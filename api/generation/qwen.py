import sys
sys.path.insert(0, ".")

import cv2
import torch
import secrets

import numpy as np

from typing import List, Iterable
from pathlib import Path
from transformers import PreTrainedTokenizer
from transformers.generation import LogitsProcessor

from api.config import config
from api.utils.image_process import get_img
from api.generation.utils import parse_messages
from api.utils.protocol import (
    Role,
    Content,
    ContentType,
    ChatMessage,
    FileException
    )


class StopWordsLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsProcessor` that enforces that when specified sequences appear, stop geration.

    Args:
        stop_words_ids (:obj:`List[List[int]]`):
            List of list of token ids of stop ids. In order to get the tokens of the words
            that should not appear in the generated text, use :obj:`tokenizer(bad_word,
            add_prefix_space=True).input_ids`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """

    def __init__(self, stop_words_ids: Iterable[Iterable[int]], eos_token_id: int):

        if not isinstance(stop_words_ids, List) or len(stop_words_ids) == 0:
            raise ValueError(
                f"`stop_words_ids` has to be a non-emtpy list, but is {stop_words_ids}."
            )
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in stop_words_ids):
            raise ValueError(
                f"`stop_words_ids` has to be a list of lists, but is {stop_words_ids}."
            )
        if any(
            any(
                (not isinstance(token_id, (int, np.integer)) or token_id < 0)
                for token_id in stop_word_ids
            )
            for stop_word_ids in stop_words_ids
        ):
            raise ValueError(
                f"Each list in `stop_words_ids` has to be a list of positive integers, but is {stop_words_ids}."
            )

        self.stop_words_ids = list(
            filter(
                lambda bad_token_seq: bad_token_seq != [eos_token_id], stop_words_ids
            )
        )
        self.eos_token_id = eos_token_id
        for stop_token_seq in self.stop_words_ids:
            assert (
                len(stop_token_seq) > 0
            ), "Stop words token sequences {} cannot have an empty list".format(
                stop_words_ids
            )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        stopped_samples = self._calc_stopped_samples(input_ids)
        for i, should_stop in enumerate(stopped_samples):
            if should_stop:
                scores[i, self.eos_token_id] = float(2**15)
        return scores

    def _tokens_match(self, prev_tokens: torch.LongTensor, tokens: List[int]) -> bool:
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        elif len(tokens) > len(prev_tokens):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False
        elif prev_tokens[-len(tokens) :].tolist() == tokens:
            # if tokens match
            return True
        else:
            return False

    def _calc_stopped_samples(self, prev_input_ids: Iterable[int]) -> Iterable[int]:
        stopped_samples = []
        for prev_input_ids_slice in prev_input_ids:
            match = False
            for stop_token_seq in self.stop_words_ids:
                if self._tokens_match(prev_input_ids_slice, stop_token_seq):
                    # if tokens do not match continue
                    match = True
                    break
            stopped_samples.append(match)

        return stopped_samples


def build_qwen_vl_chat_input(
    tokenizer: PreTrainedTokenizer,
    messages: List[ChatMessage],
    request_id: str,
    context_len: int = 8192,
    max_new_tokens: int = 256,
    max_window_size: int = 6144,
    chat_format: str = "chatml"
) -> List[int]:

    max_input_tokens = context_len - max_new_tokens
    system, rounds = parse_messages(messages)
    system = "You are a helpful assistant." if not system else system  # fix system prompt
    
    save_dir = Path(config.TEMP_DIR) / request_id
    save_dir.mkdir(exist_ok=True, parents=True)

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set(tokenizer.IMAGE_ST)
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set(tokenizer.IMAGE_ST))
        
        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        # raw_text = ""
        context_tokens = []
        for round in rounds[::-1]:

            query_messages = list(filter(lambda x: x.role == Role.USER, round))
            response_messages = list(filter(lambda x: x.role == Role.ASSISTANT, round))
            
            turn_query = get_query_text(query_messages, save_dir=save_dir)
            if len(response_messages) == 0:
                query = turn_query
                continue
            turn_response = response_messages[0].content.text if response_messages else None
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            
            if turn_response is not None:
                response_text, response_tokens_part = _tokenize_str(
                    "assistant", turn_response
                )
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

                next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                prev_chat = (
                    f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
                )
            else:
                next_context_tokens = nl_tokens + query_tokens + nl_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n"

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                # raw_text = prev_chat + raw_text
            else:
                break
        
        
        context_tokens = system_tokens + context_tokens
        # raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        # raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    return context_tokens[-max_input_tokens:]  # truncate left

def check_is_qwen(model):
    return "QWenBlock" in getattr(model, "_no_split_modules", [])

def get_query_text(messages: List[ChatMessage], save_dir: str):
    query_text = ""
    pic_idx = 1
    for query_message in messages:
        if query_message.content_type == ContentType.IMAGE:
            content = query_message.content
            
            image = get_img(content=content)

            image_name = f"tmp{secrets.token_hex(5)}.jpg"
            file_name = str(save_dir / image_name)
            
            cv2.imwrite(file_name, image)
            
            query_text += f'Picture {pic_idx}:<img>{file_name}</img>\n'
            pic_idx += 1
        if query_message.content_type == ContentType.TEXT:
            query_text += query_message.content.text
    return query_text
            
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from api.utils.protocol import ChatMessage
    model_name = "./model/models--Qwen--Qwen-VL-Chat/snapshots/96a960aacb911cd09def34dc679bdb81f60e6110/"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    build_qwen_vl_chat_input(tokenizer=tokenizer,
                             messages=[ChatMessage(role='user', content_type='image', content=Content(url='https://tse1-mm.cn.bing.net/th/id/OIP-C.OWcAiSlHu1-bzCpPo_a-OgHaJI?w=149&h=184&c=7&r=0&o=5&pid=1.7'), name=None),
                                       ChatMessage(role='user', content_type='text', content=Content(text='你好！'), name=None)],
                             request_id="chatcmpl-a858132d0eda8ff4ae26e3f2",
                             )