import gc
from typing import Iterable, Optional, List, Union

import torch
import torch.cuda
import torch.nn.functional as F

from loguru import logger
from transformers_stream_generator.main import NewGenerationMixin, StreamGenerationConfig
from transformers.generation.logits_process import LogitsProcessorList

# from api.apapter import get_prompt_adapter
# from api.generation.qwen import build_qwen_chat_input, check_is_qwen
from api.utils.constants import ErrorCode
from api.utils.protocol import ChatMessage
from api.apapter.conversation import get_prompt_adapter
from api.generation.qwen import check_is_qwen, build_qwen_vl_chat_input, StopWordsLogitsProcessor
from api.generation.utils import prepare_logits_processor, is_partial_stop, get_context_length

server_error_msg = (
    "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
)


@torch.inference_mode()
def generate_stream(
    model,
    tokenizer,
    params,
    device: str,
    context_len: int,
    stream_interval: int = 2,
):
    # Read parameters
    prompt = params["prompt"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    echo = bool(params.get("echo", True))
    stop = params.get("stop", None)
    request_id = params.get("request_id", None)
    
    max_window_size = params.get('max_window_size', None)
    if max_window_size is None:
        max_window_size = model.generation_config.max_window_size
        
    chat_format = params.get('chat_format', None)
    chat_format = model.generation_config.chat_format

    stop_token_ids = params.get("stop_token_ids", None) or []
    
    if stop is not None:
        if isinstance(stop, str):
            stop = [stop]
        stop_token_ids.extend(list(map(lambda x: tokenizer.encode(x), stop)))
    
    
    
    if tokenizer.eos_token_id not in stop_token_ids and tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)
        
    
    if stop_token_ids is not None:
            stop_words_logits_processor = StopWordsLogitsProcessor(
                stop_words_ids=stop_token_ids,
                eos_token_id=tokenizer.eos_token_id,
            )
    
    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )
    logits_processor.append(stop_words_logits_processor)


    if isinstance(prompt, list) and check_is_qwen(model):
        input_ids = build_qwen_vl_chat_input(tokenizer=tokenizer,
                                             messages=prompt,
                                             request_id=request_id,
                                             context_len=context_len,
                                             max_new_tokens=max_new_tokens,
                                             max_window_size=max_window_size,
                                             chat_format=chat_format)
        stop_token_ids.extend([tokenizer.im_end_id, tokenizer.im_start_id])
    else:
        raise NotImplementedError

    output_ids = list(input_ids)
    input_echo_len = len(input_ids)
    
    input_ids=torch.as_tensor([input_ids], device=device)

    past_key_values = None
    sent_interrupt = False
    first_tokens = None
    
    
    model.__class__.generate_stream = NewGenerationMixin.generate
    model.__class__.sample_stream = NewGenerationMixin.sample_stream
    stream_config = StreamGenerationConfig(**model.generation_config.to_dict(), do_stream=True)
    
    count = 0
    for tokens in model.generate_stream(
                    input_ids,
                    return_dict_in_generate=False,
                    generation_config=stream_config,
                    logits_processor=logits_processor,
                    seed=-1):
        
        output_ids.append(tokens[0])
        
        if tokens in stop_token_ids:
            stopped = True
        else:
            stopped = False

        # Yield the output tokens
        if count % stream_interval == 0 or count == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len(prompt) if isinstance(prompt, str) else 0
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer._decode(
                tmp_output_ids, clean_up_tokenization_spaces=True,
            )

            partially_stopped = False
            if stop:
                if isinstance(stop, str):
                    pos = output.rfind(stop, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop)
                elif isinstance(stop, Iterable):
                    for each_stop in stop:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # Prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": count,
                        "total_tokens": input_echo_len + count,
                        "first_tokens": first_tokens
                    },
                    "finish_reason": None,
                }
        count += 1
        if stopped:
            break
        elif count > max_new_tokens:
            break

    # Finish stream event, which contains finish reason
    if count == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": count,
            "total_tokens": input_echo_len + count,
            "first_tokens": first_tokens
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values
    gc.collect()
    torch.cuda.empty_cache()


class ModelServer:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        model_name,
        context_len: Optional[int] = None,
        stream_interval: Optional[int] = 2,
        prompt_name: Optional[str] = None,
    ):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name.lower()
        self.prompt_name = prompt_name.lower() if prompt_name is not None else None
        self.stream_interval = stream_interval
        self.context_len = context_len

        self.construct_prompt = True
        if check_is_qwen(self.model):
            logger.info("Using Qwen-VL Model for Chat!")
            self.construct_prompt = False
            self.generate_stream_func = generate_stream
            self.context_len = 8192 if self.context_len is None else self.context_len
        else:
            self.generate_stream_func = generate_stream

        self.prompt_adapter = get_prompt_adapter(self.model_name, prompt_name=self.prompt_name)
        if self.context_len is None:
            self.context_len = get_context_length(self.model.config)

    def count_token(self, params):
        prompt = params["prompt"]
        input_ids = self.tokenizer(prompt).input_ids
        input_echo_len = len(input_ids)

        ret = {
            "count": input_echo_len,
            "error_code": 0,
        }
        return ret

    def generate_prompt(self, messages: List[ChatMessage]) -> Union[str, List[ChatMessage]]:
        return self.prompt_adapter.generate_prompt(messages) if self.construct_prompt else messages

    def generate_stream_gate(self, params):
        if isinstance(params["prompt"], list):
            params["prompt"] = self.generate_prompt(params["prompt"])

        try:
            for output in self.generate_stream_func(
                self.model,
                self.tokenizer,
                params,
                self.device,
                self.context_len,
                self.stream_interval,
            ):
                ret = {
                    "text": output["text"],
                    "error_code": 0,
                }
                if "usage" in output:
                    ret["usage"] = output["usage"]
                if "finish_reason" in output:
                    ret["finish_reason"] = output["finish_reason"]
                if "logprobs" in output:
                    ret["logprobs"] = output["logprobs"]
                yield ret

        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield ret

        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield ret

    def generate_gate(self, params):
        if isinstance(params["prompt"], list):
            params["prompt"] = self.generate_prompt(params["prompt"])

        try:
            ret = {"text": "", "error_code": 0}
            for output in self.generate_stream_func(
                self.model,
                self.tokenizer,
                params,
                self.device,
                self.context_len,
                self.stream_interval,
            ):
                ret["text"] = output["text"]

            if "usage" in output:
                ret["usage"] = output["usage"]
            if "finish_reason" in output:
                ret["finish_reason"] = output["finish_reason"]
            if "logprobs" in output:
                ret["logprobs"] = output["logprobs"]

        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }

        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret

    @torch.inference_mode()
    def get_embeddings(self, params):
        try:
            tokenizer = self.tokenizer
            is_llama = "llama" in str(type(self.model))  # vicuna support batch inference
            is_chatglm = "chatglm" in self.model_name
            is_t5 = "t5" in str(type(self.model))
            if is_llama:
                encoding = tokenizer.batch_encode_plus(
                    params["input"], padding=True, return_tensors="pt"
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                model_output = self.model(
                    input_ids, attention_mask, output_hidden_states=True
                )
                data = model_output.hidden_states[-1]
                mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
                masked_embeddings = data * mask
                sum_embeddings = torch.sum(masked_embeddings, dim=1)
                seq_length = torch.sum(mask, dim=1)
                embedding = sum_embeddings / seq_length
                normalized_embeddings = F.normalize(embedding, p=2, dim=1)
                ret = {
                    "embedding": normalized_embeddings.tolist(),
                    "token_num": torch.sum(attention_mask).item(),
                }
            else:
                embedding = []
                token_num = 0
                for text in params["input"]:
                    input_ids = tokenizer.encode(text, return_tensors="pt").to(
                        self.device
                    )
                    if is_t5:
                        model_output = self.model(input_ids, decoder_input_ids=input_ids)
                    else:
                        model_output = self.model(input_ids, output_hidden_states=True)
                    if is_chatglm:
                        data = (model_output.hidden_states[-1].transpose(0, 1))[0]
                    elif is_t5:
                        data = model_output.encoder_last_hidden_state[0]
                    else:
                        data = model_output.hidden_states[-1][0]
                    data = F.normalize(torch.mean(data, dim=0), p=2, dim=0)
                    embedding.append(data.tolist())
                    token_num += len(input_ids[0])
                ret = {
                    "embedding": embedding,
                    "token_num": token_num,
                }
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret

    @torch.inference_mode()
    def get_other_embeddings(self, client, params):
        try:
            embeddings = client.encode(params["input"], normalize_embeddings=True)
            ret = {
                "embedding": embeddings.tolist(),
                "token_num": sum([len(i) for i in params["input"]]),
            }
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret

    @property
    def stop(self):
        return self.prompt_adapter.stop if hasattr(self.prompt_adapter, "stop") else None
