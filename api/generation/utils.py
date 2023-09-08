import sys
sys.path.insert(0, ".")

from typing import Tuple, List
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


from api.utils.protocol import (
    Role,
    ChatMessage
)

def is_partial_stop(output: str, stop_str: str):
    """Check whether the output contains a partial stop str."""
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False

def parse_messages(messages: List[ChatMessage], split_role=Role.USER) -> Tuple[str, List[List[ChatMessage]]]:
    system, rounds = "", []
    round = []
    for i, message in enumerate(messages):
        if message.role == Role.SYSTEM:
            system = message.content.text
            continue
        # if message.role == split_role and round:
        #     rounds.append(round)
        #     round = []
        round.append(message)
        if message.role == Role.ASSISTANT and round:
            rounds.append(round)
            round = []
        
    if round:
        rounds.append(round)
    return system, rounds

def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op, so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list

SEQUENCE_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_position_embeddings",
    "max_seq_len",
    "model_max_length",
]

def get_context_length(config):
    """Get the context length of a model from a huggingface model config."""
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling:
        rope_scaling_factor = config.rope_scaling["factor"]
    else:
        rope_scaling_factor = 1

    for key in SEQUENCE_LENGTH_KEYS:
        val = getattr(config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048

if __name__ == "__main__":
    import pprint
    from api.utils.protocol import ContentType, Content
    messsges = [ChatMessage(role=Role.SYSTEM, content_type=ContentType.TEXT, content=Content(text="123")),
                ChatMessage(role=Role.USER, content_type=ContentType.IMAGE, content=Content(url="https://tse1-mm.cn.bing.net/th/id/OIP-C.OWcAiSlHu1-bzCpPo_a-OgHaJI?w=149&h=184&c=7&r=0&o=5&pid=1.7")),
                ChatMessage(role=Role.USER, content_type=ContentType.TEXT, content=Content(text="这张图片里有什么？")),
                ChatMessage(role=Role.ASSISTANT, content_type=ContentType.TEXT, content=Content(text="456")),
                ChatMessage(role=Role.USER, content_type=ContentType.TEXT, content=Content(text="这张图片是什么？")),
                ChatMessage(role=Role.ASSISTANT, content_type=ContentType.TEXT, content=Content(text="789")),
                ChatMessage(role=Role.USER, content_type=ContentType.TEXT, content=Content(text="ahahah")),]
    
    res = parse_messages(messages=messsges)
    pprint.pprint(res[1])