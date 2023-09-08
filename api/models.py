from api.config import config
from api.generation.core import ModelServer

def get_generate_model():
    from api.apapter.model import load_model

    model, tokenizer = load_model(
        config.MODEL_NAME,
        model_name_or_path=config.MODEL_PATH,
        adapter_model=config.ADAPTER_MODEL_PATH,
        quantize=config.QUANTIZE,
        device=config.DEVICE,
        device_map=config.DEVICE_MAP,
        num_gpus=config.NUM_GPUs,
        load_in_8bit=config.LOAD_IN_8BIT,
        load_in_4bit=config.LOAD_IN_4BIT,
        use_ptuning_v2=config.USING_PTUNING_V2,
        dtype=config.DTYPE
    )

    return ModelServer(
        model,
        tokenizer,
        config.DEVICE,
        model_name=config.MODEL_NAME,
        context_len=config.CONTEXT_LEN,
        stream_interval=config.STREAM_INTERVERL,
        prompt_name=config.PROMPT_NAME,
    )
    
GENERATE_MDDEL = get_generate_model()