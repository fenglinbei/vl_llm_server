import sys
import torch

from loguru import logger
from typing import Optional, List
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

if sys.version_info >= (3, 9):
    from functools import cache


class BaseModelAdapter:
    """The base and the default model adapter."""

    model_names = []

    def match(self, model_name):
        return any(m in model_name for m in self.model_names) if self.model_names else True

    def load_model(self, model_name_or_path: Optional[str] = None, adapter_model: Optional[str] = None, **kwargs):
        """ Load model through transformers. """
        model_name_or_path = self.default_model_name_or_path if model_name_or_path is None else model_name_or_path
        tokenizer_kwargs = {"trust_remote_code": True, "use_fast": False}
        tokenizer_kwargs.update(self.tokenizer_kwargs)

        if adapter_model is not None:
            try:
                tokenizer = self.tokenizer_class.from_pretrained(adapter_model, **tokenizer_kwargs)
            except OSError:
                tokenizer = self.tokenizer_class.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        else:
            tokenizer = self.tokenizer_class.from_pretrained(model_name_or_path, **tokenizer_kwargs)

        config_kwargs = self.model_kwargs
        device = kwargs.get("device", "cuda")
        num_gpus = kwargs.get("num_gpus", 1)
        dtype = kwargs.get("dtype", "bf16")
        
        if dtype == "bf16":
            config_kwargs["bf16"] = True
        elif dtype == "fp16":
            config_kwargs["fp16"] = True
        elif dtype == "fp32":
            config_kwargs["fp32"] = True
        
        if device == "cuda":
            config_kwargs["device_map"] = "cuda"
            # if "torch_dtype" not in config_kwargs:
            #     config_kwargs["torch_dtype"] = torch.float16
            if num_gpus != 1:
                config_kwargs["device_map"] = "auto"
                # model_kwargs["device_map"] = "sequential"  # This is important for not the same VRAM sizes


        if kwargs.get("device_map", None) == "auto":
            config_kwargs["device_map"] = "auto"
        # config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

        print(config_kwargs)
        # Load and prepare pretrained models (without valuehead).
        model = self.model_class.from_pretrained(
            model_name_or_path,
            # config=config,
            trust_remote_code=True,
            **config_kwargs
        )

        if device == "cpu":
            model = model.float()

        # post process for special tokens
        tokenizer = self.post_tokenizer(tokenizer)


        if device == "cuda" and num_gpus == 1 and "device_map" not in config_kwargs:
            model.to(device)

        # inference mode
        model.eval()

        return model, tokenizer


    def post_tokenizer(self, tokenizer):
        return tokenizer

    @property
    def model_class(self):
        return AutoModelForCausalLM

    @property
    def model_kwargs(self):
        return {}

    @property
    def tokenizer_class(self):
        return AutoTokenizer

    @property
    def tokenizer_kwargs(self):
        return {}

    @property
    def default_model_name_or_path(self):
        return "zpn/llama-7b"


# A global registry for all model adapters
model_adapters: List[BaseModelAdapter] = []


def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls())


@cache
def get_model_adapter(model_name: str) -> BaseModelAdapter:
    """Get a model adapter for a model name."""
    for adapter in model_adapters:
        if adapter.match(model_name):
            return adapter
    raise ValueError(f"No valid model adapter for {model_name}")

class QwenModelAdapter(BaseModelAdapter):
    """ https://github.com/QwenLM/Qwen-7B """

    model_names = ["qwen-vl-chat"]

    @property
    def model_kwargs(self):
        return {"device_map": "cuda"}
    
    @property
    def default_model_name_or_path(self):
        return "Qwen/Qwen-VL-Chat"

def load_model(
    model_name: str,
    model_name_or_path: Optional[str] = None,
    adapter_model: Optional[str] = None,
    quantize: Optional[int] = 16,
    device: Optional[str] = "cuda",
    load_in_8bit: Optional[bool] = False,
    **kwargs
):
    model_name = model_name.lower()

    if "tiger" in model_name:
        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

    # get model adapter
    adapter = get_model_adapter(model_name)

    model, tokenizer = adapter.load_model(
        model_name_or_path,
        adapter_model,
        device=device,
        quantize=quantize,
        load_in_8bit=load_in_8bit,
        **kwargs
    )
    return model, tokenizer


register_model_adapter(QwenModelAdapter)

# After all adapters, try the default base adapter.
register_model_adapter(BaseModelAdapter)