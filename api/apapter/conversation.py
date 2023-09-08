from functools import cache
from typing import List, Optional, Dict

from api.utils.protocol import (
    Role,
    ChatMessage)


class BasePromptAdapter:
    """The base and the default model prompt adapter."""

    name = "default"
    system_prompt: str = "You are a helpful assistant!\n"
    user_prompt: str = "Human: {}\nAssistant: "
    assistant_prompt: str = "{}\n"
    stop = None

    def match(self, model_name):
        return True

    def generate_prompt(self, messages: List[ChatMessage]) -> str:
        prompt = self.system_prompt
        user_content = []
        for message in messages:
            role, content = message.role, message.content
            if role in [Role.USER, Role.SYSTEM]:
                user_content.append(content)
            elif role == Role.ASSISTANT:
                prompt += self.user_prompt.format("\n".join(user_content))
                prompt += self.assistant_prompt.format(content)
                user_content = []
            else:
                raise ValueError(f"Unknown role: {role}")

        if user_content:
            prompt += self.user_prompt.format("\n".join(user_content))

        return prompt



prompt_adapters: List[BasePromptAdapter] = []
prompt_adapter_dict: Dict[str, BasePromptAdapter] = {}

def register_prompt_adapter(cls):
    """Register a prompt adapter."""
    prompt_adapters.append(cls())
    prompt_adapter_dict[cls().name] = cls()

@cache
def get_prompt_adapter(model_name: str, prompt_name: Optional[str] = None):
    """Get a prompt adapter for a model name or prompt name."""
    if prompt_name is not None:
        return prompt_adapter_dict[prompt_name]
    else:
        for adapter in prompt_adapters:
            if adapter.match(model_name):
                return adapter
    raise ValueError(f"No valid prompt adapter for {model_name}")

class QwenPromptAdapter(BasePromptAdapter):
    """ https://huggingface.co/Qwen/Qwen-7B-Chat """

    name = "chatml"
    system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    user_prompt = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    assistant_prompt = "{}<|im_end|>\n"
    stop = {
        "strings": ["<|im_end|>"],
    }

    def match(self, model_name):
        return "qwen" in model_name
    
register_prompt_adapter(QwenPromptAdapter)

register_prompt_adapter(BasePromptAdapter)


