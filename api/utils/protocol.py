import secrets
import time
from datetime import datetime
from enum import Enum
from typing import Literal, Optional, List, Dict, Any, Union

from pydantic import BaseModel, Field

class FileException(Exception):
    def __init__(self, msg: str, code: int, *args: object) -> None:
        super().__init__(*args)
        self.msg = msg
        self.code = code
        

class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"

class ContentType(str, Enum):
    IMAGE = "image"
    TEXT = "text"
    BOX = "box"


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int
    
class Box(BaseModel):
    top_left_x: float = Field(..., description="左上角横坐标绝对值")
    top_left_y: float = Field(..., description="左上角纵坐标绝对值")
    botton_right_x: float = Field(..., description="右下角横坐标绝对值")
    botton_right_y: float = Field(..., description="右下角纵坐标绝对值")
    category: str = Field(..., description="目标框标签")
    
class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{secrets.token_hex(12)}")
    object: str = "model_permission"
    created: str = Field(default_factory=lambda: datetime.today().isoformat(), description="请求创建时间")
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = True
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: str = Field(default_factory=lambda: datetime.today().isoformat(), description="请求创建时间")
    owned_by: str = "dnect"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = []


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    first_tokens: Optional[Any] = None

class Content(BaseModel):
    text: Optional[str] = None
    base64_image: Optional[str] = None
    url: Optional[str] = None
    box: Optional[List[Box]] = None
    
class ChatMessage(BaseModel):
    role: str
    content_type: str
    content: Content
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None

    # Additional parameters supported by vLLM
    best_of: Optional[int] = None
    top_k: Optional[int] = -1
    ignore_eos: Optional[bool] = False
    use_beam_search: Optional[bool] = False
    
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{secrets.token_hex(12)}")
    object: str = "chat.completion"
    created: str = Field(default_factory=lambda: datetime.today().isoformat(), description="请求创建时间")
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
    
class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None