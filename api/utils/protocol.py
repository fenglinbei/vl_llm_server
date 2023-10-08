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
    """错误信息"""
    id: str = Field(default_factory=lambda: f"error-{secrets.token_hex(16)}", description="请求的唯一标识")
    object: str = Field(default="error", description="请求的返回类型")
    created: datetime = Field(default_factory=lambda: datetime.today().isoformat(), description="请求创建时间")
    message: str = Field(..., description="错误信息")
    code: int = Field(..., description="错误代码")
    
class Box(BaseModel):
    """模型输出的目标框信息"""
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
    """LLM的token使用信息"""
    prompt_tokens: int = Field(default=0, description="prompt消耗的token数")
    total_tokens: int = Field(default=0, description="总共消耗的token数")
    completion_tokens: Optional[int] = Field(default=0, description="回复消耗的token数")
    first_tokens: Optional[Any] = None

class Content(BaseModel):
    """交互的内容，包含文字、图像或目标框"""
    text: Optional[str] = Field(default=None, description="文字信息")
    base64_image: Optional[str] = Field(default=None, description="图像的base64编码")
    url: Optional[str] = Field(default=None, description="图像的url")
    box: Optional[List[Box]] = Field(default=None, description="目标框信息")
    
class ChatMessage(BaseModel):
    """交互信息块"""
    role: str = Field(..., description="信息发送者")
    content_type: str = Field(..., description="信息类型")
    content: Content = Field(..., description="信息内容")
    name: Optional[str] = Field(default=None, description="信息发送者标识")


class ChatCompletionRequest(BaseModel):
    """交互请求"""
    model: str = Field(..., description="模型名")
    messages: List[ChatMessage] = Field(..., description="交互的信息")
    temperature: Optional[float] = Field(default=0.7, description="温度系数，用于控制回复多样性，在0~1之间，越接近1多样性越强")
    top_p: Optional[float] = Field(default=1.0, description="p阈值，模型只会提取概率从大到小排列的且概率和小于p的n个token进行采样")
    n: Optional[int] = Field(default=1, description="回复的次数")
    max_tokens: Optional[int] = Field(default=1024, description="最大回复的token数，默认为1024")
    stop: Optional[Union[str, List[str]]] = Field(..., description="停止词，模型回复到该词时将自动停止回复")
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None

    # Additional parameters supported by vLLM
    best_of: Optional[int] = None
    top_k: Optional[int] = Field(default=-1, description="k阈值，模型只会提取概率从大到小的前k个token进行采样")
    ignore_eos: Optional[bool] = False
    use_beam_search: Optional[bool] = False
    
class ChatCompletionResponseChoice(BaseModel):
    """模型回复信息"""
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    """请求响应参数"""
    id: str = Field(default_factory=lambda: f"chatcmpl-{secrets.token_hex(12)}", description="请求的唯一标识")
    object: str = Field(default="chat.completion", description="请求的返回类型")
    created: str = Field(default_factory=lambda: datetime.today().isoformat(), description="请求创建时间")
    model: str
    choices: List[ChatCompletionResponseChoice] = Field(..., description="模型返回的结果")
    usage: UsageInfo
    
class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None