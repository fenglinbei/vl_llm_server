import os
import shutil
import secrets

from loguru import logger
from fastapi import APIRouter, Request
from typing import List, Dict, Optional, Union, Any, Generator

from api.config import config
from api.models import GENERATE_MDDEL
from api.utils.protocol import (
    Box,
    Role,
    Content,
    UsageInfo,
    ContentType,
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice
    )

from api.routes.utils import check_requests, create_error_response

chat_router = APIRouter(prefix="/chat")

@chat_router.post("/completions")
async def create_chat_completion(raw_request: Request):
    request = ChatCompletionRequest(**await raw_request.json())
    # logger.debug(f"Received chat completion request: {request}")
    
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret
    
    request_id = f"chatcmpl-{secrets.token_hex(12)}"
    messages = request.messages
    
    gen_params = get_gen_params(
        request.model,
        messages,
        request_id=request_id,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        max_new_tokens=request.max_tokens,
        echo=False,
        stream=request.stream,
        stop=request.stop,
    )

    # if request.stream:
    #     generator = chat_completion_stream_generator(
    #         request.model, gen_params, request.n
    #     )
    #     return StreamingResponse(generator, media_type="text/event-stream")

    choices = []
    usage = UsageInfo()
    for i in range(request.n):
        content = GENERATE_MDDEL.generate_gate(gen_params)
        if content["error_code"] != 0:
            return create_error_response(code=int(content["error_code"]), message=content["text"])
        
        finish_reason = "stop"
        boxes = get_boxes(content["text"])
        if len(boxes) == 0:
            message = ChatMessage(role=Role.ASSISTANT, content_type=ContentType.TEXT, content=Content(text=content["text"]))
        else:
            message = ChatMessage(role=Role.ASSISTANT, content_type=ContentType.BOX, content=Content(box=boxes))

        choices.append(
            ChatCompletionResponseChoice(
                index=i,
                message=message,
                finish_reason=finish_reason,
            )
        )
        task_usage = UsageInfo(**content["usage"])
        for usage_key, usage_value in task_usage.dict().items():
            if usage_key != "first_tokens":
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
        usage.first_tokens = content["usage"].get("first_tokens", None)

    delete_cache()
    return ChatCompletionResponse(id=request_id, model=request.model, choices=choices, usage=usage)

def get_boxes(text: str) -> List[Box]:
    boxes: List[Dict[str, Any]] = GENERATE_MDDEL.tokenizer._fetch_all_box_with_ref(text)
    box_list = []
    category = "box"
    for box in boxes:
        if "ref" in box:
            category = box["ref"]
        
        box_list.append(Box(top_left_x=box["box"][0] / 1000,
                            top_left_y=box["box"][1] / 1000,
                            botton_right_x=box["box"][2] / 1000,
                            botton_right_y=box["box"][3] / 1000,
                            category=category))
    return box_list

def get_gen_params(
    model_name: str,
    messages: List[ChatMessage],
    request_id: str,
    *,
    max_new_tokens: Optional[int],
    temperature: float,
    top_k: int,
    top_p: float,
    stop: Optional[Union[str, List[str]]]=None,
    echo: Optional[bool],
    stream: Optional[bool],
    ) -> Dict[str, Any]:
    
    if not max_new_tokens:
        max_new_tokens = 1024
    
    gen_params = {
        "model": model_name,
        "request_id": request_id,
        "prompt": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "echo": echo,
        "stream": stream,
    }
    
    if GENERATE_MDDEL.stop is not None:
        if "token_ids" in GENERATE_MDDEL.stop:
            gen_params["stop_token_ids"] = GENERATE_MDDEL.stop["token_ids"]

        if "strings" in GENERATE_MDDEL.stop:
            gen_params["stop"] = GENERATE_MDDEL.stop["strings"]

    if stop is not None:
        if isinstance(stop, str):
            stop = [stop]

        gen_params["stop"] = gen_params["stop"] + stop if "stop" in gen_params else stop
        gen_params["stop"] = list(set(gen_params["stop"]))

    # logger.debug(f"==== request ====\n{gen_params}")
    return gen_params
"""    
    generation_config = GenerationConfig(
    **{
        "chat_format": "chatml",
        "do_sample": True,
        "eos_token_id": 151643,
        "max_new_tokens": 512 if not max_new_tokens else max_new_tokens,
        "max_window_size": 6144,
        "pad_token_id": 151643,
        "top_k": 0 if not top_k else top_k,
        "top_p": 0.3 if not top_p else top_p,
        "transformers_version": "4.31.0"
    })
    
    if stop is not None:
        if isinstance(stop, str):
            stop = [stop]
        stop_words_ids = list(map(lambda x: tokenizer.encode(x), stop))
        
        
    query_messages: List[ChatMessage] = []
    
    for message in reversed(messages):
        if message.role == Role.USER:
            query_messages.append(message)
    
    query = ""
    save_dir = Path(config.TEMP_DIR) / request_id
    save_dir.mkdir(exist_ok=True, parents=True)
    pic_idx = 0
    query = ""
    image_list = []
    
    for query_message in reversed(query_messages):
        if query_message.role == Role.ASSISTANT:
            break
        else:
            query_messages.pop()
        if query_message.content_type == ContentType.IMAGE:
            content = query_message.content
            
            res, flag = get_img(content=content)
            if not flag:
                return res
            else:
                image = res
            
            image_name = f"tmp{secrets.token_hex(5)}.jpg"
            file_name = str(save_dir / image_name)
            
            cv2.imwrite(file_name, image)
            image_list.append(file_name)
        if query_message.content_type == ContentType.TEXT:
            query += query_message.content.text
        
    
    image_query = ""
    for image_name in image_list:
        pic_idx += 1
        q = f'Picture {pic_idx}: <img>{file_name}</img>\n'
        image_query += q
    
    query = image_query + query
    
    if messages[0].role == Role.SYSTEM:
        system = messages[0].content.text
        messages = messages[1:]
    else:
        system = "You are a helpful assistant."
    
    history = []
    history_qa = ("", "")
    pic_idx = 0
    for message in messages:
        
        if message.role == Role.USER:
            if message.content_type == ContentType.IMAGE:
                content = message.content
                
                res, flag = get_img(content=content)
                if not flag:
                    return res
                else:
                    image = res
                
                image_name = f"tmp{secrets.token_hex(5)}.jpg"
                file_name = str(save_dir / image_name)
                
                cv2.imwrite(file_name, image)
                pic_idx += 1
                history_qa[0] += f'Picture {pic_idx}: <img>{file_name}</img>\n'
            if query_message.content_type == ContentType.TEXT:
                history_qa[0] += message.content.text
        if message.role == Role.ASSISTANT:
            history_qa[1] += message.content.text
            history.append(history_qa)
            history_qa = ("", "")
            pic_idx = 0
    
    params = {"tokenizer": tokenizer,
              "query": query,
              "history": history,
              "system": system,
              "stop_words_ids": stop_words_ids,
              "generation_config": generation_config}

    return params"""

def delete_cache():
    tmp_dir_list = list(map(lambda x: os.path.join(config.TEMP_DIR, x), os.listdir(config.TEMP_DIR)))
    remove_list = sorted(tmp_dir_list, key=lambda x: os.path.getctime(x), reverse=True)[config.MAX_TEMP_NUM:]
    for dir in remove_list:
        shutil.rmtree(dir)
    

async def chat_completion_stream_generator(
    model_name: str, gen_params: Dict[str, Any], n: int
) -> Generator[str, Any, None]:
    raise NotImplementedError
    """
    Event stream format:
    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """
    _id = f"chatcmpl-{secrets.token_hex(12)}"
    finish_stream_events = []
    for i in range(n):
        # First chunk with role
        choice_data = ChatCompletionResponseStreamChoice(
            index=i,
            delta=DeltaMessage(role=Role.ASSISTANT),
            finish_reason=None,
        )
        chunk = ChatCompletionStreamResponse(
            id=_id, choices=[choice_data], model=model_name
        )
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        previous_text = ""
        with_function_call = gen_params.get("with_function_call", False)
        found_action_name = False
        for content in GENERATE_MDDEL.generate_stream_gate(gen_params):
            if content["error_code"] != 0:
                yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            decoded_unicode = content["text"].replace("\ufffd", "")
            delta_text = decoded_unicode[len(previous_text):]
            previous_text = decoded_unicode

            if len(delta_text) == 0:
                delta_text = None

            messages = []
            if with_function_call:
                if found_action_name:
                    messages.append(build_delta_message(delta_text, "arguments"))
                    finish_reason = "function_call"
                else:
                    if previous_text.rfind("\nFinal Answer:") > 0:
                        with_function_call = False

                    if previous_text.rfind("\nAction Input:") == -1:
                        continue
                    else:
                        messages.append(build_delta_message(previous_text))
                        pos = previous_text.rfind("\nAction Input:") + len("\nAction Input:")
                        messages.append(build_delta_message(previous_text[pos:], "arguments"))

                        found_action_name = True
                        finish_reason = "function_call"
            else:
                messages = [DeltaMessage(content=delta_text)]
                finish_reason = content.get("finish_reason", "stop")

            chunks = []
            for m in messages:
                choice_data = ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=m,
                    finish_reason=finish_reason,
                )
                chunks.append(ChatCompletionStreamResponse(id=_id, choices=[choice_data], model=model_name))

            if delta_text is None:
                if content.get("finish_reason", None) is not None:
                    finish_stream_events.extend(chunks)
                continue

            for chunk in chunks:
                yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_none=True, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"
            
                
            
            
            
            
    
    

    
    