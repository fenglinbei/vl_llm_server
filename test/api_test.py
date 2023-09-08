import asyncio
import os
import sys
import cv2
import time
import json
import pprint
import requests

import numpy as np

from abc import ABC
from pydantic import BaseModel
from typing import Optional, List, Tuple, Dict
from aiohttp import ClientSession

sys.path.insert(0, ".")

from api.utils.protocol import Content, ChatMessage, ChatCompletionRequest, Role, ContentType, ChatCompletionResponse, Box
from api.utils.image_process import get_img, img_to_base64

class Api:
    
    def __init__(self, server: str) -> None:
        self.server = server

    async def _chat(self,
                   messages: List[ChatMessage],
                   max_tokens: int = 0,
                   temperature: float = 0.7,
                   top_p: float = 1.0,
                   top_k: int = 5) -> str:
        params = ChatCompletionRequest(model="qwen-vl-chat",
                                       messages=messages,
                                       temperature=temperature,
                                       top_k=top_k,
                                       top_p=top_p).model_dump()

        
        async with ClientSession() as session:
            async with session.post(url=self.server + "/chat/completions", json=params, headers=None) as response:
                res = await response.text()

        response = json.loads(res)
        
        if response['object'] == 'error':
            return 'error'
        
        return response
    
    def chat(self,
                   messages: List[ChatMessage],
                   max_tokens: int = 0,
                   temperature: float = 0.7,
                   top_p: float = 1.0,
                   top_k: int = 5) -> Dict:
        params = ChatCompletionRequest(model="qwen-vl-chat",
                                       messages=messages,
                                       temperature=temperature,
                                       top_k=top_k,
                                       top_p=top_p,
                                       max_tokens=max_tokens).model_dump(exclude_unset=True)


        res = requests.post(url=self.server + "/chat/completions", json=params, headers=None)


        response = json.loads(res.text)
        
        if response['object'] == 'error':
            return 'error'
        
        return response
        
API = Api(server="http://192.168.1.210:5006/v1")

def run(main_func=None):
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        asyncio.get_event_loop().run_until_complete(main_func)
    finally:
        try:
            if hasattr(loop, "shutdown_asyncgens"):
                loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            asyncio.set_event_loop(None)
            loop.close()

def test_image(image_path: Optional[str]=None, count: int=1, url: Optional[str]=None, max_tokens: int = 0):

    if image_path:
        image_array = cv2.imread(image_path)
        image_base64_str = img_to_base64(image_array)

    t_all = 0
    
    
    async def process():
        tasks = []
    
        for i in range(count):
            
            print(i)
            if image_path:
                # res = API.image_review(content=image_base64_str, max_tokens=max_tokens)
                tasks.append(asyncio.create_task(API.image_review(content=image_base64_str, max_tokens=max_tokens)))
            elif url:
                # res = API.image_review(url=url, max_tokens=max_tokens)
                tasks.append(asyncio.create_task(API.image_review(url=url, max_tokens=max_tokens)))
        t_start = time.time()
        done, pending = await asyncio.wait(tasks)
        t_pass = time.time() - t_start
        print(f"avg process time {str(t_pass / count * 1000)[:8]}ms")
        # print(list(done)[0].result())
        return done
    
    asyncio.run(process())
    
    
    # print(f"process time {str(t_pass * 1000)[:8]}ms")
    # await res

    
    # pprint.pprint(res)
    # return res


def test_search(name: str):
    import time
    
    t_start = time.time()
    res = API.search_name(name)
    t_pass = time.time() - t_start
    
    print(f"process time {str(t_pass * 1000)[:8]} ms")
    print(res)
    
def test_delete(person_id: Optional[str] = None, face_id: Optional[str] = None):
    import time
    
    t_start = time.time()
    res = API.delete(person_id=person_id, face_id=face_id)
    t_pass = time.time() - t_start
    
    print(f"process time {str(t_pass * 1000)[:8]} ms")
    print(res)

def test_match(image_path: str, top_k: int = 10):
    
    image_array = cv2.imread(image_path)
    image_base64_str = img_to_base64(image_array)
    
    t_start = time.time()
    res = API.match(face_base64str_img=image_base64_str, top_k=top_k)
    t_pass = time.time() - t_start
    
    print(f"process time {str(t_pass * 1000)}")
    print(res)
    

from PIL import Image, ImageDraw, ImageFont
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "./SimSun.ttf", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def pad2square(image: np.ndarray, size: int = None, pad: int = None) -> Tuple[np.ndarray, int, int, float]:
    h, w, c = image.shape
    
    if pad:
        a = pad * 2 + h if h >= w else pad * 2 + w
    else:
        a = h if h >= w else w
    
    pad_h = int((a - h) / 2)
    pad_w = int((a - w) / 2)
    
    new_image = np.zeros([a, a, c], dtype=np.float32)
    new_image[pad_h : pad_h + h, pad_w : pad_w + w] = image
    
    scale = 1
    if size:
        scale = size / a
        if scale > 1:
            new_image = cv2.resize(new_image, [size, size], interpolation=cv2.INTER_LINEAR_EXACT)
        else:
            new_image = cv2.resize(new_image, [size, size], interpolation=cv2.INTER_AREA)

        # pad_h, pad_w = int(pad_h * scale), int(pad_w * scale)
    
    return new_image, pad_h, pad_w, scale


def draw_on(img: np.ndarray, boxes: List[Box], output_size: int=1280):
    
    
    dimg = img.copy()
    rh, rw, _ = dimg.shape
    dimg, pad_h, pad_w, scale = pad2square(dimg, output_size)
    dimg = dimg.astype(np.uint8)
    for box in boxes:

        color = (0, 0, 255)
        text_size = int(output_size / 35)

        box.top_left_x = int((box.top_left_x * rw + pad_w) * scale)
        box.top_left_y = int((box.top_left_y * rh + pad_h) * scale)
        box.botton_right_x = int((box.botton_right_x * rw + pad_w) * scale)
        box.botton_right_y = int((box.botton_right_y * rh + pad_h) * scale)
        
        cv2.rectangle(dimg,
                      (box.top_left_x, box.top_left_y),
                      (box.botton_right_x, box.botton_right_y), color, 2)
        dimg = cv2AddChineseText(dimg,
                                 '%s'%(box.category),
                                 (box.top_left_x - 1, box.top_left_y - text_size), (0, 255, 0), text_size)
        # cv2.putText(dimg, '%s, %s'%(face.personage, str(face.confidence)[:6]), (box.top_left_x-1, box.top_left_y-4), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    return dimg

if __name__ == "__main__":
    

    face_list = [{"image_path": '/mnt/fenglin/WiqunBot/dev/face_detection/data/image/test_07.jpg', "name": "me", "person_id": "person-1b499e7740928ffb694e6d2bd190362f"},
                 {"image_path": '/mnt/fenglin/WiqunBot/dev/face_detection/data/image/test_02.jpg', "name": "me", "person_id": "person-1b499e7740928ffb694e6d2bd190362f"},
                 {"image_path": '/mnt/fenglin/WiqunBot/dev/face_detection/data/image/test_01.jpg', "name": "ding"},
                 {"image_path": '/mnt/fenglin/WiqunBot/dev/face_detection/data/image/test_03.jpg', "name": "girl"},
                 {"image_path": '/mnt/fenglin/WiqunBot/dev/face_detection/data/image/test_04.png', "name": "meixi"}]

    # add_person()
    image_path = "./test/image/"
    image_name = "15.jpg"
    # url = "https://tse2-mm.cn.bing.net/th/id/OIP-C.eD7nCiJoGXCWf7OiFgH_lgHaHr?w=171&h=180&c=7&r=0&o=5&pid=1.7"
    # image_name = f"url_5.jpg"
    image_path = os.path.join(image_path, image_name)
    image = cv2.imread(image_path)
    base64_str = img_to_base64(img_array=image)
    messages = [ChatMessage(role=Role.USER, content_type=ContentType.IMAGE, content=Content(base64_image=base64_str)),
                ChatMessage(role=Role.USER, content_type=ContentType.TEXT, content=Content(text="框出图片中的所有中国国旗"))]
    
    res = API.chat(messages=messages, max_tokens=1000, temperature=1, top_p=0.5, top_k=50)
    pprint.pprint(res)
    # res: ChatCompletionResponse = ChatCompletionResponse.model_validate_json(json.dumps(res))
    res: ChatCompletionResponse = ChatCompletionResponse(**res)
    # res = test_image(url=url)
    
    if res.choices[0].message.content_type == ContentType.BOX:
        boxes = res.choices[0].message.content.box
    
    
    # img = cv2.imread(image_path)
    # # img, _ = get_img(content=Content(index=0,url=url))
    img = draw_on(img=image, boxes=boxes)
    
    cv2.imwrite(f"./test/image/output/output_{image_name}", img)

    