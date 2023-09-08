import cv2
import base64
import numpy as np
import urllib.request
import urllib.error

from api.config import config
from api.utils.constants import ErrorCode
from api.utils.protocol import Content, FileException

def img_to_base64(img_array):
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    encode_image = cv2.imencode(".jpg", img_array)[1]
    byte_data = encode_image.tobytes()
    base64_str = base64.b64encode(byte_data).decode("ascii")
    return base64_str

def base64_to_img(base64_str: str) -> np.ndarray:
    byte_data = base64.b64decode(base64_str)
    encode_image = np.asarray(bytearray(byte_data), dtype="uint8")
    img_array = cv2.imdecode(encode_image, cv2.IMREAD_COLOR)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return np.array(img_array)


def get_img_from_url(url: str) -> np.ndarray:
    
    resp = urllib.request.urlopen(url, timeout=config.MAX_IMAGE_LOAD_TIME)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def get_img(content: Content) -> np.ndarray:
    if content.base64_image:
        try:
            image_base64_str = content.base64_image
            image = base64_to_img(image_base64_str)
            if not image is None:
                return image
            else:
                raise FileException(msg="Failed to obtain image data by base64 data.", code=ErrorCode.INVALID_PARAM)
        except Exception as e:
            if not content.url:
                raise FileException(msg=f"Failed to obtain image data by base64 data, because {str(e)}.", code=ErrorCode.INVALID_PARAM)
            
    if content.url:
        try:
            image = get_img_from_url(content.url)
            if not image is None:
                return image
            else:
                raise FileException(msg=f"Failed to obtain image data by url {content.url}.", code=ErrorCode.INVALID_PARAM)
        except urllib.error.URLError:
            raise FileException(msg="Image load timeout, you can modify the parameter 'MAX_IMAGE_LOAD_TIME' to reduce the occurrence of this problem.", code=ErrorCode.LOAD_TIME_OUT)
        except Exception as e:
            raise FileException(msg=f"Failed to obtain image data by url {content.url}, because {str(e)}.", code=ErrorCode.INVALID_PARAM)
        
    raise FileException(msg="Please provide as least one type of image data.", code=ErrorCode.PARAM_MISSING)