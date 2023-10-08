import os

import dotenv

dotenv.load_dotenv()


DEFAULTS = {
    'HOST': '0.0.0.0',
    'PORT': 5006,
    'LOG_PATH': './api/log/',
    'MODEL_NAME': 'Qwen-VL-Chat',
    'MODEL_PATH': './model/models--Qwen--Qwen-VL-Chat/snapshots/96a960aacb911cd09def34dc679bdb81f60e6110/',
    'DEVICE': 'cuda',
    'DEVICE_MAP': "",
    'GPUS': '',
    'NUM_GPUs': 1,
    'QUANTIZE': 16,
    'CONTEXT_LEN': '',
    'LOAD_IN_8BIT': 'False',
    'LOAD_IN_4BIT': 'False',
    'USING_PTUNING_V2': 'False',
    'STREAM_INTERVERL': 2,
    'PROMPT_NAME': '',
    'PATCH_TYPE': '',
    'TRAINING_LENGTH': 4096,
    'WINDOW_SIZE': 512,
    'API_PREFIX': '/v1',
    'USE_VLLM': 'False',
    'TRUST_REMOTE_CODE': "False",
    'TOKENIZE_MODE': "auto",
    'TENSOR_PARALLEL_SIZE': 1,
    'DTYPE': "fp16",
    'EMBEDDING_SIZE': '',
    'MAX_IMAGE_LOAD_TIME': 3,
    'TEMP_DIR': './tmp/',
    'MAX_TEMP_NUM': 50
}


def get_env(key):
    return os.environ.get(key, DEFAULTS.get(key))


def get_bool_env(key):
    return get_env(key).lower() == 'true'


class Config:
    """ Configuration class. """

    def __init__(self):
        self.HOST = get_env('HOST')
        self.PORT = int(get_env('PORT'))

        self.LOG_PATH = get_env('LOG_PATH')
        self.MODEL_NAME = get_env('MODEL_NAME')
        self.MODEL_PATH = get_env('MODEL_PATH')
        self.ADAPTER_MODEL_PATH = get_env('ADAPTER_MODEL_PATH') if get_env('ADAPTER_MODEL_PATH') else None

        self.DEVICE = get_env('DEVICE')
        self.DEVICE_MAP = get_env('DEVICE_MAP') if get_env('DEVICE_MAP') else None
        self.GPUS = get_env('GPUS')
        self.NUM_GPUs = int(get_env('NUM_GPUs'))

        self.QUANTIZE = int(get_env('QUANTIZE'))
        self.EMBEDDING_NAME = get_env('EMBEDDING_NAME') if get_env('EMBEDDING_NAME') else None
        self.CONTEXT_LEN = int(get_env('CONTEXT_LEN')) if get_env('CONTEXT_LEN') else None
        self.LOAD_IN_8BIT = get_bool_env('LOAD_IN_8BIT')
        self.LOAD_IN_4BIT = get_bool_env('LOAD_IN_4BIT')
        self.USING_PTUNING_V2 = get_bool_env('USING_PTUNING_V2')

        self.STREAM_INTERVERL = int(get_env('STREAM_INTERVERL'))
        self.PROMPT_NAME = get_env('PROMPT_NAME') if get_env('PROMPT_NAME') else None
        self.PATCH_TYPE = get_env('PATCH_TYPE') if get_env('PATCH_TYPE') else None
        self.TRAINING_LENGTH = int(get_env('TRAINING_LENGTH'))
        self.WINDOW_SIZE = int(get_env('WINDOW_SIZE'))

        self.API_PREFIX = get_env('API_PREFIX')

        self.USE_VLLM = get_bool_env('USE_VLLM')
        self.TRUST_REMOTE_CODE = get_bool_env('TRUST_REMOTE_CODE')
        self.TOKENIZE_MODE = get_env('TOKENIZE_MODE')
        self.TENSOR_PARALLEL_SIZE = int(get_env('TENSOR_PARALLEL_SIZE'))
        self.DTYPE = get_env('DTYPE')

        self.EMBEDDING_SIZE = int(get_env('EMBEDDING_SIZE')) if get_env('EMBEDDING_SIZE') else None
        
        self.MAX_IMAGE_LOAD_TIME = int(get_env('MAX_IMAGE_LOAD_TIME'))
        
        self.TEMP_DIR = get_env('TEMP_DIR')
        self.MAX_TEMP_NUM = int(get_env('MAX_TEMP_NUM'))


config = Config()
# print(f"Config: {config.__dict__}")
if config.GPUS:
    if len(config.GPUS.split(",")) < config.NUM_GPUs:
        raise ValueError(
            f"Larger --num_gpus ({config.NUM_GPUs}) than --gpus {config.GPUS}!"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUS
