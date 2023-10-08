import sys
import time

sys.path.insert(0, ".")

from loguru import logger
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.config import config
from api.models import GENERATE_MDDEL
from api.routes import chat_router, model_router
from api.routes.utils import create_error_response
from api.utils.protocol import FileException
from api.utils.constants import ErrorCode

logger.remove(handler_id=None)
logger.add(config.LOG_PATH + "server.log", format="{time} {level} {message}", filter="", level="DEBUG")

app = FastAPI(title="多模LLm交互API接口 V1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time * 1000)[:6] + "ms"
    return response

@app.exception_handler(FileException)
async def file_exception_handler(request , exc: FileException):
    return create_error_response(code=exc.code, message=exc.msg)


prefix = config.API_PREFIX
app.include_router(model_router, prefix=prefix, tags=["model"])
if GENERATE_MDDEL is not None:
    app.include_router(chat_router, prefix=prefix, tags=["Chat"])


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
