"""
在开发时使用 uvicorn 的热重载功能
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
import sys
import os

# 将父目录添加到路径中，以便我们可以导入app模块
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from app.api import router
from app.cors_config import add_cors_middleware


app = FastAPI()
add_cors_middleware(app)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


