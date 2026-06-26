import os
import json
from typing import Dict, Any, AsyncGenerator
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp import FastApiMCP
from dotenv import load_dotenv
from datetime import datetime
import httpx
from utils.logger import get_logger


# 获取日志器
logger = get_logger()

# 创建FastAPI应用
app = FastAPI(
    title="星座运势分析MCP服务",
    description="基于FastAPI-MCP的星座运势分析服务"
)

# 注册跨域中间件 CORSMiddleware
"""
allow_origins=["*"]
* 代表允许所有来源域名访问后端；
开发时方便，任何本地前端、网页都能调接口；
⚠️ 生产环境不建议写 ["*"]，应填固定域名如 ["https://xxx.com"]，防止恶意网站调用你的接口。
allow_credentials=True
允许跨域请求携带 Cookie、Token、身份凭证。
如果你的接口需要登录鉴权（传 token/cookie），这个必须开 True；
注意：如果此项为 True，生产环境不能搭配 allow_origins=["*"]，浏览器会直接报错。
allow_methods=["*"]
允许所有 HTTP 请求方式：GET、POST、PUT、DELETE 等全部放行。
allow_headers=["*"]
允许前端携带任意请求头（比如 Authorization 存放 token、自定义请求头等）。
"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
"""
当前配置仅适合本地开发调试，上线部署需要修改：
# 生产安全写法
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://你的前端域名.com"], # 指定可信域名，不用*
    allow_credentials=True,
    allow_methods=["GET", "POST"], # 按需放开请求方法
    allow_headers=["Authorization", "Content-Type"], # 仅放行必要请求头
)
"""

@app.get(
    "/get_birthday_info", 
    summary="获取星座运势信息",
    operation_id="get_birthday_info",
    description="获取个人生日信息，确定星座"
)
async def get_birthday_info(
        birthday: str = Query(..., description="生日，格式为：2023-01-01"),
):
    """获取星座运势信息"""
    # 获取星座
    birthday_date = datetime.strptime(birthday, "%Y-%m-%d")
    month = birthday_date.month
    day = birthday_date.day
    return {
        "status": "success",
        "message": "获取星座成功",
        "constellation": "双子座"   # 先默认为双子座
    }

# 可以集成已有的MCP服务，比如阿里百炼的联网搜索(这个好像不太行,回头再改)
@app.get(
    "/ali_search_web", 
    summary="阿里百炼的联网搜索",
    operation_id="ali_search_web",
    description="阿里百炼的联网搜索")
async def ali_search_web(query: str = Query(..., description="搜索查询内容")):
    """访问阿里云百炼的联网搜索MCP服务，当你无法回答用户的问题或者当问的是与星座信息无关时使用！"""
    try:
        logger.info(f"API调用: ali_mcp_search({query})")
        # 使用阿里云百炼MCP WebSearch服务
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return {"error": "错误：未配置DASHSCOPE_API_KEY环境变量"}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }

        payload = {
            "query": query
        }

        url = "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp"

        logger.info(f"发送搜索请求: URL={url}, payload={payload}")

        async def stream_response():
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    print(f"Response Status: {response.status_code}")
                    print(f"Response: {response}")

                    # 检查响应状态码
                    if response.status_code != 200:
                        error_text = await response.aread()
                        yield json.dumps({
                            "error": f"搜索失败，状态码: {response.status_code}",
                            "details": error_text.decode() if isinstance(error_text, bytes) else str(error_text)
                        }) + "\n"
                        return

                    # 处理SSE流式响应
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if line.startswith("data: "):
                            data = line[6:]  # 移除 "data: " 前缀
                            if data == "[DONE]":
                                yield json.dumps({"status": "completed", "message": "搜索完成"}) + "\n"
                                break
                            try:
                                # 尝试解析JSON数据
                                parsed_data = json.loads(data)
                                yield json.dumps(parsed_data, ensure_ascii=False) + "\n"
                            except json.JSONDecodeError as e:
                                yield json.dumps({
                                    "error": f"JSON解析错误: {str(e)}",
                                    "raw_data": data
                                }) + "\n"
                        elif line and not line.startswith(":"):  # 忽略注释行
                            # 处理其他可能的数据行
                            yield json.dumps({"info": line}) + "\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    except Exception as e:
        error_result = {"error": f"搜索过程中发生错误: {str(e)}"}
        return JSONResponse(content=error_result, status_code=500)


mcp = FastApiMCP(
    app,
    name="星座运势分析MCP服务",
    description="基于FastAPI-MCP的星座运势分析服务"
)

mcp.mount_http()

mcp.setup_server()

if __name__ == "__main__":
    import uvicorn
    
    # 获取端口
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "127.0.0.1")
    
    # 启动服务
    uvicorn.run("easy_server:app", host=host, port=port)


