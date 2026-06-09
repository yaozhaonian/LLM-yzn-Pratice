from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import time
import traceback
from typing import List, Dict

from cusServe import processing_chain # 假设这里只负责分析，不负责记忆拼接，或者我们修改调用方式
from pydantic import BaseModel
import redis

app = FastAPI(title="电商客服系统")

# ======================
# Redis 配置
# ======================
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    print("Redis connected successfully.")
except Exception as e:
    print(f"Redis connection failed: {e}. Cache will be disabled.")
    redis_client = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

class FeedbackRequest(BaseModel):
    content: str
    user_id: str = "anonymous"
    session_id: str = None # 前端必须传递唯一的 session_id

# ======================
# 记忆管理辅助函数
# ======================
def get_history_from_redis(session_id: str, limit: int = 5) -> List[Dict]:
    """从 Redis 获取最近 N 轮对话历史"""
    if not redis_client or not session_id:
        return []
    
    key = f"chat_history:{session_id}"
    try:
        # 获取列表最后 limit*2 条消息 (因为一轮对话有 user 和 ai 两条)
        messages = redis_client.lrange(key, -limit*2, -1)
        # 还原为对象列表
        history = [json.loads(msg) for msg in messages]
        return history
    except Exception as e:
        print(f"Get history error: {e}")
        return []

def save_history_to_redis(session_id: str, user_msg: str, ai_msg: str):
    """将新的一轮对话存入 Redis"""
    if not redis_client or not session_id:
        return
    
    key = f"chat_history:{session_id}"
    try:
        # 保存用户消息
        redis_client.rpush(key, json.dumps({"role": "user", "content": user_msg}))
        # 保存 AI 消息
        redis_client.rpush(key, json.dumps({"role": "ai", "content": ai_msg}))
        # 设置过期时间，例如 24 小时，防止内存无限增长
        redis_client.expire(key, 86400) 
    except Exception as e:
        print(f"Save history error: {e}")

# ======================
# API 路由
# ======================
@app.post("/process-feedback")
async def process_feedback(request: FeedbackRequest):
    session_id = request.session_id or request.user_id # 如果没有 session_id，用 user_id 代替（不推荐，最好前端生成 UUID）
    
    # 1. 获取历史记录
    history = get_history_from_redis(session_id, limit=5)
    
    # 2. 准备传递给 LangChain 的数据
    # 注意：由于 cusServe.py 中的 chain 比较复杂，且主要是针对单条反馈的分析。
    # 为了实现记忆，最简单的办法是：在 Prompt 层面注入历史。
    # 但目前的 processing_chain 内部封装了 Prompt。
    # 【策略调整】：我们需要修改 cusServe.py 让 chain 接受 history，或者在这里手动构建带历史的 Prompt。
    # 鉴于修改 chain 内部逻辑较复杂，这里演示一种“外挂式”记忆：
    # 我们暂时不通过 chain 的内部 prompt 注入，而是假设 chain 输出的结果是独立的。
    # *真正的最佳实践是修改 cusServe.py 中的 generate_response 以接收 history*
    
    # 为了演示效果，我们先按原逻辑运行 chain，拿到分析结果。
    # 然后我们在 app 层做一个“后处理”或者“预处理”？
    # 不，最干净的方式是修改 cusServe.py 的 generate_response 签名。
    
    # --- 假设你已经按照上面的建议修改了 cusServe.py 中的 generate_response 和 prompt ---
    # 如果没修改 cusServe.py，下面的代码将无法利用 history。
    # 这里我提供一个兼容方案：如果 cusServe.py 没改，history 将被忽略。
    
    try:
        start = time.time()
        
        # 构造输入数据，包含历史和当前内容
        # 注意：processing_chain 需要能够处理这个字典结构
        # 如果 cusServe.py 没改，这里会报错或忽略 history。
        # 请确保 cusServe.py 的 extract_chain 能处理 dict 输入，或者我们在 invoke 前处理
        
        # 临时方案：如果 cusServe.py 还是只接受 str，我们只能放弃在 chain 内部使用历史。
        # 但为了回答你的问题，我必须假设你愿意修改 cusServe.py 以支持 dict 输入。
        
        chain_input = {
            "original_feedback": request.content,
            "history": history # 传入历史
        }
        
        # 如果 cusServe.py 中的 extract_order_id 等函数只接受 str，
        # 你需要修改 cusServe.py 中的 RunnableLambda 来适配 dict 输入。
        # 例如: RunnableLambda(lambda x: extract_order_id(x["original_feedback"]))
        
        # 【重要】由于修改 cusServe.py 较多，这里给出一个最小改动的 app.py 逻辑，
        # 它依赖于 cusServe.py 能够接收 dict 并正确提取 original_feedback
        
        result = await processing_chain.ainvoke(chain_input)
        
        elapsed = time.time() - start
        
        # 提取最终回复文本用于聊天显示
        final_response_text = result.get("final_response", "") if isinstance(result, dict) else str(result)
        
        # 提取分析数据用于面板显示 (result["result"] 里面包含了 order_id, sentiment 等)
        analysis_data = result.get("result", {}) if isinstance(result, dict) else {}
        
        # 保存历史 (使用最终回复文本)
        save_history_to_redis(session_id, request.content, final_response_text)

        return {
            "success": True,
            "source": "llm",
            "processing_time": f"{elapsed:.2f}s",
            "session_id": session_id,
            "message": final_response_text,      # 用于聊天框显示
            "analysis": analysis_data             # 用于右侧面板显示
        }

    except Exception as e:
        print("="*30)
        print("Error in process_feedback:")
        traceback.print_exc()
        print("="*30)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.get("/")
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)