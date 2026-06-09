# 使用redis缓存保存历史对话
"""
Redis 把聊天内容存在电脑内存（RAM）里,不是存在代码文件夹，默认情况也不是存在数据库文件
就是存在 Redis 服务占用的内存 里
所以：
    读取、写入极快
    重启电脑 / 关掉 Redis → 缓存全部清空
因此：
    本次聊天结束后程序结束也没问题，重启程序(只要没改user_id)还会有之前的记忆
只有两种情况会删：
    到了过期时间（3 天）
    Redis 重启 / 电脑重启
    你手动调用 clear_user_chat_history()
"""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
llm = ChatOllama(model="qwen2.5:7b",temperature=0.9)
template = ChatPromptTemplate(
    [
        ("system", "你是一个上知天文下知地理的智能聊天助手"),
        ("placeholder", "{conversation}"),
        ("human", "{input}")
    ]
)
chain = template | llm | StrOutputParser()

from typing import List, Dict
import json
import redis
# ====================== Redis 连接配置 ======================
redis_client = redis.Redis(
    host="localhost",       # 你的Redis地址
    port=6379,              # 默认端口
    db=0,                   # 使用第0个库
    password=None,             # 有密码就填，没有留空
    decode_responses=True,  # 自动解码为字符串（不需要手动bytes转str）
    socket_timeout=5
)

# 聊天缓存配置（可根据需求修改）
CACHE_EXPIRE_SECONDS = 3600 * 24 * 3  # 缓存3天过期
REDIS_CHAT_KEY_PREFIX = "chatbot:history:"  # 统一key前缀
DEFAULT_SYSTEM_PROMPT = "你是一个有用、温柔、专业的AI助手。"

# ====================== 核心工具函数 ======================

def get_user_chat_history(user_id: str) -> List[Dict]:
    """获取用户聊天历史"""
    key = f"{REDIS_CHAT_KEY_PREFIX}{user_id}"
    data = redis_client.get(key)
    return json.loads(data) if data else []

def clear_user_chat_history(user_id: str):
    """清空用户记录"""
    key = f"{REDIS_CHAT_KEY_PREFIX}{user_id}"
    redis_client.delete(key)

def append_message(user_id: str, role: str, content: str):
    """追加消息（自动刷新过期时间）"""
    history = get_user_chat_history(user_id)
    history.append({"role": role, "content": content})
    key = f"{REDIS_CHAT_KEY_PREFIX}{user_id}"
    redis_client.setex(key, CACHE_EXPIRE_SECONDS, json.dumps(history, ensure_ascii=False))

def init_user_with_prompt(user_id: str,ai_prompt: str = DEFAULT_SYSTEM_PROMPT):
    """初始化用户人设（只在全新开始时调用）"""
    clear_user_chat_history(user_id)
    append_message(user_id, "system", ai_prompt)

def update_system_prompt(user_id: str, new_prompt: str):
    """
    中途修改AI人设，不删除历史，只替换 system 指令
    """
    history = get_user_chat_history(user_id)

    # 移除旧的 system 人设（只删第一条 system）
    if history and history[0]["role"] == "system":
        history = history[1:]  # 去掉旧人设

    # 插入新人设到最前面
    new_history = [{"role": "system", "content": new_prompt}] + history

    # 存回 Redis
    key = f"{REDIS_CHAT_KEY_PREFIX}{user_id}"
    redis_client.setex(key, CACHE_EXPIRE_SECONDS, json.dumps(new_history, ensure_ascii=False))

def has_history(user_id: str) -> bool:
    """检查用户是否有历史记录"""
    return len(get_user_chat_history(user_id)) > 0

# ====================== 你要的新功能：列出所有用户 ======================
def list_all_users() -> List[str]:
    """获取所有曾经聊天的用户ID"""
    keys = redis_client.keys(f"{REDIS_CHAT_KEY_PREFIX}*")
    return [key.replace(REDIS_CHAT_KEY_PREFIX, "") for key in keys]

# ====================== 核心：带询问是否删除历史的聊天入口 ======================
def start_chat_with_user(user_id: str):
    """
    启动聊天：
    1. 检查是否有历史
    2. 有历史 → 询问是否删除
    3. 删除 → 清空并重新设置人设
    4. 不删除 → 直接续聊
    """
    print(f"\n===== 用户 {user_id} 启动对话 =====")

    # 检查是否有历史
    if has_history(user_id):
        print("✅ 检测到历史聊天记录")
        choice = input("是否删除历史记录？(y/n)：").strip().lower()

        if choice == "y":
            print("🗑️ 正在删除历史记录...")
            clear_user_chat_history(user_id)
            print("✅ 已删除，正在初始化新人设...")
            ai_prompt = str(input("请输入你希望的ai角色："))
            init_user_with_prompt(user_id,ai_prompt)
        else:
            print("✅ 继续之前的聊天")
    else:
        print("🆕 首次聊天，初始化人设...")
        ai_prompt = str(input("请输入你希望的ai角色："))
        init_user_with_prompt(user_id,ai_prompt)

    # 开始聊天循环
    while True:
        user_input = input("\n你：")
        if user_input in ["exit", "quit", "q"]:
            print("👋 结束聊天")
            break

        # ====================== 新增：中途改人设 ======================
        if user_input.startswith("#set "):
            new_prompt = user_input[5:].strip()
            update_system_prompt(user_id, new_prompt)
            print(f"✅ 人设已更新：{new_prompt}")
            continue  # 本轮不对话，直接进入下一轮

        # 获取完整上下文
        messages = get_user_chat_history(user_id)
        messages.append({"role": "user", "content": user_input})

        ai_response = chain.invoke(
            {
                "conversation": messages,
                "input": user_input
            }
        )

        append_message(user_id, "user", user_input)
        append_message(user_id, "assistant", ai_response)

        print(f"AI：{ai_response}")

def chat_history(user_id: str) -> List[Dict]:
    # 查看完整历史
    if input("是否查看当前用户的完整聊天历史？(y/n)") == "y":
        print(f"\n用户{user_id}完整聊天历史：")
        user_history = get_user_chat_history(user_id)
        if len(user_history) == 0:  # 如果没有缓存，则返回空列表
            print("没有聊天记录")
        else:  # 如果有缓存，则返回列表
            print(user_history)

if __name__ == "__main__": 
    print("📋 所有历史用户：", list_all_users())
    
    target_user = input("\n请输入用户ID：")
    start_chat_with_user(target_user)

    chat_history(target_user)
        

    if input("是否清空该用户的全部聊天记录？(y/n)") == "y":
        clear_user_chat_history(target_user)
        print("已清空聊天记录")

