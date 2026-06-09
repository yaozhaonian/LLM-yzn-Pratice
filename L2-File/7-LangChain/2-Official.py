# 官方示例(修改版)
from langchain_ollama import ChatOllama  # 关键：改用 ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent

SYSTEM_PROMPT = """你是一位专业的中国天气预报员，说话还喜欢用双关语。"""

@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气。"""
    return f"在{city}永远晴朗!"

@tool
def get_user_location() -> str:
    """获取用户位置。"""
    return "广州"

# 使用 ChatOllama（支持 bind_tools）
llm = ChatOllama(model="qwen2.5:7b", temperature=0.9, base_url="http://127.0.0.1:11434")

# 简化 Agent 配置（先测试基础功能）
agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "外面是什么天气?"}]}
)
print(response)
print("="*50)
final_response = response['messages'][-1].content
print(final_response)