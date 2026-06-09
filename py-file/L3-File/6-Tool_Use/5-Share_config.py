"""
在 LangGraph 中将运行时配置（RunnableConfig）传递给工具（Tools），从而实现基于用户上下文的状态管理。
"""
# 有些问题,回头再改看看
from langchain.agents import create_agent
from typing import List
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
model = ChatOpenAI(
    model_name="qwen2.5:7b",    # 在本地不够强大的模型一般需要去手动构造AIMessage的tool_calls字段
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    temperature=0
)

# 全局字典，用于存储用户与宠物的关系
user_to_pets = {}

# 定义更新喜爱宠物的工具
@tool(parse_docstring=True)
def update_favorite_pets(
    pets: List[str],
    config: RunnableConfig,
):
    """
    添加喜爱的宠物列表
    
    Args:
        pets: 要设置的喜爱宠物列表
    """
    print(">>> [工具执行] 进入 update_favorite_pets")
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "错误：未提供用户ID"
    user_to_pets[user_id] = pets
    print(f">>> [工具执行] 用户 {user_id} 的宠物已更新为: {pets}")
    return f"成功将 {pets} 设置为用户 {user_id} 的喜爱宠物。"
    
@tool
def list_favorite_pets(config: RunnableConfig):
    """列出喜爱的宠物。"""
    print(">>> [工具执行] 进入 list_favorite_pets")
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "错误：未提供用户ID"
    pets = user_to_pets.get(user_id, [])
    result = ", ".join(pets) if pets else "暂无记录"
    print(f">>> [工具执行] 查询用户 {user_id} 的宠物: {result}")
    return result

# 定义删除喜爱宠物的工具
@tool
def delete_favorite_pets(config: RunnableConfig) -> None:
    """删除喜爱的宠物列表。"""
    # 从配置中获取用户ID
    user_id = config.get("configurable", {}).get("user_id")
    print("看看config:\n", config)
    print(f"需要删除的用户id是:{user_id}")
    # 如果用户存在，则删除其宠物列表
    if user_id in user_to_pets:
        print(f"找到用户id{user_id}并删除")
        del user_to_pets[user_id]

# 定义列出喜爱宠物的工具
@tool
def list_favorite_pets(config: RunnableConfig):
    """当被询问时列出喜爱的宠物。"""
    # 从配置中获取用户ID
    user_id = config.get("configurable", {}).get("user_id")
    # 返回用户喜爱的宠物列表，用逗号连接
    return ", ".join(user_to_pets.get(user_id, []))

tools = [update_favorite_pets, list_favorite_pets, delete_favorite_pets, list_favorite_pets]
model_with_tools = model.bind_tools(tools)

def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    print(">>> [Agent] 调用模型...", response)
    
    # 调试：打印模型是否生成了 tool_calls
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f">>> [Agent] 模型生成工具调用: {response.tool_calls}")
    else:
        print(f">>> [Agent] 模型未生成工具调用，内容: {response.content[:50]}...")
        
    return {"messages": [response]}

def should_continue(state: MessagesState) -> str:
    message = state["messages"]
    last_message = message[-1]
    # 检查是否有工具调用
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

def tool_to_agent(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    print("调用完工具后调用模型进行回答:", response)
    return {"messages": [response]} 

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools)) # 使用标准的 ToolNode
workflow.add_node("tool_to_agent", tool_to_agent)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "tool_to_agent") # 工具执行完后回到 tool_to_agent 进行总结

graph = workflow.compile()
# graph.get_graph().draw_mermaid_png(output_file_path="./5-Share_config.png")

user_to_pets.clear()

# 测试1：设置喜爱的宠物
print("测试1：设置喜爱的宠物:")
print(f"运行前的用户信息: {user_to_pets}")

print("="*20 + " 测试1：设置喜爱的宠物 " + "="*20)
user_to_pets.clear() # 清空数据

inputs = {"messages": [HumanMessage(content="我最喜欢的宠物是狗和金鱼")]}
config = {"configurable": {"user_id": "user1"}}

try:
    for event in graph.stream(inputs, config, stream_mode="values"):
        last_msg = event["messages"][-1]
        last_msg.pretty_print()
except Exception as e:
    print(f"发生错误: {e}")

print(f"运行后的用户信息: {user_to_pets}")
print("\n")

print("="*20 + " 测试2：查询喜爱的宠物 " + "="*20)
inputs2 = {"messages": [HumanMessage(content="我最喜欢的宠物是什么？")]}
try:
    for event in graph.stream(inputs2, config, stream_mode="values"):
        last_msg = event["messages"][-1]
        last_msg.pretty_print()
except Exception as e:
    print(f"发生错误: {e}")

print(f"运行后的用户信息: {user_to_pets}")
print("B"*50)

# 测试3：删除喜爱的宠物信息
print("测试3：删除喜爱的宠物信息:")
print(f"运行前的用户信息: {user_to_pets}")

inputs = {
    "messages": [
        HumanMessage(content="请忘记我告诉你的我最喜欢的动物")
    ]
}
for chunk in graph.stream(
    inputs, config=config, stream_mode="values"
):
    chunk["messages"][-1].pretty_print()

print(f"运行后的用户信息: {user_to_pets}")
print("C"*50)

# 测试4：查询喜爱的宠物
print("测试4：查询喜爱的宠物:")
print(f"运行前的用户信息: {user_to_pets}")

inputs = {"messages": [HumanMessage(content="我最喜欢的宠物是什么？")]}
for chunk in graph.stream(
    inputs, config=config, stream_mode="values"
):
    chunk["messages"][-1].pretty_print()

print(f"运行后的用户信息: {user_to_pets}")

"""
应用场景
主要应用于需要 用户级状态管理 且 工具逻辑依赖用户身份 的场景：

个性化用户偏好设置：

适用于聊天机器人、智能助手等应用，其中用户希望保存自己的喜好（如喜欢的音乐类型、饮食禁忌、常用地址等）。
通过 config 传递 user_id，确保不同用户调用同一个工具接口时，操作的是各自独立的数据空间，避免数据混淆。
无状态服务的状态保持：

在微服务或云函数环境中，工具函数本身通常是无状态的。通过 LangGraph 的 RunnableConfig 机制，可以在不修改工具函数签名的前提下，动态注入上下文信息（如租户ID、会话ID、权限令牌等），实现逻辑上的状态保持。
隐私合规与数据隔离：

当应用需要符合 GDPR 或其他隐私法规时，必须严格隔离用户数据。此模式确保了“删除权”（Right to be Forgotten）的实现，如测试3所示，用户可以明确要求删除其个人数据，而工具能通过 user_id 精准定位并清除数据。
简化 LLM 交互复杂度：

通过将 config 设为隐藏参数，LLM 只需要关注业务参数（如 pets 列表），而不需要关心“我是谁”或“我在操作哪个用户的数据”。这降低了 Prompt 工程的复杂度，提高了工具调用的稳定性。
"""


