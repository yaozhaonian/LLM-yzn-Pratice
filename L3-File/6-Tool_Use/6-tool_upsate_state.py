"""
在 LangGraph 中如何使用 Command 对象从工具（Tool）内部直接更新图的状态（State）。
允许工具不仅返回数据供 LLM 处理，还能主动修改全局状态（如用户信息、记忆等），从而实现更复杂的控制流。
工具执行 -> 更新全局状态 -> LLM 基于新状态继续推理
"""
from langgraph.types import Command
from langchain_core.tools.base import InjectedToolCallId
from typing import Any, Annotated
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    temperature=0
)

USER_INFO = [
    {"user_id": "1", "name": "Bob Dylan", "location": "New York, US", "activity": "musician"},
    {"user_id": "2", "name": "Taylor Swift", "location": "广州, 中国", "activity": "演唱会,萤火虫漫展"},
]

USER_ID_TO_USER_INFO = {info["user_id"]: info for info in USER_INFO}

class State(MessagesState):
    user_info: dict[str, Any] = {}

@tool
def lookup_user_info(
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig
):
    """
    获取当前用户的个人资料，包括姓名和所在城市/位置。
    当用户询问关于本地推荐、天气、或基于位置的服务时，必须首先调用此工具。
    """
    user_id = config.get("configurable", {}).get("user_id")
    if user_id is None:
        raise ValueError("请提供用户ID")

    if user_id not in USER_ID_TO_USER_INFO:
        raise ValueError(f"用户 '{user_id}' 没找到")

    user_info = USER_ID_TO_USER_INFO[user_id]
    return Command(
        update={
            "user_info": user_info,
            "messages": [
                ToolMessage(
                    content=f"已加载用户信息: {user_info['name']} ({user_info['location']})",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )

tools = [lookup_user_info]
model_with_tools = model.bind_tools(tools)

# 4. 定义节点
def call_model(state: State):
    messages = state["messages"]
    # 可以在这里动态注入系统提示词，利用 state["user_info"]
    system_prompt = "你是一个智能助手。如果已知用户信息，请基于用户信息回答。"
    print("当前状态:\n", state)
    if state.get("user_info"):
        system_prompt += f"\n当前用户信息: {state['user_info']}"
    
    # 注意：bind_tools 后通常不需要手动加 system prompt 到 messages 列表，
    # 除非你使用 ChatPromptTemplate。这里为了简单，我们假设模型能记住上下文。
    # 更严谨的做法是使用 ChatPromptTemplate
    
    response = model_with_tools.invoke(messages)
    print("大模型回复:\n", response)
    return {"messages": [response]}

def should_continue(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

# 5. 构建工作流
workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent") # 工具执行后回到 agent

graph = workflow.compile()

# --- 测试 ---

print("="*20 + " 测试 User 1 " + "="*20)
config_1 = {"configurable": {"user_id": "1"}}
inputs_1 = {"messages": [("user", "你好，五一期间我在家附近有什么活动可以参加？")]}

for event in graph.stream(inputs_1, config_1, stream_mode="values"):
    last_msg = event["messages"][-1]
    if isinstance(last_msg, AIMessage):
        print(f"AI: {last_msg.content}")
    elif hasattr(last_msg, 'content'):
        print(f"Tool/System: {last_msg.content}")
    
    # 打印状态中的 user_info
    if "user_info" in event and event["user_info"]:
        print(f"[状态更新] user_info: {event['user_info']}")

print("\n" + "="*20 + " 测试 User 2 " + "="*20)
config_2 = {"configurable": {"user_id": "2"}}
inputs_2 = {"messages": [("user", "你好，五一期间我在家附近有什么活动可以参加？")]}

for event in graph.stream(inputs_2, config_2, stream_mode="values"):
    last_msg = event["messages"][-1]
    if isinstance(last_msg, AIMessage):
        print(f"AI: {last_msg.content}")
    elif hasattr(last_msg, 'content'):
        print(f"Tool/System: {last_msg.content}")
        
    if "user_info" in event and event["user_info"]:
        print(f"[状态更新] user_info: {event['user_info']}")



"""
应用场景
主要应用于需要 动态上下文加载 和 状态驱动决策 的复杂 Agent 场景：

1.个性化推荐系统：
场景：用户询问“推荐附近的餐厅”或“周末活动”。
应用：Agent 不需要用户在每次提问时都重复提供位置或个人偏好。通过工具首次调用加载用户画像（user_info）到状态中，后续的推理节点可以直接利用这些结构化数据进行精准推荐。

2.隐私保护与数据最小化：
场景：处理敏感用户数据（如医疗记录、金融信息）。
应用：工具可以将敏感数据存储在状态的私有字段中（user_info），而只向 LLM 返回一个简单的确认消息（ToolMessage）。LLM 可以通过其他机制（如自定义节点逻辑）访问状态中的数据进行推理，或者仅在必要时通过特定的 Prompt 模板引用状态字段，避免将所有敏感数据明文暴露在 LLM 的对话历史中。

3.多步工作流中的状态同步：
场景：复杂的业务流程，如订单处理、客户支持。
应用：当用户提供一个订单号时，工具可以查询数据库并将订单详情写入状态。后续的验证节点、物流节点都可以直接从状态中读取订单详情，无需重复查询数据库，提高了效率并保证了数据的一致性。

4.解耦数据获取与逻辑推理：
场景：需要将数据检索逻辑与 AI 推理逻辑分离。
应用：工具专门负责“获取数据并更新状态”，而 LLM 专门负责“基于状态进行推理和生成回答”。这种职责分离使得系统更易于维护和测试。
"""

