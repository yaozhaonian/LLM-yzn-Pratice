"""
在 LangGraph 中构建一个支持 多轮对话（Multi-turn Conversation） 和 人机协同（Human-in-the-Loop） 的多智能体系统。
"""
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.prebuilt import ToolRuntime
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal
import random
import uuid

model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    temperature=0.0 # 降低温度以提高计算确定性
)

@tool
def get_travel_recommendations():
    """获取加勒比海温暖地区的旅行目的地推荐，无需传入任何参数。"""
    return random.choice(["aruba", "turks and caicos"])

@tool
def get_hotel_recommendations(location: Literal["aruba", "turks and caicos"]):
    """获取指定目的地的精品酒店推荐"""
    return {
        "aruba": [
            "The Ritz-Carlton, Aruba (Palm Beach)",
            "Bucuti & Tara Beach Resort (Eagle Beach)"
        ],
        "turks and caicos": ["Grace Bay Club", "COMO Parrot Cay"]
    }[location]

def make_handoff_tool(*, agent_name: str):
    tool_name = f"transfer_to_{agent_name}"
    @tool(tool_name)
    def handoff_to_agent(
        runtime: ToolRuntime,  # 👈 替换旧版 InjectedState + InjectedToolCallId
    ):
        """向其他代理寻求帮助。"""
        state = runtime.state
        tool_call_id = runtime.tool_call_id

        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": tool_name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={"messages": state["messages"] + [tool_message]},
        )
    return handoff_to_agent

# --- Agent ---
travel_advisor = create_agent(
    model,
    tools=[get_travel_recommendations, make_handoff_tool(agent_name="hotel_agent")],
    state_schema=MessagesState,
    system_prompt=(
        "你是旅游顾问，负责推荐目的地和当地活动。\n"
        "问酒店 → 先说明要转给酒店专家，再调用 get_hotel_recommendations。\n"
        "不要提工具，回答自然。"
    ),
    name="travel_agent"
)

hotel_advisor = create_agent(
    model,
    tools=[get_hotel_recommendations, make_handoff_tool(agent_name="travel_agent")],
    state_schema=MessagesState,
    system_prompt=(
        "你是酒店专家，只推荐酒店。\n"
        "非酒店问题 → 说明要转回旅游顾问，再调用 get_travel_recommendations。\n"
        "回答清晰。"
    ),
    name="hotel_agent"
)

def call_travel_advisor(state: MessagesState) -> Command | dict:
    """调用 travel_advisor 子图。如果它交接，则传递命令；否则，转到 human 节点。"""
    response = travel_advisor.invoke(state)
    # 如果 travel_advisor 决定交接，它会返回一个 Command。我们必须直接传递它。
    if isinstance(response, Command):
        return response
    # 否则，它返回一个包含新消息的字典。我们将其更新并转到 human 节点。
    return Command(update=response, goto="human")

def call_hotel_advisor(state: MessagesState) -> Command | dict:
    """调用 hotel_advisor 子图。如果它交接，则传递命令；否则，转到 human 节点。"""
    response = hotel_advisor.invoke(state)
    if isinstance(response, Command):
        return response
    return Command(update=response, goto="human")

def human_node(state: MessagesState) -> Command:
    """用于收集用户输入的节点。通过检查消息历史来确定将控制权交还给哪个智能体。"""
    # 1. 使用 interrupt() 暂停图的执行，等待用户输入
    user_input = interrupt(value="等待用户输入")

    # 2. 从消息历史中确定上一个活动的智能体
    last_active_agent = ""
    # 从后往前遍历消息
    for i in range(len(state["messages"]) - 1, -1, -1):
        msg = state["messages"][i]
        # create_react_agent 会给它的 AI 消息添加一个 'name' 字段
        if msg.type == "ai" and msg.name:
            last_active_agent = msg.name
            break
            
    # 如果找不到，就回到初始智能体（作为备用方案）
    if not last_active_agent:
        # 在我们的图中，我们总是从 travel_advisor 开始
        last_active_agent = "travel_advisor" 

    # 3. 构造 Command，更新消息历史，并将控制权交还给上一个活动的智能体
    return Command(
        update={
            "messages": [
                {
                    "role": "human",
                    "content": user_input,
                }
            ]
        },
        goto=last_active_agent,  # 动态地将流程导回正确的智能体
    )

# --- 图的构建 ---
builder = StateGraph(MessagesState)
builder.add_node("travel_agent", call_travel_advisor)
builder.add_node("hotel_agent", call_hotel_advisor)
builder.add_node("human", human_node)
builder.add_edge(START, "travel_agent")
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
graph.get_graph().draw_png(output_file_path='./4-agent_multi_chat.png')

# --- 运行对话 ---
thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

inputs = [
    {
        "messages": [
            # 使用 'user' 作为 role，这在 LangChain 中会自动映射到 HumanMessage
            {"role": "user", "content": "我想去加勒比海某个温暖的地方"}
        ]
    },
    Command(
        resume="您能否推荐一个该地区的不错的酒店并告诉我它是哪个地区？"
    ),
    Command(
        resume="我喜欢第一个。你能推荐一些酒店附近可以做的事情吗？"
    ),
]

for idx, user_input in enumerate(inputs):
    print()
    print(f"--- Conversation Turn {idx + 1} ---")
    print()
    if isinstance(user_input, Command):
        print(f"User: {user_input.resume}")
    else:
        # 确保从初始字典中正确打印
        print(f"User: {user_input['messages'][0]['content']}")
    print()
    for update in graph.stream(
        user_input,
        config=thread_config,
        stream_mode="updates",
    ):
        for node_id, value in update.items():
            if isinstance(value, dict) and value.get("messages", []):
                last_message = value["messages"][-1]
                
                # 检查 last_message 是对象还是字典，并相应地获取其角色和内容
                is_ai = False
                content = ""

                if isinstance(last_message, dict):
                    # 如果是字典，通过键来访问
                    if last_message.get("role") == "ai":
                        is_ai = True
                    content = last_message.get("content", "")
                else:
                    # 如果是对象，通过属性来访问
                    if hasattr(last_message, 'type') and last_message.type == "ai":
                        is_ai = True
                    if hasattr(last_message, 'content'):
                        content = last_message.content

                # 只打印有内容的 AI 消息
                if is_ai and content:
                    print(f"{node_id}: {content}")



"""
应用场景
这段代码适用于 需要复杂交互、多领域专家协作且必须由用户主导节奏的生产级聊天机器人：

1.高级旅行规划助手：
场景：用户先问目的地，再问酒店，接着问当地美食，最后问航班。
应用：不同的专家（旅行、酒店、美食、交通）各司其职。human_node 确保用户每次发言后，系统都能准确地回到上下文相关的专家那里，或者在专家之间自动流转。

2.企业级客户支持门户：
场景：用户咨询产品技术细节，然后转向 billing（账单）问题，最后又回到技术支持。
应用：技术专家和账单专家作为独立节点。当用户切换话题时，系统能识别意图并路由；当用户在同一个话题下追问时，human_node 能记住上次是谁服务的，保持对话连贯性。

3.人机协同审核流程：
场景：AI 生成草稿，人类修改，AI 再润色。
应用：interrupt 机制允许在关键步骤暂停，让人类介入检查或提供额外信息，然后 AI 继续工作。

4.解决“路由震荡”问题：
痛点：在多智能体系统中，如果每次用户说话都重新经过一个中央路由器（Router），可能会因为上下文理解偏差导致路由错误。
解决方案：本代码采用的“回传给上一个活跃 Agent”策略（Last Active Agent Routing）是一种简单而有效的启发式方法，极大地提高了多轮对话的稳定性。

"""


