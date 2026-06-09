"""
使用 LangGraph 结合 MemorySaver 来实现带有**持久化记忆（Persistence）**的对话代理（Agent）。
其核心目标是让 AI 能够记住多轮对话中的上下文信息。
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

memory = MemorySaver()

@tool
def web_search(query: str):
    """网络搜索(以实际情况为准)"""
    if query == "上海":
        return "上海天气晴朗，温度28度，湿度65%"
    return "旅游搜索引擎推荐:https://www.dili360.com/"

tools =[web_search]

tool_node = ToolNode(tools=tools)

model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama" 
)
bound_model = model.bind_tools(tools)

def should_continue(state: MessagesState):
    """判断是否继续执行"""
    last_message = state["messages"][-1]
    print("是否需要调用工具:\n", last_message)
    if not last_message.tool_calls:
        return END
    return "action"

def call_model(state: MessagesState):
    """调用模型"""
    messages = state["messages"]
    result = bound_model.invoke(messages)
    print("调用大模型回复:\n",result)
    return {"messages": result}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_edge(START, "agent")

workflow.add_conditional_edges("agent", should_continue, ["action", END])

workflow.add_edge("action", "agent")
app = workflow.compile(checkpointer=memory)

app.get_graph().draw_png(output_file_path='./11-chat_manage.png')

config = {"configurable": {"thread_id": "1"}}
# input_message = HumanMessage(content="你好,我是小兆,我喜欢看自然风景,希望去洱海、玉龙雪山游玩")

# for event in app.stream({"messages": [input_message]}, config=config, stream_mode="values"):
#     event["messages"][-1].pretty_print()

# input_message = HumanMessage(content="你好，我计划去云南玩几天，帮我规划一下旅游路线")
# for event in app.stream({"messages": [input_message]}, config=config, stream_mode="values"):
#     event["messages"][-1].pretty_print()

# input_message = HumanMessage(content="你好，我计划去新疆玩几天，帮我规划一下旅游路线")
# for event in app.stream({"messages": [input_message]}, config=config, stream_mode="values"):
#     event["messages"][-1].pretty_print()

while True:
    user_input = input("请输入您的问题：")
    if user_input == "exit":
        break
    input_message = HumanMessage(content=user_input)
    for event in app.stream({"messages": [input_message]}, config=config, stream_mode="values"):
        event["messages"][-1].pretty_print()

"""
应用场景
该代码模式适用于以下场景：

智能客服机器人:
需要记住用户在前几轮对话中提供的订单号、问题描述或个人偏好，以便提供连贯的服务。
个人助理 Agent:
如代码所示，记住用户的姓名、喜好、日程安排等个人信息，提供个性化的交互体验。
多轮任务型对话系统:
例如预订机票或酒店，需要在多轮问答中收集完整信息（时间、地点、人数），并保持上下文一致性。
调试与回溯:
通过持久化存储，开发者可以回放整个对话过程，分析 Agent 在每一步的决策依据，便于优化 Prompt 或逻辑。
"""





