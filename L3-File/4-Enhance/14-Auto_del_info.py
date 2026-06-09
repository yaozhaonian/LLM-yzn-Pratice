"""
在 LangGraph 工作流中自动化地管理消息生命周期。
"""
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import RemoveMessage
memory = MemorySaver()


model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama" 
)

def should_continue(state: MessagesState) -> Literal[END, "dm"]:
    """判断是否继续执行"""
    last_message = state["messages"][-1]
    print("是否需要调用工具:\n", last_message)
    if not last_message.tool_calls:
        return "dm"
    return END

def delete_messages(state: MessagesState):
    """删除消息"""
    messages = state["messages"]
    if len(messages) > 2:
        print("删除消息")
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}

def call_model(state: MessagesState):
    """调用模型"""
    messages = state["messages"]
    result = model.invoke(messages)
    print("调用大模型回复:\n",result)
    return {"messages": result}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("dm", delete_messages)
workflow.add_edge(START, "agent")

workflow.add_conditional_edges("agent", should_continue)

workflow.add_edge("dm", END)
app = workflow.compile(checkpointer=memory)

# app.get_graph().draw_png(output_file_path='./14-auto_del_info.png')

config = {"configurable": {"thread_id": "3"}}
input_message = HumanMessage(content="你好，我是小兆")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    print("=======event=======", event)
    print([(message.type, message.content) for message in event["messages"]])

print("=======第2组对话=======")
input_message = HumanMessage(content="我叫什么名字？")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    print([(message.type, message.content) for message in event["messages"]])



"""
应用场景
这种“程序化自动删除”的模式适用于：

无限长对话的低成本维护:

对于闲聊机器人或非任务型助手，用户可能不介意 AI 忘记很久以前的细节。通过自动删除旧消息，可以确保 Token 消耗始终保持在低位，避免超出 LLM 上下文限制。
隐私敏感型应用:

设定严格的保留策略（例如：只保留最近 5 分钟或最近 3 轮对话）。一旦超出范围，系统自动物理删除数据，降低数据泄露风险，符合 GDPR 等隐私法规的最小化原则。
实时性强的任务:

在某些场景下，旧的历史信息不仅无用，反而可能是噪声（例如实时股票分析、即时翻译）。自动清理可以防止旧语境干扰当前判断。
作为更复杂策略的基础:

这是实现**“滑动窗口 + 摘要”**的前奏。通常的高级做法是：在删除旧消息之前，先调用 LLM 生成一段摘要，将摘要作为一条新消息插入，然后再删除原始细节消息。本代码展示了删除这一步骤。
"""




