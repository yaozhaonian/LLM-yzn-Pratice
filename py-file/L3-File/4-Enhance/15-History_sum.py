"""
项目展示：基于摘要的长期记忆管理
在 LangGraph 中实现基于摘要的长期记忆管理（Summarization-based Memory）。
这是一种高级的上下文管理策略，旨在解决长对话中 Token 消耗过大和上下文窗口限制的问题，同时保留关键的历史信息。
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import RemoveMessage
memory = MemorySaver()

model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama" 
)

class State(MessagesState):
    summary: str

def call_model(state: State):
    summary = state.get("summary", "")
    if summary:
        system_message = f"前面的对话摘要:{summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    response = model.invoke(messages)
    print("大模型回复:\n", response)
    return {"messages": [response]}

def should_continue(state: State):
    """返回下一个要执行的节点."""
    messages = state["messages"]
    print("当前对话数:\n", len(messages))
    if len(messages) > 4:  # 假设6个消息是一个长对话
        return "sc"
    else:
        return END

def summarize_conversation(state: State):
    """返回一个摘要。"""
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"这是迄今为止的对话摘要:{summary}\n\n"
            "通过考虑上述新消息来扩展摘要:"
        )
    else:
        summary_message = "创建上述对话的摘要:"   
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    print("大模型摘要:\n", summary)
    del_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": del_messages}

workflow = StateGraph(State)
workflow.add_node("call_model", call_model)
workflow.add_node("sc", summarize_conversation)
workflow.add_edge(START, "call_model")
workflow.add_conditional_edges("call_model", should_continue)

workflow.add_edge("sc", END)

app = workflow.compile(checkpointer=memory)
# app.get_graph().draw_png(output_file_path='./15-History_sum.png')

def print_update(update):
    for k, v in update.items():
        for m in v["messages"]:
            m.pretty_print()
        if "summary" in v:
            print(v["summary"])

config = {"configurable": {"thread_id": "4"}}
# input_mesage = HumanMessage(content="你好，我是小兆")
# input_mesage.pretty_print()
# for event in app.stream({"mesages": [input_mesage]}, config=config, stream_mode="updates"):
#     print_update(event)

while True:
    user_input = input("请输入您的问题：")
    if user_input == "exit":
        break
    input_message = HumanMessage(content=user_input)
    input_message.pretty_print()
    for event in app.stream({"messages": [input_message]}, config=config, stream_mode="updates"):
        print_update(event)


