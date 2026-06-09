from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

memory = MemorySaver()

model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama" 
)

@tool
def web_search(query: str):
    """网络搜索"""
    # 代替实际实现
    return "上海阳光明媚"

tools = [web_search]
tool_node = ToolNode(tools)

class AskHuman(BaseModel):
    """问人类一个问题"""
    question: str


model = model.bind_tools(tools+ [AskHuman])

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    elif last_message.tool_calls[0]["name"] == "AskHuman":
        return "ask_human"
    else:
        return "action"

def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

def ask_human(state):
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    location = interrupt("请提供您的位置:")
    tool_message = [{"tool_call_id": tool_call_id, "type": "tool", "content": location}]
    return {"messages": tool_message}


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_node("ask_human", ask_human)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
)
workflow.add_edge("action", "agent")
workflow.add_edge("ask_human", "agent")


app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "2"}}
for event in app.stream(
    {
        "messages": [
            (
                "user",
                "询问用户他们在哪里，然后使用搜索工具查看那里的天气",
            )
        ]
    },
    config,
    stream_mode="values",
):
    # event["messages"][-1].pretty_print()
    if "messages" in event and event["messages"]:
        event["messages"][-1].pretty_print()
    else:
        print(event)


for event in app.stream(Command(resume="上海"), config, stream_mode="values"):
    event["messages"][-1].pretty_print()