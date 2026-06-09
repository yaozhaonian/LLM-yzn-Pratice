from typing import Optional, Literal
from langgraph.graph import StateGraph, END, START, MessagesState, add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

try:
    print("尝试", model.invoke([HumanMessage(content="你好")]))
except:
    pass


def get_response() -> None:
    response = model.invoke([
        SystemMessage(content="You are a helpful assistant!"),
        HumanMessage(content="介绍一下广东美食")
    ])
    print("最终回复:", response.content)


if __name__ == '__main__':
    get_response()






