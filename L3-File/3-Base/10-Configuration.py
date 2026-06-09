"""
根据外部传入的配置参数，动态切换底层大语言模型（LLM）的 Agent

问题背景: 
    在传统的函数调用中，我们通常通过参数传递变量。
    但在 LangGraph/LangChain 的链式调用中，状态（State）是主要的数据载体。
    如果我们需要在不改变 State 结构的情况下，向节点传递一些“控制指令”或“环境配置”（如选择哪个模型、设置温度、用户ID等），就需要使用 config。
解决方案: 
    LangGraph 允许在 invoke 时传入一个 config 字典。
    其中 config["configurable"] 是一个特殊的键，用于存放那些不保存到检查点（Checkpoint）、仅用于当前执行上下文的可配置字段。

"""

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Sequence
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

# 初始化 LLM
qwen_llm = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

ds_llm = ChatOpenAI(
    model_name="deepseek-r1:8b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]

models = {
    "qwen": qwen_llm,
    "deepseek": ds_llm
}

def _call_model(state: AgentState, config: RunnableConfig):
    model_name = config["configurable"].get("model", "qwen")
    print('model_name(模型名称)：\n',model_name)
    model = models[model_name]
    response = model.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(AgentState)
builder.add_node("model", _call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)

graph = builder.compile()

qwen_config = {"configurable": {"model": "qwen"}}
print("="*30,"千问","="*30)
print(graph.invoke({"messages": [HumanMessage(content="你是谁？你擅长做什么？")]}, config=qwen_config))

ds_config = {"configurable": {"model": "deepseek"}}
print("="*30,"deepseek","="*30)
print(graph.invoke({"messages": [HumanMessage(content="你是谁？你擅长做什么？")]}, config=ds_config))

"""
应用场景:
A/B 测试: 同时运行两个版本的模型对比效果。
成本优化: 简单任务用便宜模型，复杂任务用昂贵模型（可通过前置节点判断复杂度并设置 config）。
多租户支持: 不同用户可能偏好或拥有不同模型的 API Key，可以通过 config 传递用户 ID，节点内部再查找对应用户的模型配置。
环境隔离: 开发环境用 Mock 模型，生产环境用真实模型。
"""

