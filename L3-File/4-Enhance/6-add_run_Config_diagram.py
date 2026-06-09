# 将运行时配置添加到图中

from typing import Annotated, Optional, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.runnables.config import RunnableConfig


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
    "ds": ds_llm,
    "qwen": qwen_llm,
}

class ConfigSchema(TypedDict):
    model: Optional[str]
    system_message: Optional[str]


def _call_model(state: AgentState, config: RunnableConfig) -> AgentState:
    model_name = config["configurable"].get("model", "qwen")    # 默认是千问
    model = models[model_name]
    messages = state["messages"]
    if "system_message" in config["configurable"]:
        messages = [SystemMessage(content=config["configurable"]["system_message"])] + messages
    response = model.invoke(messages)
    return {"messages": [response]}

builder = StateGraph(AgentState)
builder.add_node("model", _call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)

graph = builder.compile()
# graph.get_graph().draw_mermaid_png(output_file_path="./6-arcd.png")

config = {"configurable": {"model": "ds"}}
print(graph.invoke({"messages": [HumanMessage(content="你是谁?")]}, config=config))

config = {"configurable": {"system_message": "你是一个优秀的AI助手"}}
print(graph.invoke({"messages": [HumanMessage(content="你是谁?")]}, config=config))

"""
应用场景
 “运行时配置”机制极大地提升了 LangGraph 应用的灵活性和可维护性，主要应用于以下场景：

(1) A/B 测试与模型对比
场景: 你想比较不同 LLM（如 GPT-4 vs Claude 3 vs Qwen）在特定任务上的表现。
应用: 无需编写多个图或硬编码模型名称。只需定义一个通用的图结构，然后在调用时通过 config={"configurable": {"model": "gpt-4"}} 或 {"model": "claude-3"} 来切换底层模型。这使得实验和数据收集变得非常简单。
(2) 多租户 SaaS 应用
场景: 你开发了一个 AI 客服平台，供多个公司使用。不同公司可能希望使用不同的模型以控制成本，或者拥有不同的品牌语调（系统提示词）。
应用:
模型隔离: 根据用户所属的公司 ID，在中间件层解析出对应的模型名称，并通过 config 传入。
个性化提示词: 每个公司可以自定义其 Agent 的系统提示词（例如：“你是由 Company A 提供的友好助手”）。这些提示词存储在数据库中，在每次请求时动态加载并注入到 config["configurable"]["system_message"] 中。
(3) 动态角色扮演的 Chatbot
场景: 一个游戏 NPC 或教育助手，需要根据上下文扮演不同角色（老师、反派、向导）。
应用: 前端或上游逻辑根据当前剧情阶段，确定 NPC 的角色设定。在调用 LangGraph 时，将该角色的具体指令作为 system_message 传入。这样，同一个图逻辑可以服务于无数种不同的角色，无需为每个角色重写代码。
(4) 环境特定的配置 (Dev/Prod)
场景: 在开发环境中，你可能希望使用便宜、快速的模型进行测试；在生产环境中，使用高精度、高成本的模型。
应用: 通过环境变量或部署配置来决定传入的 config。例如，在测试脚本中始终传入 {"model": "fast-model"}，而在生产 API 中传入 {"model": "smart-model"}。
(5) 用户级别的偏好设置
场景: 允许最终用户在界面上选择他们喜欢的回答风格（如“简洁”、“详细”、“幽默”）。
应用: 将这些风格偏好转化为具体的 system_message 片段，并在每次用户发起对话时，将其注入到 config 中。这样，Agent 的行为会实时响应用户的个性化设置。
"""





