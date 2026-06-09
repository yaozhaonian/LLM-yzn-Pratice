# 将图状态传递给工具
from langchain.agents import create_agent, AgentState
from typing import List
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, InjectedState
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import Annotated

class State(AgentState):
    docs: List[str]

@tool
def get_context(state: Annotated[dict, InjectedState]):
    """获取回答问题的相关背景"""
    return "\n\n".join(doc for doc in state["docs"])

tools = [get_context]
tool_node = ToolNode(tools)
checkpointer = MemorySaver()

model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    temperature=0
)


agent = create_agent(
    model,
    tools,
    state_schema = State,
    checkpointer = checkpointer
)

docs = [
    "Xiaozhao公司刚刚筹集了10亿美元!",
    "Xiaozhao公司成立于2019年"
]

inputs = {
    "messages": [{"type": "user", "content": "关于Xiaozhao有什么最新消息"}],
    "docs": docs,
}

config = {"configurable": {"thread_id": "1"}}
for chunk in agent.stream(inputs, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()


"""
应用场景说明
这段代码主要应用于 需要动态上下文感知的智能体开发，具体场景包括：

企业级 RAG 问答系统：

在传统 RAG 中，检索步骤通常在链的外部完成。而在 LangGraph 中，可以将检索到的文档放入 State。
当模型需要更多细节时，它可以调用工具，该工具直接从 State 中读取已检索的文档，避免重复检索或传递大量冗余参数。
多步推理与状态共享：

在复杂的工作流中，前一个节点可能已经计算或检索了一些数据（如用户画像、历史订单、实时股价）。
后续的工具函数可以通过 InjectedState 无缝访问这些中间状态，无需在每次工具调用时手动序列化并传递所有上下文。
简化工具接口设计：

开发者无需在每个工具的签名中显式添加 context 或 history 参数。
通过依赖注入机制，工具只关注其核心逻辑（如“格式化文档”），而上下文获取由框架自动处理，提高了代码的可维护性和解耦性。
"""
