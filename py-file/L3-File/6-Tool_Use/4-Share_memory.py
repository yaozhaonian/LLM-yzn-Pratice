# 传递共享内存
"""
在 LangGraph 中构建一个支持多用户隔离和**共享内存（Store）**的 ReAct Agent。
核心在于利用 InMemoryStore 存储不同用户的上下文文档，并通过依赖注入机制让工具函数能够访问这些特定于用户的数据。
"""
from langchain.agents import create_agent
from typing import List, Tuple
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, InjectedStore
from langchain_openai import ChatOpenAI
from typing_extensions import Annotated
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver


doc_store = InMemoryStore()

# 为两个不同的用户（ID "1" 和 "2"）分别存储了一条关于 Xiaozhao 公司的文档
namespace = ("documents", "1")
doc_store.put(
    namespace=namespace,
    key="doc_0",
    value={"doc": "Xiaozhao公司刚刚筹集了20亿人民币!"}
)
namespace = ("documents", "2")  # user ID
doc_store.put(
    namespace=namespace,
    key="doc_1",
    value={"doc": "Xiaozhao公司前一个季度盈利了2亿人民币!"}
)

@tool
def get_context(
    question: str,
    config: RunnableConfig, # 参数1：注入运行时配置
    store: Annotated[BaseStore, InjectedStore()], # 参数2：注入文档存储
) -> Tuple[str, List[Document]]:
    """获取回答问题的相关背景"""
    # 从运行时配置中获取 user_id
    user_id = config.get("configurable", {}).get("user_id")
    # 从注入的 store 中根据 user_id 搜索文档
    docs = [item.value["doc"] for item in store.search(("documents", user_id))]
    return "\n\n".join(doc for doc in docs)

print("A"*50)
print(get_context.tool_call_schema.model_json_schema())

tools = [get_context]
tool_node = ToolNode(tools)
checkpointer = MemorySaver()

model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    temperature=0
)

graph = create_agent(model, tools, checkpointer=checkpointer, store=doc_store)
print("B"*50)
# 第一次调用 (user_id="1")
messages = [{"type": "user", "content": "关于Xiaozhao公司有什么最新消息"}]
config = {"configurable": {"thread_id": "1", "user_id": "1"}}
for chunk in graph.stream({"messages": messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

print("C"*50)

# 第二次调用 (user_id="2")
messages = [{"type": "user", "content": "关于Xiaozhao公司有什么最新消息"}]
config = {"configurable": {"thread_id": "2", "user_id": "2"}}
for chunk in graph.stream({"messages": messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()


"""
应用场景
这段代码典型地应用于 多租户 SaaS 应用 或 个性化智能助手 场景，具体包括：

用户数据隔离（Multi-tenancy）：

在同一个 AI 应用中服务于多个用户，确保每个用户只能访问自己的私有数据（如个人文档、历史记录、偏好设置）。
通过 namespace 和 user_id 的结合，实现了逻辑上的数据隔离。
RAG（检索增强生成）的动态上下文注入：

不同于传统的静态向量数据库检索，这里展示了如何将“业务逻辑层”的存储（InMemoryStore）直接注入到工具链中。适用于需要根据用户身份动态加载不同知识片段场景，例如：
企业客服机器人：不同等级的会员或不同部门的员工查询同一问题时，获得基于其权限或部门文档的不同回答。
个人学习助手：为每个学生存储不同的学习笔记，当学生提问时，Agent 仅基于该学生的笔记进行回答。

状态管理与长期记忆分离：
代码区分了 checkpointer（短期/会话级记忆，保存对话历史）和 store（长期/业务级数据，保存文档实体）。这种架构适合需要同时维护对话上下文和外部业务知识库的复杂 Agent 系统。
"""
"""
需求扩展:
doc_store针对用户进行记忆扩写
"""


