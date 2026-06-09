"""
在 LangGraph 中处理工具选择失败或需要动态调整工具集的复杂场景。
引入了两个关键的高级概念：重试策略（Retry Policy） 和 基于反馈的工具重新检索（Re-retrieval）。
"""
import re
import uuid
from langchain_ollama import OllamaEmbeddings
from typing import Annotated, TypedDict
from langchain_core.tools import StructuredTool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    temperature=0.8
)

def create_tool(company: str) -> dict:
    formatted_company = re.sub(r"[^\w\s]", "", company).replace(" ", "_")
    def company_tool(year: int) -> str:
        return f"{company} had revenues of $100 in {year}."
    return StructuredTool.from_function(
        company_tool,
        name=formatted_company,
        description=f"Information about {company}",
    )

s_and_p_500_companies = [
    "3M", "A.O. Smith", "Abbott", "Accenture",
    "Advanced Micro Devices", "Yum! Brands",
    "Zebra Technologies", "Zimmer Biomet", "Zoetis",
]

tool_registry = {
    str(uuid.uuid4()): create_tool(company) for company in s_and_p_500_companies
}

tool_documents = [
    Document(
        page_content=tool.description,
        id=id,
        metadata={"tool_name": tool.name},
    )
    for id, tool in tool_registry.items()
]

vector_store = InMemoryVectorStore(embedding=OllamaEmbeddings(model="bge-m3:latest", base_url="http://127.0.0.1:11434"))
document_ids = vector_store.add_documents(tool_documents)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_tools: list[str]
    retry_count: int 

builder = StateGraph(State)

def select_tools(state: State):
    last_message = state["messages"][-1]
    retry_count = state.get("retry_count", 0)
    
    # 1. 先判断是否需要增加计数
    is_retry = isinstance(last_message, AIMessage) and not last_message.tool_calls
    if is_retry:
        retry_count += 1
        print(f"--- 状态更新：重试次数增加到 {retry_count} ---")
        
    # 2. 确定查询词
    query = ""
    if isinstance(last_message, HumanMessage):
        query = last_message.content
    elif is_retry:
        query = state["messages"][0].content 
        print("--- 反思：上次未找到合适工具，重新检索 ---")
    else:
        query = state["messages"][0].content

    # 3. 向量检索
    docs = vector_store.similarity_search(query, k=3)
    selected_ids = [doc.id for doc in docs]
    
    # 4. 模拟错误：只在第一次尝试 (retry_count == 0) 时移除
    # 注意：如果是重试，retry_count 已经是 1 或更多了，所以不会移除
    if retry_count == 0 and "AMD" in query:
        amd_id = None
        for id, t in tool_registry.items():
            if t.name == "Advanced_Micro_Devices":
                amd_id = id
                break
        if amd_id in selected_ids:
            selected_ids.remove(amd_id)
            print("--- 调试：模拟第一次检索错误，故意移除 AMD 工具 ---")
        
    selected_names = [tool_registry[id].name for id in selected_ids]
    print(f"--- 检索到的候选工具: {selected_names} ---")
    
    return {"selected_tools": selected_ids, "retry_count": retry_count}

def agent_node(state: State):
    selected_tool_ids = state["selected_tools"]
    if not selected_tool_ids:
        return {"messages": [AIMessage(content="未找到任何可用工具。")]}
        
    current_tools = [tool_registry[id] for id in selected_tool_ids]
    bound_model = model.bind_tools(current_tools)
    response = bound_model.invoke(state["messages"])
    return {"messages": [response]}

# 关键：检查是否需要重试
def should_retry(state: State):
    last_message = state["messages"][-1]
    retry_count = state.get("retry_count", 0)
    
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        # 如果上一条是 ToolMessage，说明刚执行完工具，这是最终总结，结束
        if len(state["messages"]) > 1 and isinstance(state["messages"][-2], ToolMessage):
            return END
            
        if retry_count < 2: 
            print(f"--- 决定：LLM 未调用工具，触发重试 (当前次数: {retry_count}) ---")
            return "retry" # 返回字符串路由
        else:
            print("--- 决定：重试次数耗尽，结束 ---")
            return END
    
    return "tools"

builder.add_node("select_tools", select_tools)
builder.add_node("agent", agent_node)
tool_node = ToolNode(tools=list(tool_registry.values()))
builder.add_node("tools", tool_node)

# 路由逻辑
builder.add_edge(START, "select_tools")
builder.add_edge("select_tools", "agent")

# 【关键修改】使用 path_map 映射字符串 "retry" 到 "select_tools"
builder.add_conditional_edges("agent", should_retry, path_map={
    "retry": "select_tools",
    "tools": "tools",
    END: END
})

builder.add_edge("tools", "agent")

graph = builder.compile()

# 测试
user_input = "使用中文回答:你能给我一些关于 2022 年 AMD 的信息吗?"
print(f"User: {user_input}")
result = graph.invoke({"messages": [("user", user_input)], "retry_count": 0})

for message in result["messages"]:
    message.pretty_print()



"""
应用场景
主要应用于 高可靠性要求 和 复杂多步推理 的 Agent 场景：

容错性强的企业级助手：

场景：用户询问复杂问题，涉及多个模糊实体。
应用：如果第一次检索到的工具无法回答问题（例如返回“数据不存在”或报错），Agent 不会直接放弃或胡编乱造，而是回到 select_tools 节点，利用 LLM 分析失败原因，生成新的查询，重新检索更合适的工具。
动态探索性任务：

场景：数据分析或科学研究，其中下一步需要什么工具取决于上一步的结果。
应用：例如，先查询“某公司收入”，发现数据缺失，然后重新检索“某公司财报下载”工具。这种“执行-反思-再检索”的循环是自主 Agent 的核心特征。
处理检索噪声：

场景：向量检索并不完美，可能会遗漏相关工具或返回无关工具。
应用：通过引入 LLM 生成的修正查询（Refined Query），可以弥补向量检索在语义理解上的不足，提高最终工具选择的准确率。

"""











