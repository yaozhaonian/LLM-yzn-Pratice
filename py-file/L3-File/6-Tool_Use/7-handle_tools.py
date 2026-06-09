"""
处理**大规模工具集（Large Toolsets）**的问题。
当工具数量过多（例如成百上千个）时，直接将所有工具传递给 LLM 会导致上下文窗口溢出、推理速度变慢以及准确率下降
采用 “检索-选择-执行” (Retrieve-Select-Execute) 的模式：
    先通过向量检索筛选出与用户问题最相关的少量工具，再将这些工具动态绑定给 LLM 进行调用。
"""
import re
import uuid
from langchain_ollama import OllamaEmbeddings
from typing import Annotated, TypedDict
from langchain_core.tools import StructuredTool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    temperature=0.8
)

def create_tool(company: str) -> dict:
    """为占位符工具创建架构."""
    # 删除工具名称中的非字母数字字符，并用下划线替换空格
    formatted_company = re.sub(r"[^\w\s]", "", company).replace(" ", "_")

    def company_tool(year: int) -> str:
        # 返回公司和年份的静态收入信息的占位符函数
        return f"{company} had revenues of $100 in {year}."

    return StructuredTool.from_function(
        company_tool,
        name=formatted_company,
        description=f"Information about {company}",
    )

# 示例数据
s_and_p_500_companies = [
    "3M",
    "A.O. Smith",
    "Abbott",
    "Accenture",
    "Advanced Micro Devices",
    "Yum! Brands",
    "Zebra Technologies",
    "Zimmer Biomet",
    "Zoetis",
]

# 为每个公司创建一个工具，并将其存储在具有唯一 UUID 作为密钥的注册表中
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

vector_store = InMemoryVectorStore(embedding=OllamaEmbeddings(model="bge-m3:latest", base_url="http://127.0.0.1:11434" ))
document_ids = vector_store.add_documents(tool_documents)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_tools: list[str] # 存储选定工具的ID

builder = StateGraph(State)

def select_tools(state: State):
    last_user_message = state["messages"][-1]
    query = last_user_message.content
    # 在向量存储中搜索与用户问题最相关的工具描述
    tool_documents = vector_store.similarity_search(query)
     # 【调试用】打印检索到的工具名称
    selected_ids = [doc.id for doc in tool_documents]
    selected_names = [tool_registry[id].name for id in selected_ids]
    print(f"--- 检索到的候选工具: {selected_names} ---")
    
    # 提取相关工具ID，并更新到状态的selected_tools 中
    return {"selected_tools": [doc.id for doc in tool_documents]}

tools = list(tool_registry.values()) # 包含了所有工具，但agent会动态选择

def agent(state: State):
    # 从状态中获取当前轮次被选中的工具ID
    selected_tool_ids = state["selected_tools"]
    # 根据ID从完整的工具注册表中获取实际的工具对象
    current_tools = [tool_registry[id] for id in selected_tool_ids]
    # 将这些选中的工具绑定到LLM，LLM在这次调用中只会看到这些工具
    select_tool = model.bind_tools(current_tools)
    
    # 调用LLM，只使用选中的工具
    return {"messages": [select_tool.invoke(state["messages"])]}
    

builder.add_node("select_tools", select_tools)
builder.add_node("agent", agent)

tool_node = ToolNode(tools=tools)
builder.add_node("tools", tool_node)

builder.add_edge(START, "select_tools")
builder.add_conditional_edges("agent", tools_condition, path_map=["tools", "__end__"])
builder.add_edge("select_tools", "agent")
builder.add_edge("tools", "agent")

graph = builder.compile()
graph.get_graph().draw_png(output_file_path='./7-handle_tools.png')

user_input = "你能给我一些关于 2022 年 AMD 的信息吗?"
print(user_input)

result = graph.invoke({"messages": [("user", user_input)]})
print(result["selected_tools"])


for message in result["messages"]:
    message.pretty_print()

print("A"*50)
user_input = "Tesla 公司 2022 年收入多少？"

result = graph.invoke({"messages": [("user", user_input)]})
print(result["selected_tools"])


for message in result["messages"]:
    message.pretty_print()


"""
--- 检索到的候选工具: ['Advanced_Micro_Devices', '3M', 'Accenture', 'Yum_Brands'] ---
['9603b053-0197-4c53-ba14-fc957c0f1a27', '8a7b6174-6359-4eee-9b54-f486746a0b42', '5bc895f4-f5ae-4780-beb7-77ecaa5a6c58', 'ab45d56b-d7d4-4d44-b5d0-409d1ab50ef8']
================================ Human Message =================================

你能给我一些关于 2022 年 AMD 的信息吗?
================================== Ai Message ==================================
Tool Calls:
  Advanced_Micro_Devices (call_p0gfrg12)
 Call ID: call_p0gfrg12
  Args:
    year: 2022
================================= Tool Message =================================
Name: Advanced_Micro_Devices

Advanced Micro Devices had revenues of $100 in 2022.
================================== Ai Message ==================================

根据提供的信息，在2022年，AMD的营收为100亿美元。请注意这里的数字可能是一个示例数据，并非实际财报中的具体数值。如果您需要更详细的信息或者最新的财务报告，请告知我，我可以帮助进一步查找。
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
--- 检索到的候选工具: ['Zebra_Technologies', 'Accenture', '3M', 'Yum_Brands'] ---
['8b90ee3e-b80b-4d44-becf-a8b1ce0dfd36', '5bc895f4-f5ae-4780-beb7-77ecaa5a6c58', '8a7b6174-6359-4eee-9b54-f486746a0b42', 'ab45d56b-d7d4-4d44-b5d0-409d1ab50ef8']
================================ Human Message =================================

Tesla 公司 2022 年收入多少？
================================== Ai Message ==================================

对不起，目前提供的工具中没有关于特斯拉（Tesla）的直接信息。这些工具分别是关于 Zebra Technologies、Accenture、3M 和 Yum! Brands 的财务数据查询。您可以尝试提供其他公司名称或相关年份的信息。如果您对特斯拉的数据感兴趣，我可能需要使用不同的数据源来查找相关信息。是否可以帮忙确认一下？
"""


"""
应用场景
这段代码主要应用于 拥有庞大知识库或复杂 API 集合的企业级 AI 助手 场景：

企业级内部知识助手：

场景：一家大型公司拥有数千个内部微服务 API、数据库表或文档模块。
应用：不可能将所有 API 定义都放入 Prompt。通过向量检索，当员工询问“如何查询上季度销售数据”时，系统自动检索出相关的 sales_db_query 或 finance_api 工具，仅将这些工具暴露给 LLM。
动态插件系统 / App Store 模式：

场景：一个 AI 平台接入了第三方开发的成千上万个插件（如天气、股票、新闻、翻译等）。
应用：根据用户意图动态加载插件。例如，用户问“今天北京天气如何”，系统只加载“天气插件”；用户问“苹果股价多少”，系统只加载“金融插件”。这提高了响应速度并减少了误调用。
降低 LLM 成本与延迟：

场景：高并发、对延迟敏感的生产环境。
应用：减少每次请求发送给 LLM 的 Token 数量（因为只发送少量工具的 Schema），显著降低 API 调用成本并提高推理速度。
提高工具调用的准确性：

场景：工具名称或功能相似，容易混淆。
应用：通过语义检索筛选出最相关的子集，减少了 LLM 在大量无关工具中“迷路”或产生幻觉的概率，提高了工具选择的精准度。
"""

