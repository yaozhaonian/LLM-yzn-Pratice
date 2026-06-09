"""
工具节点化
ReAct (Reasoning + Acting) 模式的简化实现：
思考/行动: 模型决定是否需要调用工具。
执行: LangGraph 的 ToolNode 执行工具。
观察: 模型根据工具执行结果生成最终回复。
这种结构使得 Agent 能够处理超出纯文本生成能力的任务（如精确数学计算、数据库查询等），并通过循环机制确保任务的完整闭环。

"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from operator import add
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
"""
一个名为MessagesState的预构建状态，可轻松使用消息功能。
MessagesState仅通过一个messages键定义，该键是AnyMessage对象的列表，并使用add_messages归约器。
"""

# 初始化 LLM
llm = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

@tool
def add(a: int, b: int) -> int:
    """计算两数相加"""
    return a + b

@tool(description="计算a的b次方")
def pow(a: int, b: int) -> int:
    """计算a的b次方"""
    return a ** b

# 工具绑定
# model_tool = llm.bind_tools([add, pow])
# response = model_tool.invoke("3 的 5 次方是多少？")
# print("大模型回复:\n",response)
# tool_call = response.tool_calls[0]
# print("工具执行：",pow.invoke(tool_call))

@tool
def multiply(a: int, b: int) -> int:
    """计算两数相乘."""
    return a * b

tool_node = ToolNode([multiply, add, pow])
model_tn_tools = llm.bind_tools([multiply, add, pow])

def to_end(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    print("last_message:",last_message)
    if last_message.tool_calls:
        return "tools"
    return END

def to_tool(state: MessagesState):
    messages = state["messages"]
    response = model_tn_tools.invoke(messages)
    print("="*30,"response","="*30)
    print("大模型回复:\n",response)
    return {"messages": [response]}

builder = StateGraph(MessagesState)

builder.add_node("to_tool", to_tool)
# builder.add_node("to_end", to_end)
builder.add_node("tools", tool_node)

builder.add_edge(START, "to_tool")
builder.add_conditional_edges("to_tool", to_end, ["tools", END])    # 命中工具函数的话，总共会执行两次to_tool
builder.add_edge("tools", "to_tool")

graph = builder.compile()

# try:
#     graph.get_graph().draw_png(output_file_path='./14-ToolNode.png')
# except Exception as e:
#     print(f"错误: {e}")
#     print("请检查是否安装 graphviz 库")

print("="*30,"计算相加","="*30)
print(graph.invoke({"messages": [{"role": "user", "content": "1000+234=?"}]}))
print("="*30,"计算相乘","="*30)
print(graph.invoke({"messages": [{"role": "user", "content": "875*234=?"}]}))
print("="*30,"计算幂次方","="*30)
print(graph.invoke({"messages": [{"role": "user", "content": "8**3=?"}]}))
