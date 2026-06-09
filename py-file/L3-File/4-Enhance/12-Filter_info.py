"""
在 LangGraph 中自定义消息过滤逻辑，以控制发送给大语言模型（LLM）的上下文长度或内容。
实现“有状态存储但无状态推理”或“有限上下文窗口”的效果。
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

memory = MemorySaver()

@tool
def web_search(query: str):
    """网络搜索(以实际情况为准)"""
    if query == "上海":
        return "上海天气晴朗，温度28度，湿度65%"
    return "旅游搜索引擎推荐:https://www.dili360.com/"

tools =[web_search]

tool_node = ToolNode(tools=tools)

model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama" 
)
bound_model = model.bind_tools(tools)

def should_continue(state: MessagesState):
    """判断是否继续执行"""
    last_message = state["messages"][-1]
    print("是否需要调用工具:\n", last_message)
    if not last_message.tool_calls:
        return END
    return "action"

def filter_messages(messages: list):
    print("完整信息\n", messages)
    # print("过滤信息\n", messages[-1:])
    return messages[-1:]

def call_model(state: MessagesState):
    """调用模型"""
    print("调用模型")
    messages = filter_messages(state["messages"])
    print("过滤后的信息(最新的AI回复)\n", messages)
    response = bound_model.invoke(messages)
    print("模型返回:\n", response)
    return {"messages": response}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_edge(START, "agent")

workflow.add_conditional_edges("agent", should_continue, ["action", END])

workflow.add_edge("action", "agent")
app = workflow.compile(checkpointer=memory)

# app.get_graph().draw_png(output_file_path='./12-filter_info.png')

config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("请输入您的问题：")
    if user_input == "exit":
        break
    input_message = HumanMessage(content=user_input)
    for event in app.stream({"messages": [input_message]}, config=config, stream_mode="values"):
        event["messages"][-1].pretty_print()


"""
应用场景
这种“过滤消息”的模式适用于以下场景：

单轮独立任务处理:

每个用户请求都是独立的，不需要依赖前文。例如：翻译工具、单次代码生成、独立的问题解答。
虽然不需要记忆，但仍希望使用 LangGraph 的工作流编排能力（如工具调用、错误重试）。
基于检索增强生成（RAG）的系统:

在更复杂的场景中，filter_messages 不会被简单替换为取最后一条，而是会被替换为**“检索相关历史”**。
例如：从完整的 state["messages"] 中向量检索出与当前问题最相关的 3 条历史消息，连同当前消息一起发给 LLM。这比发送全部历史更高效且精准。
隐私保护或数据脱敏:

在发送给 LLM 之前，过滤掉包含敏感信息（如密码、身份证号）的历史消息，仅保留必要的非敏感上下文。
节省 Token 成本:

对于长对话应用，如果前文对当前问题没有参考价值，主动丢弃旧消息可以显著降低 API 调用成本。
"""


