from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

@tool
def get_weather(location: str):
    """获取当前天气."""
    if location.lower() in ["SH", "上海"]:
        return "气温23度，有雾."
    else:
        return "气温30度，阳光明媚."

@tool
def get_coolest_cities():
    """获得最凉快城市列表"""
    return "青岛, 上海"

@tool
def get_basketball_roster(team: str):
    """获得湖人的阵容."""
    if team == "湖人":
        return "詹姆斯、东契奇、里夫斯、八村塁、艾顿"
    else:
        raise ValueError("非湖人队.")



tools = [get_weather, get_coolest_cities, get_basketball_roster]
tool_node = ToolNode(tools)

# 模型调用工具
model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    temperature=0
).bind_tools(tools)

print(model.invoke("上海的天气怎么样?").tool_calls)
print("######")

def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)


workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
# 条件选择
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")
app = workflow.compile()

# app.get_graph().draw_png(output_file_path='./1-use_model.png')

# for chunk in app.stream(
#     {"messages": [("human", "上海的天气怎么样？")]}, stream_mode="values"
# ):
#     chunk["messages"][-1].pretty_print()

print('######################')

# for chunk in app.stream(
#     {"messages": [("human", "最凉快城市有哪些?天气如何？")]},
#     stream_mode="values",
# ):
#     chunk["messages"][-1].pretty_print()

print('######################')

for chunk in app.stream(
    {"messages": [("human", "湖人的轮换阵容有谁？")]},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()

print('######################')

for chunk in app.stream(
    {"messages": [("human", "马刺的轮换阵容有谁？")]},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()

# response = app.invoke({"messages": [("human", "马刺的轮换阵容有谁？")]})

# for message in response["messages"]:
#     string_representation = f"{message.type.upper()}: {message.content}\n"
#     print('string_representation:\n',string_representation)
