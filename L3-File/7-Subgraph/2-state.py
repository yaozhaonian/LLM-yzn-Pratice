# 查看和更新子图的状态
"""
结合 断点（Interrupts）、检查点（Checkpoints） 和 状态检查（State Inspection） 来实现复杂的控制流。

核心场景：
父图负责路由（判断用户意图），如果意图是“查询天气”，则进入子图处理。
子图在执行关键操作（调用天气 API）前会暂停，允许外部系统查看或干预状态，然后再恢复执行。
"""
from langgraph.graph import StateGraph, END, START, MessagesState
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal
from langgraph.checkpoint.memory import MemorySaver

model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    temperature=0
)
memory = MemorySaver()
# 子图部分 
# 天气类
class WeatherRequest(BaseModel):
    city: str = Field(description="这座城市的天气")

# 添加天气工具
@tool
def get_weather(city: str) -> str:
    """获取特定城市的天气"""
    return f"当前{city}的天气是多云!"

# 子图类(存储对话历史和提取出的城市名称)
class SubGraphState(MessagesState):
    city: str = Field(description="城市名称")

"""
功能：LLM 节点，用于从用户消息中提取城市信息。
逻辑：调用绑定了 get_weather 工具的 LLM。如果 LLM 产生了工具调用，提取其中的 city 参数并更新到状态中；否则使用默认城市 "广州"。
"""
def model_node(state: SubGraphState) -> dict:
    model_with_tools = model.bind_tools([get_weather])
    messages = [
        {"role": "system", "content": "你是一个智能天气助手。当用户询问某个城市的天气时，你必须调用 'get_weather' 工具来获取准确信息，不要直接回答。"},
        *state["messages"]
    ]
    result = model_with_tools.invoke(messages)
    print(f"LLM输出：\n{result}")
    # 从工具调用中提取城市信息
    if result.tool_calls:
        tool_call = result.tool_calls[0]
        if tool_call["name"] == "get_weather":
            print("找到城市:", tool_call["args"]["city"])
            return {"city": tool_call["args"]["city"]}
    # 如果没有工具调用或提取失败，使用默认城市
    print("未检测到有效工具调用，使用默认城市: 广州")
    return {"city": "广州"}

"""
功能：工具执行节点。
逻辑：使用状态中的 city 调用 get_weather 工具，并将结果作为助手消息添加到 messages 中。
"""
def weather_node(state: SubGraphState):
    print("天气工具执行节点")
    result = get_weather.invoke({"city": state["city"]})
    return {"messages": [{"role":"assistant", "content": result}]}

"""
子图编译配置
subgraph.compile(interrupt_before=["weather_node"])：关键点。在执行 weather_node 之前强制暂停。
这意味着 model_node 执行完后，图会停止，等待外部指令才能继续。
"""
subbuilder = StateGraph(SubGraphState)
subbuilder.add_node("weather_node", weather_node)
subbuilder.add_node("model_node", model_node)
subbuilder.add_edge(START, "model_node").add_edge("model_node", "weather_node").add_edge("weather_node", END)
subgraph = subbuilder.compile(interrupt_before=["weather_node"])


# 父图部分
# 存储路由决策
class RouterState(MessagesState):
    route: Literal["weather", "other"]

class RouteClassify(BaseModel):
    route: Literal["weather", "other"] = Field(description="将查询分类为是否与天气有关")

"""
功能：意图分类节点。
逻辑：使用 LLM 判断用户问题是否与天气有关。
通过绑定两个虚拟工具（weather_route, other_route）强制 LLM 做出选择，并根据选择更新 route 状态。包含简单的关键词回退逻辑。
"""
def router_node(state: RouterState) -> RouteClassify:
    router_prompt = """将传入的查询分类为是否与天气有关。
    如果是关于天气的，请使用工具调用“weather_route”进行响应。
    如果是关于其他事情，请使用工具调用“other_route”进行响应。
        
    仅通过工具调用进行响应，不涉及其他任何内容。"""
    router_tools = [
        {
            "type": "function",
            "function": {
                "name": "weather_route",
                "description": "用于处理与天气相关的查询"
            }
        },
        {
            "type": "function",
            "function": {
                "name": "other_route",
                "description": "用于处理与天气无关的查询"
            }
        },
    ]
    
    router_model = model.bind_tools(router_tools)
    messages = [{"role": "system", "content": router_prompt}] + state["messages"]
    result = router_model.invoke(messages)
    
    # 检查工具调用结果
    if result.tool_calls:
        tool_name = result.tool_calls[0]["name"]
        if tool_name == "weather_route":
            return {"route": "weather"}
        elif tool_name == "other_route":
            return {"route": "other"}
    
    user_content = state["messages"][-1].content.lower()
    if "weather" in user_content or "气候" in user_content or "温度" in user_content:
        return {"route": "weather"}
    else:
        return {"route": "other"}

"""
功能：默认聊天节点。
逻辑：如果路由结果为 "other"，直接调用 LLM 生成普通回复。
"""
def normal_llm_node(state:RouterState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

"""
功能：条件边函数。
逻辑：根据 state["route"] 决定下一步是进入 "weather_graph" (子图) 还是 "normal_llm_node"。
"""
def route_after_prediction(
    state: RouterState
) -> Literal["weather_graph", "normal_llm_node"]:
    if state["route"] == "other":
        return "normal_llm_node"
    else:
        return "weather_graph"

"""
父图编译配置

graph.compile(checkpointer=memory)：启用内存检查点，保存整个图（包括子图）的执行状态，支持暂停和恢复。
"""
prabuilder = StateGraph(RouterState)
prabuilder.add_node(router_node)
prabuilder.add_node(normal_llm_node)
prabuilder.add_node("weather_graph", subgraph)
prabuilder.add_edge(START, "router_node")
prabuilder.add_conditional_edges("router_node", route_after_prediction)
prabuilder.add_edge("normal_llm_node", END)
prabuilder.add_edge("weather_graph", END)
pragraph = prabuilder.compile(checkpointer=memory)
pragraph.get_graph(xray=1).draw_mermaid_png(output_file_path="./2-state.png")

print("A"*50)
config = {"configurable": {"thread_id": "1"}}
inputs = {"messages": [{"role": "user", "content": "下午好！"}]}
for update in pragraph.invoke(inputs, config=config, stream_mode="updates"):
    print(update)

print("B"*50)

# 从断点处恢复
config = {"configurable": {"thread_id": "1"}}
inputs = {"messages": [{"role": "user", "content": "北京的天气怎么样"}]}
sum = 1
try: 
    for update in pragraph.stream(input=inputs, config=config, stream_mode="updates"):
        print(f"-----------第{sum}轮-----------")
        print(update)
        sum += 1
except Exception as e:
    print(f"恢复失败: {e}")
sum = 1
print("C"*50)
config = {"configurable": {"thread_id": "3"}}
inputs = {"messages": [{"role": "user", "content": "上海的天气怎么样"}]}
for update in pragraph.stream(inputs, config=config, stream_mode="values", subgraphs=True):
    print(f"-----------第{sum}轮-----------")
    print(update)
    sum += 1

print("D"*50)
# 检查状态的代码可能会在某些情况下引发异常，所以添加异常处理
try:
    state = pragraph.get_state(config=config)
    print("state.next:", state.next)
    print("state.tasks:", state.tasks)
except:
    print("在该节点无法获取状态")
print("E"*50)
try:
    state = pragraph.get_state(config=config, subgraphs=True)
    print("state.tasks[0]:", state.tasks[0])
except:
    print("在该节点无法获取子图状态")
print("F"*50)
# 恢复执行
try:
    parent_graph_state_before_subgraph = next(
        h for h in pragraph.get_state_history(config) if h.next == ("weather_graph",)
    )
    subgraph_state_before_model_node = next(
        h
        for h in pragraph.get_state_history(parent_graph_state_before_subgraph.tasks[0].state)
        if h.next == ("model_node",)
    )
    
    print("在模型节点执行前的子图状态:\n", subgraph_state_before_model_node.next)
    print("G"*50)
    sum = 1

    for value in pragraph.stream(
        None,
        config=subgraph_state_before_model_node.config,
        stream_mode="values",
        subgraphs=True
    ):
        print(f"-----------第{sum}轮-----------")
        print(value)
        sum += 1
except:
    print("无法从特定子图节点恢复")


"""
应用场景
适用于需要 人机协同（Human-in-the-loop） 或 复杂状态监控 的企业级 Agent 场景：

1.敏感操作确认：
场景：子图中的 weather_node 替换为“发送邮件”、“支付订单”或“删除数据”。
应用：在执行这些高风险操作前设置断点。父图或外部前端可以检查子图状态（例如：确认提取的城市是否正确，或确认收件人地址），用户点击“确认”后，再调用 stream(None, config) 恢复执行。

2.多步推理的中间修正：
场景：子图是一个复杂的数据分析流程。
应用：在 model_node 提取出关键参数（如城市、日期、指标）后暂停。如果提取错误（例如将 "NYC" 误识别为 "New York City" 的缩写但拼写错误），用户可以通过 API 更新子图状态中的 city 字段，然后恢复执行，确保后续工具调用的准确性。

3.调试与可观测性：
场景：开发复杂的嵌套 Agent 系统。
应用：通过 subgraphs=True 和 get_state(..., subgraphs=True)，开发者可以深入观察子图内部的每一步状态变化，而不仅仅看到父图的宏观流程。这对于排查子图内部的逻辑错误至关重要。

4.动态路由与模块化扩展：
场景：一个通用的客服机器人，需要根据意图跳转到不同的专业技能子图（天气、退款、技术支持等）。
应用：父图作为轻量级路由器，保持简洁。每个专业技能封装在独立的子图中，可以独立开发、测试和部署。断点机制确保了在进入专业流程前的控制权。
"""

