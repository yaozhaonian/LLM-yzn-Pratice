# 设置递归限制(同样可以运用在分支上)

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from operator import add
from langgraph.errors import GraphRecursionError


class State(TypedDict):
    sum: Annotated[list, add]

def node_a(state: State) -> State:
    print(f'把"节点A"加进{state["sum"]}')
    return {"sum": ["节点A"]}

def node_b(state: State) -> State:
    print(f'把"节点B"加进{state["sum"]}')
    return {"sum": ["节点B"]}

def node_c(state: State) -> State:
    print(f'把"节点C"加进{state["sum"]}')
    return {"sum": ["节点C"]}

def node_d(state: State) -> State:
    print(f'把"节点D"加进{state["sum"]}')
    return {"sum": ["节点D"]}

builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_node("c", node_c)
builder.add_node("d", node_d)

def route(state: State) -> Literal["b", END]:
    if len(state["sum"]) >= 6:  # 6、7、8结果一样，没办法强制到END,也就是说它至少需要走完当前流程
        return END
    return "b"

builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
# builder.add_edge("b", "a")
builder.add_edge("b", "c")
builder.add_edge("b", "d")
builder.add_edge(["c", "d"], "a")
graph = builder.compile()

# graph.get_graph().draw_mermaid_png(output_file_path="./4-Recursion_limit_2.png")

try:
    result = graph.invoke({"sum": []})
    # result = graph.invoke({"aggregate": []}, {"recursion_limit": 4})
    print(result)
except GraphRecursionError:
    print("Recursion Error")


"""
应用场景
这种“带递归限制的循环图”模式在 AI Agent 开发中非常常见，主要应用于以下场景：

(1) ReAct Agent 的推理-行动循环
场景描述: Agent 需要反复进行“思考（Thought）-> 行动（Action）-> 观察（Observation）”的循环，直到找到最终答案。
对应关系:
节点 a 类似于 “LLM 思考/生成行动”。
节点 b 类似于 “执行工具/获取观察结果”。
route 类似于判断“是否已找到答案”或“是否达到最大步数”。
为什么需要递归限制: 防止 Agent 陷入死循环（例如，不断重复相同的无效操作），确保系统在有限步骤内停止并返回超时或部分结果。
(2) 自我修正代码生成 (Self-Correction)
场景描述: LLM 生成代码 -> 运行测试 -> 如果失败，LLM 根据错误信息修改代码 -> 再次运行测试。
对应关系:
a: 生成/修改代码。
b: 执行单元测试并收集错误日志。
route: 检查测试是否通过或重试次数是否过多。
价值: 避免无限次的代码修正尝试，控制 Token 消耗和执行时间。
(3) 多轮对话状态管理
场景描述: 在复杂的任务型对话中，系统可能需要多次向用户澄清意图或收集信息。
对应关系:
a: LLM 生成回复或追问。
b: 处理用户输入并更新槽位（Slot Filling）。
route: 检查所有必要信息是否已收集齐全。
价值: 防止因用户无法提供清晰信息而导致对话无限循环。
(4) 数据聚合与迭代处理
场景描述: 对一批数据进行分块处理，并将结果逐步聚合到一个列表中，直到所有数据处理完毕或达到某种阈值。
对应关系:
利用 operator.add 的累加特性，逐步构建最终结果集。
递归限制作为安全网，防止因数据量过大或逻辑错误导致的资源耗尽。
"""

"""
状态累积: 使用 operator.add 实现列表的自动合并。
循环控制: 通过条件边实现 a <-> b 的循环。
安全性: 理解并处理 GraphRecursionError，这是构建健壮 LangGraph 应用的重要一环，确保 Agent 不会无限运行下去。
"""


