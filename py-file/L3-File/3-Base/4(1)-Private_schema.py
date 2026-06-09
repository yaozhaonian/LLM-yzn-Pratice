
# 定义清晰的入口和出口，隐藏内部复杂的状态流转逻辑  --"封装"
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


    
class OverallState(TypedDict):
    oas : str
    user_input: str
    graph_output: str

class OutputState(TypedDict):
    graph_output: str

class InputState(TypedDict):
    user_input: str

class PrivateState(TypedDict):
    private_data: str


# 输入-全局-私有-输出
def node_a(state: InputState) -> OverallState:
    output = {"oas": state["user_input"] + ",于节点a开始"}
    print("="*50)
    print(f"进入节点a:\n\t输入: {state}.\n\t输出: {output}")
    return output

def node_b(state: OverallState) -> PrivateState:
    output = {"private_data": state["oas"] + ",流转到节点b"}
    print("="*50)
    print(f"进入节点b:\n\t输入: {state}.\n\t输出: {output}")
    return output

def node_c(state: PrivateState) -> OverallState:
    output = {"oas": state["private_data"] + ",节点c结束"}
    print("="*50)
    print(f"进入节点c:\n\t输入: {state}.\n\t输出: {output}")
    return output

builder = StateGraph(OverallState)

builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_node("node_c", node_c)

builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", "node_c")
builder.add_edge("node_c", END)

graph = builder.compile()

response = graph.invoke({"user_input": "你好"})
print("输出:\n", response)
"""
应用场景:
1.构建可复用的子图或工具：
当你希望将一个复杂的 Graph 封装成一个黑盒，只暴露简单的输入（InputState）和输出（OutputState）时。内部可以使用复杂的 OverallState 进行状态管理。
2.API 接口标准化：
在微服务或 Agent 架构中，确保不同模块之间的交互遵循严格的输入/输出协议，隐藏内部实现细节。
3.分阶段数据处理：
当数据需要经过多个转换阶段，每个阶段产生不同的中间形态（如 PrivateState），且这些中间形态不应该直接混入最终的全局状态直到最后一步时。
"""


