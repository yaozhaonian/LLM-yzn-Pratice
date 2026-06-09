"""
在 LangGraph 中，通常有一个全局的 State（这里定义为 OverallState）。
但有时，某些数据只需要在两个相邻节点之间传递，不需要暴露给其他节点或保留在全局状态中。
"""
# 图中一些流程流转时的数据做私有化  --"隐身"
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


    
class OverallState(TypedDict):
    oas : str

class Node_a_Output(TypedDict):
    private_data: str

class Node_b_Input(TypedDict):
    private_data: str


"""
数据隔离: 演示了如何创建“临时通道”，让数据只在需要的节点间流动（node_1 -> node_2），而不会污染全局状态或暴露给无关节点
"""
# 私有数据在node_1和node_2之间共享
# 全局数据的输入到私有数据的输出
def node_1(state: OverallState) -> Node_a_Output:
    output = {"private_data": state['oas'] + ",来自节点1"}
    print("="*50)
    print(f"进入节点1:\n\t输入: {state}.\n\t输出: {output}")
    return output

# 私有数据的输入到全局数据的输出(其实中间过程可以有好几层)
def node_2(state: Node_b_Input) -> OverallState:
    output = {"private_data": "这个是没办法看到的哟,也不会影响私有数据", "oas": "来自节点2"}
    print("="*50)
    print(f"进入节点2:\n\t输入: {state}.\n\t输出: {output}")
    return output

# node3看不到在node_1和node_2之间流转的的私有数据）
def node_3(state: OverallState) -> OverallState:
    output = {"oas": "来自节点3"}
    print("="*50)
    print(f"进入节点3:\n\t输入: {state}.\n\t输出: {output}")
    return output



builder = StateGraph(OverallState)

builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# 私有数据不合并到全局状态中，它只是“路过”
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()

# 调用具有初始状态的图形
response = graph.invoke({"oas": "初始状态"})
print()
print("输出:\n", response)

"""
应用场景:
中间计算结果: 比如 node_1 调用 API 获取原始 JSON，node_2 解析该 JSON 并提取关键字段存入全局状态。原始 JSON 不需要保留在全局状态中。
敏感数据: 某些临时令牌或密码只在两个安全节点间传递，不应记录在全局状态日志中。
复杂工作流: 当不同子流程需要不同的上下文时，使用局部 Schema 可以避免全局状态变得过于庞大和复杂。
"""




