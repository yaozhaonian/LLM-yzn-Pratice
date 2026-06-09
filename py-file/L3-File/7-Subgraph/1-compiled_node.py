"""
将一个编译好的图作为节点嵌入到另一个父图中。这种模块化设计允许开发者将复杂的逻辑封装成独立的单元，并在主流程中复用。
"""

# 添加编译后的子图节点
from langgraph.graph import START, StateGraph
from typing import TypedDict


# 定义子图
class SubgraphState(TypedDict):
    foo: str  # 请注意，此键与父图状态共享
    bar: str

def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}

def subgraph_node_2(state: SubgraphState):
    # 请注意，此节点使用的状态密钥（'bar'）仅在子图中可用，并且正在发送共享状态密钥（'foo'）的更新
    return {"foo": state["foo"] + state["bar"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

# 定义父图
class ParentState(TypedDict):
    foo: str

def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}

builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
# 将编译后的子图作为节点添加到父图中
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()
graph.get_graph().draw_png(output_file_path='./1-compiled_node.png')
for chunk in graph.stream({"foo": "foo"}):
    print(chunk)
print("A"*50)
for chunk in graph.stream({"foo": "foo"}, subgraphs=True):  # 使用 subgraphs=True参数可以清晰地看到子图内部的执行步骤。
    print(chunk)

"""
适用场景:
子图是主流程的一部分、状态结构有重叠、希望简化代码并利用自动状态同步。

应用场景
主要应用于需要 模块化、可复用性和复杂逻辑封装 的场景：

1.模块化架构设计：
场景：大型 AI 应用包含多个独立的功能模块，如“用户认证”、“数据检索”、“内容生成”、“安全过滤”等。
应用：可以将每个功能模块开发为一个独立的子图。主图（父图）负责编排这些模块的执行顺序。这样，每个子图可以独立开发、测试和维护，降低了系统的耦合度。

2.逻辑复用：
场景：多个不同的工作流都需要执行相同的复杂步骤，例如“标准化用户输入”或“调用特定的外部 API 集群”。
应用：将这个通用步骤封装成一个子图。然后在多个不同的父图中引用同一个编译好的子图对象，避免代码重复。

3.隐藏实现细节（抽象）：
场景：团队分工合作，或者向第三方提供 SDK。
应用：父图的使用者只需要知道子图节点的输入和输出（通过共享状态键 foo），而不需要了解子图内部是如何通过 subgraph_node_1 和 subgraph_node_2 协作完成任务的。子图内部的私有状态（如 bar）对外部完全透明。

4.分层调试与监控：
场景：生产环境中需要监控整体流程，但在开发或故障排查时需要深入查看某个模块的内部执行细节。
应用：默认情况下使用标准流式输出以获得简洁日志；在需要调试时，启用 subgraphs=True 以获取完整的执行轨迹，包括子图内部的每一步状态变更。
"""






