"""
转换子图的输入和输出
处理多层嵌套子图（Nested Subgraphs），特别是当父图、子图和孙子图的状态结构（State Schema）完全不同时，
通过**手动转换（Manual Transformation）**来传递数据。

“黑盒调用”或“函数式封装”，即把编译好的子图当作一个普通的 Python 函数来调用，在调用前后进行数据的映射和转换。
"""
from langgraph.graph import StateGraph, END, START
from typing import TypedDict

# 孙子图
class GrandChildState(TypedDict):
    my_grandchild_key: str

def grandchild_1(state: GrandChildState) -> GrandChildState:
    # 注意：此处无法访问子密钥或父密钥
    return {"my_grandchild_key": state["my_grandchild_key"] + ", 我是孙节点"}

grandchild = StateGraph(GrandChildState)
grandchild.add_node("grandchild_1", grandchild_1)

grandchild.add_edge(START, "grandchild_1")
grandchild.add_edge("grandchild_1", END)

grandchild_graph = grandchild.compile()
# grandchild_graph.get_graph(xray=1).draw_mermaid_png(output_file_path="./3-transform_IO_grandchild.png")
print(grandchild_graph.invoke({"my_grandchild_key": "你好，大姚"}))

# 子图
class ChildState(TypedDict):
    my_child_key: str

def call_grandchild_graph(state: ChildState) -> ChildState:
    # 注意：这里无法访问父密钥或孙密钥——将状态从子状态通道（`my_child_key`）转换为子状态通道
    grandchild_graph_input = {"my_grandchild_key": state["my_child_key"]}
    # 将状态从孙状态通道（`my_grandchild_key`）转换回子状态通道（` my_child_key `）
    grandchild_graph_output = grandchild_graph.invoke(grandchild_graph_input)
    return {"my_child_key": grandchild_graph_output["my_grandchild_key"] + "我是子节点"}

child = StateGraph(ChildState)
# 注意：我们在这里传递的是一个函数，而不仅仅是编译后的图（`child_graph`）
child.add_node("child_1", call_grandchild_graph)
child.add_edge(START, "child_1")
child.add_edge("child_1", END)
child_graph = child.compile()
# child_graph.get_graph(xray=1).draw_mermaid_png(output_file_path="./3-transform_IO_child.png")
print(child_graph.invoke({"my_child_key": "你好，小德！"}))

# 父图
class ParentState(TypedDict):
    my_key: str

def parent_1(state: ParentState) -> ParentState:
    # 注意：此处无法访问子密钥或孙密钥
    return {"my_key": "你好，父1" + state["my_key"]}

def parent_2(state: ParentState) -> ParentState:
    return {"my_key": state["my_key"] + "父2"}

def call_child_graph(state: ParentState) -> ParentState:
    # 将状态从父状态通道（my_key）转换为子状态通道（my_child_key）
    child_graph_input = {"my_child_key": state["my_key"]}
    # 将状态从子状态通道（`my_child_key`）转换回父状态通道（'my_key'）
    child_graph_output = child_graph.invoke(child_graph_input)
    return {"my_key": child_graph_output["my_child_key"] + "我是父节点"}

parent = StateGraph(ParentState)
parent.add_node("parent_1", parent_1)
# 注意：我们在这里传递的是一个函数，而不仅仅是一个编译后的图（`<code>child_graph</code>`）
parent.add_node("child", call_child_graph)
parent.add_node("parent_2", parent_2)

parent.add_edge(START, "parent_1")
parent.add_edge("parent_1", "child")
parent.add_edge("child", "parent_2")
parent.add_edge("parent_2", END)

parent_graph = parent.compile()
parent_graph.get_graph(xray=2).draw_mermaid_png(output_file_path="./3-transform_IO_2.png")
print(parent_graph.invoke({"my_key": "你好，英俊姐"}))

"""
应用场景
这种手动转换输入输出的模式适用于以下场景：
1.遗留系统或第三方库集成：
当你需要调用一个已经编译好的、状态结构固定的子图（可能来自第三方库或旧代码），而你的主图状态结构与它不兼容时。你无法修改子图的内部状态定义，因此必须在外部通过包装函数进行适配。

2.高度解耦的模块化设计：
如果希望子图完全独立，不依赖任何父图的上下文（即不共享任何状态键），可以使用这种方式。子图就像一个纯函数 Input -> Output，父图负责准备输入和处理输出。这提高了模块的可复用性和测试性。

3.复杂的数据预处理/后处理：
在调用子图之前，可能需要对数据进行复杂的清洗、格式化或增强（例如，从父图的复杂对象中提取子图所需的简单字符串）。在子图返回后，可能需要将结果整合回父图的复杂结构中。这些逻辑可以清晰地写在包装函数中。

4.跨语言或跨服务调用模拟：
如果子图实际上是一个远程 API 调用（虽然这里是用 invoke 模拟本地调用），这种模式非常自然。你序列化输入，发送请求，接收响应，反序列化输出。这种思维模型有助于将来将本地子图迁移为远程微服务。
"""
