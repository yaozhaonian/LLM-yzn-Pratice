

import random
from typing import TypedDict, Annotated, Literal
# Literal 限制变量只能取几个固定的字面量值
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
# https://docs.langchain.com/oss/python/langgraph/graph-api#command
"""
Command用途:同时更新状态并路由到其他节点
Command 是一种用于控制图执行的通用原语，它接受四个参数：
update：应用状态更新（类似于从节点返回更新内容）。
goto：导航至特定节点（类似于条件边）。
graph：从子图导航时定位到父图。
resume：提供一个值，以便在中断后恢复执行。

Command有三种使用场景：
从节点返回：使用update、goto和graph将状态更新与控制流相结合。
调用或流式处理的输入：使用 resume 以在中断后继续执行。
从工具返回：与从节点返回类似，结合工具内部的状态更新和控制流。

"""

class State(TypedDict):
    foo: int
    extra_field: int

def node_1(state: State) -> Command[Literal["node_2", "node_3"]]:
    print("node_1(节点1):\n", state)
    value = random.choice([2, 3])
    if value == 2:
        return Command(update={"foo": value}, goto="node_2")
    elif value == 3:
        return Command(update={"foo": value}, goto="node_3")


def node_2(state: State):
    print("node_2(节点2):\n", state)
    return {"extra_field": state["foo"] + 2, "foo": state["foo"]}


def node_3(state: State):
    print("node_3(节点3):\n", state)
    return {"extra_field": state["foo"] + 3, "foo": state["foo"]}
    
builder = StateGraph(State)
builder.add_node(node_1)
builder.add_node(node_2)
builder.add_node(node_3)
# 节点1、2、3之间没有边

def to_end(state: State):
    print("to_end(结束节点):\n", state)
    return state

builder.add_node(to_end)
builder.add_edge(START, "node_1")

# 2. node_2 完成后连接到 to_end (使用字符串 "node_2" 和 "to_end")
builder.add_edge("node_2", "to_end")

# 3. node_3 完成后连接到 to_end (使用字符串 "node_3" 和 "to_end")
builder.add_edge("node_3", "to_end")

# 4. to_end 连接到 END
builder.add_edge("to_end", END)

graph = builder.compile()
# try:
#     graph.get_graph().draw_png(output_file_path='./8-Command.png')
# except Exception as e:
#     print(f"错误: {e}")
#     print("请检查是否安装 graphviz 库")



num = 0
while True:    
    if num > 6:
        break
    num += 1
    graph.invoke({"foo": 0})
    print("="*50)