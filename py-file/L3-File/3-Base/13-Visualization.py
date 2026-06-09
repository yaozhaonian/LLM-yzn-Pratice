# 可视化
import random
from typing import Annotated, Literal, TypedDict

from langgraph.graph import StateGraph, START, END, add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]


class MyNode:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, state: State) -> State:
        return {
            "messages": [
                ("assistant", f"返回的节点名字: {self.name}")
            ]
        }



def route(state: State) -> Literal["entry_node", "__end__"]:
    if len(state["messages"]) > 10:
        return "__end__"
    else:
        return "entry_node"

def add_fractal_nodes(builder, current_node, level, max_level):
    if level > max_level: return
    ri = 1
    
    num_nodes = random.randint(1, 3)
    print(f"第{ri}次递归添加节点num_nodes:{num_nodes}")
    for i in range(num_nodes):
        nm = ["A", "B", "C"][i]
        node_name = f"{current_node}_{nm}"
        builder.add_node(node_name, MyNode(node_name))
        builder.add_edge(current_node, node_name)

    # 递归添加节点
    r = random.random()
    print(f"第{ri}次递归添加节点:{r}")
    ri += 1
    
    if r > 0.2 and level + 1 < max_level:
        add_fractal_nodes(builder, node_name, level + 1, max_level)
    elif r > 0.05:
        builder.add_conditional_edges(node_name, route, node_name)
    else:
        builder.add_edge(node_name, "__end__")


def build_fractal_graph(max_level: int):
    builder = StateGraph(State)
    begin_point = "entry_node"
    builder.add_node(begin_point, MyNode(begin_point))
    builder.add_edge(START, begin_point)
    
    add_fractal_nodes(builder, begin_point, 1, max_level)
    
    builder.add_edge(begin_point, END)
    
    return builder.compile()

app = build_fractal_graph(3)

try:
    app.get_graph().draw_png(output_file_path='./13-visualization.png')
except Exception as e:
    print(f"错误: {e}")
    print("请检查是否安装 graphviz 库")


print("图片已生成：13-visualization.png")



