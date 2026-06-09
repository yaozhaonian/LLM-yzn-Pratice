# 并行运行图节点(带额外步骤)
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from operator import add

class State(TypedDict):
    sum: Annotated[list, add]

def node_a(state: State):
    print(f'把"节点A"加进{state["sum"]}')
    return {"sum": ["节点A"]}

def node_b(state: State):
    print(f'把"节点B"加进{state["sum"]}')
    return {"sum": ["节点B"]}

def node_c(state: State):
    print(f'把"节点C"加进{state["sum"]}')
    return {"sum": ["节点C"]}

def node_d(state: State):
    print(f'把"节点D"加进{state["sum"]}')
    return {"sum": ["节点D"]}

def node_b_2(state: State):
    print(f'把"节点b_2"加进{state["sum"]}')
    return {"sum": ["节点b_2"]}

builder = StateGraph(State)
builder.add_node(node_a)
builder.add_node(node_b)
builder.add_node("b_2", node_b_2)
builder.add_node(node_c)
builder.add_node(node_d)

builder.add_edge(START, "node_a")

builder.add_edge("node_a", "node_b")
builder.add_edge("node_a", "node_c")

builder.add_edge("node_b", "b_2")
# builder.add_edge("node_b", "node_d")
builder.add_edge(["b_2", "node_c"], "node_d") # 带额外步骤
builder.add_edge("node_d", END)

graph = builder.compile()
# try:
#     # graph.get_graph().draw_png(output_file_path='./1-Parallel_exec_gra_nodes.png')
#     graph.get_graph().draw_png(output_file_path='./1-Parallel_2.png')
# except Exception as e:
#     print(f"错误: {e}")
#     print("请检查是否安装 graphviz 库")
    
print("="*50)
print(graph.invoke({"sum": []}, {"configurable": {"thread_id": "12321"}}))