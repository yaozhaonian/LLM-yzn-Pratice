
from typing import List
from typing import TypedDict  # 用于创建类型化的字典


from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END  # LangGraph的状态图和起始/结束节点


# 定义State
class State(TypedDict):
    # 默认行为是：覆盖（Overwrite）
    messages: List[AnyMessage]
    extra_field: int

graph_builder = StateGraph(State)

# 定义Node
def process_input_node(state: State, config: RunnableConfig):
    print("process_input_node(过程输入节点):\n",state)
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    print(f"Node 'process_input_node' processing for user(节点(过程输入)用户处理): {user_id}")
    return {"process_input": state["extra_field"] + 1}

# 定义无config的Node
def no_config_node(state: State):
    print("no_config_node(无config节点):\n",state)
    state["extra_field"] = 5
    return {"some_other_data": "done"}

def node_a(state: State):
    print("node_a(节点A):\n", state)
    return {"extra_field": state["extra_field"] + 2}

def node_b(state: State):
    print("node_b(节点B):\n", state)
    return {"extra_field": state["extra_field"] + 3}

# 路由选择
def route_tools(state: State):
    print("route_tools(路由选择):\n", state["extra_field"])
    # 判断是否为偶数（最后一位二进制位为0表示偶数）
    if state["extra_field"] & 1 == 0:
        return "even"
    else:
        return "odd"

graph_builder.add_node("processor", process_input_node)
graph_builder.add_node("finalizer", no_config_node)
graph_builder.add_node("node_a", node_a)
graph_builder.add_node("node_b", node_b) 


# 添加图的边（连接关系）：
# 从开始节点连接到业务节点
graph_builder.add_edge(START, "processor")

#Conditional Edges「条件边」:
graph_builder.add_conditional_edges("processor", route_tools, {"even": "node_b", "odd": "node_a"})

# Conditional Entry Point「条件入口点」
graph_builder.add_edge("node_a", END)
graph_builder.add_edge("node_b", "finalizer")
# 从业务节点连接到结束节点
graph_builder.add_edge("finalizer", END)

# 编译图
graph = graph_builder.compile()


print("第一种情况:\n", graph.invoke({"messages": ["My", "hobby"], "extra_field": 2}))
print("="*50)
print("第二种情况:\n", graph.invoke({"messages": ["She", "is"], "extra_field": 3}, config={"configurable": {"user_id": "user_3"}}))



