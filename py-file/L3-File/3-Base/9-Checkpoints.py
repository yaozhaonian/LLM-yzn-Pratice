"""
LangGraph 中 检查点（Checkpoints） 和 状态持久化 。
其核心目的是让图的状态能够被保存、查询历史版本，甚至支持“时间旅行”（即恢复到之前的某个状态）。

"""
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from operator import add
from langgraph.checkpoint.memory import MemorySaver
"""
MemorySaver: 这是一个内存中的检查点存储实现。它在程序运行期间将图的状态快照保存在内存中。
"""

class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}

workflow = StateGraph(State)

workflow.add_node(node_a)
workflow.add_node(node_b)

workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)


checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)


config = {"configurable": {"thread_id": "1"}}
print(graph.invoke({"foo": ""}, config))

print("="*50, "获取最新的状态快照")
# 获取最新的状态快照
print(graph.get_state(config))

print("="*50, "获取状态快照的历史")
history = list(graph.get_state_history(config))
# 获取状态快照的历史
print("历史数量:", len(history))
for i, item in enumerate(history):
    print(f"历史快照{i+1}:\n类型:{type(item)}\n内容:{item}")

print("="*50, "获取特定检查点id的状态快照")
# 获取特定检查点id的状态快照
checkpoint_id = graph.get_state(config)[2].get("configurable", {}).get("checkpoint_id", "user_1")
print(f"检查点id: {checkpoint_id},\n类型: {type(checkpoint_id)}")
config2 = {"configurable": {"thread_id": "1", 'checkpoint_id': checkpoint_id}}
print("当前的\n",graph.get_state(config2))


"""
通过在 config 中指定具体的 checkpoint_id，可以“穿越”回那个特定的时间点。
graph.get_state 会返回在该检查点时刻的状态。
应用场景:
调试: 查看中间步骤的状态以排查错误。
人机协作: 如果用户在某一步拒绝了 AI 的操作，可以回滚到上一步的状态重新生成。
分支探索: 从某个历史状态分叉出新的执行路径。
"""

# print("类型:", type(history[1]))
checkpoint_id_2 = history[1].config.get("configurable", {}).get("checkpoint_id", "user_1")
print(f"检查点id(2): {checkpoint_id_2},\n类型: {type(checkpoint_id_2)}")
config3 = {"configurable": {"thread_id": "1", 'checkpoint_id': checkpoint_id_2}}
print("="*50, "获取历史中特定检查点id的状态快照")
print(graph.get_state(config3))