"""
Reducers 归约器
reducer 函数对于告诉图如何使用每个状态更新（例如，当节点发送更新时）更新状态中的 Message 对象列表至关重要。
如果不指定reducer，则每个状态更新都将用最近提供的值覆盖消息列表。
如果只想将消息追加到现有列表中，则可以使用operator.add 作为 reducer。
"""

from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END


class State_no_Add(TypedDict):
    foo: int
    message: list[str]

class State_Add(TypedDict):
    foo: int
    # 使用Annotated标注合并策略为add（列表拼接）
    message: Annotated[list[str], add]      # PS.在此画蛇添足,但是最好用这种,在Sequence就可以区分了

def SnA(state: State_no_Add) -> State_no_Add:
    print("节点SnA:\n", state)
    return {"foo": state["foo"] + 1, "message": ["有的,兄弟有的"]}

def SA(state: State_no_Add) -> State_Add:
    print("节点SA:\n", state)
    return {"foo": state["foo"] + 1, "message": ["这么强的AI还有10个"]}

builder = StateGraph(State_Add)
builder.add_node("SnA", SnA)
builder.add_node("SA", SA)
builder.add_edge(START, "SnA")
builder.add_edge("SnA", "SA")
builder.add_edge("SA", END)

graph = builder.compile()
response = graph.invoke({"foo": 0, "message": ["开始", "装逼吧"]})
print("输出:\n", response)


"""
应用场景
聊天机器人记忆: 使用 Annotated[list[Message], add] 来自动追加每一轮的用户输入和 AI 回复，而不需要在每个节点手动读取旧列表并拼接。
并行任务结果收集: 当多个并行节点同时写入同一个列表字段时，Reducer 确保所有结果都被保留，而不是最后一个节点的结果覆盖前面的。
日志聚合: 在长运行工作流中，各个节点产生的日志片段可以自动汇聚到一个全局日志列表中。
"""