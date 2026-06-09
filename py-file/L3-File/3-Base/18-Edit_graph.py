from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    input: str

def step_1(state: State) -> dict:
    print("------step_1------")
    return {"input": state["input"] + "1"}

def step_2(state: State) -> dict:
    print("------step_2------")
    return {"input": state["input"] + "2"}

def step_3(state: State) -> dict:
    print("------step_3------")
    return {"input": state["input"] + "3"}

def step_4(state: State) -> dict:
    print("------step_4------")
    return {"input": state["input"] + "4"}

def step_5(state: State) -> dict:
    print("------step_5------")
    return {"input": state["input"] + "5"}

builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_node("step_4", step_4)
builder.add_node("step_5", step_5)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", "step_4")
builder.add_edge("step_4", "step_5")
builder.add_edge("step_5", END)

memory = MemorySaver()

graph = builder.compile(checkpointer=memory, interrupt_before=["step_2"], interrupt_after=["step_4"])
# graph.get_graph().draw_png(output_file_path='./18-Edit_graph.png')

initial_input = {"input": "hello world"}

thread = {"configurable": {"thread_id": "1"}}

# 运行graph，直到第一次中断 (在 step_2 之前)
for event in graph.stream(initial_input, thread, stream_mode="values"):
    print(event)

print("目前state!")
print(graph.get_state(thread).values)

graph.update_state(thread, {"input": "你好宇宙!"})

print("---\n---\n更新state!")
print(graph.get_state(thread).values)

print("目前state2!")
graph.update_state(thread, {"input": "你好大爆炸!"})
print(graph.get_state(thread).values)

print("---\n---\n更新state2!")
graph.update_state(thread, {"input": "你好混沌!"})
print(graph.get_state(thread).values)

# 继续执行
print("---\n---\n继续执行!")
for event in graph.stream(None, thread, stream_mode="values"):
    print(event)

print("---\n---\nstep_5要执行需要继续执行!")
for event in graph.stream(None, thread, stream_mode="values"):
    print(event)

print("="*50)
thread_2 = {"configurable": {"thread_id": "2"}}
graph2 = builder.compile(checkpointer=memory)
for event in graph2.stream(initial_input, thread_2, stream_mode="values"):
    print(event)