
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    input: str
    user_feedback: str
    step: str

def step_1(state: State):
    print("------step_1------")
    return {"step": "完成step_1"}

def human_feedback(state: State):
    print("------人类反馈------")
    feedback = interrupt("请提供反馈: ")
    print(f"反馈: {feedback}")
    return {"user_feedback": feedback}

def step_3(state: State):
    print("------step_3------")
    return {"step": "完成step_3"}

def human_feedback_2(state: State):
    print("------人类反馈2------")
    feedback = interrupt("请提供反馈2: ")
    print(f"反馈2: {feedback}")
    return {"user_feedback": feedback}

def step_5(state: State):
    print("------step_5------")
    return {"step": "完成step_5"}

builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)
builder.add_node("step_5", step_5)
builder.add_node("human_feedback_2", human_feedback_2)

builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", "human_feedback_2")
builder.add_edge("human_feedback_2", "step_5")
builder.add_edge("step_5", END)

memory = MemorySaver()

graph = builder.compile(checkpointer=memory)
graph.get_graph().draw_png(output_file_path='./1-Base.png')

initial_input = {"input": "你好，今天天气真好"}

thread = {"configurable": {"thread_id": "1"}}

for event in graph.stream(initial_input, thread, stream_mode="updates"):
    print("-----事件-----")
    print(event)
    print("="*50,"事件","="*50)

# 程序执行将变为“两阶段”阻塞式交互
for event in graph.stream(Command(resume="go to step3!"), thread, stream_mode="commands"):  # (注意：这里覆盖了之前的 step)
    print("-----命令-----")
    print(event)
    print("="*50,"命令","="*50) #这3行代码会直接被覆盖了，没有输出

for event in graph.stream(Command(resume="go to step5!"), thread, stream_mode="updates"):  # 推荐用stream_mode="updates"
    print("-----命令2-----")
    print(event)
    print("="*50,"命令2","="*50)

initial_input_2={"input": "你好，明天"}
for event in graph.stream(initial_input_2, thread, stream_mode="updates"):
    print("-----事件2-----")
    print(event)
    print("="*50,"事件2","="*50)

print("-----结束-----")
print(graph.get_state(thread).values)

"""
interrupt 的工作原理是：
节点函数执行到 interrupt()。
LangGraph 捕获异常/信号，保存状态，抛出 __interrupt__ 事件。
节点函数实际上并没有执行完，它在 interrupt 处挂起。
当你调用 Command(resume=...) 时，LangGraph 重新加载状态并重新执行该节点函数。
因此，print("------人类反馈------") 会被执行两次：
第一次：遇到 interrupt 前打印。
第二次：Resume 后，从头重新执行该节点函数时再次打印。
建议：不要在包含 interrupt 的节点中放置副作用代码（如 print、数据库写入），或者将打印放在 interrupt 之后。

 LangGraph 中最常用的几种 stream_mode 及其作用和使用场景：

1. stream_mode="updates" (最常用/推荐)
作用：只输出当前这一步执行完成的节点所返回的增量更新（Delta）。 格式：{ "节点名称": { "key": "value" } }
特点：
你可以清楚地看到哪个节点刚刚执行完毕。
你只看到该节点修改了哪些字段，而不是整个状态。
非常适合调试和观察工作流的执行步骤。

2. stream_mode="values"
作用：输出执行完当前节点后的完整状态快照（Full State Snapshot）。 格式：{ "key1": "value1", "key2": "value2", ... } (即 TypedDict 的结构)
特点：
每次迭代都包含所有状态字段。
如果你只关心“当前最新的全局状态是什么”，用这个。
缺点：如果状态很大（比如包含长文本或图片），每次打印都会非常冗长，且难以区分是哪一步引起的变化。

3. stream_mode="messages" (针对 LangChain Agent/ChatModel)
作用：专门用于流式输出 LLM 生成的 Token 或 消息片段。 格式：(AIMessageChunk, metadata) 或类似的流式块。
特点：
这是实现“打字机效果”的关键。
只有当你的节点调用了支持流式的 LLM（如 model.stream(...)）并正确 yield 时才有用。
在你的简单示例中（没有 LLM 流式调用），这个模式可能看不到明显效果或报错，除非节点内部做了特殊处理。

4. stream_mode="debug"
作用：输出详细的调试事件，包括任务开始、任务结束、检查点保存等底层信息。 格式：复杂的字典，包含 type, timestamp, payload 等。
特点：
用于深入排查 LangGraph 内部调度问题。
普通开发很少用到，除非你遇到了奇怪的并发或状态同步 Bug。

5. stream_mode="commands"
作用：主要用于恢复中断或发送外部命令时的流式反馈。 
特点：
在你之前的代码中，Command(resume=...) 本身是一个指令。
使用 stream_mode="commands" 通常是为了捕获命令执行过程中的特殊事件，但在大多数常规 Resume 场景下，它往往没有太多可见的输出，或者输出的是内部命令对象，而不是业务数据。
建议：在 Resume 时，通常也建议使用 "updates" 或 "values"，这样你能看到 Resume 后执行的节点（如 step_3）产生了什么数据变化。

"""

