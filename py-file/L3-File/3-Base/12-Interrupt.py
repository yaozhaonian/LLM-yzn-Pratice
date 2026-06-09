
from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

class FormState(TypedDict):
    age: int | None
    

# 标志图的开始与结束运行
def begin_node(state: FormState):
    print("开始运行:", state)
    print("="*50)

def end_node(state: FormState):
    print("结束运行:", state)
    print("="*50)

def get_age_node(state: FormState):
    prompt = "你的年龄是？"
    print(f"state:{state},获取年龄节点(get_age_node):{prompt}")
    inner_count = 0
    while True:
        inner_count += 1
        answer = interrupt(prompt)
        print(f"第{inner_count}次用户输入:{answer}")
        if isinstance(answer, int) and answer > 0:
            break
        prompt = f"输入的值{answer}不是正整数,请输入一个正整数:"
        print("-"*50)
    state = {"age": answer}
    print(f"年龄节点(get_age_node)第{inner_count}次循环后结束:{state}")
    return {"age": answer}
    
builder = StateGraph(FormState)
builder.add_node("begin", begin_node)
builder.add_node("get_age", get_age_node)
builder.add_node("end", end_node)

builder.add_edge(START, "begin")
builder.add_edge("begin", "get_age")
builder.add_edge("get_age", "end")
builder.add_edge("end", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "form-1"}}
first = graph.invoke({"age": None}, config=config)
print(f'第一次调用返回：{first["__interrupt__"]}')  # -> [Interrupt(value='What is your age?', ...)]
# print(f'Command模拟人类输入"thirty"')
retry = graph.invoke(Command(resume="thirty"), config=config)
print(f'第二次调用返回：{retry["__interrupt__"]}')  # -> [Interrupt(value="'thirty' is not a valid age...", ...)]
# print(f'Command第二次模拟人类输入"三十"')
third = graph.invoke(Command(resume="三十"), config=config)
print(f'第三次调用返回：{third["__interrupt__"]}')  # -> [Interrupt(value="'thirty' is not a valid age...", ...)]
# print(f'Command第三次模拟人类输入"30岁"')
fourth = graph.invoke(Command(resume="30岁"), config=config)
print(f'第四次调用返回：{fourth["__interrupt__"]}')  # -> [Interrupt(value="'thirty' is not a valid age...", ...)]
# print(f'Command第四次模拟人类输入"30"')
final = graph.invoke(Command(resume=30), config=config)
print(f'final结果：{final["age"]}')

"""
LangGraph 中 interrupt 与 while 循环结合的高级用法，实现了一个具有输入验证和重试机制的交互式表单收集功能。

其核心功能是：在一个节点内部，通过循环不断询问用户年龄，直到用户输入合法的整数为止。
每次非法输入都会导致图中断，等待新的输入，然后从断点处恢复并继续循环验证。

应用场景: 
这种模式非常适合构建复杂的聊天机器人表单、多步验证流程或需要人工反复修正数据的场景。
允许将复杂的交互逻辑封装在单个节点内，而不需要在图中创建大量的状态节点来处理每一步验证。
"""




"""
Command和interrupt()协同工作原理详解
整体工作机制
在LangGraph中，interrupt()和Command(resume=...)的协同工作遵循以下机制：
中断触发：当节点执行到interrupt()调用时，整个图的执行暂停，控制权返回给调用者
状态保存：LangGraph保存当前执行状态，包括未完成的节点信息
恢复执行：通过Command(resume=...)提供输入，图从保存的状态继续执行
重新执行：被中断的节点会从头开始重新执行，但interrupt()会直接返回resume的值而不是再次中断
LangGraph内部可能维护一个输入队列或栈，用于保存通过Command(resume=...)传递的值：
Input Queue: ["thirty"] -> ["thirty", "三十"] -> ["thirty", "三十", "30岁"] -> ["thirty", "三十", "30岁", "30"]

"""