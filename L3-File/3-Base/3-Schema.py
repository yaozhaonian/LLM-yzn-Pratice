# 输入/输出模式隔离
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

# 展示 LangGraph 中 Input/Output Schema（输入/输出模式隔离） 的概念
# 定义输入Schema
class InputState(TypedDict):
    question: str

# 定义输出Schema
class OutputState(TypedDict):
    answer: str

# 结合输入和输出，定义总体模式
class OverallState(InputState, OutputState):
    pass

# 定义处理输入并生成答案的节点
def answer_node(state: InputState):
    print("question:", state["question"])
    return {"answer_question": "回答问题", "answer": "简单回复即可.", "question": state["question"]}

# 使用指定的输入和输出模式构建图形
"""
结果过滤：
因为设置了 output_schema=OutputState，LangGraph 在最终返回结果时，丢弃了 question 字段，只保留了 answer。
"""
builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)

builder.add_node("answer", answer_node)

builder.add_edge(START, "answer")
builder.add_edge("answer", END)

graph = builder.compile()

response = graph.invoke({"question": "你好"})
print("开始:\n", response)

class OutputState2(TypedDict):
    answer_question: str

class OverallState2(InputState, OutputState2):
    pass

builder2 = StateGraph(OverallState2, input_schema=InputState, output_schema=OutputState2)
builder2.add_node("answer", answer_node)

builder2.add_edge(START, "answer")
builder2.add_edge("answer", END)

graph2 = builder2.compile()
response2 = graph2.invoke({"question": "hello"})
print("开始2:\n", response2)

class OutputState3(TypedDict):
    answer: str
    answer_question: str

class OverallState3(InputState, OutputState3):
    pass

builder3 = StateGraph(OverallState3, input_schema=InputState, output_schema=OutputState3)
builder3.add_node("answer", answer_node)

builder3.add_edge(START, "answer")
builder3.add_edge("answer", END)

graph3 = builder3.compile()
response3 = graph3.invoke({"question": "hi"})
print("开始3:\n", response3)
