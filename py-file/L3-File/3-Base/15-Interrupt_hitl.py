"""
使用 LangGraph 的 interrupt 机制实现**人机协作（Human-in-the-Loop, HITL）**的工作流。
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    text: str
    
checkpointer = MemorySaver()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 初始化 LLM
llm = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

# 人类介入节点
def human_node(state: State):
    print("人类介入:", state)
    value = interrupt({
        "text_to_revise": state["text"],
        "instructions": "输入修改要求"
    })
    print(f"介入节点值:{value}")
    # 返回用户输入的内容
    return {"text": value}

# 自动处理节点
def proceed_text(state: State):
    print("自动处理节点:", state)
    # 强化格式要求的提示词
    message = llm.invoke([
        HumanMessage(content=f"""当前请求:{state['text']}""")
    ])
    return {"text": message.content}


# 构建循环工作流
builder = StateGraph(State)
builder.add_node("proceed_text", proceed_text)
builder.add_node("human_node", human_node)

# 设置循环流程
builder.add_edge(START, "proceed_text")
builder.add_edge("proceed_text", "human_node")
builder.add_edge("human_node", "proceed_text")  # 新增循环连接


graph = builder.compile(checkpointer=checkpointer)

# graph.get_graph().draw_png(output_file_path='./15-Interrupt_hitl.png')

if __name__ == "__main__":
    thread_id = "thread_abc"
    thread_config = {"configurable": {"thread_id": thread_id}}
    
    # 初始执行
    initial_state = {"text": "使用python语法生成五五乘法表"}
    result = graph.invoke(initial_state, config=thread_config)
    print("初始执行结果\n", result)
    
    # 检查并显示中断信息
    interrupt_info = None
    if "__interrupt__" in result:
        interrupts = result["__interrupt__"]
        if interrupts:
            # 获取第一个中断的信息
            interrupt_value = interrupts[0].value
            interrupt_info = interrupt_value['instructions']
            print("中断信息:\n")
            print(f"  文本范例: {interrupt_value['text_to_revise']}")
            print(f"  提示信息: {interrupt_info}")
        else:
            print("中断列表为空")
    else:
        print("没有中断信息")
        
    # 人类输入环节(Command.resume)
    # 使用中断中的提示信息
    prompt = interrupt_info if interrupt_info else "请输入修改要求:"
    resume_result = graph.invoke(
        Command(resume=input(f"\n{prompt}")),
        config=thread_config
    )
    """
    进阶:可以使用while，再写个函数让人判断满不满意，哪里不满意，再返回进行修改
    """
    print("\n最终处理结果:\n", resume_result["text"], "\n类型:", type(resume_result))



"""
实现一个交互式 AI 助手原型：

AI 先生成初步结果。
暂停并展示给用户。
用户提出修改意见。
AI 根据意见重新生成。
（可选）重复步骤 2-4 直到满意。
这种模式非常适合代码生成、文案创作等需要多次迭代和人工审核的场景。
"""

