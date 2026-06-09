# 使用Command进行交接
"""
使用 Command 对象实现 多智能体（Multi-Agent）之间的动态交接（Handoff）。

"""

from langgraph.graph import StateGraph, END, START, MessagesState
from typing import Literal
from langchain_core.tools import tool
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langchain_core.messages import convert_to_messages

model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    temperature=0.0 # 降低温度以提高计算确定性
)

# 定义工具，用于信号传递
@tool
def transfer_to_multiplication_expert():
    """向乘法专家求助."""
    # 这个工具不会返回任何东西:我们只是在使用它
    # 作为LLM发出需要移交给另一个代理的信号的一种方式
    return

@tool
def transfer_to_addition_expert():
    """向加法专家寻求帮助."""
    return

def addition_expert(state: MessagesState,) -> Command[Literal["multiplication_expert", "__end__"]]:
    system_prompt = (
        "您是加法专家，您可以向乘法专家寻求乘法方面的帮助。 "
        "交接之前务必做好自己的那部分计算。"
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    # 让 LLM 决定是否需要调用乘法专家的工具
    ai_msg = model.bind_tools([transfer_to_multiplication_expert]).invoke(messages)
    if len(ai_msg.tool_calls) > 0: # 如果 LLM 决定需要交接
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        
        tool_msg = { # 构造一个 ToolMessage 表示交接成功
            "role": "tool",
            "content": "成功转移d",
            "tool_call_id": tool_call_id,
        }
        # 核心：返回 Command 对象
        return Command(
            goto="multiplication_expert", # 目的地：跳转到乘法专家节点
            update={"messages": [ai_msg, tool_msg]} # 有效载荷：更新消息历史
        )
    # 如果不需要交接，就正常返回对消息的更新
    return {"messages": [ai_msg]}


def multiplication_expert( state: MessagesState,) -> Command[Literal["addition_expert", "__end__"]]:
    system_prompt = (
        "您是乘法专家，您可以向加法专家寻求加法方面的帮助。 "
        "交接之前务必做好自己的那部分计算。"
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = model.bind_tools([transfer_to_addition_expert]).invoke(messages)
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        tool_msg = {
            "role": "tool",
            "content": "成功转移",
            "tool_call_id": tool_call_id,
        }
        return Command(goto="addition_expert", update={"messages": [ai_msg, tool_msg]})

    return {"messages": [ai_msg]}



builder = StateGraph(MessagesState)
builder.add_node("addition_expert", addition_expert)
builder.add_node("multiplication_expert", multiplication_expert)
builder.add_edge(START, "addition_expert")

graph = builder.compile()
graph.get_graph().draw_png(output_file_path='./1-handover_by_nodes.png')
def pretty_print_messages(update):
    if isinstance(update, tuple):
        ns, update = update
        # 跳过打印输出中的父图更新
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"从子图更新 {graph_id}:")
        print("\n")

    for node_name, node_update in update.items():
        print(f"从node更新 {node_name}:")
        print("\n")

        for m in convert_to_messages(node_update["messages"]):
            m.pretty_print()
        print("\n")

print("开始计算...")
for chunk in graph.stream(
    {"messages": [("user", "请计算：((173 + 127) * 12 + 100) * 0.5 + 2 = ?")]}
):
    pretty_print_messages(chunk)   

print("\n------结束---------")

"""
应用场景
主要应用于 复杂任务分解与协作 的多智能体场景：
1.动态路由与专家协作：
场景：用户提出一个复合问题，如 (3 + 5) * 12 = ?。
应用：
add_expert 首先被激活。它识别出需要先做加法 3+5，但也意识到后续有乘法。它可能先计算加法，或者发现需要乘法专家来处理整体结构，于是调用 transfer_to_multiplication_expert。
控制权移交给 multi_expert。它接收上下文，执行乘法逻辑。如果它在计算过程中发现需要再次确认某个加法和（虽然本例简单，但在复杂场景中可能发生），它可以调用 transfer_to_addition_expert 将控制权交还。
优势：不同于硬编码的路由逻辑，这种基于 LLM 意图的动态交接允许 Agent 根据问题的实际复杂度自主决定协作流程。

2.模块化智能体系统：
场景：一个大型客服系统，包含“订单查询专家”、“退款专家”、“技术支持专家”。
应用：每个专家是一个独立的节点。当“订单查询专家”发现用户的问题涉及退款政策时，它可以通过 Command(goto="refund_expert") 将对话无缝移交给退款专家，同时保留完整的对话历史。

3.人机协同中的权限移交：
场景：AI 助手在处理敏感操作时。
应用：常规 AI 节点处理普通对话，当检测到高风险操作（如删除数据）时，返回 Command(goto="human_approval_node")，将控制权移交给人工审核节点。
解耦的图结构：
优势：使用 Command 进行跳转，意味着你不需要在构建图时预先定义所有可能的连接边。新增一个专家节点时，只需让其他专家知道如何“呼叫”它即可，无需修改主图的边结构。这大大提高了系统的可扩展性。


实现 自主多智能体协作（Autonomous Multi-Agent Collaboration） 的关键技术，使得 Agent 能够像人类团队一样，根据任务需求动态地传递“话语权”和“控制权”。
"""

