"""
使用 LangGraph 构建一个多智能体协作网络（Multi-Agent Network）。
在这个网络中，不同的专家 Agent（节点）通过动态交接（Handoff）机制协同工作，共同完成一个复杂的用户请求。
"""
from langgraph.graph import StateGraph, END, START, MessagesState
from typing import Literal, TypedDict, List
from langgraph.prebuilt import ToolNode, tools_condition, ToolRuntime
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import convert_to_messages, HumanMessage, AnyMessage, ToolMessage
from langgraph.types import Command
import json

model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    temperature=0.0 # 降低温度以提高计算确定性
)

@tool
def transfer_to_travel_advisor():
    """向旅行顾问寻求帮助。"""
    # 此工具不返回任何内容：我们只是使用它
    # 作为 LLM 发出需要移交给其他代理的信号
    # （参见上文）
    return

@tool
def transfer_to_hotel_advisor():
    """向酒店顾问寻求帮助。"""
    return

def travel_advisor(
    state: MessagesState,
) -> Command[Literal["hotel_advisor", "__end__"]]:
    system_prompt = (
        "您是一位可以推荐旅游目的地（例如国家、城市等）的综合旅游专家。"
        "如果您需要酒店推荐，请向“hotel_advisor”寻求帮助。"
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    # LLM 绑定了可以调用 "transfer_to_hotel_advisor" 工具
    ai_msg = model.bind_tools([transfer_to_hotel_advisor]).invoke(messages)
    print(f"LLM 生成的回复: \n{ai_msg}\n")
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        
        tool_msg = { # 构造一个 ToolMessage 表示交接成功
            "role": "tool",
            "content": "成功转移",
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto="hotel_advisor",
            update={"messages": [ai_msg, tool_msg]},
        )
    
def hotel_advisor(
    state: MessagesState,
) -> Command[Literal["travel_advisor", "__end__"]]:
    system_prompt = (
        "您是一位酒店专家。您的任务是承接 'travel_advisor' 的工作。"
        "**请仔细检查完整的对话历史记录。'travel_advisor' 在调用工具将任务移交给您之前，一定会在其回复中明确推荐一个旅游目的地（例如 '巴巴多斯' 或 '巴哈马'）。**"
        "您的唯一任务是：1. 在历史记录中找到这个目的地。 2. 为这个已确定的目的地提供酒店推荐。"
        "**请不要评论“转移”这个过程本身，直接开始推荐酒店。**"
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = model.bind_tools([transfer_to_travel_advisor]).invoke(messages)
    # 如果有工具调用，LLM 需要移交给另一个代理
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        tool_msg = {
            "role": "tool",
            "content": "成功转移",
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto="travel_advisor",
            update={"messages": [ai_msg, tool_msg]},
        )
    return {"messages": [ai_msg]}



builder = StateGraph(MessagesState)
builder.add_node("travel_advisor", travel_advisor)
builder.add_node("hotel_advisor", hotel_advisor)
builder.add_edge(START, "travel_advisor")
graph = builder.compile()
# graph.get_graph().draw_png(output_file_path='./3-intelligent_agent_network.png')

def pretty_print_messages(update):
    if isinstance(update, tuple):
        ns, update = update
        if not ns: return
        graph_id = ns[-1].split(":")[0]
        print(f"[子图 {graph_id}]")

    if isinstance(update, dict):
        for node, val in update.items():
            print(f"→ 节点 {node}:")
            try:
                if isinstance(val, dict) and "messages" in val:
                    for m in convert_to_messages(val["messages"]):
                        m.pretty_print()
            except:
                pass
    print("-" * 60)


for chunk in graph.stream(
    {"messages": [("user", "我想去加勒比海某个温暖的地方")]}
):
    pretty_print_messages(chunk)

print('###################################')

for chunk in graph.stream(
    {
        "messages": [
            (
                "user",
                "我想去加勒比海某个温暖的地方。选一个目的地，然后给我推荐酒店",
            )
        ]
    }
):
    pretty_print_messages(chunk)

"""
应用场景
这段代码主要应用于 需要多领域专家协作的复杂咨询服务场景：

分层式客户服务系统：

场景：用户咨询涉及多个部门（如先咨询产品特性，再咨询价格，最后咨询物流）。
应用：
travel_advisor 类比“产品专家”，负责确定用户感兴趣的具体产品。
hotel_advisor 类比“销售/物流专家”，负责基于确定的产品提供报价或配送方案。
通过共享的消息历史，后一个专家无需用户重复提供信息，实现了无缝衔接。
复杂决策支持系统：

场景：金融理财建议。
应用：
“风险评估专家”先评估用户风险偏好。
“投资组合专家”根据评估结果推荐具体基金。
“税务专家”最后分析税务影响。
每个专家专注于自己的领域，通过 Command 传递控制权。
模块化 Prompt 工程：

优势：通过将大任务拆解为小任务（选目的地 vs 选酒店），每个节点的 System Prompt 可以更专注、更简单，从而减少 LLM 的幻觉，提高特定领域的回答质量。
上下文管理：代码展示了如何利用 LangGraph 的 MessagesState 自动维护上下文，使得后续节点可以“看到”前序节点的推理结果，这对于多步推理至关重要。
动态工作流编排：

优势：如果用户只问目的地，工作流在 travel_advisor 就结束了；如果用户问酒店，工作流会自动延伸到 hotel_advisor。这种按需激活节点的模式比线性工作流更高效、更灵活。
"""


