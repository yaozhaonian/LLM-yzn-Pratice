"""
基于 LangGraph 构建的、包含 “人机协同”（Human-in-the-Loop, HITL） 机制的智能代理工作流。
其核心目的是在 AI 执行敏感或关键操作（如调用工具）之前，引入人工审查和干预环节。
"""
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# 1. 定义系统提示词，强制要求使用工具
SYSTEM_PROMPT = """
你是一个智能助手。当用户询问天气时，你必须使用 'weather_search' 工具。
不要直接回答天气情况，而是先调用工具获取数据。
"""

@tool
def weather_search(city:str):
    """搜索天气"""
    print("-------")
    print(f"搜索的城市:{city}")
    print("-------")
    return f"{city}天气晴朗，温度28度，湿度65%"


model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama" 
)

class State(MessagesState):
    tool: str

def call_model(state: State):
    """调用模型"""
    # 2. 绑定工具，让模型感知工具存在
    model_with_tools = model.bind_tools([weather_search])
    
    # 3. 确保消息列表中包含系统提示词（如果还没有的话）
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

def human_review_node(state: State) -> Command[Literal["call_model", "run_tool"]]:
    last_message = state["messages"][-1]
    print("human_review_node中的last_message:\n", last_message)
    
    # 安全检查
    if not last_message.tool_calls:
        return Command(goto="call_model") # 或者回到模型重新生成

    # 注意：如果有多于一个工具调用，当前逻辑只审查最后一个。
    # 最佳实践是审查所有待执行的工具调用，或者逐个审查。
    # 这里为了简化，假设我们审查第一个或全部，实际业务中可能需要循环或聚合展示。
    tool_call = last_message.tool_calls[0] # 通常审查第一个，或者遍历所有
    
    human_review = interrupt(
        {
            "question": "请审查即将执行的操作",
            "tool_call": tool_call
        }
    )
    
    review_action = human_review["action"]
    review_data = human_review.get("data")
    
    # 如果获得批准,调用该工具
    if review_action == "continue":
        return Command(goto="run_tool")
    # 需要更新时更新AI信息和工具调用
    elif review_action == "update":
        update_message = {
            "role": "ai",
            "content": last_message.content,
            "tool_calls": [
                {
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "args": review_data
                }
            ],
            "id": last_message.id
        }
        return Command(goto="run_tool", update={"messages": [update_message]})
    elif review_action == "feedback":
        toolmessage = {
            "role": "tool",
            "content": review_data,
            "name": tool_call["name"],
            "tool_call_id": tool_call["id"],
        }
        return Command(goto="call_model", update={"messages": [toolmessage]})

def run_tool(state: State):
    new_messages = []
    tools = {"weather_search": weather_search}
    tool_calls = state["messages"][-1].tool_calls
    for tool_call in tool_calls:
        tool = tools[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        new_messages.append(
            {
                "role": "tool",
                "name": tool_call["name"],
                "content": result,
                "tool_call_id": tool_call["id"],
            }
        )
    return {"messages": new_messages}

def route_after_llm(state) -> Literal[END, "human_review_node"]:
    last_message = state["messages"][-1]
    # 优先检查是否有无效的工具调用
    if hasattr(last_message, 'invalid_tool_calls') and last_message.invalid_tool_calls:
        # 处理无效调用，例如记录日志或返回错误
        print("无效工具调用,结束")
        return END
        
    if not last_message.tool_calls:
        print("无工具调用,结束")
        return END
    else:
        return "human_review_node"


builder = StateGraph(State)
builder.add_node("call_model", call_model)
builder.add_node("run_tool", run_tool)
builder.add_node("human_review_node", human_review_node)
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", route_after_llm)
builder.add_edge("run_tool", "call_model")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

graph.get_graph().draw_png(output_file_path='./3-Unrestricted_use.png')

initial_input = {"messages": [{"role": "user", "content": "你好!五一假期广州天气如何，是否适合游玩呢"}]}

thread = {"configurable": {"thread_id": "1"}}

for event in graph.stream(initial_input, thread, stream_mode="updates"):
    print("event1:\n", event)
    print("---\n")

"""
应用场景
这种“无审查使用”（标题可能意指“非自动化全权代理”，即需要人工介入）的模式适用于以下高风险或高精度要求的场景：

A. 金融交易与支付
场景: 用户要求“帮我买入 100 股苹果公司股票”。
应用: LLM 解析意图并生成交易指令。在真正调用交易 API 之前，系统中断并询问用户：“确认以当前价格买入 100 股 AAPL 吗？”。
价值: 防止因 LLM 幻觉或参数错误导致错误的资金操作。允许用户在最后一步确认金额、数量或取消交易。
B. 敏感数据操作或删除
场景: 用户要求“删除数据库中所有过期的用户记录”。
应用: LLM 生成 SQL 删除语句或调用删除 API。人工审查节点展示即将被删除的记录概要，要求管理员确认。
价值: 避免不可逆的数据丢失风险。
C. 邮件发送或对外沟通
场景: 助手起草一封给重要客户的道歉信或报价单。
应用: LLM 生成邮件内容和收件人信息。在调用 send_email 工具前，中断流程，让人类编辑邮件内容或确认收件人。
价值: 确保语气得体、信息准确，避免公关危机。
D. 复杂参数校正
场景: 用户说“查一下北京明天的天气”，但 LLM 错误地解析为“北京后天的天气”或地点识别偏差。
应用: 使用 update 动作。人类发现参数错误，直接修改 city 参数为正确的值，然后让流程继续执行工具调用。
价值: 提高工具调用的准确率，无需重新整个对话。
E. 合规与法律审查
场景: 自动生成合同条款或法律建议。
应用: 在生成最终文档或调用签署服务前，由法务人员介入审查内容合规性。

"""

"""
通过 interrupt 机制实现执行流的暂停，
通过 Command 实现了灵活的恢复策略（继续、修改、反馈），
构建可信赖、可控的 Agent 系统的关键模式。
"""

print("等待分路选择!")
# 正在等待人工审核
# print(graph.get_state(thread).next)
# 工具标准使用
for event in graph.stream(
    # provide value
    Command(resume={"action": "continue"}),
    thread,
    stream_mode="updates",
):
    print("event2:\n", event)
    print("===\n")

"""
编辑工具使用
适合的应用场景
这种“先审核，后执行”的模式特别适用于对安全性、准确性或成本敏感的场景：

高风险操作执行:

数据库写入/删除: 在执行 SQL DELETE 或 UPDATE 前，让 DBA 或开发人员确认语句是否正确，防止误删数据。
金融交易: 在发起转账或股票交易前，要求用户二次确认金额和账户。
敏感信息查询:

隐私数据访问: 当 AI 需要查询包含个人身份信息 (PII) 的记录时，需经授权人员批准。
高精度需求场景:

医疗/法律建议: AI 生成的诊断建议或法律条款引用，需由专家审核后再发送给患者或客户。
参数修正: 用户问“北京明天的天气”，AI 可能错误地理解为“北京今天的天气”。人类可以在 "update" 分支中修正参数日期，而无需重新从头对话。
成本控制:

调用昂贵的外部 API（如付费数据分析服务）前，确认该调用是必要的且参数无误。
"""
memory2 = MemorySaver()
graph2 = builder.compile(checkpointer=memory2)
initial_input = {"messages": [{"role": "user", "content": "上海的天气如何?"}]}
thread = {"configurable": {"thread_id": "3"}}

# 运行graph，直到第一次interruption
for event in graph.stream(initial_input, thread, stream_mode="updates"):
    print(event)
    print("\n")

print("Pending Executions!")
print(graph.get_state(thread).next)

for event in graph2.stream(
    Command(resume={"action": "update", "data": {"city": "上海, 中国"}}),
    thread,
    stream_mode="updates",
):
    print("event3:\n", event)
    print("~~~\n")


