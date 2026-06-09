"""
在 LangGraph 中构建高级 Agent 工作流，重点在于自定义工具执行逻辑和错误恢复机制（Fallback/Retry）。
引入工程化的错误处理思维：检测错误 -> 清理现场 -> 降级重试
"""

# 自定义策略
from typing import Literal
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, ToolMessage, RemoveMessage
import json

class SevenPoemRequest(BaseModel):
    # 注意：如果希望 LLM 更容易调用，建议将 list[str] 改为 str，或者确保 LLM 能正确生成列表
    # 这里保持原样，但要注意 LLM 必须生成正确的 JSON 列表格式
    topic: str = Field(description="诗歌的主题")

model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    temperature=0
)

@tool
def master_sevenpoem_generator(topic: str):
    """基于提供的主题生成七言绝句"""
    print("生成七言绝句:\n",topic)
    chain = model | StrOutputParser()
    sevenpoem = chain.invoke(f"请生成一个关于{topic}的李白风格的七言绝句")
    return sevenpoem

tool_node = ToolNode([master_sevenpoem_generator])

model_tools = model.bind_tools([master_sevenpoem_generator])


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_tools.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

# 定义我们将在其间循环的两个节点
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

graph = workflow.compile()
# graph.get_graph().draw_png(output_file_path="./2-Custom_Strategy.png")

response = graph.invoke(
    {"messages": [("human", "请生成一个关于广西桂林山水的七言绝句")]},
    {"recursion_limit": 10}
)

for message in response["messages"]:
    string_representation = f"{message.type.upper()}: {message.content}\n"
    print('string_representation：',string_representation)

print("###################")

for event in graph.stream(
    {"messages": [("human", "给我写一首关于云南的苍山洱海的七言绝句")]},
    {"recursion_limit": 10},
):
    print(event)
    print("\n")

def call_tool(state: MessagesState):
    tools_by_name = {master_sevenpoem_generator.name: master_sevenpoem_generator}
    messages = state["messages"]
    last_message = messages[-1]
    output_messages = []
    for tool_call in last_message.tool_calls:
        try:
            tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            output_messages.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        except Exception as e:
            # 如果工具调用失败，则返回错误
            output_messages.append(
                ToolMessage(
                    content="",
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                    additional_kwargs={"error": e},
                )
            )
    return {"messages": output_messages}

def route_from_agent(state: MessagesState):
    """
    决定从 call_agent 节点去哪里：
    1. 如果有工具调用 -> 去 call_tools
    2. 如果没有工具调用 -> 结束 (END)
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # 检查是否有工具调用
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "call_tools"
    
    # 如果没有工具调用，说明任务完成或需要结束
    return END

def should_fallback(
    state: MessagesState,
) -> Literal["remove_failed_tool_call_attempt", "fallback_model"]: # 修改返回类型提示
    messages = state["messages"]
    failed_tool_messages = [
        msg
        for msg in messages
        if isinstance(msg, ToolMessage)
        and msg.additional_kwargs.get("error") is not None
    ]
    if failed_tool_messages:
        return "remove_failed_tool_call_attempt"
    
    # 如果工具执行成功，通常应该回到 agent 继续对话，或者根据业务逻辑结束
    # 在你的原逻辑中，似乎是想让成功的工具调用也进入 fallback_model? 
    # 但通常成功的工具调用应该返回给 agent 让其总结。
    # 这里假设：如果没出错，我们回到 agent (call_agent) 让 LLM 根据工具结果生成最终回复
    return "call_agent" 

def remove_failed_tool_call_attempt(state: MessagesState):
    messages = state["messages"]
    # 从最近的信息中删除所有信息
    print("从最近信息中删除所有信息")
    # AIMessage的实例。
    last_ai_message_index = next(
        i
        for i, msg in reversed(list(enumerate(messages)))
        if isinstance(msg, AIMessage)
    )
    messages_to_remove = messages[last_ai_message_index:]
    return {"messages": [RemoveMessage(id=m.id) for m in messages_to_remove]}


def call_fallback_model(state: MessagesState):
    messages = state["messages"]
    response = model_tools.invoke(messages)
    return {"messages": [response]}

# 修正 should_fallback 以匹配 work 图的节点名
def route_from_tools(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # 检查最后一条消息是否是 ToolMessage 且包含错误
    if isinstance(last_message, ToolMessage) and last_message.additional_kwargs.get("error"):
        return "remove_failed_tool_call_attempt"
    
    # 如果工具执行成功，返回给 agent 进行后续处理（如总结）
    return "call_agent"

work = StateGraph(MessagesState)

work.add_node("call_agent", call_model)
work.add_node("call_tools", call_tool) # 使用自定义的 call_tool
work.add_node("remove_failed_tool_call_attempt", remove_failed_tool_call_attempt)
work.add_node("fallback_model", call_fallback_model)

work.add_edge(START, "call_agent")

# 【修复点 1】使用新的路由函数，并明确指定可能的输出路径
# 路径包括: "call_tools" 和 END
work.add_conditional_edges(
    "call_agent", 
    route_from_agent, 
    {
        "call_tools": "call_tools",
        END: END
    }
)

# 【修复点 2】使用新的路由函数
# 路径包括: "remove_failed_tool_call_attempt" 和 "call_agent"
work.add_conditional_edges(
    "call_tools", 
    route_from_tools,
    {
        "remove_failed_tool_call_attempt": "remove_failed_tool_call_attempt",
        "call_agent": "call_agent"
    }
)

work.add_edge("remove_failed_tool_call_attempt", "fallback_model")
work.add_edge("fallback_model", "call_agent")

graph2 = work.compile()
# graph2.get_graph().draw_png(output_file_path="./2-Custom_Strategy(2).png")

print("33333333333333")
stream = graph2.stream(
    {"messages": [("human", "给我写一首关于广州白云山的七言绝句")]},
    {"recursion_limit": 10},
)

for chunk in stream:
    print(chunk)
    
"""
应用场景
这种自定义策略主要适用于以下生产级高可用场景：

1. 不稳定工具或外部 API 的容错处理
场景：Agent 需要调用一个不稳定的第三方 API（如天气接口、股票数据），该接口偶尔会超时或返回格式错误。
价值：默认 ToolNode 遇到异常可能会中断整个图。自定义 call_tools 可以捕获异常，将其转化为消息，并通过 fallback 机制让 Agent 尝试重新生成参数或告知用户服务暂时不可用，而不是直接崩溃。
2. 小模型的“自我修正” (Self-Correction)
场景：使用参数量较小的本地模型（如 7B）时，模型经常生成错误的 JSON 参数或调用不存在的工具。
价值：
当小模型调用失败时，remove_failed_tool_call_attempt 擦除错误记忆。
fallback_model 如果是更强的模型，可以纠正参数。
即使 fallback 也是同模型，清除上下文中的“错误示范”也能提高下一次生成正确的概率（避免 In-context Learning 带来的负面引导）。
3. 成本控制与分级路由
场景：为了省钱，默认使用便宜的本地小模型。只有在小模型多次尝试失败后，才调用昂贵的云端大模型。
价值：通过在 fallback_model 节点切换不同的 LLM 实例，实现分级降级策略。绝大多数简单请求由小模型完成，只有极少数复杂或出错请求才会触发大模型，从而平衡成本与效果。
4. 防止死循环
场景：模型固执地重复相同的错误工具调用。
价值：通过 RemoveMessage 强制移除错误的 AI 消息，打破了模型“看到自己刚才这么做过”的心理暗示，迫使它在新的上下文中重新思考，从而跳出死循环。
"""
