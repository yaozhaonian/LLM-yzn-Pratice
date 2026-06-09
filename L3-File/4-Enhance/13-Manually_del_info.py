"""
在 LangGraph 中手动管理和删除持久化存储中的消息。
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

memory = MemorySaver()


model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama" 
)

def call_model(state: MessagesState):
    """调用模型"""
    messages = state["messages"]
    result = model.invoke(messages)
    print("调用大模型回复:\n",result)
    return {"messages": result}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "2"}}
input_message = HumanMessage(content="你好，我是小兆")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

input_message = HumanMessage(content="我叫什么名字？")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

# 当前线程状态
messages = app.get_state(config).values["messages"]
print('当前',messages)
print('当前消息数量:', len(messages))

from langchain_core.messages import RemoveMessage
# 1. 删除第一条 HumanMessage
human_msg_id = messages[0].id
app.update_state(config, {"messages": RemoveMessage(id=human_msg_id)})
print(f'已删除 HumanMessage: {human_msg_id}')

# 2. 删除对应的 AIMessage (注意：此时消息列表在内存中没变，但在 Checkpoint 中变了)
# 我们需要引用原始列表中的第二个元素 ID
ai_msg_id = messages[1].id 
app.update_state(config, {"messages": RemoveMessage(id=ai_msg_id)})
print(f'已删除 AIMessage: {ai_msg_id}')

# 3. 验证结果
new_state = app.get_state(config)
new_messages = new_state.values["messages"]
print('现在剩余消息:', new_messages)

"""
批量删除(更高效)
# 获取当前状态
state = app.get_state(config)
all_messages = state.values["messages"]


# 假设我们要删除前两条消息 (index 0 和 1)
# 我们可以构建一个新的消息列表，不包含前两条
remaining_messages = all_messages[2:] 

# 直接更新状态为剩余的消息
# 注意：直接赋值列表会覆盖之前的消息历史，这在某些 reducer 配置下可能需要小心
# 但对于 MessagesState，通常建议使用 RemoveMessage 来保持版本链的完整性
# 如果必须批量删，可以这样：
from langchain_core.messages import RemoveMessage

updates = {"messages": [RemoveMessage(id=m.id) for m in all_messages[:2]]}
app.update_state(config, updates)

# 验证
print(app.get_state(config).values["messages"])
"""



"""
演示了如何获取当前会话状态，并使用 RemoveMessage 类从检查点（Checkpointer）中物理删除特定的历史消息。
"""

"""
应用场景
手动删除消息的功能适用于以下高级场景：

隐私合规与“被遗忘权”:

用户要求删除某些敏感个人信息（如手机号、地址）。系统可以根据消息 ID 或内容搜索，精准删除包含敏感信息的特定消息节点，而保留其他对话上下文。
上下文窗口优化（滑动窗口的高级版）:

虽然通常使用过滤函数（如上一个示例），但在某些情况下，可能需要物理删除极旧的、完全无关的消息以减小数据库存储压力，或者当消息包含大量 Token 且不再需要时。
纠错与编辑历史:

如果用户发现之前提供的信息有误（例如：“我刚才说错了，我不是小兆，我是大姚”），系统可以删除旧的错误信息消息，并插入新的修正信息，从而“重写”记忆历史，避免 AI 被旧信息误导。
清理无效的工具调用痕迹:

如果某次工具调用失败或产生了大量无用的中间日志消息，可以通过 ID 将其从历史中移除，保持对话流的整洁，防止干扰后续的 LLM 判断。
动态对话管理:

在多轮对话中，如果某个分支的话题已经结束，可以删除该分支相关的临时消息，只保留核心结论，以便开启新话题时不受旧话题干扰。
"""

