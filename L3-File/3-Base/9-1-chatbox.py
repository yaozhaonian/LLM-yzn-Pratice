"""
基本ChatBot+Tools+Memory
基于 LangGraph 构建的基础聊天机器人（Chatbot），它集成了 工具调用（Tool Use） 和 记忆管理（Memory/Checkpointing）。
一个典型的 ReAct（Reasoning + Acting）模式的简化实现，通过手动构建图结构来理解 LangGraph 的核心工作原理。
"""
from langgraph.graph import StateGraph, END, START, add_messages
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, ToolMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict
import json
from langchain_tavily import TavilySearch

model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

tool = TavilySearch(max_results=2, api_key="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")
tools = [tool]

model_with_tools = model.bind_tools(tools)
memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

builder = StateGraph(State)

def cahtbot(state: State):
    messages = state["messages"]
    
    # 如果第一条消息不是系统消息，可以 prepend 一个系统消息
    # 注意：这只是一个简单示例，实际生产中应更严谨地管理历史消息
    if not any(isinstance(m, SystemMessage) for m in messages):
        system_prompt = SystemMessage(content="你是一个智能助手。如果需要获取实时信息，请使用提供的工具。请严格按照工具调用的格式输出。")
        messages = [system_prompt] + messages
        
    return {"messages": [model_with_tools.invoke(messages)]}

class BasicToolNode:
    def __init__(self, tools: list):
        self.tools_by_name = {tool.name: tool for tool in tools}
    
    # __call__，Python中的魔法方法，使得类的对象实例可以像函数一样被调用
    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("输入中没有信息")
        outputs = []
        for tool_call in message.tool_calls:
            print("tool_call:", tool_call)
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    name=tool_call["name"],
                    content=json.dumps(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

# 路由选择
def route_tools(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


tool_node = BasicToolNode(tools=[tool])
builder.add_node("tools", tool_node)
builder.add_node("chatbot", cahtbot)
builder.add_edge("tools", "chatbot")
builder.add_edge(START, "chatbot")
builder.add_conditional_edges('chatbot', route_tools, {"tools":"tools", END: END})
# builder.add_edge("chatbot", END)

graph = builder.compile(checkpointer=memory)
graph.get_graph().draw_png(output_file_path='./chatbox.png')

# 与模型交互
def stream_graph_updata(user_input: str):
    # 定义配置
    config = {"configurable": {"thread_id": "1"}}
    
    print("\n--- 开始思考 ---")
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config=config):
        # event 是一个字典，key 是节点名称，value 是状态更新
        for node_name, state_update in event.items():
            messages = state_update.get('messages', [])
            if not messages:
                continue
                
            last_msg = messages[-1]
            
            # 判断消息类型
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                # 这是 LLM 决定调用工具
                print(f"[{node_name}] 正在调用工具: {last_msg.tool_calls[0]['name']}...")
                
            elif hasattr(last_msg, 'name') and last_msg.name: 
                # 这是工具返回的结果 (ToolMessage)
                # 为了不刷屏，我们可以选择不打印详细内容，或者只打印摘要
                print(f"[{node_name}] 工具已返回结果 (长度: {len(last_msg.content)})")
                
            else:
                # 这是 LLM 的最终文本回复
                print(f"[{node_name}] 最终回复:")
                print("-" * 20)
                print(last_msg.content)
                print("-" * 20)

# 运行图
while True:
    try:
        config = {"configurable": {"thread_id": "1"}}
        user_input = input("用户: ")
        if user_input in ['quit', 'exit', 'q']:
            print("退出程序")
            break
        # 写一个函数用于讲user_input与LLM进行交互
        stream_graph_updata(user_input)
    except Exception as e:
        print(f"错误: {e}")



"""
应用场景
这段代码主要应用于 需要实时信息检索和多轮对话能力的智能助手场景：

1,增强型问答机器人：
场景：用户询问“今天北京天气怎么样？”或“最近有什么科技新闻？”。
应用：LLM 自身知识截止或无法获取实时数据时，通过调用 TavilySearch 工具获取最新信息，然后结合上下文生成准确回答。

2.具备记忆功能的客服助手：
场景：用户在多轮对话中引用之前的内容，如“我刚才问的那个问题，再详细解释一下”。
应用：得益于 MemorySaver 和 thread_id，机器人能记住整个对话历史，提供连贯的服务体验。

3.学习与调试 LangGraph 基础架构：
场景：开发者希望深入理解 LangGraph 底层如何运作，而不是直接使用高级封装（如 create_react_agent）。
应用：此代码手动实现了节点、边、条件路由和工具执行逻辑，是理解 LangGraph “状态机”本质的最佳入门示例。它展示了数据如何在 messages 状态中流动，以及控制流如何在 LLM 和工具之间切换。
"""
"""
该文件是一个最小可行产品（MVP）级别的 ReAct Agent 实现。它虽然没有使用 LangGraph 的高级预构建组件（如 ToolNode 或 create_react_agent），但清晰地展示了构建智能 Agent 的四大支柱：

状态管理（通过 State 和 add_messages）
推理核心（LLM 节点）
行动能力（工具节点）
控制流（条件路由）
持久化记忆（Checkpointer）
"""



