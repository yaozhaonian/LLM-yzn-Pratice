from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Sequence
from operator import add
from langgraph.graph import StateGraph, START, END

# 初始化 LLM
llm = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)


class ChatState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add]

def handle_user_input(state: ChatState):
    """
    注意：在真实的 Web 服务中，不应在节点中使用 input()。
    这里为了演示控制台交互保留，但需小心处理退出逻辑。
    """
    try:
        user_input = input("用户输入(输入'退出'结束): \n").strip()
        if user_input.lower() == "退出":
            # 返回一个标记，或者直接在这里不返回新消息，由主循环判断
            # 为了保持 Graph 纯净，我们返回一个特殊的 HumanMessage 或者让主循环处理退出
            print("用户选择退出")
            # 这里我们返回一个包含退出指令的消息，供后续节点或主循环判断
            return {"messages": [HumanMessage(content="退出")]}
        
        # 正常返回用户消息
        return {"messages": [HumanMessage(content=user_input)]}
    except Exception as e:
        print(f"输入错误: {e}")
        return {"messages": [HumanMessage(content="退出")]} # 出错也退出

def handle_ai_response(state: ChatState):
    try:
        # 获取历史记录
        all_messages = state["messages"]
        
        # 简单的上下文截取：取最近 6 条消息用于发送给 LLM
        # 注意：发送给 LLM 的需要是纯消息列表，不包含其他元数据
        context_messages = all_messages[-6:]
        
        print("="*50)
        print(f"AI 正在响应... (上下文长度: {len(context_messages)})")
        print("="*50)
        
        # 调用 LLM
        response = llm.invoke(context_messages)
        
        print(f"模型回复:\n {response.content}\n")
        
        # 2. 修正：直接返回包含 AIMessage 的列表，不要嵌套列表
        return {"messages": [response]}
        
    except Exception as e:
        error_msg = f"系统暂时无法响应，请稍后再试（错误代码：{str(e)[:30]})"
        print(f"LLM 错误: {e}")
        return {"messages": [AIMessage(content=error_msg)]}

# 构建图
builder = StateGraph(ChatState)
builder.add_node("用户输入", handle_user_input)
builder.add_node("AI处理", handle_ai_response)

# 设置入口
builder.set_entry_point("用户输入")
builder.add_edge("用户输入", "AI处理")
builder.add_edge("AI处理", END)

graph = builder.compile()

if __name__ == "__main__":
    system_prompt = "你是一个专业级中文智能助手！"
    # 4. 修正：初始化状态时键名为 messages
    state = {"messages": [SystemMessage(content=system_prompt)]}
    
    num = 1
    while True:
        try:
            # 执行一步：用户输入 -> AI处理
            result = graph.invoke(state)
            
            # 获取最新产生的消息
            # 由于使用了 add reducer，result['messages'] 包含了所有历史消息
            # 我们需要找出本轮新增的消息
            prev_len = len(state["messages"])
            curr_len = len(result["messages"])
            
            if curr_len > prev_len:
                new_messages = result["messages"][prev_len:]
                
                for msg in new_messages:
                    if isinstance(msg, HumanMessage):
                        if msg.content.lower() == "退出":
                            print("检测到退出指令，结束对话。")
                            # 更新 state 以便最后打印完整记录
                            state = result 
                            raise StopIteration # 跳出 while 循环
                    
                    if isinstance(msg, AIMessage):
                        print(f"第{num}轮 AI: \n{msg.content}\n")
            
            # 更新状态
            state = result
            num += 1
            
            # 安全检查：防止无限循环或状态异常
            if not result or "messages" not in result:
                break
                
        except StopIteration:
            break
        except Exception as e:
            print(f"运行错误: {e}")
            break
    
    print("="*50, "结束", "="*50)
    print("[完整对话记录如下]")
    # 5. 修正：访问 state["messages"]
    for i, msg in enumerate(state["messages"]):
        if isinstance(msg, SystemMessage):
            continue # 跳过系统提示
        prefix = "用户：" if isinstance(msg, HumanMessage) else "AI："
        # 截断过长的内容以便展示
        content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"{i}. {prefix}{content_preview}")