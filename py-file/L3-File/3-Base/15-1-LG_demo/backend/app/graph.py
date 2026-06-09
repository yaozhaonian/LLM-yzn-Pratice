from langchain_ollama import ChatOllama
from typing import Literal, Optional
from langgraph.graph import StateGraph, END, START, MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

llm = ChatOllama(
    model='qwen2.5:7b', 
    temperature=0,
    base_url="http://127.0.0.1:11434" 
)

class DraftReviewState(MessagesState):
    human_request: str
    human_comment: Optional[str]
    status: Literal["approved", "feedback"]
    assistant_response: str

# 用户初始请求处理以及反馈请求处理
def assistant_draft(state: DraftReviewState) -> DraftReviewState:
    print(f"========> 用户请求处理\n 状态:{state}")
    user_message = HumanMessage(content=state["human_request"])
    status = state.get("status", "approved")
    
    if (status == "feedback" and state.get("human_comment")):
        human_comment = HumanMessage(content=state["human_comment"])
        
        system_message = SystemMessage(content=(f"""
        你是一个人工智能助手，正在修改你以前的草稿。仔细检查用户的反馈，并相应地更新您的回复。
        处理用户提供的所有意见、纠正或建议。确保修改后的回复充分整合了反馈，提高了清晰度，并解决了提出的任何问题。
        """))
        
        messages = [user_message] + state["messages"] + [system_message, human_comment]
        all_messages = state["messages"] + [human_comment]
    else:
        system_message = SystemMessage(content=("""
        你是一个AI助手。你的目标是充分理解和满足用户的需求
        通过准备一份相关、清晰、有用的回复草案来提出请求。专注于直接、全面地满足用户的需求。
        在此阶段，不要参考任何之前的人工反馈。
        """))
        messages = [system_message, user_message]
        all_messages = state["messages"]
        
    response = llm.invoke(messages)
    
    all_messages = all_messages + [response]
    return {
        **state,
        "messages": all_messages,
        "assistant_response": response.content,
    }

def human_feedback(state: DraftReviewState) -> None:
    print(f"========> 用户反馈处理\n 状态:{state}")

# 用户批准请求处理
def assistant_finalize(state: DraftReviewState) -> DraftReviewState:
    print(f"========> 用户批准处理\n 状态:{state}")
    system_message = """
    你是人工智能助手。用户已批准您的草稿。仔细地检查你的回复，并对清晰度、语气和完整性进行最终改进。
    确保准备作为最终答案发表的回应是完美的、专业的。
    """
    messages = [system_message] + state["messages"]
    response = llm.invoke(messages)
    
    all_messages = state["messages"] + [response]
    return {
        **state,
        "messages": all_messages,
        "assistant_response": response.content,
    }

def feedback_router(state: DraftReviewState) -> str:
    print(f"========> 反馈处理路由\n 状态:{state}")
    # 修正拼写：approved
    if state["status"] == "approved":
        return "assistant_finalize"
    else:
        return "assistant_draft"

builder = StateGraph(DraftReviewState)

builder.add_node("assistant_draft", assistant_draft)
builder.add_node("human_feedback", human_feedback)
builder.add_node("assistant_finalize", assistant_finalize)

builder.add_edge(START, "assistant_draft")
builder.add_edge("assistant_draft", "human_feedback")
builder.add_conditional_edges("human_feedback", feedback_router, {"assistant_finalize": 'assistant_finalize', 'assistant_draft': 'assistant_draft'})
builder.add_edge("assistant_finalize", END)

memory = MemorySaver()
graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)
graph.get_graph().draw_mermaid_png(output_file_path='./LG-demo.png')

# 导出
__all__ = ["graph", "DraftReviewState"]



