"""
人工介入（Human-in-the-Loop） 
具体通过 interrupt 机制实现。
核心功能是让图在执行过程中暂停，等待外部（如用户界面、API 调用或人工审核）提供决策，然后根据决策结果继续执行不同的分支。
"""
from typing import TypedDict, Optional, Literal
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

class ApprovalState(TypedDict):
    action_details: str
    approval: Optional[Literal["pending", "approved", "rejected"]]

def approval_node(state: ApprovalState) -> Command[Literal["proceed", "cancel"]]:
    print(f"节点approval_node :{state}")
    decision = interrupt({
        "question": "批准此操作?",
        "details": state["action_details"]
    })
    # 中断恢复后路由到适当的节点
    return Command(goto="proceed" if decision else "cancel")

def agree_node(state: ApprovalState):
    print(f"节点agree_node(允许) :{state}")
    return {"approval": "approved"}

def cancel_node(state: ApprovalState):
    print(f"节点cancel_node(拒绝) :{state}")
    return {"approval": "rejected"}

builder = StateGraph(ApprovalState)
builder.add_node("approval", approval_node)
builder.add_node("proceed", agree_node)
builder.add_node("cancel", cancel_node)

builder.add_edge(START, "approval")

builder.add_edge("proceed", END)
builder.add_edge("cancel", END)

# 使用持久检查点
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "reject-123"}}
initial = graph.invoke(
    {"action_details": "删除文件?", "approval": "pending"}, 
    config=config
)
print(initial["__interrupt__"])  # -> [Interrupt(value={'question': ..., 'details': ...})]

resumed = graph.invoke(Command(resume=False), config=config)

print(resumed["approval"])
print("="*50)
history = list(graph.get_state_history(config))
# 获取状态快照的历史
print("历史1数量:", len(history))
for i, item in enumerate(history):
    print(f"历史快照{i+1}:\n类型:{type(item)}\n内容:{item}")
print("="*50)

config2 = {"configurable": {"thread_id": "approve-123"}}
initial2 = graph.invoke(
    {"action_details": "删除文件?", "approval": "pending"}, 
    config=config2
)
print(initial2["__interrupt__"])  # -> [Interrupt(value={'question': ..., 'details': ...})]

resumed2 = graph.invoke(Command(resume=True), config=config2)

print(resumed2["approval"])
print("="*50)
history2 = list(graph.get_state_history(config2))
# 获取状态快照的历史
print("历史2数量:", len(history2))
for i, item in enumerate(history2):
    print(f"历史快照{i+1}:\n类型:{type(item)}\n内容:{item}")