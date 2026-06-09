"""
引入 RemainingSteps 这一“托管状态”（Managed State），让图能够在达到最大步数限制之前，主动检测到剩余步数不足并正常退出，从而返回当前已积累的状态，而不是抛出错误。
"""
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.errors import GraphRecursionError

class State(TypedDict):
    value: str
    action_result: str
    remaining_steps: RemainingSteps

"""
主动终止机制: 这里检查 state["remaining_steps"] <= 2。如果剩余步数很少（例如只剩最后几步），路由器会强制返回 END。
"""

def router(state: State):
    print("进入选择节点,remaining_steps:\n", state["remaining_steps"])
    if state["remaining_steps"] <= 2:
        return END
    if state["value"] == "end":
        return END
    else:
        return "action"

def decision_node(state):
    return {"value": "继续"}

def action_node(state: State):
    print(f"执行了{state['value']}")    # 还是执行了很多次的感觉
    return {"action_result": "执行操作"}

builder = StateGraph(State)
builder.add_node("decision", decision_node)
builder.add_node("action", action_node)
builder.add_edge(START, "decision")
builder.add_conditional_edges("decision", router, ["action", END])
builder.add_edge("action", "decision")

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="./5-Relimit.png")


try:
    # result = graph.invoke({"value": "开始"})  # 不限制的话，开始值很大
    result = graph.invoke(
        {"value": "开始"}, 
        config={"recursion_limit": 10}
    )
    print(result)
except GraphRecursionError:
    print("Recursion Error")


"""
应用场景
这种“基于剩余步数的主动退出”模式在实际生产环境中比“捕获异常”更加健壮和有用，主要应用于以下场景：

(1) 需要部分结果的 Agent 任务
场景: 一个 Agent 正在撰写长篇报告或进行复杂的数据分析，步骤非常多。
问题: 如果因为 Token 限制或时间限制导致步数耗尽，直接报错会让用户什么都得不到。
解决方案: 使用 RemainingSteps。当步数即将耗尽时，Agent 可以检测到并停止生成新内容，转而调用一个“总结节点”，将目前为止已生成的部分内容整理后返回给用户。
优势: 用户能看到“已完成的部分”，而不是一个冰冷的错误页面。
(2) 实时反馈与降级策略
场景: 在线客服机器人或交互式游戏 NPC。
问题: 对话陷入死循环或过于冗长，用户体验变差。
解决方案: 当 remaining_steps 较低时，路由到一个特殊的“收尾节点”。该节点可以输出一句：“抱歉，我好像有点迷路了，我们可以换个话题吗？”或者“由于上下文限制，我只能回答到这里。”
优势: 提供了友好的用户体验降级（Graceful Degradation），而不是程序崩溃。
(3) 调试与监控
场景: 开发复杂的 ReAct Agent。
问题: 想知道 Agent 是因为“找到了答案”而停止，还是因为“步数用尽”而被迫停止。
解决方案: 通过检查最终状态中的 remaining_steps 或记录路由路径，开发者可以明确区分正常结束和因资源限制而结束的情况，从而优化 Prompt 或调整最大步数限制。
(4) 多阶段工作流中的超时控制
场景: 一个包含多个子图的主工作流。
问题: 某个子图可能卡住，影响整体流程。
解决方案: 在子图中使用 RemainingSteps。如果子图意识到剩余步数不足以完成其内部逻辑，它可以提前退出并返回一个“超时”或“部分完成”的标志，主工作流可以根据这个标志决定是重试、跳过还是报警。
"""

