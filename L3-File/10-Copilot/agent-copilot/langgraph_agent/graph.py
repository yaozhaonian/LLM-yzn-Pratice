from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import plan_task, select_tool, extract_params, supplement_params, call_tool, check_loop, summarize
from .edges import should_continue, has_missing_params


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("plan_task", plan_task)
    workflow.add_node("select_tool", select_tool)
    workflow.add_node("extract_params", extract_params)
    workflow.add_node("supplement_params", supplement_params)
    workflow.add_node("call_tool", call_tool)
    workflow.add_node("check_loop", check_loop)
    workflow.add_node("summarize", summarize)

    workflow.set_entry_point("plan_task")
    workflow.add_edge("plan_task", "select_tool")
    workflow.add_edge("select_tool", "extract_params")
    workflow.add_conditional_edges("extract_params", has_missing_params)
    workflow.add_edge("supplement_params", "call_tool")
    workflow.add_edge("call_tool", "check_loop")
    workflow.add_conditional_edges("check_loop", should_continue)
    workflow.add_edge("summarize", END)

    return workflow


def compile_graph(workflow):
    return workflow.compile()
