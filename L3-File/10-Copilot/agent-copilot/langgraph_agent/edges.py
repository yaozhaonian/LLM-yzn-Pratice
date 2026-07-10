from typing import Dict, Any


def should_continue(state: Dict[str, Any]) -> str:
    if state.get("error"):
        return "summarize"
    
    if state.get("selected_tool") is None:
        return "summarize"
    
    if state.get("is_complete", False):
        return "summarize"
    
    return "select_tool"


def has_missing_params(state: Dict[str, Any]) -> str:
    if state.get("error"):
        return "summarize"
    
    if state.get("selected_tool") is None:
        return "summarize"
    
    missing_params = state.get("missing_params", [])
    
    if missing_params:
        return "supplement_params"
    
    return "call_tool"
