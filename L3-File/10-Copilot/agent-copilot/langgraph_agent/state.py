from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict):
    query: str
    task_description: str
    api_chain: List[dict]
    selected_tool: Optional[Any]
    params: Dict[str, Any]
    missing_params: List[Any]
    tool_result: str
    is_single_task: bool
    is_complete: bool
    loop_count: int
    summary: str
    error: Optional[str]
    topK: int
    api_planning_hub: Any
    api_selection_hub: Any
    param_extraction_hub: Any
    generate_task_hub: Any
    tool_summary_hub: Any
    tool_use_hub: Any
