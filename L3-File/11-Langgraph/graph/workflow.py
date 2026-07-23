from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from graph.state import ConversationState, init_state
from graph.nodes import (
    intent_route_node,
    data_query_node,
    shipment_check_node,
    rag_answer_node,
    manual_confirm_node,
    shipment_execution_node,
    generate_reply_node
)
from graph.param_validation import (
    param_validation_node,
    route_after_param_validation
)
from utils.logger import get_logger

logger = get_logger(__name__)

INTENT_TO_NODE = {
    "订单查询": "data_query",
    "库存查询": "data_query",
    "生产进度查询": "data_query",
    "货款查询": "data_query",
    "发货校验": "shipment_check",
    "发货申请": "shipment_check",   # 用户发起发货请求 → 系统查询库存和信用 → 人工确认 → 创建订单（幂等性检查）→ 扣减库存
    "知识库问答": "rag_answer",
    "制度流程咨询": "rag_answer",
    "闲聊/其他": "generate_reply"
}


NEEDS_VALIDATION = ["发货校验", "发货申请", "库存查询", "货款查询", "订单查询", "生产进度查询"]


def route_by_intent(state: ConversationState) -> str:
    """
    根据用户意图路由到对应业务节点
    
    Args:
        state: 当前对话状态
    
    Returns:
        str: 目标节点名称
    """
    intent = state.get("intent", "闲聊/其他")
    need_more_params = state.get("need_more_params", False)
    
    if need_more_params:
        logger.info(f"意图路由: {intent} -> 参数缺失，直接生成追问回复")
        return "generate_reply"
    
    if intent in NEEDS_VALIDATION:
        logger.info(f"意图路由: {intent} -> 参数校验节点")
        return "param_validation"
    
    target_node = INTENT_TO_NODE.get(intent, "generate_reply")
    logger.info(f"意图路由: {intent} -> {target_node}")
    return target_node


def route_after_param_validation(state: ConversationState) -> str:
    """
    参数校验后的路由
    
    根据校验结果决定下一步：
    - 校验通过 → 继续原有的业务流程
    - 校验失败 → 生成追问回复，让用户选择正确的参数
    """
    validation_result = state.get("param_validation", {})
    all_valid = validation_result.get("all_valid", True)
    
    if all_valid:
        intent = state.get("intent", "")
        target_node = INTENT_TO_NODE.get(intent, "generate_reply")
        logger.info(f"参数校验通过，路由到: {target_node}")
        return target_node
    else:
        logger.info(f"参数校验失败，生成追问回复")
        return "generate_reply"


def route_after_shipment_check(state: ConversationState) -> str:
    """
    发货校验后的路由
    
    根据校验结果决定下一步：
    - 校验通过 → 人工确认节点
    - 校验失败 → 直接生成回复
    - 系统错误重试 → 重新执行发货校验
    """
    overall_status = state.get("overall_status", "")
    retry_count = state.get("retry_count", 0)
    
    if overall_status == "通过":
        logger.info("发货校验通过，进入人工确认")
        return "manual_confirm"
    elif overall_status == "进行中" and retry_count > 0:
        logger.info(f"系统错误重试中，重试次数: {retry_count}，重新执行发货校验")
        return "shipment_check"
    else:
        logger.info(f"发货校验未通过({overall_status})，直接生成回复")
        return "generate_reply"


def route_after_manual_confirm(state: ConversationState) -> str:
    """
    人工确认后的路由
    
    根据人工确认结果决定下一步：
    - 确认通过且未执行过发货 → 发货执行节点
    - 确认通过但已执行过发货 → 直接生成回复（幂等性控制）
    - 确认取消 → 直接生成回复
    - 等待确认 → 继续等待（重新生成回复提醒用户确认）
    """
    manual_confirmation = state.get("manual_confirmation")
    shipment_executed = state.get("shipment_executed", False)
    
    if manual_confirmation is True:
        if shipment_executed:
            logger.info("人工确认通过，但发货已执行过，跳过发货执行")
            return "generate_reply"
        else:
            logger.info("人工确认通过，进入发货执行")
            return "shipment_execution"
    elif manual_confirmation is False:
        logger.info("人工确认取消，生成回复")
        return "generate_reply"
    else:
        logger.info("等待人工确认，重新生成回复")
        return "generate_reply"


def create_workflow():
    """
    创建ERP智能客服对话状态图工作流
    
    构建完整的LangGraph状态图，包含意图识别、数据查询、发货校验、
    人工确认、发货执行、知识库问答和回复生成等节点，支持多轮对话和状态持久化。
    
    Returns:
        CompiledGraph: 编译完成的状态图工作流对象，可直接调用invoke()
    
    状态图流转规则：
        1. 入口节点：intent_route_node（意图路由）
        2. 条件路由：根据state.intent分发
           - 订单查询/库存查询/生产进度查询/货款查询 → data_query_node
           - 发货校验/发货申请 → shipment_check_node
           - 知识库问答/制度流程咨询 → rag_answer_node
           - 闲聊/无关问题 → 直接进入generate_reply_node
        3. 发货校验流程：
           - 校验通过 → manual_confirm_node（人工确认）
           - 人工确认通过 → shipment_execution_node（发货执行）
           - 发货执行完成 → generate_reply_node
           - 校验失败/确认取消 → generate_reply_node
        4. 其他业务节点处理完成后统一进入generate_reply_node
        5. generate_reply_node执行完成后到达终点
    """
    try:
        logger.info("开始构建ERP智能客服工作流...")
        
        workflow = StateGraph(ConversationState)
        
        workflow.add_node("intent_route", intent_route_node)
        workflow.add_node("param_validation", param_validation_node)
        workflow.add_node("data_query", data_query_node)
        workflow.add_node("shipment_check", shipment_check_node)
        workflow.add_node("manual_confirm", manual_confirm_node)
        workflow.add_node("shipment_execution", shipment_execution_node)
        workflow.add_node("rag_answer", rag_answer_node)
        workflow.add_node("generate_reply", generate_reply_node)
        
        workflow.set_entry_point("intent_route")
        
        workflow.add_conditional_edges(
            "intent_route",
            route_by_intent,
            {
                "param_validation": "param_validation",
                "data_query": "data_query",
                "shipment_check": "shipment_check",
                "rag_answer": "rag_answer",
                "generate_reply": "generate_reply"
            }
        )
        
        workflow.add_conditional_edges(
            "param_validation",
            route_after_param_validation,
            {
                "data_query": "data_query",
                "shipment_check": "shipment_check",
                "generate_reply": "generate_reply"
            }
        )
        
        workflow.add_conditional_edges(
            "shipment_check",
            route_after_shipment_check,
            {
                "manual_confirm": "manual_confirm",
                "generate_reply": "generate_reply"
            }
        )
        
        workflow.add_conditional_edges(
            "manual_confirm",
            route_after_manual_confirm,
            {
                "shipment_execution": "shipment_execution",
                "generate_reply": "generate_reply"
            }
        )
        
        workflow.add_edge("data_query", "generate_reply")
        workflow.add_edge("shipment_execution", "generate_reply")
        workflow.add_edge("rag_answer", "generate_reply")
        
        workflow.add_edge("generate_reply", END)
        
        memory = MemorySaver()
        compiled_workflow = workflow.compile(checkpointer=memory)
        
        logger.info("ERP智能客服工作流构建完成")
        return compiled_workflow
    
    except Exception as e:
        logger.error(f"构建工作流异常: {str(e)}")
        raise
