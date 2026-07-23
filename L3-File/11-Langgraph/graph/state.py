from typing import TypedDict, List, Dict, Any, Annotated
from operator import add


class ChatHistoryItem(TypedDict):
    """
    历史对话记录项
    
    用于存储单轮对话的用户输入和助手回复。
    """
    user_input: str
    assistant_response: str


class ShipmentItem(TypedDict):
    """
    发货信息项
    
    用于存储单个商品的发货详情。
    """
    product_name: str
    quantity: int
    material_code: str
    warehouse: str


class ShipmentMissingItem(TypedDict):
    """
    待确认发货项
    
    用于存储需要用户确认的发货相关信息。
    """
    item_type: str
    description: str
    status: str


class RiskNote(TypedDict):
    """
    风险提示项
    
    用于存储发货校验过程中发现的风险点。
    """
    level: str
    message: str
    category: str


class ConversationState(TypedDict, total=False):
    """
    ERP智能客服全局对话状态
    
    定义完整的对话生命周期状态字段，覆盖从会话开始到结束的全流程。
    设置 total=False 允许各节点只返回部分状态更新。
    """

    # ========== 基础会话字段 ==========
    session_id: str
    """会话唯一标识，用于追踪整个对话过程"""
    
    user_input: str
    """当前轮次用户输入的问题或指令"""
    
    chat_history: Annotated[List[ChatHistoryItem], add]
    """历史对话记录列表，存储之前的用户-助手对话对"""

    # ========== 路由控制字段 ==========
    intent: str
    """识别出的用户意图，如：订单查询、库存查询、发货校验、货款查询等"""
    
    target_agent: str
    """目标业务节点，如：order_agent、stock_agent、shipment_check_agent等"""
    
    business_params: Dict[str, Any]
    """已收集的业务参数字典，如订单号、客户名、物料编码等"""
    
    need_more_params: bool
    """是否需要更多参数，当关键参数缺失时为True"""
    
    missing_params: List[str]
    """需要追问的参数列表，如["order_no", "customer_name"]"""

    # ========== 发货校验专属字段 ==========
    shipment_confirmed_items: List[ShipmentItem]
    """已确认的发货信息列表，包含商品名称、数量、物料编码、仓库等"""
    
    shipment_missing_items: List[ShipmentMissingItem]
    """待确认项列表，记录需要用户补充或确认的信息"""
    
    payment_check_result: Dict[str, Any]
    """货款校验结果，包含客户信用状态、可用余额等"""
    
    stock_check_result: Dict[str, Any]
    """库存校验结果，包含可用库存、是否低于安全库存等"""
    
    risk_notes: List[RiskNote]
    """风险提示列表，记录发货过程中发现的风险点"""

    # ========== 工具执行字段 ==========
    dao_query_result: Dict[str, Any]
    """DAO数据查询结果，存储从MySQL数据库获取的业务数据"""
    
    rag_retrieve_result: List[Dict[str, Any]]
    """RAG检索文档结果，存储从Milvus向量库召回的相关文档片段"""

    # ========== 发货执行字段 ==========
    shipment_order_no: str
    """自动生成的订单号"""
    
    logistics_no: str
    """自动生成的物流编码"""
    
    manual_confirmation: bool
    """人工确认状态，True表示已确认"""
    
    shipment_executed: bool
    """发货是否已执行"""
    
    # ========== 参数校验字段 ==========
    param_validation: Dict[str, Any]
    """参数校验结果，包含是否全部有效、各参数校验结果、建议选项等"""
    
    # ========== 流程输出字段 ==========
    final_response: str
    """最终生成的回复，由LLM根据查询结果生成的自然语言回答"""
    
    current_step: str
    """当前流程节点，标识对话处于哪个处理阶段"""
    
    overall_status: str
    """整体流程状态，取值：进行中、通过、失败"""


def init_state(session_id: str) -> ConversationState:
    """
    初始化对话状态
    
    创建一个新会话时，返回包含默认空值的状态字典。
    
    Args:
        session_id: 会话唯一标识
    
    Returns:
        ConversationState: 初始化后的对话状态字典
    """
    return {
        "session_id": session_id,
        "user_input": "",
        "chat_history": [],
        "intent": "",
        "target_agent": "",
        "business_params": {},
        "need_more_params": False,
        "missing_params": [],
        "shipment_confirmed_items": [],
        "shipment_missing_items": [],
        "payment_check_result": {},
        "stock_check_result": {},
        "risk_notes": [],
        "dao_query_result": {},
        "rag_retrieve_result": [],
        "shipment_order_no": "",
        "logistics_no": "",
        "manual_confirmation": False,
        "shipment_executed": False,
        "final_response": "",
        "current_step": "start",
        "overall_status": "进行中"
    }
