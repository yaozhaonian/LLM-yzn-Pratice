import json
import re
from datetime import datetime
from graph.state import ConversationState
from services.llm_client import ollama_client
from services.rag_service import rag_service
from dao.order_dao import get_order_by_order_no, get_orders_by_customer_and_time
from dao.stock_dao import get_stock_by_material_code, get_stocks_by_warehouse, get_material_code_by_product_name
from dao.production_dao import get_work_order_progress, get_in_progress_work_orders
from dao.payment_dao import get_customer_payment_by_name, update_customer_payment
from dao.order_service import create_order, update_order_status
from dao.db import get_db
from dao.models import Order, Stock
from agents.prompts import (
    INTENT_ROUTE_SYSTEM_PROMPT,     # 专业的ERP智能客服意图识别助手
    INTENT_ROUTE_USER_PROMPT,       # 对应的提交到大模型的用户提示词
    DATA_QUERY_SYSTEM_PROMPT,       # 专业的ERP数据查询助手
    DATA_QUERY_USER_PROMPT,
    SHIPMENT_CHECK_SYSTEM_PROMPT,   # 专业的ERP发货校验助手
    SHIPMENT_CHECK_USER_PROMPT,
    RAG_ANSWER_SYSTEM_PROMPT,       # 专业的ERP知识库问答助手
    RAG_ANSWER_USER_PROMPT,
    GENERAL_REPLY_SYSTEM_PROMPT,    # 专业的ERP智能客服助手
    GENERAL_REPLY_USER_PROMPT
)
from utils.logger import get_logger

logger = get_logger(__name__)


def sanitize_input(user_input: str) -> str:
    """
    输入安全过滤函数，防止提示词注入攻击
    
    对用户输入进行清洗，移除可能的注入攻击字符和模式。
    
    Args:
        user_input: 用户原始输入
        
    Returns:
        str: 清洗后的安全输入
    """
    if not user_input:
        return ""
    
    cleaned = user_input.strip()
    
    injection_patterns = [
        r"(?i)ignore\s+previous\s+instructions?",
        r"(?i)forget\s+previous\s+prompt",
        r"(?i)override\s+instructions?",
        r"(?i)bypass\s+rules?",
        r"(?i)disregard\s+instructions?",
        r"(?i)system\s+prompt",
        r"(?i)change\s+role",
        r"(?i)assume\s+role",
        r"(?i)new\s+role",
        r"(?i)act\s+as\s+a?",
        r"(?i)you\s+are\s+now",
        r"(?i)pretend\s+to\s+be",
        r"(?i)imagine\s+you\s+are",
        r"(?i)execute\s+command",
        r"(?i)run\s+code",
        r"(?i)python\s*:",
        r"(?i)javascript\s*:",
        r"(?i)shell\s*:",
        r"(?i)terminal\s*:",
        r"(?i)eval\s*\(",
        r"(?i)exec\s*\(",
        r"(?i)open\s*\(",
        r"(?i)read\s*\(",
        r"(?i)write\s*\(",
        r"(?i)delete\s*\(",
        r"(?i)rm\s+-rf",
        r"(?i)curl\s+",
        r"(?i)wget\s+",
        r"(?i)\|.*bash",
        r"(?i)\|.*sh",
        r"(?i);.*chmod",
        r"(?i)<script[^>]*>",
        r"(?i)</script>",
        r"(?i)javascript:",
        r"(?i)onerror=",
        r"(?i)onload=",
        r"(?i)data:image",
        r"(?i)data:text",
        r"(?i)alert\s*\(",
        r"(?i)prompt\s*\(",
        r"(?i)confirm\s*\(",
        r"(?i)document\.cookie",
        r"(?i)localStorage",
        r"(?i)sessionStorage",
        r"(?i)<iframe",
        r"(?i)<svg.*onload",
        r"(?i)<img.*onerror",
        r"(?i)%3c",
        r"(?i)%3e",
        r"(?i)%22",
        r"(?i)%27",
        r"(?i)--\s*[^\n]*$",
        r"(?i)/\*.*\*/",
        r"(?i)union\s+select",
        r"(?i)select.*from",
        r"(?i)insert\s+into",
        r"(?i)update.*set",
        r"(?i)delete\s+from",
        r"(?i)drop\s+table",
        r"(?i)truncate\s+table",
        r"(?i)exec\s+sp_",
        r"(?i)xp_cmdshell",
        r"(?i)1=1",
        r"(?i)or\s+1=1",
        r"(?i)';.*--",
        r"(?i)\";.*--",
        r"(?i)\(\);.*--",
        r"(?i)\{\{.*\}\}",
        r"(?i)\[\[.*\]\]",
        r"(?i)<<.*>>",
        r"(?i)<%.*%>",
        r"(?i)\$\{.*\}",
        r"(?i)`.*`",
        r"(?i)\$\(.*\)",
    ]
    
    for pattern in injection_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    if len(cleaned) > 2000:
        cleaned = cleaned[:2000] + "..."
    
    if cleaned != user_input:
        logger.warning(f"输入已被安全过滤，原始长度: {len(user_input)}, 过滤后长度: {len(cleaned)}")
    
    return cleaned


def intent_route_node(state: ConversationState) -> ConversationState:
    """
    意图路由节点
    
    调用LLM分析用户输入和对话历史，识别用户意图、目标业务场景，并提取关键业务参数。
    
    Args:
        state: 当前对话状态
    
    Returns:
        ConversationState: 更新了intent、target_agent、business_params、current_step、need_more_params、missing_params的状态
    
    支持的意图分类：
        - 订单查询
        - 库存查询
        - 生产进度查询
        - 发货校验
        - 知识库问答
        - 闲聊/其他
    """
    try:
        user_input = state.get("user_input", "")
        chat_history = state.get("chat_history", [])
        
        history_text = "\n".join(
            [f"用户: {h['user_input']}\n助手: {h['assistant_response']}" for h in chat_history]
        )
        
        import re
        
        reconfirm_keywords = ["重新确认", "确认余额", "已充值", "充值完毕", "重新查询", "再次确认", "确认发货"]
        if any(keyword in user_input for keyword in reconfirm_keywords):
            logger.info(f"检测到重新确认请求: {user_input}")
            previous_params = {}
            if chat_history:
                for h in reversed(chat_history):
                    history_content = h.get("user_input", "") + " " + h.get("assistant_response", "")
                    
                    customer_match = re.search(r'([\u4e00-\u9fff]{2,10}[科技智能制造红木机械电子商贸]?有限公司|[\u4e00-\u9fff]{2,10}集团|[\u4e00-\u9fff]{2,10}公司)', history_content)
                    if customer_match and not previous_params.get("customer_name"):
                        name = customer_match.group(1)
                        if name not in ["相关物流公司", "当前公司", "您的公司"]:
                            for prefix in ["给", "向", "为", "对"]:
                                if name.startswith(prefix):
                                    name = name[1:]
                                    break
                            previous_params["customer_name"] = name
                    
                    quantity_match = re.search(r'(发货数量|发货|订.*数量|要.*个|要.*件|要.*台|给.*发.*个|给.*发.*件|给.*发.*台)\s*[：:]?\s*(\d+)', history_content)
                    if not quantity_match:
                        quantity_match = re.search(r'(?<!库存)(?<!安全)(数量|个|件|台)\s*[：:]?\s*(\d+)', history_content)
                    if quantity_match and not previous_params.get("quantity"):
                        previous_params["quantity"] = int(quantity_match.group(2).strip())
                    
                    material_match = re.search(r'(MAT\d+)', history_content)
                    if material_match and not previous_params.get("material_code"):
                        previous_params["material_code"] = material_match.group(1).strip()
                    
                    if not previous_params.get("material_code"):
                        product_match = re.search(r'(产品名称|产品|名称)[：:]?\s*([\u4e00-\u9fff]+)', history_content)
                        if not product_match:
                            product_match = re.search(r'[\u4e00-\u9fff]{2,20}(主板|配件|零件|设备|系统|模块|板卡)', history_content)
                        if not product_match:
                            product_match = re.search(r'[\u4e00-\u9fff]{2,20}主板', history_content)
                        if product_match and not previous_params.get("product_name"):
                            product_name = product_match.group(2).strip() if len(product_match.groups()) > 1 else product_match.group(0).strip()
                            for prefix in ["个", "件", "台", "套"]:
                                if product_name.startswith(prefix):
                                    product_name = product_name[1:]
                                    break
                            previous_params["product_name"] = product_name
                    
                    payment_match = re.search(r'(付款方式|支付方式)[：:]?\s*(先付款后发货|收到货再付款)', history_content)
                    if payment_match and not previous_params.get("payment_method"):
                        previous_params["payment_method"] = payment_match.group(2).strip()
                    
                    address_match = re.search(r'(收货地址|地址)[：:]?\s*([\u4e00-\u9fff\d\s\-]+?)(?=\s*(收货人|联系人|联系电话|电话|手机号|手机|，|。|$))', history_content)
                    if not address_match:
                        address_match = re.search(r'(收货地址|地址)[：:]?\s*([\u4e00-\u9fff\d\s\-]{10,50})', history_content)
                    if address_match and not previous_params.get("shipping_address"):
                        previous_params["shipping_address"] = address_match.group(2).strip()
                    
                    receiver_match = re.search(r'(收货人|负责人|联系人)[：:]?\s*([\u4e00-\u9fff]{2,10})', history_content)
                    if receiver_match and not previous_params.get("receiver"):
                        previous_params["receiver"] = receiver_match.group(2).strip()
                    
                    phone_match = re.search(r'(联系电话|电话|手机号|手机)[：:]?\s*(\d{7,15})', history_content)
                    if phone_match and not previous_params.get("contact_phone"):
                        previous_params["contact_phone"] = phone_match.group(2).strip()
            
            quantity_update_match = re.search(r'(改为|改成|变更为|调整为|改)\s*(\d+)\s*(个|件|台|套)', user_input)
            if quantity_update_match:
                previous_params["quantity"] = int(quantity_update_match.group(2).strip())
                logger.info(f"从当前用户输入中更新数量: {previous_params['quantity']}")
            
            material_update_match = re.search(r'(MAT\d+)', user_input)
            if material_update_match:
                previous_params["material_code"] = material_update_match.group(1).strip()
                logger.info(f"从当前用户输入中更新物料编码: {previous_params['material_code']}")
            
            product_update_match = re.search(r'(产品名称|产品)\s*[：:]?\s*([\u4e00-\u9fff]+)', user_input)
            if product_update_match:
                previous_params["product_name"] = product_update_match.group(2).strip()
                logger.info(f"从当前用户输入中更新产品名称: {previous_params['product_name']}")
            
            logger.info(f"重新确认发货请求，提取的参数: {previous_params}")
            
            if previous_params.get("customer_name"):
                logger.info(f"重新确认发货请求，客户: {previous_params['customer_name']}, 数量: {previous_params.get('quantity', 1)}")
                return {
                    "intent": "发货校验",
                    "target_agent": "shipment_check_agent",
                    "business_params": previous_params,
                    "need_more_params": False,
                    "missing_params": [],
                    "current_step": "intent_route"
                }
        
        if len(user_input) <= 20 and user_input.isalnum() and not user_input.isdigit() and not any('\u4e00' <= c <= '\u9fff' for c in user_input):
            logger.info(f"检测到纯编码输入: {user_input}，尝试根据编码格式识别意图")
            # 假设订单号开头为ORD
            if user_input.startswith("ORD"):
                logger.info(f"识别为订单号: {user_input}")
                return {
                    "intent": "订单查询",
                    "target_agent": "order_agent",
                    "business_params": {"order_no": user_input},
                    "need_more_params": False,
                    "missing_params": [],
                    "current_step": "intent_route"
                }
            # 假设物料编码开头为MAT
            elif user_input.startswith("MAT"):
                logger.info(f"识别为物料编码: {user_input}")
                return {
                    "intent": "库存查询",
                    "target_agent": "stock_agent",
                    "business_params": {"material_code": user_input},
                    "need_more_params": False,
                    "missing_params": [],
                    "current_step": "intent_route"
                }
            # 假设工单号开头为WO
            elif user_input.startswith("WO"):
                logger.info(f"识别为工单号: {user_input}")
                return {
                    "intent": "生产进度查询",
                    "target_agent": "production_agent",
                    "business_params": {"work_order_no": user_input},
                    "need_more_params": False,
                    "missing_params": [],
                    "current_step": "intent_route"
                }
        
        import re
        
        user_input_cleaned = sanitize_input(user_input)
        
        order_no_match = re.search(r'(ORD\d+)', user_input_cleaned)
        material_code_match = re.search(r'(MAT\d+)', user_input_cleaned)
        work_order_no_match = re.search(r'(WO\d+)', user_input_cleaned)
        
        semantic_order_keywords = ["订单", "order"]
        semantic_material_keywords = ["物料", "库存"]
        semantic_workorder_keywords = ["工单", "生产"]
        
        has_order_semantic = any(keyword in user_input_cleaned for keyword in semantic_order_keywords)
        has_material_semantic = any(keyword in user_input_cleaned for keyword in semantic_material_keywords)
        has_workorder_semantic = any(keyword in user_input_cleaned for keyword in semantic_workorder_keywords)
        
        if order_no_match:
            order_no = order_no_match.group(1)
            if has_material_semantic and not has_order_semantic:
                logger.info(f"语义冲突：用户提到物料但提供了订单号: {order_no}")
                return {
                    "intent": "订单查询",
                    "target_agent": "order_agent",
                    "business_params": {"order_no": order_no},
                    "need_more_params": True,
                    "missing_params": ["confirm_intent"],
                    "current_step": "intent_route",
                    "risk_notes": [{
                        "level": "warning",
                        "message": f"检测到语义冲突：您提到'物料编码'，但提供的编码'{order_no}'是订单号格式（ORD开头）。您是想查询订单状态吗？",
                        "category": "语义冲突"
                    }]
                }
            logger.info(f"从中文输入中识别到订单号: {order_no}")
            return {
                "intent": "订单查询",
                "target_agent": "order_agent",
                "business_params": {"order_no": order_no},
                "need_more_params": False,
                "missing_params": [],
                "current_step": "intent_route"
            }
        elif material_code_match:
            material_code = material_code_match.group(1)
            if has_order_semantic and not has_material_semantic:
                logger.info(f"语义冲突：用户提到订单但提供了物料编码: {material_code}")
                return {
                    "intent": "库存查询",
                    "target_agent": "stock_agent",
                    "business_params": {"material_code": material_code},
                    "need_more_params": True,
                    "missing_params": ["confirm_intent"],
                    "current_step": "intent_route",
                    "risk_notes": [{
                        "level": "warning",
                        "message": f"检测到语义冲突：您提到'订单'，但提供的编码'{material_code}'是物料编码格式（MAT开头）。您是想查询库存吗？",
                        "category": "语义冲突"
                    }]
                }
            logger.info(f"从中文输入中识别到物料编码: {material_code}")
            return {
                "intent": "库存查询",
                "target_agent": "stock_agent",
                "business_params": {"material_code": material_code},
                "need_more_params": False,
                "missing_params": [],
                "current_step": "intent_route"
            }
        elif work_order_no_match:
            work_order_no = work_order_no_match.group(1)
            logger.info(f"从中文输入中识别到工单号: {work_order_no}")
            return {
                "intent": "生产进度查询",
                "target_agent": "production_agent",
                "business_params": {"work_order_no": work_order_no},
                "need_more_params": False,
                "missing_params": [],
                "current_step": "intent_route"
            }
        
        cleaned_input = sanitize_input(user_input)
        
        previous_params = {}
        if chat_history:
            for h in reversed(chat_history):
                history_content = h.get("user_input", "") + " " + h.get("assistant_response", "")
                
                customer_match = re.search(r'([\u4e00-\u9fff]{2,10}[科技智能制造红木机械电子商贸]?有限公司|[\u4e00-\u9fff]{2,10}集团|[\u4e00-\u9fff]{2,10}公司)', history_content)
                if customer_match and not previous_params.get("customer_name"):
                    name = customer_match.group(1)
                    if name not in ["相关物流公司", "当前公司", "您的公司"]:
                        for prefix in ["给", "向", "为", "对"]:
                            if name.startswith(prefix):
                                name = name[1:]
                                break
                        previous_params["customer_name"] = name
                        logger.info(f"从对话历史中提取到客户名称: {name}")
                
                address_match = re.search(r'(收货地址|地址)[：:]?\s*([\u4e00-\u9fff\d\s\-]+)', history_content)
                if address_match and not previous_params.get("shipping_address"):
                    previous_params["shipping_address"] = address_match.group(2).strip()
                    logger.info(f"从对话历史中提取到收货地址: {previous_params['shipping_address']}")
                
                receiver_match = re.search(r'(收货人|负责人|联系人)[：:]?\s*([\u4e00-\u9fff]{2,10})', history_content)
                if receiver_match and not previous_params.get("receiver"):
                    previous_params["receiver"] = receiver_match.group(2).strip()
                    logger.info(f"从对话历史中提取到收货人: {previous_params['receiver']}")
                
                phone_match = re.search(r'(联系电话|电话|手机号|手机)[：:]?\s*(\d{7,15})', history_content)
                if phone_match and not previous_params.get("contact_phone"):
                    previous_params["contact_phone"] = phone_match.group(2).strip()
                    logger.info(f"从对话历史中提取到联系电话: {previous_params['contact_phone']}")
                
                quantity_match = re.search(r'(发货数量|发货|订.*数量|要.*个|要.*件|要.*台|给.*发.*个|给.*发.*件|给.*发.*台)\s*[：:]?\s*(\d+)', history_content)
                if not quantity_match:
                    quantity_match = re.search(r'(数量|个|件|台)\s*[：:]?\s*(\d+)', history_content)
                if quantity_match and not previous_params.get("quantity"):
                    previous_params["quantity"] = int(quantity_match.group(2).strip())
                    logger.info(f"从对话历史中提取到数量: {previous_params['quantity']}")
                
                payment_match = re.search(r'(付款方式|支付方式)[：:]?\s*(先付款后发货|收到货再付款)', history_content)
                if payment_match and not previous_params.get("payment_method"):
                    previous_params["payment_method"] = payment_match.group(2).strip()
                    logger.info(f"从对话历史中提取到付款方式: {previous_params['payment_method']}")
                
                credit_confirm_match = re.search(r'(信用额度.*确认|确认.*信用额度|使用信用额度)', history_content)
                if credit_confirm_match and not previous_params.get("confirm_credit"):
                    previous_params["confirm_credit"] = True
                    logger.info(f"从对话历史中提取到信用额度确认")
                
                material_match = re.search(r'(物料编码|物料)[：:]?\s*(MAT\d+)', history_content)
                if material_match and not previous_params.get("material_code"):
                    previous_params["material_code"] = material_match.group(2).strip()
                    logger.info(f"从对话历史中提取到物料编码: {previous_params['material_code']}")
                
                product_match = re.search(r'(产品名称|产品|名称)[：:]?\s*([\u4e00-\u9fff]+)', history_content)
                if product_match and not previous_params.get("product_name"):
                    previous_params["product_name"] = product_match.group(2).strip()
                    logger.info(f"从对话历史中提取到产品名称: {previous_params['product_name']}")
        
        system_prompt = INTENT_ROUTE_SYSTEM_PROMPT
        user_prompt = INTENT_ROUTE_USER_PROMPT.format(
            history=history_text,
            user_input=cleaned_input
        )
        
        if previous_params:
            previous_params_str = "\n".join([f"- {key}: {value}" for key, value in previous_params.items()])
            user_prompt += f"\n\n注意：对话历史中已提到以下业务参数，请参考并补全：\n{previous_params_str}\n如果用户使用'当前用户'、'我'、'这个客户'等指代，请将其解析为对应的客户名称。"
        
        logger.info(f"开始意图识别，用户输入: {user_input[:50]}...")
        response = ollama_client.chat(user_prompt, system_prompt)
        
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            result = json.loads(cleaned_response)
            intent = result.get("intent", "闲聊/其他")
            target_agent = result.get("target_agent", "general_agent")
            business_params = result.get("business_params", {})
            need_more_params = result.get("need_more_params", False)
            missing_params = result.get("missing_params", [])
        except json.JSONDecodeError:
            logger.warning(f"意图识别返回非JSON格式: {response[:100]}")
            intent = "闲聊/其他"
            target_agent = "general_agent"
            business_params = {}
            need_more_params = False
            missing_params = []
        
        logger.info(f"意图识别完成，意图: {intent}, 目标节点: {target_agent}, "
                    f"需要更多参数: {need_more_params}, 缺失参数: {missing_params}")
        
        return {
            "intent": intent,
            "target_agent": target_agent,
            "business_params": business_params,
            "need_more_params": need_more_params,
            "missing_params": missing_params,
            "current_step": "intent_route"
        }
    
    except Exception as e:
        logger.error(f"意图路由节点异常: {str(e)}")
        return {
            "intent": "闲聊/其他",
            "target_agent": "general_agent",
            "business_params": {},
            "need_more_params": False,
            "missing_params": [],
            "current_step": "intent_route",
            "risk_notes": [{
                "level": "error",
                "message": f"意图识别失败: {str(e)}",
                "category": "系统错误"
            }]
        }


def data_query_node(state: ConversationState) -> ConversationState:
    """
    业务数据查询节点
    
    根据用户意图和业务参数，调用对应DAO层方法执行数据查询。
    
    Args:
        state: 当前对话状态
    
    Returns:
        ConversationState: 更新了dao_query_result的状态
    
    分支逻辑：
        - 订单查询 → 调用order_dao
        - 库存查询 → 调用stock_dao
        - 生产进度查询 → 调用production_dao
        - 货款查询 → 调用payment_dao
    """
    try:
        intent = state.get("intent", "")
        business_params = state.get("business_params", {})
        result = {}
        
        logger.info(f"开始数据查询，意图: {intent}, 参数: {business_params}")
        
        if intent == "订单查询":
            order_no = business_params.get("order_no")
            customer_name = business_params.get("customer_name")
            
            if order_no:
                result = get_order_by_order_no(order_no)
            elif customer_name:
                start_time = business_params.get("start_time")
                end_time = business_params.get("end_time")
                result = {"orders": get_orders_by_customer_and_time(
                    customer_name,
                    start_time and datetime.fromisoformat(start_time),
                    end_time and datetime.fromisoformat(end_time)
                )}
        
        elif intent == "库存查询":
            material_code = business_params.get("material_code")
            warehouse = business_params.get("warehouse")
            
            if material_code:
                result = get_stock_by_material_code(material_code)
            elif warehouse:
                result = {"stocks": get_stocks_by_warehouse(warehouse)}
        
        elif intent == "生产进度查询":
            work_order_no = business_params.get("work_order_no")
            product_name = business_params.get("product_name")
            
            if work_order_no:
                result = get_work_order_progress(work_order_no)
            elif product_name:
                result = {"work_orders": get_in_progress_work_orders(product_name)}
        
        elif intent == "货款查询":
            customer_name = business_params.get("customer_name")
            if customer_name:
                result = get_customer_payment_by_name(customer_name)
        
        logger.info(f"数据查询完成，结果: {result}")
        
        return {
            "dao_query_result": result,
            "current_step": "data_query"
        }
    
    except Exception as e:
        logger.error(f"数据查询节点异常: {str(e)}")
        return {
            "dao_query_result": {},
            "current_step": "data_query",
            "risk_notes": [{
                "level": "error",
                "message": f"数据查询失败: {str(e)}",
                "category": "数据库错误"
            }]
        }


def shipment_check_node(state: ConversationState) -> ConversationState:
    """
    发货校验节点
    
    执行发货前三项必核校验逻辑：订单信息、货款充足性、库存充足性。
    支持用户只输入商品和客户，系统自动查询库存和客户信用状态。
    
    Args:
        state: 当前对话状态
    
    Returns:
        ConversationState: 更新了发货校验相关所有字段的状态
    
    校验流程：
        1. 客户信息查询（根据客户名称查询信用状态）
        2. 库存充足性校验（根据物料编码查询库存）
        3. 订单信息校验（如果有订单号）
    
    规则：
        - 信息缺失时更新missing_items引导用户补充
        - 校验不通过时标记风险与阻断原因
        - 全部通过时标记状态为通过
        - 不需要用户输入订单号和物流编码，系统自动生成
    """
    try:
        business_params = state.get("business_params", {})
        user_input = state.get("user_input", "")
        order_no = business_params.get("order_no")
        customer_name = business_params.get("customer_name")
        material_code = business_params.get("material_code")
        product_name = business_params.get("product_name")
        quantity = business_params.get("quantity", 1)
        payment_method = business_params.get("payment_method", "")
        shipping_address = business_params.get("shipping_address")
        receiver = business_params.get("receiver")
        contact_phone = business_params.get("contact_phone")
        
        try:
            quantity = int(quantity)
        except (ValueError, TypeError):
            logger.warning(f"数量转换失败，quantity: {quantity}")
            quantity = 1
        
        logger.info(f"开始发货校验，客户名: {customer_name}, 物料编码: {material_code}, "
                    f"产品名称: {product_name}, 数量: {quantity}, 付款方式: {payment_method}")
        
        confirmed_items = []
        missing_items = []
        payment_result = {}
        stock_result = {}
        risk_notes = []
        order_result = {}
        overall_status = "进行中"
        
        if not customer_name:
            missing_items.append({
                "item_type": "customer_name",
                "description": "请提供客户名称",
                "status": "待确认"
            })
        
        if not payment_method:
            missing_items.append({
                "item_type": "payment_method",
                "description": "请选择付款方式：先付款后发货、收到货再付款",
                "status": "待确认"
            })
        
        if not shipping_address:
            missing_items.append({
                "item_type": "shipping_address",
                "description": "请提供收货地址",
                "status": "待确认"
            })
        
        if not receiver:
            missing_items.append({
                "item_type": "receiver",
                "description": "请提供收货人（负责人）",
                "status": "待确认"
            })
        
        if not contact_phone:
            missing_items.append({
                "item_type": "contact_phone",
                "description": "请提供联系电话",
                "status": "待确认"
            })
        
        if material_code and not material_code.startswith("MAT"):
            if re.search(r'[\u4e00-\u9fff]', material_code):
                logger.info(f"物料编码'{material_code}'不是MAT开头且包含中文，尝试作为产品名称查询")
                product_name = material_code
                material_code = get_material_code_by_product_name(product_name)
                if material_code:
                    logger.info(f"查询成功，物料编码: {material_code}")
                else:
                    missing_items.append({
                        "item_type": "material_code",
                        "description": f"未找到产品'{product_name}'对应的物料编码，请提供物料编码",
                        "status": "待确认"
                    })
            else:
                missing_items.append({
                    "item_type": "material_code",
                    "description": f"物料编码格式不正确，应为MAT开头（如MAT001），当前: {material_code}",
                    "status": "待确认"
                })
        
        if product_name and not material_code:
            logger.info(f"通过产品名称查询物料编码，产品名称: {product_name}")
            material_code = get_material_code_by_product_name(product_name)
            if material_code:
                logger.info(f"查询成功，物料编码: {material_code}")
            else:
                missing_items.append({
                    "item_type": "material_code",
                    "description": f"未找到产品'{product_name}'对应的物料编码，请提供物料编码",
                    "status": "待确认"
                })
        
        if not material_code and not product_name:
            missing_items.append({
                "item_type": "material_code",
                "description": "请提供物料编码或产品名称",
                "status": "待确认"
            })
        
        if material_code:
            stock_info = get_stock_by_material_code(material_code)
            logger.info(f"库存查询结果，物料编码: {material_code}, 查询结果: {stock_info}")
            if stock_info:
                stock_result = stock_info
                available_stock = int(stock_info.get("available_stock", 0))
                logger.info(f"库存充足性校验，可用库存: {available_stock}, 发货数量: {quantity}, 安全库存: {stock_info.get('safety_stock', 0)}")
                
                if available_stock < quantity:
                    risk_notes.append({
                        "level": "error",
                        "message": f"库存不足，可用库存: {available_stock}, 需求: {quantity}",
                        "category": "库存校验"
                    })
                elif stock_info.get("is_below_safety", False):
                    safety_stock = int(stock_info.get("safety_stock", 0))
                    risk_notes.append({
                        "level": "warning",
                        "message": f"库存低于安全库存，可用库存: {available_stock}, "
                                    f"安全库存: {safety_stock}",
                        "category": "库存校验"
                    })
            else:
                risk_notes.append({
                    "level": "warning",
                    "message": f"未找到物料库存信息: {material_code}",
                    "category": "库存校验"
                })
        
        if customer_name:
            payment_info = get_customer_payment_by_name(customer_name)
            logger.info(f"货款查询结果，客户名称: {customer_name}, 查询结果: {payment_info}")
            if payment_info:
                payment_result = {**payment_result, **payment_info}
                available_balance = float(payment_info.get("available_balance", 0.0))
                credit_limit = float(payment_info.get("credit_limit", 0.0))
                outstanding_amount = float(payment_info.get("outstanding_amount", 0.0))
                
                unit_price = float(stock_result.get("unit_price", 0.0)) if stock_result else 0.0
                total_amount = unit_price * quantity
                logger.info(f"货款充足性校验，可用余额: {available_balance}, 信用额度: {credit_limit}, "
                            f"未回款金额: {outstanding_amount}, 订单金额: {total_amount}, 单价: {unit_price}")
                
                if payment_method == "先付款后发货":
                    if available_balance >= total_amount:
                        payment_result["payment_status"] = "余额充足，可直接扣款"
                        payment_result["total_amount"] = total_amount
                        payment_result["payment_action"] = "deduct_balance"
                    else:
                        if available_balance + credit_limit >= total_amount:
                            confirm_credit = business_params.get("confirm_credit", False)
                            user_input_lower = user_input.lower()
                            has_credit_confirm = confirm_credit or "信用额度" in user_input_lower and ("确认" in user_input_lower or "是" in user_input_lower or "扣款" in user_input_lower)
                            
                            if has_credit_confirm:
                                payment_result["payment_status"] = "使用信用额度扣款"
                                payment_result["total_amount"] = total_amount
                                payment_result["payment_action"] = "use_credit"
                            else:
                                payment_result["payment_status"] = "余额不足，可使用信用额度"
                                payment_result["total_amount"] = total_amount
                                payment_result["payment_action"] = "use_credit"
                                missing_items.append({
                                    "item_type": "confirm_credit",
                                    "description": f"您的可用余额({available_balance})不足支付订单金额({total_amount})，但加上信用额度({credit_limit})足够。是否使用信用额度进行扣款？",
                                    "status": "待确认"
                                })
                        else:
                            payment_result["payment_status"] = "余额和信用额度均不足"
                            payment_result["total_amount"] = total_amount
                            payment_result["payment_action"] = "insufficient_funds"
                            risk_notes.append({
                                "level": "error",
                                "message": f"客户可用余额({available_balance})和信用额度({credit_limit})均不足支付订单金额({total_amount})，请充值或取消订单",
                                "category": "货款校验"
                            })
                
                elif payment_method == "收到货再付款":
                    if credit_limit >= total_amount:
                        payment_result["payment_status"] = "信用额度充足，可延迟付款"
                        payment_result["total_amount"] = total_amount
                        payment_result["payment_action"] = "add_outstanding"
                    else:
                        payment_result["payment_status"] = "信用额度不足"
                        payment_result["total_amount"] = total_amount
                        payment_result["payment_action"] = "insufficient_credit"
                        risk_notes.append({
                            "level": "error",
                            "message": f"客户信用额度({credit_limit})不足支付订单金额({total_amount})，请充值或取消订单",
                            "category": "货款校验"
                        })
                else:
                    if available_balance <= 0:
                        risk_notes.append({
                            "level": "warning",
                            "message": f"客户可用余额不足: {available_balance}",
                            "category": "货款校验"
                        })
            else:
                risk_notes.append({
                    "level": "warning",
                    "message": f"未找到客户货款信息: {customer_name}",
                    "category": "货款校验"
                })
        
        if order_no:
            order_result = get_order_by_order_no(order_no)
            if order_result:
                confirmed_items.append({
                    "product_name": order_result.get("customer_name", ""),
                    "quantity": quantity,
                    "material_code": order_result.get("order_no", ""),
                    "warehouse": ""
                })
                payment_result["order_info"] = order_result
                
                if order_result.get("status") == "已发货":
                    risk_notes.append({
                        "level": "error",
                        "message": f"订单 {order_no} 已发货，不可重复发货",
                        "category": "订单校验"
                    })
            else:
                risk_notes.append({
                    "level": "error",
                    "message": f"未找到订单: {order_no}",
                    "category": "订单校验"
                })
        
        if stock_result and payment_result:
            confirmed_items.append({
                "product_name": stock_result.get("product_name", product_name or ""),
                "quantity": quantity,
                "material_code": material_code or "",
                "warehouse": stock_result.get("warehouse", ""),
                "available_stock": stock_result.get("available_stock", 0),
                "safety_stock": stock_result.get("safety_stock", 0),
                "unit_price": stock_result.get("unit_price", 0)
            })
        
        if not missing_items and not risk_notes:
            overall_status = "通过"
            logger.info("发货校验全部通过，等待人工确认")
        elif risk_notes:
            overall_status = "失败"
            logger.info(f"发货校验失败，风险项: {len(risk_notes)}")
        
        return {
            "shipment_confirmed_items": confirmed_items,
            "shipment_missing_items": missing_items,
            "payment_check_result": payment_result,
            "stock_check_result": stock_result,
            "risk_notes": risk_notes if risk_notes else None,
            "overall_status": overall_status,
            "current_step": "shipment_check",
            "shipment_executed": False,
            "manual_confirmation": None,
            "shipment_order_no": "",
            "logistics_no": "",
            "business_params": {
                "customer_name": customer_name,
                "material_code": material_code,
                "product_name": product_name,
                "quantity": quantity,
                "shipping_address": shipping_address,
                "receiver": receiver,
                "contact_phone": contact_phone,
                "payment_method": payment_method
            }
        }
    
    except Exception as e:
        logger.error(f"发货校验节点异常: {str(e)}")
        
        retry_count = state.get("retry_count", 0)
        max_retries = 2
        
        if retry_count < max_retries:
            logger.info(f"系统错误，尝试重试，当前重试次数: {retry_count + 1}")
            return {
                "risk_notes": [{
                    "level": "warning",
                    "message": f"系统暂时出现异常，正在尝试重试... ({retry_count + 1}/{max_retries})",
                    "category": "系统错误"
                }],
                "overall_status": "进行中",
                "current_step": "shipment_check",
                "retry_count": retry_count + 1
            }
        else:
            logger.error(f"系统错误，已达到最大重试次数: {max_retries}")
            return {
                "risk_notes": [{
                    "level": "error",
                    "message": f"发货校验失败: {str(e)}，系统已尝试{max_retries}次重试仍未成功。请稍后重试或联系技术支持。",
                    "category": "系统错误"
                }],
                "overall_status": "失败",
                "current_step": "shipment_check",
                "retry_count": 0
            }


def rag_answer_node(state: ConversationState) -> ConversationState:
    """
    知识库问答节点
    
    调用RAG服务根据用户问题召回相关文档片段。
    
    Args:
        state: 当前对话状态
    
    Returns:
        ConversationState: 更新了rag_retrieve_result的状态
    """
    try:
        user_input = state.get("user_input", "")
        top_k = state.get("business_params", {}).get("top_k", 3)
        
        logger.info(f"开始知识库检索，查询: {user_input[:50]}..., top_k: {top_k}")
        
        results = rag_service.retrieve_relevant_docs(user_input, top_k)
        
        if not results:
            logger.warning("知识库检索未找到相关文档")
        
        return {
            "rag_retrieve_result": results,
            "current_step": "rag_answer"
        }
    
    except Exception as e:
        logger.error(f"知识库问答节点异常: {str(e)}")
        return {
            "rag_retrieve_result": [],
            "current_step": "rag_answer",
            "risk_notes": [{
                "level": "error",
                "message": f"知识库检索失败: {str(e)}",
                "category": "RAG服务错误"
            }]
        }


def manual_confirm_node(state: ConversationState) -> ConversationState:
    """
    人工确认节点
    
    当发货校验通过后，等待人工确认才能继续执行发货。
    
    Args:
        state: 当前对话状态
    
    Returns:
        ConversationState: 更新了manual_confirmation状态
    """
    try:
        user_input = state.get("user_input", "").strip()
        
        if user_input in ["确认", "是", "同意", "ok", "OK", "yes", "YES"]:
            logger.info("人工确认通过")
            return {
                "manual_confirmation": True,
                "current_step": "manual_confirm"
            }
        elif user_input in ["取消", "否", "拒绝", "cancel", "CANCEL", "no", "NO"]:
            logger.info("人工确认取消")
            return {
                "manual_confirmation": False,
                "current_step": "manual_confirm",
                "overall_status": "取消"
            }
        else:
            logger.info("等待人工确认")
            return {
                "manual_confirmation": None,
                "current_step": "manual_confirm"
            }
    
    except Exception as e:
        logger.error(f"人工确认节点异常: {str(e)}")
        return {
            "manual_confirmation": False,
            "current_step": "manual_confirm",
            "risk_notes": [{
                "level": "error",
                "message": f"人工确认失败: {str(e)}",
                "category": "系统错误"
            }]
        }


def shipment_execution_node(state: ConversationState) -> ConversationState:
    """
    发货执行节点
    
    人工确认后，自动生成订单号和物流编码，创建订单并扣减库存。
    根据付款方式执行相应的付款操作（扣余额、使用信用额度、增加未回款）。
    增加幂等性控制，确保一次发货申请只产生一次发货。
    
    Args:
        state: 当前对话状态
    
    Returns:
        ConversationState: 更新了发货执行结果
    """
    try:
        business_params = state.get("business_params", {})
        customer_name = business_params.get("customer_name")
        material_code = business_params.get("material_code")
        quantity = business_params.get("quantity", 1)
        shipping_address = business_params.get("shipping_address")
        receiver = business_params.get("receiver")
        contact_phone = business_params.get("contact_phone")
        
        payment_result = state.get("payment_check_result", {})
        payment_action = payment_result.get("payment_action", "")
        total_amount = payment_result.get("total_amount", 0.0)
        
        logger.info(f"开始发货执行，客户名: {customer_name}, 物料编码: {material_code}, 数量: {quantity}, "
                    f"地址: {shipping_address}, 负责人: {receiver}, 电话: {contact_phone}, "
                    f"付款操作: {payment_action}, 订单金额: {total_amount}")
        
        with get_db() as db:
            existing_order = db.query(Order).filter(
                Order.customer_name == customer_name,
                Order.material_code == material_code,
                Order.quantity == quantity,
                Order.status == "待发货"
            ).first()
            
            if existing_order:
                logger.info(f"检测到重复发货请求，已存在待发货订单: {existing_order.order_no}")
                
                if payment_action and total_amount > 0:
                    update_customer_payment(db, customer_name, payment_action, total_amount)
                
                stock_info = db.query(Stock).filter(Stock.material_code == material_code).first()
                if stock_info and stock_info.available_stock >= quantity:
                    stock_info.available_stock -= quantity
                    logger.info(f"扣减库存，物料编码: {material_code}, 扣减数量: {quantity}, 剩余库存: {stock_info.available_stock}")
                
                existing_order.status = "待发货"
                
                return {
                    "shipment_order_no": existing_order.order_no,
                    "logistics_no": existing_order.logistics_no,
                    "shipment_executed": True,
                    "overall_status": "已发货",
                    "current_step": "shipment_execution",
                    "dao_query_result": {
                        "order_no": existing_order.order_no,
                        "logistics_no": existing_order.logistics_no,
                        "customer_name": existing_order.customer_name,
                        "material_code": existing_order.material_code,
                        "product_name": existing_order.product_name,
                        "quantity": existing_order.quantity,
                        "total_amount": existing_order.total_amount,
                        "status": "待发货",
                        "shipping_address": existing_order.shipping_address,
                        "receiver": existing_order.receiver,
                        "contact_phone": existing_order.contact_phone
                    }
                }
            
            if payment_action and total_amount > 0:
                update_customer_payment(db, customer_name, payment_action, total_amount)
            
            order_result = create_order(db, customer_name, material_code, quantity, 
                                        shipping_address, receiver, contact_phone, total_amount)
            
            order_no = order_result.get("order_no", "")
            logistics_no = order_result.get("logistics_no", "")
            
            logger.info(f"发货执行完成，订单号: {order_no}, 物流编码: {logistics_no}")
            
            return {
                "shipment_order_no": order_no,
                "logistics_no": logistics_no,
                "shipment_executed": True,
                "overall_status": "已发货",
                "current_step": "shipment_execution",
                "dao_query_result": order_result
            }
    
    except Exception as e:
        logger.error(f"发货执行节点异常: {str(e)}")
        return {
            "shipment_executed": False,
            "current_step": "shipment_execution",
            "overall_status": "失败",
            "risk_notes": [{
                "level": "error",
                "message": f"发货执行失败: {str(e)}",
                "category": "发货错误"
            }]
        }


def generate_reply_node(state: ConversationState) -> ConversationState:
    """
    统一回复生成节点
    
    整合当前状态中的所有结果，调用LLM生成自然、专业的最终回复。
    
    Args:
        state: 当前对话状态
    
    Returns:
        ConversationState: 更新了final_response的状态
    
    规则：
        - 数据查询场景：结构化数据整理为清晰易读的文本，重点信息突出
        - 发货校验场景：清晰列出已确认/待确认项，异常风险明确提示
        - 知识库场景：基于召回文档作答，标注信息来源
        - 信息不足场景：明确询问用户缺失的参数
        - 参数缺失场景：根据missing_params列表生成追问回复
    """
    try:
        intent = state.get("intent", "")
        user_input = state.get("user_input", "")
        business_params = state.get("business_params", {})
        dao_result = state.get("dao_query_result", {})
        rag_result = state.get("rag_retrieve_result", [])
        shipment_missing = state.get("shipment_missing_items", [])
        risk_notes = state.get("risk_notes", [])
        overall_status = state.get("overall_status", "")
        need_more_params = state.get("need_more_params", False)
        missing_params = state.get("missing_params", [])
        
        PARAM_LABELS = {
            "order_no": "订单号",
            "customer_name": "客户名称",
            "material_code": "物料编码",
            "warehouse": "仓库",
            "work_order_no": "工单号",
            "product_name": "产品名称",
            "quantity": "发货数量"
        }
        
        param_validation = state.get("param_validation", {})
        if param_validation and not param_validation.get("all_valid", True):
            suggestions = param_validation.get("suggestions", [])
            
            system_prompt = """你是一个专业的ERP智能客服助手。当参数校验失败时，需要向用户提示错误并给出建议的正确参数。

回复规则：
1. 清晰列出每个校验失败的参数
2. 给出系统建议的候选参数
3. 询问用户是否选择建议的参数或提供正确的参数
4. 保持回复友好、简洁"""
            
            suggestions_text = "\n".join([
                f"- {s.get('param_type', '')}: {s.get('message', '')}\n  建议选项: {', '.join(s.get('suggestions', []))}"
                for s in suggestions
            ])
            
            prompt = f"""参数校验失败！

用户输入：{user_input}
意图：{intent}
校验失败的参数：
{suggestions_text}

请生成友好、简洁的回复，提示用户参数错误并给出建议的选项。"""
            
            logger.info(f"参数校验失败，生成追问回复")
            response = ollama_client.chat(prompt, system_prompt)
            
            return {
                "final_response": response,
                "current_step": "generate_reply",
                "chat_history": [{
                    "user_input": user_input,
                    "assistant_response": response
                }]
            }
        
        if need_more_params and missing_params:
            if "confirm_intent" in missing_params:
                system_prompt = """你是一个专业的ERP智能客服助手。当检测到用户输入中的语义冲突时，需要向用户确认真实意图。

回复规则：
1. 清晰指出检测到的语义冲突
2. 列出系统推测的两种可能意图
3. 明确询问用户真实意图
4. 保持回复友好、简洁
5. 给出简单的确认选项（如"是"或"否"）"""
                
                risk_messages = [note.get("message", "") for note in risk_notes if note.get("level") == "warning"]
                risk_text = "\n".join(risk_messages)
                
                prompt = f"""检测到语义冲突！

用户输入：{user_input}
已提取参数：{json.dumps(business_params, ensure_ascii=False)}
冲突提示：
{risk_text}

请生成友好、简洁的确认回复，询问用户真实意图。"""
                
                logger.info(f"检测到语义冲突，生成确认回复")
                response = ollama_client.chat(prompt, system_prompt)
            
            else:
                missing_labels = [PARAM_LABELS.get(p, p) for p in missing_params]
                missing_text = "、".join(missing_labels)
                
                PARAM_EXAMPLES = {
                    "order_no": "例如 ORD20260710001",
                    "customer_name": "例如 上海科技有限公司",
                    "material_code": "例如 MAT001",
                    "warehouse": "例如 A仓库",
                    "work_order_no": "例如 WO20260710001",
                    "product_name": "例如 工业控制主板",
                    "quantity": "例如 100"
                }
                
                examples_text = "\n".join([
                    f"- {PARAM_LABELS.get(p, p)}: {PARAM_EXAMPLES.get(p, '')}"
                    for p in missing_params
                ])
                
                system_prompt = """你是一个专业的ERP智能客服助手。当用户查询订单、库存、生产进度等业务时，如果缺少必要参数，请直接追问用户补充。

追问规则：
1. 直接列出缺少的参数，明确告诉用户需要提供什么信息
2. 使用友好、简洁的语气
3. 给出每个参数的示例格式
4. 保持回复简短，不要有多余内容
5. 确保用户知道下一步该做什么"""
                
                prompt = f"""用户意图：{intent}
用户输入：{user_input}
已提取参数：{json.dumps(business_params, ensure_ascii=False)}
缺少的参数：{missing_text}

参数格式示例：
{examples_text}

请生成简短、明确的追问回复，直接询问用户缺少的参数。"""
                
                logger.info(f"参数缺失，生成追问回复，缺失参数: {missing_params}")
                response = ollama_client.chat(prompt, system_prompt)
            
            return {
                "final_response": response,
                "current_step": "generate_reply",
                "chat_history": [{
                    "user_input": user_input,
                    "assistant_response": response
                }]
            }
        
        elif intent in ["订单查询", "库存查询", "生产进度查询", "货款查询"]:
            system_prompt = DATA_QUERY_SYSTEM_PROMPT
            prompt = DATA_QUERY_USER_PROMPT.format(
                dao_result=json.dumps(dao_result, ensure_ascii=False, default=str),
                user_input=user_input
            )
        
        elif intent == "发货校验":
            bp = business_params
            current_step = state.get("current_step", "")
            manual_confirmation = state.get("manual_confirmation")
            shipment_executed = state.get("shipment_executed", False)
            
            current_risk_notes = state.get("risk_notes", [])
            current_missing_items = state.get("shipment_missing_items", [])
            current_payment_result = state.get("payment_check_result", {})
            current_stock_result = state.get("stock_check_result", {})
            
            if current_step == "manual_confirm" and manual_confirmation is None:
                system_prompt = """你是一个专业的ERP智能客服助手。当发货校验通过后，需要等待人工确认才能继续执行发货。
                
回复规则：
1. 清晰告知用户发货校验已通过
2. 列出校验通过的关键信息（客户名称、物料、数量、库存、信用状态）
3. 明确要求用户确认是否继续发货
4. 提示用户输入"确认"或"取消"进行操作"""
                
                prompt = f"""发货校验结果：通过
客户名称：{bp.get('customer_name', '')}
物料编码：{bp.get('material_code', '')}
产品名称：{current_stock_result.get('product_name', '')}
发货数量：{bp.get('quantity', 1)}
可用库存：{current_stock_result.get('available_stock', 0)}
客户可用余额：{current_payment_result.get('available_balance', 0)}
客户信用状态：{current_payment_result.get('credit_status', '')}

请回复"确认"继续发货，或回复"取消"取消发货。"""
                
                logger.info("等待人工确认发货")
                response = ollama_client.chat(prompt, system_prompt)
            
            elif current_step == "manual_confirm" and manual_confirmation is False:
                system_prompt = """你是一个专业的ERP智能客服助手。当用户取消发货时，礼貌告知发货已取消。
                
回复规则：
1. 礼貌告知用户发货已取消
2. 询问是否有其他需求"""
                
                prompt = f"""用户已取消发货操作。
                
请礼貌地告知用户发货已取消，并询问是否需要其他帮助。"""
                
                logger.info("用户取消发货")
                response = ollama_client.chat(prompt, system_prompt)
            
            elif current_step == "shipment_execution" and shipment_executed:
                system_prompt = """你是一个专业的ERP智能客服助手。当发货执行完成后，告知用户发货详情。
                
回复规则：
1. 清晰告知用户发货已完成
2. 列出订单号、物流编码等关键信息
3. 如果有收货地址、负责人、联系电话等信息，也要显示出来
4. 提示用户可以追踪物流"""
                
                order_no = state.get("shipment_order_no", "")
                logistics_no = state.get("logistics_no", "")
                dao_result = state.get("dao_query_result", {})
                
                shipping_address = dao_result.get('shipping_address', '')
                receiver = dao_result.get('receiver', '')
                contact_phone = dao_result.get('contact_phone', '')
                
                address_info = ""
                if shipping_address:
                    address_info += f"收货地址：{shipping_address}\n"
                if receiver:
                    address_info += f"负责人：{receiver}\n"
                if contact_phone:
                    address_info += f"联系电话：{contact_phone}\n"
                
                prompt = f"""发货执行完成！

订单号：{order_no}
物流编码：{logistics_no}
客户名称：{dao_result.get('customer_name', '')}
产品名称：{dao_result.get('product_name', '')}
发货数量：{dao_result.get('quantity', 0)}
订单金额：{dao_result.get('total_amount', 0)}
{address_info}
请告知用户发货已完成，并提示可以使用物流编码追踪物流信息。"""
                
                logger.info("发货执行完成")
                response = ollama_client.chat(prompt, system_prompt)
            
            else:
                system_prompt = SHIPMENT_CHECK_SYSTEM_PROMPT
                
                payment_info_text = f"""客户名称：{bp.get('customer_name', '')}
信用状态：{current_payment_result.get('credit_status', '')}
可用余额：{current_payment_result.get('available_balance', 0.0)}元
信用额度：{current_payment_result.get('credit_limit', 0.0)}元
未回款金额：{current_payment_result.get('outstanding_amount', 0.0)}元"""
                
                stock_info_text = f"""物料编码：{bp.get('material_code', '')}
产品名称：{current_stock_result.get('product_name', '')}
库存数量：{current_stock_result.get('available_stock', 0)}件
安全库存：{current_stock_result.get('safety_stock', 0)}件"""
                
                missing_items_text = "\n".join([
                    f"- {item.get('description', '')}"
                    for item in current_missing_items
                ]) if current_missing_items else "无"
                
                risk_notes_text = "\n".join([
                    f"- {item.get('message', '')} (级别: {item.get('level', '')})"
                    for item in current_risk_notes
                ]) if current_risk_notes else "无风险提示"
                
                prompt = f"""发货校验信息：
{payment_info_text}

{stock_info_text}

发货数量：{bp.get('quantity', 1)}
付款方式：{bp.get('payment_method', '')}
收货地址：{bp.get('shipping_address', '')}
收货人：{bp.get('receiver', '')}
联系电话：{bp.get('contact_phone', '')}

待确认项：
{missing_items_text}

风险提示：
{risk_notes_text}

用户输入：{user_input}

重要规则：
1. 订单号由系统自动生成，不需要用户提供，不要向用户询问订单号
2. 以上余额、信用额度等数据均来自数据库实时查询，请严格使用这些数据，不要编造或猜测
3. 如果用户说"重新查询"或"确认余额"，请直接使用上述数据库查询结果回复，不要生成新数据

请根据校验规则输出发货校验结果。"""
        
        elif intent == "知识库问答":
            docs_text = "\n".join([
                f"文档{i+1}: {doc['content'][:200]} (相似度: {doc['score']:.4f})"
                for i, doc in enumerate(rag_result)
            ])
            system_prompt = RAG_ANSWER_SYSTEM_PROMPT
            prompt = RAG_ANSWER_USER_PROMPT.format(
                documents=docs_text,
                user_input=user_input
            )
        
        else:
            system_prompt = GENERAL_REPLY_SYSTEM_PROMPT
            prompt = GENERAL_REPLY_USER_PROMPT.format(
                user_input=user_input
            )
        
        logger.info(f"开始生成回复，意图: {intent}")
        response = ollama_client.chat(prompt, system_prompt)
        
        return {
            "final_response": response,
            "current_step": "generate_reply",
            "chat_history": [{
                "user_input": user_input,
                "assistant_response": response
            }]
        }
    
    except Exception as e:
        logger.error(f"回复生成节点异常: {str(e)}")
        error_response = f"抱歉，处理您的请求时出现错误: {str(e)}"
        return {
            "final_response": error_response,
            "current_step": "generate_reply",
            "chat_history": [{
                "user_input": user_input,
                "assistant_response": error_response
            }]
        }
