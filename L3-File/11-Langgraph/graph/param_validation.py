from typing import List, Dict, Optional, Any
from utils.logger import get_logger
from dao.stock_dao import get_stock_by_material_code, get_material_code_by_product_name
from dao.payment_dao import get_customer_payment_by_name
from dao.models import Stock, CustomerPayment
from dao.db import get_db

logger = get_logger(__name__)


class ParamValidationResult:
    """
    参数校验结果
    """
    def __init__(self):
        self.valid = True
        self.suggestions: List[Dict[str, Any]] = []
        self.corrected_params: Dict[str, Any] = {}
        self.missing_params: List[Dict[str, str]] = []


def validate_product_name(product_name: str) -> Dict[str, Any]:
    """
    验证产品名称
    
    Args:
        product_name: 产品名称
        
    Returns:
        Dict: 验证结果，包含是否有效、建议的产品名称列表、物料编码
    """
    try:
        if not product_name:
            return {"valid": False, "message": "产品名称不能为空", "suggestions": [], "material_code": None}
        
        material_code = get_material_code_by_product_name(product_name)
        if material_code:
            stock_info = get_stock_by_material_code(material_code)
            if stock_info:
                return {
                    "valid": True,
                    "message": f"产品名称验证通过，对应物料编码: {material_code}",
                    "suggestions": [],
                    "material_code": material_code,
                    "product_name": stock_info.get("product_name")
                }
        
        with get_db(read_only=True) as db:
            suggestions = db.query(Stock.product_name).filter(
                Stock.product_name.like(f"%{product_name}%")
            ).limit(5).all()
            suggestion_list = [s[0] for s in suggestions]
            
            if suggestion_list:
                return {
                    "valid": False,
                    "message": f"未找到产品'{product_name}'，是否指以下产品？",
                    "suggestions": suggestion_list,
                    "material_code": None
                }
            else:
                all_products = db.query(Stock.product_name).limit(5).all()
                all_product_list = [p[0] for p in all_products]
                return {
                    "valid": False,
                    "message": f"未找到产品'{product_name}'，系统中存在以下产品：",
                    "suggestions": all_product_list,
                    "material_code": None
                }
    except Exception as e:
        logger.error(f"验证产品名称异常: {str(e)}")
        return {"valid": False, "message": f"验证产品名称异常: {str(e)}", "suggestions": [], "material_code": None}


def validate_material_code(material_code: str) -> Dict[str, Any]:
    """
    验证物料编码
    
    Args:
        material_code: 物料编码
        
    Returns:
        Dict: 验证结果，包含是否有效、建议的物料编码列表、产品名称
    """
    try:
        if not material_code:
            return {"valid": False, "message": "物料编码不能为空", "suggestions": [], "product_name": None}
        
        stock_info = get_stock_by_material_code(material_code)
        if stock_info:
            return {
                "valid": True,
                "message": f"物料编码验证通过，对应产品: {stock_info.get('product_name')}",
                "suggestions": [],
                "product_name": stock_info.get("product_name"),
                "material_code": material_code
            }
        
        with get_db(read_only=True) as db:
            suggestions = db.query(Stock.material_code).filter(
                Stock.material_code.like(f"%{material_code}%")
            ).limit(5).all()
            suggestion_list = [s[0] for s in suggestions]
            
            if suggestion_list:
                return {
                    "valid": False,
                    "message": f"未找到物料编码'{material_code}'，是否指以下编码？",
                    "suggestions": suggestion_list,
                    "product_name": None
                }
            else:
                all_codes = db.query(Stock.material_code).limit(5).all()
                all_code_list = [c[0] for c in all_codes]
                return {
                    "valid": False,
                    "message": f"未找到物料编码'{material_code}'，系统中存在以下物料编码：",
                    "suggestions": all_code_list,
                    "product_name": None
                }
    except Exception as e:
        logger.error(f"验证物料编码异常: {str(e)}")
        return {"valid": False, "message": f"验证物料编码异常: {str(e)}", "suggestions": [], "product_name": None}


def validate_customer_name(customer_name: str) -> Dict[str, Any]:
    """
    验证客户名称
    
    Args:
        customer_name: 客户名称
        
    Returns:
        Dict: 验证结果，包含是否有效、建议的客户名称列表
    """
    try:
        if not customer_name:
            return {"valid": False, "message": "客户名称不能为空", "suggestions": [], "customer_code": None}
        
        payment_info = get_customer_payment_by_name(customer_name)
        if payment_info:
            return {
                "valid": True,
                "message": f"客户名称验证通过，信用状态: {payment_info.get('credit_status')}",
                "suggestions": [],
                "customer_code": payment_info.get("customer_code"),
                "customer_name": payment_info.get("customer_name")
            }
        
        with get_db(read_only=True) as db:
            suggestions = db.query(CustomerPayment.customer_name).filter(
                CustomerPayment.customer_name.like(f"%{customer_name}%")
            ).limit(5).all()
            suggestion_list = [s[0] for s in suggestions]
            
            if suggestion_list:
                return {
                    "valid": False,
                    "message": f"未找到客户'{customer_name}'，是否指以下客户？",
                    "suggestions": suggestion_list,
                    "customer_code": None
                }
            else:
                all_customers = db.query(CustomerPayment.customer_name).limit(5).all()
                all_customer_list = [c[0] for c in all_customers]
                return {
                    "valid": False,
                    "message": f"未找到客户'{customer_name}'，系统中存在以下客户：",
                    "suggestions": all_customer_list,
                    "customer_code": None
                }
    except Exception as e:
        logger.error(f"验证客户名称异常: {str(e)}")
        return {"valid": False, "message": f"验证客户名称异常: {str(e)}", "suggestions": [], "customer_code": None}


def validate_quantity(quantity: Any) -> Dict[str, Any]:
    """
    验证数量
    
    Args:
        quantity: 数量
        
    Returns:
        Dict: 验证结果，包含是否有效、转换后的数量值
    """
    try:
        if quantity is None:
            return {"valid": False, "message": "数量不能为空", "value": None}
        
        if isinstance(quantity, str):
            quantity = quantity.strip()
        
        if isinstance(quantity, str) and quantity.isdigit():
            quantity = int(quantity)
        
        if isinstance(quantity, int) and quantity > 0:
            return {"valid": True, "message": f"数量验证通过: {quantity}", "value": quantity}
        
        return {"valid": False, "message": f"数量格式不正确或小于等于0: {quantity}", "value": None}
    except Exception as e:
        logger.error(f"验证数量异常: {str(e)}")
        return {"valid": False, "message": f"验证数量异常: {str(e)}", "value": None}


def param_validation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    参数校验节点
    
    在查询数据库之前对业务参数进行验证，确保参数的正确性。
    如果参数无效，给出可能的候选参数供用户选择。
    
    Args:
        state: 当前对话状态
        
    Returns:
        Dict: 更新了参数校验结果的状态
    """
    try:
        intent = state.get("intent", "")
        business_params = state.get("business_params", {})
        validation_results = {}
        corrected_params = {}
        validation_suggestions = []
        all_valid = True
        
        logger.info(f"开始参数校验，意图: {intent}, 参数: {business_params}")
        
        if intent in ["发货校验", "发货申请", "库存查询"]:
            material_code = business_params.get("material_code")
            product_name = business_params.get("product_name")
            
            if material_code:
                result = validate_material_code(material_code)
                validation_results["material_code"] = result
                if result["valid"]:
                    corrected_params["material_code"] = result["material_code"]
                    corrected_params["product_name"] = result["product_name"]
                else:
                    all_valid = False
                    validation_suggestions.append({
                        "param_type": "material_code",
                        "original_value": material_code,
                        "message": result["message"],
                        "suggestions": result["suggestions"]
                    })
            
            elif product_name:
                result = validate_product_name(product_name)
                validation_results["product_name"] = result
                if result["valid"]:
                    corrected_params["product_name"] = result["product_name"]
                    corrected_params["material_code"] = result["material_code"]
                else:
                    all_valid = False
                    validation_suggestions.append({
                        "param_type": "product_name",
                        "original_value": product_name,
                        "message": result["message"],
                        "suggestions": result["suggestions"]
                    })
            
            quantity = business_params.get("quantity")
            if quantity:
                result = validate_quantity(quantity)
                validation_results["quantity"] = result
                if result["valid"]:
                    corrected_params["quantity"] = result["value"]
                else:
                    all_valid = False
                    validation_suggestions.append({
                        "param_type": "quantity",
                        "original_value": str(quantity),
                        "message": result["message"],
                        "suggestions": []
                    })
        
        if intent in ["发货校验", "发货申请", "货款查询"]:
            customer_name = business_params.get("customer_name")
            if customer_name:
                result = validate_customer_name(customer_name)
                validation_results["customer_name"] = result
                if result["valid"]:
                    corrected_params["customer_name"] = result["customer_name"]
                    corrected_params["customer_code"] = result["customer_code"]
                else:
                    all_valid = False
                    validation_suggestions.append({
                        "param_type": "customer_name",
                        "original_value": customer_name,
                        "message": result["message"],
                        "suggestions": result["suggestions"]
                    })
        
        if intent in ["订单查询"]:
            order_no = business_params.get("order_no")
            if order_no and not order_no.startswith("ORD"):
                all_valid = False
                validation_suggestions.append({
                    "param_type": "order_no",
                    "original_value": order_no,
                    "message": f"订单号格式不正确，应以ORD开头（如ORD20260710001），当前: {order_no}",
                    "suggestions": []
                })
        
        if intent in ["生产进度查询"]:
            work_order_no = business_params.get("work_order_no")
            if work_order_no and not work_order_no.startswith("WO"):
                all_valid = False
                validation_suggestions.append({
                    "param_type": "work_order_no",
                    "original_value": work_order_no,
                    "message": f"工单号格式不正确，应以WO开头（如WO20260710001），当前: {work_order_no}",
                    "suggestions": []
                })
        
        logger.info(f"参数校验完成，全部有效: {all_valid}, 校验结果: {validation_results}")
        
        return {
            "param_validation": {
                "all_valid": all_valid,
                "results": validation_results,
                "suggestions": validation_suggestions
            },
            "business_params": {**business_params, **corrected_params},
            "current_step": "param_validation"
        }
    
    except Exception as e:
        logger.error(f"参数校验节点异常: {str(e)}")
        return {
            "param_validation": {
                "all_valid": False,
                "results": {},
                "suggestions": [{
                    "param_type": "system",
                    "original_value": "",
                    "message": f"参数校验失败: {str(e)}",
                    "suggestions": []
                }]
            },
            "current_step": "param_validation",
            "risk_notes": [{
                "level": "error",
                "message": f"参数校验失败: {str(e)}",
                "category": "系统错误"
            }]
        }


def route_after_param_validation(state: Dict[str, Any]) -> str:
    """
    参数校验后的路由
    
    根据校验结果决定下一步：
    - 校验通过 → 继续原有的业务流程
    - 校验失败 → 生成追问回复，让用户选择正确的参数
    """
    validation_result = state.get("param_validation", {})
    all_valid = validation_result.get("all_valid", True)
    intent = state.get("intent", "")
    
    if all_valid:
        logger.info(f"参数校验通过，继续原业务流程，意图: {intent}")
        return "continue"
    else:
        logger.info(f"参数校验失败，生成追问回复")
        return "generate_reply"