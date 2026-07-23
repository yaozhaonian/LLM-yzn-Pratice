from dao.db import get_db
from dao.models import CustomerPayment
from utils.logger import get_logger

logger = get_logger(__name__)


def get_customer_payment_by_name(customer_name: str) -> dict:
    """
    按客户名查询可用余额与信用状态
    
    Args:
        customer_name: 客户名称（模糊匹配）
    
    Returns:
        dict: 客户货款信息字典，未找到返回空字典
    
    Raises:
        Exception: 查询异常
    """
    try:
        with get_db(read_only=True) as db:
            payment = db.query(CustomerPayment).filter(
                CustomerPayment.customer_name.like(f"%{customer_name}%")
            ).first()
            if payment:
                logger.info(f"查询客户货款成功，客户名: {customer_name}")
                return {
                    "customer_code": payment.customer_code,
                    "customer_name": payment.customer_name,
                    "credit_limit": payment.credit_limit,
                    "available_balance": payment.available_balance,
                    "outstanding_amount": payment.outstanding_amount,
                    "credit_status": payment.credit_status,
                    "used_ratio": round(payment.outstanding_amount / payment.credit_limit * 100, 2)
                    if payment.credit_limit > 0 else 0,
                    "update_time": payment.update_time.strftime("%Y-%m-%d %H:%M:%S") if payment.update_time else None
                }
            logger.warning(f"未找到客户货款信息，客户名: {customer_name}")
            return {}
    except Exception as e:
        logger.error(f"查询客户货款异常，客户名: {customer_name}, 错误: {str(e)}")
        raise


def get_customer_payment_by_code(customer_code: str) -> dict:
    """
    按客户编码查询货款信息
    
    Args:
        customer_code: 客户编码
    
    Returns:
        dict: 客户货款信息字典，未找到返回空字典
    
    Raises:
        Exception: 查询异常
    """
    try:
        with get_db(read_only=True) as db:
            payment = db.query(CustomerPayment).filter(
                CustomerPayment.customer_code == customer_code
            ).first()
            if payment:
                logger.info(f"查询客户货款成功，客户编码: {customer_code}")
                return {
                    "customer_code": payment.customer_code,
                    "customer_name": payment.customer_name,
                    "credit_limit": payment.credit_limit,
                    "available_balance": payment.available_balance,
                    "outstanding_amount": payment.outstanding_amount,
                    "credit_status": payment.credit_status,
                    "used_ratio": round(payment.outstanding_amount / payment.credit_limit * 100, 2)
                    if payment.credit_limit > 0 else 0,
                    "update_time": payment.update_time.strftime("%Y-%m-%d %H:%M:%S") if payment.update_time else None
                }
            logger.warning(f"未找到客户货款信息，客户编码: {customer_code}")
            return {}
    except Exception as e:
        logger.error(f"查询客户货款异常，客户编码: {customer_code}, 错误: {str(e)}")
        raise


def update_customer_payment(db, customer_name: str, action: str, amount: float):
    """
    更新客户货款信息
    
    Args:
        db: 数据库会话
        customer_name: 客户名称
        action: 操作类型，可选值：deduct_balance(扣余额)、use_credit(使用信用额度扣款)、add_outstanding(增加未回款)
        amount: 金额
    
    Returns:
        bool: 是否更新成功
    
    Raises:
        Exception: 更新异常
    """
    try:
        payment = db.query(CustomerPayment).filter(
            CustomerPayment.customer_name.like(f"%{customer_name}%")
        ).first()
        
        if not payment:
            logger.error(f"未找到客户货款信息，客户名: {customer_name}")
            return False
        
        if action == "deduct_balance":
            payment.available_balance -= amount
            logger.info(f"扣减客户余额，客户名: {customer_name}, 扣款金额: {amount}, 剩余余额: {payment.available_balance}")
        
        elif action == "use_credit":
            payment.available_balance -= amount
            credit_used = 0
            if payment.available_balance < 0:
                credit_used = abs(payment.available_balance)
                payment.outstanding_amount += credit_used
                payment.available_balance = 0
                payment.credit_limit -= credit_used
            else:
                payment.credit_limit -= amount
                credit_used = amount
            if payment.outstanding_amount > 0:
                payment.credit_status = "余额不足"
            logger.info(f"使用信用额度扣款，客户名: {customer_name}, 扣款金额: {amount}, "
                        f"使用信用额度: {credit_used}, 剩余余额: {payment.available_balance}, "
                        f"未回款金额: {payment.outstanding_amount}, 剩余信用额度: {payment.credit_limit}, "
                        f"账期状态: {payment.credit_status}")
        
        elif action == "add_outstanding":
            payment.outstanding_amount += amount
            payment.credit_limit -= amount
            if payment.outstanding_amount > 0:
                payment.credit_status = "余额不足"
            logger.info(f"增加客户未回款金额，客户名: {customer_name}, 增加金额: {amount}, "
                        f"未回款金额: {payment.outstanding_amount}, 剩余信用额度: {payment.credit_limit}, "
                        f"账期状态: {payment.credit_status}")
        
        else:
            logger.warning(f"未知的付款操作类型: {action}")
            return False
        
        db.commit()
        logger.info(f"客户货款信息更新成功，客户名: {customer_name}")
        return True
    
    except Exception as e:
        db.rollback()
        logger.error(f"更新客户货款信息异常，客户名: {customer_name}, 错误: {str(e)}")
        raise
