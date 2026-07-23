from datetime import datetime
from sqlalchemy import and_
from dao.db import get_db
from dao.models import Order
from utils.logger import get_logger

logger = get_logger(__name__)


def get_order_by_order_no(order_no: str) -> dict:
    """
    按订单号查询订单
    
    Args:
        order_no: 订单号
    
    Returns:
        dict: 订单信息字典，未找到返回空字典
    
    Raises:
        Exception: 查询异常
    """
    try:
        with get_db(read_only=True) as db:
            order = db.query(Order).filter(Order.order_no == order_no).first()
            if order:
                logger.info(f"查询订单成功，订单号: {order_no}")
                return {
                    "order_no": order.order_no,
                    "customer_name": order.customer_name,
                    "total_amount": order.total_amount,
                    "status": order.status,
                    "shipping_address": order.shipping_address,
                    "receiver": order.receiver,
                    "contact_phone": order.contact_phone,
                    "create_time": order.create_time.strftime("%Y-%m-%d %H:%M:%S") if order.create_time else None,
                    "update_time": order.update_time.strftime("%Y-%m-%d %H:%M:%S") if order.update_time else None
                }
            logger.warning(f"未找到订单，订单号: {order_no}")
            return {}
    except Exception as e:
        logger.error(f"查询订单异常，订单号: {order_no}, 错误: {str(e)}")
        raise


def get_orders_by_customer_and_time(
    customer_name: str,
    start_time: datetime = None,
    end_time: datetime = None
) -> list:
    """
    按客户名和时间范围查询订单列表
    
    Args:
        customer_name: 客户名称（模糊匹配）
        start_time: 开始时间
        end_time: 结束时间
    
    Returns:
        list: 订单信息字典列表
    
    Raises:
        Exception: 查询异常
    """
    try:
        with get_db(read_only=True) as db:
            query = db.query(Order).filter(Order.customer_name.like(f"%{customer_name}%"))
            
            if start_time:
                query = query.filter(Order.create_time >= start_time)
            if end_time:
                query = query.filter(Order.create_time <= end_time)
            
            orders = query.order_by(Order.create_time.desc()).all()
            
            result = []
            for order in orders:
                result.append({
                    "order_no": order.order_no,
                    "customer_name": order.customer_name,
                    "total_amount": order.total_amount,
                    "status": order.status,
                    "shipping_address": order.shipping_address,
                    "receiver": order.receiver,
                    "contact_phone": order.contact_phone,
                    "create_time": order.create_time.strftime("%Y-%m-%d %H:%M:%S") if order.create_time else None,
                    "update_time": order.update_time.strftime("%Y-%m-%d %H:%M:%S") if order.update_time else None
                })
            
            logger.info(f"查询订单列表成功，客户名: {customer_name}, 数量: {len(result)}")
            return result
    except Exception as e:
        logger.error(f"查询订单列表异常，客户名: {customer_name}, 错误: {str(e)}")
        raise
