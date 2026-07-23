from sqlalchemy.orm import Session
from dao.models import Order, Stock, CustomerPayment
from dao.db import add_operation_log, generate_order_no, generate_logistics_no
from utils.logger import get_logger

logger = get_logger(__name__)


def create_order(db: Session, customer_name: str, material_code: str, quantity: int,
                 shipping_address: str = None, receiver: str = None, contact_phone: str = None,
                 total_amount: float = None, operator: str = None) -> dict:
    """
    创建订单
    
    根据客户名称和物料编码创建订单，自动生成递增的订单号和物流编码。
    增加幂等性控制：同一客户、同一物料、同一数量的待发货订单只创建一次。
    
    Args:
        db: 数据库会话
        customer_name: 客户名称
        material_code: 物料编码
        quantity: 发货数量
        shipping_address: 收货地址
        receiver: 收货人
        contact_phone: 联系电话
        operator: 操作人
    
    Returns:
        dict: 创建的订单信息
    
    Raises:
        Exception: 创建订单失败时抛出异常
    """
    try:
        stock_info = db.query(Stock).filter(Stock.material_code == material_code).first()
        if not stock_info:
            raise Exception(f"未找到物料: {material_code}")
        
        if stock_info.available_stock < quantity:
            raise Exception(f"库存不足，可用库存: {stock_info.available_stock}, 需求: {quantity}")
        
        customer_info = db.query(CustomerPayment).filter(CustomerPayment.customer_name == customer_name).first()
        if not customer_info:
            raise Exception(f"未找到客户: {customer_name}")
        
        existing_order = db.query(Order).filter(
            Order.customer_name == customer_name,
            Order.material_code == material_code,
            Order.quantity == quantity,
            Order.status == "待发货"
        ).first()
        
        if existing_order:
            logger.info(f"检测到重复发货请求，已存在待发货订单: {existing_order.order_no}")
            
            needs_update = False
            old_data = {}
            
            if shipping_address and existing_order.shipping_address != shipping_address:
                old_data['shipping_address'] = existing_order.shipping_address
                existing_order.shipping_address = shipping_address
                needs_update = True
            
            if receiver and existing_order.receiver != receiver:
                old_data['receiver'] = existing_order.receiver
                existing_order.receiver = receiver
                needs_update = True
            
            if contact_phone and existing_order.contact_phone != contact_phone:
                old_data['contact_phone'] = existing_order.contact_phone
                existing_order.contact_phone = contact_phone
                needs_update = True
            
            if needs_update:
                new_data = {
                    'shipping_address': existing_order.shipping_address,
                    'receiver': existing_order.receiver,
                    'contact_phone': existing_order.contact_phone
                }
                add_operation_log(db, "UPDATE", "orders", existing_order.id, old_data, new_data, operator,
                                 f"更新订单收货信息，订单号: {existing_order.order_no}")
                logger.info(f"更新订单收货信息成功，订单号: {existing_order.order_no}")
            
            return {
                "order_no": existing_order.order_no,
                "logistics_no": existing_order.logistics_no,
                "customer_name": existing_order.customer_name,
                "material_code": existing_order.material_code,
                "product_name": existing_order.product_name,
                "quantity": existing_order.quantity,
                "total_amount": existing_order.total_amount,
                "status": existing_order.status,
                "shipping_address": existing_order.shipping_address,
                "receiver": existing_order.receiver,
                "contact_phone": existing_order.contact_phone
            }
        
        order_no = generate_order_no(db)
        logistics_no = generate_logistics_no(db)
        
        if total_amount is None:
            unit_price = stock_info.unit_price if stock_info.unit_price > 0 else 1000
            total_amount = unit_price * quantity
        
        order = Order(
            order_no=order_no,
            customer_name=customer_name,
            material_code=material_code,
            quantity=quantity,
            product_name=stock_info.product_name,
            total_amount=total_amount,
            status="待发货",
            logistics_no=logistics_no,
            shipping_address=shipping_address,
            receiver=receiver,
            contact_phone=contact_phone
        )
        
        db.add(order)
        
        old_stock_data = {
            "material_code": stock_info.material_code,
            "available_stock": stock_info.available_stock
        }
        
        stock_info.available_stock -= quantity
        
        new_stock_data = {
            "material_code": stock_info.material_code,
            "available_stock": stock_info.available_stock
        }
        
        add_operation_log(db, "UPDATE", "stock", stock_info.id, old_stock_data, new_stock_data, operator,
                         f"发货扣减库存，订单号: {order_no}")
        
        add_operation_log(db, "INSERT", "orders", None, None, {
            "order_no": order_no,
            "customer_name": customer_name,
            "material_code": material_code,
            "quantity": quantity,
            "product_name": stock_info.product_name,
            "total_amount": total_amount,
            "status": "待发货",
            "logistics_no": logistics_no
        }, operator, f"创建发货订单")
        
        logger.info(f"订单创建成功，订单号: {order_no}, 物流编码: {logistics_no}")
        
        return {
            "order_no": order_no,
            "logistics_no": logistics_no,
            "customer_name": customer_name,
            "material_code": material_code,
            "product_name": stock_info.product_name,
            "quantity": quantity,
            "total_amount": total_amount,
            "status": "待发货",
            "shipping_address": shipping_address,
            "receiver": receiver,
            "contact_phone": contact_phone
        }
    
    except Exception as e:
        logger.error(f"创建订单失败: {str(e)}")
        raise


def update_order_status(db: Session, order_no: str, new_status: str, operator: str = None) -> dict:
    """
    更新订单状态
    
    Args:
        db: 数据库会话
        order_no: 订单号
        new_status: 新状态
        operator: 操作人
    
    Returns:
        dict: 更新后的订单信息
    """
    try:
        order = db.query(Order).filter(Order.order_no == order_no).first()
        if not order:
            raise Exception(f"未找到订单: {order_no}")
        
        old_data = {
            "order_no": order.order_no,
            "status": order.status
        }
        
        order.status = new_status
        
        new_data = {
            "order_no": order.order_no,
            "status": order.status
        }
        
        add_operation_log(db, "UPDATE", "orders", order.id, old_data, new_data, operator,
                         f"更新订单状态: {old_data['status']} -> {new_status}")
        
        logger.info(f"订单状态更新成功，订单号: {order_no}, 状态: {new_status}")
        
        return {
            "order_no": order.order_no,
            "status": order.status
        }
    
    except Exception as e:
        logger.error(f"更新订单状态失败: {str(e)}")
        raise


def get_operation_logs(db: Session, table_name: str = None, operation_type: str = None, limit: int = 100) -> list:
    """
    查询操作记录
    
    Args:
        db: 数据库会话
        table_name: 表名筛选
        operation_type: 操作类型筛选（INSERT、UPDATE、DELETE）
        limit: 返回条数限制
    
    Returns:
        list: 操作记录列表
    """
    from dao.models import OperationLog
    
    try:
        query = db.query(OperationLog)
        
        if table_name:
            query = query.filter(OperationLog.table_name == table_name)
        
        if operation_type:
            query = query.filter(OperationLog.operation_type == operation_type)
        
        query = query.order_by(OperationLog.operation_time.desc()).limit(limit)
        
        results = query.all()
        
        return [{
            "id": log.id,
            "operation_type": log.operation_type,
            "table_name": log.table_name,
            "record_id": log.record_id,
            "old_data": log.old_data,
            "new_data": log.new_data,
            "operator": log.operator,
            "operation_time": log.operation_time.isoformat() if log.operation_time else None,
            "remark": log.remark
        } for log in results]
    
    except Exception as e:
        logger.error(f"查询操作记录失败: {str(e)}")
        raise
