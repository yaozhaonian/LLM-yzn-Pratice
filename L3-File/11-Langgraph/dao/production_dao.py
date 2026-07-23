from dao.db import get_db
from dao.models import ProductionWorkOrder
from utils.logger import get_logger

logger = get_logger(__name__)


def get_work_order_progress(work_order_no: str) -> dict:
    """
    按工单号查询工单进度
    
    Args:
        work_order_no: 工单号
    
    Returns:
        dict: 工单进度信息字典，未找到返回空字典
    
    Raises:
        Exception: 查询异常
    """
    try:
        with get_db(read_only=True) as db:
            work_order = db.query(ProductionWorkOrder).filter(
                ProductionWorkOrder.work_order_no == work_order_no
            ).first()
            if work_order:
                logger.info(f"查询工单进度成功，工单号: {work_order_no}")
                return {
                    "work_order_no": work_order.work_order_no,
                    "product_name": work_order.product_name,
                    "current_process": work_order.current_process,
                    "completion_rate": work_order.completion_rate,
                    "estimated_completion_time": work_order.estimated_completion_time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ) if work_order.estimated_completion_time else None,
                    "status": work_order.status,
                    "exception_note": work_order.exception_note,
                    "create_time": work_order.create_time.strftime("%Y-%m-%d %H:%M:%S") if work_order.create_time else None,
                    "update_time": work_order.update_time.strftime("%Y-%m-%d %H:%M:%S") if work_order.update_time else None
                }
            logger.warning(f"未找到工单，工单号: {work_order_no}")
            return {}
    except Exception as e:
        logger.error(f"查询工单进度异常，工单号: {work_order_no}, 错误: {str(e)}")
        raise


def get_in_progress_work_orders(product_name: str) -> list:
    """
    按产品名查询进行中的工单列表
    
    Args:
        product_name: 产品名称（模糊匹配）
    
    Returns:
        list: 工单信息字典列表
    
    Raises:
        Exception: 查询异常
    """
    try:
        with get_db(read_only=True) as db:
            work_orders = db.query(ProductionWorkOrder).filter(
                ProductionWorkOrder.product_name.like(f"%{product_name}%"),
                ProductionWorkOrder.status == "进行中"
            ).order_by(ProductionWorkOrder.completion_rate.asc()).all()
            
            result = []
            for work_order in work_orders:
                result.append({
                    "work_order_no": work_order.work_order_no,
                    "product_name": work_order.product_name,
                    "current_process": work_order.current_process,
                    "completion_rate": work_order.completion_rate,
                    "estimated_completion_time": work_order.estimated_completion_time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ) if work_order.estimated_completion_time else None,
                    "status": work_order.status,
                    "exception_note": work_order.exception_note,
                    "create_time": work_order.create_time.strftime("%Y-%m-%d %H:%M:%S") if work_order.create_time else None,
                    "update_time": work_order.update_time.strftime("%Y-%m-%d %H:%M:%S") if work_order.update_time else None
                })
            
            logger.info(f"查询进行中工单成功，产品名: {product_name}, 数量: {len(result)}")
            return result
    except Exception as e:
        logger.error(f"查询进行中工单异常，产品名: {product_name}, 错误: {str(e)}")
        raise
