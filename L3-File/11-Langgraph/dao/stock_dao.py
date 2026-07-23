from dao.db import get_db
from dao.models import Stock
from utils.logger import get_logger

logger = get_logger(__name__)


def get_stock_by_material_code(material_code: str) -> dict:
    """
    按物料编码查询库存
    
    Args:
        material_code: 物料编码
    
    Returns:
        dict: 库存信息字典，未找到返回空字典
    
    Raises:
        Exception: 查询异常
    """
    try:
        with get_db(read_only=True) as db:
            stock = db.query(Stock).filter(Stock.material_code == material_code).first()
            if stock:
                logger.info(f"查询库存成功，物料编码: {material_code}")
                return {
                    "material_code": stock.material_code,
                    "product_name": stock.product_name,
                    "warehouse": stock.warehouse,
                    "available_stock": stock.available_stock,
                    "safety_stock": stock.safety_stock,
                    "unit_price": stock.unit_price,
                    "is_below_safety": stock.available_stock < stock.safety_stock,
                    "update_time": stock.update_time.strftime("%Y-%m-%d %H:%M:%S") if stock.update_time else None
                }
            logger.warning(f"未找到库存，物料编码: {material_code}")
            return {}
    except Exception as e:
        logger.error(f"查询库存异常，物料编码: {material_code}, 错误: {str(e)}")
        raise


def get_material_code_by_product_name(product_name: str) -> str:
    """
    按产品名称查询物料编码
    
    Args:
        product_name: 产品名称
    
    Returns:
        str: 物料编码，未找到返回空字符串
    """
    try:
        with get_db(read_only=True) as db:
            stock = db.query(Stock).filter(Stock.product_name.like(f"%{product_name}%")).first()
            if stock:
                logger.info(f"通过产品名称查询物料编码成功，产品名称: {product_name}, 物料编码: {stock.material_code}")
                return stock.material_code
            logger.warning(f"未找到产品对应的物料编码，产品名称: {product_name}")
            return ""
    except Exception as e:
        logger.error(f"通过产品名称查询物料编码异常，产品名称: {product_name}, 错误: {str(e)}")
        return ""


def get_stocks_by_warehouse(warehouse: str) -> list:
    """
    按仓库查询全部库存
    
    Args:
        warehouse: 仓库名称（模糊匹配）
    
    Returns:
        list: 库存信息字典列表
    
    Raises:
        Exception: 查询异常
    """
    try:
        with get_db(read_only=True) as db:
            stocks = db.query(Stock).filter(Stock.warehouse.like(f"%{warehouse}%")).all()
            
            result = []
            for stock in stocks:
                result.append({
                    "material_code": stock.material_code,
                    "product_name": stock.product_name,
                    "warehouse": stock.warehouse,
                    "available_stock": stock.available_stock,
                    "safety_stock": stock.safety_stock,
                    "is_below_safety": stock.available_stock < stock.safety_stock,
                    "update_time": stock.update_time.strftime("%Y-%m-%d %H:%M:%S") if stock.update_time else None
                })
            
            logger.info(f"查询仓库库存成功，仓库: {warehouse}, 数量: {len(result)}")
            return result
    except Exception as e:
        logger.error(f"查询仓库库存异常，仓库: {warehouse}, 错误: {str(e)}")
        raise
