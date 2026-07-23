from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from config.settings import settings
from utils.logger import get_logger
import json

logger = get_logger(__name__)

engine = create_engine(
    settings.mysql.url,
    pool_size=settings.mysql.pool_size,
    max_overflow=settings.mysql.max_overflow,
    pool_pre_ping=True,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


@contextmanager
def get_db(read_only: bool = False):
    """
    获取数据库会话上下文管理器
    
    提供安全的数据库会话管理，自动处理会话的创建、提交和回滚。
    
    Args:
        read_only: 是否为只读模式，只读模式下跳过commit操作
    
    Yields:
        Session: SQLAlchemy数据库会话对象
    
    Raises:
        Exception: 数据库操作异常
    """
    db = SessionLocal()
    try:
        yield db
        if not read_only:
            db.commit()
            logger.debug("数据库会话提交成功")
    except Exception as e:
        db.rollback()
        logger.error(f"数据库操作异常，已回滚: {str(e)}")
        raise
    finally:
        db.close()
        logger.debug("数据库会话已关闭")


def init_db():
    """
    初始化数据库表结构
    
    创建所有ORM模型对应的数据库表。
    
    Raises:
        Exception: 数据库初始化异常
    """
    try:
        from dao import models
        Base.metadata.create_all(bind=engine)
        logger.info("数据库表结构初始化完成")
    except Exception as e:
        logger.error(f"数据库表结构初始化失败: {str(e)}")
        raise


def add_operation_log(db: Session, operation_type: str, table_name: str, 
                      record_id: int = None, old_data: dict = None, 
                      new_data: dict = None, operator: str = None, remark: str = None):
    """
    添加操作记录
    
    记录数据库的增、删、改操作，用于审计和追踪。
    
    Args:
        db: 数据库会话
        operation_type: 操作类型：INSERT、UPDATE、DELETE
        table_name: 操作的表名
        record_id: 操作记录的ID
        old_data: 操作前的数据（字典格式）
        new_data: 操作后的数据（字典格式）
        operator: 操作人
        remark: 备注
    """
    try:
        from dao.models import OperationLog
        
        log = OperationLog(
            operation_type=operation_type,
            table_name=table_name,
            record_id=record_id,
            old_data=json.dumps(old_data, ensure_ascii=False) if old_data else None,
            new_data=json.dumps(new_data, ensure_ascii=False) if new_data else None,
            operator=operator,
            remark=remark
        )
        db.add(log)
        logger.info(f"操作记录已添加: {operation_type} {table_name} {record_id}")
    except Exception as e:
        logger.error(f"添加操作记录失败: {str(e)}")


def generate_order_no(db: Session) -> str:
    """
    生成递增的订单号
    
    格式：ORD + 年月日 + 4位序号
    
    Args:
        db: 数据库会话
        
    Returns:
        str: 订单号
    """
    from datetime import datetime
    from dao.models import Order
    
    today = datetime.now().strftime("%Y%m%d")
    prefix = f"ORD{today}"
    
    result = db.query(func.max(Order.order_no)).filter(Order.order_no.like(f"{prefix}%")).first()
    max_no = result[0]
    
    if max_no:
        seq = int(max_no[-4:]) + 1
        return f"{prefix}{seq:04d}"
    else:
        return f"{prefix}0001"


def generate_logistics_no(db: Session) -> str:
    """
    生成递增的物流编码
    
    格式：LOG + 年月日 + 4位序号
    
    Args:
        db: 数据库会话
        
    Returns:
        str: 物流编码
    """
    from datetime import datetime
    from dao.models import Order
    
    today = datetime.now().strftime("%Y%m%d")
    prefix = f"LOG{today}"
    
    result = db.query(func.max(Order.order_no)).filter(Order.order_no.like(f"{prefix}%")).first()
    max_no = result[0]
    
    if max_no:
        seq = int(max_no[-4:]) + 1
        return f"{prefix}{seq:04d}"
    else:
        return f"{prefix}0001"


def get_operation_logs(db: Session, operation_type: str = None, table_name: str = None, 
                       limit: int = 20, offset: int = 0) -> list:
    from dao.models import OperationLog
    
    query = db.query(OperationLog).order_by(OperationLog.operation_time.desc())
    
    if operation_type:
        query = query.filter(OperationLog.operation_type == operation_type)
    if table_name:
        query = query.filter(OperationLog.table_name == table_name)
    
    return query.offset(offset).limit(limit).all()
