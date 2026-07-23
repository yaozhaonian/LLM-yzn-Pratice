from sqlalchemy import Column, String, Float, Integer, DateTime, Text
from sqlalchemy.sql import func
from dao.db import Base


class Order(Base):
    """
    订单表
    
    存储销售订单信息，包括订单基本信息、客户信息、收货信息等。
    """
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_no = Column(String(64), unique=True, nullable=False, index=True, comment="订单号")
    customer_name = Column(String(128), nullable=False, comment="客户名称")
    material_code = Column(String(64), nullable=False, comment="物料编码")
    quantity = Column(Integer, nullable=False, comment="发货数量")
    product_name = Column(String(128), comment="产品名称")
    total_amount = Column(Float, nullable=False, comment="总金额")
    status = Column(String(32), nullable=False, comment="订单状态")
    logistics_no = Column(String(64), comment="物流编码")
    shipping_address = Column(Text, comment="收货地址")
    receiver = Column(String(64), comment="收货人")
    contact_phone = Column(String(32), comment="联系电话")
    create_time = Column(DateTime, default=func.now(), nullable=False, comment="下单时间")
    update_time = Column(DateTime, default=func.now(), onupdate=func.now(), comment="更新时间")


class Stock(Base):
    """
    库存表
    
    存储商品库存信息，包括物料编码、仓库信息、库存数量等。
    """
    __tablename__ = "stock"

    id = Column(Integer, primary_key=True, autoincrement=True)
    material_code = Column(String(64), nullable=False, index=True, comment="物料编码")
    product_name = Column(String(128), nullable=False, comment="商品名称")
    warehouse = Column(String(64), nullable=False, index=True, comment="所属仓库")
    available_stock = Column(Integer, nullable=False, default=0, comment="可用库存")
    safety_stock = Column(Integer, nullable=False, default=0, comment="安全库存")
    unit_price = Column(Float, nullable=False, default=0.0, comment="单价")
    update_time = Column(DateTime, default=func.now(), onupdate=func.now(), comment="更新时间")


class ProductionWorkOrder(Base):
    """
    生产工单表
    
    存储生产工单信息，包括工单进度、工序状态、预计完工时间等。
    """
    __tablename__ = "production_work_order"

    id = Column(Integer, primary_key=True, autoincrement=True)
    work_order_no = Column(String(64), unique=True, nullable=False, index=True, comment="工单号")
    product_name = Column(String(128), nullable=False, comment="产品名称")
    current_process = Column(String(64), comment="当前工序")
    completion_rate = Column(Float, nullable=False, default=0.0, comment="完成率")
    estimated_completion_time = Column(DateTime, comment="预计完工时间")
    status = Column(String(32), nullable=False, comment="工单状态")
    exception_note = Column(Text, comment="异常说明")
    create_time = Column(DateTime, default=func.now(), nullable=False, comment="创建时间")
    update_time = Column(DateTime, default=func.now(), onupdate=func.now(), comment="更新时间")


class CustomerPayment(Base):
    """
    客户货款表
    
    存储客户信用和货款信息，包括信用额度、可用余额、未回款金额等。
    """
    __tablename__ = "customer_payment"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_code = Column(String(64), unique=True, nullable=False, index=True, comment="客户编码")
    customer_name = Column(String(128), nullable=False, comment="客户名称")
    credit_limit = Column(Float, nullable=False, default=0.0, comment="信用额度")
    available_balance = Column(Float, nullable=False, default=0.0, comment="可用余额")
    outstanding_amount = Column(Float, nullable=False, default=0.0, comment="未回款金额")
    credit_status = Column(String(32), nullable=False, comment="账期状态")
    update_time = Column(DateTime, default=func.now(), onupdate=func.now(), comment="更新时间")


class OperationLog(Base):
    """
    操作记录表
    
    记录数据库的增、删、改操作，用于审计和追踪。
    该表只能增加记录和查询，不允许修改和删除。
    """
    __tablename__ = "operation_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    operation_type = Column(String(32), nullable=False, comment="操作类型：INSERT、UPDATE、DELETE")
    table_name = Column(String(64), nullable=False, comment="操作的表名")
    record_id = Column(Integer, comment="操作记录的ID")
    old_data = Column(Text, comment="操作前的数据（JSON格式）")
    new_data = Column(Text, comment="操作后的数据（JSON格式）")
    operator = Column(String(64), comment="操作人")
    operation_time = Column(DateTime, default=func.now(), nullable=False, comment="操作时间")
    remark = Column(String(512), comment="备注")
