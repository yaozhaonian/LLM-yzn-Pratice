-- ================================================
-- ERP智能客服系统 - 数据库初始化脚本
-- 字符集: utf8mb4
-- 引擎: InnoDB
-- ================================================

-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS erp_system DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE erp_system;

-- ================================================
-- 订单表 orders
-- ================================================
CREATE TABLE IF NOT EXISTS orders (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    order_no VARCHAR(64) NOT NULL UNIQUE COMMENT '订单号',
    customer_name VARCHAR(128) NOT NULL COMMENT '客户名称',
    total_amount FLOAT NOT NULL COMMENT '总金额',
    status VARCHAR(32) NOT NULL COMMENT '订单状态',
    shipping_address TEXT COMMENT '收货地址',
    receiver VARCHAR(64) COMMENT '收货人',
    contact_phone VARCHAR(32) COMMENT '联系电话',
    create_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '下单时间',
    update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_order_no (order_no),
    INDEX idx_status (status),
    INDEX idx_customer_name (customer_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='订单表';

-- ================================================
-- 库存表 stock
-- ================================================
CREATE TABLE IF NOT EXISTS stock (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    material_code VARCHAR(64) NOT NULL COMMENT '物料编码',
    product_name VARCHAR(128) NOT NULL COMMENT '商品名称',
    warehouse VARCHAR(64) NOT NULL COMMENT '所属仓库',
    available_stock INT NOT NULL DEFAULT 0 COMMENT '可用库存',
    safety_stock INT NOT NULL DEFAULT 0 COMMENT '安全库存',
    update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_material_code (material_code),
    INDEX idx_warehouse (warehouse),
    INDEX idx_product_name (product_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='库存表';

-- ================================================
-- 生产工单表 production_work_order
-- ================================================
CREATE TABLE IF NOT EXISTS production_work_order (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    work_order_no VARCHAR(64) NOT NULL UNIQUE COMMENT '工单号',
    product_name VARCHAR(128) NOT NULL COMMENT '产品名称',
    current_process VARCHAR(64) COMMENT '当前工序',
    completion_rate FLOAT NOT NULL DEFAULT 0.0 COMMENT '完成率',
    estimated_completion_time DATETIME COMMENT '预计完工时间',
    status VARCHAR(32) NOT NULL COMMENT '工单状态',
    exception_note TEXT COMMENT '异常说明',
    create_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_work_order_no (work_order_no),
    INDEX idx_status (status),
    INDEX idx_product_name (product_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='生产工单表';

-- ================================================
-- 客户货款表 customer_payment
-- ================================================
CREATE TABLE IF NOT EXISTS customer_payment (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    customer_code VARCHAR(64) NOT NULL UNIQUE COMMENT '客户编码',
    customer_name VARCHAR(128) NOT NULL COMMENT '客户名称',
    credit_limit FLOAT NOT NULL DEFAULT 0.0 COMMENT '信用额度',
    available_balance FLOAT NOT NULL DEFAULT 0.0 COMMENT '可用余额',
    outstanding_amount FLOAT NOT NULL DEFAULT 0.0 COMMENT '未回款金额',
    credit_status VARCHAR(32) NOT NULL COMMENT '账期状态',
    update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_customer_code (customer_code),
    INDEX idx_customer_name (customer_name),
    INDEX idx_credit_status (credit_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='客户货款表';

-- ================================================
-- 操作记录表 operation_log
-- ================================================
CREATE TABLE IF NOT EXISTS operation_log (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    operation_type VARCHAR(32) NOT NULL COMMENT '操作类型：INSERT、UPDATE、DELETE',
    table_name VARCHAR(64) NOT NULL COMMENT '操作的表名',
    record_id INT COMMENT '操作记录的ID',
    old_data TEXT COMMENT '操作前的数据（JSON格式）',
    new_data TEXT COMMENT '操作后的数据（JSON格式）',
    operator VARCHAR(64) COMMENT '操作人',
    operation_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '操作时间',
    remark VARCHAR(512) COMMENT '备注',
    INDEX idx_operation_type (operation_type),
    INDEX idx_table_name (table_name),
    INDEX idx_operation_time (operation_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='操作记录表';

-- ================================================
-- 脚本执行完成
-- ================================================
