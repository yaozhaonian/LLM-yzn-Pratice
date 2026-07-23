import logging
import sys
from config.settings import settings


def get_logger(name: str) -> logging.Logger:
    """
    获取统一格式的日志记录器
    
    日志格式：时间 - 日志级别 - 模块名 - 内容
    同时输出到控制台和日志文件
    
    Args:
        name: 日志记录器名称，通常使用 __name__
    
    Returns:
        logging.Logger: 配置好的日志记录器实例
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_level = getattr(logging, settings.log.level.upper(), logging.INFO)
    logger.setLevel(log_level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(settings.log.file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
