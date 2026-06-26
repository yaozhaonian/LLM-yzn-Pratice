import logging
import time
# 通用函数执行耗时统计装饰器
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"{func.__name__} took {execution_time:.4f} seconds to execute.")
        return result

    return wrapper

