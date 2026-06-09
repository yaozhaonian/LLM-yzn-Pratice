import asyncio
import time
import httpx

"""
适合异步的场景：

数据库操作
网络请求（API 调用）
文件 I/O 操作
长时间等待的操作
不适合异步的场景：

CPU 密集型计算
简单的数据处理
没有 I/O 等待的操作
"""


# 同步方式 - 阻塞执行
def sync_fetch_data():
    start_time = time.time()
    # 模拟三个网络请求
    time.sleep(1)  # 第一个请求
    time.sleep(1)  # 第二个请求
    time.sleep(1)  # 第三个请求
    print(f"同步执行耗时: {time.time() - start_time:.2f}秒")  # 约3秒

# 异步方式 - 并发执行
async def async_fetch_data():
    start_time = time.time()
    # 三个请求并发执行
    await asyncio.gather(
        asyncio.sleep(1),  # 第一个请求
        asyncio.sleep(1),  # 第二个请求
        asyncio.sleep(1),  # 第三个请求
    )
    print(f"异步执行耗时: {time.time() - start_time:.2f}秒")  # 约1秒

# 运行示例
sync_fetch_data()  # 输出: 同步执行耗时: 3.00秒
asyncio.run(async_fetch_data())  # 输出: 异步执行耗时: 1.00秒