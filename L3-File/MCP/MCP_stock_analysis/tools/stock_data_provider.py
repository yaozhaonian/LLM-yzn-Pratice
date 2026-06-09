# stock_data_provider.py

import pandas as pd
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from utils.logger import get_logger
import time
import requests
from requests.exceptions import ConnectionError, Timeout, ChunkedEncodingError
from http.client import RemoteDisconnected
import akshare as ak
import os
import random

# 【修复】添加 HTTPException 导入
try:
    from fastapi import HTTPException
except ImportError:
    # 如果非 FastAPI 环境，定义一个兼容类或改用标准异常
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

# 获取日志器
logger = get_logger()

# 【修复】设置全局 User-Agent，模拟浏览器请求
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive"
}

class StockDataProvider:
    """
    异步股票数据提供服务
    负责获取股票、基金等金融产品的历史数据
    """
    
    def __init__(self):
        """初始化数据提供者服务"""
        logger.debug("初始化StockDataProvider")
        # 【修复】尝试设置 akshare 的请求头（如果 akshare 版本支持）
        # 注意：不同版本的 akshare 设置方式可能不同，这里主要依靠底层的 requests 设置
    
    async def get_stock_data(self, stock_code: str, market_type: str = 'A', 
                            start_date: Optional[str] = None, 
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """
        异步获取股票或基金数据
        
        Args:
            stock_code: 股票代码
            market_type: 市场类型，默认为'A'股
            start_date: 开始日期，格式YYYYMMDD，默认为一年前
            end_date: 结束日期，格式YYYYMMDD，默认为今天
            
        Returns:
            包含历史数据的DataFrame
        """
        # 使用线程池执行同步的akshare调用
        return await asyncio.to_thread(
            self._get_stock_data_sync, 
            stock_code, 
            market_type, 
            start_date, 
            end_date
        )
    
    def _get_stock_data_sync(self, stock_code: str, market_type: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        同步获取股票数据，带重试机制
        
        Args:
            stock_code: 股票代码
            market_type: 市场类型
            start_date: 开始日期 (暂未使用，保留接口一致性)
            end_date: 结束日期 (暂未使用，保留接口一致性)
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 【修复】确保每次请求前彻底清除代理
                os.environ.pop('HTTP_PROXY', None)
                os.environ.pop('HTTPS_PROXY', None)
                os.environ.pop('http_proxy', None)
                os.environ.pop('https_proxy', None)
                
                if market_type == 'A':
                    # 【修复】调用 akshare 时，部分函数允许传递 headers 或 timeout
                    # ak.stock_zh_a_hist 内部使用 requests，我们可以通过 patch 或设置全局 session 来优化
                    # 这里直接调用，但依靠下面的异常处理和重试
                    
                    # 增加随机等待，避免请求过于规律
                    if attempt > 0:
                        wait_time = 2 ** attempt + random.uniform(0, 1)
                        logger.info(f"第 {attempt + 1} 次重试，等待 {wait_time:.2f} 秒...")
                        time.sleep(wait_time)

                    df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq")
                    
                    # 【修复】检查返回数据是否有效
                    if df is None or df.empty:
                        raise ValueError("获取到的数据为空")
                        
                    return df
                
                # ... 其他市场类型 ...

            except (ConnectionError, Timeout, ChunkedEncodingError, RemoteDisconnected) as e:
                logger.warning(f"连接失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # 指数退避 + 随机抖动
                    wait_time = 2 ** attempt + random.uniform(0, 1)
                    time.sleep(wait_time)
                else:
                    logger.error("多次重试后仍无法连接数据源")
                    raise HTTPException(status_code=503, detail="数据源连接失败，请稍后重试")
            
            except ValueError as ve:
                 logger.error(f"数据验证错误: {ve}")
                 raise HTTPException(status_code=404, detail=str(ve))

            except Exception as e:
                logger.error(f"获取数据时发生未知错误: {type(e).__name__}: {e}")
                # 对于非网络错误，通常重试无效，直接抛出
                raise HTTPException(status_code=500, detail=f"获取数据失败: {str(e)}")
            
    async def get_multiple_stocks_data(self, stock_codes: List[str], 
                                     market_type: str = 'A',
                                     start_date: Optional[str] = None, 
                                     end_date: Optional[str] = None,
                                     max_concurrency: int = 5) -> Dict[str, pd.DataFrame]:
        """
        异步批量获取多只股票数据
        
        Args:
            stock_codes: 股票代码列表
            market_type: 市场类型，默认为'A'股
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            max_concurrency: 最大并发数，默认为5
            
        Returns:
            字典，键为股票代码，值为对应的DataFrame
        """
        # 使用信号量控制并发数
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def get_with_semaphore(code):
            async with semaphore:
                try:
                    return code, await self.get_stock_data(code, market_type, start_date, end_date)
                except Exception as e:
                    logger.error(f"获取股票 {code} 数据时出错: {str(e)}")
                    return code, None
        
        # 创建异步任务
        tasks = [get_with_semaphore(code) for code in stock_codes]
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks)
        
        # 构建结果字典，过滤掉失败的请求
        return {code: df for code, df in results if df is not None}