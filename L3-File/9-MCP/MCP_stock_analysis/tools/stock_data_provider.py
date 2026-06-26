# stock_data_provider.py
import sys
import os

# 【修复】确保项目根目录在 sys.path 中，以便能导入 utils
# 假设当前文件位于 tools/ 目录下，其父目录即为项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # 即 MCP_stock_analysis 目录
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import requests
import pandas as pd
import io
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Optional, Tuple, Any

# 【修复】安全导入 logger，如果失败则使用标准 logging 作为后备
try:
    from utils.logger import get_logger
    logger = get_logger()
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning("未能导入 utils.logger，已切换至标准 logging。请检查 utils/__init__.py 是否存在或路径配置。")

import time
import requests
from requests.exceptions import ConnectionError, Timeout, ChunkedEncodingError
from http.client import RemoteDisconnected
import akshare as ak
import random
import yfinance as yf


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
    
    def _get_stock_data_sync(self, stock_code: str, market_type: str, 
                             start_date: Optional[str] = None, 
                             end_date: Optional[str] = None):
        """
        同步获取股票数据
        """
        try:
            if market_type == 'A':
                return self._get_a_share_data(stock_code)
            elif market_type == 'HK':
                # 保持原有逻辑或类似实现
                return ak.stock_hk_hist(symbol=stock_code, period="daily", adjust="qfq")
            elif market_type == 'US':
                ticker = yf.Ticker(stock_code)
                # 注意：yfinance 的 history 方法可以使用 start/end 参数，这里简单起见仍用 period
                # 若需精确控制日期，可改为: ticker.history(start=start_date, end=end_date)
                return ticker.history(period="1y")
            else:
                raise ValueError(f"不支持的市场类型: {market_type}")
                
        except Exception as e:
            # 【修复】直接使用模块顶部的 logger，避免重复导入和创建
            logger.error(f"获取数据失败: {type(e).__name__}: {str(e)}")
            raise
        
    def _get_a_share_data(self, stock_code: str):
        """
        使用直接请求的方式获取 A 股数据，绕过 akshare 的反爬问题
        """
        # 确定市场前缀: 600/601/603/605/688/689 -> 1 (上海), 其他 -> 0 (深圳)
        prefix = "1" if stock_code.startswith(('600', '601', '603', '605', '688', '689')) else "0"
        secid = f"{prefix}.{stock_code}"
        
        # 【修复】使用 HTTPS 协议
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        
        params = {
            "secid": secid,
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101",         # 日K线
            "fqt": "1",           # 前复权
            "beg": "0",
            "end": "20500101",
            "lmt": "1000"         # 获取最近 1000 条数据
        }
        
        # 【修复】增强 Headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "Host": "push2his.eastmoney.com",
            "Origin": "https://quote.eastmoney.com",
            "Referer": "https://quote.eastmoney.com/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
        }
        
        # 【修复】增加重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 每次请求前都增加随机延时
                time.sleep(random.uniform(1, 2))
                
                # 【关键修复】禁用代理并关闭 SSL 验证（仅用于排查环境问题）
                response = requests.get(
                    url, 
                    params=params, 
                    headers=headers, 
                    timeout=10,
                    proxies={'http': None, 'https': None}, # 强制直连，不走系统代理
                    verify=False # 临时关闭 SSL 验证，排除证书问题
                )
                response.raise_for_status()
                
                data = response.json()
                
                if not data.get("data") or not data["data"].get("klines"):
                    raise Exception(f"未获取到股票 {stock_code} 的数据")
                    
                klines = data["data"]["klines"]
                
                # 解析数据
                records = []
                for line in klines:
                    parts = line.split(",")
                    if len(parts) >= 11:
                        records.append({
                            "Date": parts[0],
                            "Open": float(parts[1]),
                            "Close": float(parts[2]),
                            "High": float(parts[3]),
                            "Low": float(parts[4]),
                            "Volume": int(parts[5]),
                            "Turnover": float(parts[6]),
                            "Amplitude": float(parts[7]),
                            "Change_pct": float(parts[8]),
                            "Change_val": float(parts[9]),
                            "Turnover_rate": float(parts[10])
                        })
                        
                df = pd.DataFrame(records)
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)
                
                df.rename(columns={
                    "Open": "Open",
                    "Close": "Close",
                    "High": "High",
                    "Low": "Low",
                    "Volume": "Volume"
                }, inplace=True)
                
                df.sort_index(inplace=True)
                
                return df

            except (ConnectionError, Timeout, RemoteDisconnected, ChunkedEncodingError) as e:
                logger.warning(f"第 {attempt + 1} 次尝试获取 {stock_code} 数据失败: {str(e)}")
                if attempt == max_retries - 1:
                    raise Exception(f"多次尝试后仍无法连接: {str(e)}")
            except Exception as e:
                # 非网络错误直接抛出
                raise
            
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
    
if __name__ == "__main__":
    import asyncio
    
    async def main():
        logger.info("开始测试股票数据获取...")
        provider = StockDataProvider()
        
        try:
            # 测试获取 A 股数据 (例如: 贵州茅台 600519)
            logger.info("正在获取 A 股 600519 数据...")
            df_a = await provider.get_stock_data("600519", market_type='A')
            print("-" * 20 + " A 股数据预览 " + "-" * 20)
            print(df_a.head())
            print(f"A 股数据形状: {df_a.shape}")
            
            # 测试获取美股数据 (例如: Apple AAPL)
            logger.info("正在获取美股 AAPL 数据...")
            df_us = await provider.get_stock_data("AAPL", market_type='US')
            print("-" * 20 + " 美股数据预览 " + "-" * 20)
            print(df_us.head())
            print(f"美股数据形状: {df_us.shape}")
            
        except Exception as e:
            logger.error(f"测试过程中发生错误: {e}", exc_info=True)

    # 运行异步主函数
    asyncio.run(main())