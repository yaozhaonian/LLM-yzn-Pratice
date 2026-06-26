# test_data_source.py
import asyncio
import os
import sys
from dotenv import load_dotenv

# 确保环境与 MCP 服务器一致
load_dotenv()
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ['NO_PROXY'] = '*'

from tools.stock_analyzer_service import StockAnalyzerService
from models import ALI_TONGYI_URL, ALI_TONGYI_API_KEY_OS_VAR_NAME, ALI_TONGYI_PLUS_MODEL

async def main():
    print("初始化 Analyzer...")
    analyzer = StockAnalyzerService(
        custom_api_url=ALI_TONGYI_URL,
        custom_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
        custom_api_model=ALI_TONGYI_PLUS_MODEL
    )
    
    print("尝试获取 600795 数据...")
    try:
        df = await analyzer.data_provider.get_stock_data("600795", "A")
        print("成功！数据形状:", df.shape)
        print(df.head())
    except Exception as e:
        print(f"失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())