# test_manual_request.py
import requests
import os

# 清除代理
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ['NO_PROXY'] = '*'

# 东方财富的历史数据接口 (akshare 底层也是用这个)
# 注意：接口可能会变，这是当前常用的一个
url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"

params = {
    "secid": "1.600795",  # 1. 代表上海交易所, 600795 是代码
    "fields1": "f1,f2,f3,f4,f5,f6",
    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    "klt": "101",         # 日K线
    "fqt": "1",           # 前复权
    "beg": "0",
    "end": "20500101",
    "lmt": "100"          # 限制返回条数
}

# 模拟浏览器的 Headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Referer": "http://quote.eastmoney.com/",
}

print("正在尝试手动请求东方财富接口...")
try:
    response = requests.get(url, params=params, headers=headers, timeout=10)
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        if data.get("data") and data["data"].get("klines"):
            print("✅ 成功！获取到数据。")
            print("第一条数据:", data["data"]["klines"][0])
        else:
            print("❌ 失败: 返回JSON中没有数据。", data)
    else:
        print(f"❌ 失败: HTTP {response.status_code}")
        print(response.text[:200])
        
except Exception as e:
    print(f"❌ 异常: {type(e).__name__}: {e}")