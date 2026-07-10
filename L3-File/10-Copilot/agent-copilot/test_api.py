#!/usr/bin/env python3
import urllib.request
import json
import time

# 发送查询请求
url = "http://127.0.0.1:5005/api_planning"
payload = {
    "query": "查询苹果的产品信息",
    "contexts": [{"content": "查询苹果的产品信息", "role": "user"}],
    "isCopilot": True,
    "isContext": True,
    "contextNumber": 1
}

print("=== 发送查询请求 ===")
print(f"URL: {url}")
print(f"Query: {payload['query']}\n")

try:
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp_data = json.loads(resp.read().decode('utf-8'))
        print(f"Response status: {resp.status}")
        print(f"Response body: {json.dumps(resp_data, indent=2, ensure_ascii=False)[:500]}")
        task_id = resp_data.get('task_id')
    print(f"\n✓ Task ID: {task_id}\n")
    
    if task_id:
        print("=== 检查任务状态 ===")
        time.sleep(8)  # 等待任务处理
        status_url = f"http://127.0.0.1:5005/api_task_status"
        status_req = urllib.request.Request(
            url=status_url,
            data=json.dumps({"task_id": task_id}).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(status_req, timeout=10) as status_resp:
            data = json.loads(status_resp.read().decode('utf-8'))
        print(f"Task status: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
