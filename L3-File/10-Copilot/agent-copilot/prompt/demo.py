API_CONTEXT_DESC = """
    SubTask{index}: {task_description}
    PI{index}: {api_description}
    PI{index} Response: {api_response}
"""

context = [
    {"task_description": "查天气", "tool": "weather_api", "result": "晴，28℃"},
    {"task_description": "查股价", "tool": "stock_api", "result": "12.5元"}
]        
index =1 
apis = ""
for tmp in context:
    api = API_CONTEXT_DESC.format(
        index=str(index),
        task_description=tmp["task_description"],
        api_description=tmp["tool"],
        api_response=tmp["result"]
    )
    apis += api
    index += 2

print(apis)