# 大模型调用工具
import datetime
import webbrowser
from langchain.tools import tool
from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2.5:7b",temperature=0.5,base_url="http://127.0.0.1:11434")

@tool
def get_date():
    """获取时间"""
    return datetime.date.today().strftime("%Y-%m-%d")

@tool
def open_browser(url, browser_name=None):
    """ 获取浏览器,打开网站 """
    if browser_name is None:
        webbrowser.get(browser_name).open(url)
    else:
        webbrowser.open(url)

all_tools = {
    "get_date": get_date, 
    "open_browser": open_browser
}

llm_tool = llm.bind_tools([get_date, open_browser])
# resp = llm_tool.invoke("今天是几月几号？")
resp = llm_tool.invoke("帮我访问淘宝网站？")
print("resp",resp)

"""
content='' additional_kwargs={} response_metadata={'model': 'qwen2.5:7b', 'created_at': '2026-04-07T07:08:53.1829609Z', 'done': True, 'done_reason': 'stop', 'total_duration': 2667785000, 'load_duration': 209340900, 'prompt_eval_count': 177, 'prompt_eval_duration': 236171200, 'eval_count': 25, 'eval_duration': 2193410000, 'logprobs': None, 'model_name': 'qwen2.5:7b', 'model_provider': 'ollama'} id='lc_run--019d66c5-c5d1-7932-b307-38193435bf50-0' 
tool_calls=[{'name': 'open_browser', 'args': {'url': 'https://www.taobao.com'}, 'id': 'b8ebcafd-e6e2-451c-8fa8-8a2fc8f4db49', 'type': 'tool_call'}] invalid_tool_calls=[] usage_metadata={'input_tokens': 177, 'output_tokens': 25, 'total_tokens': 202}
"""

print("="*50)
# 大模型给函数调用的筛选结果，并没有直接调用工具
# 手动执行调用函数的过程
if resp.tool_calls:
    for tool_call in resp.tool_calls:
        tool = tool_call["name"]
        print(tool)
        tool_args = tool_call["args"]
        # 从字典中获取工具函数体
        selected_tool = all_tools.get(tool)
        # 手动执行函数
        result = selected_tool.invoke(tool_args)
        print("result",result)









