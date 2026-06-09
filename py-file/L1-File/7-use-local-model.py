"""
使用本地模型qwen3.5:2b
看个人电脑配置
接下来尽量用本地模型
"""

import requests
import json

# 基础配置
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"

# 1. 定义角色设定（核心：system字段）
system_prompt = """
你是一名经验丰富的“产品经理”，擅长从用户需求角度提出清晰、可落地的产品建议。
用户的身份是“软件开发工程师”，只关注技术实现难度和可行性，沟通风格直接。
请始终从产品经理的身份出发，回应工程师的问题，兼顾需求合理性和技术落地性，不要偏离角色。
"""

# 2. 初始化对话上下文(保存历史信息，事先多轮对话)
conversation_history = [
    {
        "role":"system",
        "content":system_prompt
    }
]

def chat_with_ollama(user_input):
    """
    发送用户消息，获取模型回复，并更新上下文
    :param user_input: 用户输入的问题/对话内容
    :return: 模型的回复内容
    """
    # 将用户信息加入上下文(角色为user)
    conversation_history.append({
        "role":"user",
        "content":user_input
    })

    # 构造API请求参数
    payload = {
        "model":MODEL_NAME,
        "messages":conversation_history,
        "stream":False,
        "temperature":0.7
    }
    
    try:
        # 发送API请求
        response = requests.post(OLLAMA_API_URL, json=payload,timeout=900)
        print(f"API响应:{response.text}")
        # response.raise_for_status() # 捕获HTTP错误
        # response_json = response.json()

        # 解析回复
        result = response.json()
        print(f"原始响应:{json.dumps(result, ensure_ascii = False)}")     #打印原始响应
        print(f"模型回复：{result['message']['content']}")
        assistant_repply = result["message"]["content"]

        # 将模型回复加入上下文(角色为assistant)
        conversation_history.append({
            "role":"assistant",
            "content":assistant_repply
        })

        return assistant_repply

    except requests.exceptions.Timeout:
        return "⚠️ 模型回复超时（超过900秒），请换轻量模型或检查硬件性能！"
    except requests.exceptions.HTTPError as e:
        return f"⚠️ HTTP错误：{e}\n响应内容：{response.text if 'response' in locals() else '无'}"
    except requests.exceptions.RequestException as e:
        return f"⚠️ RequestException错误：{e}\n"
    except Exception as e:
        return f"⚠️ 其他错误：{str(e)}"

# 3. 开始聊天
if __name__ == "__main__":
    print("=== 角色对话测试（产品经理 ↔ 开发工程师）===")
    print("输入 '退出' 结束对话\n")

    while True:
        user_msg = input("[开发工程师]:")
        if user_msg == "退出" or user_msg == "exit":
            break
        print('真心累')
        reply = chat_with_ollama(user_msg)
        print(f"[产品经理]:{reply}")




