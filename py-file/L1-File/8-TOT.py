# 思维树(Tree-of-thought, ToT)
'''
思维树(Tree-of-thought, ToT)
使用场景：
    1. 把问题思路设计为树结构，探索多种推理路径，最终综合选择最优解。
    2. 适用于高度复杂的决策问题，数学证明、编程调试、多路径决策
    3. 缺点：实现复杂，计算开销极大，对模型能力要求高。
'''

import requests
import json

# 基础配置
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"

prompt = """
小明100米跑成绩：10.5秒，1500米跑成绩：3分20秒，铅球成绩：12米。他适合参加哪些搏击运动训练?
请按下面步骤思考：
1.请根据以上成绩，分析候选人在速度、耐力、力量三方面素质的分档。分档包括：强（3），中（2），弱（1）三档
2.根据小明的速度、耐力、力量的分档结果，分别给小明从3个维度来推荐运动
   - 需要根据速度强度来推荐运动有哪些，给出10个例子，
   - 需要根据耐力强度来推荐运动有哪些，给出10个例子，
   - 需要根据力量强度来推荐运动有哪些，给出10个例子。
3.分别分析上面给的10个运动对速度、耐力、力量方面素质的要求: 强（3），中（2），弱（1）
根据上面的分析：生成一篇小明适合那种运动训练的分析报告, 请将思维树的步骤用树形图画出来
"""

def chat_with_ollama(prompt):

    # 构造API请求参数
    payload = {
        "model":MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream":False,
        "temperature":0.7
    }

    try:
        # 发送API请求
        print("开始请求")
        response = requests.post(OLLAMA_API_URL, json=payload,timeout=6000)
        print(f"API响应:{response}")
        print(f"响应状态码:{response.status_code}")
        print(f"响应内容长度:{len(response.text)}")
        
        response.raise_for_status()
        
        # 打印原始响应内容
        print(f"原始响应:{response.text[:500]}...")  # 只打印前 500 字符
        
        result = response.json()
        print(f"解析后的 JSON 键:{result.keys()}")  # 检查 JSON 结构
        
        # 检查 message 字段
        if "message" in result:
            assistant_reply = result["message"].get("content", "")
            print(f"回复内容长度:{len(assistant_reply)}")
            return assistant_reply
        else:
            print(f"⚠️ 响应中缺少 message 字段，完整响应:{result}")
            return ""

        

    except requests.exceptions.Timeout:
        return "⚠️ 模型回复超时（超过6000秒），请换轻量模型或检查硬件性能！"
    except requests.exceptions.HTTPError as e:
        return f"⚠️ HTTP错误：{e}\n响应内容：{response.text if 'response' in locals() else '无'}"
    except requests.exceptions.RequestException as e:
        return f"⚠️ RequestException错误：{e}\n"
    except Exception as e:
        return f"⚠️ 其他错误：{str(e)}"


if __name__ == "__main__":
    while True:
        user_text = input("请输入用户输入：")
        if user_text == "exit":
            break
        print(chat_with_ollama(user_text))
