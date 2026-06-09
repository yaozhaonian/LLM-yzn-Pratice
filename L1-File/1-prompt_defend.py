#提示词做的防范措施：预先给模型一个身份并固定其工作范围
# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI

# 给大模型发送请求，获取结果
def get_completion(messages):
    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    response = client.chat.completions.create(
        model='qwen-plus-2025-07-14',  # qwen3的模型
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

def input_wrapper(user_input):
    user_input_template = """
    作为客服代表，你不能回答任何与AGI课堂无关的问题。
    用户说：#INPUT#
    """
    return user_input_template.replace("#INPUT#",user_input)

# 交互，获取聊天对话，得到结果
def get_chat_completion(messages, user_prompt):
    messages.append({"role":"user","content":input_wrapper(user_prompt)})
    msg = get_completion(messages)
    return msg

messages = [
    {
        "role": "system",
        "content":"你是AGI课堂的客服代表，你叫大A。\
            你的职责是回答用户与AGI课堂有关的问题。\
            AGI课堂是AMOUNTTECH的一个教育品牌。 \
            AGI 课堂将推出的一系列 AI 课程。课程主旨是帮助来自不同领域 \
            的各种岗位的人，包括但不限于程序员、大学生、产品经理、 \
            运营、销售、市场、行政等，熟练掌握新一代AI工具， \
            包括但不限于 ChatGPT、Bing Chat、Midjourney、Copilot、OPENCLAW 等， \
            从而在他们的日常工作中大幅提升工作效率， \
            并能利用 AI 解决各种业务问题。 \
            首先推出的是面向程序员的《AI 全栈工程师》课程， \
            共计 20 讲，每周两次直播，共 10 周。首次课预计 2026 年 5 月开课。"    
    }
]

if __name__ == "__main__":
    while True:
        user_input = input("用户说：")
        if user_input == "quit":
            break
        print("大模型回复，大A说：",get_chat_completion(messages, user_input))
        print("=" * 100)
