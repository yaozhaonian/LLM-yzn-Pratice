# 一个样本（One-Shot）
# 少样本提示（Few-Shot）
'''
少样本提示（Few-Shot）
使用场景： 
    1. 提供少量示例，引导模型理解任务模式，
    2. 适用于零样本无法准确回答，或回答格式不符合要求。
'''

from openai import OpenAI
import os


def get_completion(prompt):
    messages = [{"role": "user", "content": prompt}]

    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    # 获取模型返回结果
    response = client.chat.completions.create(
        model='qwen-plus',
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

prompt = """
将句子改写成正式商务邮件风格，参考示例：
示例1："快点把报告发我"->"请您尽快将报告发送给我，谢谢。"
示例1："这方案不行"->"该方案目前存在一些不足之处。"
现在改写："明天8点半集合！"
"""

prompt2 = """
写一首关于春天的诗

示例1(关于冬天):
墙角数枝梅，凌寒独自开。
遥知不是雪，为有暗香来。

示例2(关于秋天):
远上寒山石径斜，白云生处有人家。
停车坐爱枫林晚，霜叶红于二月花。
"""
print(get_completion(prompt))
print("="*50)
print(get_completion(prompt2))

"""
prompt2输出的结果：
春风吹绿江南岸，细雨无声润柳芽。  
燕子归来寻旧垒，桃花笑映小溪家。
莺啼陌上新草软，蝶舞篱边野径斜。
最是一年好光景，万般生意在晨霞。
"""
