# 自我一致性(自洽性，Self-Consistency)
'''
自我一致性(自洽性，Self-Consistency)
使用场景：
    1. 生成多个推理路径或答案，选择最一致或最频繁出现的结果，减少随机性
    2. 适用于高可靠性要求任务：如科学计算、法律判断、医疗诊断。
    3. 可以与思维链结合：生成多条CoT路径，选择最优解
    4. 缺点：响应时间增加，计算成本更高。
'''

from openai import OpenAI
import os
from collections import Counter


def get_completion(prompt):
    messages = [{"role": "user", "content": prompt}]

    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    # 获取模型返回结果
    response = client.chat.completions.create(
        model='qwen-plus-2025-07-28',
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

prompt = """
现在我70岁了，当我16岁时，我的二弟是我的年龄的一半，我的三弟比我的二弟小5岁。现在我的三弟多大？请逐步思考，并将答案写在括号中《》
"""

# 请求多次:n = 3
def get_multiple_completions(prompt, n=3):
    responses = []
    for _ in range(n):
        responses.append(get_completion(prompt).strip())
    return responses

def majority_vote(responses):
    print('Vote Beginning(开始投票):')
    # res.rsplit('《')[-1].split('》')[0]   从右边开始往左搜寻第一个'《',再从左往右搜寻第一个'》',也就是有多个'《》'时会取最后一个
    counter = Counter([res.rsplit('《')[-1].split('》')[0] for res in responses])
    print('查看counter值:',counter)
    print('查看counter.most_common(1)值:',counter.most_common(1))
    # 第二个值是次数，我们这里不需要，所以取第一个值
    #  most_common(1):包含前1个最常见的元素及其出现次数
    most_common_answer = counter.most_common(1)[0]
    print("投票结果:",most_common_answer)
    return most_common_answer

def get_completion_with_self_consistency(prompt, n=5):
    responses = get_multiple_completions(prompt, n)

    print('多轮推理开始(大模型回复):')
    for i , response in enumerate(responses):
        print(f'第{i+1}轮推理过程:',response)
        print('='*50)

    final_answer = majority_vote(responses)
    return final_answer

print('查看推理结果',get_completion_with_self_consistency(prompt))
# print(get_completion(prompt))

