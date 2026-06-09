# 思维链（COT）
'''
思维链（COT）
使用场景：
    1. 引导模型像人类一样逐步推理，分解问题，再得出结论，
    2. 适用于复杂推理任务：数学应用题、逻辑推理、多步骤问题
    3. 与少样本结合：提供带推理步骤的示例(Few-Shot CoT)。
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
        temperature=0,
    )
    return response.choices[0].message.content

# 数学应用题（分步计算，CoT论文示例:https://arxiv.org/pdf/2201.11903）
prompt = """
Q：罗杰有5个网球。他又买了2罐网球。每个罐子有3个网球。他现在有多少个网球?
A：罗杰一开始有5个球。2罐3个网球，等于6个网球。5 + 6 = 11。答案是11。
Q：自助餐厅有23个苹果。如果他们用20做午餐，又买了6个，他们有多少个苹果?
A：
"""

# 逻辑推理（时间排序）
prompt2 = '''
问题：如果A比B早到，C比A晚到但比B早到，三人到达的顺序是什么？  
请分步思考：  
1. A比B早到，思考A与B的顺序 
2. C比A晚到但比B早到，思考C、A、B的顺序  
最终顺序是： 
'''

# 任务执行类（短视频文案创作）
prompt3 = '''
# 角色
你是爆款短视频剧本创作大师，你的目标任务是根据一个给定的产品或主题，创作出极具吸引力、节奏感强、易于拍摄且能引发病毒式传播的短视频剧本。

## 工作步骤
1. 定位与创意构思：明确视频目标与核心创意点。（100字内）
2. 剧本结构与细节填充：设计故事框架并撰写具体内容。（100字内）
3. 拍摄可行性审核与优化：确保剧本可执行并提出优化建议。（100字内）

# 任务
便携榨汁杯短视频剧本创作
'''

print(get_completion(prompt))
print('=' * 50)
print(get_completion(prompt2))
print('=' * 50)
print(get_completion(prompt3))


