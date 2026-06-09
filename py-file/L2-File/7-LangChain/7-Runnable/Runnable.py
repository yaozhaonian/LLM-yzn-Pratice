# 关于管道langchain_core.runnables的一些函数及用法
import langchain
from langchain_core.runnables import chain, RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableMap
from langchain_core.output_parsers import StrOutputParser
output = StrOutputParser()
from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2.5:7b",temperature=0.9)
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter

""" RunnableLambda的使用
    作用是把任意 Python 函数 / 匿名函数包装成标准 Runnable 接口，让你能把自定义逻辑无缝接入 LangChain 链
    （| 管道），和 ChatPromptTemplate、ChatOpenAI、StrOutputParser 等组件统一调用、组合。
"""
#获得字符串的长度
def length_function(text):
    return len(text)


#将两个字符串长度的数量相乘
def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)

# 包装函数：接收字典，解包调用
def multiple_length_function(_dict):
    return _multiple_length_function(_dict["text1"], _dict["text2"])

# 或者：@chain是RunnableLambda的另一种写法
@chain
def mlf(_dict):
    return _multiple_length_function(_dict["text1"], _dict["text2"])

# chat_template = ChatPromptTemplate.from_template("{a}是多少,{b}是多少, {a} + {b}是多少？")
# chain1 = ({
#     "a": ({"text1":itemgetter("wa"),"text2":itemgetter("wab")} | RunnableLambda(multiple_length_function)),
#     "b": ({"text1":itemgetter("wab"), "text2":itemgetter("wb")} | mlf)
# } | chat_template | llm | output
# )
# print(chain1.invoke({"wa": "hello", "wab": "world", "wb":"runnables"}))


""" RunnableParallel、RunnableMap(并行可运行对象)
RunnableParallel = 同时跑多个任务 → 一次性拿回所有结果
用来提速、并行处理数据、同时调用多个 LLM / 工具 / 函数。
是 LangChain 中实现并发执行的最核心组件。
"""
"""
RunnableParallel 什么都能并行：
LLM 调用
Prompt 模板
RunnableLambda
数据库查询
API 请求
工具调用
其他链
"""
def add_one(x: int) -> int:
    return x + 1

def mul_two(x: int) -> int:
    return x * 2

def mul_three(x: int) -> int:
    return x * 3

#  通过RunnableLambda包装后才能在链中使用
runnable_1 = RunnableLambda(add_one)
runnable_2 = RunnableLambda(mul_two)
runnable_3 = RunnableLambda(mul_three)

# chain2 = runnable_1 | RunnableParallel(
#     a=runnable_2, 
#     b=runnable_3
# )
# print(chain2.invoke(2))

# RunnableParallel与RunnableMap 最常用的场景如下:
prompt1 = ChatPromptTemplate.from_template("总结文本:{text},要求:少于300字")
prompt2 = ChatPromptTemplate.from_template("翻译为英文文本:{text}")
prompt3 = ChatPromptTemplate.from_template("提取关键词:{text}")


# 并行执行三个 LLM 调用！
chain3 = RunnableMap(
    summary=prompt1 | llm | output,
    translate=prompt2 | llm | output,
    keywords=prompt3 | llm | output
)

# 一次性拿到三个结果！
# result = chain3.invoke({"text": """
# 心源性猝死预防：抓住最关键的几件事
# 心源性猝死大多由恶性心律失常、急性心梗、严重心肌病引发，预防核心是：筛查高危人群 + 控制基础病 + 改掉致命习惯 + 学会急救。

# 一、先排查：你是不是高危人群
# 符合以下任意一项，风险明显升高：
# • 有冠心病、心梗、心绞痛病史
# • 有心衰、心肌病、严重心律失常
# • 家族中有人年轻时猝死/突发心脏病
# • 高血压、高血脂、糖尿病、肥胖
# • 长期熬夜、酗酒、大量吸烟、过度劳累
# • 既往体检发现：早搏过多、QT间期异常、心室肥厚

# 建议：高危人群每年做心电图 + 心脏彩超，必要时做24小时动态心电图、冠脉CT/造影。

# 二、日常最有效的预防措施
# 控制“三高”，坚持吃药
# • 高血压：血压尽量控制在 130/80mmHg 以下
# • 高血脂：尤其低密度脂蛋白LDL，冠心病患者需严格达标
# • 糖尿病：控糖、控体重，减少心血管并发症
# 不要擅自停药、减药，很多猝死就发生在自行停药后。

# 绝对戒掉/减少这些行为
# • 戒烟：吸烟是心梗、猝死的强危险因素
# • 限酒：最好不喝，尤其避免暴饮、高度酒
# • 不熬夜：尽量 23点前睡，长期睡眠不足直接诱发恶性心律失常
# • 避免极度劳累 + 情绪暴怒叠加

# 运动要科学，别“玩命练”
# • 久坐不行，但突然剧烈运动更危险
# • 推荐：快走、慢跑、骑行、游泳等中等强度运动
# • 运动中出现胸闷、胸痛、心慌、头晕、出冷汗，立刻停止就医
# • 有心脏病史者，运动前最好让医生评估

# 饮食与体重
# • 少盐、少油、少加工肉
# • 多蔬菜、水果、全谷物、鱼类
# • 控制体重，避免腹型肥胖

# 情绪与压力管理
# 长期高压、焦虑、熬夜、暴怒，是年轻人猝死的常见诱因。
# 学会放松，避免突然情绪大起大落。

# 三、出现这些信号，立刻就医（救命预警）
# 出现以下任一情况，不要扛、不要拖：
# • 突发胸闷、胸痛、压榨感，持续>10分钟不缓解
# • 心慌、心跳特别快/乱，伴头晕、眼前发黑
# • 不明原因大汗、呼吸困难、恶心呕吐
# • 活动后明显乏力、气短，休息也不缓解
# • 夜间憋醒、不能平躺

# 这些可能是心梗、恶性心律失常前兆，及时治疗可大幅降低猝死风险。

# 四、学会急救：关键时刻能救命
# • 身边有人突然倒地、呼之不应、没呼吸/濒死喘息：
#   1. 立即拨打 120
#   2. 马上开始 胸外按压
#   3. 有条件尽快使用 AED（自动体外除颤器）

# """})
# import json
# json_str = json.dumps(result, indent=2, ensure_ascii=False)
# print(json_str)

"""
RunnablePassthrough:可运行的直通器 / 透传器
功能：
    输入是什么 → 输出就是什么
    不修改、不处理、不计算
    专门用来在链里占位、衔接、保留原始输入
"""
# RunnablePassthrough原样进行数据传递
runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    modified=lambda x: x["num"] + 1,
)
# {'passed': {'num': 1}, 'modified': 2}
print(runnable.invoke({"num": 1}))

# RunnablePassthrough对数据增强后传递
# RunnablePassthrough().assign它会创建一个新的字典，包含原始的所有字段以及你新指定的字段。
runnable = RunnableParallel(
    passed=RunnablePassthrough().assign(query=lambda x: x["num"] + 2),
    modified=lambda x: x["num"] + 1,
)
print(runnable.invoke({"num": 1}))
# 以下功能偏鸡肋
next_runnable = runnable | RunnableParallel(
    new_passed = RunnablePassthrough().assign(new_query=lambda runnable: runnable["passed"]["num"] + 5),
    new_modified = lambda runnable: runnable["modified"] + 1
)
result2 = next_runnable.invoke({"num": 1})
print(repr(result2))