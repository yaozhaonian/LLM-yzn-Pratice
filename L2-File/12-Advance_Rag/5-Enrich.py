# Enrich完善问题:通过大模型多次主动与用户沟通，不断收集信息，完善对用户真实意图的理解，补全执行用户需求所需的各项参数。

# LangChain 组件
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

import json

embedding = OllamaEmbeddings(model="bge-m3:latest")
llm = ChatOllama(model='qwen2.5:7b', temperature=0.1)


# 用户的需求
user_input = "我想订一张去昆明的机票"

# 首先根据用户的需求，进行意图识别，获取对应的模板
# 示例业务模板
templates = {
    "订机票": ["起点", "终点", "时间", "座位等级", "座位偏好"],
    "订酒店": ["城市", "入住日期", "退房日期", "房型", "人数"],
}

# ======================
# 1. 意图识别
# ======================
# 意图识别的提示模版
intent_prompt = PromptTemplate.from_template(
    """
    根据用户输入的'{user_input}',选择最合适的业务模版。
    可用模版如下:
    {templates}.
    请仅返回模版名称，不要包含其他内容。
"""
)

# 创建意图识别链
intent_chain = intent_prompt | llm

# 识别意图
intent = intent_chain.invoke({"user_input": user_input, "templates": str(list(templates.keys()))}).content.strip()

print("意图：\n", intent)
# 获取对应模版
selected_template = templates.get(intent)
if not selected_template:
    print("无法识别到有效意图")
    exit()
print("模板：\n", selected_template)

# ======================
# 2. 信息补全循环
# ======================
# 手动维护简单的对话历史
conversation_history = [
    HumanMessage(content=user_input)
]

# 定义信息补全的prompt,将历史对话和当前任务结合
info_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """你是一个信息收集助手。你的任务是根据给定的【所需字段模板】，检查【对话历史】中是否包含了所有必要信息。
    
        输出必须严格遵循以下 JSON 格式，不要输出任何其他文字、解释或 Markdown 标记：
        {{
            "isComplete": boolean, // true 如果所有字段都已明确提供，否则 false
            "content": string // 如果 isComplete 为 true，这里是总结出的完整信息字符串；如果为 false，这里是一句友好的话，询问缺失的具体字段
        }}
    
        【所需字段模板】: {template_fields}
        """),
        ("placeholder", "{chat_history}"),
        ("human", "请分析以上对话，判断信息是否完整。")
    ]
)

json_parser = JsonOutputParser()
# 补充信息链
info_chain = info_prompt_template | llm | json_parser

def get_missing_info_request(history, template_fields):
    try:
        response = info_chain.invoke({
            "chat_history": history,
            "template_fields": ",".join(template_fields)
        })
        return response
    except Exception as e:
        print(f"\033[1;31m[解析错误] {e}\033[0m")
        # 如果解析失败，返回一个默认的追问，或者让 LLM 重试
        return {"isComplete": False, "content": "抱歉，我刚才没听清。能请您再重复一下吗？或者提供更多细节？"}

# 初始检查
json_data = get_missing_info_request(conversation_history, selected_template)
print("初始分析结果：", json_data)

# 循环判断是否完整,并提交用户补充信息
while not json_data.get('isComplete', False):
    # 显示引导信息
    question = json_data.get('content', '请补充缺失的信息。')
    user_answer = input(f"\033[1;33mAI:{question}]\n\033[1;34m用户: ]")

    # 将用户回答加入历史
    conversation_history.append(HumanMessage(content=user_answer))

    # 再次检查
    json_data = get_missing_info_request(conversation_history, selected_template)
    print("最新分析结果：", json_data)

# 输出最终结果
print(f"\n\033[1;32m[最终查询信息已收集完成]\033[0m")
print(f"结构化数据: {json.dumps(json_data, ensure_ascii=False, indent=2)}")


"""
意图：
 订机票
模板：
 ['起点', '终点', '时间', '座位等级', '座位偏好']
初始分析结果： {'isComplete': False, 'content': '请问您的出发地是哪里？您计划什么时候出行？您对座位有什么特殊要求或偏好吗？另外，您希望乘坐哪个航班等级的座位呢？'}
AI:请问您的出发地是哪里？您计划什么时候出行？您对座位有什么特殊要求或偏好吗？另外，您希望乘坐哪个航班等级的座位呢？]
最新分析结果： {'isComplete': False, 'content': '请问您的出发地点、希望出行的时间、对座位有特殊要求吗？以及您希望的座位等级是什么？'}
AI:请问您的出发地点、希望出行的时间、对座位有特殊要求吗？以及您希望的座位等级是什么？]计划五一在广州出发，二等座靠窗即可
最新分析结果： {'isComplete': False, 'content': '请问您的出行时间是具体的哪一天呢？'}
AI:请问您的出行时间是具体的哪一天呢？]5月1日
最新分析结果： {'isComplete': False, 'content': '起点、终点、时间、座位等级和座位偏好都已经明确，但缺少具体的座位偏好说明。'}
AI:起点、终点、时间、座位等级和座位偏好都已经明确，但缺少具体的座位偏好说明。]座位偏好是靠窗呀笨蛋
最新分析结果： {'isComplete': False, 'content': '起点和终点已明确，但时间与座位等级未提及'}
AI:起点和终点已明确，但时间与座位等级未提及]5月1日出发
最新分析结果： {'isComplete': True, 'content': '起点为广州，终点为昆明，时间是5月1日，座位等级为二等座，座位偏好为靠窗。'}

[最终查询信息已收集完成]
结构化数据: {
  "isComplete": true,
  "content": "起点为广州，终点为昆明，时间是5月1日，座位等级为二等座，座位偏好为靠窗。"
}
# 可以发现该版本中AI实际上是已经有历史记录了，不过有时还是说没有相关信息，显得有点呆,新版本是5(1)-Enrich_optimize.py
"""