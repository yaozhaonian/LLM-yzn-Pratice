# Enrich完善问题: 通过显式槽位提取解决上下文遗忘问题

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import json

# 初始化
llm = ChatOllama(model='qwen2.5:7b', temperature=0,base_url="http://127.0.0.1:11434") # 温度设为0以提高稳定性

user_input = "我想订一张去昆明的机票去那玩3天"

templates = {
    "订机票": ["起点", "终点", "出发时间", "座位等级", "座位偏好"],
    "订酒店": ["城市", "入住日期", "退房日期", "房型", "人数"],
}

# ======================
# 1. 意图识别
# ======================
intent_prompt = ChatPromptTemplate.from_template(
    "根据用户输入'{user_input}'，从以下选项中选择最合适的业务模版名称，仅返回名称：{templates}"
)
intent_chain = intent_prompt | llm
intent = intent_chain.invoke({"user_input": user_input, "templates": str(list(templates.keys()))}).content.strip()
print(f"意图: {intent}")

selected_fields = templates.get(intent)
if not selected_fields:
    print("无法识别意图")
    exit()
print("selected_fields:\n",selected_fields) 
# ======================
# 2. 动态构建槽位提取链
# ======================
slot_extraction_prompt_simple = ChatPromptTemplate.from_messages([
    ("system", """你是一个严格的信息提取助手。
    请分析【对话历史】，提取以下字段的信息。
    
    **字段列表**: {field_list}
    
    **输出要求**:
    1. 输出一个 JSON 对象。
    2. JSON 的 Key 必须是上述【字段列表】中的名称。
    3. 如果某字段已明确提及，Value 为提取到的字符串。
    4. 如果某字段未提及，Value 为 null。
    5. 不要输出任何额外文字。
    
    **示例**:
    如果字段列表是 ["城市", "天数"]，且用户说"去北京玩3天"，
    输出: {{"城市": "北京", "天数": "3"}}
    如果用户只说"去北京"，
    输出: {{"城市": "北京", "天数": null}}
    """),
    ("placeholder", "{chat_history}"),
    ("human", "请提取信息。")
])

parser = JsonOutputParser()
slot_chain = slot_extraction_prompt_simple | llm | parser

def extract_slots(history, fields):
    try:
        response = slot_chain.invoke({
            "chat_history": history,
            "field_list": ", ".join(fields) # 动态传入当前业务的字段，比如订机票就是"起点, 终点, 出发时间, 座位等级, 座位偏好"
        })
        return response
    except Exception as e:
        print(f"\033[1;31m[提取错误] {e}\033[0m")
        return {f: None for f in fields}

# ======================
# 3. 主循环
# ======================

conversation_history = [HumanMessage(content=user_input)]

# 初始化槽位
current_slots = {field: None for field in selected_fields}

print("\n开始信息收集...")

while True:
    # 1. 从历史中提取最新槽位状态
    extracted = extract_slots(conversation_history, selected_fields)
    print(f"\r\033[90m[调试] 当前提取状态: {extracted}\033[0m")
    
    # 2. 更新本地槽位状态 (用新提取的非空值覆盖旧值)
    updated = False
    for key in selected_fields:
        if key in extracted and extracted[key] is not None:
            if current_slots[key] != extracted[key]:
                current_slots[key] = extracted[key]
                updated = True
    
    # 3. 检查是否所有字段都已填充
    missing_fields = [k for k, v in current_slots.items() if v is None]
    
    if not missing_fields:
        break
        
    # 4. 生成追问
    # 为了让追问更自然，我们可以让 LLM 根据缺失字段生成一句话，
    # 或者简单地拼接缺失字段。这里为了稳定，让 LLM 生成追问，但基于明确的缺失列表。
    
    follow_up_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的助手。用户正在预订服务。"),
        ("human", """
        目前已收集信息：{current_info}
        仍然缺失的信息字段：{missing}
        
        请生成一句简短、自然的话，向用户询问这些缺失的信息。
        不要重复询问已提供的信息。
        """)
    ])
    
    follow_up_chain = follow_up_prompt | llm
    
    question = follow_up_chain.invoke({
        "current_info": {k: v for k, v in current_slots.items() if v is not None},
        "missing": missing_fields
    }).content
    
    # 5. 与用户交互
    user_answer = input(f"\033[1;33mAI: {question}\033[0m\n您: ")
    conversation_history.append(HumanMessage(content=user_answer))

# ======================
# 4. 最终结果
# ======================
print(f"\n\033[1;32m[收集完成] 最终结构化数据:\033[0m")
print(json.dumps(current_slots, ensure_ascii=False, indent=2))

"""
selected_fields 是考卷的题目（固定不变：必须要答完这5道题）。
conversation_history 是学生的答题过程（不断增加：学生写了更多字）。
extract_slots 是老师阅卷（每次根据学生目前写的所有内容，重新给每道题打分）。
current_slots 是最终成绩单（一旦某道题得分了，就记下来，防止老师下次看走眼给改成0分）。
"""
"""
个人思考:
在业务步骤或者说所需条件非常清晰时,并且需要同时满足这些条件或步骤才能顺利进行下一步就比较适合这个完善问题法。
"""