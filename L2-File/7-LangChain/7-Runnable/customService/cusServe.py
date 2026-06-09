"""
做一个Runable运用的小项目
业务场景:电商客户反馈处理系统
需求描述
    某电商平台需要自动处理客户反馈，实现以下功能：

        情感分析：判断用户反馈的情感倾向

        问题分类：识别反馈中的问题类型

        紧急程度评估：根据内容判断处理优先级
"""

import re
import json
import time

from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2.5:7b",temperature=0.5)

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableSequence

from langchain_community.chat_message_histories import ChatMessageHistory

import json

"""
业务分析:
1、先提取订单ID
2、根据订单情况分别进行情感分析、问题分类、紧急程度评估
3、生成回复草稿：根据分析结果生成初步回复
"""
# 业务处理
# 提取订单ID(目前假设用户对话中就给予订单ID,该ID有效字节长度为10;实际处理时需判断用户说的是哪个订单,根据用户的购买记录里获取对应的订单,一般不能让用户越权访问他人订单)
def extract_order_id(text: str) -> dict: 
    match = re.search(r'ORD\d{10}', text)
    return {"order_id": match.group() if match else "未知订单"}


# 情感分类
def analyze_sentiment(text: str) -> dict:
    prompt = f"""
    请分析以下客户反馈的情感倾向：
    「{text}」

    要求：
    1. 判断情感类型：POSITIVE(积极)/NEUTRAL(中性)/NEGATIVE(消极)
    2. 评估置信度(0.0-1.0)
    3. 提取3个关键短语

    返回JSON格式：
    {{
        "sentiment": "情感类型",
        "confidence": 置信度,
        "key_phrases": ["短语1", "短语2", "短语3"]
    }}
    """
    try:
        result = llm.invoke(prompt)
        print("紧急程度评估:\n",result.content)
        result_a = result.content
        json_output = JsonOutputParser().parse(result_a)
        return json_output
    except Exception as e:
        print(f"情感分析失败: {e}")
        return {
            "sentiment": "NEUTRAL",
            "confidence": 0.7,
            "key_phrases": []
        }

# 问题分类
def classify_question(text: str) -> dict:
    prompt = f"""
    作为电商客服专家，请对以下客户反馈进行分类：
    「{text}」

    分类选项：
    - 物流问题：配送延迟、物流损坏等
    - 产品质量：商品瑕疵、功能故障等
    - 客户服务：客服态度、响应速度等
    - 支付问题：扣款异常、退款延迟等
    - 退货退款：退货流程、退款金额等
    - 其他：无法归类的反馈

    要求：
    1. 选择最相关的1-2个分类
    2. 按相关性排序

    返回JSON格式：{{"categories": ["分类1", "分类2"]}}
    """
    try:
        result = llm.invoke(prompt)
        print("问题分类:\n",result.content)
        result_a = result.content
        json_output = JsonOutputParser().parse(result_a)
        return json_output
    except Exception as e:
        print(f"问题分类失败: {e}")
        return {"categories": ["其他"]}

# 紧急程度评估
def assess_urgency(text: str) -> dict:
    """使用千问模型评估紧急程度"""
    prompt = f"""
    作为客服主管，请评估以下客户反馈的紧急程度：
    「{text}」

    评估标准：
    - HIGH(高)：包含"紧急"、"立刻"、"马上"或威胁投诉
    - MEDIUM(中)：表达强烈不满但无立即行动要求
    - LOW(低)：一般反馈或建议

    返回JSON格式：
    {{
        "urgency": "紧急级别",
        "sla_hours": 响应时限(小时),
        "reason": "评估理由"
    }}
    """
    try:
        result = llm.invoke(prompt)       
        print("紧急程度评估:\n",result.content)
        result_a = result.content
        json_output = JsonOutputParser().parse(result_a)
        # 确保数值类型
        json_output["sla_hours"] = int(json_output["sla_hours"])
        return json_output
    except Exception as e:
        print(f"紧急度评估失败: {e}")
        return {
            "urgency": "MEDIUM",
            "sla_hours": 24,
            "reason": "评估失败"
        }

# 生成定制化回复(未做历史记录时)
# def generate_response(data: dict) -> dict:
#     print("生成定制化回复前:\n",data)
#     prompt_template = """
#     你是一名资深电商客服专家，请根据以下分析结果生成客户回复：

#     ### 客户反馈原文：
#     {feedback}

#     ### 分析结果：
#     - 订单ID：{order_id}
#     - 情感倾向：{sentiment} (置信度：{confidence:.2f})
#     - 问题类型：{categories}
#     - 紧急程度：{urgency} (需在{sla_hours}小时内响应)
#     {key_phrases_section}

#     ### 回复要求：
#     1. 根据情感倾向调整语气：
#        - 积极反馈：表达感谢，适当赞美
#        - 消极反馈：诚恳道歉，明确解决方案
#     2. 包含订单ID和问题分类
#     3. 明确说明处理时限和后续步骤
#     4. 长度100-150字，使用自然口语
#     5. 结尾询问是否还有其他问题

#     请直接输出回复内容，不需要额外说明。
#     """

#     # 构建"情感分类"关键短语部分
#     key_phrases = data.get("key_phrases", [])
#     if key_phrases:
#         key_phrases_section = "- 关键要点：" + "，".join(key_phrases[:3])
#     else:
#         key_phrases_section = ""
    
#     # 填充模板
#     prompt = prompt_template.format(
#         feedback=data["original_feedback"],
#         order_id=data["order_id"],
#         sentiment=data["sentiment"],
#         confidence=data.get("confidence", 0.8),
#         categories="、".join(data["categories"]),
#         urgency=data["urgency"],
#         sla_hours=data["sla_hours"],
#         key_phrases_section=key_phrases_section
#     )

#     try:
#         response = llm.invoke(prompt)
#         response = response.content
#         # 添加紧急标识
#         if data["urgency"] == "HIGH":
#             response = f"[紧急] {response}"

#         return {
#             "final_response": response,
#             "assigned_team": data["categories"][0] if data["categories"] else "General",
#             "result":data
#         }

#     except Exception as e:
#         print(f"回复生成失败: {e}")
#         return {
#             "final_response": "感谢您的反馈，我们的团队将尽快处理您的问题。",
#             "assigned_team": "General"
#         }
# 新
def generate_response(data: dict) -> dict:
    # data 现在应该包含: original_feedback, order_id, sentiment..., 以及 history
    
    history_str = ""
    if "history" in data and data["history"]:
        # 将历史记录格式化为字符串
        # 假设 history 是 [{"role": "user", "content": "..."}, {"role": "ai", "content": "..."}]
        for msg in data["history"]:
            role = "用户" if msg.get("role") == "user" else "客服"
            history_str += f"{role}: {msg.get('content', '')}\n"
        
    prompt_template = """
    你是一名资深电商客服专家。请结合以下的【对话历史】和【当前分析结果】生成回复。

    ### 对话历史：
    {history}

    ### 当前客户反馈原文：
    {feedback}

    ### 当前反馈分析结果：
    - 订单ID：{order_id}
    - 情感倾向：{sentiment} (置信度：{confidence:.2f})
    - 问题类型：{categories}
    - 紧急程度：{urgency} (需在{sla_hours}小时内响应)
    {key_phrases_section}

    ### 回复要求：
    1. 必须参考【对话历史】来理解上下文（例如代词指代）。
    2. 根据情感倾向调整语气。
    3. 包含订单ID和问题分类。
    4. 长度100-150字，使用自然口语。
    5. 结尾询问是否还有其他问题。

    请直接输出回复内容，不需要额外说明。
    """

    key_phrases = data.get("key_phrases", [])
    key_phrases_section = "- 关键要点：" + "，".join(key_phrases[:3]) if key_phrases else ""
    
    prompt = prompt_template.format(
        history=history_str, # 传入历史
        feedback=data["original_feedback"],
        order_id=data["order_id"],
        sentiment=data["sentiment"],
        confidence=data.get("confidence", 0.8),
        categories="、".join(data["categories"]),
        urgency=data["urgency"],
        sla_hours=data["sla_hours"],
        key_phrases_section=key_phrases_section
    )

    try:
        response = llm.invoke(prompt)
        print("生成定制化回复后:\n",response)
        response_content = response.content
        
        if data["urgency"] == "HIGH":
            response_content = f"[紧急] {response_content}"

        return {
            "final_response": response_content,
            "assigned_team": data["categories"][0] if data["categories"] else "General",
            "result": data
        }
    except Exception as e:
        print(f"回复生成失败: {e}")
        return {
            "final_response": "感谢您的反馈，我们的团队将尽快处理您的问题。",
            "assigned_team": "General"
        }



# 构建LCEL(LangChain Expression Language)处理链
# 1、基础信息提取(原)
# extract_chain = RunnableParallel(
#     order_id = RunnableLambda(extract_order_id),
#     original_feedback = lambda x: x
# )

# 2、并行分析任务(原)
# analyze_chain = RunnableParallel(
#     sentiment=RunnableLambda(analyze_sentiment),    # 情感分析   
#     categories=RunnableLambda(classify_question),   # 问题分类
#     urgency=RunnableLambda(assess_urgency)          # 紧急程度评估
# )

# 3、组合完整流程(原)
# processing_chain = RunnableSequence(
#     extract_chain,
#     RunnablePassthrough.assign(analyze=lambda x: analyze_chain.invoke(x["original_feedback"])),
#     RunnableParallel(
#         original_feedback=lambda x: x["original_feedback"],
#         order_id=lambda x: x["order_id"]["order_id"],
#         sentiment=lambda x: x["analyze"]["sentiment"].get("sentiment", "NEUTRAL"),
#         confidence=lambda x: x["analyze"]["sentiment"].get("confidence", 0.8),
#         key_phrases=lambda x: x["analyze"]["sentiment"].get("key_phrases", []),
#         categories=lambda x: x["analyze"]["categories"]["categories"],
#         urgency=lambda x: x["analyze"]["urgency"]["urgency"],
#         sla_hours=lambda x: x["analyze"]["urgency"]["sla_hours"],
#         urgency_reason=lambda x: x["analyze"]["urgency"].get("reason", "")
#     ),
#     RunnableLambda(generate_response)
# )


# 1、基础信息提取 (修改为适配字典输入)
extract_chain = RunnableParallel(
    order_id = RunnableLambda(lambda x: extract_order_id(x["original_feedback"])), # 从字典取值
    original_feedback = lambda x: x["original_feedback"],
    history = lambda x: x.get("history", []) # 透传历史
)

# 2、并行分析任务 (修改为适配字典输入，只取 original_feedback 进行分析)
analyze_chain = RunnableParallel(
    sentiment=RunnableLambda(lambda x: analyze_sentiment(x["original_feedback"])),   
    categories=RunnableLambda(lambda x: classify_question(x["original_feedback"])),  
    urgency=RunnableLambda(lambda x: assess_urgency(x["original_feedback"]))         
)

# 3、组合完整流程
processing_chain = RunnableSequence(
    extract_chain,
    RunnablePassthrough.assign(analyze=lambda x: analyze_chain.invoke(x)), # 传入整个字典，但函数内部只取 original_feedback
    RunnableParallel(
        original_feedback=lambda x: x["original_feedback"],
        order_id=lambda x: x["order_id"]["order_id"],
        sentiment=lambda x: x["analyze"]["sentiment"].get("sentiment", "NEUTRAL"),
        confidence=lambda x: x["analyze"]["sentiment"].get("confidence", 0.8),
        key_phrases=lambda x: x["analyze"]["sentiment"].get("key_phrases", []),
        categories=lambda x: x["analyze"]["categories"]["categories"],
        urgency=lambda x: x["analyze"]["urgency"]["urgency"],
        sla_hours=lambda x: x["analyze"]["urgency"]["sla_hours"],
        urgency_reason=lambda x: x["analyze"]["urgency"].get("reason", ""),
        history=lambda x: x.get("history", []) # 继续透传历史给 generate_response
    ),
    RunnableLambda(generate_response)
)







