"""
基于 LangGraph 框架实现的 Map-Reduce（映射-归约） 并行处理工作流
"""
"""
Send 机制
对象的数量可能在事先未知（意味着边的数量可能未知），并且输入到下游的 State 应该不同（每个生成的对象一个）。
使用Send(跳跃的节点, 参数)就可以
"""


from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# 初始化 LLM
llm = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

class OverallState(TypedDict):
    subjects: list[str]
    content: Annotated[list[str], add]
builder = StateGraph(OverallState)

#使用 Send 来并行触发多个任务，然后返回指向generate_content节点
def receive_content(state: OverallState):
    """接收内容"""
    # subjects中有几个元素，就会有几个Send对象
    # 最好一个主题做一种节点，多做会乱
    joke_content = [
        Send("generate_content",  {"subject": subject})
        for subject in state["subjects"][:1]
    ]
    sad_content = [
        Send("generate_sad_content",  {"subject": subject})
        for subject in state["subjects"][1:]
    ]
    return joke_content + sad_content
    

def generate_content(state: dict) -> dict:
    # 接收 Send 传递的局部状态 {"subject": "主题"}
    subject = state["subject"]
    
    prompt = f"请生成一个关于'{subject}'的笑话，字数在100~200之间。"
    
    response = llm.invoke([SystemMessage(content="你是一个非常擅长欧亨利结尾的幽默作家"),
                        HumanMessage(content=prompt)])
    print("="*30,"笑话","="*30)
    print(f"主题: {subject}")
    print(f"生成内容: {response.content}")
    joke_content = response.content
    
    return {"content": [joke_content]}

def generate_sad_content(state: dict) -> dict:
    # 接收 Send 传递的局部状态 {"subject": "主题"}
    subject = state["subject"]
    
    prompt = f"请生成一个关于'{subject}'的悲伤故事，字数在100~200之间。"
    
    response = llm.invoke([SystemMessage(content="你是一个优秀的作家"),
                        HumanMessage(content=prompt)])
    print("="*30,"悲伤","="*30)
    print(f"主题: {subject}")
    print(f"生成内容: {response.content}")
    joke_content = response.content
    
    return {"content": [joke_content]}


# 添加generate_content节点，该节点执行一个lambda函数，生成一个关于主题的不同情绪的故事
builder.add_node("generate_content", generate_content)
builder.add_node("generate_sad_content", generate_sad_content)

builder.add_conditional_edges(START, receive_content)

builder.add_edge("generate_content", END)
builder.add_edge("generate_sad_content", END)

graph = builder.compile()
# graph.get_graph().draw_png(output_file_path='./7-Map_Reducer.png')
result = graph.invoke({"subjects": ["电动车", "蛋仔"]})

print("="*50)
print(result)

"""
主要功能是：根据输入的主题列表，并行地生成不同风格的内容（笑话或悲伤故事），并将结果汇总。
1.状态定义 (OverallState)
2.Map 阶段 (分发任务)
3.Worker 节点 (执行任务)
4.Reduce 阶段 (自动聚合)

应用场景
1.批量内容生成：
    需要为多个产品生成描述、为多个用户生成个性化邮件、为多个关键词生成 SEO 文章摘要等。并行处理可以显著减少总耗时。
2.多策略/多模型对比：
    对同一个问题，同时发送给不同的 Prompt 模板（如本例中的“幽默” vs “悲伤”）或不同的 LLM 模型，然后收集所有结果供用户选择或进行后续评估。
3.数据并行处理：
    读取多个文档，并行提取关键信息、实体或总结，最后将提取结果汇总到一个报告中。
4.复杂路由分发：
    根据输入数据的类型（如新闻、体育、娱乐），将其分发到专门的处理节点（不同的 Prompt 或处理逻辑），最后统一输出格式。
"""


"""
============================== 悲伤 ==============================
主题: 蛋仔
生成内容: 在一个被遗忘的小岛上，住着一群形形色色的蛋仔们。其中最特别的是翠绿和紫罗兰。他们是最好的朋友，一起探索岛屿的秘密。某日，探险中他们被一道奇异光束击中，竟意外穿越到了一片荒芜之地。

这里没有阳光，没有食物，只有无尽的黑暗。两人的友情在艰难险阻中经受考验，在饥饿寒冷中逐渐淡漠，翠绿渐渐离开了紫罗兰，独自寻找生机。最后，只有紫罗兰孤零零地躺在一块岩石上，望着远方微弱的星光。

就在此时，一道熟悉的翠绿色光芒映入眼帘，是翠绿回来了。原来他曾被一阵温柔的力量指引，找到了一处被遗忘的安全岛。他带回了新鲜的食物和水源。二人相拥痛哭，他们知道，从此再也不会分开……
============================== 笑话 ==============================
主题: 电动车
生成内容: 有一天，一辆电动车去餐厅吃饭。
菜单上，它指着“鲜美的午餐肉”，用那熟悉得让人心碎的声音说，“不行哦，我刚节食呢。”
旁边的一辆燃油车不解问：“你不是刚刚吃了一顿大餐吗？”
电动车微微一笑，露出它的充电线接口：“那是我的启动电源。”
==================================================
{'subjects': ['电动车', '蛋仔'], 'content': ['有一天，一辆电动车去餐厅吃饭。\n菜单上，它指着“鲜美的午餐肉”，用那熟悉得让人心碎的声音说，“不行哦，我刚节食呢。”\n旁边的一辆燃油车不解问：“你不是刚刚吃了一顿大餐吗？”\n电动车微微一笑，露出它的充电线接口：“那是我的启动电源。”', '在一个被遗忘的小岛上，住着一群形形色色的蛋仔们。其中最特别的是翠绿和紫罗兰。他们是最好的朋友，一起探索岛屿的秘密。某日，探险中他们被一道奇异光束击中，竟意外穿越到了一片荒芜之地。\n\n这里没有阳光，没有食物，只有无尽的黑暗。两人的友情在艰难险阻中经受考验，在饥饿寒冷中逐渐淡漠，翠绿渐渐离开了紫罗兰，独自寻找生机。最后，只有紫罗兰孤零零地躺在一块岩石上，望着远方微弱的星光。\n\n就在此时，一道熟悉的翠绿色光芒映入眼帘，是翠绿回来了。原来他曾被一阵温柔的力量指引，找到了一处被遗忘的安全岛。他带回了新鲜的食物和水源。二人相拥痛哭，他们知道，从此再也不会分开……']}
"""

