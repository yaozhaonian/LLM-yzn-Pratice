# 使用条件进行分支运行
from typing_extensions import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from operator import add

class State(TypedDict):
    sum: Annotated[list, add]
    switch: str

def node_a(state: State):
    print(f'把"节点A"加进{state["sum"]}')
    return {"sum": ["节点A"]}

def node_b(state: State):
    print(f'把"节点B"加进{state["sum"]}')
    return {"sum": ["节点B"]}

def node_c(state: State):
    print(f'把"节点C"加进{state["sum"]}')
    return {"sum": ["节点C"]}

def node_d(state: State):
    print(f'把"节点D"加进{state["sum"]}')
    return {"sum": ["节点D"]}

def node_e(state: State):
    print(f'把"节点E"加进{state["sum"]}')
    return {"sum": ["节点E"]}


builder = StateGraph(State)
builder.add_node(node_a)
builder.add_node(node_b)
builder.add_node(node_c)
builder.add_node(node_d)
builder.add_node(node_e)
builder.add_edge(START, "node_a")

# 自定义走边
def route_edge(state: State) -> Sequence[str]:
    if state["switch"] == "bc":
        return ["node_b", "node_c"]
    elif state["switch"] == "bd":
        return ["node_b", "node_d"]
    else:
        return ["node_c", "node_d"]

intermediates = ["node_b", "node_c", "node_d"]


builder.add_conditional_edges(source="node_a", path=route_edge, path_map=intermediates)

builder.add_edge(intermediates, "node_e")
builder.add_edge("node_e", END)

graph = builder.compile()
graph.get_graph().draw_png(output_file_path='./2-switch.png')

print("="*50)
print(graph.invoke({"sum": [], "switch": "bc"}))
print("="*50)
print(graph.invoke({"sum": [], "switch": "bd"}))
print("="*50)
print(graph.invoke({"sum": [], "switch": "cd"}))

"""
应用场景
1. 动态多工具调用 (Dynamic Multi-Tool Invocation)
场景: 用户问：“帮我比较一下 iPhone 15 和 Samsung S24 的价格和评测。”
逻辑:
node_a：意图识别，发现需要“比价”和“评测”。
route_edge：根据意图，决定并行调用 search_price_tool 和 search_review_tool。
如果用户只问价格，则只调用 search_price_tool。
优势: 相比串行调用，并行调用可以显著减少等待时间（Latency），且能根据用户需求动态调整调用的工具组合。
2. 多维度数据验证/审核 (Multi-Perspective Validation)
场景: 生成了一段代码或文章，需要进行多方面检查。
逻辑:
node_a：生成内容。
route_edge：根据内容类型决定审核维度。
如果是代码：并行启动 syntax_check (语法检查), security_scan (安全扫描), performance_review (性能评估)。
如果是文本：并行启动 grammar_check (语法), tone_analysis (语气分析), fact_check (事实核查)。
node_e：汇总所有检查结果，生成最终报告。
优势: 不同的检查互不依赖，并行执行效率最高；且可以根据内容类型动态裁剪不必要的检查步骤。
3. 个性化信息聚合 (Personalized Information Aggregation)
场景: 新闻推荐系统或金融仪表盘。
逻辑:
node_a：获取用户画像（如：关注科技、体育）。
route_edge：
如果用户关注“科技+体育”，并行抓取 tech_news_api 和 sports_news_api。
如果用户关注“金融+政治”，并行抓取 finance_api 和 politics_api。
node_e：将抓取到的不同来源数据合并、去重、排序，返回给用户。
优势: 灵活适配不同用户的兴趣组合，避免抓取无用数据，同时利用并行加速数据获取。
4. A/B 测试或模型 ensemble (Model Ensemble)
场景: 对同一个问题，想同时尝试不同的处理策略或模型，然后综合结果。
逻辑:
node_a：接收用户输入。
route_edge：根据配置，决定并行调用 model_v1, model_v2 或者 prompt_strategy_A, prompt_strategy_B。
node_e：收集所有模型的输出，使用投票机制或另一个 LLM 进行综合评判，选出最佳答案。
优势: 提高系统的鲁棒性和准确性，同时允许动态调整参与“竞赛”的模型组合。
"""
