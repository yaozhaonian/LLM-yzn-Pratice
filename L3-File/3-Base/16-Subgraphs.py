
from typing import TypedDict
from langgraph.graph import StateGraph, START
"""
共享模式键

"""
class SubGraphState(TypedDict):
    text: str
    bar: str

def sub_graph_a(state: SubGraphState):
    return {"text": "原来如此"}

def sub_graph_b(state: SubGraphState):
    return {"text": state["text"] + " " + state["bar"]}


sub_builder = StateGraph(SubGraphState)
sub_builder.add_node(sub_graph_a)
sub_builder.add_node(sub_graph_b)
sub_builder.add_edge(START, "sub_graph_a")
sub_builder.add_edge("sub_graph_a", "sub_graph_b")


sub_graph = sub_builder.compile()
# try:
#     sub_graph.get_graph().draw_png(output_file_path='./16-subgraph.png')
# except Exception as e:
#     print(f"错误: {e}")
#     print("请检查是否安装 graphviz 库")


class ParentState(TypedDict):
    text: str

def node_a(state: ParentState):
    return {"text": state["text"] + " " + "hello,node_a!"}

# 父图和子图具有不同的模式
def node_b(state: ParentState):
    # 状态转换到子图状态
    response = sub_graph.invoke({"bar": state["text"]})
    print("不同的模式:\n", response)
    print("="*50)
    # 将响应转换回父图状态
    return {"text": response["bar"]}


par_builder = StateGraph(ParentState)
par_builder.add_node("node_a", node_a)
# par_builder.add_node("node_b", sub_graph)
par_builder.add_node("node_b", node_b)

par_builder.add_edge(START, "node_a")
par_builder.add_edge("node_a", "node_b")
par_graph = par_builder.compile()

# par_graph.get_graph().draw_png(output_file_path='./16-Subgraphs_b.png')

# for chunk in par_graph.stream({"text": "你好"}):
#     print(chunk)

# print("-"*30,"子图","-"*30)

for chunk in par_graph.stream({"text": "funny"}, subgraphs=True):
    print(chunk)

"""

应用场景:
1.  模块化与代码复用 (Modularity & Reusability)
场景描述：你有一个通用的“搜索增强生成 (RAG)”流程，或者一个“代码解释器”流程，需要在多个不同的 Agent 中使用。
应用方式：使用第二种方式（状态隔离）。
将 RAG 流程封装为一个独立的子图，定义其特定的输入（如 query）和输出（如 context）。
在不同的主 Agent（如“客服Agent”、“写作Agent”）中，通过适配节点调用这个 RAG 子图。
优势：一旦优化了 RAG子图，所有调用它的 Agent 都会受益，无需重复编写代码。
2.  复杂任务的层级分解 (Hierarchical Task Decomposition)
场景描述：构建一个“科研助手”，任务包括：文献检索、摘要生成、对比分析、报告撰写。
应用方式：
父图：控制整体流程（检索 -> 摘要 -> 分析 -> 撰写）。
子图：
“文献检索”可以是一个子图，内部包含：关键词提取、多数据库搜索、结果去重。
“对比分析”可以是另一个子图，内部包含：提取关键点、生成对比表格、总结差异。
优势：降低主图的复杂度，每个子图专注于解决一个子问题，便于调试和维护。
3.  隔离上下文与状态管理 (Context Isolation)
场景描述：在主对话中，用户突然要求执行一个复杂的数学计算或代码生成任务，这个任务需要多步交互，且不应该污染主对话的历史记录或状态。
应用方式：使用第二种方式（状态隔离）。
主图维护长期的聊天历史 messages。
当检测到需要计算时，进入一个独立的“计算器子图”。
子图拥有自己的状态（如 current_expression, step_history），与主图的 messages 隔离。
子图结束后，只将最终结果返回给主图，插入到主对话中。
优势：保持主状态干净，避免无关的中间步骤污染长期记忆。
4.  团队协作开发 (Team Collaboration)
场景描述：大型项目中，不同的小组负责不同的功能模块（如一组负责“工具调用”，另一组负责“内容审核”）。
应用方式：
各组独立开发自己的子图，定义清晰的输入/输出接口（Schema）。
主图由架构师组装，通过适配节点连接各个子图。
优势：减少代码冲突，明确责任边界，支持并行开发
5. 同步阻塞任务
场景: 子图执行的是一个必须立即得到结果的同步操作（如数据库查询、外部 API 调用封装），且不需要暴露子图内部的中间步骤给父图的流式输出。
优势: invoke 是阻塞的，它会等待子图完全执行完毕才返回结果。这使得父图的逻辑更像传统的函数调用，易于理解和调试，而不必处理复杂的流式事件嵌套。
"""
