# 将控制流和状态更新与命令结合
"""
打破了“图结构必须在编译时固定”的传统思维，允许运行时动态决定下一步去向，且这种决定权可以下放给嵌套的子模块。
"""
"""
应用场景
1. 动态路由决策中心 (Dynamic Routing Hub)
场景: 你有一个复杂的分类器或决策引擎，它被封装在一个子图中。根据子图的计算结果（如意图识别、情感分析、风险评分），主流程需要走向完全不同的分支（如“人工审核”、“自动通过”、“拒绝”）。
优势: 决策逻辑完全隔离在子图中，主图只需要等待结果并执行相应操作，无需在主图中编写复杂的条件判断边。
2. 工具调用与异常处理 (Tool Execution & Error Handling)
场景: 子图负责执行一个不稳定的外部 API 调用。
如果成功，子图返回 Command(goto="success_handler", graph=Command.PARENT)。
如果失败，子图返回 Command(goto="retry_logic" or "error_handler", graph=Command.PARENT)。
优势: 将“执行”与“流程控制”解耦。子图不仅返回数据，还返回“下一步该怎么做”的指令。
3. 多Agent 协作中的“经理”角色 (Manager Agent)
场景: 在一个多 Agent 系统中，有一个“经理 Agent”（子图）负责协调几个“工人 Agent”（父图中的节点）。
经理分析任务后，决定：“先让写手做，再让编辑做”。
经理通过 Command 指示父图依次激活 writer_node 和 editor_node。
优势: 实现了动态的工作流编排，工作流的顺序不是硬编码的，而是由智能体实时决定的。
4. 游戏引擎或交互式故事 (Game Engine / Interactive Story)
场景: 子图处理玩家的输入和游戏逻辑判定（战斗、对话选项）。
如果玩家获胜，Command(goto="victory_scene")。
如果玩家失败，Command(goto="game_over")。
优势: 游戏逻辑（子图）与场景渲染/状态展示（父图节点）分离，便于维护庞大的剧情树。
"""

import random
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START
from langgraph.types import Command
import operator

class State(TypedDict):
    foo: Annotated[str, operator.add]
    

def node_a(state: State):
    print("node_a(节点a):\n", state)
    value = random.randint(1, 2)
    print("随机值:", value)
    if value == 1:
        goto="node_b"
    else:
        goto="node_c"
    return Command(
        update={"foo": str(value)}, 
        goto=goto,
        graph=Command.PARENT
    )

subgraph = StateGraph(State).add_node(node_a).add_edge(START, "node_a").compile()

def node_b(state: State):
    print("node_b(节点b):\n", state)
    return {"foo": "节点b"}

def node_c(state: State):
    print("node_c(节点c):\n", state)
    return {"foo": "节点c"}

par_graph = StateGraph(State).add_node(node_b).add_node(node_c).add_node("subgraph", subgraph).add_edge(START, "subgraph").compile()

# try:
#     par_graph.get_graph().draw_png(output_file_path='./17-ccf_suc.png')
# except Exception as e:
#     print(f"错误: {e}")
#     print("请检查是否安装 graphviz 库")

print(par_graph.invoke({"foo": "Action"}))
