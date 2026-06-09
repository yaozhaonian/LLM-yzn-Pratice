"""
构建**多智能体系统（Multi-Agent System）**的一种更高级、更模块化的模式：基于子图（Subgraph）和动态工具生成的智能体交接。
将每个“专家”封装为独立的编译子图（Compiled Subgraph）。交接逻辑被封装在一个通用的工厂函数 make_handoff_tool 中，使得添加新的专家 Agent 变得非常容易。
"""
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.prebuilt import ToolNode, ToolRuntime
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import convert_to_messages, AnyMessage
from langgraph.types import Command
from typing import Literal

model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    temperature=0.0 # 降低温度以提高计算确定性
)

# 动态创建“交接工具”
def make_handoff_tool(*, agent_name: str):
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name)
    def handoff_to_agent(runtime: ToolRuntime):
        """向其他代理寻求帮助."""
        state = runtime.state
        tool_call_id = runtime.tool_call_id

        tool_message = {
            "role": "tool",
            "content": f"成功转移到 {agent_name}",
            "name": tool_name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={"messages": state["messages"] + [tool_message]},
        )

    return handoff_to_agent

# 创建一个标准的 ReAct 风格 Agent 子图
def make_agent(model, tools, system_prompt=None):
    model_with_tools = model.bind_tools(tools)
    # 👇 官方 ToolNode：自动支持 ToolRuntime / InjectedState
    tool_node = ToolNode(tools)

    def call_model(state: MessagesState) -> Command[Literal["tools", "__end__"]]:
        messages = state["messages"]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        response = model_with_tools.invoke(messages)
        # print(f"LLM 生成的回复: \n{response}\n")
        if response.tool_calls:
            return Command(goto="tools", update={"messages": [response]})
        return {"messages": [response]}

    # 构建图
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_edge("tools", "agent")

    return workflow.compile()


def pretty_print_messages(update):
    if isinstance(update, tuple):
        ns, update = update
        if not ns: return
        graph_id = ns[-1].split(":")[0]
        print(f"[子图 {graph_id}]")

    if isinstance(update, dict):
        for node, val in update.items():
            print(f"→ 节点 {node}:")
            try:
                if isinstance(val, dict) and "messages" in val:
                    for m in convert_to_messages(val["messages"]):
                        m.pretty_print()
            except:
                pass
    print("-" * 60)


@tool
def add(a: int, b: int) -> int:
    """将两个数相加."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """将两个数相乘."""
    return a * b

# 测试单个agent
agent = make_agent(model, [add, multiply])

print("测试单个agent:")
for chunk in agent.stream({"messages": [("user", "(3 + 5) * 12")]}):
    pretty_print_messages(chunk)

# 测试多agent交互
print("\n==== 测试多agent交互 ====")

addition_expert = make_agent(
    model,
    [add, make_handoff_tool(agent_name="multiplication_expert")],
    system_prompt="您是加法专家，只会加法。乘法请交给乘法专家。",
)

multiplication_expert = make_agent(
    model,
    [multiply, make_handoff_tool(agent_name="addition_expert")],
    system_prompt="您是乘法专家，只会乘法。加法请交给加法专家。",
)

builder = StateGraph(MessagesState)
builder.add_node("addition_expert", addition_expert)
builder.add_node("multiplication_expert", multiplication_expert)
builder.add_edge(START, "addition_expert")
graph = builder.compile()
graph.get_graph().draw_png(output_file_path='./2-tools_to_handover.png')
print("\n测试多agent交互:")
for chunk in graph.stream(
    {"messages": [("user", "(3 + 5) * 12的结果是?")]}, subgraphs=True   # 复杂点就不行了,可能跟模型有关
):
    pretty_print_messages(chunk)




"""
应用场景
这种架构适用于 大规模、可扩展的多智能体协作系统：

1.模块化智能体开发：
场景：你需要构建一个包含“搜索专家”、“代码专家”、“写作专家”的系统。
应用：使用 make_agent 可以快速生成每个专家的标准 ReAct 结构。你只需要为每个专家配置不同的 System Prompt 和工具集。

2.动态且解耦的路由：
场景：智能体之间的协作关系复杂，且可能频繁变化。
应用：通过 make_handoff_tool，交接逻辑被标准化。如果新增一个“绘图专家”，只需创建该专家子图，并在其他需要它的专家工具列表中加入 make_handoff_tool(agent_name="drawing_expert") 即可，无需修改父图结构或其他专家的内部逻辑。

3.嵌套图的状态隔离与管理：
场景：每个专家可能需要维护自己独立的短期记忆或中间状态，而不污染全局状态。
应用：由于每个专家是一个独立的子图，它们拥有独立的状态空间（虽然这里都继承自 MessagesState，但可以扩展）。父图只关心消息流的传递，而专家内部的详细推理步骤被封装在子图中，通过 subgraphs=True 可以选择性地观察。

"""

