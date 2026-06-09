# 大模型参与节点流转简单demo

from typing import Annotated  # 用于添加类型注解
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages  # 消息处理工具

llm = ChatOllama(
    model='qwen2.5:7b', 
    temperature=0,
    base_url="http://127.0.0.1:11434" 
)

# 定义状态类型，使用TypedDict来明确状态的结构
class State(TypedDict):
    # 消息列表，使用Annotated添加元数据（这里指定了消息处理方式）
    messages: Annotated[list, add_messages]

# 创建状态图构建器,传入状态类型
graph_builder = StateGraph(State)

# 定义聊天机器人节点函数
def chatbot(state: State) -> State:
    print("chatbot(聊天机器人节点):\n",state["messages"])
    bot_message = llm.invoke(state["messages"])
    print("bot_message:",{"messages": [bot_message]},"\n")
    return {"messages": [bot_message]}
# 将聊天机器人节点添加到图中
graph_builder.add_node("chatbot", chatbot)

# 添加图的边(连接关系):
# 从开始节点连接到聊天机器人节点
graph_builder.add_edge(START, "chatbot")
# 从聊天机器人节点连接到结束节点
graph_builder.add_edge("chatbot", END)

# 编译图，使其可执行
graph = graph_builder.compile()

# 定义流式处理图更新的函数
def stream_graph_update(user_input: str):
    # 使用图的流式处理功能,传入用户输入作为初始信息
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        # 遍历事件中的值
        for value in event.values():
            print("value值:", value)
            print("助手最新回复value['messages'][-1].content:\n", value['messages'][-1].content)
    
# 主交互循环
while True:
    try:
        user_input = str(input("用户输入: "))
        # 检查退出命令
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        # 处理用户输入并获取助手回复
        stream_graph_update(user_input)
    except KeyboardInterrupt:
        print("用户已退出")
        break
    except Exception as e:
        print(f"错误: {e}")
        continue

