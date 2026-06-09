# 为图添加跨线程持久性


from langgraph.store.memory import InMemoryStore
import uuid
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langchain_openai import ChatOpenAI

from langchain_ollama import OllamaEmbeddings
# 如果项目中确实存在一个名为 init_embeddings 的封装函数，请取消下面这行的注释并修正路径
# from your_module import init_embeddings 

embedding = OllamaEmbeddings(model='bge-m3:latest', base_url="http://localhost:11434")
in_memory_store = InMemoryStore(
    index={
        "embed": embedding,
        "dims": 1024  # 必须和你用的向量模型维度匹配
    }
)


llm = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama" 
)

def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    """对话处理核心逻辑,包含记忆存储和检索功能"""
    # 从配置中获取用户ID,创建专属的命名空间
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)
    
    # 在存储中搜索与当前对话相关的记忆
    memories = store.search(namespace, query=str(state["messages"][-1].content))
    print("与当前对话相关的记忆:\n", memories)
    # 将记忆数据转换为字符串格式
    info = "\n".join([memory.value["data"] for memory in memories])
    
    # 构建系统提示,包含用户记忆信息
    system_msg = f"你是一个与用户交流的好助手。用户信息: {info}"
    
    # 检查是否需要存储新记忆
    last_message = state["messages"][-1]
    print("是否需要存储新记忆:\n", last_message.content)
    content = last_message.content
    if "记住" in last_message.content.lower():
        # 生成并存储新记忆
        print("需要存储新记忆")
        if ":" in content:
            memory_content = content.split(":", 1)[1].strip()
        else:
            memory_content = content.replace("记住", "").strip()
        
        # 存入提取出的真实内容
        store.put(namespace=namespace, key=str(uuid.uuid4()), value={"data": memory_content})
        print(f"已存储记忆: {memory_content}")
        
    response = llm.invoke(
        [{"role": "system", "content": system_msg}] + state["messages"]
    )
    print("LLM 生成的回复:\n", response.content)
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

graph = builder.compile(
    checkpointer=MemorySaver(),
    store=in_memory_store
)

# graph.get_graph().draw_png(output_file_path='./9-ctp.png')

# 测试场景1：存储记忆
config = {"configurable": {"thread_id": "1", "user_id": "1"}}
input_message = {"role": "user", "content": "你好！记住:我女朋友的名字是小姚,她喜欢跑步与刷剧,讨厌吃香菜"}
print("第一次对话（存储记忆）:")
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print() 
    
# 查看存储的记忆
print("\n存储的记忆内容:")
for memory in in_memory_store.search(("memories", "1")):
    print(memory.value)

# 测试场景2：读取记忆
config = {"configurable": {"thread_id": "3", "user_id": "1"}}  # 用户2的对话配置
input_message = {"role": "user", "content": "介绍一下我的女朋友"}
print("\n第二次对话（读取记忆）:")

# 跨线程读取到记忆
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

"""
应用场景
这种架构非常适合构建具有长期记忆能力的个性化 AI 应用：

个性化私人助理：

用户今天告诉助手“我对花生过敏”，下周在新的对话窗口中问“推荐一家餐厅”，助手能通过 store 检索到过敏信息，避免推荐含花生的菜品。
智能客服系统：

记录用户的历史购买偏好、尺码信息、投诉记录。无论用户何时发起新的客服会话（新的 thread），客服机器人都能立即调取这些关键信息，提供无缝服务。
角色扮演游戏 (RPG) NPC：

NPC 需要记住玩家在游戏早期做出的关键选择（如“拯救了村庄”或“偷窃了物品”）。这些关键情节作为“记忆”存入 Store，在后续任何章节的对话中影响 NPC 的态度和剧情分支。
企业级知识管理助手：

员工可以将重要文档片段或会议纪要“标记”为记忆。之后在任何项目中提问时，助手都能跨项目、跨时间段检索这些企业私有知识。
"""



