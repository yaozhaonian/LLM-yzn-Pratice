"""
使用 LangGraph 结合 MemorySaver 实现线程级持久化（Thread-level Persistence）。
其核心目的是让 AI 助手能够“记住”同一会话（线程）中的历史对话内容，从而实现多轮对话的上下文连贯性。
"""
"""
为图、子图添加线程级持久性
"""
from typing import Annotated, Sequence, TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver


"""llm = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama" 
)

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    print("回复response:", response)
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

memory = MemorySaver()

graph = builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

input_message = {"role": "user", "content": "你好，我是小兆"}
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()


input_message = {"role": "user", "content": "我的名字是什么?"}
for chunk in graph.stream({"messages": [input_message]},{"configurable": {"thread_id": "1"}},stream_mode="values",):
    chunk["messages"][-1].pretty_print()"""



"""
 应用场景
这段代码的模式广泛应用于需要保持上下文记忆的 AI 应用中：

智能客服机器人：

用户可能在多个消息中分散提供订单号、问题描述等信息。机器人需要记住之前的对话内容才能准确解答后续问题。
每个用户的会话对应一个唯一的 thread_id。
个性化 AI 助手：

如代码示例所示，助手需要记住用户的偏好、姓名或背景信息，以便在后续对话中提供个性化服务。
多步任务执行代理 (Agent)：

在复杂的 Agent 工作流中，可能需要中断执行（例如等待用户确认或外部 API 回调）。持久化允许在中断后恢复状态，继续执行未完成的任务。
调试与回放：

开发者可以通过检查点历史记录，回溯对话的每一步状态，便于调试复杂的逻辑错误或分析用户行为路径。
"""

"""
改进建议（生产环境）
存储后端：MemorySaver 仅存储在内存中，进程重启后数据丢失。生产环境应使用数据库后端，如：

from langgraph.checkpoint.sqlite import SqliteSaver
# 或者 PostgresSaver, RedisSaver 等
with SqliteSaver.from_conn_string(":memory:") as memory: # 示例，实际应使用文件路径或连接串
    graph = builder.compile(checkpointer=memory)
异步支持：在高并发场景下，建议使用异步版本的 Checkpointer 和 Graph 执行方法 (ainvoke, astream)。
"""


class SubgraphState(TypedDict):
    foo: str
    bar: str

def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}

def subgraph_node_2(state: SubgraphState):
    return {"foo": state["foo"] + state["bar"]}

subgraph_builder = StateGraph(SubgraphState).add_node("subgraph_node_1", subgraph_node_1).add_node("subgraph_node_2", subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1").add_edge("subgraph_node_1", "subgraph_node_2").add_edge("subgraph_node_2", END)
subgraph = subgraph_builder.compile()
subgraph.get_graph().draw_mermaid_png(output_file_path="./8-Thread_persist.png")

class State(TypedDict):
    foo: str

def node_a(state: State):
    return {"foo": "hi! " + state["foo"]}

parent_builder = StateGraph(State).add_node("node_a", node_a).add_node("subgraph", subgraph)
parent_builder.add_edge(START, "node_a")
parent_builder.add_edge("node_a", "subgraph")

# 验证持久性是否生效
parent_graph = parent_builder.compile(checkpointer=MemorySaver())
# parent_graph.get_graph().draw_png(output_file_path="./8-Thread_persist.png")
config = {"configurable": {"thread_id": "1"}}
print("--- 开始流式执行 ---")
for _, chunk in parent_graph.stream({"foo": "foo"}, config, subgraphs=True):
    print(_, chunk)

print("\n--- 获取最终状态 ---")
# 通过使用与调用图相同的配置来查看父图状态。
final_state = parent_graph.get_state(config)
print(f"最终值: {final_state.values}")   # {'foo': 'hi! foobar'}
print('###')

print("\n--- 获取状态历史 ---")
# 获取父图的所有历史状态
history = list(parent_graph.get_state_history(config))
for i, s in enumerate(history):
    print(f"历史步骤 {i}: next={s.next}, values={s.values}")

print('####')

# 【修正点】查找进入子图前的状态 (next 应该是 'subgraph' 而不是 'node_2')
# 注意：get_state_history 返回的顺序通常是倒序（最新的在前），或者正序，取决于版本，建议打印确认
# 这里我们寻找 next 包含 'subgraph' 的状态
state_entering_subgraph = None
for s in history:
    if s.next and "subgraph" in s.next:
        state_entering_subgraph = s
        break

if state_entering_subgraph:
    print(f"找到进入子图前的状态: {state_entering_subgraph}")
    
    # 检索子图状态的配置
    # 注意：在较新版本的 LangGraph 中，tasks 属性可能包含子图的 checkpoint_id
    if state_entering_subgraph.tasks:
        subgraph_task = state_entering_subgraph.tasks[0]
        print(f"子图任务信息: {subgraph_task}")
        
        # 如果 task.state 是一个配置字典 (checkpoint_id)
        if hasattr(subgraph_task, 'state') and subgraph_task.state:
            subgraph_config = subgraph_task.state
            print(f"子图配置: {subgraph_config}")
            
            # 获取子图在该检查点的状态
            # 注意：你需要使用 subgraph 实例或者父图来获取子图状态，具体取决于 LangGraph 版本
            # 通常可以通过 parent_graph.get_state(subgraph_config) 获取
            try:
                subgraph_state = parent_graph.get_state(subgraph_config)
                print(f"子图内部状态: {subgraph_state.values}")
            except Exception as e:
                print(f"获取子图状态失败 (可能是版本差异): {e}")
        else:
            print("未找到子图的嵌套状态配置。")
    else:
        print("该状态没有关联的任务（tasks）。")
else:
    print("未在历史记录中找到进入子图的状态。")

print('#####')

"""
--- 开始流式执行 ---
() {'node_a': {'foo': 'hi! foo'}}
('subgraph:5cac4afb-f2d1-e41b-db47-f43b02b90ad6',) {'subgraph_node_1': {'bar': 'bar'}}
('subgraph:5cac4afb-f2d1-e41b-db47-f43b02b90ad6',) {'subgraph_node_2': {'foo': 'hi! foobar'}}
() {'subgraph': {'foo': 'hi! foobar'}}

--- 获取最终状态 ---
最终值: {'foo': 'hi! foobar'}
###

--- 获取状态历史 ---
历史步骤 0: next=(), values={'foo': 'hi! foobar'}
历史步骤 1: next=('subgraph',), values={'foo': 'hi! foo'}
历史步骤 2: next=('node_a',), values={'foo': 'foo'}
历史步骤 3: next=('__start__',), values={}
####
找到进入子图前的状态: StateSnapshot(values={'foo': 'hi! foo'}, next=('subgraph',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f14383a-4907-65ac-8001-e5ae593d79c1'}}, metadata={'source': 'loop', 'step': 1, 'parents': {}}, created_at='2026-04-29T04:26:50.590353+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f14383a-4904-6d43-8000-2bed00657318'}}, tasks=(PregelTask(id='5cac4afb-f2d1-e41b-db47-f43b02b90ad6', name='subgraph', path=('__pregel_pull', 'subgraph'), error=None, interrupts=(), state={'configurable': {'thread_id': '1', 'checkpoint_ns': 'subgraph:5cac4afb-f2d1-e41b-db47-f43b02b90ad6'}}, result={'foo': 'hi! foobar'}),), interrupts=())
子图任务信息: PregelTask(id='5cac4afb-f2d1-e41b-db47-f43b02b90ad6', name='subgraph', path=('__pregel_pull', 'subgraph'), error=None, interrupts=(), state={'configurable': {'thread_id': '1', 'checkpoint_ns': 'subgraph:5cac4afb-f2d1-e41b-db47-f43b02b90ad6'}}, result={'foo': 'hi! foobar'})
子图配置: {'configurable': {'thread_id': '1', 'checkpoint_ns': 'subgraph:5cac4afb-f2d1-e41b-db47-f43b02b90ad6'}}
子图内部状态: {'foo': 'hi! foobar', 'bar': 'bar'}
#####
"""

