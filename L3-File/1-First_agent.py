# Agent入门(使用Langchain)
# Agent就是它可以通过对话与根据你提供的工具去选择地执行下一步
import os

# LangChain 组件
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
# langchain-core/load/dump/dumps
# 返回对象的 JSON 字符串表示形式。
from langchain_core.load import dumps, loads
# /langchain-tavily/tavily_search/TavilySearch
# 一个查询 Tavily 搜索 API 并返回 JSON 数据的工具。
# Tavily需要申请api_key才可使用，并设置系统环境变量TAVILY_API_KEY
# Tavily官网:https://app.tavily.com/
from langchain_tavily import TavilySearch
# /langchain-classic/agents/tool_calling_agent/base/create_tool_calling_agent
# 创建一个使用工具的智能体
# langchain-classic/agents/agent/AgentExecutor
# 使用工具的智能体
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor, tool

llm = ChatOllama(
    model='qwen2.5:7b', 
    temperature=0,
    base_url="http://127.0.0.1:11434" 
)
embedding = OllamaEmbeddings(model="bge-m3:latest")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


"""
{agent_scratchpad}为必需，中间Agent操作和工具输出消息将在这里传递。
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个AI助手"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

@tool
def magic_function(input: str) -> str:
    """模拟工具函数"""
    if input in ["hello","hi"]:
        return input + " Agent"
    return "I am from Ollama"

tavily_search = TavilySearch(max_results = 3, tavily_api_key=TAVILY_API_KEY)
tools = [magic_function, tavily_search]

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# reult1 = agent_executor.invoke({"input": "你是一个上知天文下知地理的AI助手，magic_function('hello')的值是多少？"})
# print("reult1\n",reult1)
# reult2 = agent_executor.invoke({"input": "请问现任英国首相年龄的平方是多少？"})
# print("reult2\n",reult2)
# reult3 = agent_executor.invoke({"input": "你是谁？"})   # 没有记忆功能
# print("reult3\n",reult3)

# 添加记忆功能
# langchain-classic/memory/buffer/ConversationBufferMemory
from langchain_classic.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="input",       # 明确指定输入键
    output_key="output"      # 明确指定输出键
)


prompt_m = ChatPromptTemplate.from_messages([
    ("system", "你是一个AI助手"),
    MessagesPlaceholder(variable_name="chat_history"), # 这里插入历史记录
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent_m = create_tool_calling_agent(llm, tools, prompt_m)
agent_m_executor = AgentExecutor(agent=agent_m, tools=tools, verbose=True,memory=memory,handle_parsing_errors=True)    # 加上memory

result_m_1 = agent_m_executor.invoke({"input": "你好，我是玉树临风的女兆姚，很高兴见到你，请问现任台湾省领导人是谁？"})
print("result1 = ",result_m_1['output'])
result_m_2 = agent_m_executor.invoke({"input": "magic_function('hello')的返回值是多少？"})
print("result2 = ",result_m_2['output'])
result_m_3 = agent_m_executor.invoke({"input": "我是谁？"})
print("result3 = ",result_m_3['output'])

