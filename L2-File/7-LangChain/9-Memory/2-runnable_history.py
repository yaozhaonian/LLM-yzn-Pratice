# 使用ChatMessageHistory需要每次手动保存消息，可以使用RunnableWithMessageHistory
# 一个可以支持多人同时和大模型对话的聊天机器人
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.runnables import RunnableWithMessageHistory

llm = ChatOllama(model="qwen2.5:7b",temperature=0.9)
str_parser = StrOutputParser()

template = ChatPromptTemplate(
    [
        ("system", "你是一个聊天助手,用{language}回答所有问题"),
        ("placeholder", "{conversation}"),
        ("human", "{input}")
    ]
)

chain = template | llm | str_parser

# 保存所有用户的聊天记录
store = {}


def get_session(session_id:str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chatbot_with_his = RunnableWithMessageHistory(
    chain,
    get_session,
    input_messages_key = "input",
    history_messages_key = "history"
)

config_ch = {'configurable':{'session_id':'johnnie_chinese'}}
config_en = {'configurable':{'session_id':'johnnie_english'}}

resp = chatbot_with_his.invoke(
    {
        "input": "你好,我是广东靓仔", 
        "language":"中文"
    },
    config_ch
)
print("resp:\n",resp)
print("="*50)

resp1 = chatbot_with_his.invoke(
    {
        "input": "Hello,I'm a hansome boy from maoming, guangdong", 
        "language":"英文"
    },
    config_en
)
print("resp1:\n",resp1)
print("="*50)

resp_a = chatbot_with_his.invoke(
    {
        "input": "你好,我想去看看新鲜的最正宗的中药橘红,你知道应该去哪里看吗?", 
        "language":"中文"
    },
    config_ch
)
print("resp_a:\n",resp_a)
print("="*50)

resp1_a = chatbot_with_his.invoke(
    {
        "input": "I want to travel to Hong Kong for 2 days this weekend. Can you make a travel plan for me, including how to book flights, hotels, and which attractions to visit at what time.", 
        "language":"英文"
    },
    config_en
)
print("resp1_a:\n",resp1_a)
print("="*50)

print('内存\n',store)