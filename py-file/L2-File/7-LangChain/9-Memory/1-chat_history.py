# 给大模型添加历史对话
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory

llm = ChatOllama(model="qwen2.5:7b",temperature=0.9)
str_parser = StrOutputParser()

template = ChatPromptTemplate(
    [
        ("system", "你是一个得力的⼈⼯智能助⼿"),
        # Means the template will receive an optional list of messages under
        # the "conversation" key
        ("placeholder", "{conversation}"),
        # Equivalently:
        # MessagesPlaceholder(variable_name="conversation", optional=True)
    ]
)

chain = template | llm | str_parser

chat_history = ChatMessageHistory()
chat_history.add_user_message("你好,我是白宫前准宣传部……咳咳,老夫乃两百年前蒸汽宗宗主瓦特,如今世界发展得如何？")
response = chain.invoke({"conversation": chat_history.messages})
print(response)
chat_history.add_ai_message(response)
print("="*50)
print(chat_history.messages)

chat_history.add_user_message("你好,我似乎睡过头了，你是谁，我又是谁？")
response2 = chain.invoke({"conversation": chat_history.messages})
print("="*50)
print(response2)
chat_history.add_ai_message(response)
while True:
    user_input = input("请输入：")
    if user_input == "exit":
        break
    print("="*50)
    chat_history.add_user_message(user_input)
    print(chain.invoke({"conversation": chat_history.messages}))