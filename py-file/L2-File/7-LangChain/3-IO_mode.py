# IO输出，有多种格式
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser, JsonOutputParser, ListOutputParser, CommaSeparatedListOutputParser, XMLOutputParser
# 纯字符串输出,Pydantic 模型输出,JSON 格式输出,通用列表输出,逗号分隔列表输出,XML 格式输出

from langchain_ollama import ChatOllama 
llm = ChatOllama(model="qwen2.5:7b", temperature=0.9)

chatPrompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个翻译模型，你需要将输入的句子翻译成{language}"),
    HumanMessagePromptTemplate.from_template("{text}")
])
# 字符串的输出格式
str_parser = StrOutputParser()

# 链的形式调用
chain = chatPrompt | llm | str_parser
# 可以查看链及其组件
# print(chain.input_schema.model_json_schema)
# print("="*50)
# print(chain.output_schema.model_json_schema)
# print("="*50)
# print(chain.get_prompts)
# print("="*50)
#打印出链的结构图，该功能需要额外安装 pip install grandalf
# chain.get_graph().print_ascii()
# print("="*50)

# print(chain.invoke({"text": "I love programming.This is a program about Langchain.", "language": "中文"}))


from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel, Field

# ===== 日期输出格式（使用 PydanticOutputParser 替代）=====
class DateAnswer(BaseModel):
    """日期回答模型"""
    answer: str = Field(description="问题的答案，包含日期信息")
    date: str = Field(description="提取的日期，格式：YYYY-MM-DD")

# 创建 Pydantic 输出解析器
pydantic_parser = PydanticOutputParser(pydantic_object=DateAnswer)

template = """
你是一个全能专家，回答用户问题:{question}

{format_instructions}

请确保日期格式为 YYYY-MM-DD
"""
chatPrompt2 = PromptTemplate.from_template(
    template=template,
    partial_variables={"format_instructions":pydantic_parser.get_format_instructions()}
)
# print("chatPrompt2:",chatPrompt2)
# print("="*50)
# print("get_format_instructions:",pydantic_parser.get_format_instructions())
# print("="*50)

#使⽤RunnableSequence，与链功能一致
from langchain_core.runnables import RunnableSequence
chain2 = RunnableSequence(chatPrompt2, llm, pydantic_parser)
print(chain2.invoke({"question": "中国抗日胜利的日子是？"}))


# 逗号分隔列表输出
csl = CommaSeparatedListOutputParser()

chatprompt3 = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的程序员"),
    ("user", "{input}")
])

chain3 = chatprompt3 | llm | csl
# print(chain3.invoke({"input": "请列出10个最常用的编程语言"}))
# print("="*50)
# result = chain3.invoke({"input": "请列出3个常见的机器学习框架"})
# print(type(result))
# print(result)

# JSON 格式输出
jso = JsonOutputParser()
chatprompt4 = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的程序员,你复述一遍用户的问题后给出自己的答案"),
    ("user", "{input}")
])
chain4 = chatprompt4 | llm | jso
"""
一些预构建的结构（如传统的LangChain智能体和链）可能在内部使用输出解析器，因此即使你没有明显地实例化和使用输出解析器，也可能会遇到此错误。
"""
# result = chain4.invoke({"input": "langchain是什么?"})
# print(type(result))
# print(result)


# print("="*50)
# result_js = chain4.invoke({"input": "langchain是什么? 问题用question 回答用ans 返回一个JSON格式"})
# print(type(result_js))
# print(result_js)
"""
"system", "你是一个专业的程序员"
<class 'dict'>
{'ans': 'LangChain 是一个开源的框架，旨在简化构建、部署和集成语言模型的应用程序。它为开发者提供了与 LLMs（如大型语言模型）进行交互的功能，并允许他们将其整合到应用程序或服务中，同时提供了一系列工具来管理这些交互。”\n}'}
"""
"""
"system", "你是一个专业的程序员,你复述一遍用户的问题后给出自己的答案"
<class 'dict'>
{'question': '什么是langchain?', 'ans': 'LangChain 是一个由 LLM（大型语言模型）驱动的开放源代码平台，用于构建和扩展对话式AI应用程序。它旨在为开发者提供一系列API、工具和服务来创建交互式AI助手和其他自然语言处理应用程序。LangChain 使用 Python 编写，并且是基于 Chain 模型进行设计的，该模型通常是一个可以封装多个函数或服务的容器，每个函数或服务都用于执行一项特定的任务。”\n}'}
"""
# print("="*25,"美化格式","="*25)
# import json
# json_str = json.dumps(result_js, indent=2, ensure_ascii=False)
# print(json_str)