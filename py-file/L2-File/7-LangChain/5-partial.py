#提示模板部分格式化（Partial Formatting）：适用于需要先给某些参数赋值，其余参数后期赋值。
#非常适用于动态获得特定变量值的情况，比如日期和时间，某些系统内部配置。这些值是一般是无需用户在每次提问时输入。
#比如，实现一个学习助手，用户在提问前，先确定要问历史问题，地理问题等等
from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2.5:7b", temperature=0.9)

def get_date():
    """返回当前日期"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

from langchain_core.prompts import PromptTemplate

template = """
你是一个关于{topic}方面的顶尖故事家,请用{type}风格，讲一个符合{date}的500年前左右时间的{topic}故事。
"""

prompt = PromptTemplate(
    input_variables=["topic", "type", "date"], 
    template=template
)

partial_prompt = prompt.partial(
    date = get_date()
)

# topics = input("请输入主题：")
# types = input("请输入风格：")

# from langchain_core.output_parsers import StrOutputParser
# str_parser = StrOutputParser()

# chain = partial_prompt | llm | str_parser
# result = chain.invoke({"topic": topics, "type": types})
# print("正在生成中...")
# print(result)


partial_prompt2 = prompt.partial(
    date = get_date()
)
result2 = llm.invoke(partial_prompt2.format(topic='笑话',type='欧亨利'))
print("直接输出:",result2.content)
