# 简单的Langchain链(本地模型)


from langchain_core.globals import set_verbose
set_verbose(True)

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# 初始化Ollama模型
llm = OllamaLLM(model="qwen2.5:7b", temperature=0.9)

# 定义提示模版
prompt = PromptTemplate.from_template("请简单解释：{topic}")

# 创建链并执行
chain = prompt | llm
print(chain.invoke({"topic": "如何使用Langchain"}))
