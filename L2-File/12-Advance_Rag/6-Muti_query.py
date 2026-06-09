# Multi-Query 多路召回
"""
思路:
    利用 LLM 生成 N 个与原始查询相关的问题
    将所有问题（加上原始查询）发送给检索系统。
    通过这种方法，可以从向量库中检索到更多文档。
"""
from pathlib import Path
import json

# LangChain 组件
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableParallel
from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever
# MultiQueryRetriever:
# 1.给定一个查询，使用大语言模型（LLM）生成一组查询。
# 2.针对每个查询检索文档。返回所有检索到文档的唯一并集。


embedding = OllamaEmbeddings(model="bge-m3:latest")
llm = ChatOllama(model='qwen2.5:7b', temperature=0)

# 辅助函数：将文档列表转换为字符串
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

current_dir = Path(__file__).parent
data_file_path = current_dir.parent / "Data" / "deepseek百度百科.txt"

# 检查文件是否存在
if not data_file_path.exists():
    raise FileNotFoundError(f"找不到数据文件: {data_file_path}")

loader = TextLoader(str(data_file_path), encoding="utf-8")
doc = loader.load()

# 分割文本
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
docs = text_splitter.split_documents(doc)

# 定义回答问题的 Prompt
qa_prompt_template = """请根据下面给出的【上下文】来回答【问题】。
如果上下文中没有包含回答问题所需的信息，请说“根据已知信息无法回答该问题”。

【上下文】:
{context}

【问题】: {question}

【回答】:
"""
qa_prompt = ChatPromptTemplate.from_template(qa_prompt_template)

# 初始化向量数据库
vectorstore = Chroma.from_documents(
    persist_directory="./data/chroma_mq", 
    embedding=embedding,
    collection_name="chroma_mq",
    documents=docs
)
# 检索器
retriever = vectorstore.as_retriever()

# 检索测试
relevant_docs= retriever.invoke('deepseek的应用场景')
print("检索测试\n",relevant_docs)
# 查看一下检索到的相关文档的数量：
print("基础检索的文档数量为：",len(relevant_docs))      # 基础检索的文档数量为： 4
base_chain = (
    RunnableParallel({
    "context": (lambda x: x["question"]) | retriever | format_docs,
    "question": lambda x: x["question"]
    }) | qa_prompt | llm | StrOutputParser()
)
base_response = base_chain.invoke({"question": "deepseek的应用场景"})
print("大模型生成的回答：\n",base_response)
"""
大模型生成的回答：
 根据已知信息无法回答该问题。虽然提供了DeepSeek的详细发展历史和部分技术成果，但没有具体说明其应用场景。
"""
# 检索测试-完成

# 创建prompt模版
# template = """请根据下面给出的上下文来回答问题:
# {context}
# 问题: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)

# chain = RunnableParallel({
#     "context": lambda x:relevant_docs,
#     "question": lambda x: x["question"]
# }) | prompt | llm | StrOutputParser()

# response = chain.invoke({"question": "deepseek的应用场景"})
# print("优化前\n大模型生成的不同视角问题为：",response)

# print("="*25,"优化开始","="*25)
import logging
logging.basicConfig()
logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)

# retriever_from_llm = MultiQueryRetriever.from_llm(
#     retriever=retriever,
#     llm=llm
# )

# unique_docs = retriever_from_llm.invoke({"question": "deepseek的应用场景"})
# print("多路查询\n",unique_docs)
# print(len(unique_docs))


# ======================
# 2. Multi-Query 检索器配置
# ======================
print("\n" + "="*25, "Multi-Query 优化开始", "="*25)

base_retriever = vectorstore.as_retriever()

# 创建 MultiQueryRetriever
# 它内部会自动：
# 1. 调用 LLM 生成多个变体问题
# 2. 分别检索
# 3. 去重合并
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)
query = "deepseek的应用场景"
# 执行多路查询检索
unique_docs = multi_query_retriever.invoke({"question": query}) # 注意：MultiQueryRetriever 通常期望输入包含 query 或 question 键
print(f"Multi-Query 检索到的唯一文档数量: {len(unique_docs)}")
"""
INFO:langchain_classic.retrievers.multi_query:Generated queries: ['1. deepseek可以应用于哪些场景？', '2. deepseek在实际中有哪些使用案例？', '3. 使用深求（DeepSeek）技术可以在哪些领域发挥作用？']
Multi-Query 检索到的唯一文档数量: 8
"""


# ======================
# 3. 构建完整的 RAG Chain (动态检索 + 生成)
# ======================


# 构建 Chain
# 流程: 
# 1. 接收 input {"question": "..."}
# 2. 并行处理:
#    - context: 使用 multi_query_retriever 检索文档 -> 格式化为字符串
#    - question: 直接传递原始问题
# 3. 填入 Prompt
# 4. LLM 生成
# 5. 输出解析


rag_chain = (
    RunnableParallel({
        "context": multi_query_retriever | format_docs,  # 关键：将检索器作为 Runnable 放入
        "question": lambda x: x["question"]
    })
    | qa_prompt
    | llm
    | StrOutputParser()
)

# 执行最终问答
print("\n" + "="*25, "最终回答生成", "="*25)
final_answer = rag_chain.invoke({"question": query})
print(final_answer)

"""
INFO:langchain_classic.retrievers.multi_query:Generated queries: ['1. deepseek可以应用于哪些场景？', '2. deepseek在实际中有哪些使用案例？', '3. 使用深求（DeepSeek）技术可以在哪些领域发挥作用？']
根据提供的信息，DeepSeek的模型应用场景可能包括但不限于以下几个方面：

1. **电池相关领域**：DeepSeek在电池知识问答和文本挖掘任务上表现出色，在电池设计任务上有初步总结能力。这表明它可以在电池研发、维护等过程中提供支持。

2. **通用AI应用**：中信证券研报指出，新一代模型的发布意味着AI大模型的应用将逐步走向普惠，助力AI应用广泛落地。这意味着DeepSeek可能适用于各种需要自然语言处理和知识生成的任务场景中，如客户服务、内容创作、教育辅导等。

3. **技术限制与突破**：DeepSeek引发全球轰动和一些人的焦虑恐慌，说明其在某些特定领域的性能已经超越了现有技术水平，这可能意味着它可以在科学研究、工程设计等领域发挥重要作用。例如，在材料科学、药物研发等方面提供辅助分析或预测能力。

4. **开源共享推动技术普及**：DeepSeek公司坚持开放开源的技术路线，通过这种方式可以加速AI技术在全球范围内的应用和推广，促进更多行业和企业利用先进的人工智能技术提升效率和服务质量。

5. **电池设计与优化**：虽然DeepSeek在电池设计任务上尚欠缺科学分析能力，但其初步总结能力表明它可能有助于加快电池相关产品的开发流程，提高设计效率。

综上所述，DeepSeek的应用场景广泛且具有潜力，特别是在需要大量数据处理和复杂问题解决的领域中表现尤为突出。
"""


