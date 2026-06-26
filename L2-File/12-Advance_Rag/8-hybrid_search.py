# 混合检索
"""
根据数据特性、查询需求和场景约束，动态组合多种检索技术：
*向量检索*擅长捕捉语义相似性，但可能受限于向量空间的表示能力；
*关键词/全文检索*适合精确匹配，但对自然语言表达不友好
"""
# 例子在前面已经写过了，还用ai优化过了，现在使用langchain提供的EnsembleRetriever简单做个demo
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
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
#langchain-classic/retrievers/ensemble/EnsembleRetriever

embedding = OllamaEmbeddings(model="bge-m3:latest",base_url="http://127.0.0.1:11434")
llm = ChatOllama(model='qwen2.5:7b', temperature=0,base_url="http://127.0.0.1:11434")

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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
docs = text_splitter.split_documents(doc)

vectorstore = Chroma.from_documents(
    persist_directory="./data/chroma_hs", 
    embedding=embedding,
    collection_name="chroma_hs",
    documents=docs
)
# 检索器(默认使用MMR重排)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

question = "相关评价"
doc_vector_retriever = retriever.invoke(question)
print("="*25,"向量检索","="*25)
pretty_print_docs(doc_vector_retriever)

BM25_retriever = BM25Retriever.from_documents(docs)
BM25Retriever.k = 3
doc_BM25Retriever = BM25_retriever.invoke(question)
print("="*25,"BM25检索","="*25)
pretty_print_docs(doc_BM25Retriever)

# 混合检索
# EnsembleRetriever是Langchain集合多个检索器的检索器。
ensem_retriever = EnsembleRetriever(retrievers=[retriever, BM25_retriever], weights=[0.5, 0.5])
retriever_doc = ensem_retriever.invoke(question)
print("="*25,"混合检索","="*25)
pretty_print_docs(retriever_doc)

template = """请根据下面给出的上下文来回答问题:
{context}
问题: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain1 = RunnableParallel({
    "context": lambda x: ensem_retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()
chain2 = RunnableParallel({
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()
print("------------模型回复------------------------")
print("------------向量检索+BM25[0.5, 0.5]------------------------")
print(chain1.invoke({"question":question}))
print("------------向量检索------------------------")
print(chain2.invoke({"question":question}))


"""
MMR (Maximal Marginal Relevance, 最大边际相关性) 是信息检索与 RAG 系统中最常用的结果重排算法，核心作用是：在保证结果与查询相关的同时，最大化结果之间的多样性，避免内容重复冗余。
一、核心原理
MMR 由 Carbonell 和 Goldstein 于 1998 年提出，本质是贪心算法，每一步都选择当前 “最相关、且与已选结果最不重复” 的文档。
1. 核心公式
MMR=λ⋅Sim(Di​,Q)−(1−λ)⋅maxDj​∈S​Sim(Di​,Dj​)
λ (平衡系数，0~1)：
λ→1：只看相关性（传统检索）
λ→0：只看多样性（完全去重）
常用值：0.5~0.7（兼顾相关与多样）


Sim(Di​,Q)：文档Di​与查询Q的相关性（余弦相似度）
maxDj​∈S​Sim(Di​,Dj​)：文档Di​与已选结果集S中所有文档的最大相似度（越低越多样）

2. 执行步骤（贪心选择）

初始：结果集S为空，候选集C为全部检索结果
第一步：选相关性最高的文档加入S（无多样性约束）
循环选后续（直到取够 K 个）：
对剩余候选文档，逐个计算 MMR 分数
选MMR 分数最高的文档加入S
从候选集C中移除该文档

二、为什么要用 MMR？（解决传统检索痛点）

传统检索（仅相关性）：
问题：结果高度相似、信息片面、大量重复
例：搜 “AI”，全是 “AI + 医疗” 的文章


MMR 检索（相关 + 多样）：
效果：覆盖多维度、无重复的信息
例：搜 “AI” → 医疗、教育、伦理、大模型等多篇不同主题
"""




