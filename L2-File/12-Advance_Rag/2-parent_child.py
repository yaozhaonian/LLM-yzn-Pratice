# 父子索引(基于Langchain)
# 需求相互冲突时:
#   1. 较小的文档块以便它们Embedding以后能够最准确地反映出文档的含义
#   2. 保留较多的内容以得到全面且正确的答案
import os
from pathlib import Path
from typing import List

# LangChain 组件
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from operator import itemgetter
try:
    from langchain_chroma import Chroma
except ImportError:
    # 如果未安装新包，回退到旧包（但会有警告）
    from langchain_community.vectorstores import Chroma

# ======================
# 1. 初始化配置
# ======================
current_dir = Path(__file__).parent
data_file_path = current_dir.parent / "Data" / "deepseek百度百科.txt"

# 检查文件是否存在
if not data_file_path.exists():
    raise FileNotFoundError(f"找不到数据文件: {data_file_path}")

loader = TextLoader(str(data_file_path), encoding="utf-8")
docs = loader.load()

print(f"原始文档数量: {len(docs)}")
print(f"原始文档总长度: {sum(len(d.page_content) for d in docs)}")

embedding = OllamaEmbeddings(model="bge-m3:latest")
llm = ChatOllama(model='qwen2.5:7b', temperature=0.1)

# ======================
# 2. 父子文档检索 (Parent-Child Retrieval)
# ======================
print("\n" + "="*30 + " 开始父子文档检索实验 " + "="*30)

# 父文档：较大块，用于最终返回给 LLM 作为上下文 (提供完整语义)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

# 子文档：较小块，用于向量检索 (提高匹配精度)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)

# 向量存储
# 注意：如果之前运行过且报错，建议手动删除 ./data/chroma_db_parent_child 文件夹以确保干净
vectorstore = Chroma(persist_directory="./data/chroma_db_parent_child", embedding_function=embedding)

# 内存存储：存储父文档 ID -> 父文档内容的映射
store = InMemoryStore()

# 创建父子检索器
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 3} 
)

# 添加文档 (这会自动切分并建立索引)
# 注意：add_documents 可能会比较慢，因为要嵌入很多子块
retriever.add_documents(docs)

# 验证存储
parent_ids = list(store.yield_keys())
print(f"主文块 (Parent Chunks) 的数量: {len(parent_ids)}")
print(f"向量库中子块 (Child Chunks) 的数量: {vectorstore._collection.count()}")

# 定义 Prompt
template = """请根据下面给出的上下文来回答问题。如果上下文中没有答案，请说“我不知道”。

上下文:
{context}

问题: {question}

回答:
"""
prompt = ChatPromptTemplate.from_template(template)

# 【修复点 1】手动构建链，避免 create_retrieval_chain 的键名兼容性问题
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs, 
        "question": itemgetter("question")
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 测试查询
question = "deepseek的应用场景"
print(f"\n正在查询: {question}")

try:
    # 这里依然传入 "question"，因为我们在 chain 定义中明确指定了 "question": RunnablePassthrough()
    response = rag_chain.invoke({"question": question})
    
    print("\n【父子检索结果】:")
    print(response)

except Exception as e:
    print(f"父子检索出错: {e}")
    import traceback
    traceback.print_exc()

# ======================
# 3. 普通检索对比 (Ordinary Retrieval)
# ======================
print("\n" + "="*30 + " 开始普通检索对比实验 " + "="*30)

# 【关键修复】普通检索也必须切分文档！
ordinary_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
split_docs = ordinary_splitter.split_documents(docs)

print(f"切分后的文档片段数量: {len(split_docs)}")

# 创建向量库 (使用不同的路径以避免冲突)
db_ordinary = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory="./data/chroma_db_ordinary"
)

# 创建检索器
ordinary_retriever = db_ordinary.as_retriever(search_kwargs={"k": 3})

# 【修复点 2】同样使用手动构建链的方式
ordinary_rag_chain = (
    {
        "context": itemgetter("question") | ordinary_retriever | format_docs, 
        "question": itemgetter("question")
    }
    | prompt
    | llm
    | StrOutputParser()
)

try:
    ordinary_response = ordinary_rag_chain.invoke({"question": question})
    print("\n【普通检索结果】:")
    print(ordinary_response)
except Exception as e:
    print(f"普通检索出错: {e}")
    import traceback
    traceback.print_exc()

print("\n实验结束。")


"""
原始文档数量: 1
原始文档总长度: 20899

============================== 开始父子文档检索实验 ==============================
主文块 (Parent Chunks) 的数量: 27
向量库中子块 (Child Chunks) 的数量: 375

正在查询: deepseek的应用场景

【父子检索结果】:
根据提供的信息，DeepSeek的模型和API主要应用于大语言模型（LLM）相关的技术开发和服务中。具体来说，DeepSeek发布了多个版本的大语言模型和相关工具，如DeepSeek LLM、DeepSeek-Coder、DeepSeekMath、DeepSeek-VL、DeepSeek-V2、DeepSeek-Coder-V2等，并且还提供了API支持文档。

此外，DeepSeek还在国家超算互联网平台上线了其系列模型，包括DeepSeek-R1、V3和Coder。这表明DeepSeek的应用场景不仅限于研究和技术开发，也扩展到了实际的生产和服务领域，例如通过公共算力服务平台为用户提供服务，并且已经有多家企业接入。

总的来说，DeepSeek的应用场景涵盖了从基础的研究与开发到实际的服务提供等多个方面。

============================== 开始普通检索对比实验 ==============================
切分后的文档片段数量: 56

【普通检索结果】:
根据提供的上下文信息，DeepSeek专注于开发先进的大语言模型（LLM）和相关技术。因此，其应用场景可能涉及自然语言处理(NLP)领域，例如智能客服、虚拟助手、自动摘要生成、文本分类等。

请注意，虽然这些应用是基于提供的信息推测的，但具体的应用场景可能会根据DeepSeek实际开发的具体产品和技术而有所不同。

实验结束。
"""