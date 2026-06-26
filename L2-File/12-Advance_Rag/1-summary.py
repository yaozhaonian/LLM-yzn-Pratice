# 摘要索引,比较擅长处理结构化与半结构化数据
# 使用摘要策略的父文档检索器
from pathlib import Path
import uuid

# LangChain 组件
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.stores import InMemoryByteStore
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel

# 推荐使用新的 langchain-chroma 包以消除警告
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
llm = ChatOllama(model='qwen2.5:7b', temperature=0.1,base_url="http://127.0.0.1:11434")

# 文档分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
docs = text_splitter.split_documents(docs)  

# 创建摘要生成链
summary_chain = (
    {"docs":lambda x: x.page_content}
    | ChatPromptTemplate.from_template("""
    你是一个有用的助手，可以总结文档。
    总结以下文档:
    {docs}
    """)
    | llm
    | StrOutputParser()
)

print("准备生成文档摘要，时间稍长，请耐心等待...")
# 批量生成文档摘要
summary_docs = summary_chain.batch(docs, {"max_concurrency":5})   # 如果不做这一步生成摘要的话,整体就是父文档检索模式了
print(f"生成的文档摘要:\n {summary_docs}")

# 初始化Chroma实例（用于存储摘要向量）
vectorstore = Chroma(
    collection_name="summaries",
    embedding_function=embedding,
)

# 初始化内存字节存储（用于存储原始文档）
store = InMemoryByteStore()

# 初始化多向量检索器(结合向量存储和文档存储)
id_key = "ds_doc_id"
"""
MultiVectorRetriever 的核心设计思想是 “索引与内容分离”：
1.向量数据库 (vectorstore)：只存储摘要（Summaries）或子块（Sub-documents）的向量。这些内容较短，适合语义检索。
2.字节存储 (byte_store)：存储原始大文档（Parent Documents）的二进制序列化数据。
PS.
InMemoryByteStore 是易失性的（Volatile）
它存储在内存中。一旦你的 Python 脚本运行结束，或者服务器重启，所有存储在 store 中的原始文档都会丢失。
如果是长期运行的应用：不要使用 InMemoryByteStore，而是使用支持持久化的 Store，例如 RedisStore 或基于文件的存储。
"""
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

# 为每个文档生成唯一ID，该ID用于关联原始文档和摘要
doc_ids = [str(uuid.uuid4()) for _ in docs]

# 将文档摘要转换为LangChain中Document
summary_docum = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summary_docs)
]

print("将摘要添加到向量数据库...")
retriever.vectorstore.add_documents(summary_docum)

# 将原始文档存储到字节存储（使用ID关联）
print("准备将原始文档存储到字节存储...")
# mset：批量设置键值对
# list(zip(doc_ids, docs))：将ID和文档配对
retriever.docstore.mset(list(zip(doc_ids, docs)))


prompt =  ChatPromptTemplate.from_template("根据下面的文档回答问题:\n\n{doc}\n\n问题: {question}") 
# 生成问题回答链
# retriever.invoke将上面对摘要进行检索，但是通过关联ID获得原始文档，最终返回原始文档的过程全部都包含完成了
chain = RunnableParallel({
    "doc": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()

# 生成问题回答
query = "deepseek的企业事件"
answer = chain.invoke({"question": query})
print("-------------回答--------------")
print(answer)
# retriever.invoke(query)  1.向量数据库中检索摘要向量   2.匹配对应的原始文档并返回
retrieved_docs = retriever.invoke(query) 
print("-------------检索到的文档--------------")
print(retrieved_docs)