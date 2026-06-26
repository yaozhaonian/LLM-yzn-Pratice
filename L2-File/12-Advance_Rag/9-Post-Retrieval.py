# Post-Retrieval后检索优化
# 在完成检索后对检索出的相关知识块做必要补充处理的阶段。比如，对检索的结果借助更专业的排序模型与算法进行重排序或者过滤掉一些不符合条件的知识块等，使得最需要、最合规的知识块处于上下文的最前端，这有助于提高大模型的输出质量。

from pathlib import Path

# LangChain 组件
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
# langchain-classic/retrievers/document_compressors/chain_extract/LLMChainExtractor
# 一种使用大语言模型链提取文档相关部分的文档压缩器。
# langchain-classic/retrievers/document_compressors/chain_filter/LLMChainFilter
# 用于过滤掉与查询不相关的文档的过滤器。
# langchain-classic/retrievers/document_compressors/embeddings_filter/EmbeddingsFilter
# 一种文档压缩器，它使用嵌入向量来剔除与查询无关的文档。
# langchain-classic/retrievers/document_compressors/base/DocumentCompressorPipeline
# 使用 Transformer 管道的文档压缩器
from langchain_classic.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter, EmbeddingsFilter, DocumentCompressorPipeline
# langchain-classic/retrievers/contextual_compression/ContextualCompressionRetriever
# 一个包装基础检索器并对结果进行压缩的检索器。
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever


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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
docs = text_splitter.split_documents(doc)

vectorstore = Chroma.from_documents(
    persist_directory="./data/chroma_hs", 
    embedding=embedding,
    collection_name="chroma_hs",
    documents=docs
)
retriever = vectorstore.as_retriever()

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

lce_compress_retriever = ContextualCompressionRetriever(
    base_compressor=LLMChainExtractor.from_llm(llm),
    base_retriever=retriever
)

lcr_docs = lce_compress_retriever.invoke("deepseek的发展历程")
print("="*25,"LLMChainExtractor压缩","="*25)
pretty_print_docs(lcr_docs)


lcf_compress_retriever = ContextualCompressionRetriever(
    base_compressor=LLMChainFilter.from_llm(llm),
    base_retriever=retriever
)

lcf_docs = lcf_compress_retriever.invoke("deepseek的发展历程")
print("="*25,"LLMChainFilter压缩","="*25)
print(lcf_docs)

# EmbeddingsFilter 通过嵌入文档和查询并仅返回那些与查询具有足够相似嵌入的文档来提供更便宜且更快的选项
ebd_filter = EmbeddingsFilter(embeddings=embedding, similarity_threshold=0.7)
ebd_retriever = ContextualCompressionRetriever(
    base_compressor=ebd_filter,
    base_retriever=retriever
)

ebd_docs = ebd_retriever.invoke("deepseek的产品迭代")
print("="*25,"EmbeddingsFilter压缩","="*25)
pretty_print_docs(ebd_docs)

# langchain-community/document_transformers/embeddings_redundant_filter/EmbeddingsRedundantFilter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
rdd_filter = EmbeddingsRedundantFilter(embeddings=embedding)
rel_filter = EmbeddingsFilter(embeddings=embedding, similarity_threshold=0.7)
pipeline_compressor = DocumentCompressorPipeline(
     transformers = [splitter, rdd_filter, rel_filter]
)

cbp_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=retriever
)
cbp_docs = cbp_retriever.invoke("deepseek的历史活动")
print("="*25,"组合压缩","="*25)
print(cbp_docs)