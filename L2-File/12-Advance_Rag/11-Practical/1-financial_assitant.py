import json
import uuid
import os
from pathlib import Path
from typing import List
import time

# LangChain 组件
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.stores import InMemoryByteStore
from langchain_classic.retrievers import MultiVectorRetriever

# PDF 解析组件
import pdfplumber
from tqdm import tqdm

# --------------------------
# 配置区域
# --------------------------
CACHE_FILE = "summary_cache.json"  # 摘要缓存文件
COLLECTION_NAME = "financial_summary_rag_v2"  # Chroma 集合名称
ID_KEY = "doc_id"  # 元数据中存储父文档ID的键名

# --------------------------
# 1. 初始化模型与嵌入
# --------------------------
# 无显卡环境下，temperature 设为 0 可以减少采样计算开销，略微提升速度
llm = ChatOllama(model='qwen2.5:7b', temperature=0)
embedding = OllamaEmbeddings(model="bge-m3:latest")

# --------------------------
# 2. 文档加载与预处理
# --------------------------
current_dir = Path(__file__).parent.parent
# 请确保路径正确，如果报错请检查文件是否存在
data_file_path = current_dir.parent / "Data" / "金奥博：2025年年度报告.pdf"

if not data_file_path.exists():
    raise FileNotFoundError(f"找不到数据文件: {data_file_path}")

print(f"正在使用 pdfplumber 解析文件: {data_file_path} ...")
start_time = time.time()

def extract_text_from_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n\n"
    return full_text

raw_text = extract_text_from_pdf(data_file_path)

# 初步清洗：去除空行
lines = raw_text.split('\n')
cleaned_lines = [line.strip() for line in lines if line.strip()]
cleaned_text = "\n".join(cleaned_lines)

print(f"原始文本长度: {len(cleaned_text)} 字符, 解析耗时: {time.time() - start_time:.2f}s")

# --------------------------
# 3. 构建父文档 (Parent Documents)
# --------------------------
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)

parent_docs = parent_splitter.create_documents([cleaned_text])
print(f"生成了 {len(parent_docs)} 个父文档 (Parent Chunks)")

# 生成稳定的父文档 ID (基于内容哈希，确保重启后 ID 一致，利于缓存匹配)
def get_stable_id(text: str) -> str:
    # 取前 200 字符生成 UUID5，保证相同内容生成相同 ID
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, text[:200]))

parent_doc_ids = [get_stable_id(doc.page_content) for doc in parent_docs]
parent_docs_map = dict(zip(parent_doc_ids, parent_docs))

# --------------------------
# 4. 构建子文档与摘要 (CPU 优化版：串行 + 缓存)
# --------------------------
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)

# 初始化存储
store = InMemoryByteStore()
vectorstore = Chroma(collection_name=COLLECTION_NAME, embedding_function=embedding)

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=ID_KEY,
)

# 加载缓存
summary_cache = {}
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            summary_cache = json.load(f)
        print(f"✅ 已加载 {len(summary_cache)} 条历史摘要缓存。")
    except Exception as e:
        print(f"⚠️ 缓存加载失败: {e}")

# 创建摘要生成链
summary_prompt = ChatPromptTemplate.from_template("""
你是一个专业的金融文档分析助手。
请简要总结以下文本片段的核心内容。如果包含财务数据，请提及关键指标名称。
保持简洁，用于语义检索。

文本片段:
{text}

摘要:
""")
summary_chain = summary_prompt | llm | StrOutputParser()

# 准备最终存入向量库的子文档列表
child_documents_for_vectorstore = []
new_summaries_count = 0
cached_summaries_count = 0

print("开始处理子文档摘要...")

# 【核心优化】串行遍历，避免 CPU 过载
for i, parent_doc in enumerate(tqdm(parent_docs, desc="处理父文档")):
    p_id = parent_doc_ids[i]
    # 切割子文档
    child_chunks = child_splitter.split_documents([parent_doc])
    
    for chunk in child_chunks:
        # 生成缓存 Key (使用子文档内容的哈希)
        cache_key = str(hash(chunk.page_content))
        
        summary_text = None
        
        # 1. 尝试从缓存获取
        if cache_key in summary_cache:
            summary_text = summary_cache[cache_key]
            cached_summaries_count += 1
        else:
            # 2. 调用 LLM 生成 (串行，CPU 友好)
            try:
                # 限制输入长度，减少推理时间
                content_preview = chunk.page_content[:800]
                summary_text = summary_chain.invoke({"text": content_preview})
                new_summaries_count += 1
                
                # 存入缓存
                summary_cache[cache_key] = summary_text
            except Exception as e:
                print(f"\n⚠️ 摘要生成失败: {e}")
                summary_text = "摘要生成失败"

        # 3. 构建带有父文档 ID 元数据的子文档
        if summary_text:
            summary_doc = Document(
                page_content=summary_text,
                metadata={ID_KEY: p_id}
            )
            child_documents_for_vectorstore.append(summary_doc)

print(f"\n处理完成: 命中缓存 {cached_summaries_count} 条, 新生成 {new_summaries_count} 条。")

# 保存更新后的缓存
if new_summaries_count > 0:
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(summary_cache, f, ensure_ascii=False)
        print("💾 摘要缓存已保存到本地。")
    except Exception as e:
        print(f"⚠️ 缓存保存失败: {e}")

# --------------------------
# 5. 存入向量库与字节存储
# --------------------------
if child_documents_for_vectorstore:
    print(f"正在存入 {len(child_documents_for_vectorstore)} 个摘要向量到 Chroma...")
    try:
        # add_documents 会自动处理向量化
        retriever.vectorstore.add_documents(child_documents_for_vectorstore)
    except Exception as e:
        print(f"⚠️ 向量入库错误: {e}")

print(f"正在存入 {len(parent_docs)} 个父文档到内存存储...")
# 将父文档存入 ByteStore，Key 为 parent_doc_id
parent_items = list(parent_docs_map.items())
if parent_items:
    retriever.docstore.mset(parent_items)

print("✅ 索引构建全部完成！")

# --------------------------
# 6. 构建问答链
# --------------------------
def format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

answer_prompt = ChatPromptTemplate.from_template("""
你是一个专业的金融分析助手。请根据以下提供的【完整上下文片段】回答用户的问题。
这些片段是从年度报告中检索到的较大段落，可能包含表格数据的文本形式。

上下文信息:
{context}

用户问题:
{question}

要求:
1. 如果上下文中包含相关数据，请详细列出并计算增长率。
2. 如果上下文中没有足够信息，请如实告知，不要编造。
3. 回答要专业、准确。

请给出回答:
""")

rag_chain = (
    {
        "context": lambda x: format_docs(retriever.invoke(x["question"])),
        "question": lambda x: x["question"]
    }
    | answer_prompt
    | llm
    | StrOutputParser()
)

# --------------------------
# 7. 测试运行
# --------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🤖 金融助手已就绪 (CPU 模式)")
    print("="*50)
    while True:
        test_question = input("\n请输入问题 (输入 'exit' 退出): ")
        if test_question.lower() == 'exit':
            break
        
        if not test_question.strip():
            continue
            
        print(f"\n🔄 正在检索并思考: {test_question} ...")
        try:
            start_q_time = time.time()
            # 检索
            retrieved_parents = retriever.invoke(test_question)
            print(f"[系统] 检索到 {len(retrieved_parents)} 个相关片段")
            
            # 生成回答
            answer = rag_chain.invoke({"question": test_question})
            
            print(f"\n💡 回答:\n{answer}")
            print(f"[系统] 本次问答耗时: {time.time() - start_q_time:.2f}s")
            
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()