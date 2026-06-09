"""
对use1的优化版
优化思路
1.检查向量库是否已存在
2.记录已处理文档的哈希值/修改时间
3.对比文档是否有变化
4.支持增量添加新文档
"""
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from pathlib import Path
import hashlib
import json
import re


# ======================
# 1. 模型配置
# ======================
embedding = OllamaEmbeddings(model='bge-m3:latest')
llms = OllamaLLM(model='qwen2.5:7b', temperature=0.1)

# ======================
# 2. 配置路径
# ======================
script_dir = Path(__file__).parent
chroma_db_path = script_dir / "./chroma_db"
cache_file = script_dir / ".doc_cache.json"  # 缓存文件记录文档状态

# 查找数据文件
possible_paths = [
    (script_dir / "text.txt").resolve(),
    (script_dir / "../Data/deepseek 百度百科.txt").resolve(),
    (script_dir / "../../Data/deepseek 百度百科.txt").resolve(),
    Path(r"E:\py-file\L2-File\Data\deepseek 百度百科.txt").resolve(),
]

file_path = None
for path in possible_paths:
    if path.exists():
        file_path = path
        print(f"✓ 找到文件：{file_path}")
        break

if not file_path:
    print("✗ 未找到数据文件，创建测试文件")
    file_path = script_dir / "text.txt"
    file_path.write_text("这是一个测试文档。\nDeepSeek 是一款人工智能模型。\n", encoding='utf-8')
    print(f"✓ 已创建测试文件：{file_path}")

# ======================
# 3. 文档缓存管理
# ======================
def get_file_hash(file_path):
    """计算文件哈希值"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_cache():
    """加载缓存"""
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache_data):
    """保存缓存"""
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)

def check_document_changed(file_path, cache_data):
    """检查文档是否变化"""
    current_hash = get_file_hash(file_path)
    cached_hash = cache_data.get(str(file_path), {}).get('hash')
    return current_hash != cached_hash

# ======================
# 4. 加载或创建向量库
# ======================
cache_data = load_cache()
db = None
need_rebuild = False

# 检查向量库是否存在
if chroma_db_path.exists():
    print(f"✓ 检测到现有向量库：{chroma_db_path}")
    
    # 检查文档是否变化
    if check_document_changed(file_path, cache_data):
        print("⚠ 文档已变更，需要重新构建向量库")
        need_rebuild = True
    else:
        print("✓ 文档未变化，加载现有向量库")
        try:
            db = Chroma(
                persist_directory=str(chroma_db_path),
                embedding_function=embedding
            )
            print(f"✓ 向量库加载完成")
        except Exception as e:
            print(f"⚠ 加载失败：{e}，重新构建向量库")
            need_rebuild = True
else:
    print("⚠ 未检测到向量库，需要创建")
    need_rebuild = True

# 需要重建向量库
if need_rebuild:
    print("\n【开始构建向量库】")
    
    # 加载文档
    loader = TextLoader(str(file_path), encoding='utf-8')
    documents = loader.load()
    print(f"✓ 加载文档：{len(documents)} 个")

    # 切分文本
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30
    )
    texts = text_splitter.split_documents(documents)
    print(f"✓ 切分文本：{len(texts)} 个片段")

    # 创建向量库
    db = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=str(chroma_db_path)
    )
    print(f"✓ 向量库创建完成")

    # 更新缓存
    cache_data[str(file_path)] = {
        'hash': get_file_hash(file_path),
        'chunks': len(texts),
        'updated': str(Path(file_path).stat().st_mtime)
    }
    save_cache(cache_data)
    print(f"✓ 缓存已更新")

# ======================
# 5. 创建检索器和问答链
# ======================
retriever = db.as_retriever(search_kwargs={"k": 3})

qa_prompt = PromptTemplate.from_template("""
根据以下上下文回答问题，如果不知道答案，就说不知道：
上下文：{context}
问题：{question}
回答：
""")

rag_chain = RetrievalQA.from_chain_type(
    llm=llms,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_prompt}
)

# ======================
# 6. 开始提问
# ======================
print("\n" + "="*50)
print("RAG 问答系统已就绪，输入 '退出' 结束对话")
print("="*50)

if __name__ == '__main__':
    pattern = r"^(exit|退出|quit|结束|q)$"
    while True:
        question = input("\n请输入问题:")
        if bool(re.fullmatch(pattern, question, re.IGNORECASE)):
            print("👋 再见！")
            break
        
        try:
            result = rag_chain.invoke({"query": question})
            print("\n【回答】")
            print(result["result"])

            print("\n【参考资料】")
            for i, doc in enumerate(result["source_documents"]):
                print(f"{i+1}. {doc.page_content[:100]}...")
        except Exception as e:
            print(f"⚠ 查询出错：{e}")