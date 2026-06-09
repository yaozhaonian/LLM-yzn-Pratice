# 使用本地向量模型 bge-m3 与回答模型 qwen3.5:2b，加上向量数据库 chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# ======================
# 1. 模型配置
# ======================
llm = ChatOllama(model="qwen2.5:7b",temperature=0.5,base_url="http://127.0.0.1:11434")


embeddings = OllamaEmbeddings( 
    model="bge-m3:latest",
    base_url="http://127.0.0.1:11434"
)

# ======================
# 2. 加载本地文档
# ======================
# 新的路径写法
from pathlib import Path
script_dir = Path(__file__).parent
print(f"脚本路径：{script_dir}")

script_dir = Path(__file__).parent

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
    print("✗ 未找到数据文件，请创建或指定正确路径")
    print("可选路径:")
    for path in possible_paths:
        print(f"  - {path}")
    # 创建示例文件供测试（现代写法）
    file_path = script_dir / "text.txt"
    file_path.write_text(
        "这是一个测试文档。\nDeepSeek 是一款人工智能模型。\n",
        encoding='utf-8'
    )
    print(f"✓ 已创建测试文件：{file_path}")
r"""
# 旧的写法
import os
script_dir = os.path.dirname(os.path.abspath(__file__))   
尝试多个可能的文件路径
possible_paths = [
    os.path.join(script_dir, "text.txt"),
    os.path.join(script_dir, "../Data/deepseek 百度百科.txt"),
    os.path.join(script_dir, "../../Data/deepseek 百度百科.txt"),
    r"E:\py-file\L2-File\Data\deepseek 百度百科.txt",
]
file_path = None
for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        print(f"✓ 找到文件：{file_path}")
        break
if not file_path:
    print("✗ 未找到数据文件，请创建或指定正确路径")
    print("可选路径:")
    for path in possible_paths:
        print(f"  - {path}")
    # 创建示例文件供测试
    file_path = os.path.join(script_dir, "text.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("这是一个测试文档。\nDeepSeek 是一款人工智能模型。\n")
    print(f"✓ 已创建测试文件：{file_path}")
"""
loader = TextLoader(file_path, encoding='utf-8')
documents = loader.load()
print(f"✓ 加载文档：{len(documents)} 个")

# 切分文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30
)
texts = text_splitter.split_documents(documents)
print(f"✓ 切分文本：{len(texts)} 个片段")

# ======================
# 3. 创建 Chroma 向量库
# ======================
db = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print(f"✓ 向量库创建完成")

# 检索器
retriever = db.as_retriever(search_kwargs={"k": 3})

# ======================
# 4. RAG 问答链
# ======================
qa_prompt = PromptTemplate.from_template("""
根据以下上下文回答问题，如果不知道答案，就说不知道：
上下文：{context}
问题：{question}
回答：
""")

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_prompt}
)

# ======================
# 5. 开始提问
# ======================
import re
if __name__ == '__main__':
    pattern = r"^(exit|退出|quit|结束|q)$"
    while True:
        question = input("\n请输入问题:")
        if bool(re.fullmatch(pattern, question)):
            break
        result = rag_chain.invoke({"query": question})
        print("\n全部result:\n",result)
        print("\n【回答】")
        print(result["result"])

        print("\n【参考资料】")
        for i, doc in enumerate(result["source_documents"]):
            print(f"{i+1}. {doc.page_content}")