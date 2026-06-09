# 混合搜索，结合了 BM25 和 密集向量检索两种方法,使用本地聊天大模型与向量模型
import numpy as np
from rank_bm25 import BM25Okapi
import jieba
import json
from pathlib import Path
import chromadb
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 一开始没用langchain的，这些是回头改的，以前用rag_models.models,做法是把rag_models.models打包成一个包，然后导入使用
from langchain_ollama import OllamaEmbeddings 
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b",temperature=0.5)


embeddings = OllamaEmbeddings( 
    model="bge-m3:latest",
    base_url="http://127.0.0.1:11434"
)


# ======================
# BM25 全文检索类
# ======================
class BM25Search:
    """BM25 全文检索封装"""
    
    def __init__(self, documents: List[str]):
        """
        初始化 BM25 检索
        
        Args:
            documents: 文档列表
        """
        self.documents = documents
        # 中文分词
        self.tokenized_corpus = [jieba.lcut(doc) for doc in documents]
        # 初始化 BM25
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def search(self, query: str, n_results: Optional[int] = None) -> np.ndarray:
        """
        执行 BM25 搜索
        
        Args:
            query: 查询文本
            n_results: 返回结果数量（None 则返回所有分数）
            
        Returns:
            归一化后的 BM25 分数数组
        """
        # 查询分词
        tokenized_query = jieba.lcut(query)
        
        # 计算分数
        scores = self.bm25.get_scores(tokenized_query)
        scores = np.array(scores)
        
        # 归一化到 [0,1]
        max_score = scores.max()
        min_score = scores.min()
        
        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = scores
        
        return normalized_scores
    
    def get_top_n(self, query: str, n: int = 3) -> List[Dict]:
        """
        获取 Top-N 结果(单独时用，混合时不太能用得上)
        
        Args:
            query: 查询文本
            n: 结果数量
            
        Returns:
            包含文档、分数的结果列表
        """
        scores = self.search(query)
        top_indices = np.argsort(scores)[-n:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "score": float(scores[idx]),
                "index": int(idx)
            })
        
        return results


# ======================
# 向量数据库连接器
# ======================
class MyVectorDBConnector:
    def __init__(self, collection_name: str = "cdb_collection", persist_directory: str = "./chroma_db"):
        """
        初始化向量数据库连接
        
        Args:
            collection_name: 集合名称
            persist_directory: 数据持久化目录
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # 尝试初始化，失败则删除旧数据库
        try:
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
            print(f"✓ 向量数据库已初始化：{collection_name} @ {persist_directory}")
        except chromadb.errors.InternalError as e:
            if "mismatched types" in str(e) or "BLOB" in str(e):
                print(f"⚠ 检测到旧数据库格式不兼容，正在删除并重建...")
                import shutil
                if Path(persist_directory).exists():
                    shutil.rmtree(persist_directory)
                
                # 重新初始化
                self.chroma_client = chromadb.PersistentClient(path=persist_directory)
                self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
                print(f"✓ 向量数据库已重建：{collection_name} @ {persist_directory}")
            else:
                raise

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None,
                      ids: Optional[List[str]] = None, clear_first: bool = True):
        """
        批量添加文档到向量库
        """
        if not texts:
            print("⚠ 无文档可添加")
            return

        if clear_first:
            self.clear()
        
        print(f"正在向量化{len(texts)}个文档...")
        ad_embeddings = embeddings.embed_documents(texts)

        if ids is None:
            ids = [f'doc_{i}' for i in range(len(texts))]

        self.collection.add(
            ids=ids,
            embeddings=ad_embeddings,
            documents=texts,
            metadatas=metadatas
        )

        print(f"✓ 向量存储完成，共 {len(texts)} 个文档")

    def search(self, query: str, n_results: int = 5) -> Dict:
        """向量相似度搜索"""
        query_embedding = embeddings.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        return results
    
    def get_stats(self) -> Dict:
        """获取集合统计信息"""
        return {
            "collection_name": self.collection_name,
            "count": self.collection.count(),
            "persist_directory": self.persist_directory
        }
    
    def clear(self):
        """清空集合"""
        try:
            ids = self.collection.get()["ids"]
            if ids:
                self.collection.delete(ids=ids)
                print(f"✓ 清空集合 {self.collection_name} 成功")
        except Exception as e:
            print(f"⚠ 清空集合时出错：{e}")

    def delete(self):
        """删除集合"""
        self.chroma_client.delete_collection(self.collection_name)
        print(f"✓ 删除集合 {self.collection_name} 成功")    

# ======================
# 混合搜索类
# ======================
class HybridSearch:
    """ 混合搜索(向量 + BM25) """
    def __init__(self,vector_store: MyVectorDBConnector, bm25_search: BM25Search, alpha: float = 0.5):
        """
        初始化混合搜索
        
        Args:
            vector_store: 向量数据库连接器
            bm25_search: BM25 检索器
            alpha: 向量搜索权重（0-1）
        """
        self.vector_store = vector_store
        self.bm25_search = bm25_search
        self.alpha = alpha

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        执行混合搜索
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            
        Returns:
            排序后的结果列表
        """
        # BM25 搜索
        bm25_scores = self.bm25_search.search(query)

        # 向量搜索
        vector_results = self.vector_store.search(query, n_results=len(bm25_scores))

        # 获取向量分数(距离转相似度)
        vector_distances = vector_results["distances"][0] if vector_results["distances"] else []
        vector_scores = [1 - d for d in vector_distances]

        # 填充到相同长度
        while len(vector_scores) < len(bm25_scores): vector_scores.append(0)

        # 加权融合
        hybrid_scores = []
        for i in range(len(bm25_scores)):
            score = self.alpha * vector_scores[i] + (1 - self.alpha) * bm25_scores[i]
            hybrid_scores.append(score)

        # 获取 Top-N
        top_indices = np.argsort(hybrid_scores)[-n_results:][::-1]
        print("top_indices为:",top_indices)

        # 构建结果
        results = []
        for idx in top_indices:
             # 安全获取文档和元数据（添加空值检查）
            doc = vector_results["documents"][0][idx] if idx < len(vector_results['documents'][0]) else None
            meta = vector_results["metadatas"][0][idx] if idx < len(vector_results['metadatas'][0]) else None
            results.append({
                "document": doc,
                "metadata": meta if meta else {},
                "hybrid_score":float(hybrid_scores[idx]), 
                "bm25_score":float(bm25_scores[idx]),
                "vector_score":float(vector_scores[idx])
            })

        return results

# ======================
# 数据加载
# ======================
def find_data_file(script_dir: Path, data_path: str = "deepseek百度百科.txt") -> Path:
    """查找数据文件"""
    possible_paths = [
        # 全部使用 / 连接，不要用字符串拼接
        (script_dir / ".." / ".." / "Data" / data_path).resolve(),
        (script_dir / ".." / "Data" / data_path).resolve()
    ]
    
    print("正在查找数据文件...")
    for path in possible_paths:
        print(f"  检查：{path}")
        if path.exists():
            print(f"✓ 找到文件：{path}")
            return path
    
    # 创建测试文件
    file_path = script_dir / "deepseek百度百科.txt"
    if not file_path.exists():
        print("找不到相应的文档，创建测试文件")
        # file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            'deepseek can use tokens to answer your question\n',
            encoding='utf-8'
        )
        print(f"✓ 已创建测试文件：{file_path}")
    else:
        print(f"✓ 使用本地测试文件：{file_path}")
    return file_path

# 文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 分割长度（建议 300-800）
    chunk_overlap=50,  # 重叠长度（建议 50-100）
    separators=["\n\n", "\n", "。", "，", ""]
)

def load_data(file_path: Path) -> List[Dict]:
    """加载数据文件（支持多种格式）"""
    ext = file_path.suffix.lower()
    documents = []
    
    if ext == '.docx':
        # Word 文档
        from docx import Document
        doc = Document(file_path)
        
        full_text = ""
        for i, para in enumerate(doc.paragraphs):
            if para.text.strip():
                full_text += para.text + "\n\n"
        
        # 分割文本
        chunks = text_splitter.split_text(full_text)
        documents = [{"text": chunk, "metadata": {"source": file_path.name, "type": "docx"}} for chunk in chunks]
        
    elif ext == '.pdf':
        # PDF 文档
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        
        full_text = ""
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                full_text += f"\n\n[第{i+1}页]\n{text}"
        
        chunks = text_splitter.split_text(full_text)
        documents = [{"text": chunk, "metadata": {"source": file_path.name, "type": "pdf"}} for chunk in chunks]
        
    elif ext == '.txt':
        # 纯文本
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = text_splitter.split_text(content)
        documents = [{"text": chunk, "metadata": {"source": file_path.name, "type": "txt"}} for chunk in chunks]
        
    elif ext == '.json':
        # JSON Lines 格式
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                if line.strip():
                    try:
                        data = json.loads(line)
                        text = f"问题：{data.get('instruction', '')}\n答案：{data.get('output', '')}"
                        documents.append({
                            "text": text,
                            "metadata": {"source": file_path.name, "type": "json", "line": i, **data}
                        })
                    except:
                        documents.append({
                            "text": line.strip(),
                            "metadata": {"source": file_path.name, "type": "json", "line": i}
                        })
    
    elif ext == '.md':
        # Markdown 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = text_splitter.split_text(content)
        documents = [{"text": chunk, "metadata": {"source": file_path.name, "type": "md"}} for chunk in chunks]
    
    else:
        raise ValueError(f"不支持的文件格式：{ext}")
    
    print(f"✓ 文件 '{file_path.name}' 加载完成，共 {len(documents)} 个文档块")
    return documents

# ======================
# RAG 结果生成类
# ======================
class RAGGenerator:
    """基于检索结果的大模型回答生成器"""
    
    def __init__(self, llm_client, hybrid_search: HybridSearch):
        self.llm_client = llm_client
        self.hybrid_search = hybrid_search
    
    def build_prompt(self, query: str, search_results: List[Dict]) -> str:
        """构建 RAG prompt"""
        # 构建参考知识部分
        knowledge_text = ""
        for i, result in enumerate(search_results, 1):
            doc = result.get('document', '')
            metadata = result.get('metadata', {})
            source = metadata.get('source', '未知')
            
            if doc:
                knowledge_text += f"[参考{i}] (来源：{source})\n{doc}\n\n"
        
        
        # 构建 Prompt
        prompt = f"""你是一个专业的问答助手。请**仅根据以下参考知识**回答用户问题。

【参考知识】
{knowledge_text}

【用户问题】
{query}

【回答要求】
1. **必须基于上述参考知识回答**，不要编造信息
2. 如果参考知识中没有相关信息，直接说"根据知识库内容，没有找到相关信息"
3. 回答简洁明了，直接给出答案
4. 不要提及与问题无关的信息

【你的回答】
"""
        return prompt
    
    def generate(self, query: str, n_results: int = 3) -> Dict:
        """执行 RAG 生成"""
        search_results = self.hybrid_search.search(query, n_results=n_results)
        prompt = self.build_prompt(query, search_results)
        
        print("正在调用大模型生成回答...")
        response = self.llm_client.invoke(prompt)

        # 从 AIMessage 对象中提取 content
        if hasattr(response, 'content'):
            answer_text = response.content
        elif isinstance(response, str):
            answer_text = response
        else:
            answer_text = str(response)
        
        return {
            "query": query,
            "search_results": search_results,
            "prompt": prompt,
            "answer": answer_text
        }
    
    def chat(self, query: str, n_results: int = 3, verbose: bool = True) -> str:
        """简化版聊天接口"""
        result = self.generate(query, n_results)
        
        if verbose:
            print("\n" + "="*50)
            print("📋 检索到的参考知识：")
            for i, r in enumerate(result['search_results'], 1):
                print(f"{i}. {r['document'][:250]}...")
            print("="*50)
            print("🤖 AI 回答：")
            print(result['answer'])
            print("="*50)
        
        return result['answer']

# ======================
# 主程序
# ======================
# ======================
# 主程序
# ======================
if __name__ == "__main__":
    # 检查是否有旧数据库
    db_path = Path("./chroma_db")
    if db_path.exists():
        print(f"⚠ 发现现有数据库：{db_path}")
        response = input("是否删除并重建？(y/n): ")
        if response.lower() == 'y':
            import shutil
            shutil.rmtree(db_path)
            print("✓ 已删除旧数据库")

    # 加载数据
    script_dir = Path(__file__).parent
    file_path = find_data_file(script_dir)
    data = load_data(file_path)

    # 提取文本和元数据
    texts = [entry['text'] for entry in data]
    metadatas = [entry.get('metadata', {}) for entry in data]

    print(f"数据总量：{len(data)}")
    print(f"文档块数量：{len(texts)}")
    print('=' * 50)
    
    # 显示前 3 个文档块预览
    for i, text in enumerate(texts[:3]):
        print(f"[文档块{i+1}] {text[:100]}...")
    print('=' * 50)

    # 初始化各组件
    vector_store = MyVectorDBConnector(
        collection_name="cdb_collection",
        persist_directory="./chroma_db"
    )

    bm25_search = BM25Search(documents=texts)

    # 添加文档到向量库
    vector_store.add_documents(texts=texts, metadatas=metadatas, clear_first=True)

    # 初始化混合检索
    hybrid_search = HybridSearch(vector_store, bm25_search, alpha=0.5)
    
    # 初始化 RAG 生成器
    rag_generator = RAGGenerator(llm, hybrid_search)

    # 测试搜索（根据实际内容修改问题）
    query = input("请输入问题：")
    print(f"\n查询：{query}")
    print('=' * 50)

    # BM25 搜索
    print("BM25 搜索结果：")
    bm25_results = bm25_search.get_top_n(query, n=3)
    for i, r in enumerate(bm25_results):
        print(f"{i+1}. {r['document'][:100]}...(分数:{r['score']:.4f})")

    # 混合检索 + 大模型生成
    print("\n🤖 RAG 生成回答：")
    print('=' * 50)
    answer = rag_generator.chat(query, n_results=3, verbose=True)

    print("\n✓ 测试完成！")
