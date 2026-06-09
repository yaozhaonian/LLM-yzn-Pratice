import numpy as np
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import jieba
from langchain_ollama import ChatOllama
from rag_core.vector_store import VectorStore
from langchain_core.output_parsers import StrOutputParser

class BM25Search:
    """BM25 检索 - 与向量存储共享文档索引"""
    
    def __init__(self, documents: List[str], ids: Optional[List[str]] = None):
        self.documents = documents
        self.ids = ids or [f"doc_{i}" for i in range(len(documents))]
        self.tokenized_corpus = [jieba.lcut(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.ids)}
    
    def search(self, query: str, n_results: Optional[int] = None) -> Dict:
        """返回所有文档的 BM25 分数，按 ID 组织"""
        tokenized_query = jieba.lcut(query)
        scores = self.bm25.get_scores(tokenized_query)
        scores = np.array(scores)
        
        max_score = scores.max()
        min_score = scores.min()
        
        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = scores
        
        results = {}
        for idx, doc_id in enumerate(self.ids):
            results[doc_id] = {
                "bm25_score": float(normalized_scores[idx]),
                "document": self.documents[idx]
            }
        
        if n_results:
            top_ids = sorted(results.keys(), 
                           key=lambda x: results[x]["bm25_score"], 
                           reverse=True)[:n_results]
            return {doc_id: results[doc_id] for doc_id in top_ids}
        
        return results

class HybridSearch:
    """混合搜索 - 正确融合 BM25 和向量检索"""
    
    def __init__(self, vector_store: VectorStore, documents: List[str], 
                 doc_ids: List[str], alpha: float = 0.5):
        self.vector_store = vector_store
        self.bm25_search = BM25Search(documents, doc_ids)
        self.alpha = alpha
        self.doc_ids = doc_ids
    
    def _convert_distance_to_score(self, distance: float) -> float:
        """将余弦距离转换为相似度分数 (0~1)"""
        distance = max(0, min(2, distance))
        similarity = 1 - distance
        score = (similarity + 1) / 2
        return max(0.0, min(1.0, score))
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """正确的混合搜索流程"""
        # 步骤 1: BM25 独立检索
        bm25_results = self.bm25_search.search(query)
        
        # 步骤 2: 向量独立检索
        vector_results = self.vector_store.search_all(query)
        
        # 步骤 3: 按文档 ID 融合分数
        hybrid_results = {}
        
        for doc_id, bm25_data in bm25_results.items():
            if doc_id not in hybrid_results:
                hybrid_results[doc_id] = {
                    "document": bm25_data["document"],
                    "bm25_score": bm25_data["bm25_score"],
                    "vector_score": 0.0,
                    "metadata": {}
                }
        
        for doc_id, vector_data in vector_results.items():
            if doc_id in hybrid_results:
                hybrid_results[doc_id]["vector_score"] = vector_data["vector_score"]
                hybrid_results[doc_id]["metadata"] = vector_data.get("metadata", {})
            else:
                hybrid_results[doc_id] = {
                    "document": vector_data.get("document", ""),
                    "bm25_score": 0.0,
                    "vector_score": vector_data["vector_score"],
                    "metadata": vector_data.get("metadata", {})
                }
        
        # 步骤 4: 计算混合分数
        for doc_id in hybrid_results:
            bm25 = hybrid_results[doc_id]["bm25_score"]
            vec = hybrid_results[doc_id]["vector_score"]
            hybrid_results[doc_id]["hybrid_score"] = (
                self.alpha * vec + (1 - self.alpha) * bm25
            )
        
        # 步骤 5: 按混合分数排序返回 Top-K
        sorted_ids = sorted(hybrid_results.keys(),
                          key=lambda x: hybrid_results[x]["hybrid_score"],
                          reverse=True)[:n_results]
        
        results = []
        for doc_id in sorted_ids:
            data = hybrid_results[doc_id]
            results.append({
                "document": data["document"],
                "metadata": data["metadata"],
                "hybrid_score": float(data["hybrid_score"]),
                "bm25_score": float(data["bm25_score"]),
                "vector_score": float(data["vector_score"]),
                "doc_id": doc_id
            })
        
        return results

class RAGEngine:
    """RAG 引擎"""
    
    def __init__(self, collection_name: str = "default"):
        self.original_name = collection_name
        self.vector_store = VectorStore(collection_name)
        self.documents = []
        self.doc_ids = []
        self.hybrid_search = None
        
        # config = get_ollama_config(llm_model=OLLAMA_LLM_MODEL, temperature=LLM_TEMPERATURE)
        # self.llm_client = ModelClientFactory.get_llm_client(config.llm)
        self.llm_client = ChatOllama(model="qwen2.5:7b",temperature=0.1,base_url="http://127.0.0.1:11434")
    
    def load_documents(self, documents: List[Dict]):
        """
        加载文档 - 确保 BM25 和向量存储使用相同的文档索引
        🔴 注意：此方法内部已调用 add_documents，外部不要再调用
        """
        self.documents = documents
        
        # 生成统一的文档 ID（优先使用 metadata 中的 doc_id）
        self.doc_ids = []
        for i, doc in enumerate(documents):
            doc_id = doc.get('metadata', {}).get('doc_id') or \
                     doc.get('metadata', {}).get('file_id', f"doc_{i}")
            self.doc_ids.append(doc_id)
        
        texts = [doc['content'] for doc in documents]
        
        # 🔴 向量化存储文档（带 ID）- 只调用一次
        self.vector_store.add_documents(documents, ids=self.doc_ids)
        
        # 🔴 BM25 使用相同的文档和 ID
        self.hybrid_search = HybridSearch(
            self.vector_store, 
            texts, 
            self.doc_ids
        )
    
    def generate_answer(self, query: str, n_results: int = 12) -> Dict:
        """生成回答"""
        if not self.hybrid_search:
            return {"error": "知识库未初始化"}
        
        search_results = self.hybrid_search.search(query, n_results=n_results)
        
        knowledge_text = ""
        sources = set()
        
        for i, result in enumerate(search_results, 1):
            doc = result.get('document', '')
            metadata = result.get('metadata', {})
            source = metadata.get('source', '未知')
            sources.add(source)
            
            if doc:
                knowledge_text += f"[参考{i}] (来源：{source})\n{doc}\n\n"
        
        kb_name = self.vector_store.original_name if hasattr(self.vector_store, 'original_name') else "当前知识库"
        
        prompt = f"""你是一个专业的问答助手。请**仅根据以下来自【{kb_name}】知识库的参考知识**回答用户问题。

【参考知识】（来自知识库：{kb_name}）
{knowledge_text}

【用户问题】
{query}

【回答要求】
1. 必须基于上述参考知识回答，不要编造信息
2. 如果参考知识中没有相关信息，直接说"根据知识库内容，没有找到相关信息"
3. 回答简洁明了，**必须完整列出所有相关案例，不得遗漏任何一个**
4. 分条列出所有案例，不要合并、不要简写
5. 不要提及"闭源"、"开源"等与问题无关的信息
6. 不要比较多个知识库，只基于当前选择的知识库回答

【你的回答】
"""
        
        # print(f"\n📋 Prompt 预览（前 500 字符）：\n{prompt[:500]}...")
        print(f"\n📋 Prompt 预览:\n{prompt}...")
        chain = self.llm_client | StrOutputParser()
        answer_text = chain.invoke(prompt)
        
        return {
            "query": query,
            "knowledge_base": kb_name,
            "sources": list(sources),
            "search_results": search_results,
            "answer": answer_text
        }