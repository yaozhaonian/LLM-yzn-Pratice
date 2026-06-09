import numpy as np
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import jieba
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from rag_core.vector_store import VectorStore
from rag_core.memory_manager import MemoryManager

class BM25Search:
    """BM25 检索"""
    
    def __init__(self, documents: List[str], ids: Optional[List[str]] = None):
        self.documents = documents
        self.ids = ids or [f"doc_{i}" for i in range(len(documents))]
        self.tokenized_corpus = [jieba.lcut(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def search(self, query: str, n_results: Optional[int] = None) -> Dict:
        tokenized_query = jieba.lcut(query)
        scores = self.bm25.get_scores(tokenized_query)
        scores = np.array(scores)
        
        max_score, min_score = scores.max(), scores.min()
        normalized_scores = (scores - min_score) / (max_score - min_score) if max_score > min_score else scores
        
        results = {}
        for idx, doc_id in enumerate(self.ids):
            results[doc_id] = {
                "bm25_score": float(normalized_scores[idx]),
                "document": self.documents[idx]
            }
        
        if n_results:
            top_ids = sorted(results.keys(), key=lambda x: results[x]["bm25_score"], reverse=True)[:n_results]
            return {doc_id: results[doc_id] for doc_id in top_ids}
        
        return results

class HybridSearch:
    """混合搜索"""
    
    def __init__(self, vector_store: VectorStore, documents: List[str], doc_ids: List[str], alpha: float = 0.5):
        self.vector_store = vector_store
        self.bm25_search = BM25Search(documents, doc_ids)
        self.alpha = alpha
    
    def _convert_distance_to_score(self, distance: float) -> float:
        distance = max(0, min(2, distance))
        return max(0.0, min(1.0, (1 - distance + 1) / 2))
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        bm25_results = self.bm25_search.search(query)
        vector_results = self.vector_store.search_all(query)
        
        hybrid_results = {}
        
        for doc_id, bm25_data in bm25_results.items():
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
        
        for doc_id in hybrid_results:
            bm25 = hybrid_results[doc_id]["bm25_score"]
            vec = hybrid_results[doc_id]["vector_score"]
            hybrid_results[doc_id]["hybrid_score"] = self.alpha * vec + (1 - self.alpha) * bm25
        
        sorted_ids = sorted(hybrid_results.keys(), key=lambda x: hybrid_results[x]["hybrid_score"], reverse=True)[:n_results]
        
        return [{
            "document": hybrid_results[doc_id]["document"],
            "metadata": hybrid_results[doc_id]["metadata"],
            "hybrid_score": float(hybrid_results[doc_id]["hybrid_score"]),
            "bm25_score": float(hybrid_results[doc_id]["bm25_score"]),
            "vector_score": float(hybrid_results[doc_id]["vector_score"]),
            "doc_id": doc_id
        } for doc_id in sorted_ids]

class RAGEngine:
    """RAG 引擎（整合 Redis 记忆）"""
    
    def __init__(self, collection_name: str = "default"):
        self.original_name = collection_name
        self.vector_store = VectorStore(collection_name)
        self.documents = []
        self.doc_ids = []
        self.hybrid_search = None
        self.llm_client = ChatOllama(model="qwen2.5:7b", temperature=0.1,base_url="http://127.0.0.1:11434")
        self.memory_manager = MemoryManager()  # 新增：Redis 记忆管理
    
    def load_documents(self, documents: List[Dict]):
        self.documents = documents
        self.doc_ids = []
        
        for i, doc in enumerate(documents):
            doc_id = doc.get('metadata', {}).get('doc_id') or doc.get('metadata', {}).get('file_id', f"doc_{i}")
            self.doc_ids.append(doc_id)
        
        texts = [doc['content'] for doc in documents]
        self.vector_store.add_documents(documents, ids=self.doc_ids)
        
        self.hybrid_search = HybridSearch(self.vector_store, texts, self.doc_ids)
    
    def chat_with_memory(self, user_id: str, query: str, n_results: int = 12) -> Dict:
        """带记忆的对话（核心功能）"""
        if not self.hybrid_search:
            return {"error": "知识库未初始化"}
        
        # 1. 获取历史对话
        history = self.memory_manager.get_history(user_id, self.original_name)
        
        # 2. 混合检索
        search_results = self.hybrid_search.search(query, n_results=n_results)
        
        # 3. 构建上下文
        knowledge_text = ""
        sources = set()
        
        for i, result in enumerate(search_results, 1):
            doc = result.get('document', '')
            metadata = result.get('metadata', {})
            source = metadata.get('source', '未知')
            sources.add(source)
            
            if doc:
                knowledge_text += f"[参考{i}] (来源：{source})\n{doc}\n\n"
        
        # 4. 构建带历史的 Prompt
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-6:]])  # 最近 3 轮
        
        prompt = f"""你是一个专业的问答助手。请**仅根据以下来自【{self.original_name}】知识库的参考知识**回答用户问题。

【历史对话】
{history_text if history_text else "无历史对话"}

【参考知识】（来自知识库：{self.original_name}）
{knowledge_text}

【用户问题】
{query}

【回答要求】
1. 必须基于上述参考知识回答，不要编造信息
2. 如果参考知识中没有相关信息，直接说"根据知识库内容，没有找到相关信息"
3. 结合历史对话上下文，保持对话连贯性
4. 回答简洁明了

【你的回答】
"""
        
        # 5. 调用 LLM
        chain = self.llm_client | StrOutputParser()
        answer_text = chain.invoke(prompt)
        
        # 6. 保存对话到 Redis
        self.memory_manager.append_message(user_id, self.original_name, "user", query)
        self.memory_manager.append_message(user_id, self.original_name, "assistant", answer_text)
        
        return {
            "query": query,
            "knowledge_base": self.original_name,
            "sources": list(sources),
            "search_results": search_results,
            "answer": answer_text,
            "has_history": len(history) > 0
        }
    
    def clear_user_memory(self, user_id: str):
        """清空用户记忆"""
        self.memory_manager.clear_history(user_id, self.original_name)
    
    def generate_answer(self, query: str, n_results: int = 12) -> Dict:
        """无状态问答（兼容旧接口）"""
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
        
        prompt = f"""你是一个专业的问答助手。请**仅根据以下来自【{self.original_name}】知识库的参考知识**回答用户问题。

【参考知识】
{knowledge_text}

【用户问题】
{query}

【你的回答】
"""
        
        chain = self.llm_client | StrOutputParser()
        answer_text = chain.invoke(prompt)
        
        return {
            "query": query,
            "knowledge_base": self.original_name,
            "sources": list(sources),
            "search_results": search_results,
            "answer": answer_text
        }