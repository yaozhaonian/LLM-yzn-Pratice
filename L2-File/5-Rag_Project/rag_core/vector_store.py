# 向量化后根据余弦相似度进行检索
import chromadb
import re
from typing import List, Dict, Optional
from config import CHROMA_DB_DIR
from langchain_ollama import OllamaEmbeddings 


def sanitize_collection_name(name: str) -> str:
    """清理集合名称，使其符合 ChromaDB 要求"""
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    
    if len(sanitized) < 3:
        import hashlib
        sanitized = "kb_" + hashlib.md5(name.encode()).hexdigest()[:8]
    
    sanitized = sanitized[:63]
    sanitized = sanitized.strip('-_')
    if not sanitized[0].isalnum():
        sanitized = 'kb_' + sanitized
    if not sanitized[-1].isalnum():
        sanitized = sanitized + '_0'
    
    return sanitized

class VectorStore:
    """向量存储管理"""
    
    def __init__(self, collection_name: str = "default"):
        self.original_name = collection_name
        self.collection_name = sanitize_collection_name(collection_name)
        
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # config = get_ollama_config(embedding_model=OLLAMA_EMBEDDING_MODEL)
        # self.embedding_client = ModelClientFactory.get_embedding_client(config.embedding)
        self.embedding_client = OllamaEmbeddings( 
            model="bge-m3:latest",
            base_url="http://127.0.0.1:11434",            
        )
    
    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """向量归一化"""
        import numpy as np
        normalized = []
        for emb in embeddings:
            vec = np.array(emb)
            norm = np.linalg.norm(vec)
            if norm > 0:
                normalized.append((vec / norm).tolist())
            else:
                normalized.append(emb)
        return normalized
    
    def add_documents(self, documents: List[Dict], ids: Optional[List[str]] = None):
        """添加文档到向量库"""
        if not documents:
            return
        
        texts = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        embeddings = self.embedding_client.embed_documents(texts)
        embeddings = self._normalize_embeddings(embeddings)
        
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """向量搜索 - 返回 ChromaDB 原始格式"""
        query_embedding = self.embedding_client.embed_query(query)
        query_embedding = self._normalize_embeddings([query_embedding])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    
    def search_all(self, query: str) -> Dict:
        """
        🔴 新增：搜索所有文档，返回按 ID 组织的分数
        用于与 BM25 进行分数融合
        """
        query_embedding = self.embedding_client.embed_query(query)
        query_embedding = self._normalize_embeddings([query_embedding])[0]
        
        # 获取集合中所有文档数量
        count = self.collection.count()
        
        # 查询所有文档
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=count,
            include=["documents", "metadatas", "distances"]
        )
        
        # 按 ID 组织结果
        id_results = {}
        if results["ids"] and results["ids"][0]:
            print('按 ID 组织结果',results["ids"])
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                vector_score = self._convert_distance_to_score(distance)
                
                id_results[doc_id] = {
                    "document": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "vector_score": vector_score
                }
        
        return id_results
    
    def _convert_distance_to_score(self, distance: float) -> float:
        """将余弦距离转换为相似度分数 (0~1)"""
        distance = max(0, min(2, distance))
        similarity = 1 - distance
        score = (similarity + 1) / 2
        return max(0.0, min(1.0, score))
    
    def clear(self):
        """清空集合"""
        ids = self.collection.get()['ids']
        if ids:
            self.collection.delete(ids=ids)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "collection_name": self.original_name,
            "internal_name": self.collection_name,
            "count": self.collection.count()
        }