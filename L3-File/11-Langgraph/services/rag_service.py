import os
import json
import subprocess
from typing import List, Optional
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.ollama import Ollama
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGServiceError(Exception):
    """RAG服务异常基类"""
    pass


class MilvusConnectionError(RAGServiceError):
    """Milvus连接异常"""
    pass


class DocumentLoadError(RAGServiceError):
    """文档加载异常"""
    pass


class EmbeddingError(RAGServiceError):
    """嵌入模型调用异常"""
    pass


class CustomOllamaEmbedding(BaseEmbedding):
    """
    自定义Ollama嵌入模型，使用subprocess调用curl
    
    解决ollama Python客户端502错误问题。
    """

    def __init__(self, model_name: str, base_url: str, **kwargs):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._base_url = base_url.replace("http://", "").replace("https://", "")

    def _get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入向量"""
        url = f"http://{self._base_url}/api/embeddings"
        data = json.dumps({
            "model": self._model_name,
            "prompt": text
        })
        
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", url,
             "-H", "Content-Type: application/json",
             "-d", data],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            raise EmbeddingError(f"嵌入调用失败: {result.stderr}")
        
        try:
            response = json.loads(result.stdout)
            return response.get("embedding", [])
        except json.JSONDecodeError:
            raise EmbeddingError(f"嵌入响应解析失败: {result.stdout[:200]}")

    def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询嵌入向量"""
        return self._get_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """获取文本嵌入向量"""
        return self._get_embedding(text)

    @property
    def _model_name(self) -> str:
        """模型名称"""
        return self.__model_name

    @_model_name.setter
    def _model_name(self, value: str):
        self.__model_name = value

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本嵌入向量"""
        return [self._get_embedding(text) for text in texts]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步获取查询嵌入向量"""
        return self._get_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步获取文本嵌入向量"""
        return self._get_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """异步批量获取文本嵌入向量"""
        return [self._get_embedding(text) for text in texts]


class RAGService:
    """
    RAG向量知识库服务
    
    基于LlamaIndex和Milvus实现文档向量化存储与检索，
    使用Ollama本地模型作为LLM和Embedding数据源。
    """

    def __init__(self):
        """初始化RAG服务"""
        self._milvus_host = settings.milvus.host
        self._milvus_port = settings.milvus.port
        self._milvus_username = settings.milvus.username
        self._milvus_password = settings.milvus.password
        self._collection_name = settings.milvus.collection_name
        self._embedding_model = settings.embedding.model
        self._embedding_dim = settings.embedding.dimension
        self._llm_model = settings.ollama.model
        self._ollama_base_url = settings.ollama.base_url.replace("http://", "").replace("https://", "")
        self._top_k = settings.rag.top_k

        self._vector_store = None
        self._index = None

        self._init_llama_index_settings()
        logger.info(f"RAG服务初始化完成，Milvus: {self._milvus_host}:{self._milvus_port}, "
                    f"嵌入模型: {self._embedding_model}, 向量维度: {self._embedding_dim}")

    def _init_llama_index_settings(self):
        """
        初始化LlamaIndex全局设置
        
        配置LLM、Embedding模型和文档分割器。
        """
        Settings.llm = Ollama(
            model=self._llm_model,
            base_url=f"http://{self._ollama_base_url}"
        )
        Settings.embed_model = CustomOllamaEmbedding(
            model_name=self._embedding_model,
            base_url=self._ollama_base_url
        )
        Settings.text_splitter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50
        )

    def _ensure_vector_store(self) -> MilvusVectorStore:
        """
        确保Milvus向量存储已创建
        
        懒加载模式，首次调用时创建并缓存。
        
        Returns:
            MilvusVectorStore: Milvus向量存储实例
        
        Raises:
            MilvusConnectionError: Milvus连接失败
        """
        if self._vector_store is None:
            try:
                from pymilvus import MilvusClient
                client = MilvusClient(
                    uri=f"http://{self._milvus_host}:{self._milvus_port}",
                    user=self._milvus_username or None,
                    password=self._milvus_password or None
                )
                if client.has_collection(self._collection_name):
                    collection_info = client.describe_collection(self._collection_name)
                    old_dim = collection_info.get('dimension', 0)
                    if old_dim != self._embedding_dim:
                        logger.warning(f"集合维度不匹配，重建集合: {self._collection_name}")
                        client.drop_collection(self._collection_name)
                
                self._vector_store = MilvusVectorStore(
                    host=self._milvus_host,
                    port=self._milvus_port,
                    user=self._milvus_username or None,
                    password=self._milvus_password or None,
                    collection_name=self._collection_name,
                    dim=self._embedding_dim,
                    overwrite=False
                )
                self._vector_store.client.load_collection(self._collection_name)
                logger.info(f"Milvus向量存储创建成功，集合: {self._collection_name}")
            except Exception as e:
                logger.error(f"Milvus连接失败: {str(e)}")
                raise MilvusConnectionError(f"Milvus连接失败，请检查服务是否启动: {str(e)}")
        return self._vector_store

    def _ensure_index(self) -> VectorStoreIndex:
        """
        确保向量索引已创建
        
        懒加载模式，首次调用时创建并缓存。
        
        Returns:
            VectorStoreIndex: 向量索引实例
        
        Raises:
            MilvusConnectionError: Milvus连接失败
        """
        if self._index is None:
            vector_store = self._ensure_vector_store()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context
            )
            logger.info(f"向量索引创建成功，集合: {self._collection_name}")
        return self._index

    def build_index_from_docs(self, doc_dir: str) -> None:
        """
        从文档目录构建向量索引
        
        加载指定目录下的所有文档，自动切片、向量化，存入Milvus向量库。
        
        Args:
            doc_dir: 文档目录路径
        
        Raises:
            DocumentLoadError: 文档加载失败或目录为空
            MilvusConnectionError: Milvus连接失败
            EmbeddingError: 嵌入模型调用失败
            RAGServiceError: 其他服务异常
        """
        try:
            if not os.path.exists(doc_dir):
                logger.error(f"文档目录不存在: {doc_dir}")
                raise DocumentLoadError(f"文档目录不存在: {doc_dir}")

            if not os.listdir(doc_dir):
                logger.error(f"文档目录为空: {doc_dir}")
                raise DocumentLoadError(f"文档目录为空: {doc_dir}")

            logger.info(f"开始加载文档，目录: {doc_dir}")
            documents = SimpleDirectoryReader(doc_dir).load_data()
            logger.info(f"文档加载完成，数量: {len(documents)}")

            if not documents:
                logger.error(f"未加载到任何文档: {doc_dir}")
                raise DocumentLoadError(f"未加载到任何文档: {doc_dir}")

            vector_store = self._ensure_vector_store()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            logger.info("开始构建向量索引...")
            self._index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context
            )
            logger.info(f"向量索引构建完成，文档数: {len(documents)}")

        except DocumentLoadError:
            raise
        except MilvusConnectionError:
            raise
        except Exception as e:
            logger.error(f"构建向量索引异常: {str(e)}")
            if "embedding" in str(e).lower() or "ollama" in str(e).lower():
                raise EmbeddingError(f"嵌入模型调用失败: {str(e)}")
            raise RAGServiceError(f"构建向量索引异常: {str(e)}")

    def retrieve_relevant_docs(self, query: str, top_k: int = None) -> List[dict]:
        """
        根据用户问题召回最相关的文档片段
        
        Args:
            query: 用户查询问题
            top_k: 返回的文档数量，默认使用配置值
        
        Returns:
            List[dict]: 包含文档内容、相似度分数等信息的字典列表
        
        Raises:
            MilvusConnectionError: Milvus连接失败
            EmbeddingError: 嵌入模型调用失败
            RAGServiceError: 其他服务异常
        """
        try:
            if not query or not query.strip():
                logger.warning("查询问题为空")
                return []

            k = top_k if top_k is not None else self._top_k
            logger.info(f"开始检索相关文档，查询: {query[:50]}..., top_k: {k}")

            index = self._ensure_index()
            retriever = index.as_retriever(similarity_top_k=k)
            nodes = retriever.retrieve(query)

            results = []
            for node in nodes:
                results.append({
                    "content": node.text,
                    "score": node.score,
                    "metadata": node.metadata,
                    "id": node.node_id
                })

            logger.info(f"检索完成，返回文档数: {len(results)}")
            return results

        except MilvusConnectionError:
            raise
        except Exception as e:
            logger.error(f"检索文档异常: {str(e)}")
            if "embedding" in str(e).lower() or "ollama" in str(e).lower():
                raise EmbeddingError(f"嵌入模型调用失败: {str(e)}")
            raise RAGServiceError(f"检索文档异常: {str(e)}")


rag_service = RAGService()
