# models.py
"""
通用模型接口封装 - 修复版
支持：本地 Ollama 模型、阿里云百炼在线模型、其他 OpenAI 兼容接口
"""
import os
import inspect
from typing import Optional, List, Dict, Any, Iterator
from dataclasses import dataclass, field
from enum import Enum


# ======================
# 1. 模型类型枚举
# ======================
class ModelProvider(Enum):
    """模型提供商"""
    OLLAMA = "ollama"
    ALI_BAILIAN = "ali_bailian"
    OPENAI = "openai"
    CUSTOM = "custom"


# ======================
# 2. 模型配置类
# ======================
@dataclass
class EmbeddingConfig:
    """Embedding 模型配置"""
    provider: ModelProvider = ModelProvider.OLLAMA
    model_name: str = "bge-m3:latest"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    dimension: int = 1024


@dataclass
class LLMConfig:
    """LLM 模型配置"""
    provider: ModelProvider = ModelProvider.OLLAMA
    model_name: str = "qwen2.5:7b"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2048


@dataclass
class ModelConfig:
    """总模型配置"""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


# ======================
# 3. 阿里云百炼配置常量
# ======================
ALI_BAILIAN_API_KEY_VAR = "DASHSCOPE_API_KEY"
ALI_BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALI_BAILIAN_EMBEDDING_MODEL = "text-embedding-v4"
ALI_BAILIAN_LLM_MODEL = "qwen-plus"


# ======================
# 4. LangChain 兼容的 LLM 封装
# ======================
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun


class LangChainOllamaLLM(LLM):
    """LangChain 兼容的 Ollama LLM 包装器"""
    
    model_name: str = "qwen2.5:7b"
    temperature: float = 0.1
    _client: Any = None
    
    def __init__(self, model_name: str = "qwen2.5:7b", temperature: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.temperature = temperature
        from langchain_ollama import OllamaLLM
        self._client = OllamaLLM(model=model_name, temperature=temperature)
    
    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """执行 LLM 调用"""
        return self._client.invoke(prompt)
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature}


class LangChainOpenAICompatibleLLM(LLM):
    """LangChain 兼容的 OpenAI 兼容接口 LLM 包装器"""
    
    model_name: str = "qwen-plus"
    temperature: float = 0.1
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    _client: Any = None
    
    def __init__(
        self,
        model_name: str = "qwen-plus",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv(ALI_BAILIAN_API_KEY_VAR)
        self.base_url = base_url or ALI_BAILIAN_BASE_URL
        
        from openai import OpenAI
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    @property
    def _llm_type(self) -> str:
        return "openai_compatible"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """执行 LLM 调用"""
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=2048
        )
        return response.choices[0].message.content
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "base_url": self.base_url
        }


# ======================
# 5. Embedding 客户端（保持原样，Chroma 需要的是 embed_documents 方法）
# ======================
class OllamaEmbeddingClient:
    """Ollama Embedding 客户端"""
    
    def __init__(self, config: EmbeddingConfig):
        from langchain_ollama import OllamaEmbeddings
        self.model = config.model_name
        self.client = OllamaEmbeddings(model=self.model)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化"""
        return self.client.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """单条向量化"""
        return self.client.embed_query(text)


class OpenAICompatibleEmbeddingClient:
    """OpenAI 兼容接口 Embedding 客户端"""
    
    def __init__(self, config: EmbeddingConfig):
        from openai import OpenAI
        
        api_key = config.api_key or os.getenv(ALI_BAILIAN_API_KEY_VAR)
        base_url = config.base_url or ALI_BAILIAN_BASE_URL
        model = config.model_name or ALI_BAILIAN_EMBEDDING_MODEL
        
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化"""
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [x.embedding for x in response.data]
    
    def embed_query(self, text: str) -> List[float]:
        """单条向量化"""
        return self.embed_documents([text])[0]


# ======================
# 6. 客户端工厂类
# ======================
class ModelClientFactory:
    """模型客户端工厂"""
    
    @staticmethod
    def get_embedding_client(config: EmbeddingConfig):
        """获取 Embedding 客户端"""
        if config.provider == ModelProvider.OLLAMA:
            return OllamaEmbeddingClient(config)
        elif config.provider in [ModelProvider.ALI_BAILIAN, ModelProvider.OPENAI, ModelProvider.CUSTOM]:
            return OpenAICompatibleEmbeddingClient(config)
        else:
            raise ValueError(f"不支持的 Embedding 提供商：{config.provider}")
    
    @staticmethod
    def get_llm_client(config: LLMConfig, use_langchain_wrapper: bool = True):
        """
        获取 LLM 客户端
        use_langchain_wrapper: 是否返回 LangChain 兼容的 LLM 实例（用于 RetrievalQA）
        """
        if config.provider == ModelProvider.OLLAMA:
            if use_langchain_wrapper:
                return LangChainOllamaLLM(
                    model_name=config.model_name,
                    temperature=config.temperature
                )
            else:
                from langchain_ollama import OllamaLLM
                return OllamaLLM(model=config.model_name, temperature=config.temperature)
        
        elif config.provider in [ModelProvider.ALI_BAILIAN, ModelProvider.OPENAI, ModelProvider.CUSTOM]:
            if use_langchain_wrapper:
                return LangChainOpenAICompatibleLLM(
                    model_name=config.model_name,
                    temperature=config.temperature,
                    api_key=config.api_key,
                    base_url=config.base_url
                )
            else:
                from openai import OpenAI
                api_key = config.api_key or os.getenv(ALI_BAILIAN_API_KEY_VAR)
                base_url = config.base_url or ALI_BAILIAN_BASE_URL
                return OpenAI(api_key=api_key, base_url=base_url)
        
        else:
            raise ValueError(f"不支持的 LLM 提供商：{config.provider}")


# ======================
# 7. 便捷配置函数
# ======================
def get_ollama_config(embedding_model: str = "bge-m3:latest",
                      llm_model: str = "qwen2.5:7b",
                      temperature: float = 0.1) -> ModelConfig:
    """获取 Ollama 本地模型配置"""
    return ModelConfig(
        embedding=EmbeddingConfig(
            provider=ModelProvider.OLLAMA,
            model_name=embedding_model
        ),
        llm=LLMConfig(
            provider=ModelProvider.OLLAMA,
            model_name=llm_model,
            temperature=temperature
        )
    )


def get_ali_bailian_config(embedding_model: str = ALI_BAILIAN_EMBEDDING_MODEL,
                           llm_model: str = ALI_BAILIAN_LLM_MODEL,
                           api_key: Optional[str] = None,
                           temperature: float = 0.1) -> ModelConfig:
    """获取阿里云百炼模型配置"""
    return ModelConfig(
        embedding=EmbeddingConfig(
            provider=ModelProvider.ALI_BAILIAN,
            model_name=embedding_model,
            api_key=api_key,
            base_url=ALI_BAILIAN_BASE_URL
        ),
        llm=LLMConfig(
            provider=ModelProvider.ALI_BAILIAN,
            model_name=llm_model,
            api_key=api_key,
            base_url=ALI_BAILIAN_BASE_URL,
            temperature=temperature
        )
    )


def get_custom_config(base_url: str,
                      api_key: str,
                      embedding_model: str,
                      llm_model: str,
                      temperature: float = 0.1) -> ModelConfig:
    """获取自定义 OpenAI 兼容接口配置"""
    return ModelConfig(
        embedding=EmbeddingConfig(
            provider=ModelProvider.CUSTOM,
            model_name=embedding_model,
            api_key=api_key,
            base_url=base_url
        ),
        llm=LLMConfig(
            provider=ModelProvider.CUSTOM,
            model_name=llm_model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature
        )
    )


# ======================
# 8. Chroma 向量存储封装
# ======================
class ChromaVectorStore:
    """Chroma 向量数据库封装"""
    
    def __init__(self, collection_name: str = "rag_collection", persist_directory: str = "./chroma_db"):
        from chromadb import PersistentClient
        self.client = PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def add_documents(self, texts: List[str], embeddings: List[List[float]], 
                      ids: Optional[List[str]] = None, metadatas: Optional[List[Dict]] = None):
        """添加文档到向量库"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
    
    def search(self, query_embedding: List[float], n_results: int = 5) -> Dict:
        """向量相似度搜索"""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
    
    def delete_collection(self):
        """删除集合"""
        self.client.delete_collection(self.collection.name)