# e:\py-file\my-lib\rag_models\__init__.py
"""
RAG 模型工具包
通用模型接口封装，支持 Ollama、阿里云百炼、OpenAI 等
"""

from .models import (
    # 配置类
    ModelConfig,
    EmbeddingConfig,
    LLMConfig,
    ModelProvider,
    
    # 工厂类
    ModelClientFactory,
    ChromaVectorStore,
    
    # 便捷函数
    get_ollama_config,
    get_ali_bailian_config,
    get_custom_config,
    
    # 常量
    ALI_BAILIAN_API_KEY_VAR,
    ALI_BAILIAN_BASE_URL,
    ALI_BAILIAN_EMBEDDING_MODEL,
    ALI_BAILIAN_LLM_MODEL,
)

__version__ = "0.1.0"
__author__ = "Johnnie Yao"
__all__ = [
    "ModelConfig",
    "EmbeddingConfig",
    "LLMConfig",
    "ModelProvider",
    "ModelClientFactory",
    "ChromaVectorStore",
    "get_ollama_config",
    "get_ali_bailian_config",
    "get_custom_config",
]