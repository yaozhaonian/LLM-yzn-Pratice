from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class OllamaSettings(BaseSettings):
    host: str = "http://localhost"
    port: int = 11434
    model: str = "qwen2.5:7b"
    timeout: int = 120

    model_config = SettingsConfigDict(
        env_prefix="OLLAMA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def base_url(self) -> str:
        return f"{self.host}:{self.port}"


class EmbeddingSettings(BaseSettings):
    model: str = "qwen2.5:7b"
    dimension: int = 3584

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class MilvusSettings(BaseSettings):
    host: str = "localhost"
    port: int = 19530
    username: Optional[str] = ""
    password: Optional[str] = ""
    collection_name: str = "erp_knowledge"

    model_config = SettingsConfigDict(
        env_prefix="MILVUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class MySQLSettings(BaseSettings):
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: Optional[str] = ""
    database: str = "erp_system"
    pool_size: int = 5
    max_overflow: int = 10

    model_config = SettingsConfigDict(
        env_prefix="MYSQL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def url(self) -> str:
        if self.password:
            return f"mysql+mysqlconnector://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        return f"mysql+mysqlconnector://{self.user}@{self.host}:{self.port}/{self.database}"


class LogSettings(BaseSettings):
    level: str = "INFO"
    file: str = "app.log"

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class RAGSettings(BaseSettings):
    top_k: int = 3

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


ollama_settings = OllamaSettings()
embedding_settings = EmbeddingSettings()
milvus_settings = MilvusSettings()
mysql_settings = MySQLSettings()
log_settings = LogSettings()
rag_settings = RAGSettings()


class Settings:
    ollama = ollama_settings
    embedding = embedding_settings
    milvus = milvus_settings
    mysql = mysql_settings
    log = log_settings
    rag = rag_settings


settings = Settings()
