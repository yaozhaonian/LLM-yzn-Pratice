from pathlib import Path

BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
CHROMA_DB_DIR = BASE_DIR / "data" / "chroma_db"

# 文件限制
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.json', '.md'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Redis 配置
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
CACHE_EXPIRE_SECONDS = 3600 * 24 * 3  # 3 天

# 模型配置
OLLAMA_LLM_MODEL = "qwen2.5:7b"
OLLAMA_EMBEDDING_MODEL = "bge-m3:latest"
LLM_TEMPERATURE = 0.1

# 确保目录存在
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)