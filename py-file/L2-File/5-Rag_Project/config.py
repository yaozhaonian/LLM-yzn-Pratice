# 配置文件
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent

# 知识库目录
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
DOCUMENTS_DIR = KNOWLEDGE_BASE_DIR / "documents"
CHROMA_DB_DIR = KNOWLEDGE_BASE_DIR / "chroma_db"

# 确保目录存在
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# 模型配置
OLLAMA_EMBEDDING_MODEL = "bge-m3:latest"
OLLAMA_LLM_MODEL = "qwen2.5:7b"
LLM_TEMPERATURE = 0.1

# 上传配置
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx", ".json", ".md"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB