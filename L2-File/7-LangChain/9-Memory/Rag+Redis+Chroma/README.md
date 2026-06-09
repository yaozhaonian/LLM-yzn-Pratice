# 结合rag技术、redis记忆缓存、chroma数据存储的项目
结合之前的项目使用ai进行改写优化

## 项目结构
Rag+Redis+Chroma/
├── app.py                      # FastAPI 主应用
├── config.py                   # 配置文件
├── requirements.txt            # 依赖包
├── rag_core/
│   ├── __init__.py
│   ├── rag_engine.py           # RAG 引擎（整合 Redis 记忆）
│   ├── vector_store.py         # Chroma 向量存储
│   ├── document_loader.py      # 文档加载器
│   └── memory_manager.py       # Redis 记忆管理（新增）
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   └── chat.html
├── static/
│   ├── css/
│   └── js/
├── data/
│   ├── documents/              # 上传文档存储
│   └── chroma_db/              # Chroma 持久化存储
└── logs/




