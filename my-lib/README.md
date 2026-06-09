# rag-models

RAG 通用模型接口封装，支持多种大模型提供商。

## 安装为本地包

### 方式一：开发模式安装（推荐）

```bash
# 进入包目录
cd e:\\py-file\\my-lib

# 开发模式安装（代码修改后无需重新安装）
pip install -e .
```

### 方式二：正常安装
```bash
cd e:\\py-file\\my-lib
pip install .
```

### 方式三：仅添加路径（不安装）
```python
# 在需要使用的项目中添加
import sys
sys.path.insert(0, r"e:\py-file\my-lib")

from rag_models import get_ollama_config
```

## 使用示例
安装后，在任意项目中导入：

```python
# ChromaDB_use5.py
from rag_models import get_ollama_config, ModelClientFactory

model_config = get_ollama_config(
    embedding_model="bge-m3:latest",
    llm_model="qwen2.5:7b",
    temperature=0.1
)

embedding = ModelClientFactory.get_embedding_client(model_config.embedding)
llm = ModelClientFactory.get_llm_client(model_config.llm)
```

## 发布到 PyPI（可选）
如果想让其他人也能 pip install rag-models：

```bash
# 安装构建工具
pip install build twine

# 构建包
cd e:\\py-file\\my-lib
python -m build

# 测试上传（TestPyPI）
twine upload --repository testpypi dist/*

# 正式上传（PyPI）
twine upload dist/*
```

## 最终目录结构
```bash
e:\py-file\
├── my-lib/                    # 包源码目录
│   ├── rag_models/
│   │   ├── __init__.py
│   │   └── models.py
│   ├── pyproject.toml
│   └── README.md
│
└── ChromaDB/                  # 使用包的项目
    └── ChromaDB_use5.py       # 直接 import rag_models

```
### 快速验证
```bash
# 1. 安装包
cd e:\\py-file\\my-lib
pip install -e .

# 2. 测试导入
python -c "from rag_models import get_ollama_config; print('导入成功！')"
```

## 常见问题
| 问题 | 解决方案 |
| :------: | :------: |
| ModuleNotFoundError | 确认已执行 pip install -e . |
| 导入路径错误 | 检查 __init__.py 中的 from .models |
| 依赖缺失 | 在 pyproject.toml 的 dependencies 中添加 |
| 代码修改不生效 | 开发模式安装会自动生效，否则需重新 pip install |

创建完成后，任何项目只需 pip install 即可使用你的 models.py！
