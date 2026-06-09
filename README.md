# 🤖 PY-FILE - AI Agent & LLM Learning Project

这是一个 Python 项目文件夹，拥有多个可运行的py文件，包含了从基础到进阶的 LLM / LangChain / LangGraph AI智能体相关代码练习与实验。

## 🛠️ 技术栈与核心能力
| 类别 | 技术/工具 | 掌握程度 |
| :--- | :--- | :--- |
| **语言与环境** | Python 3.10+, Git, VSCode, Virtualenv | ★★★★★ |
| **大模型应用** | 提示词工程（COT/TOT/Few-Shot）、本地模型调用 | ★★★★★ |
| **RAG 技术** | 文档分块、向量数据库（Chroma）、基础/高级 RAG、Ragas 评估 | ★★★★★ |
| **框架工具** | LangChain、LangGraph、Ollama、AutoGen 多智能体 | ★★★★☆ |
| **配套能力** | Python 爬虫、SQL 基础、数据处理 | ★★★★☆ |

---

## 📁 项目结构
PY-FILE/
├── .venv/ # Python 虚拟环境（.gitignore 已忽略）
├── chroma_db/ # 向量数据库（.gitignore 已忽略）
├── data/ # 通用数据文件
├── evals/ # 模型与 RAG 评估相关代码
├── L1-File/ # 入门级基础练习
│ ├── Algorithm_Fundamentals/ # 算法基础练习
│ ├── exercise/ # 通用 Python 练习
│ ├── spider/ # 爬虫相关练习
│ ├── sql/ # SQL 数据库基础练习
│ ├── 1-prompt_defend.py # Prompt 注入防御实践
│ ├── 2-rednote.py # 小红书相关数据处理
│ ├── 3-webscrap.py # 基础网页爬虫实践
│ ├── 4-Few-Shot.py # Few-Shot 提示词工程
│ ├── 5-COT.py # Chain-of-Thought 思维链
│ ├── 6-Self-Consistency_Counter.py # 自一致性采样
│ ├── 7-use-local-model.py # 本地模型调用实践
│ └── 8-TOT.py # Tree-of-Thought 思维树
├── L2-File/ # 中级进阶练习
│ ├── 1-doc_split/ # 文档分块与预处理
│ ├── 2-Ollama/ # Ollama 本地大模型部署
│ ├── 3-ChromaDB/ # Chroma 向量数据库使用
│ ├── 4-base_knowledge/ # 基础知识库构建
│ ├── 5-Rag_Project/ # 基础 RAG 项目实现
│ ├── 6-Ragas/ # RAG 效果评估（Ragas 框架）
│ ├── 7-LangChain/ # LangChain 基础与进阶
│ ├── 11-FunctionCall/ # LLM 函数调用实践
│ ├── 12-Advance_Rag/ # 高级 RAG 技术（如 HyDE / 多查询）
│ ├── Data/ # RAG 项目专用数据集
│ └── 向量数据库对比.md # 向量数据库选型对比笔记
├── L3-File/ # 高级 / 项目级实践
│ ├── 1-First_agent.py # 第一个自定义 Agent 实现
│ ├── 2-Func/ # 高级函数调用与工具集成
│ ├── 3-Base/ # Agent 基础逻辑构建
│ ├── 4-Enhance/ # Agent 增强功能（记忆 / 反思）
│ ├── 5-Human_in_loop/ # 人机协同工作流
│ ├── 6-Tool_Use/ # 复杂工具链集成
│ ├── 7-Subgraph/ # LangGraph 子图与状态管理
│ ├── 8-multi-agent/ # 多智能体协作框架
│ └── 分析代码常用 ai 语.txt # 代码分析与提示词模板
├── my-lib/ # 自定义工具库与通用函数
├── node_modules/ # Node.js 依赖（.gitignore 已忽略）
├── requirements.txt # Python 项目依赖清单
├── package.json # Node.js 配置文件
├── package-lock.json # Node.js 依赖锁文件
├── summary_cache.json # 摘要缓存文件
├── .gitignore # Git 忽略配置
├── LICENSE # MIT 开源协议
└── README.md # 项目说明（本文件）


---

## 📚 主要学习内容
### ✅ L1-File：LLM与Python基础
- Python 算法与数据结构、爬虫、SQL基础
- 提示词工程核心技术：Few-Shot、COT、TOT、自一致性
- Prompt注入防御、本地模型调用

### ✅ L2-File：RAG与LangChain进阶
- 文档分块、向量数据库（Chroma）使用
- Ollama本地大模型部署
- 基础RAG到高级RAG的完整实现
- Ragas框架评估RAG效果
- LLM函数调用与LangChain核心组件

### ✅ L3-File：Agent与多智能体项目
- 单Agent构建、工具链集成
- LangGraph状态管理与子图
- AutoGen多智能体协作
- 人机协同（Human-in-the-loop）工作流
- 企业级AI辅助系统项目实践

---

## 📦 快速开始
### 1. 克隆仓库
```bash
git clone https://github.com/yaozhaonian/LLM-yzn-Pratice.git
cd PY-FILE


License
本项目采用 MIT License 开源协议，可自由学习与二次修改。
