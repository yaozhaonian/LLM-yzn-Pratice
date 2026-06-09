"""
ChromaDB RAG 问答系统 - 完整优化版(继续优化use3版)
功能：
1. 多文档支持与增量更新
2. 配置类集中管理参数
3. 日志系统记录运行状态
4. 对话历史记录
5. 向量库统计信息
"""
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from pathlib import Path
import hashlib
import json
import re
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict


# ======================
# 0. 日志配置
# ======================
def setup_logger(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """配置日志系统"""
    logger = logging.getLogger("RAG")
    logger.setLevel(level)
    
    # 清除现有处理器
    logger.handlers.clear()
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ======================
# 1. 配置类
# ======================
@dataclass
class ModelConfig:
    """模型配置"""
    embedding_model: str = "bge-m3:latest"
    llm_model: str = "qwen2.5:7b"
    temperature: float = 0.1


@dataclass
class ChunkConfig:
    """文本切分配置"""
    chunk_size: int = 150
    chunk_overlap: int = 30


@dataclass
class RetrievalConfig:
    """检索配置"""
    top_k: int = 3
    score_threshold: Optional[float] = None


@dataclass
class PathConfig:
    """路径配置"""
    script_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    vector_store_dir: str = "./chroma_db"
    cache_file: str = ".doc_cache.json"
    log_file: str = ".rag.log"
    data_dirs: List[str] = field(default_factory=lambda: [
        "text.txt",
        "../Data/deepseek 百度百科.txt",
        "../../Data/deepseek 百度百科.txt",
        r"E:\py-file\L2-File\Data\deepseek 百度百科.txt",
    ])
    
    def get_vector_store_path(self) -> Path:
        return self.script_dir / self.vector_store_dir
    
    def get_cache_path(self) -> Path:
        return self.script_dir / self.cache_file
    
    def get_log_path(self) -> Path:
        return self.script_dir / self.log_file


@dataclass
class Config:
    """总配置类"""
    model: ModelConfig = field(default_factory=ModelConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    path: PathConfig = field(default_factory=PathConfig)
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ======================
# 2. 文档缓存管理类
# ======================
class DocumentCacheManager:
    """管理多文档缓存，记录文件哈希和状态"""
    
    def __init__(self, cache_file: Path, logger: logging.Logger):
        self.cache_file = cache_file
        self.logger = logger
        self.cache_data = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """加载缓存文件"""
        if self.cache_file.exists():
            try:
                content = self.cache_file.read_text(encoding='utf-8')
                self.logger.info(f"加载缓存文件：{self.cache_file}")
                return json.loads(content)
            except Exception as e:
                self.logger.warning(f"缓存文件加载失败：{e}")
                return {}
        self.logger.info("缓存文件不存在，创建新缓存")
        return {}
    
    def save_cache(self):
        """保存缓存到文件"""
        try:
            self.cache_file.write_text(
                json.dumps(self.cache_data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            self.logger.info(f"缓存已保存：{self.cache_file}")
        except Exception as e:
            self.logger.error(f"缓存保存失败：{e}")
    
    @staticmethod
    def get_file_hash(file_path: Path) -> str:
        """计算文件 MD5 哈希值"""
        return hashlib.md5(file_path.read_bytes()).hexdigest()
    
    def is_changed(self, file_path: Path) -> bool:
        """检查文件是否发生变化"""
        key = str(file_path.resolve())
        current_hash = self.get_file_hash(file_path)
        cached_info = self.cache_data.get(key)
        
        if cached_info is None:
            self.logger.debug(f"新文件：{file_path}")
            return True
        
        cached_hash = cached_info.get('hash')
        changed = current_hash != cached_hash
        self.logger.debug(f"文件 {file_path.name} 变化检测：{changed}")
        return changed
    
    def update_cache(self, file_path: Path, chunks: int, doc_count: int = 1):
        """更新缓存记录"""
        key = str(file_path.resolve())
        self.cache_data[key] = {
            'hash': self.get_file_hash(file_path),
            'chunks': chunks,
            'doc_count': doc_count,
            'updated': datetime.now().isoformat(),
            'file_size': file_path.stat().st_size
        }
        self.save_cache()
        self.logger.info(f"更新缓存：{file_path.name}, 片段数：{chunks}")
    
    def remove_cache(self, file_path: Path):
        """移除缓存记录"""
        key = str(file_path.resolve())
        if key in self.cache_data:
            del self.cache_data[key]
            self.save_cache()
            self.logger.info(f"移除缓存：{file_path}")
    
    def get_all_files(self) -> List[Dict]:
        """获取所有缓存文件信息"""
        return [
            {'path': k, **v} for k, v in self.cache_data.items()
        ]
    
    def get_total_chunks(self) -> int:
        """获取总片段数"""
        return sum(info.get('chunks', 0) for info in self.cache_data.values())


# ======================
# 3. 向量库管理类
# ======================
class VectorStoreManager:
    """管理 Chroma 向量库的创建、加载和重建"""
    
    def __init__(self, vector_store_dir: Path, embedding, logger: logging.Logger):
        self.vector_store_dir = vector_store_dir
        self.embedding = embedding
        self.logger = logger
        self.vector_store: Optional[Chroma] = None
    
    def load_existing(self) -> bool:
        """加载已存在的向量库"""
        if not self.vector_store_dir.exists():
            self.logger.info("向量库目录不存在")
            return False
        
        try:
            self.vector_store = Chroma(
                persist_directory=str(self.vector_store_dir),
                embedding_function=self.embedding
            )
            count = self.vector_store._collection.count() if hasattr(self.vector_store, '_collection') else "未知"
            self.logger.info(f"向量库加载完成：{self.vector_store_dir}, 文档数：{count}")
            print(f"✓ 向量库加载完成：{self.vector_store_dir}")
            return True
        except Exception as e:
            self.logger.error(f"向量库加载失败：{e}")
            print(f"⚠️ 向量库加载失败：{e}")
            return False
    
    def create_from_documents(self, documents, text_splitter) -> int:
        """从文档创建向量库"""
        self.logger.info("开始创建向量库")
        
        texts = text_splitter.split_documents(documents)
        self.logger.info(f"文档切分完成：{len(texts)} 个片段")
        print(f"✓ 文档切分完成：{len(texts)} 个片段")
        
        # 如果目录存在则先清理
        if self.vector_store_dir.exists():
            import shutil
            shutil.rmtree(self.vector_store_dir)
            self.logger.info("清理旧向量库目录")
        
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embedding,
            persist_directory=str(self.vector_store_dir)
        )
        
        self.logger.info(f"向量库创建完成：{self.vector_store_dir}")
        print(f"✓ 向量库创建完成：{self.vector_store_dir}")
        
        return len(texts)
    
    def rebuild(self, file_paths: List[Path], text_splitter, cache_manager: DocumentCacheManager):
        """重建向量库（支持多文档）"""
        self.logger.info(f"开始重建向量库，文档数：{len(file_paths)}")
        print("\n【开始构建向量库】")
        
        all_documents = []
        total_chunks = 0
        
        for file_path in file_paths:
            self.logger.info(f"加载文档：{file_path}")
            loader = TextLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
            print(f"✓ 加载文档：{file_path.name} ({len(documents)} 个)")
            all_documents.extend(documents)
        
        # 创建向量库
        total_chunks = self.create_from_documents(all_documents, text_splitter)
        
        # 更新所有文档的缓存
        for file_path in file_paths:
            cache_manager.update_cache(file_path, total_chunks, len(file_paths))
        
        print(f"✓ 缓存更新完成")
        self.logger.info(f"向量库重建完成，总片段数：{total_chunks}")
    
    def get_retriever(self, k: int = 3):
        """获取向量库检索器"""
        if self.vector_store is None:
            self.logger.error("向量库未加载")
            raise RuntimeError("向量库未加载")
        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
    def get_stats(self) -> Dict:
        """获取向量库统计信息"""
        if self.vector_store is None:
            return {"status": "未初始化"}
        
        try:
            count = self.vector_store._collection.count() if hasattr(self.vector_store, '_collection') else 0
            return {
                "status": "正常",
                "document_count": count,
                "persist_directory": str(self.vector_store_dir)
            }
        except Exception as e:
            return {"status": "错误", "error": str(e)}


# ======================
# 4. 对话历史管理类
# ======================
class ConversationHistory:
    """管理对话历史记录"""
    
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.history: List[Dict] = []
    
    def add(self, question: str, answer: str, sources: List[str]):
        """添加对话记录"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "sources": sources
        }
        self.history.append(record)
        
        # 限制历史记录数量
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_recent(self, n: int = 5) -> List[Dict]:
        """获取最近 n 条对话"""
        return self.history[-n:]
    
    def export(self, file_path: Path):
        """导出对话历史"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
    
    def clear(self):
        """清空对话历史"""
        self.history.clear()


# ======================
# 5. RAG 问答机器人类
# ======================
class RAGChatbot:
    """基于 RAG 的问答机器人，管理问答链和用户交互"""
    
    def __init__(self, llm, retriever, logger: logging.Logger, history: ConversationHistory = None):
        self.llm = llm
        self.retriever = retriever
        self.logger = logger
        self.history = history or ConversationHistory()
        self.chain = self._build_chain()
    
    def _build_chain(self):
        """构建问答链"""
        qa_prompt = PromptTemplate.from_template("""
        根据以下上下文回答问题，如果不知道答案，就说不知道：
        上下文：{context}
        问题：{question}
        回答：
        """)
        
        self.logger.info("构建问答链")
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_prompt}
        )
    
    def ask(self, question: str) -> Dict:
        """提问并获取回答"""
        self.logger.info(f"用户提问：{question[:50]}...")
        result = self.chain.invoke({"query": question})
        self.logger.info("获取回答成功")
        
        # 记录对话历史
        sources = [doc.page_content[:100] for doc in result.get("source_documents", [])]
        self.history.add(question, result["result"], sources)
        
        return result
    
    def print_result(self, answer: Dict, max_content_len: int = 100):
        """打印回答"""
        print(f"\n【回答】")
        print(f"Answer: {answer['result']}")
        
        print("\n【参考资料】")
        for i, doc in enumerate(answer['source_documents']):
            content = doc.page_content[:max_content_len]
            print(f'{i+1}. {content}...')
    
    def show_history(self, n: int = 5):
        """显示最近对话历史"""
        recent = self.history.get_recent(n)
        if not recent:
            print("暂无对话历史")
            return
        
        print("\n" + "="*50)
        print(f"最近 {len(recent)} 条对话历史:")
        for i, record in enumerate(recent):
            print(f"\n[{i+1}] {record['timestamp']}")
            print(f"Q: {record['question']}")
            print(f"A: {record['answer'][:50]}...")
        print("="*50)
    
    def export_history(self, file_path: Path):
        """导出对话历史"""
        self.history.export(file_path)
        print(f"✓ 对话历史已导出：{file_path}")
    
    def start_chat(self, exit_pattern: str = r"^(exit|退出|quit|结束|q)$"):
        """启动聊天循环"""
        print("\n" + "="*50)
        print("RAG 问答系统已就绪")
        print("命令：退出/历史/导出/清空")
        print("="*50)
        
        pattern = re.compile(exit_pattern, re.IGNORECASE)
        
        while True:
            try:
                question = input("\n请输入问题:")
                
                # 处理特殊命令
                if pattern.match(question):
                    print("👋 再见！")
                    break
                elif question.lower() in ["历史", "history", "h"]:
                    self.show_history()
                    continue
                elif question.lower() in ["导出", "export", "e"]:
                    self.export_history(Path("conversation_history.json"))
                    continue
                elif question.lower() in ["清空", "clear", "c"]:
                    self.history.clear()
                    print("✓ 对话历史已清空")
                    continue
                
                result = self.ask(question)
                self.print_result(result)
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                self.logger.error(f"查询出错：{e}")
                print(f"❌ 查询出错：{e}")


# ======================
# 6. 主程序入口
# ======================
class RAGApplication:
    """RAG 应用主控制器"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.logger = setup_logger(self.config.path.get_log_path())
        self.logger.info("="*50)
        self.logger.info("RAG 系统启动")
        self.logger.info(f"配置：{self.config.to_dict()}")
        
        # 初始化组件
        self.embedding = OllamaEmbeddings(model=self.config.model.embedding_model)
        self.llm = OllamaLLM(model=self.config.model.llm_model, temperature=self.config.model.temperature)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk.chunk_size,
            chunk_overlap=self.config.chunk.chunk_overlap
        )
        
        # 管理器
        self.cache_manager = DocumentCacheManager(
            self.config.path.get_cache_path(), 
            self.logger
        )
        self.store_manager = VectorStoreManager(
            self.config.path.get_vector_store_path(), 
            self.embedding,
            self.logger
        )
        self.conversation_history = ConversationHistory()
        self.chatbot: Optional[RAGChatbot] = None
    
    def find_documents(self) -> List[Path]:
        """查找所有数据文件"""
        found_paths = []
        
        for path_str in self.config.path.data_dirs:
            path = Path(path_str)
            if not path.is_absolute():
                path = self.config.path.script_dir / path
            path = path.resolve()
            
            if path.exists():
                found_paths.append(path)
                self.logger.info(f"找到文件：{path}")
                print(f"✓ 找到文件：{path}")
        
        if not found_paths:
            # 创建测试文件
            file_path = self.config.path.script_dir / "text.txt"
            file_path.write_text("这是一个测试文档。\nDeepSeek 是一款人工智能模型。\n", encoding='utf-8')
            found_paths.append(file_path)
            self.logger.info(f"创建测试文件：{file_path}")
            print(f"✓ 已创建测试文件：{file_path}")
        
        return found_paths
    
    def initialize(self):
        """初始化 RAG 应用"""
        self.logger.info("开始初始化 RAG 系统")
        print("\n" + "="*50)
        print("[初始化 RAG 系统]")
        print("="*50)
        
        # 查找文档
        file_paths = self.find_documents()
        self.logger.info(f"找到 {len(file_paths)} 个文档")
        
        # 检查是否需要重建向量库
        need_rebuild = False
        
        if self.store_manager.load_existing():
            # 检查是否有文档变化
            for file_path in file_paths:
                if self.cache_manager.is_changed(file_path):
                    print("⚠ 文档已变更，需要重新构建向量库")
                    need_rebuild = True
                    break
            if not need_rebuild:
                print("✓ 文档未变化，使用现有向量库")
        else:
            print("⚠ 向量库不存在，需要重新构建")
            need_rebuild = True
        
        # 重建向量库
        if need_rebuild:
            self.store_manager.rebuild(file_paths, self.text_splitter, self.cache_manager)
        
        # 显示向量库统计
        stats = self.store_manager.get_stats()
        print(f"\n【向量库统计】")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        
        # 创建聊天机器人
        retriever = self.store_manager.get_retriever(k=self.config.retrieval.top_k)
        self.chatbot = RAGChatbot(
            self.llm, 
            retriever, 
            self.logger,
            self.conversation_history
        )
        
        self.logger.info("RAG 系统初始化完成")
        print("\n" + "="*50)
        print("[RAG 系统初始化完成]")
        print("="*50)
    
    def run(self):
        """运行 RAG 应用"""
        try:
            self.initialize()
            if self.chatbot:
                self.chatbot.start_chat()
        except Exception as e:
            self.logger.error(f"系统运行错误：{e}")
            raise
        finally:
            self.logger.info("RAG 系统关闭")


# ======================
# 7. 程序入口
# ======================
if __name__ == '__main__':
    # 可选：自定义配置
    # config = Config()
    # config.model.llm_model = "qwen2.5:7b"
    # config.chunk.chunk_size = 200
    
    app = RAGApplication()
    app.run()