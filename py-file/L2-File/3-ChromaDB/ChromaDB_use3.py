"""
对use2的优化 -- 面向对象
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
from typing import Optional, List, Dict

# ======================
# 1. 文档缓存管理类
# ======================
class DocumentCacheManager:
    """管理文档缓存，记录文件哈希和状态"""
    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.cache_data = self._load_cache()
        
    def _load_cache(self) -> Dict:
        """加载缓存文件"""
        if self.cache_file.exists():
            cache_path = Path(self.cache_file)
            return json.loads(cache_path.read_text(encoding='utf-8'))
        return {}
    
    def save_cache(self):
        """保存缓存到文件"""
        Path(self.cache_file).write_text(
            json.dumps(self.cache_data, ensure_ascii=False, indent=2),
            encoding='utf-8'        
        )

    @staticmethod
    def get_file_hash(file_path: Path) -> str:
        """计算文件 MD5 哈希值"""
        return hashlib.md5(file_path.read_bytes()).hexdigest()
    
    def is_changed(self, file_path: Path) -> bool:
        """检查判断文件是否有变化"""
        current_hash = self.get_file_hash(file_path)
        cached_hash = self.cache_data.get(str(file_path))
        return current_hash!= cached_hash
    
    def update_cache(self, file_path: Path,chunks: int):
        """更新缓存记录"""
        self.cache_data[str(file_path)] = {
            'hash': self.get_file_hash(file_path),
            'chunks': chunks,
            'updated': str(file_path.stat().st_mtime)
        }
        self.save_cache()

    def get_info(self, file_path: Path) -> Optional[Dict]:
        """获取文件缓存信息"""
        return self.cache_data.get(str(file_path))

# ======================
# 2. 向量库管理类
# ======================
class VectorStoreManager:
    """管理 Chroma 向量库的创建、加载和重建"""
    def __init__(self, vector_store_dir: Path, embedding):
        self.vector_store_dir = vector_store_dir
        self.embedding = embedding
        self.vector_store: Optional[Chroma] = None

    def load_existing(self) -> bool:
        """加载已存在的向量库"""
        if not self.vector_store_dir.exists():
            return False
        
        try:
            self.vector_store = Chroma(
                persist_directory=str(self.vector_store_dir),
                embedding_function=self.embedding
            )
            print(f"✓ 向量库加载完成：{self.vector_store_dir}")
            return True
        except Exception as e:
            print(f"⚠️ 向量库加载失败：{e}")
            return False
        
    def create_from_documents(self, documents, text_splitter):
        """从文档创建向量库"""
        print(f"正在创建向量库...")

        texts = text_splitter.split_documents(documents)
        print(f"分割文档完成，共 {len(texts)} 个片段")

        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embedding,
            persist_directory=str(self.vector_store_dir)
        )

        return len(texts) 
    
    def rebuild(self, file_path: Path, text_splitter, cache_manager: DocumentCacheManager):
        """重建向量库"""
        # 加载文档
        loader = TextLoader(str(file_path),encoding='utf-8')
        documents = loader.load()
        print(f"✓ 加载文档：{len(documents)} 个")

        # 创建/更新向量库
        chunks = self.create_from_documents(documents, text_splitter)

        # 更新缓存
        cache_manager.update_cache(file_path, chunks)
        print(f"✓ 缓存更新完成")

    def get_retriever(self, k: int = 3):
        """获取向量库检索器"""
        if self.vector_store is None:
            raise RuntimeError("向量库未加载")
        return self.vector_store.as_retriever(search_kwargs={"k": k})

# ======================
# 3. RAG 问答机器人类
# ======================
class RAGChatbot:
    """基于 RAG 的问答机器人,管理问答链和用户交互"""

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.chain = self._build_chain()

    def _build_chain(self):
        """构建问答链"""
        qa_prompt = PromptTemplate.from_template("""
        根据以下上下文回答问题，如果不知道答案，就说不知道：
        上下文：{context}
        问题：{question}
        回答：
        """)

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_prompt}
        )
    
    def ask(self, context: str, question: str) -> Dict:
        """提问并获取回答"""
        return self.chain.invoke({"query":question})
    
    def print_result(self, answer: Dict, max_content_len: int = 100):
        """打印回答"""
        print(f"\n[回答]\nAnswer: {answer['result']}")

        print("\n[参考资料]\n")
        for i, doc in enumerate(answer['source_documents']):
            content = doc.page_content[:max_content_len]
            print(f'{i+1}.{content}...')

    def start_chat(self, exit_pattern: str = r"^(exit|退出|quit|结束|q)$"):
        """启动聊天循环"""
        print("\n" + "="*50)
        print("RAG 问答系统已就绪，输入 '退出' 结束对话")
        print("="*50)

        pattern = re.compile(exit_pattern, re.IGNORECASE)

        while True:
            try:
                question = input("\n请输入问题:")
                if pattern.match(question):
                    print("👋 再见！")
                    break

                result = self.ask(context="", question=question)
                self.print_result(result)

            except KeyboardInterrupt:
                print("\n👋 再见！")
            except Exception as e:
                print(f"❌ 查询出错：{e}")

# ======================
# 4. 主程序入口
# ======================
class RAGApplication:
    """RAG 应用主控制器"""
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.vector_store_dir = self.script_dir / "./chroma_db"
        self.cache_file = self.script_dir / ".doc_cache.json"

        # 初始化组件
        self.embedding = OllamaEmbeddings(model='bge-m3:latest')
        self.llm = OllamaLLM(model='qwen2.5:7b',temperature=0.1)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 150,
            chunk_overlap = 30
        )

        # 管理器
        self.cache_manager = DocumentCacheManager(self.cache_file)
        self.vector_store_manager = VectorStoreManager(self.vector_store_dir, self.embedding)
        self.chatbot:Optional[RAGChatbot] = None

    def find_document(self) -> Path:
        """查找数据文件"""
        possible_paths = [
            (self.script_dir / "text.txt").resolve(),
            (self.script_dir / "../Data/deepseek 百度百科.txt").resolve(),
            (self.script_dir / "../../Data/deepseek 百度百科.txt").resolve(),
            Path(r"E:\py-file\L2-File\Data\deepseek 百度百科.txt").resolve(),
        ]

        for path in possible_paths:
            if path.exists():
                print(f"✓ 找到文件：{path}")
                return path

        # 创建测试文件
        file_path = self.script_dir / "text.txt"
        file_path.write_text("这是一个测试文档。\nDeepSeek 是一款人工智能模型。\n", encoding='utf-8')
        print(f"✓ 已创建测试文件：{file_path}")
        return file_path

    def run(self):
        """运行 RAG 应用"""
        self.initialize()
        if self.chatbot:
            self.chatbot.start_chat()

    def initialize(self):
        """初始化 RAG 应用"""
        print("="*20,"[初始化 RAG 系统]","="*20)

        # 查找文档
        file_path = self.find_document()

        # 检查是否需要重建向量库
        need_rebuild = False

        if self.vector_store_manager.load_existing():
            if self.cache_manager.is_changed(file_path):
                print("⚠ 文档已变更，需要重新构建向量库")
                need_rebuild = True
            else:
                print("✓ 文档未变化，使用现有向量库")
        else:
            print("⚠ 向量库不存在，需要重新构建")
            need_rebuild = True

        # 重建向量库
        if need_rebuild:
            self.vector_store_manager.rebuild(file_path, self.text_splitter, self.cache_manager)

        # 创建聊天机器人
        self.chatbot = RAGChatbot(self.llm, self.vector_store_manager.get_retriever(k=3))

        print("="*20,"[RAG 系统初始化完成]","="*20)

# ======================
# 5. 程序入口
# ======================
if __name__ == '__main__':
    app = RAGApplication()
    app.run()
