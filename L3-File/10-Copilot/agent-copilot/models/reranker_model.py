# 使用Post-Retrieval后检索-重排序，本地模型
import json
import hashlib
from typing import List, Optional

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import chain, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

class RerankerModel:
    def __init__(
        self, 
        vectorstore: Optional[Chroma] = None,   # 移除了 texts
        llm_model: str = 'qwen2.5:7b',
        embedding_model: str = "bge-m3:latest",
        base_url: str = "http://127.0.0.1:11434",
        persist_directory: str = "./chroma_db",
        k_retrieval: int = 4,
        k_rrf: int = 60
    ):
        self.k_rrf = k_rrf
        self.base_url = base_url
        
        # 1. 初始化模型组件（这里补全了完整参数，不再是 ...）
        self.llm = ChatOllama(
            model=llm_model, 
            temperature=0, 
            base_url=base_url
        )
        self.embedding = OllamaEmbeddings(
            model=embedding_model, 
            base_url=base_url
        )
        
        # 2. 初始化向量存储
        if vectorstore:
            self.vectorstore = vectorstore
        else:
            # 仅尝试从磁盘加载，不自动创建
            self.vectorstore = Chroma(
                embedding_function=self.embedding,
                persist_directory=persist_directory
            )
            if self.vectorstore._collection.count() == 0:
                raise ValueError(f"路径 {persist_directory} 下没有数据，请使用 RerankerModel.create_from_texts() 方法创建。")
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k_retrieval})
        
        # 3. 初始化查询生成链（与你的原代码一致）
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个能根据单个输入查询生成多个搜索查询的有用助手。"),
            ("user", "生成多个与 '{original_query}' 相关的搜索查询。"),
            ("user", "请仅输出4个查询，每行一个，不要包含编号或其他文字。")
        ])
        
        self.generate_queries_chain = (
            self.prompt 
            | self.llm 
            | StrOutputParser() 
            | self.parse_queries
        )

    @staticmethod
    def parse_queries(output: str) -> List[str]:
        queries = [q.strip() for q in output.split("\n") if q.strip()]
        cleaned_queries = []
        for q in queries:
            if len(q) > 1 and q[0].isdigit() and q[1] in ['.', '、']:
                q = q[2:].strip()
            if q:
                cleaned_queries.append(q)
        return cleaned_queries[:4]

    @staticmethod
    def get_doc_hash(doc: Document) -> str:
        content = doc.page_content
        try:
            meta_str = json.dumps(doc.metadata, sort_keys=True, default=str)
        except Exception:
            meta_str = ""
            raise ValueError("metadata无法序列化。")
        combined = f"{content}|||{meta_str}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()

    @staticmethod
    def reciprocal_rank_fusion_func(results: List[List[Document]], k: int = 60) -> List[tuple]:
        fused_scores = {}
        doc_map = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_key = RerankerModel.get_doc_hash(doc)
                if doc_key not in fused_scores:
                    fused_scores[doc_key] = 0
                    doc_map[doc_key] = doc
                fused_scores[doc_key] += 1 / (k + rank)
        reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_map[key], score) for key, score in reranked_results]

    def invoke(self, original_query: str) -> List[tuple]:
        rrf_lambda = RunnableLambda(
            lambda x: self.reciprocal_rank_fusion_func(x, k=self.k_rrf)
        )
        full_chain = self.generate_queries_chain | self.retriever.map() | rrf_lambda
        return full_chain.invoke({"original_query": original_query})

    def add_texts(self, texts: List[str]):
        if hasattr(self, 'vectorstore'):
            self.vectorstore.add_texts(texts)
        else:
            raise RuntimeError("Vectorstore not initialized properly.")

    # ---------- 重点：新增的工厂方法，专门用于首次创建 ----------
    # 工厂方法，是一个专门用来“生产”对象的“函数”
    @classmethod
    def create_from_texts(
        cls, 
        texts: List[str], 
        persist_directory: str, 
        **kwargs  # 用来接收 llm_model, base_url, k_rrf 等其他参数
    ):
        """
        静态工厂方法：先用文本建好向量库，再初始化 RerankerModel
        """
        # 注意：这里必须和 __init__ 里的模型名保持一致
        embedding_model = kwargs.get('embedding_model', 'bge-m3:latest')
        base_url = kwargs.get('base_url', 'http://127.0.0.1:11434')
        
        # 临时构建嵌入模型，用来建库
        temp_embedding = OllamaEmbeddings(
            model=embedding_model, 
            base_url=base_url
        )
        
        # 创建持久化的向量库
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=temp_embedding,
            persist_directory=persist_directory
        )
        
        # 调用正常的 __init__，把建好的 vectorstore 传进去，同时透传其他参数
        return cls(
            vectorstore=vectorstore, 
            persist_directory=persist_directory, 
            **kwargs
        )

# 测试代码
if __name__ == "__main__":
    # 示例文本
    sample_texts = [
        "人工智能在医疗诊断中的应用。",
        "人工智能如何提升供应链效率。",
        "NBA季后赛最新赛况分析。",
        "传统法式烘焙的五大技巧。",
        "红楼梦人物关系图谱分析。",
        "人工智能在金融风险管理中的应用。",
        "人工智能如何影响未来就业市场。",
        "人工智能在制造业的应用。",
        "今天天气怎么样",
        "人工智能伦理：公平性与透明度。",
        "人工智能可以应用在地理教学中",
        "中国女足挺进世界杯",
        "中国男足开了4个号,依旧与世界杯无缘",
        "世界杯席位增至48,中国仍无缘上榜",
        "某音热点:国足是用来平衡国运的",
        "人工智能在精细化制作中的应用",
        "人工智能在交通指挥的应用"   
    ]

    # 1. 初始化类
    reranker = RerankerModel.create_from_texts(
        texts=sample_texts,
        persist_directory="./chroma_db",
        llm_model='qwen2.5:7b',    # 这些参数会透传给 __init__
        base_url='http://127.0.0.1:11434',
        k_rrf=60
    )
    
    # 2. 执行查询
    query = "人工智能的应用"
    print(f"--- 原始查询: {query} ---")
    
    results = reranker.invoke(query)
    
    # 3. 输出结果
    print("\n--- 重排序后的结果 (Top Documents) ---")
    for i, (doc, score) in enumerate(results):
        print(f"{i+1}. [Score: {score:.4f}] {doc.page_content}")