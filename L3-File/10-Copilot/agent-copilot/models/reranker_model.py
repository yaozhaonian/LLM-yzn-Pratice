# 使用Post-Retrieval后检索-重排序，本地模型
import json
import hashlib
from typing import List, Optional

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

class RerankerModel:
    def __init__(
        self, 
        texts: Optional[List[str]] = None, 
        vectorstore: Optional[Chroma] = None,
        llm_model: str = 'qwen2.5:7b',
        embedding_model: str = "bge-m3:latest",
        base_url: str = "http://127.0.0.1:11434",
        persist_directory: str = "./chroma_db",
        k_retrieval: int = 4,
        k_rrf: int = 60
    ):
        """
        初始化重排序模型
        :param texts: 原始文本列表，如果提供则创建新的向量库
        :param vectorstore: 已有的 Chroma 向量库对象，如果提供则直接使用，忽略 texts
        :param llm_model: LLM 模型名称
        :param embedding_model: 嵌入模型名称
        :param base_url: Ollama 服务地址
        :param persist_directory: 向量库持久化路径
        :param k_retrieval: 每个子查询检索的文档数量
        :param k_rrf: RRF 算法的平滑参数
        """
        self.k_rrf = k_rrf
        self.base_url = base_url
        
        # 1. 初始化模型组件
        self.llm = ChatOllama(model=llm_model, temperature=0, base_url=base_url)
        self.embedding = OllamaEmbeddings(model=embedding_model, base_url=base_url)
        
        # 2. 初始化向量存储和检索器
        if vectorstore:
            self.vectorstore = vectorstore
        elif texts:
            # 如果提供了文本，则创建新的向量库
            # 注意：如果 texts 很大，建议先检查 persist_directory 是否已存在，避免重复创建
            self.vectorstore = Chroma.from_texts(
                texts=texts, 
                embedding=self.embedding, 
                persist_directory=persist_directory
            )
        else:
            # 如果既没有 vectorstore 也没有 texts，尝试从 persist_directory 加载
            try:
                self.vectorstore = Chroma(
                    embedding_function=self.embedding,
                    persist_directory=persist_directory
                )
                # 检查是否为空
                if self.vectorstore._collection.count() == 0:
                    raise ValueError("向量库为空，没有提供文本。")
            except Exception as e:
                raise ValueError(f"必须提供 texts 或 vectorstore，或确保 persist_directory ({persist_directory}) 中有有效数据。错误: {str(e)}")
            
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k_retrieval})
        
        # 3. 初始化查询生成链
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
        """解析LLM输出的查询列表"""
        queries = [q.strip() for q in output.split("\n") if q.strip()]
        # 过滤掉可能存在的编号 (如 "1. xxx")
        cleaned_queries = []
        for q in queries:
            if len(q) > 1 and q[0].isdigit() and q[1] in ['.', '、']:
                q = q[2:].strip()
            if q:
                cleaned_queries.append(q)
        return cleaned_queries[:4]

    @staticmethod
    def get_doc_hash(doc: Document) -> str:
        """生成文档的唯一哈希标识"""
        content = doc.page_content
        # 确保 metadata 是可序列化的，并排序 key 以保证一致性
        try:
            meta_str = json.dumps(doc.metadata, sort_keys=True, default=str)
        except Exception:
            meta_str = ""
            raise ValueError("metadata无法序列化。")
        combined = f"{content}|||{meta_str}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()

    @staticmethod
    def reciprocal_rank_fusion_func(results: List[List[Document]], k: int = 60) -> List[tuple]:
        """
        互逆排序融合算法 (静态函数，便于在 Chain 中使用)
        """
        fused_scores = {}
        doc_map = {}
        
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_key = RerankerModel.get_doc_hash(doc)
                
                if doc_key not in fused_scores:
                    fused_scores[doc_key] = 0
                    doc_map[doc_key] = doc
                
                # RRF 公式: 1 / (k + rank)
                fused_scores[doc_key] += 1 / (k + rank)
                
        # 按分数降序排序
        reranked_results = sorted(
            fused_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 返回 (Document, score) 格式
        return [(doc_map[key], score) for key, score in reranked_results]

    def invoke(self, original_query: str) -> List[tuple]:
        """
        执行 RAG Fusion 流程
        :param original_query: 用户原始查询
        :return: 重排序后的文档列表，每个元素为 (Document, score)
        """
        from langchain_core.runnables import RunnableLambda
        
        # 使用 RunnableLambda 包装静态方法，以便传入 self.k_rrf
        rrf_lambda = RunnableLambda(
            lambda x: self.reciprocal_rank_fusion_func(x, k=self.k_rrf)
        )
        
        # 构建完整链：生成查询 -> 并行检索 -> RRF 重排序
        full_chain = self.generate_queries_chain | self.retriever.map() | rrf_lambda
        
        return full_chain.invoke({"original_query": original_query})

    def add_texts(self, texts: List[str]):
        """
        向现有向量库添加新文本
        :param texts: 新的文本列表
        """
        if hasattr(self, 'vectorstore'):
            self.vectorstore.add_texts(texts)
        else:
            raise RuntimeError("Vectorstore not initialized properly.")

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
    reranker = RerankerModel(texts=sample_texts)
    
    # 2. 执行查询
    query = "人工智能的应用"
    print(f"--- 原始查询: {query} ---")
    
    results = reranker.invoke(query)
    
    # 3. 输出结果
    print("\n--- 重排序后的结果 (Top Documents) ---")
    for i, (doc, score) in enumerate(results):
        print(f"{i+1}. [Score: {score:.4f}] {doc.page_content}")