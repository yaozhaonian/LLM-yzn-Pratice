# Post-Retrieval后检索-重排序RAG-Fusion
from pathlib import Path
import json

# LangChain 组件
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableParallel, chain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
# langchain-core/load/dump/dumps
# 返回对象的 JSON 字符串表示形式。
from langchain_core.load import dumps, loads


llm = ChatOllama(model='qwen2.5:7b', temperature=0)
embedding = OllamaEmbeddings(model="bge-m3:latest")

texts=[
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

vectorstore = Chroma.from_texts(
    texts=texts, embedding= embedding
)  
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个能根据单个输入查询生成多个搜索查询的有用助手。"),
    ("user", "生成多个与 {original_query} 相关的搜索查询"),
    ("user", "输出（4个查询）:")
])

# 1. 优化查询生成解析逻辑，去除空行
def parse_queries(output: str) -> list[str]:
    # 按换行符分割，去除空白字符，过滤空字符串
    queries = [q.strip() for q in output.split("\n") if q.strip()]
    # 确保只取前4个，防止过多
    return queries[:4]

generate_queries = (
    prompt | llm | StrOutputParser() | parse_queries
)
original_query = "人工智能的应用"
queries = generate_queries.invoke({"original_query": original_query})
print(f"原始查询：{original_query}\n生成的查询：{queries}")


@chain
def reciprocal_rank_fusion(results: list[list], k=60):
    """互逆排序融合算法，用于合并多个排序文档列表"""
    fused_scores = {}
    # 用于存储 doc_str 到原始 Document 对象的映射，避免反复 loads
    doc_map = {}
    
    for docs in results:
        for rank, doc in enumerate(docs):
            # 【优化点】使用 page_content + str(metadata) 作为唯一键
            # 这比 dumps/loads 更稳定，且不依赖 Beta API
            # 注意：如果 metadata 中有不可哈希类型，可能需要 json.dumps(metadata, sort_keys=True)
            try:
                # 尝试生成一个稳定的字符串标识
                doc_key = f"{doc.page_content}|||{json.dumps(doc.metadata, sort_keys=True)}"
            except:
                #  fallback: 如果 metadata 无法序列化，仅使用内容
                doc_key = doc.page_content
            
            if doc_key not in fused_scores:
                fused_scores[doc_key] = 0
                doc_map[doc_key] = doc # 保存原始对象引用
            
            fused_scores[doc_key] += 1 / (rank + k)
            
    # 按融合分数降序排序
    reranked_results = [
        (doc_map[doc_key], score) for doc_key, score in sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        )
    ]
    
    return reranked_results

'''
generate_queries会生成4个多角度的query,
retriever.map()的作用是根据generate_queries的结果映射出4个retriever(可以理解为同时复制出4个retriever)
与generate_queries生成的4个query对应，
并为每个query检索出来的一组相关文档集，
那么4个query总共可以生成16个相关文档。
最后会经过RRF算法重新排序后输出最相关的文档
'''
generate_chain = generate_queries | retriever.map() | reciprocal_rank_fusion



# 输入结果列表
result_list = generate_chain.invoke({"original_query": original_query})
# 提取文档内容和对应分数
print("先看输出result_list:\n",result_list)
contents = [doc[0].page_content for doc in result_list]
scores = [doc[1] for doc in result_list]

combined_tuples = list(zip(contents, scores))
print("="*30,"最相关的文档及其得分：")
for item in combined_tuples:
    print(item)

