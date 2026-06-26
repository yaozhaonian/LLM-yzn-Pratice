# 假设性问题索引
# 基于一个或多个假设的情况或前提来使用llm提出问题,用问题向量替换文档的块向量,候选的相关性问题和切片的内容强相关
"""
核心适用场景（按行业 / 场景分类）
1. 企业内部知识库 / 智能问答系统
场景痛点：员工提问方式口语化、多样化，和知识库的正式文档表述差异大（比如员工问 “报销打车费要啥材料”，文档写 “差旅交通费用报销凭证要求”），传统检索容易漏匹配。
应用价值：提前为每个文档块生成 3-5 个用户真实会问的假设问题，用户提问时直接匹配问题向量，精准定位对应文档，大幅提升内部问答机器人的准确率。
典型场景：企业 HR 制度问答、IT 运维知识库、行政流程查询、产品手册问答。
2. 智能客服 / 售后支持系统
场景痛点：用户提问五花八门（比如 “我的快递没收到怎么办”“怎么退货”“退款多久到账”），而知识库是结构化的售后政策文档，直接用用户问题匹配文档向量容易语义错位。
应用价值：为每个售后规则切片生成用户高频提问，用户咨询时快速匹配对应问题，定位到正确的政策条款，提升客服响应效率和准确率，减少人工介入。
典型场景：电商平台售后客服、SaaS 产品用户支持、运营商 / 银行客服机器人。
3. 法律 / 合规 / 金融专业知识库
场景痛点：专业文档（合同、法规、监管政策）术语密集、表述严谨，用户 / 律师的提问是业务场景化的（比如 “这个违约金条款适用于跨境合同吗”），直接匹配难度极高。
应用价值：为每个法律条款 / 金融规则生成场景化假设问题，快速定位相关条款，辅助合同审查、合规检查、法律咨询。
典型场景：律所合同审查系统、企业合规知识库、金融监管政策检索、法律条文智能问答。
4. 学术 / 科研文献检索系统
场景痛点：论文内容专业度高，研究者的提问是针对研究细节的（比如 “这个实验的样本量是多少”“该研究用了什么统计方法”），传统全文检索难以精准定位段落。
应用价值：为论文切片生成研究相关的假设问题，帮助研究者快速定位到论文中对应细节，提升文献检索效率。
典型场景：学术数据库问答、论文辅助阅读工具、科研知识库。
5. 产品 / 技术文档检索（To B/To C）
场景痛点：技术文档（API 手册、产品说明书、运维指南）内容专业，用户提问是问题导向的（比如 “怎么解决这个报错”“这个接口怎么调用”），直接匹配文档容易找错段落。
应用价值：为每个技术点生成用户可能遇到的问题，用户提问时精准定位对应解决方案，提升自助排查效率。
典型场景：SaaS 产品技术文档、开源项目文档、硬件设备说明书问答。
6. 电商 / 内容推荐系统
场景痛点：用户用自然语言描述需求（比如 “适合油皮的夏天用的防晒霜”），而商品详情页是产品参数，语义匹配难度大。
应用价值：为每个商品 / 内容切片生成用户可能的搜索问题，用问题向量匹配用户需求，提升搜索精准度。
典型场景：电商商品搜索、内容平台知识问答、导购机器人。
"""
"""
技术层面的适用场景（解决的核心问题）
用户提问与文档表述语义不匹配：用户用口语化、场景化提问，文档是正式、书面化内容，传统向量检索跨域匹配效果差，假设性问题索引通过「问题 - 问题」对称检索解决该问题。
提升 RAG 系统的召回率和准确率：解决传统 RAG 中 “用户问题短、语义模糊，无法精准匹配长文档” 的问题，通过多假设问题覆盖用户提问的多种可能性。
多轮对话 / 复杂问题检索：用户提问是多轮、上下文相关的，提前生成的假设问题可以覆盖多轮提问的潜在语义，提升复杂问题的检索效果。
低资源 / 冷启动知识库：新知识库没有历史用户提问数据，通过 LLM 自动生成假设问题，快速搭建高质量检索索引，无需人工标注问答对。
"""
from typing import List
from pathlib import Path
import uuid
from pydantic import BaseModel, Field

# LangChain 组件
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.stores import InMemoryByteStore
from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel

embedding = OllamaEmbeddings(model="bge-m3:latest",base_url="http://127.0.0.1:11434")
llm = ChatOllama(model='qwen2.5:7b', temperature=0.1,base_url="http://127.0.0.1:11434")

current_dir = Path(__file__).parent
data_file_path = current_dir.parent / "Data" / "deepseek百度百科.txt"

# 检查文件是否存在
if not data_file_path.exists():
    raise FileNotFoundError(f"找不到数据文件: {data_file_path}")

loader = TextLoader(str(data_file_path), encoding="utf-8")
doc = loader.load()

# 分割文本
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
docs = text_splitter.split_documents(doc)

# 初始化向量数据库
vectorstore = Chroma(
    persist_directory="./data/chroma_pqi", 
    embedding_function=embedding
)
# 初始化内存存储(存储原始文档)
bytestore = InMemoryByteStore()

id_key = "ds_doc_id"

# 配置多向量检索器
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=bytestore,
    id_key=id_key,
)

# 为每个原始文档生成唯一ID
doc_ids = [str(uuid.uuid4()) for _ in docs]

class HypotheticalQuestions(BaseModel):
    """约束生成假设性问题的格式"""
    questions: List[str] = Field(..., description="List of questions")

prompt = ChatPromptTemplate.from_template(
        """请基于以下文档生成3个假设性问题（必须使用JSON格式）:
        {doc}
        
        要求：
        1. 输出必须为合法JSON格式，包含questions字段
        2. questions字段的值是包含3个问题的数组
        3. 使用中文提问
        示例格式：
        {{
            "questions": ["问题1", "问题2", "问题3"]
        }}"""
)

# 创建假设性问题链
'''
其中的llm.with_structured_output可以理解为输出解析器的一种更高级用法
将大模型的输出转换为HypotheticalQuestions所限定的格式，
而HypotheticalQuestions要求的格式是：
定义了一个字段 questions，它具有以下特性：
类型注解：List[str] 表示 questions 字段应该是一个字符串列表。
必需性：Field(...) 中的省略号 ... 表示这个字段是必需的。
描述信息：description="List of questions" 为该字段添加了描述，这对于生成文档或帮助理解模型结构很有用。
'''
chain = (
    {"doc": lambda x: x.page_content}
    | prompt
    # 将LLM输出构建为字符串列表
    | llm.with_structured_output(
        HypotheticalQuestions
    )
    # 提取问题列表
    | (lambda x: x.questions)
)

# 测试-在单个文档上调用链，链的最终输出是大模型答复的假设性问题列表
print("测试：",docs[0])
print("测试生成问题：",chain.invoke(docs[0]))

# 批量处理所有文档生成假设性问题（最大并行数5），每个切块后的文档块都对应的生成三个问题
hypothetical_questions = chain.batch(docs, {"max_concurrency": 5})
print("假设性问题列表：",hypothetical_questions)

# 将生成的问题转换为带元数据的文档对象
question_docs = []
for i, question_list in enumerate(hypothetical_questions):
    question_docs.extend([
        Document(page_content=s, metadata={id_key: doc_ids[i]}) for s in question_list
    ])

print('文档对象\n',question_docs)
 # 将问题文档存入向量数据库
retriever.vectorstore.add_documents(question_docs)
 # 将原始文档存入字节存储（通过ID关联）
retriever.docstore.mset(list(zip(doc_ids, docs)))

# query = "DeepSeek公司发布了多个模型版本，请问这些新发布的模型分别在哪些领域提供了支持？"
query = str(input("请输入问题："))
sub_docs = retriever.vectorstore.similarity_search(query)
print("=========相似性：=========")
print("测试-执行相似性搜索：",sub_docs)

promptqa = ChatPromptTemplate.from_template(
    """请基于以下文档回答问题：
    {doc}
    问题：{question}
    """
)

chainqa = RunnableParallel(
    {
        "doc": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"]
    }
) | promptqa | llm | StrOutputParser()

# 生成问题回答
answer = chainqa.invoke({"question": query})
print("=========回答=========")
print(answer)
#  返回的是知识块
retrieved_docs = retriever.invoke(query) 
print("=========检索到的问题=========")
print(retrieved_docs)


