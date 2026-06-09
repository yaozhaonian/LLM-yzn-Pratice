# 元数据索引
import json
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_core.documents import Document
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.chains.query_constructor.base import get_query_constructor_prompt, StructuredQueryOutputParser


# 1. 初始化组件
embedding = OllamaEmbeddings(model="bge-m3:latest")
# 建议：如果可能，使用参数量更大的模型，如 qwen2.5:7b 或 14b，或者使用线上模型，小模型很难严格遵守复杂格式
llm = ChatOllama(model='qwen2.5:7b', temperature=0.1) 

# 2. 加载文档
docs = [
    Document(
        page_content="作者A团队开发出基于人工智能的自动驾驶决策系统，在复杂路况下的响应速度提升300%",
        metadata={"year": 2024, "rating": 9.2, "genre": "AI", "author": "A"},
    ),
    Document(
        page_content="区块链技术成功应用于跨境贸易结算，作者B主导的项目实现交易确认时间从3天缩短至30分钟",
        metadata={"year": 2023, "rating": 9.8, "genre": "区块链", "author": "B"},
    ),
    Document(
        page_content="云计算平台实现量子计算模拟突破，作者C构建的新型混合云架构支持百万级并发计算",
        metadata={"year": 2022, "rating": 8.6, "genre": "云", "author": "C"},
    ),
    Document(
        page_content="大数据分析预测2024年全球经济趋势，作者A团队构建的模型准确率超92%",
        metadata={"year": 2023, "rating": 8.9, "genre": "大数据", "author": "A"},
    ),
    Document(
        page_content="人工智能病理诊断系统在胃癌筛查中达到三甲医院专家水平，作者B获医疗科技创新奖",
        metadata={"year": 2024, "rating": 7.1, "genre": "AI", "author": "B"},
    ),
    Document(
        page_content="基于区块链的数字身份认证系统落地20省市，作者C设计的新型加密协议通过国家级安全认证",
        metadata={"year": 2022, "rating": 8.7, "genre": "区块链", "author": "C"},
    ),
    Document(
        page_content="云计算资源调度算法重大突破，作者A研发的智能调度器使数据中心能效提升40%",
        metadata={"year": 2023, "rating": 8.5, "genre": "云", "author": "A"},
    ),
    Document(
        page_content="大数据驱动城市交通优化系统上线，作者B团队实现早晚高峰通行效率提升25%",
        metadata={"year": 2024, "rating": 7.4, "genre": "大数据", "author": "B"},
    )
]

vectorstore = Chroma.from_documents(docs, embedding)

# 3. 定义元数据字段
metadata_field_info = [
    AttributeInfo(
        name="year",
        description="文章的出版年份 (integer)",
        type="integer",
    ),
    AttributeInfo(
        name="rating",
        description="技术价值的评分 (float, 1-10分)",
        type="float",
    ),
    AttributeInfo(
        name="author",
        description="署名文章的作者姓名 (string)",
        type="string",
    ),
    AttributeInfo(
        name="genre",
        description="文章的技术领域 (string)，可选值: 'AI', '区块链', '云', '大数据'",
        type="string",
    )
]

document_content_description = "技术文章简述"

# 4. 【关键修改】手动构建更稳健的查询构造链
# 原始的 from_llm 可能因为 Prompt 不够强导致小模型出错。
# 我们使用标准的 get_query_constructor_prompt，但确保 LLM 温度低且模型足够强。

prompt = get_query_constructor_prompt(
    document_contents=document_content_description,
    attribute_info=metadata_field_info,
)

output_parser = StructuredQueryOutputParser.from_components()

# 构建链
query_constructor = prompt | llm | output_parser

# 5. 测试查询
test_query = "作者C在2022年发布的文章"
print(f"用户查询: {test_query}")
print("=" * 50)

try:
    # 调用链
    structured_query = query_constructor.invoke({"query": test_query})
    
    print("结构化查询对象:")
    print(structured_query)
    print("=" * 50)
    
    # 6. 使用 SelfQueryRetriever 执行检索
    # 注意：这里我们可以直接用刚才生成的 structured_query，或者重新让 retriever 生成
    # 为了演示完整流程，我们通常让 retriever 自己处理，但如果想调试，可以手动翻译
    
    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        verbose=True # 开启verbose可以看到中间过程
    )
    
    # 执行检索
    results = retriever.invoke(test_query)
    
    print(f"检索到 {len(results)} 条结果:")
    for doc in results:
        print(f"- 作者: {doc.metadata['author']}, 年份: {doc.metadata['year']}, 评分: {doc.metadata['rating']}")
        print(f"  内容: {doc.page_content[:50]}...")

except Exception as e:
    print(f"发生错误: {e}")
    print("\n建议:")
    print("1. 检查 LLM 模型是否足够强大 (推荐 qwen2.5:7b 或更高)。")
    print("2. 确保已安装 lark: pip install lark")
    print("3. 检查 metadata_field_info 中的 type 是否与数据实际类型一致。")