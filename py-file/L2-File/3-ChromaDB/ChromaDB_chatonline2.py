# 不用本地模型，需要消耗tokens的简单demo,针对非键值对的普通文档
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from openai import OpenAI
import inspect
from pathlib import Path
ALI_TONGYI_API_KEY_OS_VAR_NAME = "DASHSCOPE_API_KEY"
ALI_TONGYI_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALI_TONGYI_EMBEDDING_V4 = "text-embedding-v4"
ALI_TONGYI_TURBO_MODEL = "qwen-turbo-latest"
def get_oline_client(api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
                      base_url=ALI_TONGYI_URL,
                      verbose=False, debug=False):
    """
    使用原生api获得指定平台的客户端，但未指定具体模型，缺省平台为阿里云百炼
    也可以通过传入api_key，base_url两个参数来覆盖默认值
    verbose，debug两个参数，分别控制是否输出调试信息，是否输出详细调试信息，默认不打印
    """
    function_name = inspect.currentframe().f_code.co_name
    if (verbose):
        print(f"{function_name}-平台：{base_url}")
    if (debug):
        print(f"{function_name}-平台：{base_url},key：{api_key}")
    return OpenAI(api_key=api_key, base_url=base_url)


script_dir = Path(__file__).parent
chroma_db_path = script_dir / "./chroma_db"
cache_file = script_dir / ".doc_cache.json"  # 缓存文件记录文档状态

# 查找数据文件
possible_paths = [
    (script_dir / "../Data/train.json").resolve(),
    (script_dir / "../../Data/train.json").resolve(),
    Path(r"E:\py-file\L2-File\Data\train.json").resolve(),
]

file_path = None
for path in possible_paths:
    if path.exists():
        file_path = path
        print(f"✓ 找到文件：{file_path}")
        break

if not file_path:
    print("✗ 未找到数据文件，创建测试文件")
    file_path = script_dir / "train.json"
    file_path.write_text("这是一个测试文档。\nDeepSeek 是一款人工智能模型。\n", encoding='utf-8')
    print(f"✓ 已创建测试文件：{file_path}")
# 1.加载文档，读取数据
with open('../Data/deepseek百度百科.txt', 'r', encoding='utf-8') as f:
    content = f.read()
print(len(content))


# 负责和向量数据库打交道，接收文档转为向量，并保存到向量数据库中，然后根据需要从向量库中检索出最相似的记录
class MyVectorDBConnector:
    # 初始化，传入集合名称，和向量化函数名
    def __init__(self, collection_name):
        # 当前配置中，数据保存在内存中，如果需要持久化到磁盘，需使用 PersistentClient创建客户端
        chroma_client = chromadb.Client(Settings(allow_reset=True))

        # 持久化到磁盘
        # chroma_client = chromadb.PersistentClient(path="./chroma_data")
        # 为了演示，实际不需要每次 reset()
        # chroma_client.reset()

        # 创建一个 collection
        self.collection = chroma_client.get_or_create_collection(name=collection_name)

        # 连接大模型的客户端
        self.client = get_oline_client()

    # 向量化
    def get_embeddings(self, texts, model=ALI_TONGYI_EMBEDDING_V4):
        '''封装 OpenAI 的 Embedding 模型接口'''
        data = self.client.embeddings.create(input=texts, model=model).data
        return [x.embedding for x in data]

    # get_embeddings函数的变体版，因为各个模型对一次能处理的文本条数有限制且每个平台不一致，新增一个batch_size参数用以控制。
    def get_embeddings_batch(self,texts, model=ALI_TONGYI_EMBEDDING_V4, batch_size=10):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_text = texts[i:i + batch_size]
            data = self.client.embeddings.create(input=batch_text, model=model).data
            all_embeddings.extend([x.embedding for x in data])
        return all_embeddings

    # 添加文档与向量
    def add_documents(self, documents):
        '''向 collection 中添加文档与向量'''
        embeddings = self.get_embeddings_batch(documents)

        self.collection.add(
            embeddings=embeddings,  # 每个文档的向量
            documents=documents,  # 文档的原文
            ids=[f"id{i}" for i in range(len(documents))]  # 每个文档的 id
        )
        print("self.collection.count():", self.collection.count())

    # 检索向量数据库
    def search(self, query, top_k):
        ''' 检索向量数据库
           query是用户的查询，
           top_k指查出top_k个相似高的记录
        '''
        results = self.collection.query(
            query_embeddings=self.get_embeddings_batch([query]),
            n_results=top_k
        )
        return results

# 创建一个向量数据库对象
vector_db = MyVectorDBConnector("demo")


# 2.切分文档
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 分割长度
    chunk_overlap=50,  # 重叠长度 /重叠窗口大小
    separators=["\n\n", "\n", "。", "?", "，", ""],
)
texts = splitter.split_text(content)
# for i, chunk in enumerate(texts):
#     print(f"块 {i + 1} - 长度{len(chunk)}，内容: {chunk}")


# 3.将文档存入向量数据库
# 向 向量数据库 中添加文档
vector_db.add_documents(texts)


# 4. 开始检索
user_query = "DeepSeek的全称是什么?"
user_query = "deepseek的发展历程"
user_query = '360集团创始人周鸿祎说了什么?'
results = vector_db.search(user_query, 5)
# print(results)
# print('-' * 100)

# for i in results['documents'][0]:
#     print(i + "\n----\n")
# 相关文档
contents = '\n'.join(results['documents'][0])

# 5. llm
prompt = f"""
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。
确保你的回复完全依据下述已知信息。不要编造答案。
如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

已知信息:
{contents}

----
用户问：
{user_query}

请用中文回答用户问题。
"""

def get_completion(prompt, model=ALI_TONGYI_TURBO_MODEL):
    '''封装 openai 接口'''
    messages = [{"role": "user", "content": prompt}]
    client = get_oline_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
    )
    return response.choices[0].message.content

response = get_completion(prompt)
print(response)
