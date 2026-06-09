# 进一步:向量化
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings 
embeddings = OllamaEmbeddings( 
    model="bge-m3:latest",
    base_url="http://127.0.0.1:11434"
) 
# text = "This is a test query." 
# query_result = embeddings.embed_query(text)
# print(query_result)
# print(f"向量维度：{len(query_result)}")

loader = WebBaseLoader(
    web_path=["http://www.banyuetan.org/yw/detail/20260401/1000200033137441775007039426865103_1.html"],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=("detail_tit","detail_content"))
    )
)
docs = loader.load()
# print(docs)
# print()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
documents = splitter.split_documents(docs)

# 实例化向量空间
# 网址：https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma
from langchain_chroma import Chroma
db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)
# 检索器
retriver = db.as_retriever()

# 注意这里的prompt模板中包含 {context} 和 {input} 的模板
#   需要使用{context}，这个变量，来表示上下文，这个变量，会自动从retriever中获取。
#   而human中也限定了变量{input}，链的必须使用这个变量。
system_prompt = """
您是问答任务的助理。使用以下的上下文来回答问题，
上下文：<{context}>
如果你不知道答案，不要其他渠道去获得答案，就说你不知道。
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

llm = ChatOllama(model="qwen2.5:7b",temperature=0.1)

# 创建文档链
chain1 = create_stuff_documents_chain(llm=llm, prompt=prompt)
# 创建检索链
# 网址:https://reference.langchain.com/python/langchain-classic/chains/retrieval/create_retrieval_chain
from langchain_classic.chains import create_retrieval_chain
chain2 = create_retrieval_chain(retriever=retriver, combine_docs_chain=chain1)
# 对于有多个匹配的,不知道为什么有不少丢失的,可能内置的是只输出得分最高的
# resp = chain2.invoke({"input": "请回答：回答一下习近平强调了什么？要求:全面无误，一字不改"})

resp = chain2.invoke({"input": "请回答：回答一下政府工作报告提出了什么？"})
print(type(resp))
print("="*50)
print(resp['input'])
print("="*50)
print(resp['context'])
print("="*50)
print(resp['answer'])