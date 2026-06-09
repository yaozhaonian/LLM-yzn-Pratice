# 爬虫结合langchain链
from langchain_core.prompts import ChatPromptTemplate
# 相关网址:https://reference.langchain.com/python/langchain-classic/chains/combine_documents/stuff/create_stuff_documents_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_ollama import ChatOllama


"""
create_stuff_documents_chain
构建链  这个链将文档作为输入，并使用之前定义的提示模板和初始化的大模型来生成答案
链要求输入是一个字典，必须包含context,默认就是将context中的内容作为文档给大模型
"""
prompt = ChatPromptTemplate.from_messages(
    [("system", "根据提供的上下文: {context} \n\n 回答问题"),
     ("user","问题：{input}")
    ]
)
model = ChatOllama(model="qwen2.5:7b",temperature=0.9)
chain = create_stuff_documents_chain(model, prompt)

from langchain_community.document_loaders import WebBaseLoader
import bs4
# 加载文档  bs :Beautiful Soup 解析器
loader = WebBaseLoader(
    # 网页网址,一般用官方网站，这个是十五五规划
    web_path="https://www.gov.cn/gongbao/2025/issue_12386/202511/content_7047415.html?f_link_type=f_linkinlinenote&flow_extra=eyJpbmxpbmVfZGlzcGxheV9wb3NpdGlvbiI6MCwiZG9jX3Bvc2l0aW9uIjowLCJkb2NfaWQiOiI5OTBhMjE5NTlhYWRkM2FjLTJlMTMwNzRmNGUyYmE5ZDQifQ%3D%3D",
    # bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="UCAP-CONTENT"))
    bs_kwargs={"parse_only":bs4.SoupStrainer(id="UCAP-CONTENT")}
)

docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
# 分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
documents = text_splitter.split_documents(docs)
print(len(documents))

res = chain.invoke({"input":"总结一下文档内容，要求：600字以内，简洁、不缺失重点","context": documents})
print(res)

