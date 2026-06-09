# 利用LangChain部署应用成为WEB服务


# 使用fastapi框架
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

from langchain_core.prompts import ChatPromptTemplate
chatprompt = ChatPromptTemplate.from_messages([
    ("system", "请将以下的内容翻译成{language}"),
    ("user", "{text}")
])

from langchain_ollama import ChatOllama 
llm = ChatOllama(model="qwen2.5:7b", temperature=0.9)

chain = chatprompt | llm | parser

#部署为服务
app = FastAPI(title="LangChain Ollama Demo", version="V1.0", description="A simple demo of LangChain with Ollama")
# 添加路由，给当前的程序添加一个访问路径
from langserve import add_routes
add_routes(app, chain,path="/tslServer")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8088)





