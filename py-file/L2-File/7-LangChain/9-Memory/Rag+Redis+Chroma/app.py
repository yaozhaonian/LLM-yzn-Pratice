from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
import uuid
from typing import List
import hashlib

from config import BASE_DIR, DOCUMENTS_DIR, ALLOWED_EXTENSIONS, MAX_FILE_SIZE
from rag_core.document_loader import DocumentLoader
from rag_core.rag_engine import RAGEngine

app = FastAPI(title="RAG+Redis+Chroma 知识库问答系统")

# 挂载静态文件和模板
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# 全局 RAG 引擎（按知识库名称管理）
rag_engines = {}
knowledge_bases = {}

# ====================== 页面路由 ======================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    has_knowledge = len(knowledge_bases) > 0
    return templates.TemplateResponse("index.html", {
        "request": request,
        "has_knowledge": has_knowledge,
        "knowledge_bases": knowledge_bases
    })

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    if not knowledge_bases:
        raise HTTPException(status_code=400, detail="请先创建知识库")
    
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "knowledge_bases": knowledge_bases
    })

# ====================== API 路由 ======================
@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...), knowledge_name: str = Form(...)):
    if not knowledge_name:
        raise HTTPException(status_code=400, detail="知识库名称不能为空")
    
    uploaded_docs = []
    
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型：{ext}")
        
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"文件过大：{file.filename}")
        
        if ext == ".docx" and not content.startswith(b'PK'):
            raise HTTPException(status_code=400, detail=f"文件 {file.filename} 不是有效的 .docx 格式")
        
        file_id = str(uuid.uuid4())[:8]
        file_path = DOCUMENTS_DIR / f"{file_id}_{file.filename}"
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        try:
            docs = DocumentLoader.load_file(file_path, chunk_size=500)
        except Exception as e:
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=400, detail=f"文件解析失败：{file.filename}, 错误：{str(e)}")
        
        for idx, doc in enumerate(docs):
            doc['metadata']['file_id'] = file_id
            doc['metadata']['filename'] = file.filename
            doc['metadata']['chunk_index'] = idx
            content_hash = hashlib.md5(doc['content'].encode()).hexdigest()[:8]
            doc['metadata']['doc_id'] = f"{file_id}_{idx}_{content_hash}"
        
        uploaded_docs.extend(docs)
    
    if knowledge_name in rag_engines:
        rag_engines[knowledge_name].vector_store.clear()
        del rag_engines[knowledge_name]
    
    rag_engine = RAGEngine(collection_name=knowledge_name)
    rag_engine.load_documents(uploaded_docs)
    rag_engines[knowledge_name] = rag_engine
    
    knowledge_bases[knowledge_name] = {
        "name": knowledge_name,
        "document_count": len(uploaded_docs),
        "files": list(set([doc['metadata']['filename'] for doc in uploaded_docs]))
    }
    
    return JSONResponse({
        "message": "上传成功",
        "knowledge_name": knowledge_name,
        "document_count": len(uploaded_docs),
        "files": knowledge_bases[knowledge_name]['files']
    })

@app.post("/api/chat")
async def chat(query: str = Form(...), knowledge_name: str = Form(...), user_id: str = Form("default_user")):
    """带记忆的聊天问答"""
    if knowledge_name not in rag_engines:
        raise HTTPException(status_code=400, detail="知识库不存在")
    
    rag_engine = rag_engines[knowledge_name]
    result = rag_engine.chat_with_memory(user_id, query, n_results=12)
    
    return JSONResponse(result)

@app.post("/api/chat/clear_memory")
async def clear_memory(knowledge_name: str = Form(...), user_id: str = Form("default_user")):
    """清空用户记忆"""
    if knowledge_name not in rag_engines:
        raise HTTPException(status_code=400, detail="知识库不存在")
    
    rag_engine = rag_engines[knowledge_name]
    rag_engine.clear_user_memory(user_id)
    
    return JSONResponse({"message": "记忆已清空"})

@app.get("/api/knowledge")
async def get_knowledge_list():
    return JSONResponse({"knowledge_bases": list(knowledge_bases.values())})

@app.delete("/api/knowledge/{name}")
async def delete_knowledge(name: str):
    if name in rag_engines:
        rag_engines[name].vector_store.clear()
        del rag_engines[name]
    if name in knowledge_bases:
        del knowledge_bases[name]
    
    return JSONResponse({"message": "删除成功"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)