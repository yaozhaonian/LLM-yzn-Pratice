from pathlib import Path
from typing import List, Dict
import json

class DocumentLoader:
    """文档加载器"""
    
    @staticmethod
    def load_text(file_path: Path, chunk_size: int = 500) -> List[Dict]:
        """加载文本文件（按语义块分割）"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        documents = []
        
        # 尝试解析 JSON Lines
        if file_path.suffix.lower() == '.json':
            for i, line in enumerate(content.strip().split('\n')):
                if line.strip():
                    try:
                        data = json.loads(line)
                        # 合并 instruction 和 output 作为一个完整文档
                        text = f"问题：{data.get('instruction', '')}\n答案：{data.get('output', '')}"
                        documents.append({
                            "content": text,
                            "metadata": {"source": file_path.name, "line": i, **data}
                        })
                    except:
                        documents.append({
                            "content": line,
                            "metadata": {"source": file_path.name, "line": i}
                        })
        else:
            # 按段落分割（空行分隔）
            paragraphs = content.split('\n\n')
            for i, para in enumerate(paragraphs):
                if para.strip():
                    # 如果段落太长，按行合并成块
                    lines = para.strip().split('\n')
                    current_chunk = ""
                    chunk_start_line = i
                    
                    for j, line in enumerate(lines):
                        if len(current_chunk) + len(line) > chunk_size and current_chunk:
                            # 保存当前块
                            documents.append({
                                "content": current_chunk,
                                "metadata": {"source": file_path.name, "paragraph": chunk_start_line}
                            })
                            current_chunk = line
                        else:
                            current_chunk += "\n" + line if current_chunk else line
                    
                    # 保存最后一块
                    if current_chunk.strip():
                        documents.append({
                            "content": current_chunk,
                            "metadata": {"source": file_path.name, "paragraph": chunk_start_line}
                        })
        
        return documents
    
    @staticmethod
    def load_file(file_path: Path, chunk_size: int = 500) -> List[Dict]:
        """根据文件类型加载"""
        ext = file_path.suffix.lower()
        
        if ext in ['.txt', '.json', '.md']:
            return DocumentLoader.load_text(file_path, chunk_size)
        elif ext == '.pdf':
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            documents = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    documents.append({
                        "content": text,
                        "metadata": {"source": file_path.name, "page": i}
                    })
            return documents
        elif ext == '.docx':
            from docx import Document
            doc = Document(file_path)
            documents = []
            for i, para in enumerate(doc.paragraphs):
                if para.text.strip():
                    documents.append({
                        "content": para.text,
                        "metadata": {"source": file_path.name, "paragraph": i}
                    })
            return documents
        
        return []