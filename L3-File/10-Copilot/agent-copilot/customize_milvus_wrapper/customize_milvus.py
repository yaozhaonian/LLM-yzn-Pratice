from typing import List

from pymilvus import MilvusClient, DataType
from pymilvus.milvus_client.index import IndexParams
from tqdm import tqdm

from entity import Tool
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.remote_embedding_model import RemoteEmbeddingModel
from utils import logger

class CustomizeMilvus:
    def __init__(self, uri, db_name):
        self.client = MilvusClient(uri=uri)
        self.embeddingModel = RemoteEmbeddingModel()
        databases = self.client.list_databases()
        if db_name not in databases:
            self.client.create_database(db_name)
        self.client.use_database(db_name)
    
    def list_collections(self):
        return self.client.list_collections()
    
    def has_collection(self, collection_name):
        return collection_name in self.list_collections()
    
    def load_collection_into_memory(self, collection_name):
        try:
            load_state = self.client.get_load_state(collection_name)
            if load_state.get('state') == 'NotLoaded' or load_state.get('state') == 1:
                self.client.load_collection(collection_name)
        except Exception:
            try:
                self.client.load_collection(collection_name)
            except Exception:
                pass
    
    def create_collection(self, collection_name):
        embedding_dim = 1024
        
        schema = self.client.create_schema(
            auto_id=False,
            description="Tool collection"
        )
        schema.add_field("tool_id", DataType.INT64, is_primary=True)
        schema.add_field("operation_summary", DataType.VARCHAR, max_length=3000)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=embedding_dim)
        
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema
        )
        
        index_params = IndexParams()
        index_params.add_index("embedding", index_type="FLAT", metric_type="COSINE")
        
        self.client.create_index(
            collection_name=collection_name,
            index_params=index_params
        )
        
        logger.info(f"集合 {collection_name} 创建成功")
    
    def insert_embeddings(self, collection_name, embeddings, datas):
        self.load_collection_into_memory(collection_name)
        batch_datas = []
        for data, embedding in tqdm(zip(datas, embeddings)):
            data['embedding'] = embedding
            batch_datas.append(data)
            if len(batch_datas) >= 10:
                self.client.insert(collection_name=collection_name, data=batch_datas)
                batch_datas = []
        if batch_datas:
            self.client.insert(collection_name=collection_name, data=batch_datas)
            batch_datas = []

    def drop_collection(self, collection_name):
        if self.has_collection(collection_name):
            self.client.drop_collection(collection_name=collection_name)

    def embed_chunked_data(self, tools: List[Tool]):
        datas = []
        chunk_texts = []
        for tool in tools:
            new_data = {
                "tool_id": tool.tool_id,
                "operation_summary": f"{tool.operationId}: {tool.name_for_human}: {tool.description}"
            }
            datas.append(new_data)
            chunk_texts.append(new_data['operation_summary'])
        embedding = self.embeddingModel.get_embedding(chunk_texts)
        # 规范化返回的嵌入：确保为嵌套列表，每个向量为 float 列表
        norm_embeddings = []
        try:
            for emb in embedding:
                # 如果嵌套一层（如 [[...],[...]] 中的每项仍是列表），直接使用
                if isinstance(emb, list):
                    norm_embeddings.append([float(x) for x in emb])
                else:
                    # 如果返回单个向量不是列表，尝试转换为 float
                    norm_embeddings.append([float(emb)])
        except Exception as e:
            logger.error(f"embed_chunked_data: 嵌入向量格式不正确: {e}; raw={embedding}")
            raise
        return norm_embeddings, datas


    def insert_tools(self, collection_name, tools: List[Tool]):
        if not self.has_collection(collection_name):
            self.create_collection(collection_name)
        embeddings, datas = self.embed_chunked_data(tools)
        # 确保每个 embedding 都是 float 列表
        try:
            sanitized = [[float(x) for x in emb] for emb in embeddings]
        except Exception as e:
            logger.error(f"insert_tools: 批量嵌入向量格式错误: {e}; sample={embeddings[:2]}")
            raise
        self.insert_embeddings(collection_name, sanitized, datas)

    def get_docs(self, collection_name, query_text, top_k=5):
        self.load_collection_into_memory(collection_name)
        query_embedding = self.embeddingModel.get_embedding(query_text)
        # 规范化查询向量：如果返回 batch（嵌套列表），取第一个向量；并确保元素为 float
        try:
            if isinstance(query_embedding, list) and len(query_embedding) > 0 and isinstance(query_embedding[0], list):
                qe = query_embedding[0]
            else:
                qe = query_embedding
            qe = [float(x) for x in qe]
        except Exception as e:
            logger.error(
                f"get_docs: 查询嵌入向量格式错误: {e}; raw_query_embedding={query_embedding}; query_text={query_text!r}"
            )
            raise
        query_embedding = qe
        try:
            response = self.client.search(
                collection_name=collection_name, 
                data=[query_embedding], 
                limit=top_k,           
                search_params={
                    "metric_type": "COSINE",
                    "params": {"level": 1}
                },
                output_fields=["tool_id", "operation_summary"],
                consistency_level="Bounded"
            )
        except Exception as e:
            logger.error(
                f"get_docs: Milvus search failed for query_text={query_text!r}; vector_length={len(query_embedding)}; vector_sample={query_embedding[:8]}; error={e}"
            )
            raise
        docs = []
        for res in response:
            for result in res:
                chunk = result["entity"]
                docs.append(int(chunk["tool_id"]))
        return docs

    def delete_tools(self, tools: List[Tool]):
        ids = []
        for tool in tools:
            ids.append(tool.tool_id)
        self.client.delete(collection_name="tools", ids=ids)

    def get_all_entity(self, ids):
        return self.client.get(collection_name="tools", ids=ids, output_fields=["tool_id", "operation_summary"])
