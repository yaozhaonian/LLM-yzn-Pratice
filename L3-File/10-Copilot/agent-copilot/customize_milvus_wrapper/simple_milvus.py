from typing import List
import numpy as np
import os
import json
from entity import Tool
from models.simple_embedding_model import SimpleEmbeddingModel
from utils import logger

class SimpleMilvus:
    def __init__(self, uri, db_name):
        self.uri = uri
        self.db_name = db_name
        self.embeddingModel = SimpleEmbeddingModel()
        self.collections = {}
        self._load_from_disk()
    
    def _get_data_dir(self):
        return os.path.join(os.path.dirname(__file__), '..', 'data', self.db_name)
    
    def _load_from_disk(self):
        data_dir = self._get_data_dir()
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith('.json'):
                    collection_name = filename[:-5]
                    filepath = os.path.join(data_dir, filename)
                    with open(filepath, 'r') as f:
                        self.collections[collection_name] = json.load(f)
    
    def _save_to_disk(self):
        data_dir = self._get_data_dir()
        os.makedirs(data_dir, exist_ok=True)
        for collection_name, data in self.collections.items():
            filepath = os.path.join(data_dir, f'{collection_name}.json')
            with open(filepath, 'w') as f:
                json.dump(data, f)
    
    def list_collections(self):
        return list(self.collections.keys())
    
    def has_collection(self, collection_name):
        return collection_name in self.collections
    
    def create_collection(self, collection_name):
        if collection_name not in self.collections:
            self.collections[collection_name] = {
                'data': [],
                'embeddings': []
            }
            self._save_to_disk()
        logger.info(f"集合 {collection_name} 创建成功")
    
    def insert_embeddings(self, collection_name, embeddings, datas):
        if collection_name not in self.collections:
            self.create_collection(collection_name)
        
        for data, embedding in zip(datas, embeddings):
            self.collections[collection_name]['data'].append(data)
            self.collections[collection_name]['embeddings'].append(embedding)
        
        self._save_to_disk()
    
    def drop_collection(self, collection_name):
        if collection_name in self.collections:
            del self.collections[collection_name]
            self._save_to_disk()
    
    def load_collection_into_memory(self, collection_name):
        pass
    
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
        embeddings = self.embeddingModel.get_embedding(chunk_texts)
        return embeddings, datas
    
    def insert_tools(self, collection_name, tools: List[Tool]):
        if not self.has_collection(collection_name):
            self.create_collection(collection_name)
        embeddings, datas = self.embed_chunked_data(tools)
        self.insert_embeddings(collection_name, embeddings, datas)
    
    def get_docs(self, collection_name, query_text, top_k=5):
        if collection_name not in self.collections:
            return []
        
        data = self.collections[collection_name]['data']
        embeddings = self.collections[collection_name]['embeddings']
        
        if not data or not embeddings:
            return []
        
        query_embedding = self.embeddingModel.get_embedding(query_text)[0]
        query_vec = np.array(query_embedding)
        embeddings_mat = np.array(embeddings)
        
        if len(embeddings_mat) == 0:
            return []
        
        norm_query = np.linalg.norm(query_vec)
        norm_vectors = np.linalg.norm(embeddings_mat, axis=1)
        
        mask = norm_vectors > 0
        if not np.any(mask):
            return []
        
        cos_sims = np.zeros(len(embeddings_mat))
        cos_sims[mask] = np.dot(query_vec, embeddings_mat[mask].T) / (norm_query * norm_vectors[mask])
        
        sorted_indices = np.argsort(cos_sims)[::-1]
        top_indices = sorted_indices[:top_k]
        
        result = []
        for idx in top_indices:
            if cos_sims[idx] > 0.1:
                result.append(int(data[idx]['tool_id']))
        
        return result
    
    def delete_tools(self, tools: List[Tool]):
        if 'tools' not in self.collections:
            return
        
        tool_ids = {t.tool_id for t in tools}
        data = self.collections['tools']['data']
        embeddings = self.collections['tools']['embeddings']
        
        new_data = []
        new_embeddings = []
        for d, e in zip(data, embeddings):
            if d['tool_id'] not in tool_ids:
                new_data.append(d)
                new_embeddings.append(e)
        
        self.collections['tools']['data'] = new_data
        self.collections['tools']['embeddings'] = new_embeddings
        self._save_to_disk()
    
    def get_all_entity(self, ids):
        if 'tools' not in self.collections:
            return []
        
        data = self.collections['tools']['data']
        result = []
        for d in data:
            if d['tool_id'] in ids:
                result.append(d)
        return result
