from typing import List

from pymilvus import db, MilvusClient, DataType
from tqdm import tqdm

from entity import Tool
import sys
import os
# 将项目根目录添加到系统路径(测试用)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import RemoteEmbeddingModel
from pymilvus.client.types import LoadState

class CustomizeMilvus:
    def __init__(self, uri, db_name):
        """
        CustomizeMilvus 类的初始化，用于初始化 Milvus 数据库连接。
        
        参数:
        - uri: Milvus 数据库的 URI，格式为 "http://{host}:{port}"。
        - db_name: Milvus 数据库的名称。
        """
        databases = db.list_database()
        self.embeddingModel = RemoteEmbeddingModel()
        if db_name not in databases:
            db.create_database(db_name)
        self.client = MilvusClient(uri=uri, db_name=db_name)
        
    def list_collections(self):
        """
        列出数据库中的所有集合
        返回:
            list: 所有集合的名称列表
        """
        return self.client.list_collections()
    
    def has_collection(self, collection_name):
        """
        检查数据库中是否存在指定名称的集合
        参数:
        - collection_name: 集合的名称
        返回:
        - bool: 如果集合存在则返回 True，否则返回 False
        """
        return collection_name in self.list_collections()
    
    def load_collection_into_memory(self, collection_name):
        """
        将指定名称的集合加载到内存中
        参数:
        - collection_name: 集合的名称
        返回:
        - None
        """
        if self.client.get_load_state(collection_name)['state'] == LoadState.NotLoaded:
            self.client.load_collection(collection_name)
        
    def create_collection(self, collection_name):
        """
        创建一个名为 collection_name 的集合
        参数:
        - collection_name: 集合的名称
        返回:
        - None
        """
        schema = MilvusClient.create_schema(enabled_dynamic_field=False)    # 关闭动态字段
        schema.add_field(field_name="tool_id", data_type=DataType.INT64, is_primary=True)
        schema.add_field(field_name="operation_summary", data_type=DataType.VARCHAR, max_length=4096, description="工具名称")
        schema.add_field(field_name="embedding", data_type=DataType.FLOAT_VECTOR, dim=1024, description="工具描述向量")
        
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        self.client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
        
    def insert_embeddings(self, collection_name, embeddings, datas):
        """
        插入嵌入数据
        参数:
        - collection_name: 集合的名称
        - embeddings: 嵌入数据列表
        - datas: 数据列表
        返回:
        - None
        
        分批提交，控制内存
        - 不一次性组装几万条数据再插入，最多缓存 10 条，海量数据不会爆内存。
        进度可视化
        - tqdm 直观看到插入进度，方便排查卡死、数据量问题。
        数据不丢失兜底
        - 循环结束判断剩余缓存，不足一批的数据也会入库。
        """
        self.load_collection_into_memory(collection_name)
        batch_datas = []
        # zip(datas, embeddings)：配对遍历原始业务数据与对应向量，保证顺序一一对应；
        for data, embedding in tqdm(zip(datas, embeddings)):
            data['embedding'] = embedding
            batch_datas.append(data)
            if len(batch_datas) >= 10:  # 每10条数据提交一次
                self.client.insert(collection_name=collection_name, data=batch_datas)
                batch_datas = []
        if batch_datas:  # 剩余的数据提交
            self.client.insert(collection_name=collection_name, data=batch_datas)
            batch_datas = []

    def drop_collection(self, collection_name):
        """
        删除集合
        参数:
        - collection_name: 集合的名称
        返回:
        - None
        """
        self.client.drop_collection(collection_name=collection_name)        

    def embed_chunked_data(self, tools: List[Tool]):
        """
        将工具信息列表进行分块处理，并返回嵌入向量和数据列表
        参数:
        - tools: 工具列表
        返回:
        - embeddings: 嵌入向量列表
        - datas: 数据列表
        """
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
        return embedding, datas


    def insert_tools(self, collection_name, tools: List[Tool]):
        """
        将工具信息列表存入指定的 Milvus 集合中
        参数:
        - collection_name: 集合的名称
        - tools: 工具列表
        返回:
        - None
        """
        if not self.has_collection(collection_name):
            self.create_collection(collection_name)
        embeddings, datas = self.embed_chunked_data(tools)
        self.insert_embeddings(collection_name, embeddings, datas)

    def get_docs(self, collection_name, query_text, top_k=5):
        """
        根据查询文本从指定的 Milvus 集合中获取相关文档的ID
        代码流程逻辑：
            1. 确保目标集合已加载到内存中，便于后续搜索操作；
            2. 对输入的查询语句进行嵌入处理，将其转换为向量表示
            3. 使用转换后的查询向量在 Milvus 集合中进行搜索，设置搜索参数、返回结果数量等；
            4. 解析搜索结果，提取每个匹配文档的工具ID；
            5. 将提取的工具 ID 手机到列表中并返回。
        参数:
        - collection_name: 集合的名称
        - query_text: 查询文本
        - top_k: 返回最相关的文档数量
        返回:
        - list: 包含文档ID的列表
        """
        self.load_collection_into_memory(collection_name)
        query_embedding = self.embeddingModel.get_embedding(query_text)
        """
            search_params 检索核心配置
            metric_type: "COSINE"：使用余弦相似度匹配，和建表、向量归一化逻辑统一
            params: {"level":1}：AUTOINDEX/HNSW 索引专属参数，level 代表检索精度；1 是基础速度优先，数值越大召回越准、速度越慢
        """ 
        response = self.client.search(
            collection_name=collection_name, 
            data=[query_embedding], 
            limit=top_k,           
            search_params={
                "metric_type": "COSINE",
                "params": {"level": 1}
            },
            output_fields=["tool_id", "operation_summary"],
            consistency_level="Bounded" # 事务一致性级别，平衡查询性能与数据实时性，Milvus 常用配置
        )
        docs = []
        for res in response:
            for result in res:
                chunk = result["entity"]
                docs.append(int(chunk["tool_id"]))
        return docs

    def delete_tools(self, tools: List[Tool]):
        """
        删除指定的工具列表
        参数:
        - tools: 工具列表
        返回:
        - None
        """
        ids = []
        for tool in tools:
            ids.append(tool.tool_id)
        self.client.delete(collection_name="tools", ids=ids)

    def get_all_entity(self, ids):
        return self.client.get(collection_name="tools", ids=ids, output_fields=["tool_id", "operation_summary"])







