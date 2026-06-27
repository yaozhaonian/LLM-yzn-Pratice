from utils import logger
import json
import time
from customize_milvus_wrapper import CustomizeMilvus
from entity import Parameter, Tool
from typing import List, Optional
from models import RerankerModel
import os
from mongoengine import *
import threading
from cachetools import TTLCache
from langchain_chroma import Chroma

class ToolManager:
    def __init__(
        self, 
        mongo_host, mongo_db, mongo_port,
        milvus_uri, milvus_db_name,
        texts: Optional[List[str]] = None,  # 这里必须要保留！作为“投递口”
        vectorstore: Optional[Chroma] = None,
        # 下面是 Reranker 的配置参数，暴露给上层
        reranker_persist_dir: str = "./chroma_db",
        llm_model: str = 'qwen2.5:7b',
        base_url: str = "http://127.0.0.1:11434"
    ):
        # 1. 初始化 MongoDB 和 Milvus（你的原有逻辑）
        try:
            # 检查是否已有连接
            existing_connection = get_connection()
            if existing_connection:
                disconnect()  # 断开现有连接
        except Exception as e:
            logger.warning(f"检查ToolManager的已有MongoDB连接: {e}，已重建连接 。")
        self.mongoClient = connect(mongo_db, host=mongo_host, port=mongo_port)
        self.db_name = mongo_db
        self.cache_lock = threading.Lock()
        self.tool_cache = TTLCache(maxsize=100, ttl=3600)
        self.milvus = CustomizeMilvus(milvus_uri, milvus_db_name)
        
        # 2. 核心：根据参数灵活初始化 Reranker
        if vectorstore:
            # 如果直接给的是现成的 vectorstore，直接加载
            self.reranker = RerankerModel(vectorstore=vectorstore)
        elif texts:
            # 如果给的是原始文本，调用工厂方法（新建库）
            self.reranker = RerankerModel.create_from_texts(
                texts=texts,
                persist_directory=reranker_persist_dir,
                llm_model=llm_model,
                base_url=base_url
            )
        else:
            # 如果都没给，从默认磁盘路径加载现有数据
            self.reranker = RerankerModel(
                persist_directory=reranker_persist_dir,
                llm_model=llm_model,
                base_url=base_url
            )

    def clear_cache(self):
        """
        工具缓存清除方法。该方法清除工具缓存中的所有工具。
        """
        with self.cache_lock:
            tool_ids = self.tool_cache.keys()
            for tool_id in tool_ids:
                self.tool_cache.pop(tool_id, None)
           
    def delete_all_tools(self):
        """
        工具删除方法。该方法根据提供的工具 ID，从数据库中删除对应的工具。
        如果工具 ID 为空，则返回 None。
        参数:
            tool_id (int): 工具 ID，用于唯一标识要删除的工具。
        返回:
            None
        """
        Tool.objects.delete()
        self.milvus.drop_collection(collection_name="tools")
        self.clear_cache()
        
    def get_raw_all_tools(self):
        return Tool.objects.all()
