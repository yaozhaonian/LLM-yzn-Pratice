import os
import sys
from utils.config import mongo_user as mongo_user, mongo_password, auth_source
if __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from customize_milvus_wrapper import CustomizeMilvus
from models import LargeLanguageModel
from prompt import create_prompt_hub
from tools.tool_manager import ToolManager
from utils import logger
import traceback

class ApiSelectionHub:
    def __init__(self, milvus_uri, model_path, milvus_db_name, model, temperature, top_p,
            mongo_host, mongo_db, mongo_port, api_url, api_key,
            mongo_user=None, mongo_password=None, auth_source=None):
        self.ToolManager = ToolManager(
            mongo_host,
            mongo_db,
            mongo_port,
            milvus_uri,
            milvus_db_name,
            mongo_user=mongo_user,
            mongo_password=mongo_password,
            auth_source=auth_source,
        )       
        self.milvus = CustomizeMilvus(milvus_uri, milvus_db_name)
        self.LargeLanguageModel = LargeLanguageModel(api_url, api_key)
        self.PromptModelHub = create_prompt_hub(model)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    def _fallback_keyword_match(self, query, topK):
        all_tools = self.ToolManager.get_all_tools()
        if not all_tools:
            return []
        
        query_lower = query.lower()
        scored_tools = []
        
        for tool in all_tools:
            score = 0
            search_text = f"{tool.name_for_human} {tool.description} {tool.operationId}".lower()
            
            keywords = [
                ("产品", 2), ("商品", 2), ("订单", 2), ("供应商", 2), ("物流", 3), ("物流公司", 3),
                ("库存", 2), ("价格", 2), ("配送", 3), ("区域", 2), ("发货", 3), 
                ("下单", 3), ("购买", 2), ("订购", 2), ("快递", 2), ("运输", 2), ("运送", 2),
                ("荔枝", 3), ("苹果", 3), ("香蕉", 3), ("橙子", 3), ("水果", 2),
                ("北京", 2), ("上海", 2), ("广东", 2), ("城市", 2), ("地址", 2), ("收货", 2),
                ("创建", 2), ("生成", 2), ("新增", 2), ("京东", 3), ("顺丰", 3), ("邮政", 3)
            ]
            for kw, weight in keywords:
                if kw in query_lower and kw in search_text:
                    score += weight
            
            for char in query_lower:
                if char in search_text:
                    score += 1
            
            scored_tools.append((tool, score))
        
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, score in scored_tools[:topK * 2] if score > 0]

    def get_tool_coarse_and_fine(self, query, required_argument, topK):
        try:
            try:
                tool_ids = self.milvus.get_docs("tools", query, topK * 2)
                vector_search_tools = self.ToolManager.get_tools_by_ids(tool_ids)
                logger.info(f"[向量检索] 查询: {query}, 候选工具数量: {len(vector_search_tools)}")
            except Exception as e:
                logger.warning(f"[向量检索失败，使用关键词匹配降级] {e}")
                vector_search_tools = self._fallback_keyword_match(query, topK)
                logger.info(f"[关键词匹配] 查询: {query}, 候选工具数量: {len(vector_search_tools)}")
            
            if not vector_search_tools:
                logger.warning(f"[工具选择] 没有找到候选工具，使用所有工具")
                vector_search_tools = self.ToolManager.get_all_tools()[:topK * 2]
            
            for i, tool in enumerate(vector_search_tools[:5]):
                logger.debug(f"[候选工具] {i+1}: [{tool.tool_id}] {tool.operationId}-{tool.name_for_human}")
            
            try:
                reranked_tools = self.ToolManager.search_tools_with_rerank(query, top_k=topK * 2, final_top_n=topK)
                logger.info(f"[重排序] 重排序后工具数量: {len(reranked_tools)}")
            except Exception as e:
                logger.warning(f"[重排序失败，使用向量检索结果] {e}")
                reranked_tools = None
            
            tools = reranked_tools if reranked_tools else vector_search_tools
            
            if required_argument is None:
                prompt = self.PromptModelHub.gen_tool_selection_prompt(query, tools)
            else:
                prompt = self.PromptModelHub.gen_required_argument_tool_selection_prompt(query, required_argument, tools)
            
            model_output = self.LargeLanguageModel.chat_completions(prompt, self.model, self.temperature, self.top_p)
            final_tool = self.PromptModelHub.post_process_tool_selection_result(model_output, tools)
            
            if final_tool:
                logger.info(f"查询: [{query}]被选择工具 : [{final_tool.tool_id}] {final_tool.operationId}-{final_tool.name_for_human}")
            else:
                logger.warning(f"查询: [{query}]未选择到工具")
            
            return final_tool
            
        except Exception as e:
            logger.error(f"检索并选择工具[{query}：{required_argument}]失败: {e}\n{traceback.format_exc()}")
            return None

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    milvus_uri = os.getenv("milvus_uri", "http://localhost:19530")
    logger.info(milvus_uri)
    model_path = os.getenv("model_path", "model")
    milvus_db_name = os.getenv("milvus_db_name", "tool_db")
    model = os.getenv("model", "qwen2.5:7b")
    temperature = float(os.getenv("temperature", "0.01"))
    top_p = float(os.getenv("top_p", "0.01"))
    mongo_host = os.getenv("mongo_host", "127.0.0.1")
    mongo_db = os.getenv("mongo_db", "tools")
    mongo_port = int(os.getenv("mongo_port", "27017"))
    topK = int(os.getenv("topK", "5"))
    api_key = os.getenv("api_key", "")
    base_url = os.getenv("base_url", "")
    apiSelectionHub = ApiSelectionHub(milvus_uri, model_path, milvus_db_name, model, temperature, top_p,
                                      mongo_host, mongo_db, mongo_port, base_url,api_key)
    tool = apiSelectionHub.get_tool_coarse_and_fine("查询苹果产品信息", None, topK)
