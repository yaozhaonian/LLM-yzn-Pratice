import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apis.api_planning_hub import ApiPlanningHub
from apis.api_selection_hub import ApiSelectionHub
from param_extraction.param_extraction_hub import ParamExtractionHub
from tasks.generate_task_hub import GenerateTaskHub
from tools.tool_summary_hub import ToolSummaryHub
from tools.tool_use_hub import ToolUseHub
from utils import logger

from .graph import build_graph, compile_graph


class LanggraphAgent:
    def __init__(self):
        load_dotenv()
        
        self._load_config()
        self._init_components()
        self._build_graph()
    
    def _load_config(self):
        self.milvus_uri = os.getenv("milvus_uri", "http://localhost:19530")
        self.model_path = os.getenv("model_path", "model")
        self.milvus_db_name = os.getenv("milvus_db_name", "tool_db")
        self.model = os.getenv("model", "qwen2.5:7b")
        self.temperature = float(os.getenv("temperature", "0.01"))
        self.top_p = float(os.getenv("top_p", "0.01"))
        self.mongo_host = os.getenv("mongo_host", "127.0.0.1")
        self.mongo_db = os.getenv("mongo_db", "tools")
        self.mongo_port = int(os.getenv("mongo_port", "27017"))
        self.topK = int(os.getenv("topK", "5"))
        self.api_key = os.getenv("api_key", "")
        self.api_url = os.getenv("base_url", "http://127.0.0.1:11434")
        
        self.mongo_user = os.getenv("mongo_user")
        self.mongo_password = os.getenv("mongo_password")
        self.auth_source = os.getenv("auth_source")
        
        logger.info(f"配置加载完成: model={self.model}, milvus={self.milvus_uri}, mongo={self.mongo_host}:{self.mongo_port}")
    
    def _init_components(self):
        logger.info("初始化 Langgraph Agent 组件...")
        
        mongo_kwargs = {
            "mongo_user": self.mongo_user,
            "mongo_password": self.mongo_password,
            "auth_source": self.auth_source
        }
        
        self.api_planning_hub = ApiPlanningHub(
            self.milvus_uri,
            self.model_path,
            self.milvus_db_name,
            self.model,
            self.temperature,
            self.top_p,
            self.mongo_host,
            self.mongo_db,
            self.mongo_port,
            self.topK,
            self.api_url,
            self.api_key,
            **mongo_kwargs
        )
        
        self.api_selection_hub = ApiSelectionHub(
            self.milvus_uri,
            self.model_path,
            self.milvus_db_name,
            self.model,
            self.temperature,
            self.top_p,
            self.mongo_host,
            self.mongo_db,
            self.mongo_port,
            self.api_url,
            self.api_key,
            **mongo_kwargs
        )
        
        self.param_extraction_hub = ParamExtractionHub(
            self.model,
            self.temperature,
            self.top_p,
            self.api_url,
            self.api_key
        )
        
        self.generate_task_hub = GenerateTaskHub(
            self.model,
            self.temperature,
            self.top_p,
            self.api_url,
            self.api_key,
            self.mongo_host,
            self.mongo_db,
            self.mongo_port,
            self.milvus_uri,
            self.milvus_db_name,
            **mongo_kwargs
        )
        
        self.tool_summary_hub = ToolSummaryHub(
            self.model,
            self.temperature,
            self.top_p,
            self.api_url,
            self.api_key
        )
        
        self.tool_use_hub = ToolUseHub("")
        
        logger.info("Langgraph Agent 组件初始化完成")
    
    def _build_graph(self):
        logger.info("构建 Langgraph 工作流...")
        self.workflow = build_graph()
        self.app = compile_graph(self.workflow)
        logger.info("Langgraph 工作流构建完成")
    
    def run(self, query: str) -> dict:
        logger.info(f"Langgraph Agent 开始执行: {query}")
        
        initial_state = {
            "query": query,
            "task_description": "",
            "api_chain": [],
            "selected_tool": None,
            "params": {},
            "missing_params": [],
            "tool_result": "",
            "is_single_task": False,
            "is_complete": False,
            "loop_count": 0,
            "summary": "",
            "error": None,
            "topK": self.topK,
            "api_planning_hub": self.api_planning_hub,
            "api_selection_hub": self.api_selection_hub,
            "param_extraction_hub": self.param_extraction_hub,
            "generate_task_hub": self.generate_task_hub,
            "tool_summary_hub": self.tool_summary_hub,
            "tool_use_hub": self.tool_use_hub
        }
        
        try:
            result = self.app.invoke(initial_state)
            logger.info(f"Langgraph Agent 执行完成")
            
            return {
                "success": True,
                "summary": result.get("summary", ""),
                "api_chain": result.get("api_chain", []),
                "loop_count": result.get("loop_count", 0),
                "error": result.get("error", None)
            }
        except Exception as e:
            logger.error(f"Langgraph Agent 执行失败: {e}")
            return {
                "success": False,
                "summary": "",
                "api_chain": [],
                "loop_count": 0,
                "error": str(e)
            }


if __name__ == "__main__":
    logger.info("启动 Langgraph Agent 测试...")
    
    agent = LanggraphAgent()
    
    test_query = "查询苹果的产品信息"
    logger.info(f"测试查询: {test_query}")
    
    result = agent.run(test_query)
    logger.info(f"测试结果: {result}")
    
    if result["success"]:
        logger.info(f"总结: {result['summary']}")
        for i, api_call in enumerate(result["api_chain"]):
            logger.info(f"API调用 {i+1}: {api_call['tool']} - {api_call['result'][:200]}")
    else:
        logger.error(f"失败原因: {result['error']}")
