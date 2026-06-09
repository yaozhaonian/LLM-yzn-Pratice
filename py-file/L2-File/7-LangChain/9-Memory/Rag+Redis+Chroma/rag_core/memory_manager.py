from typing import List, Dict, Optional
import json
import redis
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, CACHE_EXPIRE_SECONDS

class MemoryManager:
    """Redis 对话记忆管理"""
    
    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT, db: int = REDIS_DB):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_timeout=5
        )
        self.key_prefix = "rag:chat:"
        self.expire_seconds = CACHE_EXPIRE_SECONDS
    
    def _get_key(self, user_id: str, knowledge_name: str) -> str:
        """生成 Redis Key"""
        return f"{self.key_prefix}{knowledge_name}:{user_id}"
    
    def get_history(self, user_id: str, knowledge_name: str) -> List[Dict]:
        """获取对话历史"""
        key = self._get_key(user_id, knowledge_name)
        data = self.redis_client.get(key)
        return json.loads(data) if data else []
    
    def append_message(self, user_id: str, knowledge_name: str, role: str, content: str):
        """追加消息"""
        history = self.get_history(user_id, knowledge_name)
        history.append({"role": role, "content": content})
        
        # 限制历史长度（保留最近 20 轮）
        if len(history) > 40:  # 用户+AI 算 2 条
            history = history[-40:]
        
        key = self._get_key(user_id, knowledge_name)
        self.redis_client.setex(
            key,
            self.expire_seconds,
            json.dumps(history, ensure_ascii=False)
        )
    
    def clear_history(self, user_id: str, knowledge_name: str):
        """清空历史"""
        key = self._get_key(user_id, knowledge_name)
        self.redis_client.delete(key)
    
    def has_history(self, user_id: str, knowledge_name: str) -> bool:
        """检查是否有历史"""
        return len(self.get_history(user_id, knowledge_name)) > 0
    
    def list_all_sessions(self) -> List[str]:
        """列出所有会话"""
        keys = self.redis_client.keys(f"{self.key_prefix}*")
        return [key.replace(self.key_prefix, "") for key in keys]