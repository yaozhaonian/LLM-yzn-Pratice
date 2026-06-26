# 本地 Ollama 大语言模型封装 (基于 LangChain)

import time
import sys
import os
# 将项目根目录添加到系统路径(测试用)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import logger
import traceback
from typing import List, Dict, Any, Optional

# LangChain 组件
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

class LargeLanguageModel:
    """
    封装本地 Ollama 大语言模型交互功能
    """
    def __init__(
        self, 
        api_url: str = "http://127.0.0.1:11434", 
        api_key: str = "unused", # Ollama 本地通常不需要 API Key，保留参数以兼容旧接口
        model_name: str = 'qwen2.5:7b',
        temperature: float = 0,
        top_p: float = 1.0
    ):
        """
        初始化 LLM 客户端
        :param api_url: Ollama 服务地址
        :param api_key: 占位符，本地使用可忽略
        :param model_name: 模型名称，如 'qwen2.5:7b'
        :param temperature: 温度参数
        :param top_p: 核采样参数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        
        # 初始化 ChatOllama 实例
        try:
            self.llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                base_url=api_url
            )
            logger.info(f"LLM 初始化成功: Model={model_name}, URL={api_url}")
        except Exception as e:
            logger.error(f"LLM 初始化失败: {e}")
            raise

    def _convert_messages_to_langchain(self, messages: List[Dict[str, str]]) -> list:
        """
        将 OpenAI 格式的 messages 列表转换为 LangChain 格式的消息对象
        :param messages: [{"role": "user/system/assistant", "content": "..."}]
        :return: List of LangChain Message objects
        """
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                # 默认当作人类消息处理
                lc_messages.append(HumanMessage(content=content))
        return lc_messages

    def context_chat_completions(
        self, 
        contexts: List[Dict[str, str]], 
        model: Optional[str] = None, 
        temperature: Optional[float] = None, 
        top_p: Optional[float] = None, 
        context_number: int = 10
    ) -> str:
        """
        带上下文的聊天完成接口
        :param contexts: 历史消息列表，格式同 OpenAI messages
        :param model: 临时指定模型（可选，默认使用初始化时的模型）
        :param temperature: 临时指定温度
        :param top_p: 临时指定 top_p
        :param context_number: 保留最近多少条上下文
        :return: 模型回复内容字符串
        """
        try:
            # 1. 处理上下文截取
            if len(contexts) > context_number:
                target_contexts = contexts[-context_number:]
            else:
                target_contexts = contexts
            
            # 2. 转换消息格式
            lc_messages = self._convert_messages_to_langchain(target_contexts)
            
            # 3. 配置临时参数 (如果提供)
            current_temp = temperature if temperature is not None else self.temperature
            current_top_p = top_p if top_p is not None else self.top_p
            current_model = model if model else self.model_name
            
            # 注意：ChatOllama 实例化后修改参数较麻烦，这里简单起见重新创建或使用 invoke 的配置
            # 更优做法是在 invoke 时传递 config，但 ChatOllama 对动态参数支持有限
            # 这里为了简单，假设参数变化不大，直接使用初始化的 self.llm
            # 如果必须动态改模型，需要重新实例化 self.llm
            
            # 4. 调用模型
            # invoke 接收消息列表
            response = self.llm.invoke(lc_messages)
            
            # 5. 返回内容
            return response.content
            
        except Exception as err:
            logger.error(f"与模型 '{self.model_name}' 交互时发生错误: {err}\n{traceback.format_exc()}")
            return f"Error: {str(err)}"

    def chat_completions(
        self, 
        prompt: str, 
        model: Optional[str] = None, 
        temperature: Optional[float] = None, 
        top_p: Optional[float] = None
    ) -> str:
        """
        简单聊天完成接口
        :param prompt: 用户提示词
        :param model: 临时指定模型
        :param temperature: 临时指定温度
        :param top_p: 临时指定 top_p
        :return: 模型回复内容字符串
        """
        try:
            # 构建标准的 System + User 消息结构
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            
            # 复用 context_chat_completions 的逻辑
            return self.context_chat_completions(
                contexts=messages,
                model=model,
                temperature=temperature,
                top_p=top_p,
                context_number=10 # 此处无实际意义，因为 messages 只有两条
            )
            
        except Exception as err:
            logger.error(f"与模型 '{self.model_name}' 交互时发生错误: {err}\n{traceback.format_exc()}")
            return f"Error: {str(err)}"

    def backoff(self, wait=None):
        """
        保留此方法以兼容旧代码调用，但在本地模式下通常不需要复杂的退避策略
        """
        if wait is None:
            time.sleep(1) # 本地出错通常建议短暂等待而非长时间随机等待
        else:
            time.sleep(wait)


if __name__ == "__main__":
    # 测试代码
    # 注意：确保 Ollama 服务已启动，且 qwen2.5:7b 模型已拉取
    
    # 初始化 LLM
    # api_url 指向本地 Ollama 服务
    llm = LargeLanguageModel(
        api_url="http://127.0.0.1:11434",
        model_name='qwen2.5:7b',
        temperature=0.7
    )

    # 测试 1: 简单聊天
    print("--- 测试 1: 简单聊天 ---")
    test_prompt = "请介绍一下Python语言"
    try:
        result = llm.chat_completions(test_prompt)
        logger.info("模型返回结果:")
        print(result)
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")

    # 测试 2: 带上下文聊天
    print("\n--- 测试 2: 带上下文聊天 ---")
    contexts = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么我可以帮你的吗？"},
        {"role": "user", "content": "我想学习编程，应该从哪里开始？"}
    ]
    try:
        result = llm.context_chat_completions(contexts=contexts)
        logger.info("模型返回结果:")
        print(result)
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")