import time
import sys
import os
import json
import subprocess
import traceback
from typing import List, Dict, Any, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import logger


class LargeLanguageModel:
    def __init__(
        self, 
        api_url: str = "http://127.0.0.1:11434", 
        api_key: str = "unused",
        model_name: str = 'qwen2.5:7b',
        temperature: float = 0,
        top_p: float = 1.0
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.api_url = api_url
        logger.info(f"LLM 初始化成功: Model={model_name}, URL={api_url}")

    def _call_ollama_curl(self, messages: List[Dict[str, str]], temperature: float, top_p: float) -> str:
        try:
            data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False
            }
            
            cmd = [
                "curl",
                "-X", "POST",
                f"{self.api_url}/api/chat",
                "-H", "Content-Type: application/json",
                "-d", json.dumps(data, ensure_ascii=False)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                try:
                    response = json.loads(result.stdout)
                    if "message" in response and "content" in response["message"]:
                        return response["message"]["content"]
                    else:
                        logger.error(f"Ollama响应格式错误: {result.stdout[:500]}")
                        return f"Error: 响应格式错误"
                except json.JSONDecodeError:
                    logger.error(f"Ollama响应解析失败: {result.stdout[:500]}")
                    return f"Error: 响应解析失败"
            else:
                logger.error(f"Ollama调用失败: {result.stderr[:500]}")
                return f"Error: 调用失败"
                
        except subprocess.TimeoutExpired:
            logger.error("Ollama调用超时")
            return "Error: 调用超时"
        except Exception as e:
            logger.error(f"Ollama调用异常: {e}\n{traceback.format_exc()}")
            return f"Error: {str(e)}"

    def _convert_messages_to_ollama(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        ollama_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                ollama_messages.append({"role": "system", "content": content})
            elif role == "user":
                ollama_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                ollama_messages.append({"role": "assistant", "content": content})
            else:
                ollama_messages.append({"role": "user", "content": content})
        return ollama_messages

    def context_chat_completions(
        self, 
        contexts: List[Dict[str, str]], 
        model: Optional[str] = None, 
        temperature: Optional[float] = None, 
        top_p: Optional[float] = None, 
        context_number: int = 10
    ) -> str:
        try:
            if len(contexts) > context_number:
                target_contexts = contexts[-context_number:]
            else:
                target_contexts = contexts
            
            ollama_messages = self._convert_messages_to_ollama(target_contexts)
            
            current_temp = temperature if temperature is not None else self.temperature
            current_top_p = top_p if top_p is not None else self.top_p
            
            return self._call_ollama_curl(ollama_messages, current_temp, current_top_p)
            
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
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            
            return self.context_chat_completions(
                contexts=messages,
                model=model,
                temperature=temperature,
                top_p=top_p,
                context_number=10
            )
            
        except Exception as err:
            logger.error(f"与模型 '{self.model_name}' 交互时发生错误: {err}\n{traceback.format_exc()}")
            return f"Error: {str(err)}"

    def backoff(self, wait=None):
        if wait is None:
            time.sleep(1)
        else:
            time.sleep(wait)


if __name__ == "__main__":
    llm = LargeLanguageModel(
        api_url="http://127.0.0.1:11434",
        model_name='qwen2.5:7b',
        temperature=0.7
    )

    print("--- 测试 1: 简单聊天 ---")
    test_prompt = "请介绍一下Python语言"
    try:
        result = llm.chat_completions(test_prompt)
        logger.info("模型返回结果:")
        print(result)
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")

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
