import json
import subprocess
from typing import Generator
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class OllamaClientError(Exception):
    """Ollama客户端调用异常基类"""
    pass


class OllamaConnectionError(OllamaClientError):
    """Ollama服务连接异常"""
    pass


class OllamaModelNotFoundError(OllamaClientError):
    """模型不存在异常"""
    pass


class OllamaTimeoutError(OllamaClientError):
    """调用超时异常"""
    pass


class OllamaClient:
    """
    Ollama本地大模型调用客户端
    
    封装Ollama API调用，提供同步单轮对话和流式输出两种调用方式，
    统一处理连接超时、模型不存在、服务未启动等异常。
    """

    def __init__(self):
        """初始化Ollama客户端"""
        self._host = settings.ollama.host.replace("http://", "").replace("https://", "")
        self._port = settings.ollama.port
        self._model = settings.ollama.model
        self._timeout = settings.ollama.timeout
        logger.info(f"Ollama客户端初始化完成，服务地址: {self._host}:{self._port}, 模型: {self._model}")

    def _build_url(self) -> str:
        """构建完整的API地址"""
        return f"http://{self._host}:{self._port}/api/chat"

    def chat(self, prompt: str, system_prompt: str = "") -> str:
        """
        同步单轮对话调用
        
        向Ollama模型发送提示词，等待完整响应后返回纯文本回答。
        
        Args:
            prompt: 用户输入的提示词
            system_prompt: 系统提示词，用于设定模型行为
        
        Returns:
            str: 模型返回的纯文本回答
        
        Raises:
            OllamaConnectionError: Ollama服务连接失败
            OllamaModelNotFoundError: 指定的模型不存在
            OllamaTimeoutError: 调用超时
            OllamaClientError: 其他调用异常
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            data = {
                "model": self._model,
                "messages": messages,
                "stream": False
            }

            logger.info(f"开始调用Ollama模型，提示词长度: {len(prompt)}")
            
            result = subprocess.run(
                ["curl", "-s", "-X", "POST", self._build_url(),
                 "-H", "Content-Type: application/json",
                 "-d", json.dumps(data)],
                capture_output=True,
                text=True,
                timeout=self._timeout
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                if "connection refused" in error_msg.lower() or "can't connect" in error_msg.lower():
                    logger.error(f"Ollama服务连接失败: {error_msg}")
                    raise OllamaConnectionError(f"Ollama服务连接失败，请检查服务是否启动")
                logger.error(f"Ollama调用失败: {error_msg}")
                raise OllamaClientError(f"Ollama调用失败: {error_msg}")

            try:
                response = json.loads(result.stdout)
                answer = response.get("message", {}).get("content", "")
            except json.JSONDecodeError:
                logger.error(f"Ollama响应解析失败: {result.stdout[:200]}")
                raise OllamaClientError(f"Ollama响应解析失败")

            logger.info(f"Ollama调用完成，响应长度: {len(answer)}")
            return answer

        except subprocess.TimeoutExpired:
            logger.error(f"Ollama调用超时: {self._timeout}秒")
            raise OllamaTimeoutError(f"调用超时，超过 {self._timeout} 秒")
        except OllamaConnectionError:
            raise
        except OllamaClientError:
            raise
        except Exception as e:
            logger.error(f"Ollama调用异常: {str(e)}")
            raise OllamaClientError(f"Ollama调用异常: {str(e)}")

    def stream_chat(self, prompt: str, system_prompt: str = "") -> Generator[str, None, None]:
        """
        流式输出对话结果
        
        向Ollama模型发送提示词，以流式方式返回回答片段。
        
        Args:
            prompt: 用户输入的提示词
            system_prompt: 系统提示词，用于设定模型行为
        
        Yields:
            str: 模型返回的文本片段
        
        Raises:
            OllamaConnectionError: Ollama服务连接失败
            OllamaModelNotFoundError: 指定的模型不存在
            OllamaTimeoutError: 调用超时
            OllamaClientError: 其他调用异常
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            data = {
                "model": self._model,
                "messages": messages,
                "stream": True
            }

            logger.info(f"开始流式调用Ollama模型，提示词长度: {len(prompt)}")
            
            process = subprocess.Popen(
                ["curl", "-s", "-X", "POST", self._build_url(),
                 "-H", "Content-Type: application/json",
                 "-d", json.dumps(data)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            for line in process.stdout:
                line = line.strip()
                if line:
                    try:
                        response = json.loads(line)
                        content = response.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if response.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

            process.wait(timeout=self._timeout)
            
            if process.returncode != 0:
                error_msg = process.stderr.read() if process.stderr else ""
                logger.error(f"Ollama流式调用失败: {error_msg}")
                raise OllamaClientError(f"Ollama流式调用失败")

            logger.info("Ollama流式调用完成")

        except subprocess.TimeoutExpired:
            logger.error(f"Ollama流式调用超时: {self._timeout}秒")
            raise OllamaTimeoutError(f"调用超时，超过 {self._timeout} 秒")
        except Exception as e:
            logger.error(f"Ollama流式调用异常: {str(e)}")
            raise OllamaClientError(f"Ollama流式调用异常: {str(e)}")


ollama_client = OllamaClient()
