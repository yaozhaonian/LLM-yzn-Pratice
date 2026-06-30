from requests import Response

from entity import Tool
import requests
from utils import logger, RESPONSE_STATUS_CODE_ERROR
import traceback
from tools.tool_manager import ToolManager
sim_api_key = "hihachengfeng"
class ToolUseHub:
    def __init__(self, name):
        self.name = name
        self.retries = 3
        
    def tool_use(self, tool: Tool, requestBody: dict):
        """
        工具调用方法。该方法根据提供的工具对象和请求参数，调用工具的 API 接口。
        如果工具对象的 API 接口为空，则返回 None。
        参数:
            tool (Tool): 工具对象，包含工具的 API 接口、方法、路径等信息。
            requestBody (dict): 请求参数，以字典形式表示。
        返回:
            requests.Response: 工具 API 接口的响应对象。
        """
        # 初始化响应对象
        response = Response()
        response.status_code = RESPONSE_STATUS_CODE_ERROR

        try:
            url = f"{tool.api_url}{tool.path}"
            logger.info(f"准备调用工具[{url}：{requestBody}]....")
            param_request = {}
            for param in tool.request_body:
                if param.name in requestBody:
                    if param.in_ == "path":
                        url = url.replace("{"+param.name+"}", str(requestBody[param.name]))
                        requestBody.pop(param.name)
                    elif param.in_ == "query":
                        param_request[param.name] = requestBody[param.name]
                        requestBody.pop(param.name)

            # 创建包含API密钥的请求头
            headers = {"X-API_Key": sim_api_key}
            # 对于需要JSON请求体的请求，添加content-type头部
            if len(requestBody.keys()) > 0:
                headers["content-type"] = "application/json"

            if len(requestBody.keys()) == 0:
                if len(param_request.keys()) == 0:  # 无请求体 + 无查询参数，示例：GET https://xxx/api/user/1
                    response = requests.request(tool.method.upper(), url, headers=headers)
                else:   # 无请求体 + 有查询参数，示例：GET https://xxx/api/list?page=1&size=10
                    response = requests.request(tool.method.upper(), url,params=param_request, headers=headers)
            else:
                if len(param_request.keys()) == 0:  # 有请求体 + 无查询参数，示例：POST https://xxx/api/add 携带 body {"name":"xxx"}
                    response = requests.request(tool.method.upper(), url, json=requestBody, headers=headers)
                else:   # 有请求体 + 同时带查询参数
                    response = requests.request(tool.method.upper(), url, params=param_request,json=requestBody, headers=headers)

            # 读取接口剩余调用次数
            remaining_calls = response.headers.get('X-Remaining-Calls', 'N/A')
            logger.info(f"API 调用成功，剩余调用次数: {remaining_calls}")

            logger.info(f"API调用返回状态码: {response.status_code}")
        except Exception as e:
            logger.error(f"调用工具[{url}：{requestBody}]失败: {e}\n{traceback.format_exc()}")
            return response

if __name__ == "__main__":
    toolUseHub = ToolUseHub("test")
    toolManager = ToolManager('localhost', "tools", 27017, "http://127.0.0.1:19530", "tool_db")
    tool = toolManager.get_tools_by_ids([15])[0]
    response = toolUseHub.tool_use(tool, {"productId": 2})
    logger.info("接口回复:", response)





