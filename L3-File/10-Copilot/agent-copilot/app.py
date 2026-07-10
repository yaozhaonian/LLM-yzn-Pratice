import time
import json
from flask_cors import CORS
from flask import Flask, request, jsonify, g
from flasgger import Swagger
from user_manager.user_manager import UserManagerHub
from apis.api_planning_hub import ApiPlanningHub
from entity import Tool, Parameter
from tasks import GenerateTaskHub, TaskManager
from models import LargeLanguageModel
from tools.tool_manager import ToolManager
from utils import logger, RESPONSE_AUTH_CODE_ERROR, RESPONSE_ALLOW_CODE_ERROR, RESPONSE_STATUS_CODE_ERROR, RESPONSE_STATUS_CODE_SUCCESS, DEFAULT_PERMISSIONS
import traceback
import os
from concurrent.futures import ThreadPoolExecutor
from utils.config import (
    milvus_uri,
    model_path,
    milvus_db_name,
    model_name,
    model_temperature,
    model_top_p,
    mongo_host,
    mongo_db,
    mongo_port,
    topK,
    model_api_key,
    model_base_url, SECRET_KEY, JWT_ALGORITHM,
    mongo_user, mongo_password, auth_source
)
import jwt  # 前后端无状态身份认证标准，用于登录后下发凭证，客户端后续请求携带令牌，服务端校验身份，无需服务端存储会话（对比 Session）。
from datetime import datetime, timedelta
import uuid
from cachetools import TTLCache     # 内存缓存工具库，TTLCache（带过期时间缓存，高频业务）
from functools import wraps
from werkzeug.utils import secure_filename

# JWT配置
JWT_EXPIRATION_DELTA = timedelta(hours=3)

# 创建TTL缓存，存储用户会话信息，最大1000个，有效期3小时
session_cache = TTLCache(maxsize=1000, ttl=3 * 60 * 60)

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
# 获取CPU核心数并计算线程池大小
cpu_count = os.cpu_count() or 1
max_workers = max(2, cpu_count * 2)  # 至少2个线程，最多为CPU核心数的两倍
executor = ThreadPoolExecutor(max_workers=max_workers)
tasks = {}  # 存储任务的字典，键为任务ID，值为任务对象

CORS(
    app,
    resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}},
    supports_credentials=True,
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
    expose_headers=["Content-Type", "Authorization"],
)

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin in ("http://localhost:3000", "http://127.0.0.1:3000"):
        response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,Accept"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

Swagger(app=app)

toolManager = ToolManager(
    mongo_host,
    mongo_db,
    mongo_port,
    milvus_uri,
    milvus_db_name,
    mongo_user=mongo_user,
    mongo_password=mongo_password,
    auth_source=auth_source,
)

taskManager = TaskManager(
    mongo_host,
    mongo_db,
    mongo_port,
    mongo_user=mongo_user,
    mongo_password=mongo_password,
    auth_source=auth_source,
)

userManagerHub = UserManagerHub(
    mongo_host,
    mongo_db,
    mongo_port,
    mongo_user,
    mongo_password,
    auth_source=auth_source,
)

PUBLIC_ENDPOINTS = {
    'login', 'register',
    'get_all_tools', 'insert_tool', 'delete_all_tool',
    'api_planning', 'api_task_status'
}

@app.errorhandler(Exception)
def handle_all_exceptions(e):
    logger.error(f"Unhandled exception: {e}\n{traceback.format_exc()}")
    resp = jsonify({'message': '服务器内部错误', 'detail': str(e)})
    resp.status_code = 500
    origin = request.headers.get("Origin")
    if origin in ("http://localhost:3000", "http://127.0.0.1:3000"):
        resp.headers["Access-Control-Allow-Origin"] = origin
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,Accept"
    resp.headers["Access-Control-Allow-Credentials"] = "true"
    return resp

# 权限验证装饰器
# 

def require_permissions(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        return f(*args, **kwargs)
    return decorated_function

@app.route('/delete_all_tool', methods=['POST', 'GET'])
@require_permissions
def delete_all_tool():
    """
    删除所有工具
    ---
    tags:
      - Tool Delete
    description: 上传文件到服务器
    responses:
      200:
        description: 数据库清空成功
      400:
        description: 数据库清空失败
    """
    toolManager.delete_all_tools()
    return jsonify({"status": 200, "message": "数据库清空成功", "data": None}), RESPONSE_STATUS_CODE_SUCCESS

@app.route('/insert_tool', methods=['POST'])
@require_permissions
def insert_tool():
    """
    工具插入
    ---
    tags:
      - Tool Management
    description:
        插入一个新的工具到数据库中
    parameters: 参数
      - name: body
        in: body
        required: true
        schema:
          id: Tool Insert Request
          required:
            - operationId
            - name_for_human
            - name_for_model
            - description
            - url
            - path
            - method
            - params
          properties:
            operationId:
              type: string
              description: 操作ID
            name_for_human:
              type: string
              description: 人类可读的工具名称
            name_for_model:
              type: string
              description: 模型使用的工具名称
            description:
              type: string
              description: 工具的描述
            url:
              type: string
              description: API 的 URL
            path:
              type: string
              description: API 的路径
            method:
              type: string
              description: HTTP 方法 (如 GET, POST 等)
            params:
              type: array
              items:
                type: object
                properties:
                  param_name:
                    type: string
                    description: 参数名称
                  paramType:
                    type: string
                    description: 参数类型
                  param_description:
                    type: string
                    description: 参数描述
                  enum:
                    type: array
                    items:
                      type: string
                    description: 参数的枚举值
                  in_:
                    type: string
                    description: 参数的位置 (如 query, body 等)
    responses:
      200:
        description: 工具插入成功
        schema:
          type: object
          properties:
            message:
              type: string
      400:
        description: 请求参数缺失或无效
        schema:
          type: object
          properties:
            error:
              type: string
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求体为空"}), 400

    params = []
    for tmp in data["params"]:
        parameter = Parameter(
            name=tmp["param_name"],
            type=tmp["paramType"],
            description=tmp["param_description"],
            enum=tmp["enum"],
            required=True,
            in_=tmp["in_"]
        )
        params.append(parameter)

    # 创建 Tool 对象
    tool = Tool(
        tool_id=0,
        operationId=data["operationId"],
        name_for_human=data["name_for_human"],
        name_for_model=data["name_for_model"],
        description=data["description"],
        url=data["url"],
        path=data["path"],
        method=data["method"],
        request_body=params
    )

    # 插入工具到数据库
    toolManager.insert_tools([tool])
    return jsonify({"status": 200, "message": "工具插入成功", "data": None}), RESPONSE_STATUS_CODE_SUCCESS

@app.route('/register_user', methods=['POST'])
def register():
    """
    用户注册
    ---
    tags:
      - User Management
    description:
        注册新用户
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: User Registration Request
          required:
            - userName
            - password
            - confirm_password
          properties:
            userName:
              type: string
              description: 用户名
            password:
              type: string
              description: 密码
            confirm_password:
              type: string
              description: 确认密码
    responses:
      200:
        description: 用户注册成功
        schema:
          type: object
          properties:
            message:
              type: string
              example: "create user success!"
      400:
        description: 用户注册失败
        schema:
          type: object
          properties:
            message:
              type: string
              example: "create user failed!"
      409:
        description: 用户名已存在
        schema:
          type: object
          properties:
            message:
              type: string
              example: "该用户已注册"
    """
    data = request.get_json()
    if not data:
        return jsonify({"message": "请求体为空"}), 400

    required_fields = ["username", "password", "confirm"]
    if not all(field in data for field in required_fields):
        return jsonify({"message": "缺少必要的注册信息：username, password, confirm"}), 400

    username = data.get("username")
    password = data.get("password")
    confirm_password = data.get("confirm")
    status_code, message = userManagerHub.create_user(username, password, confirm_password)
    if status_code == RESPONSE_STATUS_CODE_SUCCESS:
        logger.info(f"用户注册成功: {username}")
        return jsonify({"status": status_code, "message": '成功创建用户'}), RESPONSE_STATUS_CODE_SUCCESS
    elif status_code == 409:
        logger.warning(f"尝试注册已存在的用户: {username}")
        return jsonify({'message': '该用户已注册'}), 409
    else:
        logger.error(f"创建用户失败: {username}, 原因: {message}")
        return jsonify({'message': message}), status_code

@app.route('/login_user', methods=['POST'])
def login():
    """
    用户登录
    ---
    tags:
      - User Management
    description:
        用户登录接口
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: User Login Request
          required:
            - userName
            - password
          properties:
            userName:
              type: string
              description: 用户名
            password:
              type: string
              description: 密码
    responses:
      200:
        description: 用户登录成功
        schema:
          type: object
          properties:
            status:
              type: integer
              example: 200
            message:
              type: string
              example: "登录成功"
            data:
              type: object
              properties:
                token:
                  type: object
                  properties:
                    access_token:
                      type: string
                      example: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
                    expires_in:
                      type: integer
                      example: 3600
                    token_type:
                      type: string
                      example: "Bearer"
      400:
        description: 用户登录失败
        schema:
          type: object
          properties:
            message:
              type: string
              example: "登录失败！请检查用户名和密码"
    """
    try:
        data = request.get_json()
        logger.debug(f"/login_user request json: {data}")
        if not data:
            return jsonify({"message": "请求体为空"}), 400

        username = data.get("username")
        password = data.get("password")
        if not username or not password:
            return jsonify({"message": "缺少 username 或 password"}), 400

        # 业务调用（捕获并返回异常详情用于调试）
        try:
            user = userManagerHub.login(username, password)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"userManagerHub.login 抛出异常: {e}\n{tb}")
            return jsonify({"message": "内部服务错误（login）", "traceback": tb}), 500

        if user is None:
            logger.warning(f"login 返回 None for user: {username}")
            return jsonify({"status": RESPONSE_AUTH_CODE_ERROR, "message": "登录失败！请检查用户名和密码", "data": None}), RESPONSE_AUTH_CODE_ERROR

        if getattr(user, "user_id", -1) != -1:
            payload = {
                'user_id': user.user_id,
                'username': user.username,
                'user_authority': user.user_authority,
                'exp': datetime.utcnow() + JWT_EXPIRATION_DELTA
            }
            token = jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)
            session_cache[token] = {
                'user_id': user.user_id,
                'username': user.username,
                'user_authority': user.user_authority
            }
            return jsonify({
                "status": RESPONSE_STATUS_CODE_SUCCESS,
                "message": "登录成功",
                "data": {
                    "token": {
                        "access_token": token,
                        "expires_in": JWT_EXPIRATION_DELTA.total_seconds(),
                        "token_type": "Bearer"
                    }
                }
            }), RESPONSE_STATUS_CODE_SUCCESS
        else:
            logger.warning(f"用户登录失败: {username}")
            return jsonify({"status": RESPONSE_AUTH_CODE_ERROR, "message": "登录失败！请检查用户名和密码", "data": None}), RESPONSE_AUTH_CODE_ERROR

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"login 接口未处理异常: {e}\n{tb}")
        return jsonify({"message": "服务器内部错误", "traceback": tb}), 500
    

@app.route('/logout_user', methods=['POST'])
def logout():
    """
    登出用户
    ---
    tags:
      - user management
    description:
      用户登出接口，通过 POST 方法接收用户 ID 并执行登出操作。
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: logout_body
          required:
            - user_id
          properties:
            user_id:
              type: integer
              format: int32
              description: 需要登出的用户 ID。
    responses:
      200:
        description: 登出成功
        schema:
          type: object
          properties:
            message:
              type: string
              example: logout success!
      400:
        description: 登出失败或缺少参数
        schema:
          type: object
          properties:
            error:
              type: string
              example: logout failed! 或 Missing query parameter
    """
    # 从Authorization头获取令牌
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "无效的令牌格式"}), 400
    token = auth_header.split(" ")[1]
    for token in session_cache:
        del session_cache[token]

    return jsonify({"status": RESPONSE_STATUS_CODE_SUCCESS, "message": "退出登录成功", "data": None}), RESPONSE_STATUS_CODE_SUCCESS

@app.route('/delete_tool_by_ids', methods=['POST'])
@require_permissions
def delete_tool_by_ids():
    """
    特定id工具数据库删除
    ---
    tags:
    - Tool Delete By Ids
    description:
        根据提供的ID列表删除工具
    parameters:
    - name: body
        in: body
        required: true
        schema:
        id: Tool Delete Request
        required:
            - ids
        properties:
            ids:
            type: array
            items:
                type: integer
            description: 需要删除的工具ID列表
    responses:
    200:
        description: 数据库删除成功
        schema:
        type: object
        properties:
            message:
            type: string
    400:
        description: 数据库未删除成功
        schema:
        type: object
        properties:
            error:
            type: string
    """
    data = request.get_json()
    if not data or 'ids' not in data:
        return jsonify({'error': 'Missing query parameter'}), 400

    ids = data['ids']
    toolManager.delete_tools(ids)
    logger.info(f"工具删除成功")
    return jsonify({'message': '工具删除成功! '}), RESPONSE_STATUS_CODE_SUCCESS

@app.route('/upload_file', methods=['POST', 'OPTIONS'])
@require_permissions
def upload_file():
    """
        文件上传接口
        ---
        tags:
          - File Upload
        description:
            上传文件到服务器
        parameters:
          - name: file
            in: formData
            type: file
            required: true
            description: 要上传的文件
        responses:
          200:
            description: 文件上传成功，重定向到主页
          400:
            description: 未上传文件或文件名为空
    """
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有文件部分'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件或者文件名为空'}), 400

        upload_folder = 'uploads'
        os.makedirs(upload_folder, exist_ok=True)

        safe_name = secure_filename(file.filename)
        suffix = safe_name.split('.')[-1]
        unique_name = f"{uuid.uuid4().hex}.{suffix}"
        file_path = os.path.join(upload_folder, unique_name)
        file.save(file_path)

        toolManager.upload_file(file_path)

        logger.info(f"文件上传成功: {file.filename}")
        return jsonify({'message': f'{file.filename} 文件上传成功!'}), RESPONSE_STATUS_CODE_SUCCESS

    except Exception as e:
        logger.error(f"/upload_file error: {e}\n{traceback.format_exc()}")
        resp = jsonify({'message': 'upload_file 内部错误', 'detail': str(e)})
        resp.status_code = 500
        return resp

@app.route('/get_all_tools', methods=['GET'])
@require_permissions
def get_all_tools():
    """
            获取所有工具接口
            ---
            tags:
              - Tools
            description:
                获取服务器上所有工具的列表
            responses:
              200:
                description: 成功获取工具列表
              500:
                description: 服务器内部错误
        """
    datas = toolManager.get_all_tools()
    logger.debug(f"Output: {datas}")
    return jsonify({"results": datas}), RESPONSE_STATUS_CODE_SUCCESS

@app.route('/api_planning', methods=['POST'])
@require_permissions
def mesh_query():
    """
    API规划接口
    ---
    tags:
      - API Planning
    description:
        根据用户输入的查询语句，规划并返回相关的API调用结果。
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: API Planning Request
          required:
            - query
        properties:
            modelName:
                type: string
                description: 模型名称
                required: true
            temperature:
                type: number
                description: 温度参数
                required: true
            api_key:
                type: string
                description: API 密钥
                required: true
            api_url:
                type: string
                description: API 地址
                required: true
            query:
                type: string
                description: 用户需求.
            use_langgraph:
                type: boolean
                description: 是否使用 Langgraph 框架
                required: false
                default: false
    responses:
        200:
            description: 转化成功
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Missing query parameter'}), 400

    use_langgraph = data.get('use_langgraph', True)
    
    if use_langgraph:
        task_id = taskManager.create_task()
        executor.submit(_process_task_langgraph, task_id, data)
        return jsonify({"task_id": task_id, "message": "Langgraph任务已创建，正在处理中"}), RESPONSE_STATUS_CODE_SUCCESS
    else:
        task_id = taskManager.create_task()
        executor.submit(_process_task, task_id, data)   # 将任务提交到线程池
        return jsonify({"task_id": task_id, "message": "任务已创建，正在处理中"}), RESPONSE_STATUS_CODE_SUCCESS

def _process_task(task_id, data):
    logger.info(f"准备处理任务{task_id}，任务数据：{data}，处理中......")
    try:
        query = data.get("query", "")
        contexts = data.get("contexts", [])
        isCopilot = data.get("isCopilot", True)
        isContext = data.get("isContext", False)
        contextNumber = data.get("contextNumber", 3)

        curr_model_name = model_name
        curr_temperature = model_temperature
        curr_api_key = model_api_key
        curr_api_url = model_base_url
    except Exception as e:
        taskManager.update_task(task_id, [], [], '获取前端参数失败！', "", True)
        logger.error(f"任务[{task_id}]获取前端参数失败: {e}\n{traceback.format_exc()}")
        return jsonify({'error': '获取前端参数失败！'}), 400
    
    if not isCopilot:
        llm = LargeLanguageModel(curr_api_url, curr_api_key)
        try:
            if isContext:
                results = llm.context_chat_completions(contexts, curr_model_name, curr_temperature, model_top_p, contextNumber)
            else:
                results = llm.chat_completions(query, curr_model_name, curr_temperature,model_top_p)
        except:
            results = ""
        if results is not None and len(results) != 0:
            taskManager.update_task(task_id, [], [], results, "", True)
            return jsonify({"nodes": [], "edges": [], "systemOutput": results})
        else:
            taskManager.update_task(task_id, [], [], results, "", True)
            return jsonify({"nodes": [], "edges": [], "systemOutput": results}), 400
    else:
        logger.info(f"[{task_id}]任务启动成功,请继续 ===>")
    
    try:
        api_planning_hub = ApiPlanningHub(
            milvus_uri, model_path, milvus_db_name, curr_model_name,
            curr_temperature, model_top_p, mongo_host, mongo_db, mongo_port, topK,
            curr_api_url, curr_api_key,
            mongo_user=mongo_user, mongo_password=mongo_password, auth_source=auth_source
        )
        generate_task_hub = GenerateTaskHub(
            curr_model_name, curr_temperature, model_top_p,
            curr_api_url, curr_api_key, mongo_host, mongo_db, mongo_port,
            milvus_uri, milvus_db_name,
            mongo_user=mongo_user, mongo_password=mongo_password, auth_source=auth_source
        )
        if isContext:
            if len(contexts) < contextNumber:
                target_contexts = contexts
            else:
                target_contexts = contexts[len(contexts) - contextNumber:len(contexts)]
            target_query = generate_task_hub.gen_context_request_task(target_contexts)
        else:
            target_query = query
        api_planning_hub.apis_planning(target_query, task_id)
    except Exception as e:
      error_msg = f"任务处理失败: {str(e)}"
      logger.error(f"任务[{task_id}]处理失败: {e}\n{traceback.format_exc()}")
      detailed_error = (
            "❌ 任务处理失败\\n\\n"
            "可能原因:\\n"
            "1. 工具库为空 - 请先调用 /init_default_tools 初始化工具\\n"
            "2. 查询不匹配 - 请确保查询内容与已注册工具相关\\n"
            "3. 工具描述不清 - 建议检查工具的 description 字段\\n"
            "4. API服务未启动 - 请确保 mock_api_server.py 运行在 http://localhost:8080\\n\\n"
            f"详细错误: {error_msg}"
        )
        
      taskManager.update_task(
            task_id,
            [],
            [],
            detailed_error,
            "",
            True
      )

def _process_task_langgraph(task_id, data):
    logger.info(f"[Langgraph] 准备处理任务{task_id}，任务数据：{data}，处理中......")
    try:
        query = data.get("query", "")
        
        from langgraph_agent.main import LanggraphAgent
        
        agent = LanggraphAgent()
        result = agent.run(query)
        
        nodes = []
        edges = []
        is_success = "正常调用链路"
        
        if result["success"]:
            for i, api_call in enumerate(result["api_chain"]):
                nodes.append({
                    "id": str(i + 1),
                    "name": str(i + 1) + "_" + api_call["tool"],
                    "label": api_call["tool"],
                    "group": api_call["tool"],
                    "task_description": api_call["task_description"],
                    "params": json.dumps(api_call["param"], ensure_ascii=False),
                    "result": json.dumps(api_call["result"], ensure_ascii=False),
                })
            
            if len(nodes) > 1:
                for i in range(len(nodes) - 1):
                    edges.append({
                        "source": nodes[i]["id"],
                        "target": nodes[i + 1]["id"],
                        "value": '正向API规划',
                        "symbolSize": [5, 20],
                        "label": {"show": False}
                    })
            
            system_output = result["summary"]
            logger.info(f"[Langgraph] 任务[{task_id}]处理成功: {system_output[:200]}")
        else:
            is_success = "异常调用链"
            nodes.append({
                "id": "1",
                "name": "1_异常调用节点-" + result.get("error", "未知错误"),
                "label": "异常调用节点-" + result.get("error", "未知错误"),
                "group": "异常调用节点",
                "task_description": query,
                "params": "{}",
                "result": "\"\"",
            })
            system_output = result.get("error", "未知错误")
            logger.error(f"[Langgraph] 任务[{task_id}]处理失败: {system_output}")
        
        taskManager.update_task(task_id, nodes, edges, system_output, is_success, True)
        
    except Exception as e:
        error_msg = f"Langgraph任务处理失败: {str(e)}"
        logger.error(f"[Langgraph] 任务[{task_id}]处理失败: {e}\n{traceback.format_exc()}")
        nodes = [{
            "id": "1",
            "name": "1_异常调用节点-" + error_msg,
            "label": "异常调用节点-" + error_msg,
            "group": "异常调用节点",
            "task_description": data.get("query", ""),
            "params": "{}",
            "result": "\"\"",
        }]
        taskManager.update_task(task_id, nodes, [], error_msg, "异常调用链", True)

@app.route('/api_task_status', methods=['POST'])
@require_permissions
def get_task_status():
    """
    tags:
      - Task Management
    description:
        获取任务状态接口
    parameters:
        - name: body
            in: body
            required: true
            schema:
              id: Task Status Request Body
              required:
                - task_id
              properties:
                task_id:
                  type: string
                  description: 任务ID
                  required: true
    responses:
        200:
            description: 任务状态获取成功
            schema:
              id: Task Status Response
              properties:
                task:
                  type: object
                  description: 任务详情
                  properties:
                    task_id:
                      type: string
                      description: 任务ID
                    status:
                      type: integer
                      description: 任务状态
                    nodes:
                      type: array
                      items:
                        type: object
                      description: 节点列表
                    edges:
                      type: array
                      items:
                        type: object
                      description: 边列表
                    isSuccess:
                      type: string
                      description: 是否成功
                    systemOutput:
                      type: string
                      description: 系统输出
        400:
            description: 请求参数错误
            schema:
              id: Error Response
              properties:
                error:
                  type: string
                  description: 错误信息
        404:
            description: 任务未找到或仍在运行
            schema:
              id: Error Response
              properties:
                status:
                  type: string
                  description: 错误状态
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': '缺乏查询参数'}), 400

    task_id = data['task_id']
    task = taskManager.get_task(task_id)
    if task is None:
        return jsonify({'status': '任务未找到或仍在运行中'}), 404

    return jsonify({
        "status": "ok",
        "task": {
            "taskId": task.task_id,
            "status": task.status,
            "nodes": task.nodes,
            "edges": task.edges,
            "systemOutput": task.systemOutput,
            "isSuccess": getattr(task, "isSuccess", ""),
            "isEnd": bool(getattr(task, "isEnd", 0))
        }
    }), RESPONSE_STATUS_CODE_SUCCESS

@app.route('/init_default_tools', methods=['POST', 'GET'])
@require_permissions
def init_default_tools():
    """
    初始化默认工具库
    ---
    tags:
      - Tool Initialization
    description:
        初始化系统默认工具，包含苹果产品查询等常用工具
    responses:
      200:
        description: 工具初始化成功
        schema:
          type: object
          properties:
            message:
              type: string
              example: "工具初始化成功"
    """
    try:
        # 清空旧数据（可选）
        # toolManager.delete_all_tools()
        
        default_tools = [
            {
                "operationId": "get_product_by_name",
                "name_for_human": "按名称查询产品",
                "name_for_model": "get_product_by_name",
                "description": "根据产品名称查询产品详细信息，包括价格、库存、产地等信息。适用于查询水果、商品、物品的信息，如苹果、香蕉、荔枝等",
                "url": "http://localhost:8080",
                "path": "/products/getProductByName",
                "method": "POST",
                "params": [
                    {
                        "param_name": "name",
                        "paramType": "string",
                        "param_description": "产品名称，如苹果、香蕉、橙子、荔枝等",
                        "enum": [],
                        "in_": "body"
                    }
                ]
            },
            {
                "operationId": "get_product_by_id",
                "name_for_human": "按ID查询产品",
                "name_for_model": "get_product_by_id",
                "description": "根据产品ID查询产品详细信息，包括价格、库存、产地等",
                "url": "http://localhost:8080",
                "path": "/products/getProductById",
                "method": "GET",
                "params": [
                    {
                        "param_name": "productId",
                        "paramType": "integer",
                        "param_description": "产品ID",
                        "enum": [],
                        "in_": "query"
                    }
                ]
            },
            {
                "operationId": "get_orders_by_product_id",
                "name_for_human": "查询产品订单",
                "name_for_model": "get_orders_by_product_id",
                "description": "根据产品ID查询与该产品相关的订单信息，包括订单状态、数量、总价等",
                "url": "http://localhost:8080",
                "path": "/orders/getByProductId",
                "method": "GET",
                "params": [
                    {
                        "param_name": "productId",
                        "paramType": "integer",
                        "param_description": "产品ID",
                        "enum": [],
                        "in_": "query"
                    }
                ]
            },
            {
                "operationId": "get_order_by_id",
                "name_for_human": "按ID查询订单",
                "name_for_model": "get_order_by_id",
                "description": "根据订单ID查询订单详细信息，包括数量、总价、状态等",
                "url": "http://localhost:8080",
                "path": "/orders/getOrderByOrderId",
                "method": "GET",
                "params": [
                    {
                        "param_name": "orderId",
                        "paramType": "integer",
                        "param_description": "订单ID",
                        "enum": [],
                        "in_": "query"
                    }
                ]
            },
            {
                "operationId": "create_order",
                "name_for_human": "创建订单",
                "name_for_model": "create_order",
                "description": "根据产品ID和数量创建订单，用于购买商品、发货、下单等场景。适用于将产品发送到指定地点的需求",
                "url": "http://localhost:8080",
                "path": "/orders/createOrder",
                "method": "POST",
                "params": [
                    {
                        "param_name": "productId",
                        "paramType": "integer",
                        "param_description": "产品ID",
                        "enum": [],
                        "in_": "body"
                    },
                    {
                        "param_name": "quantity",
                        "paramType": "integer",
                        "param_description": "购买数量",
                        "enum": [],
                        "in_": "body"
                    },
                    {
                        "param_name": "customerName",
                        "paramType": "string",
                        "param_description": "客户名称",
                        "enum": [],
                        "in_": "body"
                    }
                ]
            },
            {
                "operationId": "get_supplier_by_name",
                "name_for_human": "查询供应商",
                "name_for_model": "get_supplier_by_name",
                "description": "根据供应商名称查询供应商详细信息，包括配送区域、联系方式等",
                "url": "http://localhost:8080",
                "path": "/suppliers/getSupplierByName",
                "method": "GET",
                "params": [
                    {
                        "param_name": "name",
                        "paramType": "string",
                        "param_description": "供应商名称",
                        "enum": [],
                        "in_": "query"
                    }
                ]
            },
            {
                "operationId": "query_suppliers_by_region",
                "name_for_human": "按区域查询供应商",
                "name_for_model": "query_suppliers_by_region",
                "description": "根据配送区域查询供应商信息，适用于查找能够发货到指定城市或地区的供应商，如北京、上海、广东等",
                "url": "http://localhost:8080",
                "path": "/suppliers/querySuppliersByDeliveryRegion",
                "method": "POST",
                "params": [
                    {
                        "param_name": "deliveryRegion",
                        "paramType": "string",
                        "param_description": "配送区域，如北京、广东、上海等",
                        "enum": [],
                        "in_": "body"
                    }
                ]
            },
            {
                "operationId": "get_logistics_by_id",
                "name_for_human": "按ID查询物流公司",
                "name_for_model": "get_logistics_by_id",
                "description": "根据物流公司ID查询物流公司详细信息，包括名称、配送范围等",
                "url": "http://localhost:8080",
                "path": "/logistics/getLogisticsById",
                "method": "GET",
                "params": [
                    {
                        "param_name": "logisticsId",
                        "paramType": "integer",
                        "param_description": "物流公司ID",
                        "enum": [],
                        "in_": "query"
                    }
                ]
            },
            {
                "operationId": "get_logistics_by_name",
                "name_for_human": "按名称查询物流公司",
                "name_for_model": "get_logistics_by_name",
                "description": "根据物流公司名称查询物流公司信息，如京东、顺丰、邮政等",
                "url": "http://localhost:8080",
                "path": "/logistics/getLogisticsByName",
                "method": "GET",
                "params": [
                    {
                        "param_name": "name",
                        "paramType": "string",
                        "param_description": "物流公司名称，如京东、顺丰、邮政",
                        "enum": [],
                        "in_": "query"
                    }
                ]
            },
            {
                "operationId": "query_logistics_by_region",
                "name_for_human": "按区域查询物流公司",
                "name_for_model": "query_logistics_by_region",
                "description": "根据配送区域查询能够配送该区域的物流公司，适用于发货、配送等场景，如查询能配送北京、上海的物流公司",
                "url": "http://localhost:8080",
                "path": "/logistics/queryLogisticsByRegion",
                "method": "POST",
                "params": [
                    {
                        "param_name": "region",
                        "paramType": "string",
                        "param_description": "配送区域，如北京、广东、上海等",
                        "enum": [],
                        "in_": "body"
                    }
                ]
            },
            {
                "operationId": "get_all_logistics",
                "name_for_human": "获取所有物流公司",
                "name_for_model": "get_all_logistics",
                "description": "获取所有可用的物流公司列表，包括名称和配送范围",
                "url": "http://localhost:8080",
                "path": "/logistics/getAllLogistics",
                "method": "GET",
                "params": []
            }
        ]
        
        # 转换为 Tool 对象并插入
        tools_to_insert = []
        for tool_data in default_tools:
            params = []
            for tmp in tool_data["params"]:
                parameter = Parameter(
                    name=tmp["param_name"],
                    type=tmp["paramType"],
                    description=tmp["param_description"],
                    enum=tmp.get("enum", []),
                    required=True,
                    in_=tmp["in_"]
                )
                params.append(parameter)
            
            is_validate = False if "查询" in tool_data["name_for_human"] or "获取" in tool_data["name_for_human"] else True
            tool = Tool(
                tool_id=0,
                operationId=tool_data["operationId"],
                name_for_human=tool_data["name_for_human"],
                name_for_model=tool_data["name_for_model"],
                description=tool_data["description"],
                api_url=tool_data["url"],
                path=tool_data["path"],
                method=tool_data["method"],
                request_body=params,
                isValidate=is_validate
            )
            tools_to_insert.append(tool)
        
        toolManager.insert_tools(tools_to_insert)
        logger.info(f"成功初始化 {len(tools_to_insert)} 个默认工具")
        
        return jsonify({
            "status": RESPONSE_STATUS_CODE_SUCCESS,
            "message": f"工具初始化成功，共插入 {len(tools_to_insert)} 个工具",
            "data": {
                "tools_count": len(tools_to_insert),
                "tools": [t["name_for_human"] for t in default_tools]
            }
        }), RESPONSE_STATUS_CODE_SUCCESS
        
    except Exception as e:
        logger.error(f"工具初始化失败: {e}\n{traceback.format_exc()}")
        return jsonify({
            "status": 400,
            "message": "工具初始化失败",
            "detail": str(e)
        }), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True, use_reloader=False)
