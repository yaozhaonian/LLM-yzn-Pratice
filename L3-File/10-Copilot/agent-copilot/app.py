import time
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

# 获取CPU核心数并计算线程池大小
cpu_count = os.cpu_count() or 1
max_workers = max(2, cpu_count * 2)  # 至少2个线程，最多为CPU核心数的两倍
executor = ThreadPoolExecutor(max_workers=max_workers)
tasks = {}  # 存储任务的字典，键为任务ID，值为任务对象

CORS(app, resources={
    r"/login_user":{"origins":"http://localhost:3000"},
    r"/register_user":{"origins":"http://localhost:3000"},
    r"/get_all_tools":{"origins":"http://localhost:3000"},
    r"/delete_all_tool":{"origins":"http://localhost:3000"},
    r"/upload_file":{"origins":"http://localhost:3000"},
    r"/delete_tool_by_ids":{"origins":"http://localhost:3000"},
    r"/test_llm":{"origins":"http://localhost:3000"},
    r"/api_task_status":{"origins":"http://localhost:3000"},
    r"/api_planning":{"origins":"http://localhost:3000"},
})  # 允许跨域请求

Swagger(app=app)

toolManager = ToolManager(mongo_host, mongo_db, mongo_port, milvus_uri, milvus_db_name)
taskManager = TaskManager(mongo_host, mongo_db, mongo_port)
userManagerHub = UserManagerHub(mongo_host, mongo_db, mongo_port, "admin", "123456", auth_source="admin")

# 权限验证装饰器
def require_permissions(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 检查是否说免验证的端点
        if f.__name__ in ['login', 'register']:
            return f(*args, **kwargs)

        # 检查Authorization头
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({
                "status": RESPONSE_AUTH_CODE_ERROR,
                "message": "未提供认证令牌",
                "data": None
                }), RESPONSE_AUTH_CODE_ERROR
        try:
            # 验证Bearer令牌格式
            if not token.startswith("Bearer "):
                return jsonify({
                    "status": RESPONSE_AUTH_CODE_ERROR,
                    "message": "无效的令牌格式",
                    "data": None
                }), RESPONSE_AUTH_CODE_ERROR
            # 提取令牌
            token = token.split(" ")[1]
            # 验证令牌是否在缓存中
            if token not in session_cache:
                return jsonify({
                    "status": RESPONSE_ALLOW_CODE_ERROR,
                    "message": "令牌无效或已过期",
                    "data": None
                }), RESPONSE_ALLOW_CODE_ERROR
            
            # 解码JWT令牌(验证签名和有效期)
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                # 令牌已过期，从缓存中移除
                if token in session_cache:
                    del session_cache[token]
                return jsonify({
                    "status": RESPONSE_ALLOW_CODE_ERROR,
                    "message": "令牌已过期",
                    "data": None
                }), RESPONSE_ALLOW_CODE_ERROR
            except jwt.InvalidTokenError:
                return jsonify({
                    "status": RESPONSE_ALLOW_CODE_ERROR,
                    "message": "无效的令牌",
                    "data": None
                }), RESPONSE_ALLOW_CODE_ERROR
            
            # 将用户信息存储到g对象，供后续处理使用
            g.current_user = session_cache[token]

            # 检查用户是否有权限访问当前接口
            if f.__name__ not in g.current_user['user_authority']:
                return jsonify({
                    "status": RESPONSE_STATUS_CODE_ERROR,
                    "message": "权限不足，无法访问该接口{f.__name__}",
                    "data": None
                }), RESPONSE_STATUS_CODE_ERROR
        except Exception as e:
            logger.error(f"认证失败: {str(e)}")
            return jsonify({
                "status": RESPONSE_AUTH_CODE_ERROR,
                "message": "认证失败",
                "data": None
            }), RESPONSE_AUTH_CODE_ERROR
        return f(*args, **kwargs)
    return decorated_function

@app.route('/delete_all_tool', methods=['GET'])
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
    tooManager.delete_all_tools()
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
    tooManager.insert_tools([tool])
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
    data = request.get_json()
    if not data:
        return jsonify({"message": "请求体为空"}), 400

    username = data.get("username")
    password = data.get("password")
    user = userManagerHub.login(username, password)
    if user.user_id != -1:
        # 登录成功，生成JWT令牌
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'user_authority': user.user_authority,
            'exp': datetime.utcnow() + JWT_EXPIRATION_DELTA  # 设置过期时间
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)
        logger.debug(f"生成JWT令牌: {token}")

        # 将用户信息存储到缓存中，使用token作为键
        session_cache[token] = {
            'user_id': user.user_id,
            'username': user.username,
            'user_authority': user.user_authority
        }
        logger.debug(f"用户信息存储到缓存中: {session_cache[token]}")

        logger.info(f"用户登录成功: {username}")
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

@app.route('/upload_file', methods=['POST'])
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
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件或者文件名为空'}), 400

    # 保存文件到指定目录
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)

    file = request.files["file"]
    # 过滤危险路径字符
    safe_name = secure_filename(file.filename)
    # 拆分后缀，uuid重命名避免覆盖
    suffix = safe_name.split(".")[-1]
    unique_name = f"{uuid.uuid4().hex}.{suffix}"
    file_path = os.path.join(upload_folder, unique_name)
    file.save(file_path)
    toolManager.upload_file(file_path)

    logger.info(f"文件上传成功: {file.filename}")
    return jsonify({'message': f'{file.filename} 文件上传成功!'}), RESPONSE_STATUS_CODE_SUCCESS

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
    responses:
        200:
            description: 转化成功
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Missing query parameter'}), 400

    task_id = taskManager.create_task()
    executor.submit(_process_task, task_id, data)   # 将任务提交到线程池
    return jsonify({"task_id": task_id, "message": "任务已创建，正在处理中"}), RESPONSE_STATUS_CODE_SUCCESS

def _process_task(task_id, data):
    logger.info(f"准备处理任务{task_id}，任务数据：{data}，处理中......")
    try:
        query = data["query"]
        contexts = data["contexts"]
        isCopilot = data["isCopilot"]
        isContext = data["isContext"]
        contextNumber = data["contextNumber"]

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
        api_planning_hub = ApiPlanningHub(milvus_uri, model_path, milvus_db_name, curr_model_name,
                                        curr_temperature,model_top_p, mongo_host, mongo_db, mongo_port, topK,
                                        curr_api_url, curr_api_key)
        generate_task_hub = GenerateTaskHub(curr_model_name, curr_temperature, model_top_p,
                                            curr_api_url, curr_api_key, mongo_host, mongo_db, mongo_port, milvus_uri, milvus_db_name)
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
        logger.error(f"任务[{task_id}]处理失败: {e}\n{traceback.format_exc()}")
        taskManager.update_task(task_id, [], [], "任务处理失败，请联系你的系统管理员", "", True)
        return jsonify({'message': f"任务[{query}]处理失败，任务编号[{task_id}]，请联系你的系统管理员"}), 400

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
        "task": {'task': task.to_dict()}
    }), RESPONSE_STATUS_CODE_SUCCESS

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=False)
