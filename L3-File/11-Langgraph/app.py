#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERP智能客服Flask接口服务

提供RESTful API接口，集成LangGraph工作流，支持多轮对话交互。
"""

import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from graph.workflow import create_workflow
from graph.state import init_state
from utils.logger import get_logger

# 获取日志记录器
logger = get_logger(__name__)

# 创建Flask应用实例
app = Flask(__name__, static_folder='static', static_url_path='')

# 配置CORS跨域，支持前端页面访问
CORS(app, resources={r"/api/*": {"origins": "*"}})

# HTTP异常处理（404、405等）
@app.errorhandler(HTTPException)
def handle_http_exception(e):
    """
    HTTP异常处理
    
    处理Flask内置的HTTP异常（如404、405），保持原有的HTTP状态码。
    
    Args:
        e: HTTP异常对象
    
    Returns:
        JSON: 错误响应，包含状态码和错误信息
    """
    logger.warning(f"HTTP异常: {e.code} - {e.description}")
    return jsonify({
        "code": e.code,
        "msg": f"请求错误: {e.description}",
        "data": {}
    }), e.code

# 全局业务异常处理
@app.errorhandler(Exception)
def handle_exception(e):
    """
    全局业务异常捕获处理
    
    捕获所有未处理的业务异常，返回统一的500错误响应。
    注意：HTTP异常（如404）已在上一个处理器中处理，不会走到这里。
    
    Args:
        e: 异常对象
    
    Returns:
        JSON: 错误响应，包含状态码和错误信息
    """
    logger.error(f"全局业务异常捕获: {str(e)}", exc_info=True)
    return jsonify({
        "code": 500,
        "msg": f"服务器内部错误: {str(e)}",
        "data": {}
    }), 500

# 初始化LangGraph工作流（延迟加载，首次请求时初始化）
_workflow = None

def get_workflow():
    """
    获取LangGraph工作流实例（单例模式）
    
    延迟初始化工作流，避免应用启动时的耗时操作。
    
    Returns:
        CompiledGraph: LangGraph工作流实例
    """
    global _workflow
    if _workflow is None:
        logger.info("初始化LangGraph工作流...")
        _workflow = create_workflow()
        logger.info("LangGraph工作流初始化完成")
    return _workflow

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    对话接口
    
    接收用户消息，调用LangGraph工作流处理，返回对话回复。
    
    请求参数（JSON格式）:
        session_id: 会话ID，用于维持多轮对话上下文
        user_input: 用户输入的消息内容
    
    返回参数（JSON格式）:
        code: 状态码，0表示成功，非0表示失败
        msg: 提示信息
        data: 数据对象，包含response字段（对话回复）
    """
    try:
        # 获取请求数据
        data = request.get_json()
        
        # 参数校验
        if not data:
            return jsonify({
                "code": 400,
                "msg": "请求数据为空",
                "data": {}
            }), 400
        
        session_id = data.get("session_id")
        user_input = data.get("user_input")
        
        if not session_id:
            return jsonify({
                "code": 400,
                "msg": "缺少session_id参数",
                "data": {}
            }), 400
        
        if not user_input:
            return jsonify({
                "code": 400,
                "msg": "缺少user_input参数",
                "data": {}
            }), 400
        
        logger.info(f"收到对话请求，session_id: {session_id}, user_input: {user_input[:50]}...")
        
        # 获取工作流实例
        workflow = get_workflow()
        
        # 配置thread_id使用session_id，维持多轮对话上下文
        config = {"configurable": {"thread_id": session_id}}
        
        # 构建输入状态
        input_state = {
            **init_state(session_id),
            "user_input": user_input
        }
        
        # 执行LangGraph工作流
        result = workflow.invoke(input_state, config)
        
        # 提取回复内容
        response = result.get("final_response", "")
        
        logger.info(f"对话请求处理完成，session_id: {session_id}, response_length: {len(response)}")
        
        # 返回成功响应
        return jsonify({
            "code": 0,
            "msg": "success",
            "data": {
                "response": response
            }
        })
    
    except Exception as e:
        logger.error(f"对话接口异常，session_id: {session_id}, error: {str(e)}", exc_info=True)
        return jsonify({
            "code": 500,
            "msg": f"对话处理失败: {str(e)}",
            "data": {}
        }), 500

@app.route('/')
def index():
    """
    默认路由，返回前端页面
    
    访问根路径时，返回静态文件夹中的index.html页面。
    
    Returns:
        HTML: 前端页面内容
    """
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    """
    启动Flask应用
    
    默认端口5011，开启开发模式热重载。
    
    启动命令:
        python app.py
    """
    logger.info("启动ERP智能客服Flask服务...")
    app.run(
        host='0.0.0.0',
        port=5011,
        debug=True
    )