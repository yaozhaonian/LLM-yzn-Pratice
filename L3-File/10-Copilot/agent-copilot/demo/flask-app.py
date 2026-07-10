from flask import Flask, request, jsonify, redirect, url_for
from langchain_ollama import ChatOllama
llm = ChatOllama(model='qwen2.5:7b', temperature=0.1,base_url="http://127.0.0.1:11434")
from flasgger import Swagger

app = Flask(__name__)
Swagger(app=app)

@app.route("/")
def index():
    return redirect(url_for('hello'))

@app.route("/hello")
def hello():
    """
    示例测试接口
    ---
    responses:
      200:
        description: 返回问候语
    """
    return {"msg": "Hello Swagger + Flask"}

@app.route('/api/data', methods=['GET'])
def get_data():
    """
    获取样本数据
    这是一个简单的 GET 方法，用于返回样本数据。
    ---
    tags:
      - Sample API
    responses:
      200:
        description: 成功返回样本数据
        schema:
          type: object
          properties:
            message:
              type: string
              example: This is a sample data response
    """
    data = {"message": "简单的数据读取"}
    return jsonify(data)

@app.route('/api', methods=['GET'])
def get_data_index():
    """
    获取指定 ID 的数据
    这是一个带有路径参数的 GET 方法，用于根据 ID 返回特定数据。
    ---
    tags:
      - Sample API
    parameters:
      - name: item_id
        in: query
        type: string
        required: true
        description: 数据的唯一标识符
    responses:
      200:
        description: 成功返回指定 ID 的数据
        schema:
          type: object
          properties:
            message:
              type: string
      404:
        description: 未找到指定 ID 的数据
        schema:
          type: object
          properties:
            error:
              type: string
              example: Item not found
    """
    item_id = request.args.get('item_id')
    if item_id == "123":
        data = {"message": f"Data for item_id: {item_id}"}
    else:
        return jsonify({"error": "Item not found"}), 404
    return jsonify(data)

@app.route('/sentiment_analysis', methods=['POST'])
def mesh_query():
    """
    
    情感分析
        ---
        tags:
          - sentiment API
        description:
            情感分析接口，json格式
        parameters:
          - name: body
            in: body
            required: true
            schema:
              id: 情感分析body
              required:
                - query
              properties:
                query:
                  type: string
                  description: 分析语句.

        responses:
          200:
              description: 转化成功
              schema:
                type: object
                properties:
                  message:
                    type: string"""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': '缺乏query参数'}), 400
    
    query = data['query']
    content = query + """\n
    -------------------
    请分析上述文本的情感
    请直接输出结果，不需要输出额外的解释
    """
    try:
        response = llm.invoke(content)
        print("大模型回复内容",response.content)
        return {"message": response.content}
    except Exception as e:
        return jsonify({'message': str(e),'fail':'is fail'})

if __name__ == "__main__":
    print("Swagger访问地址：http://127.0.0.1:5005/apidocs")
    app.run(debug=False, host='0.0.0.0', port=5005)
