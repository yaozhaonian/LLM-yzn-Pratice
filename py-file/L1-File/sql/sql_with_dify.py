"""
房产数据分析助手的服务
"""
from flask import Flask, request, jsonify
import pymysql

app = Flask(__name__)

# 配置数据库连接参数（需要修改）
db_config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '123456', # 数据库密码
    'db': 'dify_test' # 数据库名
}

def execute_query(sql):
    # 连接数据库
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            # 执行 SQL 查询
            cursor.execute(sql)
            result = cursor.fetchall()
            return result
    finally:
        # 关闭数据库连接
        connection.close()

@app.route('/execute_query', methods=['GET'])
def get_data():
    # 从查询参数中获取 SQL 查询
    sql_query = request.args.get('sql_query', default='', type=str)
    print("接收到查询：",sql_query)
    # 执行 SQL 查询并获取结果
    result = execute_query(sql_query)
    print("执行结果：",result)
    # 返回 JSON 格式的响应
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5001)