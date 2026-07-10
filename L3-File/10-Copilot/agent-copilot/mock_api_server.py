from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
import json

# ===================== MongoDB 配置 =====================
MONGO_HOST = "127.0.0.1"
MONGO_PORT = 27017
DB_NAME = "product_db"

# 连接Mongo
client = MongoClient(
    MONGO_HOST,
    MONGO_PORT,
    username="admin",    # 你的Mongo用户名
    password="123456", # 你的Mongo密码
    authSource="admin"
)
db = client[DB_NAME]

# 三个集合
col_product = db["products"]
col_order = db["orders"]
col_supplier = db["suppliers"]
col_logistics = db["logistics"]

# 初始测试数据（仅第一次运行插入，避免重复）
init_products = [
    {"productId": 1, "name": "苹果", "address_province": "山东省", "address_city": "烟台市", "price": 12.99, "inventory": 100},
    {"productId": 2, "name": "香蕉", "address_province": "海南省", "address_city": "海口市", "price": 6.99, "inventory": 200},
    {"productId": 3, "name": "橙子", "address_province": "江西省", "address_city": "赣州市", "price": 8.99, "inventory": 150},
    {"productId": 4, "name": "橘子", "address_province": "广西省", "address_city": "南宁市", "price": 5.99, "inventory": 300},
    {"productId": 5, "name": "葡萄", "address_province": "新疆省", "address_city": "喀什市", "price": 15.99, "inventory": 80},
    {"productId": 6, "name": "西瓜", "address_province": "宁夏省", "address_city": "", "price": 29.99, "inventory": 50},
    {"productId": 7, "name": "草莓", "address_province": "云南省", "address_city": "丽江市", "price": 25.99, "inventory": 60},
    {"productId": 8, "name": "蓝莓", "address_province": "辽宁省", "address_city": "沈阳市", "price": 39.99, "inventory": 40},
    {"productId": 8, "name": "荔枝", "address_province": "广东省", "address_city": "高州市", "price": 45.88, "inventory": 300},
]

init_orders = [
    {"orderId": 1, "productId": 1, "quantity": 10, "status": "已完成", "totalPrice": 129.90, "customerName": "张三"},
    {"orderId": 2, "productId": 2, "quantity": 20, "status": "处理中", "totalPrice": 139.80, "customerName": "李四"},
    {"orderId": 3, "productId": 3, "quantity": 5, "status": "运输中", "totalPrice": 44.95, "customerName": "王五"},
    {"orderId": 4, "productId": 1, "quantity": 15, "status": "待付款", "totalPrice": 194.85, "customerName": "赵六"},
    {"orderId": 5, "productId": 5, "quantity": 8, "status": "已取消", "totalPrice": 127.92, "customerName": "钱七"},
]

init_suppliers = [
    {"supplierId": 1, "name": "山东果品公司", "deliveryRegion": ["北京", "天津", "河北"], "contact": "李经理"},
    {"supplierId": 2, "name": "海南热带水果", "deliveryRegion": ["广东", "广西", "海南"], "contact": "王经理"},
    {"supplierId": 3, "name": "新疆果业集团", "deliveryRegion": ["新疆", "甘肃", "青海"], "contact": "张经理"},
    {"supplierId": 4, "name": "云南草莓基地", "deliveryRegion": ["全国"], "contact": "刘经理"},
]

init_logistics = [
    {"logisticsId": 1, "name": "京东", "range": "广东省、北京市、湖南省"},
    {"logisticsId": 2, "name": "顺丰", "range": "广西省、辽宁省、广东省、山东省"},
    {"logisticsId": 3, "name": "邮政", "range": "广东省、北京市、湖南省、广西省、辽宁省、山东省、新疆省、江西省、宁夏省、海南省"},
]

# 初始化数据：集合为空时插入初始数据
if col_product.count_documents({}) == 0:
    col_product.insert_many(init_products)
if col_order.count_documents({}) == 0:
    col_order.insert_many(init_orders)
if col_supplier.count_documents({}) == 0:
    col_supplier.insert_many(init_suppliers)
if col_logistics.count_documents({}) == 0:
    col_logistics.insert_many(init_logistics)

# ===================== Flask 初始化 =====================
app = Flask(__name__)
CORS(app)

def json_resp(data):
    return Response(
        json.dumps(data, ensure_ascii=False, default=str),
        mimetype="application/json; charset=utf-8"
    )

# 鉴权装饰器（预留扩展）
def authenticate(func):
    def wrapper(*args, **kwargs):
        # 后续可加token校验逻辑
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

# 工具函数：去除Mongo _id，方便返回json
def clean_mongo_item(item):
    if item:
        item.pop("_id", None)
    return item

def clean_mongo_list(items):
    return [clean_mongo_item(i) for i in items]

# ===================== 商品接口 =====================
@app.route('/products/getProductByName', methods=['GET', 'POST'])
@authenticate
def get_product_by_name():
    name = ""
    # GET 从url参数拿
    if request.method == "GET":
        name = request.args.get('name', '').strip()
    # POST 从json body拿
    else:
        # 防止无json体报错
        if request.is_json:
            data = request.get_json() or {}
            name = data.get('name', '').strip()
    # 模糊匹配
    cursor = col_product.find({"name": {"$regex": name}})
    return json_resp(clean_mongo_list(list(cursor)))

@app.route('/products/getProductById', methods=['GET'])
@authenticate
def get_product_by_id():
    try:
        product_id = int(request.args.get('productId'))
    except:
        return jsonify([])
    item = col_product.find_one({"productId": product_id})
    res = [clean_mongo_item(item)] if item else []
    return json_resp(res)

@app.route('/products/getBatchProductByProductIds', methods=['POST'])
@authenticate
def get_batch_products():
    data = request.get_json() or {}
    ids = data.get('productIds', [])
    cursor = col_product.find({"productId": {"$in": ids}})
    return json_resp(clean_mongo_list(list(cursor)))

@app.route('/products/getProductSubstitutes', methods=['GET'])
@authenticate
def get_product_substitutes():
    try:
        product_id = int(request.args.get('productId'))
    except:
        return jsonify([])
    product = col_product.find_one({"productId": product_id})
    if not product or product.get("substituteProductId") is None:
        return jsonify([])
    sub_id = product["substituteProductId"]
    sub = col_product.find_one({"productId": sub_id})
    return json_resp([clean_mongo_item(sub)] if sub else [])

@app.route('/products/addProduct', methods=['POST'])
@authenticate
def add_product():
    data = request.get_json() or {}
    # 获取最大productId自增
    max_p = col_product.find_one(sort=[("productId", -1)])
    new_id = max_p["productId"] + 1 if max_p else 1
    new_product = {
        "productId": new_id,
        "name": data.get('name'),
        "description": data.get('description', ""),
        "price": data.get('price', 0),
        "quantityInStock": data.get('quantityInStock', 0),
        "substituteProductId": data.get('substituteProductId', None)
    }
    col_product.insert_one(new_product)
    return json_resp({"success": True, "productId": new_id})

@app.route('/products/removeProductByName', methods=['DELETE'])
@authenticate
def remove_product():
    data = request.get_json() or {}
    name = data.get('name')
    if not name:
        return jsonify({"success": False})
    res = col_product.delete_many({"name": name})
    return jsonify({"success": res.deleted_count > 0})

# ===================== 订单接口 =====================
@app.route('/orders/createOrder', methods=['POST'])
@authenticate
def create_order():
    data = request.get_json() or {}
    product_id = data.get('productId')
    quantity = data.get('quantity')
    if not isinstance(product_id, int) or not isinstance(quantity, int) or quantity <= 0:
        return json_resp({"success": False, "message": "参数非法"}), 400

    product = col_product.find_one({"productId": product_id})
    if not product or product["quantityInStock"] < quantity:
        return json_resp({"success": False, "message": "库存不足"}), 400

    # 生成订单ID
    max_o = col_order.find_one(sort=[("orderId", -1)])
    new_order_id = max_o["orderId"] + 1 if max_o else 1
    total = product["price"] * quantity
    new_order = {
        "orderId": new_order_id,
        "productId": product_id,
        "quantity": quantity,
        "status": "待付款",
        "totalPrice": total,
        "customerName": data.get('customerName', '匿名用户')
    }
    col_order.insert_one(new_order)
    # 扣库存
    col_product.update_one(
        {"productId": product_id},
        {"$inc": {"quantityInStock": -quantity}}
    )
    return json_resp({"success": True, "orderId": new_order_id})

@app.route('/orders/getOrderByOrderId', methods=['GET'])
@authenticate
def get_order_by_id():
    try:
        order_id = int(request.args.get('orderId'))
    except:
        return jsonify([])
    item = col_order.find_one({"orderId": order_id})
    return json_resp([clean_mongo_item(item)] if item else [])

@app.route('/orders/getByProductId', methods=['GET'])
@authenticate
def get_orders_by_product():
    try:
        product_id = int(request.args.get('productId'))
    except:
        return jsonify([])
    cursor = col_order.find({"productId": product_id})
    return json_resp(clean_mongo_list(list(cursor)))

@app.route('/orders/updateOrderStatus', methods=['PUT'])
@authenticate
def update_order_status():
    data = request.get_json() or {}
    order_id = data.get('orderId')
    status = data.get('status')
    if not isinstance(order_id, int) or not status:
        return jsonify(False)
    res = col_order.update_one(
        {"orderId": order_id},
        {"$set": {"status": status}}
    )
    return json_resp(res.modified_count > 0)

@app.route('/orders/cancelOrder', methods=['DELETE'])
@authenticate
def cancel_order():
    data = request.get_json() or {}
    order_id = data.get('orderId')
    if not isinstance(order_id, int):
        return json_resp(False)
    order = col_order.find_one({"orderId": order_id})
    if not order:
        return json_resp(False)
    # 取消订单：恢复库存
    col_product.update_one(
        {"productId": order["productId"]},
        {"$inc": {"quantityInStock": order["quantity"]}}
    )
    col_order.update_one({"orderId": order_id}, {"$set": {"status": "已取消"}})
    return json_resp(True)

# ===================== 供应商接口 =====================
@app.route('/suppliers/getSupplierById', methods=['GET'])
@authenticate
def get_supplier_by_id():
    try:
        sid = int(request.args.get('supplierId'))
    except:
        return jsonify([])
    item = col_supplier.find_one({"supplierId": sid})
    return json_resp([clean_mongo_item(item)] if item else [])

@app.route('/suppliers/getSupplierByName', methods=['GET'])
@authenticate
def get_supplier_by_name():
    name = request.args.get('name', '')
    cursor = col_supplier.find({"name": {"$regex": name}})
    return json_resp(clean_mongo_list(list(cursor)))

@app.route('/suppliers/querySuppliersByDeliveryRegion', methods=['POST'])
@authenticate
def query_suppliers_by_region():
    data = request.get_json() or {}
    region = data.get('deliveryRegion', '')
    # 匹配包含该地区 或 全国
    cursor = col_supplier.find({
        "$or": [
            {"deliveryRegion": region},
            {"deliveryRegion": "全国"}
        ]
    })
    return json_resp(clean_mongo_list(list(cursor)))

# 健康检查
@app.route('/health', methods=['GET'])
def health():
    try:
        client.admin.command("ping")
        mongo_ok = True
    except Exception as e:
        mongo_ok = False
    return json_resp({
        "status": "ok",
        "mongo_connected": mongo_ok,
        "db": DB_NAME
    })

# 启动入口
if __name__ == '__main__':
    print("=" * 60)
    print(f"✅ MongoDB 连接成功 | 数据库：{DB_NAME} 端口：{MONGO_PORT}")
    print("           Flask API 服务启动中...")
    print("=" * 60)
    print(f"服务地址: http://127.0.0.1:8080")
    print(f"商品集合数量: {col_product.count_documents({})}")
    print(f"订单集合数量: {col_order.count_documents({})}")
    print(f"供应商集合数量: {col_supplier.count_documents({})}")
    print(f"物流集合数量: {col_logistics.count_documents({})}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8080, debug=True)