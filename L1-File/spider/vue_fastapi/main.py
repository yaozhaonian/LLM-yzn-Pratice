# 可以与vue结合
# uvicorn main:app --reload
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模拟数据库数据
fake_db = [
    {"id": 1, "name": "Python 学习", "price": 99},
    {"id": 2, "name": "Vue 教程", "price": 88},
    {"id": 3, "name": "FastAPI 实战", "price": 128}
]

# 1. GET 接口：获取数据
@app.get("/api/products")
def get_products():
    return {"code": 200, "data": fake_db, "msg": "获取成功"}

# 2. POST 接口：提交数据
class Product(BaseModel):
    name: str
    price: float

@app.post("/api/products")
def add_product(product: Product):
    fake_db.append({
        "id": len(fake_db) + 1,
        "name": product.name,
        "price": product.price
    })
    return {"code": 200, "msg": "添加成功", "data": product}


@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!", "docs": "/docs"}

@app.get("/api/products/{product_id}")
def get_one_product(product_id: int):
    for item in fake_db:
        if item["id"] == product_id:
            return item
    return {"msg": "未找到"}