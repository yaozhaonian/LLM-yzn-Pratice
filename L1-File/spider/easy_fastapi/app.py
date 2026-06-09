from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# ================= 数据模型定义 =================

# 用于接收创建用户时的数据（不需要前端传id）
class UserCreate(BaseModel):
    name: str
    email: str
    article: Optional[str] = None # 可选字段

# 用于接收更新用户时的数据
class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    article: Optional[str] = None

# 用于返回给用户的数据结构（包含id）
class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    article: Optional[str] = None

# ================= 模拟数据库 =================

# 初始数据
users_db = [
    {
        'id': 1,
        'name': 'Alice',
        'email': 'Alice@main.com',
        'article': '《春雨》、《咏鹅》'
    },
    {
        'id': 2,
        'name': 'Bob',
        'email': 'Bob@main.com',
        'article': '《百年孤独》、《三国演义》'
    }
]

# 使用一个可变对象来存储当前最大ID，避免 global 关键字的复杂性
# 或者简单点，每次生成ID时遍历数据库找最大ID
def get_next_id():
    if not users_db:
        return 1
    return max(user['id'] for user in users_db) + 1

# ================= 辅助函数 =================

def find_user_by_id(user_id: int):
    for user in users_db:
        if user['id'] == user_id:
            return user
    return None

# ================= 路由接口 =================

@app.get("/users", response_model=List[UserResponse])
async def get_users():
    """获取所有用户"""
    return users_db

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    """获取特定用户"""
    user = find_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user

@app.post("/users", status_code=201)
async def create_user(user: UserCreate):
    """创建新用户"""
    new_id = get_next_id()
    new_user = {
        'id': new_id,
        'name': user.name,
        'email': user.email,
        'article': user.article or ""
    }
    users_db.append(new_user)
    return {"message": "用户添加成功", "user": new_user}

@app.put("/users/{user_id}")
async def update_user(user_id: int, user_update: UserUpdate):
    """更新用户信息"""
    user = find_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    # 只有当字段不为 None 时才更新
    if user_update.name is not None:
        user['name'] = user_update.name
    if user_update.email is not None:
        user['email'] = user_update.email
    if user_update.article is not None:
        user['article'] = user_update.article
        
    return {"message": "用户更新成功", "user": user}

@app.delete("/users/{user_id}")
async def delete_user(user_id: int):
    """删除用户"""
    user = find_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    users_db.remove(user)
    return {"message": "用户删除成功"}

# ================= 嵌套资源 (文章) =================

@app.get("/users/{user_id}/posts")
async def get_user_posts(user_id: int):
    """获取用户的文章"""
    user = find_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    # 返回文章字符串，或者你可以解析成列表
    return {"user_id": user_id, "articles": user.get('article', '')}

@app.post("/users/{user_id}/posts")
async def create_user_post(user_id: int, post_data: dict):
    """更新/添加用户的文章"""
    # 这里简化处理，假设传入 {"article": "新文章内容"}
    user = find_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    if 'article' in post_data:
        user['article'] = post_data['article']
        return {"message": "文章更新成功", "article": user['article']}
    else:
        raise HTTPException(status_code=400, detail="请提供 article 字段")