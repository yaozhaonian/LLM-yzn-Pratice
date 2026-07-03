# 用户管理

# 测试用的路径
import os
import sys
if __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mongoengine import *
from pymongo import ReturnDocument
import threading
from entity import User
from cachetools import TTLCache
import bcrypt
from utils import logger, DEFAULT_PERMISSIONS
import traceback

def hash_password(password: str) -> str:
    """对密码进行哈希加密"""
    # 生成盐值并加密密码
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password=password.encode('utf-8'),salt=salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    """验证密码是否正确"""
    return bcrypt.checkpw(
        password=password.encode('utf-8'),
        hashed_password=hashed_password.encode('utf-8')
    )

def easy_verify_password(password: str, cache_password: str) -> bool:
    """无加密版密码验证"""
    return password == cache_password

class UserManagerHub:
    def __init__(self, mongo_host, mongo_db, mongo_port, mongo_user, mongo_password, auth_source='admin'):
        self.mongo_host = mongo_host
        self.mongo_db = mongo_db
        self.mongo_port = mongo_port
        self.mongo_user = mongo_user
        self.mongo_password = mongo_password
        self.auth_source = auth_source
        self.mongoClient = connect(
            mongo_db,
            host=mongo_host,
            port=mongo_port,
            username=mongo_user,
            password=mongo_password,
            authentication_source=auth_source,
        )
        self.db_name = mongo_db  # 保存数据库名称
        self.cache_lock = threading.Lock()  # 上锁
        self.user_cache = TTLCache(maxsize=100, ttl=3600)

    def _ensure_connection(self):
        """确保MongoDB连接有效，如果连接已关闭则重新连接"""
        try:
            # 尝试执行一个简单的操作来检查连接是否有效
            self.mongoClient.admin.command('ping')
        except Exception:
            # 连接已关闭或无效，重新连接
            try:
                self.mongoClient = connect(self.mongo_db, host=self.mongo_host, port=self.mongo_port, username=self.mongo_user, password=self.mongo_password, authentication_source=self.auth_source)
            except Exception as e:
                logger.error(f"重新连接MongoDB失败: {e}")
                raise

    def get_next_user_id(self):
        self._ensure_connection()
        db = self.mongoClient[self.db_name]
        # 使用 find_one_and_update 原子操作
        counter = db.counters.find_one_and_update(
            {"_id": "user_id"},
            {"$inc": {"sequence_value": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER
        )
        return counter["sequence_value"]
    
    def create_user(self, user_name, password, confirm_password, permissions=None):
        if password != confirm_password:
            return 409, "两次输入的密码不一致"
        
        try:
            # 先检查用户名是否存在
            if User.objects(username=user_name).first():
                return 409, "该用户已注册"
            
            # 原子获取用户ID
            user_id = self.get_next_user_id()
            # password = hash_password(password)
            # 设置默认权限
            if permissions is None:
                permissions = DEFAULT_PERMISSIONS
            # 创建用户
            current_user = User(user_id=user_id, username=user_name, password=password, user_authority=permissions)
            current_user.save()
            return 201, "注册成功"
        except NotUniqueError:
            return 409, "用户已存在"
        except Exception as e:
            logger.error(f"注册失败: {e}\n{traceback.format_exc()}")
            return 500, f"注册失败: {str(e)}"

    def login(self, user_name, password):
        try:
            users = User.objects(username=user_name)
            if users:
                user = users.first()
                stored_password = user.password
                if isinstance(stored_password, str):
                    stored_password = stored_password.encode('utf-8')
                # if bcrypt.checkpw(password.encode('utf-8'), stored_password):
                #     with self.cache_lock:
                #         self.user_cache[user.user_id] = user
                #     return user
                if easy_verify_password(password, stored_password.decode('utf-8')):
                    with self.cache_lock:
                        self.user_cache[user.user_id] = user
                    return user
        except Exception as e:
            logger.error(f"登录失败: {e}\n{traceback.format_exc()}")
        return User(user_id=-1)

    def logout(self, user_id):
        with self.cache_lock:
            self.user_cache.pop(user_id, None)
        return True
    
    def islogin(self, user_id):
        user = self.user_cache.get(user_id)
        if user:
            return True
        else:
            return False


if __name__ == "__main__":
    # 测试代码
    user_manager = UserManagerHub(
        mongo_host="localhost",
        mongo_db="test_db",
        mongo_port=27017,
        mongo_user="admin",
        mongo_password="123456",
        auth_source="admin"
    )
    status, message = user_manager.create_user("testuserA", "password456", "password456")
    print(f"创建用户: Status={status}, Message={message}")
    
    user = user_manager.login("testuserA", "password456")
    if user.user_id != -1:
        print(f"登录成功，用户ID: {user.user_id}")
        is_logged_in = user_manager.islogin(user.user_id)
        print(f"是否登录成功: {is_logged_in}")
        
        logout_success = user_manager.logout(user.user_id)
        print(f"退出登录成功: {logout_success}")
        
        is_logged_in_after_logout = user_manager.islogin(user.user_id)
        print(f"退出登录后是否登陆: {is_logged_in_after_logout}")
    else:
        print("登录失败.")
        




