import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.tool_manager import ToolManager
from utils.config import (
    milvus_uri,
    milvus_db_name,
    mongo_host,
    mongo_db,
    mongo_port,
    mongo_user,
    mongo_password,
    auth_source
)

def init_tools_from_json(json_file_path):
    """从JSON文件初始化工具数据"""
    if not os.path.exists(json_file_path):
        print(f"错误: 文件不存在 {json_file_path}")
        return False

    print(f"正在初始化工具数据，文件: {json_file_path}")
    
    tool_manager = ToolManager(
        mongo_host=mongo_host,
        mongo_db=mongo_db,
        mongo_port=mongo_port,
        milvus_uri=milvus_uri,
        milvus_db_name=milvus_db_name,
        mongo_user=mongo_user,
        mongo_password=mongo_password,
        auth_source=auth_source
    )

    print("1. 清空现有工具数据...")
    tool_manager.delete_all_tools()
    print("   ✓ 已清空")

    print("2. 上传新工具数据...")
    tool_manager.upload_file(json_file_path)
    print("   ✓ 上传完成")

    time.sleep(3)

    print("3. 验证工具数据...")
    tools = tool_manager.get_all_tools()
    print(f"   ✓ 成功加载 {len(tools)} 个工具")
    for tool in tools:
        print(f"     - [{tool['key']}] {tool['name']} ({tool['method']})")

    return True

def create_sample_users():
    """创建示例用户"""
    try:
        from user_manager.user_manager import UserManagerHub
        
        user_manager = UserManagerHub(
            mongo_host=mongo_host,
            mongo_db=mongo_db,
            mongo_port=mongo_port,
            mongo_user=mongo_user,
            mongo_password=mongo_password,
            auth_source=auth_source
        )

        sample_users = [
            {"username": "admin", "password": "admin123"},
            {"username": "test", "password": "test123"}
        ]

        print("4. 创建示例用户...")
        for user in sample_users:
            status, message = user_manager.create_user(user["username"], user["password"], user["password"])
            if status == 200:
                print(f"   ✓ 创建用户 {user['username']} 成功")
            elif status == 409:
                print(f"   ⚠ 用户 {user['username']} 已存在")
            else:
                print(f"   ✗ 创建用户 {user['username']} 失败: {message}")
        
        return True
    except Exception as e:
        print(f"   ✗ 创建用户失败: {e}")
        return False

def main():
    print("=" * 60)
    print("           Agent Copilot 数据初始化工具")
    print("=" * 60)

    json_file_path = os.path.join(os.path.dirname(__file__), "api_data", "dataset_apis_aliyun.json")
    
    if not os.path.exists(json_file_path):
        print(f"错误: API定义文件不存在")
        print(f"请确保文件存在: {json_file_path}")
        sys.exit(1)

    print(f"\n配置信息:")
    print(f"  MongoDB: {mongo_host}:{mongo_port}/{mongo_db}")
    print(f"  Milvus: {milvus_uri}/{milvus_db_name}")
    print(f"  API定义文件: {json_file_path}")
    print()

    try:
        success = init_tools_from_json(json_file_path)
        if success:
            create_sample_users()
        
        print("\n" + "=" * 60)
        print("           数据初始化完成！")
        print("=" * 60)
        print("\n下一步操作:")
        print("  1. 启动服务: python app.py")
        print("  2. 测试API规划:")
        print("     curl -X POST http://localhost:5005/api_planning")
        print("       -H \"Content-Type: application/json\"")
        print("       -d '{\"query\": \"查询产品名为苹果的产品信息\", ...}'")
        
    except Exception as e:
        print(f"\n初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
