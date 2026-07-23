from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from dao.db import engine, get_db, init_db
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def create_database():
    """创建数据库（如果不存在）"""
    try:
        db_url_parts = settings.mysql.url.split('/')
        base_url = '/'.join(db_url_parts[:-1]) + '/'
        
        from sqlalchemy import create_engine as sa_create_engine
        temp_engine = sa_create_engine(base_url)
        
        with temp_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {settings.mysql.database}"))
            conn.commit()
        temp_engine.dispose()
        print(f"   ✅ 数据库 '{settings.mysql.database}' 创建成功")
    except Exception as e:
        print(f"   ⚠️ 创建数据库时出现警告: {str(e)}")


def test_db_connection():
    """
    测试数据库连接是否正常
    
    验证MySQL数据库连接、会话创建和简单查询执行。
    """
    print("=" * 60)
    print("         数据库连接测试")
    print("=" * 60)
    
    try:
        print("\n1. 创建数据库（如果不存在）...")
        create_database()
        
        print("\n2. 测试数据库引擎连接...")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            print(f"   ✅ 连接成功，测试结果: {row[0]}")
        
        print("\n3. 测试数据库会话...")
        with get_db(read_only=True) as db:
            result = db.execute(text("SELECT VERSION() as version"))
            row = result.fetchone()
            print(f"   ✅ MySQL版本: {row[0]}")
        
        print("\n4. 初始化数据库表结构...")
        init_db()
        print("   ✅ 表结构初始化完成")
        
        print("\n5. 测试DAO查询...")
        with get_db(read_only=True) as db:
            result = db.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result.fetchall()]
            if tables:
                print(f"   ✅ 数据库表列表: {tables}")
            else:
                print("   ⚠️ 数据库中暂无表")
        
        print("\n✅ 数据库连接测试全部通过！")
        logger.info("数据库连接测试成功")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        logger.error(f"数据库连接测试失败: {str(e)}")
        raise


if __name__ == "__main__":
    test_db_connection()
