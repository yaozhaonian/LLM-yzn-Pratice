# 手搓与数据库的连接
# PyMySQL 是 Python 连接 MySQL 数据库的纯 Python 驱动库
import pymysql
from datetime import datetime


HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = '0213_pra'
USERNAME = 'root'
PASSWORD = '123456'
MYSQL_URI = f'mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}'

# 建立一次连接，贯穿整个脚本
db_connect = pymysql.connect(
    host=HOSTNAME,
    port=int(PORT),
    user=USERNAME,
    password=PASSWORD,
    database=DATABASE,
    charset='utf8mb4'
)
db_cursor = db_connect.cursor()

try:
    # ==========================
    # 1. 建表部分
    # ==========================
    database_schema_string = """
    CREATE TABLE IF NOT EXISTS Classes (
        class_id INT PRIMARY KEY COMMENT '班级的ID编号',
        class_name VARCHAR(100) NOT NULL COMMENT '班级的名称'
    ) ENGINE=InnoDB COMMENT = '班级表';

    CREATE TABLE IF NOT EXISTS Students (
        student_id INT PRIMARY KEY COMMENT '学生的唯一性ID编号',
        name VARCHAR(100) NOT NULL COMMENT '学生姓名',
        class_id INT COMMENT '学生所在班级的ID编号，和班级表中的班级ID编号对应'
    ) ENGINE=InnoDB COMMENT = '学生表';

    CREATE TABLE IF NOT EXISTS Scores (
        score_id INT PRIMARY KEY COMMENT '学生成绩表的唯一性ID编号',
        student_id INT COMMENT '学生个人的ID编号，和学生的唯一性ID编号对应',
        subject VARCHAR(100) NOT NULL COMMENT '考试科目，中文名称标识',
        score FLOAT NOT NULL COMMENT '考试科目的分数'
    ) ENGINE=InnoDB COMMENT = '学生科目成绩表';
    """

    sql_statements = [stmt.strip() for stmt in database_schema_string.split(';') if stmt.strip()]
    for stmt in sql_statements:
        if stmt: # 确保语句非空
            db_cursor.execute(stmt)
    db_connect.commit()
    print("表创建成功！")

    # ==========================
    # 2. 数据插入部分
    # ==========================
    print(f"数据插入开始：{datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    
    try:
        db_cursor.execute("SET FOREIGN_KEY_CHECKS = 0;") # 暂时禁用外键检查
        db_cursor.execute("TRUNCATE TABLE Scores;")
        db_cursor.execute("TRUNCATE TABLE Students;")
        db_cursor.execute("TRUNCATE TABLE Classes;")
        db_cursor.execute("SET FOREIGN_KEY_CHECKS = 1;") # 恢复外键检查
        print("旧数据已清空")
    except pymysql.Error as e:
        print("清空表时出错（可能表不存在）:", e)

    insert_commands = {
        "classes": "INSERT IGNORE INTO Classes (class_id, class_name) VALUES (%s, %s)",
        "students": "INSERT IGNORE INTO Students (student_id, name, class_id) VALUES (%s, %s, %s)",
        "scores": "INSERT IGNORE INTO Scores (score_id, student_id, subject, score) VALUES (%s, %s, %s, %s)"
    }

    data_pool = {
        "classes": [(1, '一班'), (2, '二班'), (3, '三班'), (4, '四班'), (5, '五班')],
        "students": [
            (1, '张三', 1), (2, '李四', 1), (3, '王五', 2),
            (4, '赵六', 3), (5, '钱七', 4)
        ],
         "scores": [
            # 确保 score_id 唯一：1, 2, 3, 4, 5
            (1, 1, '数学', 95.5),
            (2, 1, '英语', 96.0),
            (3, 2, '语文', 78.0), 
            (4, 3, '英语', 78.5),
            (5, 4, '数学', 92.5) 
        ]
    }

    for table in ['classes', 'students', 'scores']:
        # 注意：如果表中有主键冲突，重复运行此脚本会报错。
        # 生产环境通常先 TRUNCATE 表或使用 INSERT IGNORE / ON DUPLICATE KEY UPDATE
        result = db_cursor.executemany(insert_commands[table], data_pool[table])
        print(f"[{table.upper()}] 表插入了 {result} 条数据")
    
    db_connect.commit()
    print(f"数据插入结束：{datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
        # 在 finally 之前添加这段调试代码
    db_cursor.execute("SHOW TABLES")
    tables = db_cursor.fetchall()
    print("当前数据库中的表:", tables)
    
    if tables:
        db_cursor.execute("SELECT COUNT(*) FROM Classes")
        count = db_cursor.fetchone()[0]
        print(f"Classes 表中的数据行数: {count}")
    else:
        print("警告：数据库中没有任何表！")

except pymysql.Error as e:
    print("发生错误:", e)
    db_connect.rollback()

finally:
    # 只有在所有操作结束后，才关闭连接
    if db_cursor:
        db_cursor.close()
    if db_connect:
        db_connect.close()
    print("数据库连接已关闭")