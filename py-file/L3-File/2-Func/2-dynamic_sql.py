# 不用Langchain的一次对数据库的操作
import pymysql
import json
from openai import OpenAI

# 1. 初始化 OpenAI 客户端 (指向 Ollama)
# 2-llm_lc_fc.py中有ChatOllama的相关用法
client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

# 2. 定义数据库 Schema (优化点：增加了注释说明数据格式)
database_schema_string = """
CREATE TABLE Classes (
    class_id INT PRIMARY KEY,
    class_name VARCHAR(100) -- 示例: '一班', '二班'
);
CREATE TABLE Students (
    student_id INT PRIMARY KEY,
    name VARCHAR(100), -- 示例: '张三', '李四'
    class_id INT
);
CREATE TABLE Scores (
    score_id INT PRIMARY KEY,
    student_id INT,
    subject VARCHAR(100), -- 重要：科目名称存储为中文，例如: '数学', '英语', '语文'
    score FLOAT
);
"""

# 3. 定义真正的执行函数
def execute_sql_query(sql_query: str) -> str:
    if not sql_query.strip().upper().startswith("SELECT"):
        return "错误：为了安全，仅允许执行 SELECT 查询。"
    
    conn = None
    cursor = None
    try:
        conn = pymysql.connect(
            host='127.0.0.1',
            port=3306,
            user='root',
            password='123456',
            database='0213_pra',
            charset='utf8mb4'
        )
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        
        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
        else:
            columns = []
            
        return json.dumps({"columns": columns, "data": results}, ensure_ascii=False)
        
    except Exception as e:
        return f"数据库查询错误: {str(e)}"
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

# 4. 定义工具描述 (优化点：强化了对中文匹配的指令)
tools_definition = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql_query",
            "description": "当用户询问关于学生、班级或成绩的数据时，使用此函数执行 SQL 查询。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": f"""
                        根据以下数据库架构编写的 MySQL 查询语句:
                        {database_schema_string}
                        
                        【重要规则】:
                        1. 数据库中的 'subject' (科目) 字段存储的是**中文** (如 '数学', '英语')，**严禁**使用英文 (如 'Math', 'English')。
                        2. 数据库中的 'class_name' (班级) 字段存储的是**中文** (如 '一班')。
                        3. 只返回纯 SQL 文本，不要包含 Markdown 格式或解释。
                        4. 确保 JOIN 条件正确。
                        """
                    }
                },
                "required": ["sql_query"]
            }
        }
    }
]

def run_agent(user_question):
    messages = [
        {"role": "system", "content": "你是一个智能 SQL 助手。注意：数据库中的所有文本字段（如姓名、班级、科目）均使用**简体中文**存储。生成 SQL 时必须使用中文进行匹配。"},
        {"role": "user", "content": user_question}
    ]

    # 第一轮：让 LLM 决定是否需要调用工具
    response = client.chat.completions.create(
        model="qwen2.5:7b",
        messages=messages,
        tools=tools_definition,
        tool_choice="auto"
    )

    assistant_message = response.choices[0].message
    
    if assistant_message.tool_calls:
        print(f"🤖 LLM 决定调用工具: {assistant_message.tool_calls[0].function.name}")
        
        tool_call = assistant_message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        if function_name == "execute_sql_query":
            sql_query = function_args.get("sql_query")
            print(f"🔍 生成的 SQL: {sql_query}")
            
            observation = execute_sql_query(sql_query)
            print(f"📊 数据库返回结果: {observation}") 
            
            messages.append(assistant_message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": observation
            })
            
            # 第二轮：让 LLM 根据工具结果生成最终回答
            final_response = client.chat.completions.create(
                model="qwen2.5:7b",
                messages=messages
            )
            
            return final_response.choices[0].message.content
    else:
        return assistant_message.content

# --- 主程序 ---
if __name__ == "__main__":
    question = "查询一班的学生数学成绩是多少？"
    print(f"❓ 用户问题: {question}\n")
    
    try:
        answer = run_agent(question)
        print("\n" + "="*30)
        print(f"✅ 最终回答: {answer}")
    except Exception as e:
        print(f"❌ 发生错误: {e}")