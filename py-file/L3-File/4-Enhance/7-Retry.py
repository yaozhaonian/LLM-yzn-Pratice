"""
在实际的 AI 应用开发中，外部依赖（如数据库、API）可能会因为网络波动、锁竞争或临时故障而失败。
通过内置的重试机制，可以显著提高系统的健壮性和成功率，而无需在节点函数内部编写复杂的 try-except 循环。
"""
"""
LangGraph Text-to-SQL 自我修正循环示例
结合元数据感知与错误反馈机制，提高复杂 SQL 生成的鲁棒性。
"""

from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_community.utilities import SQLDatabase
import re

# ----------------------
# 配置项
# ----------------------
MAX_RETRIES = 3
DB_URI = "sqlite:///:memory:"   # 用来生成临时数据库
LLM_MODEL = "qwen2.5:7b"
LLM_BASE_URL = "http://127.0.0.1:11434/v1"

# ----------------------
# 1. 初始化环境
# ----------------------
db = SQLDatabase.from_uri(DB_URI)
db.run("""CREATE TABLE Artist (artist_id INTEGER PRIMARY KEY, Name NVARCHAR(120));""")
db.run("""CREATE TABLE Album (album_id INTEGER PRIMARY KEY, title NVARCHAR(160), artist_id INTEGER, FOREIGN KEY (artist_id) REFERENCES Artist(artist_id));""")
db.run("INSERT INTO Artist (Name) VALUES ('Louis Armstrong'), ('Duke Ellington');")
db.run("INSERT INTO Album (title, artist_id) VALUES ('Satchmo', 1), ('Ellington Indigos', 2), ('浮夸', 1);")

ds_llm = ChatOpenAI(
    model_name=LLM_MODEL,
    base_url=LLM_BASE_URL,
    api_key="ollama",
    temperature=0,
    timeout=180,
    max_retries=0 
)

# ----------------------
# 2. 定义状态
# ----------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    schema_info: str
    sql_query: str
    error_message: str
    retry_count: int
    final_answer: str  # 新增：存储最终格式化后的答案

# ----------------------
# 3. 工具函数
# ----------------------
def get_schema():
    return db.get_table_info()

def clean_sql(raw_content: str) -> str:
    """更健壮的 SQL 清洗，处理思考过程"""
    if not raw_content:
        return ""
    
    # 1. 尝试提取 ```sql ... ``` 块
    match = re.search(r'```sql\s*(.*?)\s*```', raw_content, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1).strip()
    else:
        # 如果没有 markdown 块，尝试去除可能的思考标签 (如 <think>...</think>)
        sql = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL | re.IGNORECASE).strip()
        # 去除其他非 SQL 文本（简单策略：取最后一行看起来像 SQL 的部分）
        # 这里简化处理，假设剩余内容主要是 SQL
        sql = sql.strip()

    # 确保以分号结尾
    if sql and not sql.endswith(';'):
        sql += ';'
    
    return sql

# ----------------------
# 4. 节点定义
# ----------------------

def prepare_context(state: AgentState):
    schema = get_schema()
    print(f"[Context] Schema Loaded.")
    return {
        "schema_info": schema,
        "retry_count": 0,
        "error_message": "",
        "sql_query": "",
        "final_answer": ""
    }

def generate_sql(state: AgentState):
    schema = state["schema_info"]
    user_question = state["messages"][0].content
    error_msg = state.get("error_message", "")
    retry_count = state.get("retry_count", 0)
    
    print(f"\n--- [Generate SQL] Attempt {retry_count + 1}/{MAX_RETRIES} ---")
    
    # 【核心优化】重新格式化 Schema，使其更易读，减少歧义
    # 原始 schema 可能包含 CREATE TABLE 语法，LLM 容易忽略细节
    # 我们手动构建一个更清晰的视图（在实际生产中可以用工具解析 schema）
    # 这里为了演示，我们假设 schema 字符串已经包含表信息，我们通过 Prompt 强调
    
    system_prompt = f"""
    你是一个严格的 SQLite SQL 生成专家。
    
    可用数据库结构 (请逐字匹配表名和列名):
    {schema}
    
    ⚠️ 绝对禁止的行为:
    1. 严禁使用 "artist_name", "album_name", "AlbumID" 等常见但不在 Schema 中的列名。
    2. 严禁在 SELECT 子句中使用未在 FROM/JOIN 中定义的表别名。
    3. 必须确保 SELECT 中的列名完全存在于对应的表中。
    
    🧠 思考步骤 (必须在生成 SQL 前执行):
    1. 确定涉及的表名 (检查 Schema)。
    2. 确定涉及的列名 (检查 Schema，逐字匹配)。
    3. 检查 JOIN 条件是否正确。
    4. 生成最终 SQL。
    
    输出格式要求:
    请先输出简短的思考过程 (Thinking)，然后输出 SQL 代码块。
    例如:
    Thinking: 需要 Artist 表的 Name 列和 Album 表的计数。关联键是 artist_id。
    ```sql
    SELECT ...
    ```
    """
    
    if retry_count == 0:
        user_content = f"问题: {user_question}"
    else:
        # 【增强反馈】明确指出之前的具体错误点
        hint = ""
        if "no such table" in error_msg:
            # 提取出错的表名
            match = re.search(r"no such table: (\w+)", error_msg)
            wrong_table = match.group(1) if match else "未知表"
            hint = f"\n⚠️ 严重错误: 表 '{wrong_table}' 不存在。请严格参照 Schema 中的表名。"
        elif "no such column" in error_msg:
            # 提取出错的列名
            match = re.search(r"no such column: ([\w\.]+)", error_msg)
            wrong_col = match.group(1) if match else "未知列"
            hint = f"\n⚠️ 严重错误: 列 '{wrong_col}' 不存在。请检查该列是否属于正确的表，并严格参照 Schema。"
        
        user_content = f"""
        问题: {user_question}
        
        上一次执行失败，错误信息:
        "{error_msg}"
        
        {hint}
        
        请重新思考并修正 SQL。
        """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]
    
    try:
        response = ds_llm.invoke(messages)
        raw_content = response.content
        print(f"[LLM Raw Output]:\n{raw_content[:200]}...") # 打印部分原始输出以调试
        
        sql = clean_sql(raw_content)
        print(f"[LLM] Final SQL: {sql}")
        
        return {
            "sql_query": sql,
            "messages": [response],
            "error_message": "" 
        }
    except Exception as e:
        print(f"[LLM Error] {str(e)}")
        return {
            "sql_query": "",
            "error_message": f"LLM Generation Failed: {str(e)}",
            "retry_count": retry_count + 1
        }

def execute_sql(state: AgentState):
    sql = state["sql_query"]
    retry_count = state["retry_count"]
    
    # 如果 SQL 为空，说明生成阶段就失败了，直接返回错误，不再次增加计数（因为生成阶段已加）
    if not sql:
        print("[Execute] Skipped: Empty SQL from generator.")
        return {
            "error_message": state.get("error_message", "Empty SQL"),
            # 保持 retry_count 不变，因为生成节点已经增加了
            "retry_count": retry_count 
        }

    print(f"[Execute] Running: {sql}")
    
    try:
        result = db.run(sql)
        print(f"[Execute] Success: {result}")
        return {
            "messages": [AIMessage(content=str(result))],
            "error_message": "",
            "retry_count": retry_count,
            "sql_query": sql # 保留成功的 SQL
        }
    except Exception as e:
        error_str = str(e)
        print(f"[Execute] Failed: {error_str}")
        return {
            "error_message": error_str,
            "retry_count": retry_count + 1 # 执行失败，增加计数
        }

def format_result(state: AgentState):
    """可选节点：格式化最终结果"""
    last_msg = state["messages"][-1]
    raw_result = last_msg.content
    
    # 简单示例：将 Python 元组列表转换为更易读的形式
    # 在实际生产中，可以使用 pandas 或更复杂的解析
    try:
        # 尝试评估为 Python 对象（注意安全，仅用于可信内部环境）
        # 这里仅作演示，实际建议用 csv/json 解析
        formatted = f"查询结果:\n{raw_result}"
    except:
        formatted = raw_result
        
    return {"final_answer": formatted}

# ----------------------
# 5. 路由逻辑
# ----------------------

def should_retry(state: AgentState):
    # 1. 如果没有错误，成功
    if not state.get("error_message"):
        return "format_result" # 走向格式化节点
    
    # 2. 检查重试次数
    if state["retry_count"] >= MAX_RETRIES:
        print("[Router] Max retries reached. Ending.")
        return END
        
    # 3. 需要重试
    print(f"[Router] Retrying... ({state['retry_count']}/{MAX_RETRIES})")
    return "generate_sql"

# ----------------------
# 6. 构建图
# ----------------------

builder = StateGraph(AgentState)

builder.add_node("prepare_context", prepare_context)
builder.add_node("generate_sql", generate_sql)
builder.add_node("execute_sql", execute_sql)
builder.add_node("format_result", format_result)

builder.add_edge(START, "prepare_context")
builder.add_edge("prepare_context", "generate_sql")
builder.add_edge("generate_sql", "execute_sql")

# 条件边：执行后决定去向
builder.add_conditional_edges(
    "execute_sql",
    should_retry,
    ["generate_sql", "format_result", END]
)

# 格式化后结束
builder.add_edge("format_result", END)

graph = builder.compile()

# ----------------------
# 7. 测试
# ----------------------

print("--- 测试: 复杂查询 ---")
try:
    result = graph.invoke({
        "messages": [HumanMessage(content="查询每个艺术家有多少张专辑，返回艺术家名字和专辑数量")]
    })
    
    if result.get("final_answer"):
        print(f"✅ 最终答案:\n{result['final_answer']}")
    elif result.get("error_message"):
        print(f"❌ 失败: {result['error_message']}")
    else:
        print(f"⚠️ 未知状态: {result}")
        
except Exception as e:
    print(f"❌ 系统异常: {e}")


"""
import operator
import sqlite3
from typing import Annotated, Sequence
import re

from langgraph.types import RetryPolicy
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
# RetryPolicy()

db = SQLDatabase.from_uri("sqlite:///:memory:")
# 创建表
db.run("CREATE TABLE Artist (ArtistId INTEGER PRIMARY KEY, Name NVARCHAR(120));")
# 表中添加数据
db.run("\""CREATE TABLE Artist (artist_id INTEGER PRIMARY KEY, Name NVARCHAR(120));"\"")
db.run("\""CREATE TABLE Album (album_id INTEGER PRIMARY KEY, title NVARCHAR(160), artist_id INTEGER, FOREIGN KEY (artist_id) REFERENCES Artist(artist_id));"\"")
db.run("INSERT INTO Artist (Name) VALUES ('Louis Armstrong'), ('Duke Ellington');")
db.run("INSERT INTO Album (title, artist_id) VALUES ('Satchmo', 1), ('Ellington Indigos', 2), ('浮夸', 1);")


model = ChatOpenAI(
    model_name="deepseek-v3",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("API_BASE_URL")
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def query_database(state):
    # 提取纯SQL查询，去除Markdown标记
    raw_sql = state['messages'][1].content
    # 使用正则表达式提取SQL语句
    sql_match = re.search(r'(SELECT.*?;)', raw_sql, re.DOTALL | re.IGNORECASE)
    if sql_match:
        clean_sql = sql_match.group(1)
    else:
        # 如果没有找到明确的SQL语句，使用原始内容但去除常见的标记
        clean_sql = re.sub(r'```.*?\n', '', raw_sql).strip()
        clean_sql = re.sub(r'```', '', clean_sql).strip()
    
    print('执行的SQL:', clean_sql)
    query_result = db.run(clean_sql)
    print('query_result:',query_result)
    return {"messages": [AIMessage(content=query_result)]}


def call_model(state):
    response = model.invoke(state["messages"])
    print('response:',response)
    return {"messages": [response]}

builder = StateGraph(AgentState)
builder.add_node("query_database",query_database,retry=RetryPolicy(retry_on=sqlite3.OperationalError))
builder.add_node("model", call_model, retry=RetryPolicy(max_attempts=5))

builder.add_edge(START, "model")
builder.add_edge("model", "query_database")
builder.add_edge("query_database", END)
graph = builder.compile()

from IPython.display import display, Image
try:
    display(Image(graph.get_graph().draw_png(output_file_path='../imgs/示例9.png')))
except:
    pass

result= graph.invoke({"messages": [HumanMessage(content="查询Artist表格中前十位艺术家，只返回SQL查询语句，不要返回其他内容，特别是```sql\n不要出现")]})
print('result：',result)

"""

