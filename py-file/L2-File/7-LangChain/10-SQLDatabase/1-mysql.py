# Langchain访问MySQL数据库
# 可以生成SQL语句，再从数据库查询数据

"""
PS.
使用提示词让大模型生成数据库语句时，在用户端尽量把数据库中的表结构提供给大模型，这样大模型才能更准确地生成SQL。
例子：
对于system(即大模型)角色
你是数据分析专家，精通MySQL。根据用户问题生成SQL查询。

核心规则：
1. 仅使用用户提到的表和字段
2. 确保SQL兼容MySQL
3. 只输出一个完整SQL语句，确保输出是一个可直接执行的SQL文本,不含注释等其它任何多余信息

注意：
- 检查表名和字段名，严格按照表结构描述写，比如用户提到单价，你需要写成单价（元）
- 字符字段用LIKE N'%关键词%'
- 所有除法用指定模板
- 根据需求选择正确的聚合函数


对于用户user提示词：

表结构描述：
表名称：fangchang
id，整数
地区 ，字符串
房价（万），小数
配置，字符串
大小（平米），小数
单价（元），整数
方位，字符串
层数，字符串
装修，字符串
其他，字符串

问题:**修改这里**

请将问题转成SQL语句

请只返回执行语句
例如： SELECT * FROM fangchang WHERE price BETWEEN 50 AND 100;
请不要有任何其他信息，特别是前面不能有sql\n,否则会受到惩罚
"""

from langchain_community.utilities import SQLDatabase

# 数据库配置
HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = '0213_pra'
USERNAME = 'root'
PASSWORD = '123456'
MYSQL_URI = f'mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}'
db = SQLDatabase.from_uri(MYSQL_URI)
# print(db.get_usable_table_names())
# print(db.run('select * from country limit 5'))

from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2.5:7b",temperature=0.9)
# template = ChatPromptTemplate(
#     [
#         ("system", "你是一个上知天文下知地理的智能聊天助手"),
#         ("placeholder", "{conversation}"),
#         ("human", "{input}")
#     ]
# )



# def get_table_names():
#     return db.get_usable_table_names()

# get_usable_table_names_tool = Tool(
#     name="获取表名",
#     description="获取数据库中所有可用的表名",
#     func=get_table_names
# )

# client_with_tools = llm.bind_tools([get_usable_table_names_tool])
# resp = client_with_tools.invoke([HumanMessage(content="请从国家表中查询出China的所有数据")])
# print(resp)
# print("**",resp.content)
# print("**",resp.tool_calls)

'''
从上面的执行结果可以看到，大模型判断出为了回答问题，需要使用工具，但是工具的使用不是大模型负责的，而应该是应用负责的。
所以在大模型使用工具回答问题的过程中，往往需要多次和大模型交互才能得到最终的结果。
针对这种情况，可以在LangChain的链调用工具
'''

# 创建一个用于生成 SQL 查询的链:https://reference.langchain.com/python/langchain-classic/chains/sql_database/query/create_sql_query_chain
from langchain_classic.chains import create_sql_query_chain
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import re

"""
对于上述问题:"请从国家表中查询出China的所有数据",要分为几步才能出结果:
1、大模型判断这个问题需要调用工具查询数据库，获得所有的表名和表中的字段名，目的是看哪个表才是国家表，国家表有哪些字段；
2、工具执行后，把其执行结果交给模型；
3、大模型根据国家表及其字段，生成SQL语句；
4、SQL语句的执行依然需要使用工具；
5、工具执行后，把工具执行结果交给大模型，大模型生成最终答案
"""


# print("先查看这个查询会返回什么\n",response,"\ntype:",type(response))
# print("="*25,"再优化返回结果","="*25)
# 返回的response结果:SQLQuery: SELECT * FROM `country` WHERE `Name` = 'China'

def parse(text: str) -> str:
    """清理 LLM 返回的 SQL，去除多余前缀和标记"""
    pattern = r'```sql(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        sql = match.group(1).strip()
        # 去除 SQLQuery 的前缀
        sql = re.sub(r'^SQLQuery:', '', sql, flags=re.MULTILINE).strip()
        return sql
    text = re.sub(r'^SQLQuery:', '', text).strip()
    return text    

# sql_make_chain2 = create_sql_query_chain(llm, db) | SQLCleaner()
# response2 = sql_make_chain.invoke({"question":"请从国家表中查询出China的所有数据"})
# print("再查看这个查询会返回什么\n",response2,"\ntype:",type(response))
# print("="*25,"看着优化返回结果","="*25)


# 创建一个执行SQL的工具
dic = {"question":"请从国家表中查询出China的所有数据"}
sql_make_chain = create_sql_query_chain(llm, db)
# response = sql_make_chain.invoke(dic)
# response_format = parse(response)
# print("格式化的SQL：", response_format)
execute_sql_tools = QuerySQLDataBaseTool(db=db)
# response_Query = execute_sql_tools.invoke({
#     "query":response_format
# })
# print('最后的查询结果:\n',response_Query)
# 做成一个langchain链
# ==================== 组装字典函数 ====================
chain = sql_make_chain | RunnableLambda(parse) | {"query":RunnablePassthrough()} | execute_sql_tools
response_chain = chain.invoke(dic)
print('response_chain的查询结果:\n',response_chain)

# 配合SQL Agent使用(官方说该方法给于智能体的权限比较大，需注意风险)
# from langchain_community.agent_toolkits import create_sql_agent

# agent = create_sql_agent(
#     llm, 
#     db=db, 
#     verbose=True
# )

# response_sa = agent.invoke("从国家表中查询出China的所有数据")
# print('response_sa:\n',response_sa) #可以把结果提取整理放在md文档中就可以得出表格数据了