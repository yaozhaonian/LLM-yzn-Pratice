from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
import os
import logging


# 配置日志级别
logging.getLogger("mcp").setLevel(logging.WARNING)

# 配置 MCP Server启动参数或称为服务器连接参数
# 相当于通过 python 工具运行指定的 MCP Server,并建立 stdio 连接
# calculator_params = StdioServerParameters(
#     command="python",
#     args=[os.path.join(os.path.dirname(os.path.dirname(__file__)), "server", "calculator.py")],
#     env=None
# )

# web_search_params = StdioServerParameters(
#     command="python",
#     args=[os.path.join(os.path.dirname(os.path.dirname(__file__)), "server", "web_search.py")],
#     env=None
# )

SERVER_CONFIGS = [
    {
        "name": "calculator",
        "params": StdioServerParameters(
            command="python",
            args=[os.path.join(os.path.dirname(os.path.dirname(__file__)), "server", "calculator.py")],
        )
    },
    {
        "name": "web_search",
        "params": StdioServerParameters(
            command="python",
            args=[os.path.join(os.path.dirname(os.path.dirname(__file__)), "server", "web_search.py")],
        )
    },
    # 在这里继续添加新工具，无需改动业务代码
    # {
    #     "name": "time_tool",
    #     "params": StdioServerParameters(command="python", args=[xxx])
    # }
]

# async def main():
#     print("正在启动MCP Client...")
    
#     # 建立计算器服务的stdio客户端连接
#     # 使用stdio_client建立连接，返回calc_read和calc_write两个异步函数，分别用于从子进程读取和向子进程写入数据
#     async with stdio_client(calculator_params) as (calc_read, calc_write):
#         # 创建计算器服务的客户端会话
#         async with ClientSession(calc_read, calc_write) as calc_session:
#             # 启动网页搜索服务的stdio客户端连接
#             async with stdio_client(web_search_params) as (search_read, search_write):
#                 # 创建网页搜索服务的客户端会话
#                 async with ClientSession(search_read, search_write) as search_session:
#                     # 初始化会话
#                     await calc_session.initialize()
#                     await search_session.initialize()
                    
#                     # 列出可用的工具
#                     print("\n正在列出服务器工具...")
                    
#                     print("\n 计算机工具:")
#                     calc_tools = await calc_session.list_tools()
#                     for tool in calc_tools.tools:
#                         print(f" - 可用工具: {tool.name}\n描述: {tool.description}")
                        
#                     print("\n 网页搜索工具:")
#                     search_tools = await search_session.list_tools()
#                     for tool in search_tools.tools:
#                         print(f" - 可用工具: {tool.name}\n描述: {tool.description}")
                        
#                     # 等待用户输入
#                     while True:
#                         print("\n请选择功能:")
#                         print("1. 计算器")
#                         print("2. 网页搜索")
#                         print("q. 退出")
#                         choice = input("请输入选项(1/2/q): ")
                        
#                         if choice.lower() == 'q':
#                             break
#                         elif choice == '1':
#                             print("计算器功能")
#                             expression = input("请输入要计算的表达式：")
#                             try:
#                                 result = await calc_session.call_tool("calculate", {"expression": expression})
#                                 print(f"计算结果：{result.content}")
#                             except Exception as e:
#                                 print(f"调用计算器工具出错：{str(e)}")
#                         elif choice == '2':
#                             print("网页搜索功能")
#                             query = input("请输入要搜索的查询：")
#                             try:
#                                 result = await search_session.call_tool("web_search", {"query": query})
#                                 print(f"搜索结果：\n{result.content}")
#                             except Exception as e:
#                                 print(f"调用网页搜索工具出错：{str(e)}")
#                         else:
#                             print("无效选项，请重新输入。")


async def main():
    # 用来存放所有会话：key=服务名，value=ClientSession
    sessions = {}
    # 用来存放所有 stdio 连接（必须保持引用，防止被回收）
    stdio_managers = []

    # ========== 2. 循环创建所有连接与会话（扁平代码，无嵌套） ==========
    for cfg in SERVER_CONFIGS:
        name = cfg["name"]
        params = cfg["params"]
        # 逐个创建 stdio 连接与会话
        stdio_ctx = stdio_client(params)
        read, write = await stdio_ctx.__aenter__()
        sess_ctx = ClientSession(read, write)
        session = await sess_ctx.__aenter__()
        
        # 初始化会话
        await session.initialize()
        
        # 保存引用
        stdio_managers.append((stdio_ctx, sess_ctx))
        sessions[name] = session

    # ========== 3. 正常使用所有会话（和原来逻辑一致） ==========
    print("=== 列出所有工具 ===")
    for name, sess in sessions.items():
        tools = await sess.list_tools()
        print(f"\n【{name} 工具列表】")
        for tool in tools.tools:
            print(f"- {tool.name}: {tool.description}")

    # 交互逻辑示例
    while True:
        print("\n可选服务：calculator / web_search / q 退出")
        choice = input("请输入服务名：").strip()
        if choice == "q":
            break
        if choice not in sessions:
            print("无效服务名")
            continue

        sess = sessions[choice]
        if choice == "calculator":
            expr = input("输入计算表达式：")
            res = await sess.call_tool("calculate", {"expression": expr})
            print("结果：", res.content)
        elif choice == "web_search":
            query = input("输入搜索词：")
            res = await sess.call_tool("web_search", {"query": query})
            print("搜索结果：", res.content)

    # ========== 4. 统一释放所有资源（关闭会话、子进程） ==========
    for stdio_ctx, sess_ctx in reversed(stdio_managers):
        await sess_ctx.__aexit__(None, None, None)
        await stdio_ctx.__aexit__(None, None, None)




if __name__ == "__main__":
    asyncio.run(main())


