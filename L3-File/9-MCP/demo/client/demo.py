from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
import os
import logging

# 配置日志
logging.getLogger("mcp").setLevel(logging.WARNING)

SERVER_CONFIGS = [
    {
        "name": "calculator",
        "params":StdioServerParameters(
            command="python",
            # 从当前脚本文件，向上两级找到项目根目录，再拼接 server 文件夹下的 calculator.py，生成该文件的完整绝对路径。
            # __file__:内置变量，代表当前 .py 脚本文件的完整绝对路径
            args=[os.path.join(os.path.dirname(os.path.dirname(__file__)), "server", "calculator.py")],
        )
    },
    {
        "name": "web_search",
        "params": StdioServerParameters(
            command="python",
            args=[os.path.join(os.path.dirname(os.path.dirname(__file__)), "server", "web_search.py")],
        )
    }
]

async def main():
    # 用来存放所有会话:key=服务名,value=ClientSession
    sessions = {}
    # 用来存放所有 stdio 连接(必须保持引用,防止被回收)
    stdio_managers = []
    
    # ========== 循环创建所有连接与会话(扁平代码,无嵌套) ==========
    for cfg in SERVER_CONFIGS:
        # 启动一个本地子进程服务，建立 RPC 会话，初始化连接，把连接缓存起来，最后遍历所有已连会话读取可用工具列表
        name = cfg["name"]
        params = cfg["params"]
        # 逐个创建 stdio 连接与会话
        # 返回值 stdio_ctx 是一个异步上下文管理器对象，内部封装了：启动子进程、绑定标准输入输出管道、关闭资源等逻辑
        stdio_ctx = stdio_client(params)
        read, write = await stdio_ctx.__aenter__()
        """
        创建MCP 协议会话上下文管理器：
            封装 JSON-RPC 消息编解码、请求响应收发、消息排队；
            sess_ctx 是会话管理器，控制 RPC 会话的创建与销毁。
        """       
        sess_ctx = ClientSession(read, write)
        """
        调用会话管理器的异步进入方法：
            在读写流基础上初始化 RPC 通信通道；
            返回可直接调用工具的操作会话实例 session；
            后续所有调用工具、查询工具都靠这个 session 对象。
        """       
        session = await sess_ctx.__aenter__()
        
        """
        执行 MCP 标准握手初始化：
            客户端与服务端交换协议版本、能力声明；
            完成连接校验，确认双方可以正常收发工具调用请求；
            不执行这一步，后面 list_tools() 会报错。
        """       
        await session.initialize()
        print(f"{name} 服务会话已初始化。")
        
        # 保存引用
        """
        缓存两个管理器对象到列表：
            stdio_ctx：子进程管道管理器（用来关闭子进程）
            sess_ctx：RPC 会话管理器（用来关闭 RPC 连接）
            作用：程序退出时统一遍历这个列表，逐个释放进程和会话，防止资源泄漏。
        """       
        stdio_managers.append((stdio_ctx, sess_ctx))
        sessions[name] = session
        
    # ======= 正常使用所有会话(和原来逻辑一致) =======
    print("===== 列出所有工具 =====")
    for name, sess in sessions.items():
        # list_tools() 是 MCP 标准接口，异步请求服务端导出的所有工具
        tools = await sess.list_tools()
        print(f"\n【{name} 工具列表】")
        for tool in tools.tools:
            print(f"- {tool.name}: {tool.description}")\
                    
    while True:
        print("\n可选服务：calculator / web_search / q 退出")
        choice = input("请输入服务名：").strip()
        if choice == "q":
            break
        if choice not in sessions:
            print("无效服务名，请重试")
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
                
        # ========== 统一释放所有资源(关闭会话、子进程) ==========
        for stdio_ctx, sess_ctx in reversed(stdio_managers):
            # 关闭 MCP RPC 会话，停止消息读写、释放协议层资源；
            await sess_ctx.__aexit__(None, None, None)
            # 关闭 stdio 管道、终止底层子进程（calculator.py 这类服务程序）；
            await stdio_ctx.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())


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