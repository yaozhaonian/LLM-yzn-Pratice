from mcp.server.fastmcp import FastMCP
from ddgs import DDGS

mcp = FastMCP("网页搜索")



@mcp.tool()
def web_search(query: str, max_results: int = 3) -> str:
    """
    执行网页搜索
    参数：
        - query: 搜索关键词
        - max_results: 最大返回结果数
    
    返回：
        搜索结果摘要
    """
    try:
        # 使用DDGS搜索
        with DDGS() as ddgs:
            results = []  # 初始化结果列表
            # 【修复】遍历搜索结果
            for r in ddgs.text(query, max_results=max_results, region="cn-zh"):
                result_text = []
                if 'title' in r:
                    result_text.append(f"标题: {r['title']}")
                if 'href' in r:
                    result_text.append(f"链接: {r['href']}")
                if 'body' in r:
                    result_text.append(f"内容: {r['body']}")
                
                # 【关键修复】将当前条目的文本加入 results 列表
                if result_text:  # 确保有内容才添加
                    results.append("\n".join(result_text))
                
            # 如果列表为空，说明没搜到或处理失败
            if not results:
                return "没有找到相关结果。"
            
            # 【修复】返回所有结果的拼接字符串，用换行符分隔不同条目
            return "\n\n".join(results)
        
    except Exception as e:
        return f"搜索出错：{str(e)}"

"""
Stdio 方式：通过标准输入（stdin）和标准输出（stdout）实现双向通信。特点是 Client 启动 Server 子进程，且 Server 进程只能与启动它的 MCP Client 通信，适用于本地快速集成测试场景。
"""
if __name__ == "__main__":
    mcp.run(transport='stdio')
    