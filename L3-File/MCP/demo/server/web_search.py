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
            results = []
            for r in ddgs.text(query, max_results=max_results):
                result_text = []
                if 'title' in r:
                    result_text.append(f"标题: {r['title']}")
                if 'href' in r:
                    result_text.append(f"链接: {r['href']}")
                if 'body' in r:
                    result_text.append(f"内容: {r['body']}")
                result_text = "\n".join(result_text)
                
            if not results:
                return "没有找到相关结果。"
            
            return "\n".join(results)
        
    except Exception as e:
        return f"搜索出错：{str(e)}"
    
if __name__ == "__main__":
    mcp.run(transport='stdio')
    