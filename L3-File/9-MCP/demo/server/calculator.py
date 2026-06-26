from mcp.server.fastmcp import FastMCP


mcp = FastMCP("计算器")

@mcp.tool()
def calculate(expression: str) -> float:
    """
    计算四则运算表达式
    参数：
        - expression: 数学表达式字符串，如"1 + 2 * (8 - 3)"
    返回：
        - 计算结果，浮点数
    """
    try:
        # 直接使用eval计算表达式
        # PS.在实际生产环境中，应安全地评估表达式
        result = eval(expression)
        return float(result)
    except Exception as e:
        return ValueError(f"计算错误：{str(e)}")

if __name__ == "__main__":
    mcp.run(transport='stdio')

