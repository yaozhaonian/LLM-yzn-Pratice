# stdio_server.py
import sys
import os
import asyncio
import traceback
import logging

# 1. 设置日志文件
log_file_path = "mcp_server_debug.log"
log_file = open(log_file_path, "w", encoding="utf-8")

# 2. 重定向 stderr 到日志文件
sys.stderr = log_file

# 3. 关键：配置根 logger，使其不再向 stdout 输出，而是输出到日志文件
# 这样可以捕获 utils.logger 或其他库产生的日志
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# 移除默认的 handler (如果存在)
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# 添加文件 handler
file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# 同时也重定向 print 语句，防止漏网之鱼
def silent_print(*args, **kwargs):
    # 将 print 的内容写入日志文件而不是 stdout
    message = " ".join(str(a) for a in args)
    log_file.write(f"PRINT: {message}\n")
    log_file.flush()

# 注意：不要重定向 sys.stdout 到文件，因为 MCP 库需要使用 sys.stdout 发送 JSON-RPC 消息
# 我们只重定向 stderr 和 logging，并覆盖 print

async def run_stdio():
    try:
        # 记录启动信息
        root_logger.info("Starting Stdio Server...")
        
        # 导入主服务
        from integration_in_client_mcp_server import mcp, analyzer
        root_logger.info("Import successful.")
        root_logger.info(f"Analyzer initialized: {analyzer is not None}")

        from mcp.server.stdio import stdio_server
        
        root_logger.info("Entering stdio_server context...")
        async with stdio_server() as (read_stream, write_stream):
            root_logger.info("Stdio streams created. Running server...")
            await mcp.server.run(read_stream, write_stream, mcp.server.create_initialization_options())
            root_logger.info("Server run finished.")
            
    except Exception as e:
        root_logger.error(f"CRITICAL ERROR: {str(e)}")
        root_logger.error(traceback.format_exc())
        raise
    finally:
        root_logger.info("Server shutting down.")
        log_file.close()

if __name__ == "__main__":
    try:
        asyncio.run(run_stdio())
    except Exception as e:
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"Startup Error: {str(e)}\n")
            f.write(traceback.format_exc())