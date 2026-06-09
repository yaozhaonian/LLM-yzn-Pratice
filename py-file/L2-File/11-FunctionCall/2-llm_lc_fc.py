import gradio as gr
import os
import webbrowser
import json
import random
import datetime
import subprocess
from typing import List, Dict, Any

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# ======================
# 1. 定义工具 (使用 @tool 装饰器，LangChain 标准写法),相当于Agent了
# ======================

@tool
def get_date() -> str:
    """获取当前日期，格式为 YYYY-MM-DD"""
    return datetime.date.today().strftime("%Y-%m-%d")

@tool
def open_calc() -> str:
    """打开本地计算器应用程序"""
    try:
        # Windows
        subprocess.Popen(['calc.exe'])
        # Mac 取消注释下面这行
        # subprocess.Popen(['open', '-a', 'Calculator'])
        return "计算器已成功打开"
    except Exception as e:
        return f"打开计算器失败: {str(e)}"

@tool
def open_browser(url: str) -> str:
    """
    打开浏览器访问指定网址
    
    Args:
        url: 要访问的完整网址 (例如: https://www.taobao.com)
    """
    try:
        webbrowser.open(url)
        return f"已成功在默认浏览器中打开 {url}"
    except Exception as e:
        return f"打开浏览器失败: {str(e)}"

@tool
def recom_drink(location: str = "当前位置") -> str:
    """
    推荐附近的饮料店
    
    Args:
        location: 用户所在的大致位置描述
    """
    result = f'''在{location}附近推荐以下饮品：
    1. 瑞幸咖啡 (距离200米)
    2. 蜜雪冰城 (距离200米)
    3. 茶颜悦色 (距离200米)
    另外50米处有惠民便利店，可购买矿泉水。
    '''
    return result

# 将所有工具放入列表
tools = [get_date, open_calc, open_browser, recom_drink]

# ======================
# 2. 初始化 LLM 并绑定工具
# ======================

# 注意：确保你的 Ollama 中拉取了 qwen2.5:7b 或更高版本
# qwen2.5 对 function call 支持较好
llm = ChatOllama(model="qwen2.5:7b", temperature=0.5)

# 关键步骤：将工具绑定到 LLM
# 这会让 LLM 知道它有哪些工具可用，并在需要时返回 tool_calls
llm_with_tools = llm.bind_tools(tools)

# ======================
# 3. 构建 Prompt
# ======================

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的AI助手。你可以使用提供的工具来帮助用户完成任务。如果工具返回了结果，请根据结果回答用户的问题。"),
    ("user", "{input}"),
    # MessagesPlaceholder 用于在后续对话中插入历史消息和工具调用结果
    MessagesPlaceholder(variable_name="agent_scratchpad", optional=True), 
])

# ======================
# 4. 核心处理逻辑 (Agent Loop)
# ======================

def run_agent(user_input: str, history: List[List[str]]) -> List[List[str]]:
    """
    运行 Agent 逻辑
    :param user_input: 用户当前输入
    :param history: Gradio 聊天历史记录 [[user_msg, ai_msg], ...]
    :return: 更新后的历史记录
    """
    
    # 1. 构建当前对话的消息列表
    # 我们需要将 Gradio 的历史记录转换为 LangChain 的 Message 对象
    messages = []
    for h in history:
        if h[0]: messages.append(HumanMessage(content=h[0]))
        if h[1]: messages.append(AIMessage(content=h[1]))
    
    # 添加当前用户输入
    messages.append(HumanMessage(content=user_input))
    
    # 2. 第一次调用 LLM
    try:
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        return history + [[user_input, f"LLM 调用错误: {str(e)}"]]

    # 3. 检查是否有工具调用
    if hasattr(response, 'tool_calls') and response.tool_calls:
        # 有工具调用
        print("有工具调用")
        tool_messages = []
        tool_results_str = ""
        
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            # 查找对应的工具函数
            selected_tool = {t.name: t for t in tools}.get(tool_name)
            
            if selected_tool:
                try:
                    # 执行工具
                    observation = selected_tool.invoke(tool_args)
                    tool_messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call['id']))
                    tool_results_str += f"\n[调用工具: {tool_name}, 参数: {tool_args}, 结果: {observation}]"
                except Exception as e:
                    error_msg = f"工具 {tool_name} 执行出错: {str(e)}"
                    tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call['id']))
                    tool_results_str += f"\n{error_msg}"
            else:
                error_msg = f"未知工具: {tool_name}"
                tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call['id']))
                tool_results_str += f"\n{error_msg}"
        
        # 4. 第二次调用 LLM (将工具结果传回给 LLM 生成最终回复)
        # 组合所有消息：历史 + 当前用户输入 + AI的工具调用请求 + 工具执行结果
        final_messages = messages + [response] + tool_messages
        
        try:
            final_response = llm_with_tools.invoke(final_messages)
            ai_reply = final_response.content
        except Exception as e:
            ai_reply = f"LLM 二次调用错误: {str(e)}"
            
        # 返回更新后的历史：用户输入 -> AI最终回复
        # 注意：Gradio 通常只显示最终的人机对话，中间的工具调用过程可以选择性显示或不显示
        # 这里我们只显示最终结果，保持界面整洁
        return history + [[user_input, ai_reply]]
        
    else:
        # 没有工具调用，直接返回 LLM 的回答
        return history + [[user_input, response.content]]

# ======================
# 5. Gradio 界面
# ======================

def process_message(message, history):
    """Gradio 回调函数"""
    if not message:
        return history
    
    # 调用 Agent 逻辑
    new_history = run_agent(message, history)
    return new_history

with gr.Blocks(title="Ollama Function Call 演示") as demo:
    gr.Markdown("# 🤖 Ollama 本地模型 Function Call 演示")
    gr.Markdown("模型: Qwen2.5-3B | 功能: 日期查询、打开计算器、打开浏览器、饮料推荐")
    
    chatbot = gr.Chatbot(
        value=[["你好", "你好！我是你的AI助手，我可以帮你查日期、开计算器、开浏览器或推荐饮料。试试问我吧！"]],
        height=500,
        label="对话记录"
    )
    
    msg = gr.Textbox(label="请输入指令", placeholder="例如：帮我打开淘宝、现在几点了、我渴了...")
    
    with gr.Row():
        submit_btn = gr.Button("发送")
        clear_btn = gr.ClearButton([msg, chatbot])

    # 绑定事件
    msg.submit(process_message, [msg, chatbot], [chatbot])
    submit_btn.click(process_message, [msg, chatbot], [chatbot])

if __name__ == '__main__':
    # 启动服务
    demo.launch(server_port=7779)