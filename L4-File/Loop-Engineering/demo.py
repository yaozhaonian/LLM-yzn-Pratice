from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 代码生成模型
code_agent = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.2,
    base_url="http://127.0.0.1:11434"
)
# 评分模型（独立实例，上下文隔离）
judge_agent = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.1,  # 打分温度更低，更严谨
    base_url="http://127.0.0.1:11434"
)

TASK_GOAL = "编写可直接运行的Python冒泡排序代码，逻辑正确、注释详细规范"
PASS_SCORE = 85
MAX_LOOP = 3

context = [
    SystemMessage(content=f"任务：{TASK_GOAL}。仅输出代码与注释，不要额外文字。"),
    HumanMessage(content="开始编写代码")
]

def clean_text(txt: str) -> str:
    lines = txt.replace("```python", "").replace("```", "").splitlines()
    valid_lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(valid_lines)

def evaluate_result(output: str) -> tuple[int, str]:
    eval_prompt = f"""
满分100，三项打分：
1. 代码可运行无语法错误：40
2. 排序逻辑正确：30
3. 注释完整清晰：30

内容：{output}

格式要求：
第一行：纯数字总分
第二行：简短修改建议，禁止输出代码。
"""
    resp = judge_agent.invoke([HumanMessage(content=eval_prompt)])
    res = resp.content.strip()
    eval_lines = [line.strip() for line in res.splitlines() if line.strip()]

    try:
        score = int(eval_lines[0])
        suggestion = " ".join(eval_lines[1:]) if len(eval_lines) > 1 else "无建议"
    except:
        score = 0
        suggestion = "评测异常"
    return score, suggestion

def run_loop():
    loop_count = 0
    print("=== 启动 AI 自动循环任务 ===")

    while loop_count < MAX_LOOP:
        loop_count += 1
        print(f"\n----- 第 {loop_count} 轮执行 -----")

        exec_resp = code_agent.invoke(context)
        current_output = exec_resp.content.strip()
        clean_code = clean_text(current_output)
        print("本轮输出：\n", current_output)

        score, advice = evaluate_result(clean_code)
        print(f"本轮评分：{score} | 优化建议：{advice}")

        if score >= PASS_SCORE:
            print(f"\n✅ 任务达标！循环结束，最终结果：\n{current_output}")
            return

        context.append(AIMessage(content=current_output))
        context.append(HumanMessage(content=f"按建议修改代码：{advice}"))

    print(f"\n⚠️ 已达到最大循环次数，强制结束循环")

if __name__ == "__main__":
    run_loop()