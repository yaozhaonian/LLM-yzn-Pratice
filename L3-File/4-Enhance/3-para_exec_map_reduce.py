# 基于 LangGraph 构建 MapReduce 模式工作流
"""
工作流主要分为四个阶段：
主题分解 (Map) -> 并行执行 (Parallel Execution) -> 结果汇聚 (Reduce/Collect) -> 最终决策 (Final Selection)。
"""
import re
from typing_extensions import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from langgraph.types import Send
from pydantic import BaseModel, Field   # 使用更明确的结构化输出定义

"""
生成与{topic}相关的1到3个示例的逗号分隔列表。
生成一个关于{subject}的笑话
下面是一些关于{topic}的笑话。选择最好的一个！返回最佳的ID。
{jokes}
"""
# deepseek-r1:8b
llm = ChatOpenAI(
    model_name="qwen2.5:7b",
    api_key="ollama",
    base_url="http://127.0.0.1:11434/v1"
)

subjects_prompt = """作为主题专家，请为{topic}生成1-3个最相关的子主题，以逗号分隔。只返回子主题列表。"""
joke_prompt = """作为专业喜剧编剧，请创作一个关于{subject}的高质量笑话。要求：
1. 简洁幽默
2. 适合大众
3. 长度不超过5句话"""
best_joke_prompt = """作为幽默感评委，请从以下关于{topic}的笑话中选出最佳作品。评判标准：
1. 创意性(40%)
2. 幽默感(40%) 
3. 语言表达(20%)

{jokes}

请只返回最佳笑话的编号(从0开始)。"""


# 数据模型定义
class Subject(BaseModel):
    subjects: str = Field(..., description="主题")
    

class Joke(BaseModel):
    joke: str = Field(description="笑话")

class BestJoke(BaseModel):
    id: int = Field(description="最佳笑话的索引号，从0开始")


# 状态定义
class OverallState(TypedDict):
    topic: str
    subjects: list
    jokes: Annotated[list, add]
    best_selected_joke: str
    best_joke_id: int

class JokeState(TypedDict):
    subject: str

def generate_topics(state: OverallState):
    """生成相关子主题"""
    print("正在生成相关子主题...")
    prompt = subjects_prompt.format(topic=state["topic"])
    print("generate_topics提示词:", prompt)
    response = llm.invoke(prompt)
    
    # 1. 先获取原始内容
    content = response.content
    
    # 2. 移除 DeepSeek R1 等模型的 <think> 标签内容
    if "<think>" in content and "</think>" in content:
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
    # 3. 解析子主题
    subjects = [s.strip() for s in content.split(",") if s.strip()]
    
    # 4. 过滤掉可能的空字符串或非主题内容
    subjects = [s for s in subjects if len(s) > 1] 
    print("subjects值:", subjects)
    
    return {"subjects": subjects[:3]}

def generate_joke(state: JokeState):
    """为每个子主题生成笑话"""
    print("为每个子主题生成笑话...")
    prompt = joke_prompt.format(subject=state["subject"])
    print("generate_joke提示词:", prompt)
    response = llm.invoke(prompt)
    print("笑话回复:", response.content)
    return {"jokes": [response.content]}

def continue_to_jokes(state: OverallState):
    """路由决定下一步"""
    print("路由决定下一步...", state["subjects"])
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

def best_joke(state: OverallState):
    """选择最佳笑话"""
    print("选择最佳笑话...")
    jokes = "\n\n".join([f"ID {i}: {j}" for i, j in enumerate(state["jokes"])]) # 明确标出 ID
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    print("best_joke提示词:", prompt)
    
    try:
        response = llm.invoke(prompt)
        content = response.content
        print("选择最佳笑话中(1):\n", prompt)
        
        # 移除思考过程
        if "<think>" in content and "</think>" in content:
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        # 尝试提取数字
        best_id = 0
        # 查找第一个出现的独立数字
        match = re.search(r'\b(\d+)\b', content)
        if match:
            best_id = int(match.group(1))
        
        # 确保ID在有效范围内
        if not state["jokes"]: # 防止空列表
             return {"best_selected_joke": "", "best_joke_id": -1}
             
        best_id = max(0, min(best_id, len(state["jokes"]) - 1))
        best_joke = state["jokes"][best_id]
        print("选择最佳笑话中(2):\nid:", best_id, "\njoke:", best_joke)
        return {
            "best_selected_joke": best_joke, 
            "best_joke_id": best_id
            }
    except Exception as e:
        print(f"选择最佳笑话时出错: {e}")
        # 默认返回第一个，防止崩溃
        if state["jokes"]:
            return {"best_selected_joke": state["jokes"][0], "best_joke_id": 0}
        else:
            return {"best_selected_joke": "", "best_joke_id": -1}


graph = StateGraph(OverallState)
graph.add_node("generate_topics", generate_topics)
graph.add_node("generate_joke", generate_joke)
graph.add_node("best_joke", best_joke)

graph.add_edge(START, "generate_topics")
graph.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
graph.add_edge("generate_joke", "best_joke")
graph.add_edge("best_joke", END)
app = graph.compile()

# app.get_graph().draw_png(output_file_path="./3-map_reduce.png")

if __name__ == "__main__":
    inputs = {"topic": "小白兔"}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"{key}: {value}")
        print("-" * 20)


"""
关键技术点
Send API: 这是 LangGraph v0.2+ 中用于实现动态并行分支的核心功能。它允许根据前一个节点的输出动态决定创建多少个并行任务。
operator.add Reducer: 在 OverallState 中定义 jokes: Annotated[list, operator.add] 至关重要。它告诉 LangGraph 当多个并行节点返回结果时，如何将它们合并到全局状态中（即列表追加），而不是覆盖。
动态路由: continue_to_jokes 不是简单的条件判断（去 A 或去 B），而是生成一组任务指令，实现了“一对多”的分发。

用应用场景
这种 MapReduce 架构非常适合需要发散思维生成多个选项，然后收敛评估选出最优解的场景：

内容创作与优化:

标题生成: 输入文章主旨，并行生成 5-10 个不同风格的标题，最后由 LLM 选出最吸引人的一个。
广告文案: 针对产品的不同卖点（子主题）生成多条文案，再筛选最佳组合。
创意写作: 为故事开头生成多个情节发展方向，评估后选择一个继续展开。
代码生成与审查:

多方案代码生成: 针对一个编程问题，并行生成 Python、Java、C++ 等不同语言的实现，或者同一语言的不同算法实现，然后选择效率最高或最简洁的代码。
Bug 修复建议: 针对一个错误日志，并行生成多种可能的修复方案，并评估其安全性。
数据分析与报告:

多维度分析: 输入一个商业指标，并行生成关于市场、竞争、用户反馈等多个维度的分析摘要，最后汇总成一份综合报告。
摘要生成: 对长文档的不同章节并行生成摘要，最后合并并润色成全文摘要。
搜索与信息聚合:

多源查询: 将用户问题拆解为多个子问题，并行搜索或查询不同知识库，最后整合答案并去重/排序。
"""

