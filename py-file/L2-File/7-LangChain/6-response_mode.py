"""
LCEL:Langchain表达式(LangChain Expression Language)
stream: 流式返回响应的块
invoke: 接受输⼊返回输出
batch: 接受批量输⼊返回输出列表
"""
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2.5:7b",temperature=0.9)

from langchain_core.prompts import PromptTemplate
chatprompt = PromptTemplate.from_template("""
用5句话介绍{topic}
""")

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

# stream: 流式返回响应的块
chain_stream = RunnableSequence(chatprompt, llm, parser)

# topic = "量子计算"
# print(f"开始生成【{topic}】介绍:")
# for chunk in chain_stream.stream({"topic": topic}):
#     print(chunk, end="", flush=True)

# batch: 接受批量输⼊返回输出列表
"""
介绍{topic}的起源、定义、特点、作用以及未来前景
"""
chatprompt2 = PromptTemplate.from_template("""
用3句话介绍{topic}
""")
chain = RunnableSequence(chatprompt2, llm, parser)

# 准备批量输入
topics = ["人工智能", "区块链", "量子计算", "基因编辑", "低空经济", "三生教育", "青藏高原", "广东广州", "台风", "Apifox", "五年规划"]
inputs = [{"topic": topic} for topic in topics]
# 单次调用计时
import time
start_invoke = time.time()
single_results = [chain.invoke({"topic": topic}) for topic in topics]
single_time = time.time() - start_invoke

# 注意事项
# 输入列表中的所有字典必须有相同的键结构
# 批量处理不适合有状态的操作（如带记忆的对话链）
start_batch = time.time()
batch_results = chain.batch(inputs)
batch_time = time.time() - start_batch

# 结果对比
print(f"\n=== 单次调用耗时: {single_time:.2f}s ===")
for i, res in enumerate(single_results):
    print(f"{topics[i]}: {res}")

print(f"\n=== 批量调用耗时: {batch_time:.2f}s (加速 {single_time/batch_time:.1f}x) ===")
for i, res in enumerate(batch_results):
    print(f"{topics[i]}: {res}")

# 结论：在用本地模型的情况下，任务越复杂、并行数量越少批量调用反而越慢，并行数量多才速度更快