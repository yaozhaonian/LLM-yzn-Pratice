# 从huggingface 加载数据集

from datasets import load_dataset
import pandas as pd

# # 1) 下载 ChnSentiCorp（中文小数据集）
# dataset = load_dataset("lansinuote/ChnSentiCorp")
# # 或中文QA：
# # dataset = load_dataset("zhouzhouyang/ChineseSimpleQA")

# # 查看结构
# print(dataset)
# # 一般是: dataset['train'], dataset['test']

# # 看几条数据
# for i in range(10):
#     print(dataset['train'][i])

# 加载多路召回专用测试集（自动下载，本地缓存）
# dataset = load_dataset("C-MTEB/DuRetrieval")#64.4M

# print("缓存路径:\n",dataset.cache_files)
# print("查看数据结构:\n",dataset)

# corpus = dataset["corpus"]       # 知识库：id + text
# queries = dataset["queries"]     # 问题：id + text


# print("知识库条数:", len(corpus),"\n类型:", type(corpus), "\n查看知识库第一条:", corpus[0], "\n")
# print("查询条数:", len(queries))
# # 看一条
# print("\n查询示例:", queries[0])


#["train"]["features"]
# ---------------------- 核心配置 ----------------------
# 加载 【中文多跳问答HotpotQA】（100%可下载，无报错）
# 专门用于：复杂问题分解 + 多步推理COT
# ds = load_dataset("hotpot_qa", 'distractor')

# print("可用拆分 keys:", ds.keys()) # 打印出来确认有哪些拆分，通常是 dict_keys(['train', 'validation'])
# print("缓存路径:\n", ds.cache_files)
# print("查看数据结构:\n", ds)

# # 【修改点】使用 "validation" 而不是 "test"
# if "validation" in ds:
#     split_name = "validation"
# elif "test" in ds:
#     split_name = "test"
# else:
#     raise KeyError("数据集中既没有 validation 也没有 test 拆分！")

# print(f"正在使用拆分: {split_name}")

# # 转换为 Pandas DataFrame
# df = ds[split_name].to_pandas()

# print(f"原始数据条数: {len(df)}")

# # 固定随机抽样 200 条（如果数据不足200条，会报错，所以加个判断）
# n_samples = min(200, len(df))
# if n_samples < 200:
#     print(f"警告：数据只有 {len(df)} 条，无法抽取200条，将抽取全部数据。")

# sample_200 = df.sample(n=n_samples, random_state=42)

# # HotpotQA 的列名通常是: 
# # 'question', 'answer', 'context' (context是一个嵌套列表，包含title和sentences)
# # 检查列是否存在
# required_columns = ["question", "answer", "context"]
# for col in required_columns:
#     if col not in sample_200.columns:
#         print(f"错误：列 '{col}' 不存在。当前列有: {sample_200.columns.tolist()}")
#         # 如果是英文 HotpotQA，列名可能不同，或者 context 结构复杂，需要特殊处理
#         # 这里假设标准列名存在

# # 精简列：只保留 问题+上下文+答案
# sample_clean = sample_200[required_columns]

# # 导出 CSV
# output_filename = "cot_200_hotpotqa_en.csv" # 注意：HotpotQA是英文数据集，文件名建议改一下
# sample_clean.to_csv(output_filename, index=False, encoding="utf-8-sig")

# # ---------------------- 输出验证 ----------------------
# print(f"✅ 导出成功！文件：{output_filename}")
# print("\n📌 测试数据示例（第1条）：")
# print("复杂问题：", sample_clean.iloc[0]["question"])
# # context 在 HotpotQA 中是一个列表，直接打印前100字符可能会乱，这里简单处理
# ctx = str(sample_clean.iloc[0]["context"])
# print("知识库上下文（预览）：", ctx[:100], "...")
# print("标准答案：", sample_clean.iloc[0]["answer"])

# ds = load_dataset("nbraham/system-prompts-task-decomposition-v1")
# ds = load_dataset("vibrantlabsai/WikiEval")
original_column_names = [
    "question", 
    "answer", 
    "context_v1", 
    "context_v2", 
    "ungrounded_answer", 
    "source", 
    "poor_answer"
]

ds = load_dataset(
    "csv", 
    data_files="hf://datasets/InfiniFlow/medical_QA/*.csv",
    column_names=original_column_names, # 这里必须包含CSV中所有的列，按顺序
    header=None, # 修改点：使用 None 而不是 False
    split="train"
)  # 医疗相关

print("缓存路径:\n", ds.cache_files) 
print("查看数据结构:\n", ds)
# 取测试集
df = ds.to_pandas()
df_small = df.head(500)
print("当前列名:", df_small.columns.tolist())
# 抽取 200 条（固定随机，可复现）
# sample_200 = df.sample(n=10, random_state=10)

# if "train" in ds:
#     split_name = "train"
# elif "test" in ds:
#     split_name = "test"
# else:
#     raise KeyError("数据集中既没有 validation 也没有 test 拆分！")

# 清理列：问题 + 思维链（分解步骤） + 答案
cols_to_keep = ['question', 'answer']

# 导出 CSV（Excel 直接打开不乱码）
existing_cols = [c for c in cols_to_keep if c in df_small.columns]
sample_clean = df_small[existing_cols]
# ---------------------- 输出预览 ----------------------
# 导出
sample_clean.to_csv("medical_QA_500.csv", index=False, encoding="utf-8-sig")
# print("✅ 成功导出")
print("\n📌 第一条示例：", sample_clean.iloc[0])


