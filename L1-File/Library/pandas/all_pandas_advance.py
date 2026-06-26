# all_pandas_advance.py
import pandas as pd
import numpy as np
# 完整打印配置
pd.set_option('display.max_rows', None)       # 显示所有行，None=不限制
pd.set_option('display.max_columns', None)    # 显示所有列
pd.set_option('display.width', None)           # 终端宽度自动拉满，不自动折行列错位
pd.set_option('display.max_colwidth', 20)      # 每一列文字最大宽度，避免超长文字挤乱表格
pd.set_option('display.unicode.east_asian_width', True)  # 修复中文对齐错乱（关键！）
pd.set_option('display.unicode.ambiguous_as_wide', True)
# ====================== 1. 构造测试数据集 ======================
df = pd.DataFrame({
    "产品": ["A", "B", "A", "C", "B", "A", "C", "B"],
    "区域": ["华东", "华南", "华东", "华北", "华南", "华北", "华北", "华东"],
    "销量": [120, 90, 150, 88, 110, 135, 95, 102],
    "价格": [29.9, 39.9, 29.9, 49.9, 39.9, 29.9, 49.9, 39.9],
    "日期": pd.date_range("2026-01-01", periods=8)
})
# 手动添加缺失值用于演示缺失值处理
df.loc[2, "销量"] = np.nan
df.loc[5, "价格"] = np.nan
print("原始数据：")
print(df)
print("-" * 60)

# ====================== 2. 高级筛选：多条件 / isin / contains / query ======================
# 2.1 多条件与&、或|
res_and = df[(df["产品"] == "A") & (df["销量"] > 100)]
print("多条件且筛选(产品A且销量>100)：")
print(res_and)

res_or = df[(df["产品"] == "A") | (df["区域"] == "华北")]
print("\n多条件或筛选(产品A或区域华北)：")
print(res_or)

# 2.2 isin 多值匹配、取反~
res_isin = df[df["产品"].isin(["A", "C"])]
print("\n筛选产品A/C：")
print(res_isin)
res_not_in = df[~df["产品"].isin(["A", "C"])]
print("\n排除产品A/C：")
print(res_not_in)

# 2.3 字符串模糊匹配
res_like = df[df["区域"].str.contains("南")]
print("\n区域包含南：")
print(res_like)

# 2.4 query表达式筛选（可读性强）
min_sale = 120
res_query = df.query("产品 == 'A' and 销量 > @min_sale")
print("\nquery筛选：")
print(res_query)
print("-" * 60)

# ====================== 3. 缺失值全套处理 ======================
print("缺失值占比：")
print(df.isnull().mean())

# 删除空值行
df_drop = df.dropna(axis=0, how="any")
print("\n删除含空值行：")
print(df_drop)

# 按列填充空值
fill_rule = {"销量": df["销量"].mean(), "价格": df["价格"].median()}
df_fill = df.fillna(fill_rule)
print("\n均值/中位数填充空值：")
print(df_fill)

# 前向填充时序数据(用前面的数据填后面的空值)
df_ffill = df.fillna(method="ffill")
print("\n前向填充：")
print(df_ffill)
print("-" * 60)

# ====================== 4. 新增列 map / apply / assign ======================
# map单列映射
df["产品等级"] = df["产品"].map({"A": "高端", "B": "普通", "C": "普通"})

# apply单列函数
def sale_level(x):
    if pd.isna(x):
        return "未知"
    return "高销" if x > 120 else "低销"
df["销量等级"] = df["销量"].apply(sale_level)

# apply行运算 axis=1
def calc_revenue(row):
    if pd.isna(row["销量"]) or pd.isna(row["价格"]):
        return np.nan
    return row["销量"] * row["价格"]
df["营收"] = df.apply(calc_revenue, axis=1)

# assign链式新增（不改动原df）
df_new = df.assign(单价翻倍=lambda x: x["价格"] * 2)
print("新增多列后数据：")
print(df_new[["产品", "销量", "价格", "营收", "产品等级"]])
print("-" * 60)

# ====================== 5. GroupBy分组聚合、filter、transform ======================
# 多聚合函数agg
group_agg = df.groupby("产品").agg(
    销量均值=("销量", "mean"),
    总销量=("销量", "sum"),
    单价中位数=("价格", "median")
).reset_index()
print("单分组聚合：")
print(group_agg)

# 双字段分组
group_double = df.groupby(["产品", "区域"])["销量"].sum().reset_index()
print("\n产品+区域双分组求和：")
print(group_double)

# filter过滤分组（只保留总销量>200的产品）
def filter_group(g):
    return g["销量"].sum() > 200
df_filter_group = df.groupby("产品").filter(filter_group)
print("\n过滤分组（总销量大于200）：")
print(df_filter_group)

# transform 分组均值匹配原表行数
df["同产品平均销量"] = df.groupby("产品")["销量"].transform("mean")
print("\ntransform分组均值新增列：")
print(df[["产品", "销量", "同产品平均销量"]])
print("-" * 60)

# ====================== 6. 表合并 merge / concat ======================
df_supplier = pd.DataFrame({"产品": ["A", "B", "C"], "供应商": ["甲", "乙", "丙"]})
# 左连接
df_merge = pd.merge(df, df_supplier, on="产品", how="left")
print("merge左连接：")
print(df_merge[["产品", "区域", "销量", "供应商"]])

# concat上下拼接
df_top3 = df.head(3)
df_concat_row = pd.concat([df, df_top3], axis=0, ignore_index=True)
print("\n上下拼接行，总行数：", len(df_concat_row))
print("-" * 60)

# ====================== 7. 透视表 pivot_table / 交叉表crosstab ======================
pivot = df.pivot_table(
    index="产品",
    columns="区域",
    values="销量",
    aggfunc="sum",
    fill_value=0
)
print("数据透视表：")
print(pivot)

cross = pd.crosstab(df["产品"], df["区域"])
print("\n交叉计数表：")
print(cross)
print("-" * 60)

# ====================== 8. 索引进阶、排序 ======================
# 设置时间索引
df_time_idx = df.set_index("日期")
print("按日期索引筛选2026-01-03 ~ 2026-01-06：")
print(df_time_idx.loc["2026-01-03":"2026-01-06"])

# 多列排序
df_sort = df.sort_values(by=["销量", "价格"], ascending=[False, True])
print("\n按销量降序、价格升序排序：")
print(df_sort[["产品", "销量", "价格"]])
print("-" * 60)

# ====================== 9. 时间列提取年/月/星期 ======================
df["年"] = df["日期"].dt.year
df["月"] = df["日期"].dt.month
df["星期"] = df["日期"].dt.day_name()
print("时间拆分字段：")
print(df[["日期", "年", "月", "星期"]])
print("-" * 60)

# ====================== 10. 去重、随机采样 ======================
df_drop_dup = df.drop_duplicates(subset=["产品", "区域"], keep="last")
print("按产品区域去重后行数：", len(df_drop_dup))

sample3 = df.sample(n=3)
print("\n随机抽取3行：")
print(sample3[["产品", "区域", "销量"]])
print("-" * 60)

# ====================== 11. 窗口函数 rolling 滑动均值 ======================
df_sort_date = df.sort_values("日期")
df_sort_date["3日滑动平均销量"] = df_sort_date["销量"].rolling(window=3).mean()
print("滑动窗口3日均值：")
print(df_sort_date[["日期", "销量", "3日滑动平均销量"]])
print("-" * 60)

# ====================== 12. 链式流水线一站式处理 ======================
df_result = (df
             .fillna({"销量": 0, "价格": 0})
             .query("销量 > 90")
             .assign(营收=lambda x: x["销量"] * x["价格"])
             .groupby("产品")["营收"].sum()
             .reset_index()
             .sort_values("营收", ascending=False)
            )
print("链式处理最终汇总结果：")
print(df_result)

# ====================== 13. 文件读写（取消注释使用） ======================
# df.to_csv("销售数据.csv", index=False, encoding="utf-8-sig")
# df.to_excel("销售数据.xlsx", sheet_name="销售明细", index=False)
# chunk读取大文件
# chunk_iter = pd.read_csv("大文件.csv", chunksize=1000)
# for chunk in chunk_iter:
#     print(chunk["销量"].sum())