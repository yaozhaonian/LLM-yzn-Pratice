# pandas_fast_template.py
import pandas as pd
import numpy as np

# 1. 模拟业务数据，替换成你的读取文件代码即可
df = pd.DataFrame({
    "品类": ["数码", "服饰", "数码", "食品", "服饰"],
    "城市": ["广州", "深圳", "广州", "佛山", "深圳"],
    "销售额": [1200, 850, 1500, 600, 920],
    "利润": [240, 170, 300, 90, 184]
})

# ---------------------- 模块1：多条件筛选模板 ----------------------
# 模板1：且 / 或
filter_1 = df[(df["品类"] == "数码") & (df["销售额"] > 1000)]
# 模板2：包含多个值
filter_2 = df[df["城市"].isin(["广州", "深圳"])]
# 模板3：query简洁写法
min_sales = 900
filter_3 = df.query("品类 != '食品' and 销售额 > @min_sales")
# 模板4：模糊匹配文本
filter_4 = df[df["城市"].str.contains("广")]

print("【多条件筛选结果】")
print(filter_3)
print("-" * 50)

# ---------------------- 模块2：GroupBy分组聚合万能模板 ----------------------
# 单维度分组，多指标聚合
group_single = df.groupby("品类").agg(
    销售总额=("销售额", "sum"),
    平均销售额=("销售额", "mean"),
    总利润=("利润", "sum"),
    订单数量=("销售额", "count")
).reset_index()

# 双维度分组
group_double = df.groupby(["品类", "城市"]).agg(
    销售总额=("销售额", "sum")
).reset_index()

print("【单维度分组聚合】")
print(group_single)
print("【双维度分组聚合】")
print(group_double)
print("-" * 50)

# ---------------------- 模块3：数据透视表模板（Excel透视表等价） ----------------------
pivot_table = df.pivot_table(
    index="品类",        # 行维度
    columns="城市",      # 列维度
    values="销售额",     # 需要计算的值
    aggfunc="sum",      # 聚合方式 sum/mean/count/max/min
    fill_value=0         # 空白填充0
)
print("【透视表】")
print(pivot_table)

# 交叉统计表（计数专用）
cross_tab = pd.crosstab(df["品类"], df["城市"])
print("\n【交叉计数表】")
print(cross_tab)