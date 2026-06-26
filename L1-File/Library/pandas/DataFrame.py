import pandas as pd

# 字典: 键=列名，值=列数据
data = {
    "姓名": ["张三", "李四", "王五"],
    "年龄": [18, 20, 22],
    "性别": ["男", "女", "男"],
    "城市": ["北京", "上海", "广州"]
}

# 生成二维表格
df = pd.DataFrame(data)
print(df)
print(df.columns)  # 输出列名
print(df["姓名"])  # 输出姓名列数据
print("="*20)

df2 = pd.DataFrame({
    "产品": ["A","B","A","C","B"],
    "销量": [120, 90, 150, 88, 110],
    "价格": [29.9, 39.9, 29.9, 49.9, 39.9]
})

# 1. 基础数值统计
print(df2.describe())
print("="*20)

# 数值统计
print("====数值指标====")
print(df2.describe())
# 文本分类统计
print("====分类指标====")
print(df2.describe(include="O"))
print("="*20)

# 3. 自定义分位数
print(df2.describe(percentiles=[0.05, 0.95]))
print("="*20)

# 4. 单独一列统计
print(df2["销量"].describe())

"""
count：非空数据条数
mean：平均值
std：标准差（数据波动大小）
min：最小值
25%：下四分位数
50%：中位数
75%：上四分位数
max：最大值
"""












