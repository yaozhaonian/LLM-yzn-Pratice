# 使用python的rank_bm25库实现BM25算法
# 从 rank_bm25 库中导入 BM25Okapi 类，用于计算 BM25 相似度得分
from rank_bm25 import BM25Okapi
# 导入 jieba 库，用于中文分词
import jieba

# 定义一个包含多个文档的语料库，每个文档是一个字符串
corpus = [
    "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
    "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
    "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
    "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
    "我国首次在空间站开展舱外辐射生物学暴露实验",
    "这是第六个文档",
    "中华人民共和国对台独分子发出严肃的警告",
    "中央人民政府公布了关于三胎的福利政策"
]

# 对语料库中的每个文档进行分词操作，使用 jieba.lcut() 函数将文档分割成词语列表
# 最终得到一个包含多个词语列表的列表，每个子列表对应一个文档的分词结果
tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
print("tokenized_corpus:", tokenized_corpus)


# 定义一个查询语句，即要查找相关文档的关键词
query = "联合国发警告"
# 对查询语句进行分词操作，将其转换为词语列表
tokenized_query = jieba.lcut(query)
print("tokenized_query:", tokenized_query)
print('=' * 50)

# 使用分词后的语料库初始化 BM25Okapi 对象，后续将使用该对象进行相似度计算
bm25 = BM25Okapi(tokenized_corpus)
# 调用 BM25Okapi 对象的 get_scores 方法，计算查询语句与语料库中每个文档的相似度得分
# 得到一个包含多个得分的列表，每个得分对应语料库中的一个文档
scores = bm25.get_scores(tokenized_query)
# 打印计算得到的相似度得分列表
print("scores:", scores)
print('=' * 50)

# 调用 BM25Okapi 对象的 get_top_n 方法，根据查询语句的相似度得分从语料库中选取前 n 个最相关的文档
# 这里 n 设置为 1，表示只选取最相关的一个文档
top_n = bm25.get_top_n(tokenized_query,corpus,n=1)
# 打印选取的最相关文档列表
print("top_n:", top_n)
