import numpy as np
from numpy import dot
from numpy.linalg import norm
from ollama import Client

client = Client(host='http://127.0.0.1:11434')

def get_embedding(text, model='bge-m3:latest'):
    # 使用ollama库获取嵌入向量
    response = client.embed(model, text)
    embedding = response['embeddings']
    return embedding

query = "国际争端"
# query = "global conflicts"
documents = [
    "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
    "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
    "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
    "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
    "我国首次在空间站开展舱外辐射生物学暴露实验",
]

def cos_sim(a, b):
    """ 余弦相似度 -- 越大越相似 """
    return dot(a, b) / (norm(a) * norm(b))

def l2(a, b):
    """ 欧式距离 -- 越小越相似 """
    x = np.asarray(a) - np.asarray(b)
    return norm(x)

query_vec = get_embedding(query)[0]
doc_vec = get_embedding(documents)
# print("输出:\n",doc_vec)

print("余弦相似度:")
print("query 和自己的相似度:",cos_sim(query_vec, query_vec))
for i, vec in enumerate(doc_vec):
    print(f"query 和第{i+1}行的相似度:{cos_sim(query_vec, vec)}")

print("\n欧氏距离:")
print("query 和自己的欧氏距离:",l2(query_vec, query_vec))
for i, vec in enumerate(doc_vec):
    print(f"query 和第{i+1}行的欧氏距离:{l2(query_vec, vec)}")





