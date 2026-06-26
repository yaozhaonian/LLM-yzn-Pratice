# 远程嵌入模型

import numpy as np
from ollama import Client

client = Client(host='http://127.0.0.1:11434')
model='bge-m3:latest'

class RemoteEmbeddingModel:
    def __init__(self):
        """
        初始化 RemoteEmbeddingModel 类
        创建一个 OpenAI 客户端实例，用于后续嵌入向量的生成操作
        """
        self.client = Client(host='http://127.0.0.1:11434')

    def get_embedding(self, text):
        """
        根据输入的文本获取对应的嵌入向量
        Args(参数):
        text (str): 输入的文本
        Returns(返回):
        list: 嵌入向量列表
        """        
        response = self.client.embed(model, text)
        embedding = response['embeddings']
        return embedding
    
    def get_batch_embeddings(self, texts, batch_size=10):
        """
        批量获取文本嵌入向量
        Args(参数):
        texts (list): 输入的文本列表
        batch_size (int, optional): 每批次处理的文本数量。默认为 10。
        Returns(返回):
        list: 批量处理后的嵌入向量列表
        """
        all_embeddings = []
        
        # 将文本列表分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            embeddings = self.get_embedding(batch_texts)
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def get_similarity(self, target_texts, vectors, query, recall_num=5, threshold=0.5):
        query = np.array(query)
        vectors = np.array(vectors)
        
        # 计算余弦相似度
        norm_query = np.linalg.norm(query)
        norm_vectors = np.linalg.norm(vectors, axis=1)
        cos_sims = np.dot(query, vectors.T) / (norm_query * norm_vectors)
        
        # 按相似度降序排序
        sorted_indices = np.argsort(cos_sims)[::-1]
        
        result_docs = []
        for i in sorted_indices:
            if cos_sims[i] >= threshold:
                result_docs.append(target_texts[i])
            if recall_num != -1 and len(result_docs) >= recall_num:
                break
            
        return result_docs


if __name__ == '__main__':
    # 测试
    embeddingModel = RemoteEmbeddingModel()
    x = embeddingModel.get_batch_embeddings(["你好！", "hello"])
    print("x:",x)
    print("="*50)
    embeddingModel = RemoteEmbeddingModel()
    y = embeddingModel.get_embedding("hello")
    print("y",y)  
    print("y的维度:",len(y[0]))  






