from typing import List
import numpy as np
import sys
import os
# 将项目根目录添加到系统路径(测试用)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.function_util import timing_decorator
import torch
# 优先cuda，无GPU自动切CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("当前设备：", device)
# print(torch.cuda.is_available())
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import math
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

def list_split(lst, chunk_size=10):
    # 无限循环，直到剩余列表不足一块时退出
    while True:
        # 判断当前剩余列表长度是否大于分块大小
        if len(lst) > chunk_size:
            yield lst[:chunk_size]
            lst = lst[chunk_size:]
        else:
            yield lst
            break

class EmbeddingModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.embedding_model = AutoModel.from_pretrained(model_path)
        self.embedding_model.eval()
        self.embedding_model.to(self.device)
        self.similarity_model = SentenceTransformer(model_path)
        self.similarity_model.to(self.device)
        self.headers = {'Content-Type': 'application/json'}
        
    @timing_decorator
    def get_batch_embeddings(self, texts, batch_size=128):
        all_embeddings = []
        chunk_id = 0
        total_chunks = math.ceil(len(texts) / 10)
        for content in list_split(texts, 10):
            chunk_id+=1
            """ 
            Tokenizer 分词编码:
            - padding=True：短文本自动补 0 对齐长度
            - truncation=True：超长文本截断到模型最大长度
            - return_tensors='pt'：输出 PyTorch 张量，而非普通 list
            - 返回字典：input_ids（文本数字编码）、attention_mask（有效 token 掩码）、部分模型（BERT）多token_type_ids
            """
            encoded_input = self.tokenizer(content, padding=True, truncation=True, return_tensors="pt")
            """
            兼容两种模型构建数据集 TensorDataset
            - BERT、RoBERTa 等需要token_type_ids区分句子 A/B；Qwen、E5、大部分通用嵌入模型不需要该参数
            - TensorDataset：把多个张量打包成数据集，方便 DataLoader 分批读取
            """            
            if 'token_type_ids' in encoded_input:
                dataset = TensorDataset(encoded_input['input_ids'], encoded_input['attention_mask'], encoded_input['token_type_ids'])
            else:
                dataset = TensorDataset(encoded_input['input_ids'], encoded_input['attention_mask'])
            # 构建 DataLoader 内层分批
            dataloader = DataLoader(dataset, batch_size=batch_size)
            lst_embeddings = []
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f'chunk {chunk_id:02d}/{total_chunks}'):
                    # 张量迁移到 GPU/CPU 设备
                    batch = tuple(t.to(self.device) for t in batch)
                    inputs = {'input_ids': batch[0], 'attention_mask':batch[1]}
                    if 'token_type_ids' in encoded_input:
                        inputs['token_type_ids'] = batch[2]
                        
                    # 模型前向传播获取输出
                    model_output = self.embedding_model(**inputs)
                    embeddings = model_output[0][:, 0]
                    # L2 归一化向量,每条向量模长 = 1
                    # 好处：后续计算余弦相似度时，等价于直接点积，简化检索计算，嵌入模型通用标准操作。
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    lst_embeddings.append(embeddings)
                    
            """
            torch.cat：拼接当前 chunk 内所有批次向量，恢复 10 条文本完整张量
            - .cpu()：把 GPU 张量转移回 CPU 内存，释放显存
            - extend：追加到全局向量列表all_embeddings
            """            
            lst_embeddings = torch.cat(lst_embeddings, dim=0)
            all_embeddings.extend(lst_embeddings.cpu())
            
        return all_embeddings
    
    @timing_decorator
    def get_embedding(self, text):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')

        input_ids = encoded_input['input_ids'].to(self.device)
        attention_mask = encoded_input['attention_mask'].to(self.device)

        if 'token_type_ids' in encoded_input:
            token_type_ids = encoded_input['token_type_ids'].to(self.device)
        else:
            token_type_ids = None

        with torch.no_grad():
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            if token_type_ids is not None:
                inputs['token_type_ids'] = token_type_ids

            model_output = self.embedding_model(**inputs)
            embedding = model_output[0][:, 0]
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding.cpu()
    
    # 文本召回、相似度匹配
    """
    参数说明：
    - sentences：候选文本列表，待匹配的一堆句子
    - source_sentence：查询句 / 源句子，用来和候选做对比
    - recall_num=-1：最多返回几条；-1代表不限制，返回全部符合条件的
    - threshold=0：相似度阈值，只保留相似度大于该值的结果
    """    
    @timing_decorator
    def get_similarity(self, sentences: List[str], source_sentence: str, recall_num: int = -1, threshold: float = 0):
        text_embeddings = self.similarity_model.encode(sentences)
        query_embedding = self.similarity_model.encode(source_sentence)
        
        cosine_similarities = np.dot(text_embeddings, query_embedding.T) / (np.linalg.norm(text_embeddings, axis=1) * np.linalg.norm(query_embedding))
        sorted_indices = cosine_similarities.argsort()[::-1]
        
        result_docs = []
        for i in sorted_indices:
            if cosine_similarities[i] > threshold:
                result_docs.append(sentences[i])
        
        if recall_num == -1:
            return result_docs
        if len(result_docs) > recall_num:
            return result_docs[0:recall_num]
        else:
            return result_docs


