import numpy as np
import hashlib

class SimpleEmbeddingModel:
    def __init__(self, dim=4096):
        self.dim = dim
    
    def get_embedding(self, text):
        if isinstance(text, list):
            return [self._text_to_embedding(t) for t in text]
        return [self._text_to_embedding(text)]
    
    def _text_to_embedding(self, text):
        text = str(text)
        hash_value = hashlib.md5(text.encode('utf-8')).digest()
        embedding = []
        for i in range(self.dim):
            byte_idx = i % 16
            hash_segment = hash_value[byte_idx]
            char_idx = (i // 16) % len(text) if text else 0
            char_val = ord(text[char_idx]) if text else 0
            val = (hash_segment + char_val + i * 137) % 256
            embedding.append(val / 255.0)
        return embedding
    
    def get_batch_embeddings(self, texts, batch_size=10):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            embeddings = self.get_embedding(batch_texts)
            all_embeddings.extend(embeddings)
        return all_embeddings
    
    def get_similarity(self, target_texts, vectors, query, recall_num=5, threshold=0.5):
        query = np.array(query)
        vectors = np.array(vectors)
        
        norm_query = np.linalg.norm(query)
        norm_vectors = np.linalg.norm(vectors, axis=1)
        cos_sims = np.dot(query, vectors.T) / (norm_query * norm_vectors)
        
        sorted_indices = np.argsort(cos_sims)[::-1]
        
        result_docs = []
        for i in sorted_indices:
            if cos_sims[i] >= threshold:
                result_docs.append(target_texts[i])
            if recall_num != -1 and len(result_docs) >= recall_num:
                break
        
        return result_docs

if __name__ == '__main__':
    model = SimpleEmbeddingModel()
    emb = model.get_embedding("查询产品信息")
    print(f"嵌入向量维度: {len(emb[0])}")
    print(f"前10个值: {emb[0][:10]}")
