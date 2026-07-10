import numpy as np
import json
import subprocess
import time
import traceback
import os
from utils import logger

model = 'bge-m3:latest'


class RemoteEmbeddingModel:
    def __init__(self):
        pass

    def _call_embed_api(self, text, max_retries=3, retry_delay=3):
        url = 'http://127.0.0.1:11434/api/embed'
        data = {'model': model, 'input': text}
        
        cmd = [
            'curl', '-s', '-X', 'POST', url,
            '-H', 'Content-Type: application/json',
            '-H', 'Connection: keep-alive',
            '-d', json.dumps(data, ensure_ascii=False)
        ]

        for attempt in range(max_retries):
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env={**os.environ, 'CURL_TIMEOUT': '60'})

            if result.returncode != 0:
                logger.error(
                    f"Embedding API request failed (attempt {attempt+1}/{max_retries}): "
                    f"returncode={result.returncode}, stderr={result.stderr[:300]}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise RuntimeError(f"Embedding API request failed: {result.stderr.strip()}")

            try:
                response = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Embedding API returned invalid JSON (attempt {attempt+1}/{max_retries}): "
                    f"{e}; stdout={result.stdout[:300]}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise

            embedding = response.get('embeddings') or response.get('embedding')
            if embedding is None:
                logger.error(f"Embedding API response missing embeddings (attempt {attempt+1}/{max_retries}): {response}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise ValueError(f"Embedding API response missing embeddings: {response}")

            if isinstance(embedding, list) and len(embedding) == 0:
                logger.error(f"Embedding API returned empty embeddings (attempt {attempt+1}/{max_retries}): {response}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise ValueError(f"Embedding API returned empty embeddings: {response}")

            return embedding

        raise RuntimeError("Embedding API request failed after all retries")

    def get_embedding(self, text):
        try:
            embedding = self._call_embed_api(text)

            if isinstance(embedding, list):
                if isinstance(text, list):
                    if isinstance(embedding[0], list):
                        return [[float(x) for x in emb] for emb in embedding]
                    return [[float(x) for x in embedding]]

                if isinstance(embedding[0], list):
                    return [float(x) for x in embedding[0]]
                return [float(x) for x in embedding]

            raise ValueError(f"Unexpected embedding format: {embedding}")

        except Exception as e:
            logger.error(
                f"Embedding format conversion failed: {e}\n{traceback.format_exc()}"
            )
            raise

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
    embeddingModel = RemoteEmbeddingModel()
    x = embeddingModel.get_batch_embeddings(["你好！", "hello"])
    print("x:", x)
    print("="*50)
    embeddingModel = RemoteEmbeddingModel()
    y = embeddingModel.get_embedding("hello")
    print("y", y)
    print("y的维度:", len(y))
