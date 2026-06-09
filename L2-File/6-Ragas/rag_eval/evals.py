# 使用 Ragas 进行评估 (优化版 v0.4+)
from pathlib import Path
from langchain_ollama import ChatOllama, OllamaEmbeddings
import jieba
import pandas as pd
from datetime import datetime
import asyncio
import numpy as np # 🔴 新增：用于处理 NaN

# 🔴 修复：使用 HuggingFace datasets 创建 Dataset
from datasets import Dataset as HFDataset
from ragas import evaluate

# 🔴 修复：直接导入指标类并实例化
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)

# 🔴 新增：导入 Ragas 的 LangChain 包装器
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# ======================
# 配置
# ======================

# Data 文件夹路径
DATA_DIR = Path(__file__).parent.parent.parent / "Data"
print(f"📁 Data 目录：{DATA_DIR}")

# Ollama 配置
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
MODEL_NAME = "qwen2.5:7b"      # 聊天模型
EMBED_MODEL_NAME = "nomic-embed-text" # 嵌入模型 (请确保已运行 ollama pull nomic-embed-text)

# Ollama 客户端 (用于 RAG 生成回答)
chat_client = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)

# Ollama 嵌入客户端 (用于 Ragas 评估中的向量计算)
embed_client = OllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=OLLAMA_BASE_URL)

# 🔴 关键修复：将 LangChain 对象包装为 Ragas 兼容对象
ragas_llm = LangchainLLMWrapper(chat_client)
ragas_embeddings = LangchainEmbeddingsWrapper(embed_client)


# ======================
# 简单的文档加载和检索 (保持不变)
# ======================

class SimpleRAG:
    """简单的 RAG 实现，不依赖外部库"""
    
    def __init__(self, data_file: str):
        self.data_file = Path(data_file)
        self.documents = []
        self.load_documents()
    
    def load_documents(self):
        """加载文档并分块"""
        if not self.data_file.exists():
            print(f"❌ 文件不存在：{self.data_file}")
            return
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 简单按双换行符分割
        paragraphs = content.split('\n\n')
        self.documents = [p.strip() for p in paragraphs if len(p.strip()) > 50]
        
        print(f"✅ 加载了 {len(self.documents)} 个文档块")
    
    def _bm25_score(self, query: str, doc: str) -> float:
        """简单的关键词重叠相似度"""
        query_words = set(jieba.lcut(query))
        doc_words = set(jieba.lcut(doc))
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & doc_words)
        return overlap / len(query_words)
    
    def search(self, query: str, n_results: int = 3) -> list:
        """检索相关文档"""
        scores = []
        for i, doc in enumerate(self.documents):
            score = self._bm25_score(query, doc)
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scores[:n_results]:
            results.append({
                "content": self.documents[idx],
                "score": score
            })
        
        return results
    
    def query(self, question: str, n_results: int = 3) -> dict:
        """执行 RAG 查询"""
        search_results = self.search(question, n_results)
        contexts = [r["content"] for r in search_results]
        
        context_text = "\n\n".join(contexts)
        prompt = f"""请根据以下参考知识回答问题。如果参考知识中没有相关信息，请说"根据参考知识，没有找到相关信息"。

【参考知识】
{context_text}

【问题】
{question}

【回答】
"""
        
        try:
            # 使用 LangChain 标准的 invoke 方法
            from langchain_core.messages import HumanMessage
            
            response = chat_client.invoke([HumanMessage(content=prompt)])
            answer = response.content
            
        except Exception as e:
            print(f"❌ LLM 调用失败：{e}")
            answer = f"调用 LLM 失败：{str(e)}"
        
        return {
            "answer": answer,
            "contexts": contexts,
            "search_results": search_results
        }


# ======================
# 测试数据
# ======================

def create_test_dataset():
    """创建测试数据集"""
    test_data = [
        {
            "question": "周鸿祎对 DeepSeek 有什么表示？",
            "ground_truth": "2025 年 1 月 29 日，360 集团创始人周鸿祎表示，如果 DeepSeek 有需要，360 愿意提供网络安全方面的全力支持。"
        },
        {
            "question": "DeepSeek 是什么时候成立的？",
            "ground_truth": "DeepSeek 成立于 2023 年 7 月 17 日，由知名量化资管巨头幻方量化创立。"
        },
        {
            "question": "DeepSeek-R1 模型有什么特点？",
            "ground_truth": "DeepSeek-R1 在后训练阶段大规模使用了强化学习技术，在仅有极少标注数据的情况下，极大提升了模型推理能力，成本价格低廉，性能与 OpenAI 相当。"
        },
    ]
    return test_data


# ======================
# 保存结果到 CSV
# ======================

def save_results_to_csv(results, eval_samples, output_dir: Path):
    """保存评估结果到多个 CSV 文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []
    
    try:
        # 1. 保存详细数据 (包含原始问答)
        detailed_path = output_dir / f"eval_details_{timestamp}.csv"
        detailed_data = []
        for i, sample in enumerate(eval_samples, 1):
            detailed_data.append({
                "id": i,
                "question": sample["question"],
                "answer": sample["answer"],
                "ground_truth": sample["ground_truth"],
                "contexts": " ||| ".join(sample["contexts"]),
            })
        df_detailed = pd.DataFrame(detailed_data)
        df_detailed.to_csv(detailed_path, index=False, encoding='utf-8-sig')
        saved_files.append(f"📝 详细数据：{detailed_path.name}")
        
        # 2. 保存分数统计
        scores_path = output_dir / f"eval_scores_{timestamp}.csv"
        scores_data = []
        
        # 🔴 修复：统一将结果转换为 DataFrame 处理
        df_results = None
        if isinstance(results, pd.DataFrame):
            df_results = results
        elif hasattr(results, 'to_pandas'):
            # 某些版本的 Ragas Result 对象有 to_pandas 方法
            df_results = results.to_pandas()
        else:
            # 如果结果是字典或其他结构，尝试手动转换
            print(f"⚠️ 未知的结果类型: {type(results)}，尝试直接打印")
            print(results)
            return saved_files

        if df_results is not None and not df_results.empty:
            # 识别数值列（通常是指标分数）
            numeric_cols = df_results.select_dtypes(include=['number']).columns
            
            for col in numeric_cols:
                # 丢弃 NaN 值进行统计
                valid_scores = df_results[col].dropna()
                if not valid_scores.empty:
                    scores_data.append({
                        "metric": col,
                        "avg_score": round(valid_scores.mean(), 4),
                        "min_score": round(valid_scores.min(), 4),
                        "max_score": round(valid_scores.max(), 4),
                        "count": len(valid_scores)
                    })
                else:
                    scores_data.append({
                        "metric": col,
                        "avg_score": np.nan,
                        "min_score": np.nan,
                        "max_score": np.nan,
                        "count": 0
                    })

            if scores_data:
                df_scores = pd.DataFrame(scores_data)
                df_scores.to_csv(scores_path, index=False, encoding='utf-8-sig')
                saved_files.append(f"📈 分数统计：{scores_path.name}")
            
    except Exception as e:
        print(f"⚠️ 结果保存失败：{e}")
        import traceback
        traceback.print_exc()
    
    return saved_files


# ======================
# 主函数
# ======================

async def main():
    print("=" * 60)
    print("开始 RAGAS 评估实验 (优化版)")
    print("=" * 60)
    
    # 1. 初始化 RAG
    data_file = DATA_DIR / "deepseek百度百科.txt"
    rag = SimpleRAG(str(data_file))
    
    if not rag.documents:
        print("❌ 文档加载失败，退出")
        return
    
    # 2. 创建测试数据集
    test_data = create_test_dataset()
    print(f"\n✅ 创建测试数据集，共 {len(test_data)} 条")
    
    # 3. 执行 RAG 查询
    print("\n🔍 执行 RAG 查询...")
    eval_samples = []
    
    for i, item in enumerate(test_data, 1):
        print(f"\n[{i}/{len(test_data)}] 问题：{item['question']}")
        result = rag.query(item['question'])
        print(f"📝 回答：{result['answer'][:80]}...")
        
        eval_samples.append({
            "question": item["question"],
            "answer": result["answer"],
            "contexts": result["contexts"],
            "ground_truth": item["ground_truth"]
        })
    
    # 4. 使用 Ragas 评估
    print("\n" + "=" * 60)
    print("开始 RAGAS 评估...")
    print("=" * 60)
    
    # 实例化指标对象
    metrics = [
        Faithfulness(),          
        AnswerRelevancy(),      
        ContextPrecision(),     
    ]
    
    metric_names = [m.__class__.__name__ for m in metrics]
    print(f"📊 使用指标：{metric_names}")
    
    try:
        # 创建 HuggingFace Dataset
        dataset_dict = {
            "question": [s["question"] for s in eval_samples],
            "answer": [s["answer"] for s in eval_samples],
            "contexts": [s["contexts"] for s in eval_samples],
            "ground_truth": [s["ground_truth"] for s in eval_samples],
        }
        
        dataset = HFDataset.from_dict(dataset_dict)
        print(f"✅ Dataset 创建成功：{len(dataset)} 条样本")
        
        # 🔴 关键修复：传入包装后的 llm 和 embeddings
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,           
            embeddings=ragas_embeddings, 
            raise_exceptions=False
        )
        
        print("\n✅ 评估完成！")
        print(f"\n📊 评估结果预览:")
        print(results)
        
        # 保存结果
        output_dir = Path(__file__).parent
        saved_files = save_results_to_csv(results, eval_samples, output_dir)
        
        # 打印总结
        print("\n" + "=" * 60)
        print("📊 评估总结")
        print("=" * 60)
        
        # 🔴 修复：直接从 DataFrame 打印分数
        df_results = None
        if isinstance(results, pd.DataFrame):
            df_results = results
        elif hasattr(results, 'to_pandas'):
            df_results = results.to_pandas()
            
        if df_results is not None and not df_results.empty:
            numeric_cols = df_results.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                valid_scores = df_results[col].dropna()
                if not valid_scores.empty:
                    avg = valid_scores.mean()
                    print(f"{col}: {avg:.4f} (有效样本数: {len(valid_scores)})")
                else:
                    print(f"{col}: NaN (无有效分数)")
        else:
            print("⚠️ 无法解析评估结果以打印摘要")
        
        print("\n✅ 生成的文件:")
        for file in saved_files:
            print(f"   {file}")
        
    except Exception as e:
        print(f"\n❌ 评估失败：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())