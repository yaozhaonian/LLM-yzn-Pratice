# 使用 Ragas 进行评估
import os
import sys
from pathlib import Path
from openai import OpenAI
import jieba
import numpy as np
import pandas as pd
from datetime import datetime

# 🔴 修复：使用 HuggingFace datasets 创建 Dataset
from datasets import Dataset as HFDataset
from ragas import evaluate
from ragas.llms import llm_factory

# ======================
# 配置
# ======================

# Data 文件夹路径
DATA_DIR = Path(__file__).parent.parent.parent / "Data"
print(f"📁 Data 目录：{DATA_DIR}")
print(f"📁 目录是否存在：{DATA_DIR.exists()}")

# 🔴 检查 Ollama 可用模型
def check_ollama_models():
    """检查 Ollama 中已安装的模型"""
    try:
        client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
        models = client.models.list()
        model_names = [m.id for m in models.data]
        print(f"📦 Ollama 可用模型：{model_names}")
        return model_names
    except Exception as e:
        print(f"❌ 无法连接 Ollama: {e}")
        return []

# 获取可用模型
available_models = check_ollama_models()

# 🔴 选择可用的模型
if "qwen2.5:7b" in available_models:
    MODEL_NAME = "qwen2.5:7b"
elif "qwen2.5" in available_models:
    MODEL_NAME = "qwen2.5"
elif "llama3:latest" in available_models:
    MODEL_NAME = "llama3:latest"
elif "llama3" in available_models:
    MODEL_NAME = "llama3"
else:
    MODEL_NAME = available_models[0] if available_models else "qwen2.5:7b"

print(f"🤖 使用模型：{MODEL_NAME}")

# Ollama 客户端
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

# LLM 配置
llm = llm_factory(MODEL_NAME, provider="openai", client=client)


# ======================
# 简单的文档加载和检索
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
        
        paragraphs = content.split('\n\n')
        self.documents = [p.strip() for p in paragraphs if len(p.strip()) > 50]
        
        print(f"✅ 加载了 {len(self.documents)} 个文档块")
    
    def _bm25_score(self, query: str, doc: str) -> float:
        """简单的 BM25 相似度"""
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
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            answer = response.choices[0].message.content
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
            "question": "周鸿祎对 DeepSeek 有什么评价？",
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
        # 方法 1: 保存完整评估结果
        output_path = output_dir / f"eval_results_{timestamp}.csv"
        results.to_csv(output_path)
        saved_files.append(f"📊 评估结果：{output_path.name}")
        print(f"💾 评估结果已保存到：{output_path}")
    except Exception as e:
        print(f"⚠️ 评估结果保存失败：{e}")
    
    # 方法 2: 保存详细数据
    detailed_path = output_dir / f"eval_details_{timestamp}.csv"
    detailed_data = []
    for i, sample in enumerate(eval_samples, 1):
        detailed_data.append({
            "id": i,
            "question": sample["question"],
            "answer": sample["answer"],
            "ground_truth": sample["ground_truth"],
            "contexts": " ||| ".join(sample["contexts"]),
            "contexts_count": len(sample["contexts"])
        })
    
    df_detailed = pd.DataFrame(detailed_data)
    df_detailed.to_csv(detailed_path, index=False, encoding='utf-8-sig')
    saved_files.append(f"📝 详细数据：{detailed_path.name}")
    print(f"💾 详细数据已保存到：{detailed_path}")
    
    # 方法 3: 保存分数统计
    scores_path = output_dir / f"eval_scores_{timestamp}.csv"
    scores_data = []
    
    try:
        if hasattr(results, 'scores'):
            column_names = results.scores.column_names if hasattr(results.scores, 'column_names') else []
            
            for metric_name in column_names:
                try:
                    scores = results.scores[metric_name]
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        scores_data.append({
                            "metric": metric_name,
                            "avg_score": round(avg_score, 4),
                            "min_score": round(min(scores), 4),
                            "max_score": round(max(scores), 4),
                            "count": len(scores)
                        })
                except Exception as e:
                    print(f"⚠️ 提取 {metric_name} 分数失败：{e}")
        
        if scores_data:
            df_scores = pd.DataFrame(scores_data)
            df_scores.to_csv(scores_path, index=False, encoding='utf-8-sig')
            saved_files.append(f"📈 分数统计：{scores_path.name}")
            print(f"💾 分数统计已保存到：{scores_path}")
    except Exception as e:
        print(f"⚠️ 分数统计保存失败：{e}")
    
    return saved_files


# ======================
# 主函数
# ======================

async def main():
    print("=" * 60)
    print("开始 RAGAS 评估实验（独立版本）")
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
        print(f"📝 回答：{result['answer'][:100]}...")
        
        eval_samples.append({
            "question": item["question"],
            "answer": result["answer"],
            "contexts": result["contexts"],
            "ground_truth": item["ground_truth"]
        })
    
    # 4. 使用 ragas 评估
    print("\n" + "=" * 60)
    print("开始 RAGAS 评估...")
    print("=" * 60)
    
    # 🔴 修复：使用 RAGAS 0.4+ 正确的 metrics 实例化方式
    from ragas.metrics import AspectCritic
    
    metrics = [
        AspectCritic(
            name="faithfulness",
            llm=llm,
            definition="检查回答是否基于提供的上下文，没有编造信息",
        ),
        AspectCritic(
            name="context_precision",
            llm=llm,
            definition="检查检索到的上下文是否与问题相关",
        ),
        AspectCritic(
            name="answer_relevancy",
            llm=llm,
            definition="检查回答是否与问题相关，是否直接回答了用户的问题",
        ),
    ]
    
    print(f"📊 使用指标：{[m.name for m in metrics]}")
    
    try:
        # 🔴 修复：使用 HuggingFace Dataset 创建数据集
        dataset_dict = {
            "question": [s["question"] for s in eval_samples],
            "answer": [s["answer"] for s in eval_samples],
            "contexts": [s["contexts"] for s in eval_samples],
            "ground_truth": [s["ground_truth"] for s in eval_samples],
        }
        
        # 🔴 使用 HuggingFace 的 Dataset
        dataset = HFDataset.from_dict(dataset_dict)
        print(f"✅ Dataset 创建成功：{len(dataset)} 条样本")
        print(f"📋 Dataset 列：{dataset.column_names}")
        
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm
        )
        
        print("\n✅ 评估完成！")
        print(f"\n📊 评估结果:")
        print(results)
        
        # 保存结果
        output_dir = Path(__file__).parent
        saved_files = save_results_to_csv(results, eval_samples, output_dir)
        
        # 打印总结
        print("\n" + "=" * 60)
        print("📊 评估总结")
        print("=" * 60)
        
        try:
            if hasattr(results, 'scores'):
                column_names = results.scores.column_names if hasattr(results.scores, 'column_names') else []
                for metric_name in column_names:
                    scores = results.scores[metric_name]
                    if scores:
                        avg = sum(scores) / len(scores)
                        print(f"{metric_name}: {avg:.4f}")
        except Exception as e:
            print(f"⚠️ 打印分数失败：{e}")
        
        print("\n✅ 生成的文件:")
        for file in saved_files:
            print(f"   {file}")
        
    except Exception as e:
        print(f"\n❌ 评估失败：{e}")
        import traceback
        traceback.print_exc()
        
        # 即使评估失败，也保存原始数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_path = Path(__file__).parent / f"eval_failed_{timestamp}.csv"
        
        failed_data = []
        for i, sample in enumerate(eval_samples, 1):
            failed_data.append({
                "id": i,
                "question": sample["question"],
                "answer": sample["answer"],
                "ground_truth": sample["ground_truth"],
                "contexts": " ||| ".join(sample["contexts"])
            })
        
        df_failed = pd.DataFrame(failed_data)
        df_failed.to_csv(failed_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 原始数据已保存到：{failed_path}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())