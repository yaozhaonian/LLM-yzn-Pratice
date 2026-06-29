"""
句子嵌入模型微调（Fine-tuning）
利用标注数据，对预训练的中文模型 BAAI/bge-large-zh-v1.5 进行进一步训练，使其在特定任务（例如：API 匹配、语义检索）上表现得更好。
"""

import json
"""
sentence_transformers 核心组件：

SentenceTransformer：加载预训练模型。

InputExample：封装单个训练样本（包含文本对和标签）。

losses：提供各种损失函数，用于计算预测与标签之间的差异。

evaluation：提供评估器，用于在训练过程中监控模型性能。

torch.utils.data.DataLoader：PyTorch 的数据加载器，用于批量打乱和迭代训练样本。
"""
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def train(train_file):  # 准备训练数据
    # 假设你有一个包含句子对的 CSV 文件，其中第一列为句子1，第二列为句子2，第三列为相似度标签(0或1)
    with open(train_file, encoding='utf-8') as f:
        datas = json.load(f)
        all_examples = []
        for data in datas:
            all_examples.append(InputExample(texts=[data["sentences"], data["api"]], label=1))
            all_examples.append(InputExample(texts=[data["negative_sentences"], data["api"]], label=0))

    # 划分训练集和验证集(20%用于验证)
    train_examples, val_examples = train_test_split(all_examples, test_size=0.2, random_state=42)
    print(f"训练集数量:{train_examples}\n验证集数量:{val_examples}")
    
    model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

    # 定义训练参数
    train_loss = losses.CosineSimilarityLoss(model=model)
    train_dataloader= DataLoader(train_examples, shuffle=True, batch_size=16)
    num_epochs = 3

    # 定义评估器
    # 假设有一个验证集，用于评估模型性能
    # EmbeddingSimilarityEvaluator 是 sentence_transformers 内置的评估器，用于在验证集上评估模型的嵌入质量。
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name="validation")

    # 训练模型
    """
    model.fit() 是 SentenceTransformer 类提供的训练方法，负责执行完整的训练循环。
    epochs：训练轮数。

    evaluator：传入刚才定义的评估器，训练过程中会定期评估。

    evaluation_steps：每隔 1000 个训练步骤（batch 迭代）执行一次评估。

    save_best_model：如果评估指标有提升（例如 Spearman 相关系数增大），则自动保存当前最佳模型。

    show_progress_bar：显示训练进度条，便于观察。
    """
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        evaluator=evaluator,
        evaluation_steps=len(train_dataloader), # 每个epoch结束评估一次
        save_best_model=True,
        show_progress_bar=True
    )

    # 保存模型
    model.save("output/bge_model")

if __name__ =="__main__":
    train("new_dataset_train.json")
