# 针对数据集

import os
import time
from datasets import Dataset
from ragas import evaluate, RunConfig

# 这个导入有警告不用管,能用
from ragas.metrics import (
    Faithfulness,              # 忠实度        
    AnswerRelevancy,           # 忠实度       
    ContextRecall,           # 上下文召回       
    ContextPrecision,        # 上下文精确度    
)
from pathlib import Path
import json
import uuid
import numpy as np

# LangChain 组件
from langchain_community.document_loaders import PyPDFLoader
"""
一种集成多个检索器的检索器。采用了排序融合技术。
"""
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
# langchain-classic/retrievers/document_compressors/chain_extract/LLMChainExtractor
# 一种使用 LLM 链提取文档相关部分的文档压缩器。
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_text_splitters import RecursiveCharacterTextSplitter
"""
FAISS 向量检索,基于语义相似度（Embeddings 向量距离）检索 TopK 文档,擅长语义匹配，能理解问题的 “意图”
"""

from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

# --------------------------
# 1. 初始化 模型与嵌入、ragas(自己电脑配置可以的话就不用base_url=...)
# --------------------------

MAC_IP = "192.113.1.164"
llm = ChatOllama(
    model='qwen2.5:7b', 
    temperature=0,
    base_url=f"http://{MAC_IP}:11434"
)
embedding = OllamaEmbeddings(
    model="bge-m3:latest",
    base_url=f"http://{MAC_IP}:11434"
)

vectordb = None

faithfulness_metric = Faithfulness()
answer_relevancy_metric = AnswerRelevancy()
context_recall_metric = ContextRecall()
context_precision_metric = ContextPrecision()
# --------------------------
# 2. 文档处理   
#       加载 PDF 文档、文本分块、FAISS 向量索引的创建 / 加载
# --------------------------
current_dir = Path(__file__).parent.parent
# 请确保路径正确，如果报错请检查文件是否存在
data_file_path = current_dir.parent / "Data" / "初赛训练数据集.pdf"

if not data_file_path.exists():
    raise FileNotFoundError(f"找不到数据文件: {data_file_path}")

print(f"正在使用 pdfplumber 解析文件: {data_file_path} ...")
loader = PyPDFLoader(data_file_path)
docs = loader.load()
print("文档个数：", len(docs))
# 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
)
split_docs = text_splitter.split_documents(docs)
print("分块个数：", len(split_docs))

index_folder_path = "data/faiss_index"
index_name = "3"
index_file_path = os.path.join(index_folder_path, f"{index_name}.faiss")
# 检查索引文件是否存在
if os.path.exists(index_file_path):
    print("索引文件已存在，直接加载...")
    vectordb = FAISS.load_local(index_folder_path, embedding, index_name, allow_dangerous_deserialization=True)
else:
    print("索引文件不存在，创建并保存索引...")
    # 创建向量存储
    vectordb = FAISS.from_documents(split_docs, embedding)
    # 保存索引
    vectordb.save_local(index_folder_path, index_name)
    print("向量化完成....")

# --------------------------
# 3. 评估配置
# 定义待测试问题、标准答案（ground truth）、检索 TopK 参数等
# questions/ground_truths：评估用的 “问题 - 标准答案” 对，是 ragas 评估的核心参照；
# --------------------------
topK_doc_count = 4

questions = ["如何使用安全带？", "车辆如何保养？"]
ground_truths = [
    '''调节座椅到合适位置，缓慢拉出安全带，将锁舌插入锁扣中，直到听见“咔哒”声。
    使腰部安全带应尽可能低的横跨于胯部。确保肩部安全带斜跨整个肩部，穿过胸部。
    将前排座椅安全带高度调整至合适的位置。
    请勿将座椅靠背太过向后倾斜。
    请在系紧安全带前检查锁扣插口是否存在异物（如：食物残渣等），若存在异物请及时取出。
    为确保安全带正常工作，请务必将安全带插入与之匹配的锁扣中。
    乘坐时，安全带必须拉紧，防止松垮，并确保其牢固贴身，无扭曲。
    切勿将安全带从您的后背绕过、从您的胳膊下面绕过或绕过您的颈部。安全带应远离您的面部和颈部，但不得从肩部滑落。
    如果安全带无法正常使用，请联系Lynk & Co领克中心进行处理。''',
    "为了保持车辆处于最佳状态，建议您定期关注车辆状态，包括定期保养、洗车、内部清洁、外部清洁、轮胎的保养、低压蓄电池的保养等。"
    
]
'''
, "座椅太热怎么办？"
您好，如果您的座椅太热，1、通过中央显示屏，设置座椅加热强度或关闭座椅加热功能，
    在中央显示屏中点击座椅进入座椅加热控制界面，可在“关-低-中-高”之间循环。
    2、登录Lynk & Co App，按下前排座椅加热图标图标可以打开/关闭前排座椅加热。
    3、在中央显示屏中唤起空调控制界面然后点击舒适选项，降低座椅加热时间。'''


# --------------------------
# 4. 评估类（QAEvaluator）
# 封装「问答生成」和「指标计算」逻辑，核心是 RetrievalQA 链和 ragas 评估
# --------------------------
class QAEvaluator:
    def __init__(self, llm, retriever, embeddings):
        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            # input_key="question",
            # output_key="answer",
            return_source_documents=True,
        )
        self.embeddings = embeddings
        
    # 生成答案
    def generate_answers(self, questions):
        answers = []
        contexts = []
        for question in questions:
            print("问题：", question)
            response = self.chain.invoke(question)
            print("大模型答复：", response['result'], "\n")
            answers.append(response['result'])
            contexts.append([doc.page_content for doc in response['source_documents']])
        return answers, contexts
    
    # 评估
    def method_evaluate(self, questions, answers, contexts, ground_truths):
        evaluate_data = {
            "question":questions,
            "answer":answers,
            "contexts":contexts,
            "ground_truth":ground_truths,
        }
        evaluate_dataset = Dataset.from_dict(evaluate_data)
        # 【优化】配置运行参数，增加超时时间和重试次数
        # max_retries: 失败后重试次数
        # timeout: 单个请求的超时时间（秒）
        run_config = RunConfig(
            timeout=120,      # 增加到 120 秒，适应远程网络延迟
            max_retries=2,    # 失败重试 2 次
            max_wait=60,      # 最大等待时间
            max_workers=1     # 注意：新版 ragas 可能在 RunConfig 中支持 max_workers，如果不支持请移除该行
        )

        try:
            # 【注意】新版 ragas.evaluate 签名可能略有不同
            # 如果 RunConfig 不起作用，可以尝试直接传递 timeout (取决于具体版本)
            # 这里尝试使用 run_config 参数
            evaluate_result = evaluate(
                dataset=evaluate_dataset,
                metrics=[
                    faithfulness_metric,
                    answer_relevancy_metric,
                    context_recall_metric,
                    context_precision_metric
                ],
                llm=llm,
                embeddings=self.embeddings,
                run_config=run_config, # 使用 run_config 替代 max_workers/timeout
                raise_exceptions=False # 忽略单个样本的错误，继续执行其他样本
            )
        except TypeError as e:
            # 兼容处理：如果 run_config 也不支持，回退到最简调用
            print(f"RunConfig 参数可能不兼容，尝试基本调用... 错误信息: {e}")
            try:
                evaluate_result = evaluate(
                    dataset=evaluate_dataset,
                    metrics=[
                        faithfulness_metric,
                        answer_relevancy_metric,
                        context_recall_metric,
                        context_precision_metric
                    ],
                    llm=llm,
                    embeddings=self.embeddings,
                    raise_exceptions=False
                )
            except Exception as e2:
                print(f"评估过程中发生严重错误: {e2}")
                return None
        except Exception as e:
            print(f"评估过程中发生错误: {e}")
            return None
        return evaluate_result


# --------------------------
# 5. 工具函数
# 执行评估(检索评估)、计算上下文精准率 / 召回率的 F1(在本案例忠实度与回答相关性都很高了)
# --------------------------
def exec_eval(retriever):
    qa_evaluator = QAEvaluator(llm, retriever, embedding)
    answers, contexts = qa_evaluator.generate_answers(questions)
    return qa_evaluator.method_evaluate(questions, answers, contexts, ground_truths)

def calc_f1(evaluate_result):
    if evaluate_result is None:
        print("评估结果为空，无法计算 F1。")
        return 0.0
    
    context_precisions = evaluate_result["context_precision"]
    context_recalls = evaluate_result["context_recall"]
    
    # 检查是否包含 nan
    if any(np.isnan(x) for x in context_precisions) or any(np.isnan(x) for x in context_recalls):
        print("警告: 评估结果中包含 NaN，可能是 LLM 超时或错误导致。请检查 max_workers 和 timeout 设置。")
        # 可以选择返回 0 或者跳过
        return 0.0
    
    print("内容精度context_precisions=",context_precisions)
    print("内容召回context_recalls=",context_recalls)
    
     # 过滤掉可能的 nan 值再计算平均值（虽然上面已经检查过，但作为双重保险）
    valid_precisions = [x for x in context_precisions if not np.isnan(x)]
    valid_recalls = [x for x in context_recalls if not np.isnan(x)]
    
    if not valid_precisions or not valid_recalls:
        return 0.0
    
    context_precision_score = sum(context_precisions) / len(context_precisions)
    context_recall_score = sum(context_recalls) / len(context_recalls)
    
    if (context_precision_score + context_recall_score) == 0:
        return 0.0
    
    f1_scores = []
    for p, r in zip(context_precisions, context_recalls):
        if p + r == 0:
            f1_scores.append(0.0)
        else:
            f1 = (2 * p * r) / (p + r)
            f1_scores.append(f1)
    # 返回每个问题F1的平均分
    return sum(f1_scores) / len(f1_scores)
    
    # f1_score = (2 * context_precision_score * context_recall_score) / (context_precision_score + context_recall_score)
    # return f1_score

# --------------------------
# 6. 检索器创建 + 评估执行与结果输出
# 构建 3 类检索器（FAISS 向量检索、BM25+FAISS 混合检索、上下文压缩检索）
# 逐个执行检索器评估，输出核心指标（faithfulness 等）和 F1 分数，对比性能
# --------------------------
# FAISS 向量检索
faiss_retriever = vectordb.as_retriever(search_kwargs={"k": topK_doc_count, "score_threshold": 0.7})
faiss_eval_result = exec_eval(faiss_retriever)
print("FAISS 向量检索评估结果：", faiss_eval_result,"\nf1分数：",calc_f1(faiss_eval_result))
time.sleep(5)

# 全文检索(BM25检索)
bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k=topK_doc_count

# BM25+FAISS 混合检索
mix_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],weight=[0.2, 0.8])
mix_evaluate_result = exec_eval(mix_retriever)
print("混合检索mix_retriever评估结果：", mix_evaluate_result," \nf1分数：",calc_f1(mix_evaluate_result))
time.sleep(5)

# 上下文压缩检索
# 这个需要电脑较好配置
# 文档压缩器,遍历最初返回的文档,并仅从每个文档中提取与查询相关的内容。
# 上下文压缩检索器：需要传入一个文档压缩器和基本检索器
# 上下文压缩检索器将查询传递到基本检索器，获取初始文档并将它们传递到文档压缩器。
# 文档压缩器获取文档列表，并通过减少文档内容或完全删除文档来缩短文档列表。
compression_retriever = ContextualCompressionRetriever(
    base_compressor=LLMChainExtractor.from_llm(llm), 
    base_retriever=mix_retriever
)
compression_evaluate_result = exec_eval(compression_retriever)
print("compression_retriever评估结果：", compression_evaluate_result," ，f1分数：",calc_f1(compression_evaluate_result))


"""
1. Ragas的作用什么?
Ragas的作用就是评估，通过我们给定的数据格式
如：大模型的输入，大模型的输出，知识库的返回结果，我们这个问题的真正回答
借由评估器进行对比，同样的评估器也是大模型去使用，通过特定的提示词，让大模型对比回答和输出之类的方法，获得特定的分数
那这个分数就是我们的评估值
2. Ragas的使用步骤(最好能够参考官网)
RAGAS的使用步骤
setup pip install ragas
构造数据集->构造评估器->进行评估
（1） 生成，制作测试集
 首先
 ragas的测试数据集的格式分别是4个列表
   "question": questions, 我们的问题
    "answer": answers,  我们自己要评估的rag的回答
    "contexts": contexts, rag的知识库根据问题搜出来的结果
    "ground_truth": ground_truth 标准答案
    
举个例子
问题：板蓝根颗粒的用处
回答： 板蓝根可以治疗感冒
文档： xxx(板蓝根的药方)
标准回答 ： 板蓝根颗粒具有xxx的功能，能够治疗xxx病症

自动生成数据的代码如下：

而我们ragas自动生成测试集的原理如下，
我们将文档加载后交给ragas设置的大模型，大模型根据预先设定的提示词，对文档内容提问，并生成：
问题、问题对应文档内容(文档)、这个大模型给出的回答(期望回答)，
这3个结果。
然后，我们将生成的问题，传给我们自己要评估的RAG，生成我们要评估的RAG的回答
这时候我们就凑出了4个数据

然后呢，去做成特定格式(pandas的字典格式)
注意，ragas自动生成的测试集需要这个函数去处理：testdataset.to_pandas()，到此我们就可以获得我们的测试集了

事实上我们不需要一定用自动生成的测试集，只要能够给出以下格式，就可以进行转化成对应数据集，里面的数据我们可以手动塞入。
数据格式类似于
    "question": ['我们的问题']
    "answer":   ['我们自己要评估的rag的回答']
    "contexts": [['rag的知识库根据问题搜出来的结果']]
    "ground_truth": [['我们希望它的回答']]
根据ragas的版本不同，数据结构可能会有些许变化，比如从2维列表变为1维列表
但最终肯定是要去进行处理，因为处理后查询速度更快更方便
就是这个函数了：Dataset.from_dict

（2） 评估器构造
以下是一个标准的评估器，dataset就是我们之前处理好的数据集
我们的评估是evaluate函数：from ragas import evaluate
其中，metrics是一个列表，我们会将我们需要用到的评估方法加入进去，ragas提供了自带的几种函数，metrics列表里面接受的是一个函数
比如context_precision等，我们可以通过名字来知道它是评估什么数值的，
下面的列表说明它有4个评估项目，且是默认的4个评估器，如果不写的话，也是默认这4个评估器
这几个默认评估器已经可以包括9成的需求了。
也可以去查看metrics中ragas提供的几个评估器
我们也可以去自定义评估器，
此外，它还有llm参数和embedding参数，用于指定评估使用的模型

result = evaluate(
    dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)


（3） 进行评估运行
那么，它是如何进行评估的呢，简单来说，它每个评估器都会调用一个默认的模型，并且有特定的提示词进行输入
类似以下提示词：
根据问题,文档，和我希望的回答，为如下回答给出一个分数
评分标准为 这个回答是否和问题有关/与期望回答相同/等等
问题：{问题}
回答：{回答}
文档：{文档}
期望回答：{期望回答}

然后它就能为每一个回答提交分数
（4）检查评估结果
评估结果类似以下图片：https://i-blog.csdnimg.cn/blog_migrate/4d25205ba7361327f769192d43edcbf9.png

数据流向如下： 原数据->ragas自带的RAG(生成测试集)->我们要评估的RAG(生成回答)->评估(遍历每一个评估器->询问问题和回答能获得多高的分数)->返回结果

此外，请注意：ragas默认用的gpt4，可能在中文语境的评估会有不足，导致在某些评估没有生成返回值，导致评估结果为空
"""
