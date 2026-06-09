# 问题分解
# 用提示词工程中的CoT策略，把用户的问题拆成一个一个小问题来理解接下来可以使用并行与串行两个策略来执行子任务。
# PS.用于较复杂的需要推理的场景

from typing import List

# LangChain 组件
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import BasePromptTemplate, PromptTemplate, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
# LineListOutputParser用于行列表的输出解析器
from langchain_classic.retrievers.multi_query import LineListOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

embedding = OllamaEmbeddings(model="bge-m3:latest")
llm = ChatOllama(model='qwen2.5:7b', temperature=0)

# 辅助函数：将文档列表转换为字符串
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# 格式化输出内容
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

documents = [
    Document(page_content="番茄炒蛋的食材：\n\n- 新鲜鸡蛋：3-4个（根据人数调整）\n- 番茄：2-3个中等大小\n- 盐：适量\n- 白糖：一小勺（可选，用于提鲜）\n- 食用油：适量,最好选择橄榄油\n- 葱花：少许（可选，用于增香）\n\n这些是最基本的材料，当然也可以根据个人口味添加其他调料或配料。"),
    Document(page_content="番茄炒蛋的步骤：鸡蛋打入碗中，加入少许盐，用筷子或打蛋器充分搅拌均匀；\n   - 番茄洗净后切成小块备用。\n\n3. **炒鸡蛋**：锅内倒入适量食用油加热至温热状态，然后将搅拌好的鸡蛋液缓缓倒入锅中。待鸡蛋凝固时轻轻翻动几下，让其受热均匀直至完全熟透，随后盛出备用。\n\n4. **炒番茄**：在同一锅里留下的底油中放入切好的番茄块，中小火慢慢翻炒至出汁，可根据个人口味加一点点白糖提鲜。\n\n5. **合炒**：当番茄炒至软烂并开始释放大量汤汁时，再把之前炒好的鸡蛋倒回锅里，快速与番茄混合均匀，同时加入适量的盐调味。如果喜欢的话还可以撒上一些葱花增加香气。\n\n6. **完成**：最后检查一下味道是否合适，确认无误后即可关火装盘享用美味的番茄炒蛋啦！"),
    Document(page_content="技巧与注意事项：1. **选材**：选择新鲜的鸡蛋和成熟的番茄。新鲜的食材是做好这道菜的基础。\n2. **打蛋液**：将鸡蛋打入碗中后加入少许盐（根据个人口味调整），然后充分搅拌均匀。这样做可以让蛋更加松软且味道更佳。\n3. **处理番茄**：番茄最好先用开水稍微焯一下皮，然后去皮切块。这样可以去除表皮的硬质部分，让番茄更容易入味，并且口感更好。\n4. **热锅冷油**：先用中小火把锅烧热，再倒入适量食用油，待油温五成热时下蛋液。这样的做法可以使蛋快速凝固形成漂亮的形状而不易粘锅。\n5. **分步烹饪**：通常建议先炒鸡蛋至半熟状态取出备用；接着利用剩下的底油继续翻炒番茄至出汁，最后再将之前炒好的鸡蛋倒回锅里与番茄混合均匀加热即可。\n6. **调味品**：除了基本的盐之外，还可以根据喜好添加少量糖来提鲜或者一点酱油增色添香。注意调味料不宜过多以免掩盖了食材本身的味道。\n7. **出锅前加葱花**：如果喜欢的话，在即将完成时撒上一些葱花不仅能增加菜品色泽还能增添香气。"),
    Document(page_content="番茄炒蛋的工具:1.俗话说:工欲善其事先利其器；一个好用的不粘锅是新手必备的好伙伴，它能防止新手炒菜时因为食材粘锅而手忙脚乱。\n2.抽油烟机也是炒菜的必备家具，毕竟谁也不愿意待在满是油烟的厨房里干活。")
]

vectorstore = Chroma.from_documents(
    documents=documents, 
    embedding=embedding,
    collection_name="tomato_cook_decomposition"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
base_prompt = PromptTemplate.from_template(
    "根据提供的文档{documents}，回答问题：{question}"
)
basae_chain =  RunnableParallel({
    "documents":(lambda x: x["question"]) | retriever | format_docs,
    "question":lambda x: x["question"]
}) | base_prompt | llm | retriever

print("========检索到的文档（拆解前）========-")
#从实际来说，该问题的答案应该包括原材料、步骤，以及注意事项才算完整
pretty_print_docs(retriever.invoke("新手如何制作番茄炒蛋？"))

"""
Document 1:

番茄炒蛋的步骤：鸡蛋打入碗中，加入少许盐，用筷子或打蛋器充分搅拌均匀；
   - 番茄洗净后切成小块备用。

3. **炒鸡蛋**：锅内倒入适量食用油加热至温热状态，然后将搅拌好的鸡蛋液缓缓倒入锅中。待鸡蛋凝固时轻轻翻动几下，让其受热均匀直至完全熟透，随后盛出备用。

4. **炒番茄**：在同一锅里留下的底油中放入切好的番茄块，中小火慢慢翻炒至出汁，可根据个人口味加一点点白糖提鲜。

5. **合炒**：当番茄炒至软烂并开始释放大量汤汁时，再把之前炒好的鸡蛋倒回锅里，快速与番茄混合均匀，同时加入适量的盐调味。如果喜欢的话还可以撒上一些葱花增加香气。

6. **完成**：最后检查一下味道是否合适，确认无误后即可关火装盘享用美味的番茄炒蛋啦！
"""


print("开始问题拆解的处理=======================>")
template = """
你是一名AI语言模型助理。你的任务是将输入问题分解成至少5个子问题，通过一个个解决这些子问题从而解决完整的问题。
子问题需要在矢量数据库中检索相关文档。通过分解用户问题生成子问题，你的目标是帮助用户克服基于距离的相似性搜索的一些局限性。
请提供这些用换行符分隔的子问题本身，不需要额外内容。
原始问题: {question}
"""
DEFAULT_QUERY_PROMPT =PromptTemplate(
    input_variables=["question"],
    template=template,
)
chain = DEFAULT_QUERY_PROMPT | llm | LineListOutputParser()
result = chain.invoke({"question":"新手如何制作番茄炒蛋？"})
print("问题拆解：",result)
print("========完成测试大模型对问题的拆解========-")


DEFAULT_SUB_QUESTION_PROMPT = PromptTemplate(
    input_variables=["question", "sub_question", "documents"],
    template="""要解决主要问题{question}，需要先解决子问题{sub_question}。
    以下是为支持您的推理而提供的参考文档：{documents}。请直接给出当前子问题的答案。不需要额外内容。""",
)

class DecompositionRetriever(BaseRetriever):
    # 向量数据库检索器
    retriever: BaseRetriever
    # 生成子问题链
    make_sub_chain: Runnable
    # 解决子问题链
    resolve_sub_chain: Runnable

    # 一个便捷构造方法，也叫工厂方法。@classmethod = 把一个方法标记成 “属于类本身”，而不是属于某个实例。
    @classmethod
    def from_llm(
            cls,
            retriever: BaseRetriever,
            llm: BaseLanguageModel,
            prompt: BasePromptTemplate = DEFAULT_QUERY_PROMPT,
            sub_prompt: BasePromptTemplate = DEFAULT_SUB_QUESTION_PROMPT
    ) -> "DecompositionRetriever":
        output_parser = LineListOutputParser()
        return cls(
            retriever=retriever,
            make_sub_chain=prompt | llm | output_parser,
            resolve_sub_chain=sub_prompt | llm
        )
    
    # 生成子问题
    def generate_queries(self, query: str) -> List[str]:
        sub_questions = self.make_sub_chain.invoke({"question": query})
        print("生成的子问题：\n",sub_questions)
        # return [q.strip() for q in sub_questions]
        return sub_questions
    
    # 获得子问题答案
    def retrieve_documents(self, query: str, sub_queries: List[str]) -> List[Document]:
        sub_llm_chain = RunnableLambda(
            # 传入子问题，检索文档并回答
            lambda sub_query: self.resolve_sub_chain.invoke({
                "question": query, 
                "sub_question": sub_query,
                "documents": [doc.page_content for doc in self.retriever.invoke(sub_query)]
            })
        )
        # 批量执行所有的子问题
        responses = sub_llm_chain.batch(sub_queries)
        # 将子问题和答案合并作为解决主问题的文档
        documents = [
            Document(page_content=f"{sub_query}\n{response.content}")
            for sub_query, response in zip(sub_queries, responses)
        ]
        return documents
    
    # 继承自BaseRetriever的函数
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        # 生成子问题
        sub_queries = self.generate_queries(query)
        # 解决子问题
        documents = self.retrieve_documents(query, sub_queries)
        return documents

print("============使用DecompositionRetriever分解解决问题=================")
DecomRetri = DecompositionRetriever.from_llm(retriever=retriever, llm=llm)
decom_docs = DecomRetri.invoke("新手如何制作番茄炒蛋？")
print("========检索到的文档（拆解后）========")
pretty_print_docs(decom_docs)

# 创建prompt模板
template = """请根据以下文档回答问题:
### 文档:
{context}
### 问题:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

print("========回答========")
question = "新手如何制作番茄炒蛋？"
response = chain.invoke({"context": [doc.page_content for doc in decom_docs], "question": question})
print(response.content)


"""
========检索到的文档（拆解前）========-
Document 1:

番茄炒蛋的步骤：鸡蛋打入碗中，加入少许盐，用筷子或打蛋器充分搅拌均匀；
   - 番茄洗净后切成小块备用。

3. **炒鸡蛋**：锅内倒入适量食用油加热至温热状态，然后将搅拌好的鸡蛋液缓缓倒入锅中。待鸡蛋凝固时轻轻翻动几下，让其受热均匀直至完全熟透，随后盛出备用。

4. **炒番茄**：在同一锅里留下的底油中放入切好的番茄块，中小火慢慢翻炒至出汁，可根据个人口味加一点点白糖提鲜。

5. **合炒**：当番茄炒至软烂并开始释放大量汤汁时，再把之前炒好的鸡蛋倒回锅里，快速与番茄混合均匀，同时加入适量的盐调味。如果喜欢的话还可以撒上一些葱花增加香气。

6. **完成**：最后检查一下味道是否合适，确认无误后即可关火装盘享用美味的番茄炒蛋啦！
开始问题拆解的处理=======================>
问题拆解： ['如何选择新鲜的番茄？', '如何挑选合适的鸡蛋？', '制作番茄炒蛋需要哪些基本厨具？', '番茄炒蛋的烹饪步骤是什么？', '如何根据个人口味调整番茄炒蛋的味道？']
========完成测试大模型对问题的拆解========-
============使用DecompositionRetriever分解解决问题=================
生成的子问题：
 ['如何选择新鲜的番茄？', '如何挑选合适的鸡蛋？', '制作番茄炒蛋需要哪些基本厨具？', '番茄炒蛋的烹饪步骤是什么？', '如何根据个人口味调整番茄炒蛋的味道？']
========检索到的文档（拆解后）========
Document 1:

如何选择新鲜的番茄？
要选择新鲜的番茄，可以注意以下几点：
1. 观察番茄的颜色是否鲜艳，成熟度高的番茄颜色更加鲜亮。
2. 按压番茄看其硬度，成熟的番茄会有一定的弹性但不会太硬或太软。
3. 闻一闻番茄是否有自然的新鲜果香，避免选择有异味的番茄。
----------------------------------------------------------------------------------------------------
Document 2:

如何挑选合适的鸡蛋？
为了挑选合适的鸡蛋，可以关注以下几个方面：
1. 观察蛋壳：新鲜的鸡蛋蛋壳表面干净、颜色均匀，没有斑点或裂纹。
2. 检查蛋重：同样大小的鸡蛋，越重的新鲜度越高。
3. 敲听声音：轻轻敲击鸡蛋在耳边听其声音，清脆声表示新鲜，沉闷声则可能不新鲜。
4. 观察蛋白和蛋黄：打开后观察内部结构，新鲜鸡蛋的蛋白浓稠、透明，蛋黄饱满且颜色鲜亮。
----------------------------------------------------------------------------------------------------
Document 3:

制作番茄炒蛋需要哪些基本厨具？
制作番茄炒蛋需要的基本厨具包括：

- 炒锅
- 铲子或 spatula（翻锅工具）
- 刀和砧板（用于切番茄）
- 勺子（用于加调料）

这些基本厨具能够帮助你顺利完成番茄炒蛋的烹饪。
----------------------------------------------------------------------------------------------------
Document 4:

番茄炒蛋的烹饪步骤是什么？
番茄炒蛋的烹饪步骤如下：
1. 鸡蛋打入碗中，加入少许盐，用筷子或打蛋器充分搅拌均匀；番茄洗净后切成小块备用。
2. **炒鸡蛋**：锅内倒入适量食用油加热至温热状态，然后将搅拌好的鸡蛋液缓缓倒入锅中。待鸡蛋凝固时轻轻翻动几下，让其受热均匀直至完全熟透，随后盛出备用。
3. **炒番茄**：在同一锅里留下的底油中放入切好的番茄块，中小火慢慢翻炒至出汁，可根据个人口味加一点点白糖提鲜。
4. **合炒**：当番茄炒至软烂并开始释放大量汤汁时，再把之前炒好的鸡蛋倒回锅里，快速与番茄混合均匀，同时加入适量的盐调味。如果喜欢的话还可以撒上一些葱花增加香气。
5. **完成**：最后检查一下味道是否合适，确认无误后即可关火装盘享用美味的番茄炒蛋啦！
----------------------------------------------------------------------------------------------------
Document 5:

如何根据个人口味调整番茄炒蛋的味道？
要根据个人口味调整番茄炒蛋的味道，可以在步骤4和5中进行以下调整：

- 在炒番茄时（步骤4），可以根据个人口味加入一点点白糖来提鲜。
- 在合炒时（步骤5），除了加盐调味外，还可以根据喜好添加其他调料，如少许鸡精或胡椒粉增加风味。如果喜欢酸甜口味，可以适当增加糖的量；如果喜欢更浓郁的味道，可以在最后撒入一些番茄酱。

通过这些调整，您可以制作出符合个人口味的番茄炒蛋。
========回答========
根据文档中的信息，新手制作番茄炒蛋可以遵循以下步骤：

1. **准备食材和工具**：
   - 鸡蛋：打入碗中。
   - 番茄：洗净后切成小块备用。
   - 基本厨具：炒锅、铲子或 spatula（翻锅工具）、刀和砧板（用于切番茄）、勺子（用于加调料）。

2. **炒鸡蛋**：
   - 在锅内倒入适量食用油，加热至温热状态。
   - 将搅拌好的鸡蛋液缓缓倒入锅中。待鸡蛋开始凝固时轻轻翻动几下，让其受热均匀直至完全熟透。
   - 炒好后将鸡蛋盛出备用。

3. **炒番茄**：
   - 在同一锅里留下的底油中放入切好的番茄块，用中小火慢慢翻炒至出汁。可以根据个人口味加一点点白糖提鲜。

4. **合炒**：
   - 当番茄炒至软烂并开始释放大量汤汁时，将之前炒好的鸡蛋倒回锅里。
   - 快速与番茄混合均匀，并加入适量的盐调味。如果喜欢的话还可以撒上一些葱花增加香气。

5. **完成**：
   - 最后检查一下味道是否合适，确认无误后即可关火装盘享用美味的番茄炒蛋啦！

通过以上步骤，即使是烹饪新手也能轻松制作出美味的番茄炒蛋。可以根据个人口味在炒番茄和合炒时进行适当的调整，以达到更满意的味道。

"""
"""
个人感觉这回答也没什么出彩的，不知道是不是模型进化得太厉害了，这么简单的都没有什么辨别度了
"""
