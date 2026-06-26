from prompt import PromptModelHub
from utils import logger

class QwenModelPromptHub(PromptModelHub):
    def __init__(self, system_prompt, model_name):
        self.system_prompt = system_prompt
        self.stop_label = None
        self.use_desc = f"使用{model_name}模型提示词===>"

    def get_root_task_prompt_text(self):
        logger.info(f"{self.use_desc}get_root_task_prompt_text")
        return """
        你是专业的API工具规划专家，我会给你用户需求以及可用工具列表。
        请结合用户需求和工具列表判断：该需求是只调用一个API就能完成的单工具任务，还是需要调用多个API的多工具任务。

        如果是单API工具任务，请回答Yes；
        如果是多API工具任务，请回答No，并且输出第一条子任务的自然语言描述，用来匹配对应API。

        注意规则：
        1. 单纯基础信息查询，一般属于单工具任务；
        2. 查询单个商品、单个订单、单个物流供应商相关信息，属于单工具任务；
        3. 创建单个订单、新增单个商品，只用一个工具就能完成，属于单工具任务。

        可用工具列表如下：
        [
        {tool_descs}
        ]

        输出必须严格遵守下面格式：
        Single API tool task: Yes/No
        First subtask description: 用于匹配对应API、用自然语言描述的第一条子任务

        示例1：
        用户需求：先分别查询苹果和梨子的商品信息，再查询商品ID为3的商品详情
        示例1输出：
        Single API tool task: No
        First subtask description: 查询苹果的商品信息

        用户需求：{query}
        直接给出结果，不要输出任何思考过程！
        """
    
    def get_param_task_prompt_text(self):
        logger.info(f"{self.use_desc}get_param_task_prompt_text")
        return """
        你是专业的API工具调用专家。我会提供原始用户需求、当前已提取到的API请求参数。
        请针对**缺失的参数**生成一句自然语言子任务查询描述。
        要求：语句里不能出现已经提取出来的参数名和参数值；句子要带上必要查询条件，方便匹配对应的API接口。

        遵循以下规则：
        1. 生成的子任务描述格式参照「查询XX相关的XX信息」这种句式。
        举例：缺失物流供应商ID时，结合用户需求生成类似“请查询XXX的物流供应商信息”，XXX根据用户原文需求补充限定条件。

        示例1：
        原始用户需求：创建一个产品ID为21，数量109，配送区域南京的订单
        当前已提取参数：
            "quantity": 109,
            "supplierId": "",
            "productId": 21,
            "orderRegion": "南京"
        缺失参数：supplierId：物流供应商ID

        示例1输出：
        请查询能够配送南京的物流供应商信息

        下面是待处理内容，请严格按照示例和规则输出：
        原始用户需求：{query}
        当前已提取参数：{params}
        缺失参数：{missing_param}

        直接输出一句自然语言查询语句，不要输出思考、推理过程。
        """

    def gen_subtask_context_prompt(self, query, context):
        """
        生成用于子任务上下文的提示词。
        该函数根据输入的用户请求和已调用API的上下文信息，生成一个提示词，用于询问模型当前任务是否已完成。
        如果输入的请求为空，则返回停止标签。
        参数:
            query (str): 用户输入的请求内容。
            context (str): 已调用API的上下文信息。
        返回:
            str: 生成的提示词字符串；如果query为空，则返回 self.stop_label。
        """
        logger.info(f"{self.use_desc}gen_subtask_context_prompt")
        REACT_PROMPT = """
        你是资深API工具规划专家，我会给到一条需要多轮调用接口才能做完的用户需求，同时提供前面已经执行完毕的所有API调用上下文记录。
        请结合已有历史调用信息，判断用户整体需求是否全部完成：
        1. 若全部完成，只填写Yes；
        2. 若未完成，填写No，并且用自然语言写出下一条待执行子任务，用于匹配对应API工具。

        重要规则：
        1. 只要为某条子任务选对了匹配的API，且接口正常返回结果，哪怕返回内容为空、空列表[]、提示查无数据，都判定该子任务已完成，代表对应信息不存在；
        2. 各个子任务之间无数据依赖，上一步API返回结果不会影响下一条子任务。

        输出固定格式：
        Has the task been completed: Yes/No
        The Next Subtask Request: 用自然语言描述的下一条待执行子任务

        待处理内容：
        用户原始需求：{query}
        已执行API上下文记录：
        {context}

        直接输出结果，禁止输出任何推理、思考过程！
        """
        API_CONTEXT_DESC = """
        SubTask{index}: {task_description}
        API{index}: {api_description}
        API{index} Response: {api_response}        
        """
        if len(query) == 0:
            return self.stop_label
        
        index = 1
        apis = ""
        for tmp in context:
            api = API_CONTEXT_DESC.format(index=str(index), api_description=tmp["tool"],
                                    api_response=tmp["result"], task_description=tmp["task_description"])
            apis += api
            index += 1

        prompt = REACT_PROMPT.format(query=query, context=apis)
        logger.info(f"[{query}]子任务上下文提示词: {prompt}")
        return prompt
        
    def get_all_parameters_prompt_text(self):
        logger.info(f"{self.use_desc}get_all_parameters_prompt_text")
        return """
        尽可能准确回答以下要求：

        需要提取的参数列表：{arguments}
        将提取出的参数整理为JSON格式对象

        必须严格遵守以下规则：
        1. JSON的键名必须和我给出的参数名称完全一致，保持原有格式不能修改
        2. 提取到的参数值，必须是用户提问原文中实际出现过的文字
        3. 提取商品名称参数时，值里面不要带上“产品”这两个字
        4. 如果参数类型格式为 date-time，日期时间必须遵循示例格式：2025-08-12T13:58:04.094Z
        5. 如果参数类型是枚举enum，参数值只能从给定的枚举列表里挑选其中一项

        输出必须使用规定格式：
        {output}

        用户问题：{query}
        前置子任务的执行结果：{predecessor_tasks}
        """
    


