from mongoengine import EmbeddedDocument, EmbeddedDocumentField, StringField, BooleanField, ListField, DictField, IntField, IntField, Document


# 
class Parameter(EmbeddedDocument):
    """参数实体类，表示任务的参数信息"""
    name = StringField()  # 参数名称
    value = StringField()  # 参数值
    type = StringField()   # 参数类型（如字符串、整数等）
    required = BooleanField(default=False)  # 是否为必填参数
    format = StringField()  # 参数格式（如日期格式、数值范围等）
    description = StringField()  # 参数描述信息
    enum = ListField(StringField())  # 枚举值列表，表示参数的可选值
    in_ = StringField()  # 参数位置（如query、path、body等）

class Tool(Document):
    """工具实体类，表示一个工具的基本信息"""
    tool_id = IntField()  # 工具唯一标识
    name_for_human = StringField()  # 工具名称（面向用户的可读名称）
    name_for_model = StringField()  # 工具名称（面向模型的内部名称）
    description = StringField()  # 工具描述信息
    # category = StringField()  # 工具类别（如数据处理、分析、可视化等）
    # version = StringField()  # 工具版本号
    # author = StringField()  # 工具作者或开发者信息
    # license = StringField()  # 工具使用许可信息
    # tags = ListField(StringField())  # 工具标签列表，用于分类和搜索
    isValidate = BooleanField(default=True)  # 工具是否有效，默认为 True
    operationId = StringField() # 工具操作 ID，用于唯一标识工具的操作
    api_url = StringField() # 工具 API 接口 URL，用于调用工具的接口
    path = StringField()    # 工具 API 访问路径
    method = StringField(required=True) 
    request_body = ListField(EmbeddedDocumentField(Parameter))  # 工具请求体参数列表



