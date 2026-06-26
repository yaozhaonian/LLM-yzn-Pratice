from mongoengine import StringField, IntField, Document, ListField, DateTimeField

class User(Document):
    user_id = IntField(required=True)
    # 用户名 / 登录账号，unique=True 强制账号不能重复，保证登录唯一；
    user_name = StringField(unique=True,required=True)
    password = StringField(required=True)
    user_authority = ListField(StringField(),default=[])
    # create_time = DateTimeField()  # 创建时间
    # last_login = DateTimeField()   # 最后登录时间
    # status = IntField(default=1)  # 账号状态 1正常 0禁用
    # avatar = StringField()         # 头像地址
    # phone = StringField()          # 手机号
