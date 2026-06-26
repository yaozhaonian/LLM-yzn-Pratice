from mongoengine import Document, StringField, ListField, DictField, IntField

class Task(Document):
    task_id = StringField()         # 任务唯一标识，字符串类型
    status = IntField()             # 任务状态码，数字（例：0待执行、1运行中、2成功、3失败）
    nodes = ListField(DictField())  # 节点数组，数组内每个元素是字典
    # ListField(DictField())：存储形如 [{}, {}, {}] 的数据
    edges = ListField(DictField())  # 连线/边数组，数组内每个元素是字典
    isSuccess = StringField()       # 是否执行成功，字符串（"true"/"false"）
    systemOutput = StringField()    # 系统输出日志、执行结果文本
    
    def to_dict(self):
        """将 Task 对象转换为字典"""
        return {
            'task_id':self.task_id,
            'status': self.status,
            'nodes': self.nodes,
            'edges': self.edges,
            'isSuccess': self.isSuccess,
            'systemOutput': self.systemOutput
        }

"""
这个是工作流 / 流程图任务存储实体：
每条记录代表一次完整流程任务；
nodes 存储流程图所有节点信息；
edges 存储节点之间的连线关系；
status、isSuccess、systemOutput 记录任务运行状态和日志；
to_dict 对外提供标准字典格式，用于接口返回。
"""
