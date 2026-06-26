from mongoengine import connect, Document, StringField, ListField, DictField, IntField

# 1. 连接数据库
connect(db="task_db")

# 2. 定义模型
class Task(Document):
    task_id = StringField(required=True, unique=True)
    status = IntField(default=0)
    nodes = ListField(DictField(), default=[])
    edges = ListField(DictField(), default=[])
    isSuccess = StringField(default="false")
    systemOutput = StringField(default="")

    meta = {
        "indexes": ["task_id"],
        "collection": "task_list"
    }

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "status": self.status,
            "nodes": self.nodes,
            "edges": self.edges,
            "isSuccess": self.isSuccess,
            "systemOutput": self.systemOutput
        }

# 3. 新增
import time
unique_task_id = f"task_{int(time.time())}"

Task.objects.create(
    task_id=unique_task_id,  # 使用唯一的 ID
    status=0,
    nodes=[{"id": "n1", "name": "起始节点"}],
    edges=[{"source": "n1", "target": "n2"}]
)

# 4. 查询单条
task = Task.objects.get(task_id=unique_task_id)
print("查询单条:",task.to_dict())

# 5. 更新
Task.objects(task_id=unique_task_id).update(status=2, isSuccess="true", systemOutput="运行成功")

# 6.insert () 批量插入多条      (PS.因为task_id设置为主键，所以多次运行相同代码会出错)
task_batch = [
    Task(
        task_id="task_009",
        status=1,
        nodes=[{"id": "n2", "name": "计算节点"}],
        edges=[{"source": "n2", "target": "n3"}]
    ),
    Task(
        task_id="task_010",
        status=2,
        isSuccess="true",
        systemOutput="流程执行完毕"
    )
]
insert_result = Task.objects.insert(task_batch)
print("批量插入生成的主键ID列表：", insert_result)

print("批量插入完成，当前数据库中所有任务记录：")
# 6. 批量查询
success_tasks = Task.objects().all()
for item in success_tasks:
    print(item.to_dict())

# 7. 删除
Task.objects(task_id=unique_task_id).delete()