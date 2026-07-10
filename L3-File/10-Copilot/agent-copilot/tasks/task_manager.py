import uuid
from mongoengine import *
import threading
from entity import Task
from utils import logger
import traceback

class TaskManager:
    """
    task实例和状态管理，提供task的生命周期管理方法，在数据库中新建任务、更新任务等。

    __init__ 方法获得访问数据库的连接和设置并发控制。
    """
    def __init__(
        self,
        mongo_host,
        mongo_db,
        mongo_port,
        mongo_user=None,
        mongo_password=None,
        auth_source=None,
    ):
        self.mongo_host = mongo_host
        self.mongo_db = mongo_db
        self.mongo_port = mongo_port
        self.mongo_user = mongo_user
        self.mongo_password = mongo_password
        self.auth_source = auth_source

        try:
            disconnect(alias="default")
        except Exception:
            pass

        kwargs = {
            "host": self.mongo_host,
            "db": self.mongo_db,
            "port": self.mongo_port,
            "alias": "default",
        }
        if self.mongo_user and self.mongo_password:
            kwargs.update(
                {
                    "username": self.mongo_user,
                    "password": self.mongo_password,
                    "authentication_source": self.auth_source or self.mongo_db,
                }
            )

        self.mongo_client = connect(**kwargs)
        
    def create_task(self):
        while True:
            # 创建任务 ID
            task_id = str(uuid.uuid4())
            task = Task.objects(task_id=task_id).first()
            if task is None:
                break
        task = Task()
        task.task_id = task_id
        task.status = 0
        task.edges = []
        task.nodes = []
        task.systemOutput = "初始化任务"
        task.save()
        return task_id
    
    def update_task(self, task_id, nodes, edges, systemOutput, errMsg, isEnd):
        task = Task.objects(task_id=task_id).first()
        if task is None:
            logger.error(f"任务[{task_id}]未找到")
            return False

        if isEnd:
            task.status = -1
        else:
            task.status = task.status + 1
        task.nodes = nodes
        task.edges = edges
        task.systemOutput = systemOutput
        task.isEnd = 1 if isEnd else 0
        task.save()
        return True

    def get_task_by_id(self, task_id):
        try:
            task = Task.objects.get(task_id=task_id)
            if not task:
                logger.warning(f"任务 {task_id} 不存在")
                return None
            return task
        except Exception as e:
            logger.error(f"任务[{task_id}]获取失败: {e}\n{traceback.format_exc()}")
            return None
        
    def get_task(self, task_id):
        return Task.objects(task_id=task_id).first()

if __name__ == "__main__":
    task_manager = TaskManager("localhost", "tools", 27017)
    task_id = task_manager.create_task()
    logger.info(task_id)
