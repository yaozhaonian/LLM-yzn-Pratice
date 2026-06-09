"""
为 LangGraph 实现一个基于 Redis 的自定义检查点保存器（Custom Checkpoint Saver）。
将对话状态持久化到 Redis 数据库中。
使得 AI 应用具备生产级的高可用性、持久性和可扩展性。
"""
from contextlib import asynccontextmanager, contextmanager
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Iterator,
    List,
    Optional,
    Tuple,
    Literal
)

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

REDIS_KEY_SEPARATOR = "$"

"""
作用：生成存储检查点（Checkpoint）主数据的 Redis Key。
格式：checkpoint$thread_id$namespace$checkpoint_id。
目的：确保每个线程的每个状态快照都有唯一的标识符。
"""
def _make_redis_checkpoint_key(
    thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> str:
    return REDIS_KEY_SEPARATOR.join(["checkpoint", thread_id, checkpoint_ns, checkpoint_id])

"""
作用：生成存储中间写入（Pending Writes，即节点执行过程中的临时输出）的 Redis Key。
格式：writes$thread_id$namespace$checkpoint_id$task_id$idx。
目的：将特定任务在特定检查点产生的中间结果隔离存储，支持断点续传时的状态恢复。
"""
def _make_redis_checkpoint_writes_key(
    thread_id: str, checkpoint_ns: str, checkpoint_id: str, task_id: str, idx: Optional[int]) -> str:
    if idx is None:
        return REDIS_KEY_SEPARATOR.join(["writes", thread_id, checkpoint_ns, checkpoint_id, task_id])
    else:
        return REDIS_KEY_SEPARATOR.join(["writes", thread_id, checkpoint_ns, checkpoint_id, task_id, str(idx)])

"""
作用：反向解析 Redis Key，提取出 thread_id, namespace, checkpoint_id。
目的：在遍历或检索 Key 时，能够从字符串中还原出结构化的配置信息。
"""
def _parse_redis_checkpoint_key(redis_key: str) -> dict:
    namespace, thread_id, checkpoint_ns, checkpoint_id = redis_key.split(
        REDIS_KEY_SEPARATOR
    )
    
    if namespace != "checkpoint":
        raise ValueError("期望检查点密钥以“checkpoint”开头")

    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": checkpoint_id,
    }

"""
作用：反向解析中间写入的 Redis Key，提取出 task_id 和索引 idx。
目的：用于重组和排序同一任务产生的多个中间写入数据。
"""
def _parse_redis_checkpoint_writes_key(redis_key: str) -> dict:
    namespace, thread_id, checkpoint_ns, checkpoint_id, task_id, idx = redis_key.split(
        REDIS_KEY_SEPARATOR
    )
    
    if namespace != "writes":
        raise ValueError("期望检查点密钥以“writes”开头")

    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": checkpoint_id,
        "task_id": task_id,
        "idx": idx
    }

"""
作用：对获取到的 Redis Key 列表进行过滤和排序。
逻辑：
如果指定了 before，只保留 ID 小于该值的检查点。
按 checkpoint_id 降序排序（最新的在前）。
如果指定了 limit，截取前 N 个。
目的：支持 list 方法中的分页和历史回溯功能。
"""
def _filter_keys(
    keys: List[str], before: Optional[RunnableConfig], limit: Optional[int]) -> list:
    """根据可选标准过滤和排序Redis关键字."""
    if before:
        keys = [key for key in keys 
                if _parse_redis_checkpoint_key(key.decode())["checkpoint_id"]
                < before["configurable"]["checkpoint_id"]]
        
    keys = sorted(keys, key=lambda x: _parse_redis_checkpoint_key(x.decode())["checkpoint_id"], reverse=True)
    return keys[:limit] if limit else keys

"""
作用：将从 Redis 读取的原始字节数据反序列化为 PendingWrite 对象列表。
目的：将存储的中间结果还原为 LangGraph 引擎可识别的对象。
"""
def _load_writes(
    serde: SerializerProtocol, task_id_to_data: dict[tuple[str, str], dict]
    ) -> list[PendingWrite]:
    """反序列化准备的写入"""
    writes = [
        (
            task_id,
            data[b"channel"].decode(),
            serde.loads_typed((data[b"type"].decode(), data[b"value"])),
        )
        for (task_id, _), data in task_id_to_data.items()
    ]
    return writes

"""
作用：将从 Redis Hash 中获取的所有字段（checkpoint, metadata, parent_id 等）组装成一个完整的 CheckpointTuple 对象。
目的：这是 get_tuple 的核心数据处理步骤，负责最终的数据交付格式。
"""
def _parse_redis_checkpoint_data(
                                 serde: SerializerProtocol,
                                 key: str,
                                 data: dict,
                                 pending_writes: Optional[List[PendingWrite]] = None,
                                 ) -> Optional[CheckpointTuple]:
    """解析从Redis检索的checkpoint数据"""
    if not data: return None
    
    parsed_key = _parse_redis_checkpoint_key(key)
    thread_id = parsed_key["thread_id"]
    checkpoint_ns = parsed_key["checkpoint_ns"]
    checkpoint_id = parsed_key["checkpoint_id"]
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_ns": checkpoint_ns,
        }
    }
    
    try:
        checkpoint = serde.loads_typed((data[b"type"].decode(), data[b"checkpoint"]))
    except Exception as e:
        print(f"警告: 未能反序列化密钥的检查点 {key}: {e}")
        return None

    # 修复：更稳健地反序列化 metadata
    metadata = {}
    try:
        # 始终期望 metadata_type 和 metadata 成对存在，因为 put 方法中是这样存储的
        if b"metadata_type" in data and b"metadata" in data:
            meta_type = data[b"metadata_type"].decode()
            meta_data = data[b"metadata"]
            metadata = serde.loads_typed((meta_type, meta_data))
        else:
            # 如果确实没有 metadata 字段（极罕见情况），保持空字典
            metadata = {}
    except Exception as e:
        print(f"警告: 未能反序列化密钥的元数据 {key}: {e}. 使用空字典.")
        metadata = {}

    # 确保 metadata 中包含 LangGraph 所需的基本结构，如果缺失则补充默认值
    # 注意：通常 LangGraph 会在 put 时传入完整的 metadata，这里主要是防止解析失败导致的关键键缺失
    if isinstance(metadata, dict):
        if "step" not in metadata:
            metadata["step"] = -1 # 默认初始步骤
        if "source" not in metadata:
            metadata["source"] = "input" 

    parent_checkpoint_id = data.get(b"parent_checkpoint_id", b"").decode()
    parent_config = (
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": parent_checkpoint_id,
            }
        }
        if parent_checkpoint_id
        else None
    )
    
    return CheckpointTuple(
        config=config,
        checkpoint=checkpoint,
        metadata=metadata,
        parent_config=parent_config,
        pending_writes=pending_writes,
    )

class RedisSaver(BaseCheckpointSaver):
    """基于Redis的检查点保护程序实现"""
    conn: Redis
    
    def __init__(self, conn: Redis) -> None:
        """初始化Redis检查点保存程序"""
        super().__init__()
        self.conn = conn

    """
    作用：一个上下文管理器工厂方法。
    目的：简化 Redis 连接的创建和关闭流程，确保在使用结束后自动释放连接资源。
        方法定义：
            cls：类方法固定第一个参数（代表类本身，不是实例）
            *, host:str...：强制关键字参数，必须用 host=xxx 传参，不能按位置传
            返回值：迭代器，生成 RedisSaver 对象
    """   
    @classmethod        # 类方法，不需要实例化类就能调用
    @contextmanager     # 上下文管理器装饰器，配合 yield 实现 with 语法
    def from_conn_info(cls, *, host: str, port: int, db: int) -> Iterator["RedisSaver"]:
        conn = None
        try:
            conn = Redis(host=host, port=port, db=db)
            yield RedisSaver(conn)
        finally:
            if conn:
                conn.close()
                
    """
    作用：保存检查点。
    逻辑：序列化状态数据和元数据，使用 HSET 存入 Redis。
    触发时机：当 LangGraph 完成一个节点的执行并更新状态时调用。
    """   
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """将检查点保存到Redis.

        Args:
            config (RunnableConfig): 与检查点关联的配置.
            checkpoint (Checkpoint): 要保存的检查点.
            metadata (CheckpointMetadata): 与检查点一起保存的附加元数据.
            new_versions (ChannelVersions): 本次写入时的新Channel版本.

        Returns:
            RunnableConfig: 存储检查点后更新了配置.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        key = _make_redis_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        meta_type_, serialized_metadata = self.serde.dumps_typed(metadata)
        
        data = {
            "checkpoint": serialized_checkpoint,
            "type": type_,
            # 修复 2: 存储 metadata 的类型和数据
            "metadata": serialized_metadata,
            "metadata_type": meta_type_, 
            "parent_checkpoint_id": parent_checkpoint_id
            if parent_checkpoint_id
            else "",
        }
        self.conn.hset(key, mapping=data)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    """
    作用：保存中间写入。
    逻辑：将节点执行过程中产生的临时通道值（Channel Values）存入 Redis。使用 HSET 或 HSETNX 防止冲突。
    触发时机：在节点执行过程中，向通道写入数据时调用。
    """
    def put_writes(
        self,
        config: RunnableConfig,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """存储链接到检查点的中间写入.

        Args:
            config (RunnableConfig): 相关检查点的配置.
            writes (Sequence[Tuple[str, Any]]): 要存储的写入列表, 比如 (channel, value) 对.
            task_id (str): 创建写入的任务的标识符.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        for idx, (channel, value) in enumerate(writes):
            key = _make_redis_checkpoint_writes_key(
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                WRITES_IDX_MAP.get(channel, idx),
            )
            type_, serialized_value = self.serde.dumps_typed(value)
            data = {"channel": channel, "type": type_, "value": serialized_value}
            if all(w[0] in WRITES_IDX_MAP for w in writes):
                # 使用HSET将覆盖现有值
                self.conn.hset(key, mapping=data)
            else:
                # 使用HSETNX，它不会覆盖现有值
                for field, value in data.items():
                    self.conn.hsetnx(key, field, value)

    """
    作用：获取单个检查点。
    逻辑：根据 config 查找对应的 Key。如果未指定具体 ID，则通过 _get_checkpoint_key 查找该线程最新的检查点。同时加载关联的中间写入。
    触发时机：当 LangGraph 启动或恢复一个线程时，需要加载上一刻的状态。
    """
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """从Redis获取检查点元组。

            该方法基于提供的配置。如果配置包含“checkpoint_id”密钥，则带有
            检索匹配的thread ID和checkpoint_id。否则，最新的检查点检索给定thread ID.

        Args:
            config (RunnableConfig): 用于检索检查点的配置.

        Returns:
            Optional[CheckpointTuple]: 检索的检查点元组，如果找不到匹配的检查点，则为None.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        checkpoint_key = self._get_checkpoint_key(
            self.conn, thread_id, checkpoint_ns, checkpoint_id
        )
        if not checkpoint_key:
            return None

        checkpoint_data = self.conn.hgetall(checkpoint_key)

        # 加载待处理写入
        checkpoint_id = (
            checkpoint_id
            or _parse_redis_checkpoint_key(checkpoint_key)["checkpoint_id"]
        )
        pending_writes = self._load_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )
        return _parse_redis_checkpoint_data(
            self.serde, checkpoint_key, checkpoint_data, pending_writes=pending_writes
        )

    """
    作用：列出历史检查点。
    逻辑：使用 keys pattern 匹配所有属于该线程的检查点 Key，经过 _filter_keys 处理后，逐个返回 CheckpointTuple。
    用途：用于调试、回放对话历史或实现“回到过去”的功能。
    """   
    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        # TODO: implement filtering
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """列出数据库中的检查点。
        该方法从基于Redis的数据库中检索检查点元组列表
        在提供的配置上。检查点按检查点ID降序排列(最新的排在最前面).

        Args：
            config（RunnableConfig）：用于列出检查点的配置。
            filter（可选[Dict[str，Any]]）：元数据的其他过滤条件。默认为“无”。
            before（Optional[RunnableConfig]）：如果提供，则只返回指定检查点ID之前的检查点。默认为“无”。
            limit（可选[int]）：要返回的检查点的最大数量。默认为“无”。

        Yields：
            Iterator[CheckpointTuple]：检查点元组的迭代器。
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        pattern = _make_redis_checkpoint_key(thread_id, checkpoint_ns, "*")

        keys = _filter_keys(self.conn.keys(pattern), before, limit)
        for key in keys:
            data = self.conn.hgetall(key)
            if data and b"checkpoint" in data and b"metadata" in data:
                # load pending writes
                checkpoint_id = _parse_redis_checkpoint_key(key.decode())[
                    "checkpoint_id"
                ]
                pending_writes = self._load_pending_writes(
                    thread_id, checkpoint_ns, checkpoint_id
                )
                yield _parse_redis_checkpoint_data(
                    self.serde, key.decode(), data, pending_writes=pending_writes
                )

    """作用：内部辅助方法，加载指定检查点关联的所有中间写入数据。"""
    def _load_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> List[PendingWrite]:
        writes_key = _make_redis_checkpoint_writes_key(
            thread_id, checkpoint_ns, checkpoint_id, "*", None
        )
        matching_keys = self.conn.keys(pattern=writes_key)
        parsed_keys = [
            _parse_redis_checkpoint_writes_key(key.decode()) for key in matching_keys
        ]
        pending_writes = _load_writes(
            self.serde,
            {
                (parsed_key["task_id"], parsed_key["idx"]): self.conn.hgetall(key)
                for key, parsed_key in sorted(
                    zip(matching_keys, parsed_keys), key=lambda x: x[1]["idx"]
                )
            },
        )
        return pending_writes

    """作用：内部辅助方法，确定要获取哪个 Key。如果 ID 缺失，则寻找最新的 Key。"""
    def _get_checkpoint_key(
        self, conn, thread_id: str, checkpoint_ns: str, checkpoint_id: Optional[str]
    ) -> Optional[str]:
        """Determine the Redis key for a checkpoint."""
        if checkpoint_id:
            return _make_redis_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        all_keys = conn.keys(_make_redis_checkpoint_key(thread_id, checkpoint_ns, "*"))
        if not all_keys:
            return None

        latest_key = max(
            all_keys,
            key=lambda k: _parse_redis_checkpoint_key(k.decode())["checkpoint_id"],
        )
        return latest_key.decode()


class AsyncRedisSaver(BaseCheckpointSaver):
    """异步基于redis的检查点保护程序实现"""

    conn: AsyncRedis

    def __init__(self, conn: AsyncRedis):
        super().__init__()
        self.conn = conn

    """作用：异步上下文管理器，用于创建和清理异步连接。"""
    @classmethod
    @asynccontextmanager
    async def from_conn_info(
        cls, *, host: str, port: int, db: int
    ) -> AsyncIterator["AsyncRedisSaver"]:
        conn = None
        try:
            conn = AsyncRedis(host=host, port=port, db=db)
            yield AsyncRedisSaver(conn)
        finally:
            if conn:
                await conn.aclose()

    """作用：异步版本的 put,下方函数也是"""
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """ 异步将检查点保存到数据库。
            此方法将检查点保存到Redis。检查点已关联
            使用提供的配置及其父配置（如果有的话）。
            Args：
            config（RunnableConfig）：与检查点关联的配置。
            checkpoint（checkpoint）：要保存的检查点。
            元数据（CheckpointMetadata）：与检查点一起保存的其他元数据。
            new_versions（ChannelVersions）：截至本文撰写时的新频道版本。
            返回：
            RunnableConfig：存储检查点后更新配置。
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        key = _make_redis_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        meta_type_, serialized_metadata = self.serde.dumps_typed(metadata)
        
        data = {
            "checkpoint": serialized_checkpoint,
            "type": type_,
            "metadata": serialized_metadata,
            "metadata_type": meta_type_, # 新增：存储元数据类型
            "parent_checkpoint_id": parent_checkpoint_id
            if parent_checkpoint_id
            else "",
        }

        await self.conn.hset(key, mapping=data)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        for idx, (channel, value) in enumerate(writes):
            key = _make_redis_checkpoint_writes_key(
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                WRITES_IDX_MAP.get(channel, idx),
            )
            type_, serialized_value = self.serde.dumps_typed(value)
            data = {"channel": channel, "type": type_, "value": serialized_value}
            if all(w[0] in WRITES_IDX_MAP for w in writes):
                # Use HSET which will overwrite existing values
                await self.conn.hset(key, mapping=data)
            else:
                # Use HSETNX which will not overwrite existing values
                for field, value in data.items():
                    await self.conn.hsetnx(key, field, value)

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """ 异步从Redis获取检查点元组。
            此方法基于以下内容从Redis检索检查点元组
            提供配置。如果配置包含“checkpoint_id”键，则检查点为
            检索匹配的线程ID和检查点ID。否则，最新检查点
            因为检索到给定的线程ID。
            Args：
            config（RunnableConfig）：用于检索检查点的配置。
            返回：
            可选[CheckpointTuple]：检索到的检查点元组，如果找不到匹配的检查点，则为None。
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        checkpoint_key = await self._aget_checkpoint_key(
            self.conn, thread_id, checkpoint_ns, checkpoint_id
        )
        if not checkpoint_key:
            return None
        checkpoint_data = await self.conn.hgetall(checkpoint_key)

        # load pending writes
        checkpoint_id = (
            checkpoint_id
            or _parse_redis_checkpoint_key(checkpoint_key)["checkpoint_id"]
        )
        pending_writes = await self._aload_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )
        return _parse_redis_checkpoint_data(
            self.serde, checkpoint_key, checkpoint_data, pending_writes=pending_writes
        )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        # TODO: implement filtering
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncGenerator[CheckpointTuple, None]:
        """异步列出Redis中的检查点。

            此方法从基于Redis的检查点元组列表中检索
            在提供的配置上。检查点按检查点ID降序排列（最新的第一个）。

            Args：
            config（可选[RunnableConfig]）：过滤检查点的基本配置。
            filter（可选[Dict[str，Any]]）：元数据的其他过滤条件。
            before（Optional[RunnableConfig]）：如果提供，则只返回指定检查点ID之前的检查点。默认为“无”。
            limit（可选[int]）：要返回的检查点的最大数量。

            Yields：
            AsyncIterator[CheckpointTuple]：匹配检查点元组的异步迭代器。
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        pattern = _make_redis_checkpoint_key(thread_id, checkpoint_ns, "*")
        keys = _filter_keys(await self.conn.keys(pattern), before, limit)
        for key in keys:
            data = await self.conn.hgetall(key)
            if data and b"checkpoint" in data and b"metadata" in data:
                checkpoint_id = _parse_redis_checkpoint_key(key.decode())[
                    "checkpoint_id"
                ]
                pending_writes = await self._aload_pending_writes(
                    thread_id, checkpoint_ns, checkpoint_id
                )
                yield _parse_redis_checkpoint_data(
                    self.serde, key.decode(), data, pending_writes=pending_writes
                )

    async def _aload_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> List[PendingWrite]:
        writes_key = _make_redis_checkpoint_writes_key(
            thread_id, checkpoint_ns, checkpoint_id, "*", None
        )
        matching_keys = await self.conn.keys(pattern=writes_key)
        parsed_keys = [
            _parse_redis_checkpoint_writes_key(key.decode()) for key in matching_keys
        ]
        pending_writes = _load_writes(
            self.serde,
            {
                (parsed_key["task_id"], parsed_key["idx"]): await self.conn.hgetall(key)
                for key, parsed_key in sorted(
                    zip(matching_keys, parsed_keys), key=lambda x: x[1]["idx"]
                )
            },
        )
        return pending_writes

    async def _aget_checkpoint_key(
        self, conn, thread_id: str, checkpoint_ns: str, checkpoint_id: Optional[str]
    ) -> Optional[str]:
        """Asynchronously determine the Redis key for a checkpoint."""
        if checkpoint_id:
            return _make_redis_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        all_keys = await conn.keys(
            _make_redis_checkpoint_key(thread_id, checkpoint_ns, "*")
        )
        if not all_keys:
            return None

        latest_key = max(
            all_keys,
            key=lambda k: _parse_redis_checkpoint_key(k.decode())["checkpoint_id"],
        )
        return latest_key.decode()

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def get_weather(city: Literal["贵州", "拉萨"]):
    """用来获取天气信息"""
    if city == "贵州":
        return "贵州现在也许是阴天"
    elif city == "拉萨":
        return "拉萨一直都是晴天"
    else:
        raise AssertionError("不知道的城市")


tools = [get_weather]
model = ChatOpenAI(
    model_name="qwen2.5:7b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama" 
)

with RedisSaver.from_conn_info(host="localhost", port=6379, db=0) as checkpointer:
    import redis
    r = redis.Redis(host="localhost", port=6379, db=0)
    keys = r.keys("checkpoint$1$*") + r.keys("writes$1$*")
    if keys:
        r.delete(*keys)
        print("已清空 thread_id=1 的历史数据")
    
    graph = create_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    res = graph.invoke({"messages": [("human", "拉萨的天气怎么样")]}, config)

    latest_checkpoint = checkpointer.get(config)
    latest_checkpoint_tuple = checkpointer.get_tuple(config)
    checkpoint_tuples = list(checkpointer.list(config))
    print(latest_checkpoint)


"""
数据控制权：你可以自定义 Key 的前缀、过期策略（TTL）、序列化方式（如使用 JSON 而非 Pickle 以提高兼容性）。
性能优化：可以根据业务需求优化 Redis 命令的使用（如批量写入、管道操作）。
兼容性：适配特定版本的 Redis 集群或云服务商的特殊要求。
"""

"""
应用场景
使用 Redis 作为持久化后端主要适用于以下生产环境场景：

高可用聊天机器人服务：

场景：用户正在与客服机器人对话，服务器突然重启或部署新版本。
优势：由于状态存在 Redis 中，用户刷新页面或重新连接后，机器人能从断点处继续对话，用户体验无缝衔接。MemorySaver 在此场景下会导致所有对话历史丢失。
分布式 Agent 系统：

场景：多个工作节点（Workers）共同处理大量的 Agent 任务。
优势：Redis 作为共享存储，允许任何工作节点读取任意线程的状态。如果节点 A 在处理任务时崩溃，节点 B 可以从 Redis 加载最新状态并接管任务（故障转移）。
长周期任务与人工介入（Human-in-the-loop）：

场景：Agent 执行到一个关键步骤（如“确认转账”），暂停等待用户审批。这个过程可能持续几小时甚至几天。
优势：Redis 支持设置 Key 的过期时间（TTL），可以自动清理过期的临时会话，同时保证等待期间的状态不丢失。
会话历史分析与审计：

场景：需要分析用户的行为路径或审计 AI 的决策过程。
优势：Redis 中的数据可以被其他分析工具读取。通过 list 方法，可以轻松回溯某个用户完整的状态变更历史。


建议
虽然当前测试正常，但在生产环境中，建议注意以下几点：

TTL（过期时间）：Redis 内存宝贵，建议给 Key 设置过期时间。可以在 put 方法中添加 self.conn.expire(key, seconds=3600)。
Thread ID 管理：在实际应用中，不要硬编码 thread_id="1"，应使用用户 ID 或会话 ID 生成唯一的 thread_id。
异常处理：目前的 _parse_redis_checkpoint_data 已经比较健壮，但可以考虑增加日志记录而不是仅仅 print，以便在生产环境中追踪问题。
"""