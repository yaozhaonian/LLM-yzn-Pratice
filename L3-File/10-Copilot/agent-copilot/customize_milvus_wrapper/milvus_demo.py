import numpy as np
from pymilvus import db, MilvusClient, DataType, connections
from pymilvus.client.types import LoadState
from urllib.parse import urlparse

if __name__ == '__main__':
    uri = "http://127.0.0.1:19530"
    db_name = "test_db_v1"
    collection_name = "test_collection_v1"
    url_parsed = urlparse(uri)
    conn = connections.connect(host=url_parsed.hostname, port=url_parsed.port)
    # 查看当前milvus中的数据库
    databases = db.list_database()
    print("当前milvus中的数据库:", databases)
    # 如果db_name 不在数据库中，就创建数据库
    if db_name not in databases:
        db.create_database(db_name)
    # 创建milvus客户端
    client = MilvusClient(uri=uri, db_name=db_name)
    # 创建collection schema:创建集合结构模板，定义字段类型、主键、向量维度、动态字段开关
    schema = MilvusClient.create_schema(enabled_dynamic=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=1024)
    
    # 创建索引
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="IP",
        index_type="FLAT",
        params={"nlist": 2048}
    )
    # 创建collection
    if collection_name in client.list_collections():
        client.drop_collection(collection_name)
    
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    
    # 随机生成10条数据
    datas = []
    query_array = None
    for i in range(10):
        vectors_array = np.random.rand(1024)
        if query_array is None:
            query_array = vectors_array
        datas.append({
            "id": i,
            "embedding": vectors_array
        })
        
    # 如果 collection 未加载在内存中,请先加载内存
    if client.get_load_state(collection_name)['state'] == LoadState.NotLoad:
        client.load_collection(collection_name)
        
    # 插入这十条数据
    client.insert(collection_name=collection_name, data=datas, batch_size=len(datas))
    
    # 生成随机的查询向量
    vectors_array = np.random.rand(1024)
    
    # 从milvus中进行查询操作
    results = client.search(
        collection_name=collection_name, 
        data=[vectors_array.tolist()],
        search_params={"metric_type": "IP", "params": {"nprobe": 16}},
        limit=100,
        output_fields=["id"],
        consistency_level="Bounded"
    )
    # 对milvus结果进行解析
    docs = []
    print("查询结果1:\n", results)
    for result in results:
        for res in result:
            chunk = res["entity"]
            docs.append({"id": chunk["id"]})
    print("查询结果2:\n", docs)
    print("再次看一下当前milvus中的数据库", db.list_database())
            
    








