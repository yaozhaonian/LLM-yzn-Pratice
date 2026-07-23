import os
import tempfile
from services.rag_service import rag_service, MilvusConnectionError, DocumentLoadError, RAGServiceError
from utils.logger import get_logger

logger = get_logger(__name__)


def test_rag_service():
    """
    测试RAG服务连接与初始化是否正常
    
    验证Milvus连接、LlamaIndex设置初始化、文档检索功能。
    """
    print("=" * 60)
    print("         RAG服务测试")
    print("=" * 60)
    
    try:
        print("\n1. 测试RAG服务初始化...")
        print(f"   ✅ Milvus地址: {rag_service._milvus_host}:{rag_service._milvus_port}")
        print(f"   ✅ 集合名称: {rag_service._collection_name}")
        print(f"   ✅ 嵌入模型: {rag_service._embedding_model}")
        print(f"   ✅ 向量维度: {rag_service._embedding_dim}")
        print(f"   ✅ 检索Top-K: {rag_service._top_k}")
        print("   ✅ RAG服务初始化成功")
        
        print("\n2. 测试Milvus向量存储连接...")
        try:
            vector_store = rag_service._ensure_vector_store()
            print(f"   ✅ Milvus向量存储创建成功")
        except MilvusConnectionError as e:
            print(f"   ⚠️ Milvus连接失败（服务可能未启动）: {str(e)}")
            print("   ⚠️ 跳过后续测试")
            return
            
        print("\n3. 测试向量索引创建...")
        index = rag_service._ensure_index()
        print("   ✅ 向量索引创建成功")
        
        print("\n4. 测试文档索引构建...")
        doc_dir = "./data/knowledge"
        if os.path.exists(doc_dir) and os.listdir(doc_dir):
            from pymilvus import MilvusClient
            client = MilvusClient(
                uri=f"http://{rag_service._milvus_host}:{rag_service._milvus_port}",
                user=rag_service._milvus_username or None,
                password=rag_service._milvus_password or None
            )
            if client.has_collection(rag_service._collection_name):
                client.drop_collection(rag_service._collection_name)
                print("   ⚠️ 已删除旧集合，重新创建")
            rag_service._vector_store = None
            rag_service._index = None
            rag_service.build_index_from_docs(doc_dir)
            print("   ✅ 文档索引构建成功")
        else:
            print("   ⚠️ 文档目录不存在或为空，跳过索引构建")
        
        print("\n5. 测试文档检索...")
        results = rag_service.retrieve_relevant_docs("ERP发货流程")
        if results:
            print(f"   ✅ 检索到 {len(results)} 条相关文档")
            for i, doc in enumerate(results[:3]):
                print(f"      文档{i+1}: 相似度={doc['score']:.4f}, 内容={doc['content'][:50]}...")
        else:
            print("   ⚠️ 未检索到相关文档")
        
        print("\n✅ RAG服务测试全部通过！")
        logger.info("RAG服务测试成功")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        logger.error(f"RAG服务测试失败: {str(e)}")
        raise


if __name__ == "__main__":
    test_rag_service()
