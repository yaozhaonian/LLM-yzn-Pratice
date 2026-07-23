from services.llm_client import ollama_client
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """
    LLM客户端测试脚本
    
    验证Ollama本地模型能否正常响应。
    """
    print("=" * 60)
    print("         LLM客户端测试")
    print("=" * 60)
    
    try:
        test_prompt = "你好，请问你是谁？"
        print(f"\n测试问题: {test_prompt}")
        print("正在调用模型，请稍候...")
        
        response = ollama_client.chat(test_prompt)
        
        print(f"\n模型响应:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        
        print("\n✅ 测试成功！模型可以正常返回响应")
        logger.info("LLM测试成功")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        logger.error(f"LLM测试失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()
