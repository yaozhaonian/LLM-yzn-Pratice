from config.settings import settings
from graph.workflow import create_workflow
from utils.logger import get_logger

logger = get_logger(__name__)


def test_chat_flow():
    """
    测试ERP智能客服系统的多轮对话流程
    
    模拟用户输入，验证系统追问机制是否正常工作。
    """
    print("=" * 60)
    print("         ERP智能客服系统 - 多轮对话测试")
    print("=" * 60)
    
    try:
        logger.info(f"ERP智能客服系统启动，Ollama模型: {settings.ollama.model}")
        workflow = create_workflow()
        
        thread_id = "erp_chat_session"
        
        print("\n--- 测试场景1: 查询订单状态（需要追问订单号）---")
        user_input = "查询订单状态"
        print(f"\n用户输入: {user_input}")
        print("-" * 60)
        
        result = workflow.invoke(
            {"user_input": user_input},
            config={"configurable": {"thread_id": thread_id}}
        )
        
        response = result.get("final_response", "")
        intent = result.get("intent", "未知")
        need_more_params = result.get("need_more_params", False)
        missing_params = result.get("missing_params", [])
        
        print(f"识别意图: {intent}")
        print(f"需要更多参数: {need_more_params}")
        print(f"缺失参数: {missing_params}")
        print(f"回复：{response}")
        
        print("\n--- 测试场景2: 提供订单号后查询 ---")
        user_input = "ORD20260710001"
        print(f"\n用户输入: {user_input}")
        print("-" * 60)
        
        result = workflow.invoke(
            {"user_input": user_input},
            config={"configurable": {"thread_id": thread_id}}
        )
        
        response = result.get("final_response", "")
        intent = result.get("intent", "未知")
        dao_result = result.get("dao_query_result", {})
        
        print(f"识别意图: {intent}")
        print(f"数据查询结果: {dao_result}")
        print(f"回复：{response}")
        
        print("\n--- 测试场景3: 查询库存（需要追问物料编码）---")
        user_input = "库存情况怎么样"
        print(f"\n用户输入: {user_input}")
        print("-" * 60)
        
        result = workflow.invoke(
            {"user_input": user_input},
            config={"configurable": {"thread_id": thread_id}}
        )
        
        response = result.get("final_response", "")
        intent = result.get("intent", "未知")
        need_more_params = result.get("need_more_params", False)
        missing_params = result.get("missing_params", [])
        
        print(f"识别意图: {intent}")
        print(f"需要更多参数: {need_more_params}")
        print(f"缺失参数: {missing_params}")
        print(f"回复：{response}")
        
        print("\n--- 测试场景4: 提供物料编码后查询库存 ---")
        user_input = "MAT001"
        print(f"\n用户输入: {user_input}")
        print("-" * 60)
        
        result = workflow.invoke(
            {"user_input": user_input},
            config={"configurable": {"thread_id": thread_id}}
        )
        
        response = result.get("final_response", "")
        intent = result.get("intent", "未知")
        dao_result = result.get("dao_query_result", {})
        
        print(f"识别意图: {intent}")
        print(f"数据查询结果: {dao_result}")
        print(f"回复：{response}")
        
        print(f"\n{'='*60}")
        print("✅ 多轮对话测试完成！")
        logger.info("多轮对话测试完成")
        
    except Exception as e:
        error_msg = f"系统启动失败: {str(e)}"
        print(f"❌ 错误：{error_msg}")
        logger.error(error_msg)


if __name__ == "__main__":
    test_chat_flow()
