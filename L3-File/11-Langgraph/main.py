from config.settings import settings
from graph.workflow import create_workflow
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """
    ERP智能客服系统主入口
    
    基于LangGraph的MemorySaver机制，实现真正的多轮对话，
    通过固定thread_id维持会话状态，支持上下文延续。
    """
    try:
        logger.info(f"ERP智能客服系统启动，Ollama模型: {settings.ollama.model}")
        workflow = create_workflow()
        
        thread_id = "erp_chat_session"
        
        print("=" * 60)
        print("         ERP智能客服系统")
        print("=" * 60)
        print("  支持：订单查询、库存查询、生产进度查询、发货校验、知识库问答")
        print("  输入 exit/quit/q 退出系统")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n请输入您的问题：")
                
                if user_input.lower().strip() in ["exit", "quit", "q"]:
                    print("\n感谢使用ERP智能客服系统！")
                    logger.info("用户主动退出会话")
                    break
                
                if not user_input.strip():
                    print("请输入有效的问题")
                    continue
                
                logger.info(f"用户输入: {user_input[:50]}...")
                
                result = workflow.invoke(
                    {"user_input": user_input},
                    config={"configurable": {"thread_id": thread_id}}
                )
                
                response = result.get("final_response", "")
                print(f"\n回复：{response}")
                
            except KeyboardInterrupt:
                print("\n\n感谢使用ERP智能客服系统！")
                logger.info("用户通过Ctrl+C退出会话")
                break
            except Exception as e:
                error_msg = f"处理请求时出现异常: {str(e)}"
                print(f"\n错误：{error_msg}")
                logger.error(error_msg)
    
    except Exception as e:
        error_msg = f"系统启动失败: {str(e)}"
        print(f"错误：{error_msg}")
        logger.error(error_msg)


if __name__ == "__main__":
    main()
