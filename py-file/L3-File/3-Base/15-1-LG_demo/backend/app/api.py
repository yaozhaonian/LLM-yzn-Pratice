# API端点（启动、回复、状态等）


from fastapi import FastAPI, APIRouter
from uuid import uuid4
from app.http_req_resp import StartRequest, GraphResponse, ResumeRequest
from app.graph import graph

router = APIRouter()

# 进行graph的调用，处理请求或者反馈
def run_graph_and_response(input_state, config):
    result = graph.invoke(input_state, config)
    print(f"Graph执行后，准备返回用户答复 result:{result}")
    state = graph.get_state(config)
    print(f"Graph执行后，准备返回用户答复 state:{state}")
    next_nodes = state.next
    thread_id = config["configurable"]["thread_id"]
    if next_nodes and "human_feedback" in next_nodes:
        run_status = "user_feedback"
    else:
        run_status = "finished"
    return GraphResponse(thread_id=thread_id, run_status=run_status, assistant_response=result["assistant_response"])

# 用户发出第一次请求
@router.post("/graph/start", response_model=GraphResponse)
def start_graph(request: StartRequest):
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"human_request": request.human_request}
    
    return run_graph_and_response(initial_state, config)

# 用户发出后续请求，比如修改反馈或者批准等
@router.post("/graph/resume", response_model=GraphResponse)
def resume_graph(request: ResumeRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    state = {"status": request.review_action}
    if request.human_comment is not None:
        state["human_comment"] = request.human_comment
    print(f"用户发出后续请求，State to update: {state}")
    graph.update_state(config, state)
    return run_graph_and_response(None, config)





