from pydantic import BaseModel
from typing import Literal, Optional

# 用户初始请求
class StartRequest(BaseModel):
    human_request: str

# 用户后续请求
class ResumeRequest(BaseModel):
    thread_id: str
    human_comment: Optional[str] = None
    review_action: Literal["approved", "feedback"]

# 给前端界面的答复
class GraphResponse(BaseModel):
    thread_id: str
    assistant_response: Optional[str] = None
    run_status: Literal["finished", "user_feedback"]

