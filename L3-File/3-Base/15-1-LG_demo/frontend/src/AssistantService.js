// 为助理会话/对话API调用提供集中服务

const BASE_URL = "http://localhost:8000";

export default class AssistantService {
  static async startConversation(human_request) {
    const response = await fetch(`${BASE_URL}/graph/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({human_request:human_request})
    });
    if (!response.ok) throw new Error("网络响应异常");
    return response.json();
  }

  static async submitReview({ thread_id, review_action, human_comment }) {
    const body = { thread_id, review_action };
    if (human_comment) body.human_comment = human_comment;
    const response = await fetch(`${BASE_URL}/graph/resume`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (!response.ok) throw new Error("网络响应异常");
    return response.json();
  }
}






