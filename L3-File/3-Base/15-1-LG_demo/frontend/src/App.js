import React, {useState, useRef, useEffect} from "react";
import AssistantService from "./AssistantService";
import ReactMarkdown from "react-markdown";
/*
React + 3 个核心 Hook：
    useState：管理页面所有状态（输入框、对话、UI 状态等）
    useRef：获取 DOM 元素（用于自动聚焦反馈输入框）
    useEffect：监听状态变化，执行副作用（自动聚焦）
AssistantService：后端接口封装（发送问题、提交反馈、批准）
ReactMarkdown：把 AI 返回的 Markdown 文本渲染成富文本（标题、列表、代码块等）
*/


/*
uiState：控制整个页面显示什么（空闲 / 等待 / 反馈 / 完成）
history：存储完整对话流，用户消息 + AI 消息 + 加载占位
threadId：保证同一轮对话不会混乱
*/
const App = () =>{
    // UI 状态: 空闲、等待中、用户反馈、已完成
    const [uiState, setUiState] = useState("idle");

    // 用户问题
    const [question, setQuestion] = useState("");

    // AI 最终回答
    const [assistantResponse, setAssistantResponse] = useState("");

    // 用户反馈内容
    const [feedback, setFeedback] = useState("");

    // 对话唯一ID(后端用来关联同一场对话)
    const [threadId, setThreadId] = useState("");

    // 对话历史数组: [{role: "user"/"assistant", content: "内容"}]
    const [history, setHistory] = useState([]);


    // 当 UI 切换到反馈模式时，自动把光标定位到反馈输入框，提升体验
    const feedbackInputRef = useRef(null);
    useEffect(() => { 
        if (uiState === "feedback_form" && feedbackInputRef.current){
            feedbackInputRef.current.focus();
        }
    }, [uiState]);


    /* 
    handleStart：用户发送第一个问题
    作用：
        用户点「发送」触发
        先显示加载动画
        请求后端
        拿到 AI 回答后更新到页面
    */
    const handleStart = async () =>{
        setUiState("waiting");                  // 切换到加载状态
        setHistory([                          // 先把用户问题显示到界面
            {role: "user", content: question},
            {role: "assistant", content: null} // null = 加载动画
        ]);
        try {
            // 调用后端接口发起对话
            const data = await AssistantService.startConversation(question);

            // 更新 AI 回答、对话历史、对话 ID
            setAssistantResponse(data.assistant_response);
            setUiState("idle");
            setThreadId(data.thread_id);
            setHistory([
                {role: "user", content: question},
                {role: "assistant", content: data.assistant_response}
            ]);
        } catch (error) {
            // 遇到错误，切换到错误状态
            alert("handleStart无法联系后端");
            console.error("handleStart错误信息:", error);
            setUiState("idle");
        }
    };



    /* 
    handleApprove：用户批准 AI 回答
    作用：
        用户觉得回答不好，输入改进意见
        提交反馈 → AI 重新生成答案
        回到可再次审核的状态
    */
    const handleApprove = async () =>{
        setUiState("waiting");
        // 把用户反馈加入历史, 并显示加载
        setHistory([  
            ...history,
            {role: "assistant", content: null}
        ]);

        try {
            // 提交批准给后端
            const data = await AssistantService.submitReview({
                thread_id: threadId,
                review_action: "approved"
            });

            // 更新回答
            setAssistantResponse(data.assistant_response);
            setUiState("finished"); // 切换到「已完成」
            // 替换加载动画为真实回答
            setHistory(prev => [
                ...prev.slice(0, -1), // 删除加载动画
                {role: "assistant", content: data.assistant_response}
            ]);
        } catch (error) {
            alert("handleApprove无法联系后端");
            console.error("handleApprove错误信息:", error);
            setUiState("idle");
        }
    };

    /*
    handleFeedback：用户提交改进反馈
    作用：
        用户觉得回答不好，输入改进意见
        提交反馈 → AI 重新生成答案
        回到可再次审核的状态
    */
    const handleFeedback = async () => {
        setUiState("waiting");
        // 把用户反馈加入历史, 并显示加载
        setHistory([  
            ...history,
            {role: "user", content: feedback},
            {role: "assistant", content: null}
        ]);

        try {
            // 提交反馈给后端
            const data = await AssistantService.submitReview({
                thread_id: threadId,
                review_action: "feedback",
                human_comment: feedback
            })
            // 获取新的回答
            setAssistantResponse(data.assistant_response);
            setUiState("idle");
            setHistory(prev => [
                ...prev.slice(0, -1),
                {role: "assistant", content: data.assistant_response}
            ]);
            setFeedback("");
        } catch (error) {
            alert("handleFeedback无法联系后端");
            console.error("handleFeedback错误信息:", error);
            setUiState("idle");
        }

    }

    /*
    页面渲染（JSX）
    根据 uiState 和 history 动态显示不同界面。
    整体布局
        左侧：图片 + 标题「节点关系图」
        右侧：AI 对话主界面
    特色功能
        加载动画：AI 回复中显示旋转圆圈
        Markdown 渲染：AI 回答支持富文本
        对话标签：自动标注「初始请求」「反馈」「你已批准」
        开始新对话：一键清空所有状态
    */ 
    return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start', margin: '40px auto', fontFamily: 'sans-serif' }}>
      <div style={{ flex: '0 0 320px', maxWidth: 320, marginRight: 32, background: '#fafbfc', borderRadius: 8, border: '1px solid #eee', padding: 16, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <img src="/LG-demo.png" alt="HITL Graph" style={{ width: '75%', height: 'auto', borderRadius: 6, boxShadow: '0 2px 12px #0001', marginBottom: 16 }} />
        <div style={{ fontSize: 16, color: '#444', textAlign: 'center' }}>节点关系图</div>
      </div>
      <div style={{ maxWidth: 600, width: '95%', padding: 24, border: '1px solid #eee', borderRadius: 8, position: 'relative', background: '#fff' }}>
        <button
          onClick={() => {
            setUiState("idle");
            setQuestion("");
            setAssistantResponse("");
            setFeedback("");
            setThreadId(null);
            setHistory([]);
          }}
          style={{ position: "absolute", top: 24, right: 24, padding: "8px 18px", fontSize: 16, borderRadius: 6, background: "#f5f5f5", border: "1px solid #ddd", cursor: "pointer" }}
        >
          开始新的对话
        </button>
        <h2>支持人类干预的AI助手</h2>
        {uiState === "idle" && history.length === 0 && (
          <div>
            <input
              type="text"
              placeholder="有什么可以帮你的？向我提问吧..."
              value={question}
              onChange={e => setQuestion(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter") handleStart(); }}
              style={{ width: "70%", padding: 12, fontSize: 18, borderRadius: 6, border: '1px solid #bbb', marginRight: 8 }}
            />
            <button onClick={handleStart} style={{ padding: '12px 32px', fontSize: 20, borderRadius: 6, border: '1px solid #bbb', background: '#f5f5f5', cursor: 'pointer', height: 48 }}>发送</button>
          </div>
        )}

        {history.length > 0 && (
          <div style={{ margin: "24px 0" }}>
            {history.map((msg, idx) => {
              // Hide the last assistant message if in finished state (it's shown in Final Version block)
              if (uiState === "finished" && msg.role === "assistant" && idx === history.length - 1) {
                return null;
              }
              // Determine hint for each message
              let hint = null;
              if (msg.role === "user") {
                hint = idx === 0 ? "初始请求" : "反馈";
              } else if (msg.role === "assistant" && idx === history.length - 1 && uiState === "finished") {
                hint = "你已批准";
              }
              return (
                <div key={idx} style={{ textAlign: msg.role === "user" ? "right" : "left", margin: "8px 0" }}>
                  {hint && (
                    <div style={{ fontSize: 12, color: "#888", marginBottom: 2 }}>{hint}</div>
                  )}
                  <span style={{
                    fontWeight: msg.role === "user" ? 600 : 700,
                    color: msg.role === "assistant" ? '#1976d2' : undefined,
                    background: msg.role === "assistant" ? 'rgba(25, 118, 210, 0.08)' : undefined,
                    padding: msg.role === "assistant" ? '2px 8px' : undefined,
                    borderRadius: msg.role === "assistant" ? 4 : undefined
                  }}>
                    {msg.role === "user" ? "用户: " : "AI助手: "}
                  </span>
                  {msg.role === "assistant" && msg.content === null ? (
                    <div style={{ display: "inline-block", verticalAlign: "middle", marginLeft: 6 }}>
                      <div style={{
                        border: "4px solid #eee",
                        borderTop: "4px solid #333",
                        borderRadius: "50%",
                        width: 24,
                        height: 24,
                        animation: "spin 1s linear infinite",
                        display: "inline-block"
                      }} />
                      <style>{`
                        @keyframes spin {
                          0% { transform: rotate(0deg); }
                          100% { transform: rotate(360deg); }
                        }
                      `}</style>
                    </div>
                  ) : msg.role === "assistant" ? (
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  ) : (
                    msg.content
                  )}
                </div>
              );
            })}
          </div>
        )}
        {uiState === "user_feedback" && (
          <div style={{ marginTop: 24, background: '#f8fafd', border: '1px solid #e3eaf2', borderRadius: 6, padding: 18 }}>
            <div style={{ marginBottom: 8, fontWeight: 600 }}>请提供反馈以改进助手的回答:</div>
            <textarea
              ref={feedbackInputRef}
              value={feedback}
              onChange={e => setFeedback(e.target.value)}
              rows={3}
              style={{ width: '95%', padding: 12, fontSize: 18, borderRadius: 6, border: '1px solid #bbb', resize: 'vertical' }}
              placeholder="您的反馈..."
            />
            <div style={{ marginTop: 8 }}>
              <button
                onClick={handleFeedback}
                style={{ marginRight: 8, padding: "8px 24px", height: 48, fontSize: 20 }}
              >
                提交反馈
              </button>
              <button
                onClick={() => {
                  setUiState("idle");
                  setFeedback("");
                }}
                style={{ padding: "8px 24px", height: 48, fontSize: 20 }}
              >
                取消
              </button>
            </div>
          </div>
        )}
        {uiState === "waiting" && null}
        {uiState === "idle" && (
          // Only show review buttons if there is an assistant response
          (assistantResponse || (history.length > 0 && history[history.length - 1].role === "assistant" && history[history.length - 1].content)) && (
            <div style={{ marginTop: 24, textAlign: 'right' }}>
              <button
                onClick={handleApprove}
                style={{ marginRight: 8, padding: "8px 24px", height: 48, fontSize: 20 }}
              >
                批准
              </button>
              <button
                onClick={() => setUiState("user_feedback")}
                style={{ padding: "8px 24px", height: 48, fontSize: 20 }}
              >
                发送反馈
              </button>
            </div>
          )
        )}
        {uiState === "finished" && (
          <div>
            <div style={{ textAlign: "right", margin: "8px 0" }}>
              <div style={{ fontSize: 12, color: "#888", marginBottom: 2 }}>你已批准</div>
              <span style={{ fontWeight: 600 }}>用户: </span>已认可答案
            </div>
            <div style={{ marginTop: 24, padding: 12, background: "#f6f6f6", borderRadius: 6 }}>
              <span style={{
                fontWeight: 700,
                color: '#1976d2',
                background: 'rgba(25, 118, 210, 0.08)',
                padding: '2px 8px',
                borderRadius: 4,
                marginBottom: 8,
                display: 'inline-block'
              }}>
                AI助手:
              </span>
              <strong style={{ marginLeft: 8 }}>最终答案:</strong>
              <ReactMarkdown>{assistantResponse}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;