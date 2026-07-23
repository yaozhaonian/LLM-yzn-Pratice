/**
 * ERP智能客服前端应用
 * 
 * 实现与后端API的交互，管理对话状态和UI渲染。
 */

// 会话ID，使用时间戳和随机数生成
let sessionId = generateSessionId();

// 是否正在发送消息
let isSending = false;

/**
 * 生成会话ID
 * 
 * 使用时间戳和随机数生成唯一的会话标识符。
 * 
 * @returns {string} 会话ID
 */
function generateSessionId() {
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substring(2, 9);
    return `session_${timestamp}_${random}`;
}

/**
 * 处理键盘事件
 * 
 * 监听Enter键，配合Shift键实现换行，单独Enter键发送消息。
 * 
 * @param {KeyboardEvent} event 键盘事件
 */
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

/**
 * 发送消息
 * 
 * 获取用户输入，调用后端API发送消息，并更新UI。
 */
async function sendMessage() {
    const input = document.getElementById('userInput');
    const btnSend = document.getElementById('btnSend');
    const userInput = input.value.trim();

    // 验证输入
    if (!userInput) {
        return;
    }

    // 检查是否正在发送
    if (isSending) {
        return;
    }

    // 设置发送状态
    isSending = true;
    btnSend.disabled = true;
    btnSend.innerHTML = '<span class="loading"><span></span><span></span><span></span></span>';

    // 清空输入框
    input.value = '';

    // 添加用户消息到UI
    addMessage('user', userInput);

    try {
        // 调用后端API
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: sessionId,
                user_input: userInput
            })
        });

        const result = await response.json();

        // 处理API响应
        if (result.code === 0 && result.data && result.data.response) {
            addMessage('bot', result.data.response);
        } else {
            addMessage('bot', `❌ 出错了：${result.msg || '未知错误'}`);
        }

    } catch (error) {
        console.error('发送消息失败:', error);
        addMessage('bot', '❌ 网络连接失败，请稍后重试');
    } finally {
        // 恢复发送状态
        isSending = false;
        btnSend.disabled = false;
        btnSend.innerHTML = '<span>发送</span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 2L11 13"></path><path d="M22 2L15 22L11 13L2 9L22 2Z"></path></svg>';
    }
}

/**
 * 添加消息到UI
 * 
 * 创建消息元素并添加到对话列表中，自动滚动到最新消息。
 * 
 * @param {string} type 消息类型：'user'或'bot'
 * @param {string} text 消息内容
 */
function addMessage(type, text) {
    const chatMessages = document.getElementById('chatMessages');
    
    // 创建消息容器
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type === 'user' ? 'user-message' : 'bot-message'}`;
    
    // 创建头像
    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'avatar';
    avatarDiv.textContent = type === 'user' ? '👤' : '🤖';
    
    // 创建消息内容
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // 创建消息文本
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.innerHTML = formatMessage(text);
    
    // 创建消息时间
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = getCurrentTime();
    
    // 组装消息元素
    contentDiv.appendChild(textDiv);
    contentDiv.appendChild(timeDiv);
    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);
    
    // 添加到对话列表
    chatMessages.appendChild(messageDiv);
    
    // 自动滚动到最新消息
    scrollToBottom();
}

/**
 * 格式化消息内容
 * 
 * 将纯文本消息转换为HTML格式，支持换行和基本的markdown格式。
 * 
 * @param {string} text 原始消息文本
 * @returns {string} 格式化后的HTML
 */
function formatMessage(text) {
    // 替换换行符
    let formatted = text.replace(/\n/g, '<br>');
    
    // 加粗处理（**文本**）
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // 列表处理
    formatted = formatted.replace(/^\s*-\s(.+)$/gm, '<li>$1</li>');
    formatted = formatted.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
    
    return formatted;
}

/**
 * 获取当前时间
 * 
 * 返回格式化的当前时间字符串。
 * 
 * @returns {string} 时间字符串，格式如"14:30"
 */
function getCurrentTime() {
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    return `${hours}:${minutes}`;
}

/**
 * 滚动到底部
 * 
 * 将对话列表滚动到最新消息的位置。
 */
function scrollToBottom() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * 清空对话
 * 
 * 清空对话列表，重新生成会话ID，重置对话状态。
 */
function clearChat() {
    const chatMessages = document.getElementById('chatMessages');
    
    // 保留欢迎消息，移除其他消息
    const welcomeMessage = chatMessages.querySelector('.bot-message');
    chatMessages.innerHTML = '';
    chatMessages.appendChild(welcomeMessage);
    
    // 重新生成会话ID
    sessionId = generateSessionId();
    
    // 滚动到顶部
    chatMessages.scrollTop = 0;
}

/**
 * 页面加载完成后初始化
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('ERP智能客服系统初始化完成');
});