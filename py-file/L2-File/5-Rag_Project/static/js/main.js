// 全局工具函数
function showNotification(message, type = 'info') {
    const div = document.createElement('div');
    div.className = `notification ${type}`;
    div.textContent = message;
    document.body.appendChild(div);
    // div.style.transition = 'opacity 0.3s';
    // document.body.appendChild(div);

    // 3秒后淡出再删除
    // setTimeout(() => {
    //     div.style.opacity = 0;
    //     setTimeout(() => div.remove(), 300);
    // }, 3000);
    
    setTimeout(() => div.remove(), 3000);
}

// 页面加载时检查知识库状态
document.addEventListener('DOMContentLoaded', async () => {
    // 可以在这里添加初始化逻辑
    // RAG项目初始化逻辑示例：检查知识库状态
    // try {
    //     const res = await fetch('/api/rag/knowledge-base/status');
    //     const data = await res.json();
    //     if (data.ready) {
    //         showNotification('知识库已就绪', 'success');
    //     } else {
    //         showNotification('知识库加载中...', 'info');
    //     }
    // } catch (err) {
    //     showNotification('检查知识库失败：' + err.message, 'error');
    // }
});