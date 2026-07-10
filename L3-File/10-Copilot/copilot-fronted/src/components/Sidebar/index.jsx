
import { useState } from 'react';
// 作用：用于在函数组件中定义、管理页面本地状态（比如侧边栏展开收起、菜单选中项等）
import { Layout, Menu, Button } from 'antd';
/**
 * Layout：页面布局容器（侧边栏 + 头部 + 内容区经典后台布局）
 * Menu：侧边 / 顶部导航菜单
 * Button：通用按钮组件
 */
import { useNavigate } from 'react-router-dom';
/**
 * useNavigate：路由跳转钩子，替代老版本 useHistory
 * 作用：代码里控制页面跳转，不用 a 标签
 */
import { ApiOutlined, ThunderboltOutlined, LogoutOutlined } from '@ant-design/icons';
/**
 * 从 antd 图标库导入 4 个线性图标，放在菜单 / 按钮前面做装饰：
 * ApiOutlined：接口 / API 图标
 * ThunderboltOutlined：闪电、快速执行
 * LogoutOutlined:登出图标，一般放在右上角退出登录按钮旁
 */

const { Sider } = Layout;


export default function AppSider() {
    const navigate = useNavigate();
    const [collapsed, setCollapsed] = useState(false)

    const menuItems = [
        {
            key: 'api-tools',
            icon: <ApiOutlined />,
            label: ' API工具',
            onClick: () => navigate('/dashboard/api-tools')
        },
        {
            key: 'copilot',
            icon: <ThunderboltOutlined />,
            label: ' Copilot执行',
            onClick: () => navigate('/dashboard/coplilot')
        }
    ];

    const handleLogout = () => {
        // 退出登录逻辑（需根据实际认证方案实现）
        localStorage.removeItem('token');
        window.location.href = '/login';
    };

    return (
        <Sider
            collapsible
            collapsed={collapsed}
            onCollapse={(value) => setCollapsed(value)}
            className="glow-sidebar"
            width={250}
        >
            <div className='header-content' style={{ display: 'flex', alignItems: 'center', margin: '10px' }}>
                <img src="/logo192.png" className='nav-logo' style={{ width: 32, marginRight: 12,marginLeft:12 }} alt="192的图" />
                <span className='platfrom-name' style={{ color: '#fff', fontSize: 18 }}>Copilot智能辅助助手</span>               
            </div>
            <div className="sidebar-header" style={{ position: 'relative' }}>
                <Button
                    type="primary"
                    danger
                    icon={<LogoutOutlined />}
                    onClick={handleLogout}
                    style={{
                    position: 'absolute',
                    top: 'calc(30% - 48px)',
                    left: 24,
                    right: 24,
                    width: 'calc(100% - 48px)'
                    }}
                >退出系统</Button>
            </div>
            <Menu
                theme="dark"
                mode="inline"
                items={menuItems}
                className="nav-menu"
                style={{ marginTop: '40px', gap: '8px' }}
             />
        </Sider>
    );
}








