import './App.css';
/**
 * 路由相关（react-router-dom v6）
 * BrowserRouter as Router：路由根容器，开启前端路由模式（history 模式）
 * Routes：路由匹配容器，替代旧版 Switch
 * Route：单条路由规则，配置路径与对应页面组件
 */
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/Home';
import LoginPage from './pages/Login';
import RegisterPage from './pages/Register';
import { Layout, ConfigProvider } from 'antd';
import { useWindowSize } from 'react-use';
import Dashboard from './pages/Dashboard';
import { useState } from 'react';
import ApiTools from './pages/ApiTools';
import Copilot from './pages/Copilot';

function App() {
    return (
        <ConfigProvider theme={{
            token: {
                colorPrimary: '#1890ff',
                borderRadius: 8
            },
        }}>
            <Layout style={{ minHeight: '100vh' }}>
                <Router>
                    <Routes>
                        <Route path="/" element={<HomePage />} />
                        <Route path="/login" element={<LoginPage />} />
                        <Route path="/register" element={<RegisterPage />} />

                        <Route path="/dashboard" element={<Dashboard />}>
                            <Route path="api-tools" element={<ApiTools />} />
                            <Route path="copilot" element={<Copilot />} />
                        </Route>
                    </Routes>
                </Router>
            </Layout>
        </ConfigProvider>
    );
}

export default App;