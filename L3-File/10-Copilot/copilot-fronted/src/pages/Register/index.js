/**
 * 整体功能：React + antd + react-router 注册页面
 * 实现表单校验、密码一致性校验、调用后端注册接口、成功跳转登录、失败弹出错误提示。
 */
// Form/Input/Input.Password：表单、输入框、密码框,Typography.Title：标题文字,message：全局轻提示（成功 / 失败弹窗）
import { Form, Input, Button, Typography,message } from 'antd';
// UserOutlined 用户昵称图标、LockOutlined 密码锁图标
import { UserOutlined, LockOutlined } from '@ant-design/icons';
// Link：页面跳转标签（不刷新页面）,useNavigate：编程式跳转函数（JS 代码里跳转页面）
import { Link, useNavigate } from 'react-router-dom';
import { RegisterApi } from '../../api/agent';
import './style.css';

export default function Register() {
    /**
    * antd Form 表单实例，用来手动获取表单所有输入值、重置表单、校验等；
    * form.getFieldsValue() 可以拿到全部表单字段。
    */
    const [form] = Form.useForm();
    // 路由跳转方法，navigate('/login') 跳转到登录页
    const navigate = useNavigate();
    // antd 新版消息提示写法，必须在页面渲染 {contextHolder} 才能正常弹出提示框。
    const [messageApi, contextHolder] = message.useMessage();

    const handleSubmit = () => {
        console.log("表单数据", form.getFieldsValue())
        RegisterApi(form.getFieldsValue()).then(res => {
            if(res.status === 200) {
                navigate('/login');
            }
        }).catch(err => {
            // 优先级读取错误文案：后端返回的 message > 请求自带错误信息 > 默认 “注册失败”
            let errorMessage = '注册失败';

            // 优先读取后端返回的错误信息
            if(err.response && err.response.data && err.response.data.message){
                errorMessage = err.response.data.message;
            }else if(err.message){
                errorMessage = err.message;
            }

            // 弹出错误提示
            messageApi.open({
                type: 'error',
                content: errorMessage,
                duration: 5     // 提示框5秒后自动消失
            });

            console.error('注册失败:', err.response || err);
        })
    };

    return (
        <div className='register-container'>
            {contextHolder}
            <div className='register-card'>
                <Typography.Title level={2} className="neon-title">
                    用户注册
                </Typography.Title>

                <Form form={form} className="dark-form">
                    <Form.Item 
                        name="username"
                        rules={[{ required: true, message: '请输入昵称!'}]}>
                        <Input
                            prefix={<UserOutlined />}
                            placeholder="昵称"
                        />
                    </Form.Item>
                    <Form.Item 
                        name="password"
                        rules={[{ required: true, message: '请输入密码!'}]}>
                        <Input.Password
                            prefix={<LockOutlined />}
                            placeholder="密码"
                        />
                    </Form.Item>
                    <Form.Item 
                        name="confirm"
                        dependencies={['password']}
                        rules={[
                            { required: true, message: '请输入密码!'},
                            ({ getFieldsValue }) => ({
                                validator(_, value) {
                                    if(!value || getFieldsValue('password') === value){
                                        return Promise.resolve();
                                    }
                                    return Promise.reject('两次输入的密码不一致!')
                                },
                            }),
                        ]}>
                        <Input.Password
                            prefix={<LockOutlined />}
                            placeholder="确认密码"
                        />
                    </Form.Item>
                    <Button
                        type="primary"
                        block
                        onClick={handleSubmit}
                        className="register-btn"
                    >
                        立即注册
                    </Button>
                    <div className='login-link'>
                        已有账号?<Link to="/login">立即登录</Link>
                    </div>
                </Form>
            </div>
        </div>
    );
}