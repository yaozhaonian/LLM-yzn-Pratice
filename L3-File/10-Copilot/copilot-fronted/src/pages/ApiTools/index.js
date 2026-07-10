// AI Agent 工具列表管理页，展示所有 API 工具、单条删除、一键清空全部、上传 JSON 文件批量导入工具。
// 技术栈：React 函数组件 + useState/useEffect + antd 组件库 + 后端接口请求。
import { Table, Button, Upload, message } from 'antd';
import { DeleteOutlined, UploadOutlined } from '@ant-design/icons';
import { ToolsApi, DeleteAllToolApi, UploadFileApi, DeleteOneToolApi } from '../../api/agent';
import { useState, useEffect } from 'react';

const demoData = [
    {
        "key": "1",
        "index": 1,
        "name": "用户查询API",
        "description": "通过用户ID查询详细信息",
        "params": "123",
        "method": "GET"
    },
    {
        "key": "2",
        "index": 2,
        "name": "订单创建API",
        "description": "创建新的订单记录",
        "params": "123",
        "method": "POST"
    },
    {
        "key": "3",
        "index": 3,
        "name": "订单创建API",
        "description": "创建新的订单记录",
        "params": "123",
        "method": "POST"
    }, {
        "key": "4",
        "index": 4,
        "name": "订单创建API",
        "description": "创建新的订单记录",
        "params": "123",
        "method": "POST"
    }
];

export default function ApiTools() {
    const [data, setData] = useState(demoData);
    const delete_all_tool = () => {
        DeleteAllToolApi().then(res => {
            if(res.status === 200){
                handleGetAllData();
                message.success('清空成功');
            }
        }).catch(err=>{
            console.log('ApiTools出错:', err);
            message.error('清空失败');
        })
    };

    const handleGetAllData = () => {
        ToolsApi().then(res => {
            console.log('当前参数:', res);
            if(res.status === 200){
                setData(res.data.results);
            }
        }).catch(err => {
            console.log('handleGetAllData出错:', err);
        })
    };

    useEffect(() => {
        handleGetAllData();
    }, []);

    // 表格列配置
    const columns = [
        { title: '序号', dataIndex: 'index' },
        { title: 'API名称', dataIndex: 'name' },
        { title: 'API描述', dataIndex: 'description' },
        { title: '请求参数', dataIndex: 'params' },{ title: '请求方法', dataIndex: 'method' },
        {
            title: '操作',
            render: (_, record) => (
                <>
                    <Button danger icon={<DeleteOutlined />} onClick = {() => {
                        console.log('record', record);
                        DeleteOneToolApi({"ids":[record.key]}).then(res => {
                            if(res.status === 200){
                                handleGetAllData();
                                message.success('删除成功');
                            }
                        }).catch(err => {
                            console.log('删除失败', err);
                            message.error('删除失败');
                        })
                    }} />
                </>
            )
        }
    ];

    return (
        <div className='api-tools-page'>
            { /* 顶部工具栏区域 */ }
            <div className='table-header'>
                <h3 className='neon-title'></h3>
                <div className='toolbar-right'>
                    <Button
                        danger
                        className='danger-btn'
                        onClick={delete_all_tool}
                        style={{ marginRight: 16 }}
                    >清空所有工具</Button>
                    <Upload
                        name='file'
                        customRequest={({ file }) => {
                            UploadFileApi(file).then(res => {
                                if(res.status === "success"){
                                    message.success('上传成功');
                                    handleGetAllData();
                                }
                            }).catch(err => {
                                message.error('上传失败' + err.message)
                            })
                        }}
                        showUploadList={false}
                        beforeUpload={file => {
                            const isJSON = file.type === 'application/json';
                            if(!isJSON){
                                message.error('仅支持JSON格式文件');
                            }
                            return isJSON;
                        }}
                    >
                        <Button
                            type='primary'
                            className='upload-btn'
                            icon={<UploadOutlined />}
                        >上传所有工具</Button>
                    </Upload>
                </div>
            </div>
            <Table 
                columns={columns}        // 绑定上面定义的列配置
                dataSource={data}        // 绑定表格数据源state
                bordered
                pagination={true}
                className='api-table'
                rowClassName={() => {'table-row'}}
            />
        </div>
    )
}
/**
 * 整体业务流程梳理
 * 页面加载触发 useEffect → 调用 handleGetAllData 请求后端工具列表，渲染表格
 * 功能 1：单条删除 → 每行操作删除按钮 → 调用 DeleteOneToolApi → 刷新列表
 * 功能 2：一键清空全部 → 顶部红色按钮 → DeleteAllToolApi → 清空表格
 * 功能 3：批量导入工具 → 上传 JSON 文件 → Upload 组件校验格式 → UploadFileApi 上传 → 刷新列表
 */





