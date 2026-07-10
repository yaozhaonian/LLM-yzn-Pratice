/**
 * API 工具管理页面，用 antd Table 展示工具列表，支持删除单行、编辑弹窗上传 JSON 文件、顶部批量清空 / 批量上传。
 */
// Table：数据表格,Button：按钮,Modal：弹窗（确认框、信息弹窗）,Upload：文件上传组件
import { Table, Button, Modal, Upload } from 'antd';
// DeleteOutlined/EditOutlined：删除、编辑图标
import { DeleteOutlined, EditOutlined } from '@ant-design/icons';
import { useState } from 'react';


/**
 * 组件定义与状态
 * data：表格数据源，初始空数组，setData 修改表格列表
 * selectedRow：保存当前点击删除的那一行完整数据，初始 null
 */
export default function ApiTools() {
    const [data, setData] = useState([]);
    const [selectedRow, setSelectedRow] = useState(null); 

    // 删除确认函数 confirmDelete
    const confirmDellete = () => {
        Modal.confirm({
            title: '确认删除该API工具',
            content: '此操作不可撤销',
            onOk() {
                /**
                 * data.filter(...)：数组过滤，保留 key 不等于选中行 key 的所有数据
                 * filter 返回新数组，符合 React 不可变原则，直接传给 setData 更新表格
                 */
                setData(data.filter(item => item.key != selectedRow.key))
            }
        })
    };

    // 表格列配置 columns
    const columnns = [
        {title: '序号', dataIndex: 'index'},
        {title: 'API名称', dataIndex: 'name'},
        {title: 'API描述', dataIndex: 'description'},
        // 操作列（核心交互部分）
        {
            title: '操作', 
            render: (
                <>
                {/* 删除按钮 */}
                <Button danger icon={<DeleteOutlined />} onClick={() => {
                    setSelectedRow(record);
                    confirmDellete();
                }} />

                {/** 编辑按钮 */}
                <Button icon={<EditOutlined />} 
                    onClick={() => Modal.info({
                        title: '编辑API工具',
                        content: <Upload beforeUpload={file => {
                            //处理文件逻辑
                        }} />
                    })} />
                </>
            )
        }
    ];

    return(
        <div className='api-tools-page'>
            {/** 顶部工具栏 */}
            <div className='toolbar'>
                <Button danger>清空所有工具</Button>
                <Button type='primarry'>上传所有工具</Button>
            </div>
            {/** 表格 */}
            <Table 
                columnns={columnns}
                dataSource={data}
                bordered
                pagination={false}
            />
        </div>
    );
};


