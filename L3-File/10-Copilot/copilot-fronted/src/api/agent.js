// 统一接口请求封装

import { BaseRoot } from './baseRoot';
import axios from 'axios';
import { getAuthHeader, clearAccessToken } from './auth';

// 添加请求拦截器确保所有请求都包含认证头
// 请求拦截器 request.interceptors（发请求前自动执行）
// 原理：每次调用 axios.get/post 时，代码会先进入这里修改请求头。
axios.interceptors.request.use(
  (config) => {
    // 为所有非登录/注册请求添加认证头
    if (!config.url.includes('/login_user') && !config.url.includes('/register_user')) {
      // const authHeader = getAuthHeader();
      config.headers = {
        ...config.headers,
        // ...authHeader
      };
    }
    console.log('请求配置:', config); // 调试日志
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 添加响应拦截器处理401和403错误
// 响应拦截器 response.interceptors（后端返回数据后统一处理）
axios.interceptors.response.use(
  (response) => response,   // 对于成功的响应，直接返回
  (error) => {
    if (error.response && error.response.status === 401) {
      alert('安全校验未通过:' + (error.response.data.message || '你无权访问系统'));
      // 清除本地存储的访问令牌
      clearAccessToken();
      // 可选：重定向到登录页面或显示登录提示
      if (typeof window !== 'undefined') {
        window.location.href = '/login';
      }
    }
    if (error.response && error.response.status === 403) {
      // 处理403错误，例如显示权限不足提示
      if (typeof window !== 'undefined' && window.alert) {
        console.error('权限不足');
        alert('权限不足:' + (error.response.data.message || '您没有权限执行此操作。'));
      }
    }
    return Promise.reject(error);
  }
);

// 封装登录 POST 请求的接口函数，导出给页面组件直接调用，统一管理接口地址。
export const LoginApi = (loginData) => {
    return axios.post(`${BaseRoot}/login_user`, loginData);
}
// 2种写法
export const RegisterApi = registerData => axios.post(`${BaseRoot}/register_user`, registerData)

// 获取全部工具列表
export const ToolsApi = () => axios.get(`${BaseRoot}/get_all_tools`)
// 删除单个/批量工具，直接传递数组ids作为请求体
export const DeleteOneToolApi = (ids) => axios.post(`${BaseRoot}/delete_tool_by_ids`, ids)
// 删除全部工具
export const DeleteAllToolApi = () => axios.post(`${BaseRoot}/delete_all_tool`)

export const UploadFileApi = (file) => {
  const formData = new FormData();
  formData.append('file', file);

  return axios.post(`${BaseRoot}/upload_file`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  }).then(response => {
    // 成功后统一包装返回自定义对象
    if (response.status === 200) {
      console.log('文件上传成功:', response.data);
      return {
        status: 'success',
        message: '上传成功'
      };
    }
  }).catch(error => {
    return {
        status: 'error',
        message: '上传失败' + (error.response?.data?.message || error.message)
      };
  });
}
// 测试大模型接口
export const TestLLMApi = (data) => axios.post(`${BaseRoot}/test_llm`, data)
// 创建API规划任务
export const ApiPlanningApi = async (data) => {
  return axios.post(`${BaseRoot}/api_planning`, data)
};
// 查询任务执行状态
export const TestTaskStatusApi = async (data) => {
  return axios.post(`${BaseRoot}/api_task_status`, data)
}; 

/*
文件整体价值
统一管理接口地址：后端接口路径全部集中在一个文件，后端改地址只改一处，不用全局搜代码；
统一鉴权逻辑：不用每个页面手动携带 Token，拦截器自动处理；
统一错误处理：401 自动跳登录、403 自动弹窗，减少重复业务代码；
统一请求格式：所有接口封装成函数，页面直接 import 调用，代码整洁；
特殊请求单独封装：文件上传单独处理 FormData、统一返回成功 / 失败结构，降低页面复杂度。
*/