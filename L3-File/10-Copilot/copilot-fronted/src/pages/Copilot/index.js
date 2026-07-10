/**
 * AI Copilot 智能推理绘图页面，技术栈:
 * React 函数组件 + Ant Design X（AI 对话组件库）+ echarts-for-react（关系力导向图）+ antd-style（CSS-in-JS 样式）+ 后端异步任务轮询架构。
 * 业务流程:
 * 用户右侧聊天框输入自然语言指令
 * 提交请求 ApiPlanningApi 创建后端异步任务，返回 taskId
 * 前端轮询 TestTaskStatusApi 实时拉取任务进度、图节点 / 连线数据、标题
 * 任务完成后流式输出 AI 回复文字，左侧 ECharts 自动渲染流程图 / 关系图
 */

import {
  OpenAIFilled,
  UserOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import ReactEcharts from 'echarts-for-react'; // echarts React封装，渲染关系图
import { Switch, Typography, InputNumber } from 'antd'; // 基础组件

// Ant Design X AI对话专用组件
import {
  Bubble, // 聊天气泡
  Prompts, // 快捷推荐提问
  Sender, // 底部输入框
  Suggestion, // 输入联想下拉
  Welcome, // 空白欢迎页
} from '@ant-design/x';
import { Button, Space, Spin } from 'antd';
import { createStyles } from 'antd-style'; // 组件级样式方案，基于token主题
import { useEffect, useRef, useState } from 'react';
import { ApiPlanningApi, TestTaskStatusApi } from '../../api/agent';

const fooAvatar = {
  color: '#f56a00',
  backgroundColor: '#fde3cf',
}; // 用户气泡头像
const barAvatar = {
  color: '#fff',
  backgroundColor: '#87d068',
}; // AI气泡头像

// 模拟历史对话会话列表
const MOCK_SESSION_LIST = [
  {
    key: '5',
    label: '我想知道苹果产品的库存信息',
    group: 'Today',
  },
  {
    key: '4',
    label: '查询ID为1的物流供应商信息',
    group: 'Today',
  },
  {
    key: '3',
    label: '查询名为京东的物流供应商信息',
    group: 'Today',
  },
  {
    key: '2',
    label: '查询订单ID为1的订单信息',
    group: 'Yesterday',
  },
  {
    key: '1',
    label: '查询产品ID为1的订单信息',
    group: 'Yesterday',
  },
];
// 输入框 / 触发的联想下拉菜单，支持分组子选项
const MOCK_SUGGESTIONS = [
  { label: '写报告', value: '报告' },
  { label: '画图', value: '图片' },
  {
    label: '查看技术栈',
    value: '技术栈',
    icon: <OpenAIFilled />,
    children: [
      { label: '关于React', value: 'react' },
      { label: '关于Ant Design', value: 'antd' },
    ],
  },
];
// 首页空白时展示的快捷提问，点击直接发送指令
const MOCK_QUESTIONS = [
  '查询苹果的产品信息',
  '查询名为京东的物流供应商信息',
];
// AI 加载中占位文案
const AGENT_PLACEHOLDER = 'Generating content, please wait...';
// useCopilotStyle（右侧聊天面板样式）
const useCopilotStyle = createStyles(({ token, css }) => {
  return {
    copilotChat: css`
      display: flex;
      flex-direction: column;
      background: ${token.colorBgContainer};
      color: ${token.colorText};
    `,
    // chatHeader 样式
    chatHeader: css`
      height: 52px;
      box-sizing: border-box;
      border-bottom: 1px solid ${token.colorBorder};
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 10px 0 16px;
    `,
    headerTitle: css`
      font-weight: 600;
      font-size: 15px;
    `,
    headerButton: css`
      font-size: 18px;
    `,
    conversations: css`
      width: 300px;
      .ant-conversations-list {
        padding-inline-start: 0;
      }
    `,
    // chatList 样式
    chatList: css`
      overflow: auto;
      padding-block: 16px;
      flex: 1;
    `,
    chatWelcome: css`
      margin-inline: 16px;
      padding: 12px 16px;
      border-radius: 2px 12px 12px 12px;
      background: ${token.colorBgTextHover};
      margin-bottom: 16px;
    `,
    loadingMessage: css`
      background-image: linear-gradient(90deg, #ff6b23 0%, #af3cb8 31%, #53b6ff 89%);
      background-size: 100% 2px;
      background-repeat: no-repeat;
      background-position: bottom;
    `,
    // chatSend 样式
    chatSend: css`
      padding: 12px;
    `,
    sendAction: css`
      display: flex;
      align-items: center;
      margin-bottom: 12px;
      gap: 8px;
    `,
    speechButton: css`
      font-size: 18px;
      color: ${token.colorText} !important;
    `,
  };
});
const Copilot = props => {
    // 父组件 CopilotDemo 传递全部全局状态与修改函数
  const { setNodes, setEdges, copilotOpen,  isCopilot, isNotContext, setIsCopilot, setIsNotContext,contextNumber,setContextNumber, setTitle,taskId,setTaskId, setTaskStatus } = props;

  const { styles } = useCopilotStyle();
  const abortController = useRef(null);

  // ==================== State ====================
  const [loading, setLoading] = useState(false);
  const [inputValue, setInputValue] = useState('');
  
  
  const [messages, setMessages ] = useState([]);
  // ==================== Event ====================
  // 新增状态跟踪流式请求;// 定时器状态，用于清理轮询、流式打字
  const [streamInterval, setStreamInterval] = useState(null); // AI文字逐字输出定时器
  const taskInterval = useState(null); // 后端任务轮询定时器
  // AI 文字流式打字输出函数:后端任务执行完毕，拿到完整 AI 回复字符串，模拟打字机逐字渲染气泡
  const testUserSubmit = (systemOutput,nodes,edges,title) => {
    // 创建一条空AI消息，状态标记为 streaming 流式加载
    const newMessage = {
      id: Date.now(),
      message: { content: '', role: 'assistant' },
      status: 'streaming'
    };
    /**
     * 当多次连续调用 setMessages 时，React 会批量合并更新，如果直接依赖 messages 变量，会拿到过期旧值；
     * 而 prev 是 React 内部保证的最新上一轮状态快照，永远准确，适合聊天这种高频追加消息场景。
     * 参数 prev:固定代表更新前最新的 messages 数组，不用自己读取 messages。
     * [...prev, newMessage] 展开运算符原理
     * ... 是 ES6 数组展开语法，作用:不修改原数组，创建全新数组
     * ...prev:把旧数组 prev 里所有元素原样复制出来；
     * , newMessage:在复制完旧元素后，尾部追加新消息对象；
     * 整体生成一个全新数组，地址引用完全改变。
     */
    setMessages(prev => [...prev, newMessage]);
  
    let accumulated = ''; // 累计输出文字
    // 分片输出逻辑
    const processChunk = (chunk) => {
      accumulated += chunk;
      // 更新对应消息的content；遍历整个消息数组，map 会返回全新数组（符合 React 不可变原则，不能直接改原数组）
      setMessages(prev => prev.map(msg => 
        msg.id === newMessage.id 
          ? {...msg, message: { ...msg.message, content: accumulated }}
          : msg
      ));
    };
    // 50ms输出一个字符，模拟打字
    let index = 0;
    const interval = setInterval(() => {
      if (index < systemOutput.length) {
        processChunk(systemOutput[index]);
        index++;
      } else {
        clearInterval(interval);
        setMessages(prev => prev.map(msg =>
          msg.id === newMessage.id
            ? {...msg, status: 'results'}
            : msg
        ));
        setLoading(false);
        setNodes(nodes)
        setEdges(edges)
        setTitle(title)
      }
    }, 50);
    setStreamInterval(interval);
    return () => clearInterval(interval);
  };

/**
 * 用户点击发送按钮触发完整请求链路:
 * 1.校验:无内容 /loading 中直接 return 阻止重复提交
 * 2.清空旧图表、标题、开启 loading 加载态
 * 3.创建 user 用户消息，存入 messages 列表渲染气泡
 * 4.组装请求参数:
 * - query:用户输入文本
 * - contexts:历史对话上下文
 * - isCopilot:推理模式开关
 * - isContext:是否启用上下文记忆（取反 isNotContext）
 * - contextNumber:记忆条数上限
 * 5.调用 ApiPlanningApi 创建后端异步任务
 * - 成功拿到 taskId → setTaskId 触发下方轮询 useEffect
 * - 请求失败:推送错误 AI 气泡，关闭 loading，清空图表
 */
const handleUserSubmitV2 = val => {
    if (!val || loading) return;

    console.log('1. [Submit] handleUserSubmitV2由值触发:', val);

    setLoading(true);
    setEdges([]);
    setNodes([]);
    setTitle("");
    // 新增用户消息
    const userMessage = {
      id: Date.now(),
      message: { content: val, role: "user" },
      status: "local"
    };
    setMessages(prevMessages => [...prevMessages, userMessage]);

    // 注意:这里的 contexts 应该是包含了当前发送的这条消息
    // 拼接上下文
    const contexts = [...messages, userMessage].map(m => m.message);

    const data = {
      query: val,
      contexts,
      isCopilot: isCopilot,
      isContext: !isNotContext,
      contextNumber: contextNumber
    };

    console.log('2. [提交] 向ApiPlanningApi发送数据:', data);
    // 发起创建任务接口
    ApiPlanningApi(data).then(res => {
      console.log('3. [提交] ApiPlanning Api成功！回应:', res);
      if (res.status === 200 && res.data && res.data.task_id) {
        console.log('4. [提交] 收到 task_id:', res.data.task_id, '.更新状态中...');
        // 🚀 更新 taskId，触发 useEffect
        setTaskId(res.data.task_id);
      } else {
        console.error('5. [提交] 错误:响应正常,但缺少task_id.', res.data);
        setLoading(false); // 出错了也要停止加载状态
      }
    }).catch(err => {
      // 网络/接口报错处理
      console.error('5. [提交] 失败:ApiPlanningApi请求失败！', err);
      const errorMessage = {
        id: Date.now(),
        message: { content: "系统:API规划错误", role: "assistant" },
        status: "results"
      };
      setMessages(prev => [...prev, errorMessage]);
      setNodes([]);
      setEdges([]);
      setTitle("");
      setLoading(false);
    });
  };

  const onChangeCopilot = (checked) => {
    setIsCopilot(checked)
  };
  const onChangeContext = (checked) => {
    setIsNotContext(checked)
  };

  const onChangeNuber = value => {
  setContextNumber(value)
};
const reload = ()=>{
  setMessages([])
  setNodes([])
  setEdges([])
}

  // ==================== Nodes ====================
  //JSX 渲染模块
  const chatHeader = (
    <div className={styles.chatHeader}>
      <div className={styles.headerTitle}>✨ AI Copilot</div>
      <Space size={0}>
        <Button type="text" icon={<ReloadOutlined />} className={styles.headerButton} onClick={reload} />
      </Space>
    </div>
  );
  const chatList = (
    <div className={styles.chatList}>
      {(messages === null || messages === void 0 ? void 0 : messages.length) ? (
        /** 消息列表 */
        <Bubble.List
          style={{ height: '100%', paddingInline: 16 }}
          items={
            messages === null || messages === void 0
              ? void 0
              : messages.map(i =>
                  Object.assign(Object.assign({}, i.message), {
                    classNames: {
                      content: i.status === 'loading' ? styles.loadingMessage : '',
                    },
                    typing:
                      i.status === 'loading' ? { step: 5, interval: 20, suffix: <>💗</> } : false,
                  }),
                )
          }
          roles={{
            assistant: {
              placement: 'start',
              avatar: { icon: <UserOutlined />, style: barAvatar },
              loadingRender: () => (
                <Space>
                  <Spin size="small" />
                  {AGENT_PLACEHOLDER}
                </Space>
              ),
            },
            user: { placement: 'end',avatar: { icon: <UserOutlined />, style: fooAvatar }, },
            
          }}
        />
      ) : (
        /** 没有消息时的 welcome */
        <>
          <Welcome
            variant="borderless"
            title="👋 欢迎使用Copilot智能操作平台"
            description="请在输入框中输入你的问题，我们将为您服务，希望您使用的轻松愉快"
            className={styles.chatWelcome}
          />

          <Prompts
            vertical
            title="我能帮您:"
            items={MOCK_QUESTIONS.map(i => ({ key: i, description: i }))}
            onItemClick={info => {
              var _a;
              return handleUserSubmitV2(
                (_a = info === null || info === void 0 ? void 0 : info.data) === null ||
                  _a === void 0
                  ? void 0
                  : _a.description,
              );
            }}
            style={{
              marginInline: 16,
            }}
            styles={{
              title: { fontSize: 14 },
            }}
          />
        </>
      )}
    </div>
  );
  const chatSender = (
    <div className={styles.chatSend}>

      

      <Switch  checkedChildren="单句模式" unCheckedChildren="上下文模式" onChange={onChangeContext} defaultChecked={isNotContext} style={{marginBottom: 5,marginLeft:10}}/>

    {!isNotContext&&<InputNumber min={1} max={100} defaultValue={1} onChange={onChangeNuber} style={{marginBottom: 5,marginLeft:10}}/>}


      <Suggestion items={MOCK_SUGGESTIONS} onSelect={itemVal => setInputValue(`[${itemVal}]:`)}>
        {({ onTrigger, onKeyDown }) => (
          <Sender
            loading={loading}
            value={inputValue}
            onChange={v => {
              onTrigger(v === '/');
              setInputValue(v);
            }}
            onSubmit={() => {
              console.log(inputValue);
              handleUserSubmitV2(inputValue);
              setInputValue('');
            }}
            onCancel={() => {
              var _a;
              (_a = abortController.current) === null || _a === void 0 ? void 0 : _a.abort();
            }}
            placeholder="询问或输入/使用技能"
            onKeyDown={onKeyDown}
            prefix={
              <div>
                <Switch checkedChildren="copilot模式" unCheckedChildren="chat模式" onChange={onChangeCopilot} defaultChecked />

              </div>
            }
            actions={(_, info) => {
              const { SendButton, LoadingButton } = info.components;
              return (
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  {loading ? <LoadingButton onClick={() => {
            clearInterval(taskInterval);
            clearInterval(streamInterval);
            setLoading(false);
            setMessages(prev => prev.map(msg =>
              msg.status === 'streaming' 
                ? {...msg, status: 'interrupted'}
                : msg
            ));
          }}type="default" /> : <SendButton type="primary" />}
                </div>
              );
            }}
          />
        )}
      </Suggestion>
    </div>
  );

  // 轮询核心 useEffect（监听 taskId）
  useEffect(() => {
    console.log('A. [结果]useEffect正在运行。当前taskId:', taskId, ' | 当前加载状态:', loading);

    // 当没有 taskId 或正在提交第一个请求时，直接退出
    if (!taskId) {
      console.log('B. [结果] 没有 taskId, 直接退出.');
      return;
    }
    
    // 如果 taskId 存在，说明可以开始轮询了
    console.log('C. [结果] taskId 存在! 开始轮询,设置轮询间隔...');
    let targetStatus = 0;
    const intervalId = setInterval(() => {
      console.log(`D. [投票] 给taskId投票: ${taskId}`);
      TestTaskStatusApi({ task_id: taskId }).then(res => {
        console.log('E. [投票] 收到状态响应:', res.data);
        if (res.status === 200 && res.data.task) {
          const task = res.data.task;
          
          if (task.status !== -1 && !task.isEnd) {
            console.log('F. [投票] 任务仍在运行。状态:', task.status);
            // ... (更新中间状态的代码)
            if (task.status !== targetStatus) {
              targetStatus = task.status;
              setTaskStatus(task.status);
              setNodes(task.nodes);
              setEdges(task.edges);
              setTitle(task.isSuccess);
            }
          } else {
            console.log('G. [投票] 成功：任务完成！停止投票并显示结果.');
            clearInterval(intervalId);
            setTaskId(''); // 清空taskId
            testUserSubmit(task.systemOutput, task.nodes, task.edges, task.isSuccess);
            setTaskStatus(-1);
          }
        }
      }).catch(err => {
        console.error('H. [投票] 失败:轮询API请求失败！', err);
        clearInterval(intervalId);
        setTaskId('');
        setLoading(false);
      });
    }, 2000); // 轮询间隔调整为2秒，方便观察

    return () => {
      console.log('I. [清除结果] 清理taskId间隔:', taskId);
      clearInterval(intervalId);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskId]); // 依赖项保持不变
  return (
    <div className={styles.copilotChat} style={{ width: copilotOpen ? 600 : 0 }}>
      {/** 对话区 - header */}
      {chatHeader}

      {/** 对话区 - 消息列表 */}
      {chatList}

      {/** 对话区 - 输入框 */}
      {chatSender}
    </div>
  );
};
/**
 * 页面左右分栏布局:
 * 左侧 workarea:占剩余宽度，放 ECharts 关系图
 * 右侧 Copilot 聊天面板:宽度由 copilotOpen 控制，开启 600px，关闭宽度 0 隐藏
 * workareaHeader:顶部标题栏，带渐变按钮（当前无绑定事件）
 * workareaBody:图表容器，设置滚动、高度自适应
 */
// useWorkareaStyle（左侧图表工作区外层布局）
const useWorkareaStyle = createStyles(({ token, css }) => {
  return {
    copilotWrapper: css`
      min-width: 1000px;
      height: 100vh;
      display: flex;
    `,
    workarea: css`
      flex: 1;
      background: ${token.colorBgLayout};
      display: flex;
      flex-direction: column;
    `,
    workareaHeader: css`
      box-sizing: border-box;
      height: 52px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 48px 0 28px;
      border-bottom: 1px solid ${token.colorBorder};
    `,
    headerTitle: css`
      font-weight: 600;
      font-size: 15px;
      color: ${token.colorText};
      display: flex;
      align-items: center;
      gap: 8px;
    `,
    headerButton: css`
      background-image: linear-gradient(78deg, #8054f2 7%, #3895da 95%);
      border-radius: 12px;
      height: 24px;
      width: 93px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #fff;
      cursor: pointer;
      font-size: 12px;
      font-weight: 600;
      transition: all 0.3s;
      &:hover {
        opacity: 0.8;
      }
    `,
    workareaBody: css`
      flex: 1;
      padding: 16px;
      background: ${token.colorBgContainer};
      border-radius: 16px;
      min-height: 0;
    `,
    bodyContent: css`
      overflow: auto;
      height: 100%;
      padding-right: 10px;
    `,
    bodyText: css`
      color: ${token.colorText};
      padding: 8px;
    `,
  };
});
const CopilotDemo = () => {
  const { styles: workareaStyles } = useWorkareaStyle();
  // ==================== State =================
  const [copilotOpen, setCopilotOpen] = useState(true);
  const [isCopilot,setIsCopilot] = useState(true);
  const [isNotContext,setIsNotContext] = useState(false);
  const [contextNumber,setContextNumber] = useState(1);
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [title, setTitle] = useState('');
  const [taskId,setTaskId] = useState('');
  const [taskStatus,setTaskStatus] = useState(0);


  // 图的配置选项
  /**
   * ECharts 力导向图配置 options
   * 使用 graph 关系图，force 力导向布局：
   * symbolSize 节点大小、repulsion 节点斥力、edgeLength 连线长度
   * 鼠标悬浮 tooltip 展示节点信息
   * 连线带箭头、阴影美化、支持鼠标拖拽缩放漫游 roam:true
   * 实时绑定全局 state nodes / edges，后端轮询更新后图表自动刷新
   */
  const options = {
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        if (params.dataType === 'node') {
          return `<div>
            <p><strong>节点名称:</strong>${params.data.name}</p>
            <p><strong>类型:</strong>${params.data.group}</p>
            <p><strong>描述:</strong>${params.data.group}</p>
          </div>`;
        }
        else{
          return `<div>${params.data.value}</div>`
        }
      },
    },
  series: [
    {
       type: 'graph',
        layout: 'force',
        symbolSize: 50,
        roam: true,
       
        label: {
        show: true,
        position: 'bottom',
        formatter: (params) => {
          return params.data.name;
        }
      },
        edgeSymbol: ['','arrow'],
        edgeSymbolSize: [4, 10],
        lineStyle: {
          color: '#888',
          curveness: 0.3,
          width: 2,
        },
        force: {
          repulsion: 8000,
          edgeLength: [100, 200],
          gravity: 0.1,
        },
        itemStyle: {
          borderColor: '#fff',
          borderWidth: 2,
          shadowBlur: 10,
          shadowColor: 'rgba(0, 0, 0, 0.3)',
        },
      data: nodes,
      // links: [],
      links: edges,
      lineStyle: {
        opacity: 0.9,
        width: 2,
        curveness: 0
      }
    }
  ],

  };

  // ==================== Render =================
  return (
    <div>
    {/* Copilot 组件最外层 DOM */}
    <div className={workareaStyles.copilotWrapper}>
      {/** 左侧:图表工作区 */}
      <div className={workareaStyles.workarea}>
        <div className={workareaStyles.workareaHeader}>
          <div className={workareaStyles.headerTitle}>
            <img
              src="https://mdn.alipayobjects.com/huamei_iwk9zp/afts/img/A*eco6RrQhxbMAAAAAAAAAAAAADgCCAQ/original"
              draggable={false}
              alt="logo"
              width={20}
              height={20}
            />
            系统操作推理与运行路径
          </div>
        </div>

        <div
          className={workareaStyles.workareaBody}
          style={{ margin: 16 }}
        >
          <Typography style={{fontSize:18,fontWeight:'bold',color:'#FF0000'}}>{title}</Typography>
          <div className={workareaStyles.bodyContent}>
      <ReactEcharts
        option={options}
        notMerge
        lazyUpdate
        style={{height:'calc(100vh - 150px)'}}
      />
          </div>
        </div>
      </div>
      {/** 右侧对话区 */}
      <Copilot 
        nodes={nodes}
        edges={edges}
        isCopilot={isCopilot}
        isNotContext={isNotContext}
        contextNumber={contextNumber}
        setContextNumber={setContextNumber}
        setIsCopilot={setIsCopilot}
        setIsNotContext={setIsNotContext}
        setNodes={setNodes}
        setEdges={setEdges}
        title={title}
        setTitle={setTitle}
        taskId={taskId}
        setTaskId={setTaskId}
        taskStatus={taskStatus}
        setTaskStatus={setTaskStatus}
        copilotOpen={copilotOpen} setCopilotOpen={setCopilotOpen} />
    </div>
    </div>
  );
};
export default CopilotDemo;