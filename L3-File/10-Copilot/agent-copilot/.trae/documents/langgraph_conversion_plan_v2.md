# Langgraph 框架转换计划 v2

## 一、项目分析

### 当前项目状态
项目已经有初步的Langgraph实现，但存在以下问题：
1. **循环控制不完善**：需要确保循环次数限制在8次以内
2. **通用查询接口**：需要完善API调用封装，参考mock_api_server.py
3. **组件初始化复杂**：LanggraphAgent初始化多个Hub组件，容易出错
4. **工具调用流程**：需要确保AI能正确调用查询接口

### 核心需求
1. **AI规划任务**：使用Langgraph实现循环规划，最大循环8次
2. **通用查询接口**：提供产品、订单、供应商等查询接口给AI调用
3. **与现有系统集成**：保留原有的工具管理和向量检索逻辑

---

## 二、Langgraph 架构设计

### 状态定义 (`AgentState`)
```python
class AgentState(TypedDict):
    query: str              # 用户原始查询
    task_description: str   # 当前任务描述
    api_chain: List[dict]   # API调用链历史
    selected_tool: Any      # 当前选择的工具
    params: Dict[str, Any]  # 提取的参数
    missing_params: List    # 缺失的参数
    tool_result: str        # 工具调用结果
    is_single_task: bool    # 是否单任务
    is_complete: bool       # 任务是否完成
    loop_count: int         # 当前循环次数（最大8次）
    summary: str            # 最终总结
    error: Optional[str]    # 错误信息
    # 组件引用
    api_planning_hub: Any
    api_selection_hub: Any
    param_extraction_hub: Any
    generate_task_hub: Any
    tool_summary_hub: Any
    tool_use_hub: Any
```

### 图结构
```
START → [plan_task] → [select_tool] → [extract_params]
                                          │
                    ┌──────────────────────┴──────────────────────┐
                    ↓ (有缺失参数)                                 ↓ (无缺失参数)
            [supplement_params]                            [call_tool]
                    │                                           │
                    └───────────────┬───────────────────────────┘
                                    ↓
                              [check_loop]
                                    │
                    ┌───────────────┴───────────────┐
                    ↓ (继续循环)                     ↓ (完成/循环>=8)
            [select_tool] ←←←←←←←←←←←←←←      [summarize] → END
```

### 节点功能
| 节点 | 功能 | 循环次数影响 |
|------|------|-------------|
| `plan_task` | 任务规划，判断单/多任务 | 初始化 loop_count=0 |
| `select_tool` | 选择合适的工具 | 无 |
| `extract_params` | 提取工具参数 | 无 |
| `supplement_params` | 补齐缺失参数 | 无 |
| `call_tool` | 调用工具API | 无 |
| `check_loop` | 判断是否继续循环 | loop_count++，超过8次终止 |
| `summarize` | 总结结果 | 结束 |

---

## 三、实现步骤

### 步骤 1：完善状态定义 (`langgraph_agent/state.py`)
确保状态定义完整，包含所有必要字段和组件引用。

### 步骤 2：完善节点实现 (`langgraph_agent/nodes.py`)
- `plan_task`: 初始化循环计数
- `check_loop`: 严格限制循环次数不超过8次
- 所有节点添加错误处理

### 步骤 3：完善条件边 (`langgraph_agent/edges.py`)
- `should_continue`: 判断是否继续循环或进入总结
- `has_missing_params`: 判断是否需要参数补齐

### 步骤 4：构建图 (`langgraph_agent/graph.py`)
使用 `StateGraph` 构建完整流程，确保循环控制正确。

### 步骤 5：增强通用查询接口 (`langgraph_agent/api_handler.py`)
基于 `mock_api_server.py`，提供更完善的API调用封装：
- 产品查询接口
- 订单查询接口
- 供应商查询接口
- 通用HTTP请求方法

### 步骤 6：优化入口文件 (`langgraph_agent/main.py`)
简化组件初始化，提供清晰的API接口。

### 步骤 7：修改 `app.py` 添加 Langgraph 路由
确保前端可以通过 `use_langgraph=true` 参数使用Langgraph流程。

### 步骤 8：测试验证
- 测试单任务流程
- 测试多任务循环流程（验证8次限制）
- 测试工具调用失败处理

---

## 四、通用查询接口设计

基于 `mock_api_server.py`，提供以下接口：

### 产品接口
| 接口 | 方法 | 参数 | 说明 |
|------|------|------|------|
| `/products/getProductByName` | POST | `name`: 产品名称 | 按名称查询产品 |
| `/products/getProductById` | POST | `productId`: 产品ID | 按ID查询产品 |
| `/products/getBatchProductByProductIds` | POST | `productIds`: ID列表 | 批量查询 |
| `/products/addProduct` | POST | `name`, `description`, `price`, `quantityInStock` | 添加产品 |

### 订单接口
| 接口 | 方法 | 参数 | 说明 |
|------|------|------|------|
| `/orders/getOrderByOrderId` | POST | `orderId`: 订单ID | 查询订单 |
| `/orders/getByProductId` | POST | `productId`: 产品ID | 查询产品订单 |
| `/orders/getByTimeRange` | POST | `startDate`, `endDate` | 按时间查询 |
| `/orders/createOrder` | POST | `productId`, `quantity`, `customerName` | 创建订单 |

### 供应商接口
| 接口 | 方法 | 参数 | 说明 |
|------|------|------|------|
| `/suppliers/getSupplierByName` | GET | `name`: 供应商名称 | 查询供应商 |
| `/suppliers/getSupplierById` | GET | `supplierId`: 供应商ID | 查询供应商 |
| `/suppliers/querySuppliersByDeliveryRegion` | POST | `region`: 配送区域 | 查询区域供应商 |

---

## 五、循环控制设计

### 关键逻辑
```python
MAX_LOOP_COUNT = 8

def check_loop(state):
    loop_count = state["loop_count"]
    
    # 单任务直接完成
    if state["is_single_task"]:
        return {"is_complete": True}
    
    # 超过最大循环次数
    if loop_count >= MAX_LOOP_COUNT:
        logger.warning(f"循环次数达到上限 {MAX_LOOP_COUNT}")
        return {"is_complete": True}
    
    # 判断是否还有后续任务
    is_complete, task_description = generate_task_hub.gen_from_context_task(...)
    
    return {
        "is_complete": is_complete,
        "task_description": task_description,
        "loop_count": loop_count + 1
    }
```

### 流程图
```
loop_count = 0
┌─────────────────────────────────────────────────────────────┐
│                       plan_task                             │
│                  loop_count = 0                             │
└───────────────────┬─────────────────────────────────────────┘
                    ↓
              select_tool
                    ↓
              extract_params
                    ↓
              call_tool
                    ↓
              check_loop ─────────────┐
                    │                 │
           ┌────────┴────────┐        │
           ↓                 ↓        │
     is_complete?      loop_count >= 8?
           │                 │        │
          YES               YES       │
           └────────┬────────┘        │
                    ↓                 │
              summarize               │
                    ↓                 │
                  END ←───────────────┘
```

---

## 六、文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `langgraph_agent/state.py` | 更新 | 完善状态定义 |
| `langgraph_agent/nodes.py` | 更新 | 完善节点实现，添加循环控制 |
| `langgraph_agent/edges.py` | 更新 | 完善条件边逻辑 |
| `langgraph_agent/graph.py` | 更新 | 构建完整流程图 |
| `langgraph_agent/api_handler.py` | 更新 | 增强通用查询接口 |
| `langgraph_agent/main.py` | 更新 | 优化入口和初始化 |
| `app.py` | 更新 | 添加Langgraph路由支持 |

---

## 七、验证计划

1. **单元测试**：测试每个节点的独立功能
2. **循环测试**：验证多任务场景下循环次数不超过8次
3. **功能测试**：使用"查询苹果的产品信息"验证端到端流程
4. **异常测试**：测试工具调用失败、参数缺失等异常情况

---

## 八、风险与注意事项

### 潜在风险
1. **循环次数限制**：需确保 `check_loop` 节点正确更新 `loop_count`
2. **组件初始化**：多个Hub组件初始化可能失败，需添加错误处理
3. **工具调用失败**：需处理API调用异常情况
4. **LLM响应解析**：需确保LLM返回格式正确

### 注意事项
1. 保留原有的工具管理和向量检索逻辑
2. LLM调用使用现有的 `LargeLanguageModel` 封装
3. 确保与原项目的数据格式兼容
4. 提供详细的日志记录便于调试
