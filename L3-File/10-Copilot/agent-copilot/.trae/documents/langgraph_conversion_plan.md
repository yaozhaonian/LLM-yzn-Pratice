# Langgraph 框架转换计划

## 一、项目分析

### 当前项目核心流程
```
用户查询 → 任务规划 → 工具选择 → 参数提取 → 参数补齐 → 工具调用 → 循环判断 → 结果总结
```

### 关键组件
| 组件 | 职责 | 文件 |
|------|------|------|
| ApiPlanningHub | 整体规划流程 | apis/api_planning_hub.py |
| ApiSelectionHub | 工具选择（向量检索+LLM） | apis/api_selection_hub.py |
| ParamExtractionHub | 参数提取 | param_extraction/param_extraction_hub.py |
| GenerateTaskHub | 任务生成与规划 | tasks/generate_task_hub.py |
| ToolUseHub | 工具调用 | tools/tool_use_hub.py |
| ToolSummaryHub | 结果总结 | tools/tool_summary_hub.py |
| LargeLanguageModel | LLM封装 | models/llm.py |
| ToolManager | 工具管理 | tools/tool_manager.py |

### 用户需求
1. **转换为 Langgraph 框架**：使用 Langgraph 的 State、Node、Edge 概念重构
2. **AI 规划任务**：核心流程使用 Langgraph 实现循环规划
3. **循环次数限制**：最大循环 8 次
4. **通用查询接口**：参考 mock_api_server.py 提供可被 AI 调用的 API

---

## 二、Langgraph 架构设计

### 核心概念映射

| Langgraph 概念 | 当前实现 | 说明 |
|----------------|----------|------|
| State | api_chain + query + task_description | 图的共享状态 |
| Node | 各个 Hub 的方法 | 图中的处理节点 |
| ConditionalEdge | 循环判断逻辑 | 根据状态决定下一个节点 |
| EndNode | 总结节点 | 任务结束时的总结 |

### 状态定义 (AgentState)
```python
class AgentState(TypedDict):
    query: str              # 用户原始查询
    task_description: str   # 当前任务描述
    api_chain: list         # API调用链历史
    selected_tool: Tool     # 当前选择的工具
    params: dict            # 提取的参数
    missing_params: list    # 缺失的参数
    tool_result: str        # 工具调用结果
    is_single_task: bool    # 是否单任务
    is_complete: bool       # 任务是否完成
    loop_count: int         # 当前循环次数
    summary: str            # 最终总结
```

### 图结构设计
```
START → [任务规划] → {单任务?} → YES → [工具选择] → [参数提取] → [参数补齐] → [工具调用] → [总结] → END
                          │                           │
                          NO                          │
                          ↓                           ↓
                    [循环处理] ←←←←←←←←←←←←←←←←←←←←←←←
                          │
                          ↓ (循环次数>=8)
                      [总结] → END
```

### 节点设计

| 节点名称 | 功能 | 对应原方法 |
|----------|------|------------|
| `plan_task` | 判断单/多任务，生成任务描述 | GenerateTaskHub.gen_root_task |
| `select_tool` | 选择合适的工具 | ApiSelectionHub.get_tool_coarse_and_fine |
| `extract_params` | 提取工具参数 | ParamExtractionHub.extraction_params |
| `supplement_params` | 补齐缺失参数 | ApiPlanningHub.supplement_parameters |
| `call_tool` | 调用工具API | ToolUseHub.tool_use |
| `check_loop` | 判断是否继续循环 | GenerateTaskHub.gen_from_context_task + loop_count检查 |
| `summarize` | 总结结果 | ToolSummaryHub.tool_summary |

### 条件边设计

| 条件边 | 判断逻辑 | 下一个节点 |
|--------|----------|------------|
| `is_single_task` | 单任务 → 工具选择；多任务 → 工具选择（进入循环） | select_tool |
| `has_missing_params` | 有缺失参数 → 参数补齐；无缺失 → 工具调用 | supplement_params / call_tool |
| `should_continue` | 未完成且循环<8 → 继续；完成或循环>=8 → 总结 | select_tool / summarize |

---

## 三、实现步骤

### 步骤 1：安装依赖
```bash
pip install langgraph langgraph-checkpoint-sqlite
```

### 步骤 2：创建 Langgraph 核心文件

**文件结构：**
```
langgraph_agent/
├── __init__.py
├── state.py          # 状态定义
├── nodes.py          # 各个节点实现
├── edges.py          # 条件边实现
├── graph.py          # 图定义与编译
└── main.py           # 入口与测试
```

### 步骤 3：实现状态定义 (`state.py`)
定义 `AgentState` TypedDict，包含所有需要共享的状态字段。

### 步骤 4：实现节点 (`nodes.py`)
封装各个处理逻辑为独立函数，每个函数接收 `AgentState` 并返回更新后的状态。

### 步骤 5：实现条件边 (`edges.py`)
定义条件判断函数，决定流程走向。

### 步骤 6：构建图 (`graph.py`)
使用 `StateGraph` 构建完整的流程图，设置最大循环次数为 8。

### 步骤 7：创建通用查询接口 (`api_handler.py`)
基于 mock_api_server.py，提供统一的 API 调用封装，方便 AI 调用。

### 步骤 8：创建入口文件 (`main.py`)
提供测试接口，验证 Langgraph 流程是否正常运行。

---

## 四、关键代码设计

### 4.1 状态定义
```python
from typing import TypedDict, Optional, List
from entity import Tool

class AgentState(TypedDict):
    query: str
    task_description: str
    api_chain: List[dict]
    selected_tool: Optional[Tool]
    params: dict
    missing_params: List[dict]
    tool_result: str
    is_single_task: bool
    is_complete: bool
    loop_count: int
    summary: str
```

### 4.2 任务规划节点
```python
def plan_task(state: AgentState) -> AgentState:
    is_single, task_desc = generate_task_hub.gen_root_task(state["query"])
    return {
        **state,
        "is_single_task": is_single,
        "task_description": task_desc,
        "is_complete": False,
        "loop_count": 0
    }
```

### 4.3 循环检查条件边
```python
def should_continue(state: AgentState) -> str:
    if state["is_complete"] or state["loop_count"] >= 8:
        return "summarize"
    return "select_tool"
```

### 4.4 图构建
```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)
workflow.add_node("plan_task", plan_task)
workflow.add_node("select_tool", select_tool)
workflow.add_node("extract_params", extract_params)
workflow.add_node("supplement_params", supplement_params)
workflow.add_node("call_tool", call_tool)
workflow.add_node("check_loop", check_loop)
workflow.add_node("summarize", summarize)

workflow.set_entry_point("plan_task")
workflow.add_edge("plan_task", "select_tool")
workflow.add_conditional_edges("extract_params", has_missing_params)
workflow.add_edge("supplement_params", "call_tool")
workflow.add_edge("call_tool", "check_loop")
workflow.add_conditional_edges("check_loop", should_continue)
workflow.add_edge("summarize", END)

app = workflow.compile()
```

---

## 五、通用查询接口设计

基于 `mock_api_server.py`，创建统一的 API 处理模块：

| 接口 | 方法 | 功能 |
|------|------|------|
| `/products/getProductByName` | POST | 根据名称查询产品 |
| `/products/getProductById` | POST | 根据ID查询产品 |
| `/products/getBatchProductByProductIds` | POST | 批量查询产品 |
| `/orders/getOrderByOrderId` | POST | 查询订单 |
| `/orders/getByTimeRange` | POST | 按时间范围查询订单 |
| `/suppliers/getSupplierByName` | GET | 查询供应商 |

---

## 六、风险与注意事项

### 潜在风险
1. **循环次数限制**：需确保在 `check_loop` 节点正确更新 `loop_count`
2. **状态同步**：Langgraph 的状态是不可变的，需正确返回新状态
3. **工具调用失败**：需处理 API 调用异常情况
4. **参数补齐失败**：需处理参数补齐失败时的回退逻辑

### 注意事项
1. 保留原有的工具管理和向量检索逻辑
2. LLM 调用使用现有的 `LargeLanguageModel` 封装
3. 确保与原项目的数据格式兼容
4. 提供详细的日志记录便于调试

---

## 七、验证计划

1. **单元测试**：测试每个节点的独立功能
2. **集成测试**：测试完整的 Langgraph 流程
3. **功能测试**：使用 "查询苹果的产品信息" 验证端到端流程
4. **循环测试**：测试多任务场景的循环控制

---

## 八、文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `langgraph_agent/state.py` | 新建 | 状态定义 |
| `langgraph_agent/nodes.py` | 新建 | 节点实现 |
| `langgraph_agent/edges.py` | 新建 | 条件边实现 |
| `langgraph_agent/graph.py` | 新建 | 图定义 |
| `langgraph_agent/main.py` | 新建 | 入口文件 |
| `langgraph_agent/api_handler.py` | 新建 | 通用API调用封装 |
| `app.py` | 修改 | 添加 Langgraph 模式的路由 |
