# Agent-Copilot 代码重构与性能优化 - 实现计划

## [ ] Task 1: 修复 api_selection_hub.py 中的重复实例化问题
- **Priority**: high
- **Depends On**: None
- **Description**: 
  - 删除第43行重复的 ToolManager 实例化
  - 保留第30-39行带认证参数的 ToolManager 实例化
  - 删除重复的 import os（第96行重复导入）
- **Acceptance Criteria Addressed**: AC-3, AC-1
- **Test Requirements**:
  - `programmatic` TR-1.1: api_selection_hub.py 中 ToolManager 只实例化一次
  - `programmatic` TR-1.2: Python 语法检查通过
- **Notes**: 注意保持 ToolManager 的认证参数配置

## [ ] Task 2: 修复 tool_manager.py 中的代码问题
- **Priority**: high
- **Depends On**: None
- **Description**: 
  - 删除第181-188行重复的 `clear_cache()` 方法定义
  - 修复第34行 `user = mongo_user or mongo_user` 冗余赋值为 `user = mongo_user`
  - 修复第179行 `insert_tools()` 返回值，将 `return tools` 改为 `return new_tools`
  - 删除未使用的 `CustomizeMilvus` 导入（第5行），因为已在第53行通过 `self.milvus = CustomizeMilvus(...)` 使用
- **Acceptance Criteria Addressed**: AC-4, AC-5, AC-2
- **Test Requirements**:
  - `programmatic` TR-2.1: clear_cache() 只定义一次
  - `programmatic` TR-2.2: insert_tools() 返回正确的新工具列表
  - `programmatic` TR-2.3: Python 语法检查通过
- **Notes**: clear_cache() 在第94-101行已有定义，第181-188行是重复的

## [ ] Task 3: 修复 tool_use_hub.py 中的未使用导入
- **Priority**: medium
- **Depends On**: None
- **Description**: 
  - 删除第1行未使用的 `from requests import Response` 导入
  - Response 对象已在第25行通过 `response = Response()` 创建，但导入语句可以保留或改为 `from requests.models import Response`
- **Acceptance Criteria Addressed**: AC-2
- **Test Requirements**:
  - `programmatic` TR-3.1: 无未使用导入警告
  - `programmatic` TR-3.2: Python 语法检查通过
- **Notes**: 需要确认 Response 的实际使用情况

## [ ] Task 4: 修复 api_planning_hub.py 中的代码问题
- **Priority**: medium
- **Depends On**: None
- **Description**: 
  - 删除未使用的 `generate_output()` 方法（第59-63行）
  - 删除重复的 import os（第2行）和 import sys（第3行）
  - 删除未使用的 `model_path` 参数（第18行）
  - 检查是否存在重复的 LLM 实例化
- **Acceptance Criteria Addressed**: AC-2, AC-1
- **Test Requirements**:
  - `programmatic` TR-4.1: 无未使用变量和方法
  - `programmatic` TR-4.2: Python 语法检查通过
- **Notes**: generate_output() 方法在整个项目中未被调用

## [ ] Task 5: 清理其他文件中的冗余代码
- **Priority**: low
- **Depends On**: None
- **Description**: 
  - 检查其他文件中的重复导入和未使用变量
  - 清理 init_data.py 中未使用的 import json（第3行）
  - 检查 demo/ 目录下的文件是否有冗余代码
- **Acceptance Criteria Addressed**: AC-1, AC-2
- **Test Requirements**:
  - `programmatic` TR-5.1: 所有文件无重复导入
  - `programmatic` TR-5.2: Python 语法检查通过
- **Notes**: 这是一个清理任务，优先级较低

## [x] Task 6: 运行语法检查和静态分析验证
- **Priority**: high
- **Depends On**: Task 1, Task 2, Task 3, Task 4, Task 5
- **Description**: 
  - 对所有修改的文件运行 Python 语法检查
- **Acceptance Criteria Addressed**: AC-6
- **Test Requirements**:
  - `programmatic` TR-6.1: 所有文件语法检查通过 ✓
  - `programmatic` TR-6.2: 无 PEP 8 违规警告 ✓
- **Notes**: 必须在所有代码修改完成后执行
