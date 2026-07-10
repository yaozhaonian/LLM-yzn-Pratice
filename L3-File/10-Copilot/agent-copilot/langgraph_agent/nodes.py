import json
import traceback
from typing import Dict, Any

from utils import logger, TASK_SUCCESS_CODE, TASK_ERROR_CODE, RESPONSE_STATUS_CODE_SUCCESS

MAX_LOOP_COUNT = 8


def plan_task(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["query"]
    logger.info(f"[plan_task] 开始规划任务: {query}")
    
    try:
        is_single, task_desc = state["generate_task_hub"].gen_root_task(query)
        logger.info(f"[plan_task] 任务规划结果: is_single={is_single}, task_desc={task_desc}")
        
        return {
            **state,
            "is_single_task": is_single,
            "task_description": task_desc,
            "is_complete": False,
            "loop_count": 0,
            "api_chain": [],
            "params": {},
            "missing_params": [],
            "tool_result": "",
            "error": None
        }
    except Exception as e:
        logger.error(f"[plan_task] 任务规划失败: {e}\n{traceback.format_exc()}")
        return {
            **state,
            "error": f"任务规划失败: {str(e)}",
            "is_complete": True
        }


def select_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["query"]
    task_desc = state["task_description"]
    
    logger.info(f"[select_tool] 选择工具: query={query}, task_desc={task_desc}")
    
    try:
        search_query = task_desc if task_desc else query
        tool = state["api_selection_hub"].get_tool_coarse_and_fine(search_query, None, topK=state["topK"])
        
        if tool is None:
            logger.error(f"[select_tool] 无法找到合适的工具")
            return {
                **state,
                "selected_tool": None,
                "error": "无法找到合适的API工具",
                "is_complete": True,
                "summary": "无法找到合适的API工具"
            }
        
        logger.info(f"[select_tool] 选择工具成功: [{tool.tool_id}] {tool.operationId}-{tool.name_for_human}")
        
        return {
            **state,
            "selected_tool": tool,
            "error": None
        }
    except Exception as e:
        logger.error(f"[select_tool] 工具选择失败: {e}\n{traceback.format_exc()}")
        return {
            **state,
            "selected_tool": None,
            "error": f"工具选择失败: {str(e)}",
            "is_complete": True
        }


def extract_params(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["query"]
    task_desc = state["task_description"]
    tool = state["selected_tool"]
    api_chain = state["api_chain"]
    
    if tool is None:
        return state
    
    logger.info(f"[extract_params] 提取参数: tool={tool.operationId}")
    
    try:
        full_query = query + (" " + task_desc if task_desc else "")
        params, missing_params = state["param_extraction_hub"].extraction_params(
            full_query, tool, api_chain
        )
        logger.info(f"[extract_params] 参数提取结果: params={params}, missing_params={[p.name for p in missing_params]}")
        
        return {
            **state,
            "params": params,
            "missing_params": missing_params,
            "error": None
        }
    except Exception as e:
        logger.error(f"[extract_params] 参数提取失败: {e}\n{traceback.format_exc()}")
        return {
            **state,
            "params": {},
            "missing_params": [],
            "error": f"参数提取失败: {str(e)}",
            "is_complete": True
        }


def supplement_params(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["query"]
    tool = state["selected_tool"]
    params = state["params"]
    missing_params = state["missing_params"]
    
    if tool is None or not missing_params:
        return state
    
    logger.info(f"[supplement_params] 补齐缺失参数: {[p.name for p in missing_params]}")
    
    try:
        for missing_param in missing_params:
            gen_param_query = state["generate_task_hub"].gen_param_task(
                query,
                json.dumps(params, ensure_ascii=False, indent=4),
                f'{missing_param.name}: {missing_param.description}'
            )
            
            supplement_param, supplement_param_tool, supplement_param_result = state["api_planning_hub"].supplement_parameters(
                gen_param_query, missing_param.name
            )
            
            if supplement_param is not None:
                logger.info(f"[supplement_params] 参数 {missing_param.name} 已补齐: {supplement_param}")
                params[missing_param.name] = supplement_param
                if supplement_param_result:
                    state["api_chain"].append(supplement_param_result)
            else:
                logger.error(f"[supplement_params] 参数 {missing_param.name} 补齐失败")
                return {
                    **state,
                    "error": f"参数 {missing_param.name} 补齐失败",
                    "is_complete": True,
                    "summary": f"缺少必要参数 {missing_param.name}"
                }
        
        return {
            **state,
            "params": params,
            "missing_params": [],
            "error": None
        }
    except Exception as e:
        logger.error(f"[supplement_params] 参数补齐失败: {e}\n{traceback.format_exc()}")
        return {
            **state,
            "error": f"参数补齐失败: {str(e)}",
            "is_complete": True
        }


def call_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["query"]
    tool = state["selected_tool"]
    params = state["params"]
    
    if tool is None:
        return state
    
    logger.info(f"[call_tool] 调用工具: {tool.operationId}, params={params}")
    
    try:
        inject_flag, reason = state["generate_task_hub"].gen_judge_task(query, tool, params)
        if inject_flag:
            logger.error(f"[call_tool] 检测到提示注入攻击: {reason}")
            return {
                **state,
                "error": f"提示注入攻击: {reason}",
                "is_complete": True,
                "summary": "检测到提示注入攻击"
            }
        
        tool_response = state["tool_use_hub"].tool_use(tool, params)
        
        if tool_response is None:
            logger.error("[call_tool] 工具调用返回None")
            return {
                **state,
                "tool_result": "",
                "error": "工具调用返回None",
                "is_complete": True,
                "summary": "工具调用失败"
            }
        
        if tool_response.status_code != RESPONSE_STATUS_CODE_SUCCESS:
            logger.error(f"[call_tool] 工具调用失败，状态码: {tool_response.status_code}")
            return {
                **state,
                "tool_result": "",
                "error": "调用外部系统失败",
                "is_complete": True,
                "summary": "调用外部系统失败"
            }
        
        tool_result = tool_response.text
        logger.info(f"[call_tool] 工具调用成功: {tool_result[:200]}...")
        
        api_chain_entry = {
            "code": TASK_SUCCESS_CODE,
            "tool": tool.name_for_human,
            "result": tool_result,
            "missing_param": [],
            "param": params,
            "query": query,
            "task_description": state["task_description"]
        }
        
        return {
            **state,
            "tool_result": tool_result,
            "api_chain": state["api_chain"] + [api_chain_entry],
            "error": None
        }
    except Exception as e:
        logger.error(f"[call_tool] 工具调用异常: {e}\n{traceback.format_exc()}")
        return {
            **state,
            "tool_result": "",
            "error": f"工具调用异常: {str(e)}",
            "is_complete": True
        }


def check_loop(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["query"]
    api_chain = state["api_chain"]
    loop_count = state["loop_count"]
    is_single_task = state["is_single_task"]
    
    logger.info(f"[check_loop] 检查循环: loop_count={loop_count}, is_single_task={is_single_task}, max_loop={MAX_LOOP_COUNT}")
    
    try:
        if is_single_task:
            logger.info(f"[check_loop] 单任务模式，直接完成")
            return {
                **state,
                "is_complete": True,
                "error": None
            }
        
        if loop_count >= MAX_LOOP_COUNT:
            logger.warning(f"[check_loop] 循环次数达到上限 {MAX_LOOP_COUNT}，终止任务")
            return {
                **state,
                "is_complete": True,
                "error": None
            }
        
        is_complete, task_description = state["generate_task_hub"].gen_from_context_task(query, api_chain)
        logger.info(f"[check_loop] 循环检查结果: is_complete={is_complete}, task_description={task_description}")
        
        new_loop_count = loop_count + 1
        logger.info(f"[check_loop] 循环次数更新: {loop_count} -> {new_loop_count}")
        
        return {
            **state,
            "is_complete": is_complete,
            "task_description": task_description,
            "loop_count": new_loop_count,
            "error": None
        }
    except Exception as e:
        logger.error(f"[check_loop] 循环检查失败: {e}\n{traceback.format_exc()}")
        return {
            **state,
            "is_complete": True,
            "error": f"循环检查失败: {str(e)}"
        }


def summarize(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["query"]
    api_chain = state["api_chain"]
    error = state["error"]
    
    logger.info(f"[summarize] 开始总结: query={query}, api_chain_length={len(api_chain)}, loop_count={state['loop_count']}")
    
    try:
        if error:
            summary = f"任务执行失败: {error}"
        elif not api_chain:
            summary = "未执行任何API调用"
        else:
            summary = state["tool_summary_hub"].tool_summary(query, api_chain)
        
        logger.info(f"[summarize] 总结完成: {summary[:200]}...")
        
        return {
            **state,
            "summary": summary,
            "is_complete": True,
            "error": None
        }
    except Exception as e:
        logger.error(f"[summarize] 总结失败: {e}\n{traceback.format_exc()}")
        return {
            **state,
            "summary": f"总结失败: {str(e)}",
            "is_complete": True,
            "error": str(e)
        }
