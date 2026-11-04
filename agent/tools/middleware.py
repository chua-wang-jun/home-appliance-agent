import json
from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import (
    ModelRequest,
    before_model,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts, load_system_prompts


def _context_value(context, key: str, default=None):
    """兼容字典和对象形式的 runtime context。"""
    if isinstance(context, dict):
        return context.get(key, default)
    return getattr(context, key, default)


def _context_set(context, key: str, value) -> None:
    """兼容字典和对象形式的 runtime context 写入。"""
    if isinstance(context, dict):
        context[key] = value
    elif context is not None:
        setattr(context, key, value)


def _context_setdefault(context, key: str, value) -> None:
    """只在上下文没有有效值时写入，保证调用方显式传入的数据优先。"""
    if value and not _context_value(context, key):
        _context_set(context, key, value)


def _tool_result_text(result: ToolMessage | Command) -> str:
    """从 ToolMessage 或 Command 中提取工具返回的文本。"""
    if isinstance(result, ToolMessage):
        content = result.content
    elif isinstance(result, Command):
        update = result.update if isinstance(result.update, dict) else {}
        messages = update.get("messages", [])
        content = messages[-1].content if messages else ""
    else:
        content = getattr(result, "content", result)

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content) if content is not None else ""


def _append_knowledge_context(context, text: str) -> None:
    """累积 RAG 检索结果，并限制写入报告提示词的长度。"""
    if not text:
        return

    current = _context_value(context, "knowledge_context", "") or ""
    if text in current:
        return

    combined = f"{current}\n\n{text}".strip() if current else text
    limit = int(agent_conf.get("max_knowledge_context_chars", 6000))
    _context_set(context, "knowledge_context", combined[:limit])


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Command],
) -> ToolMessage | Command:
    """记录工具调用，并把工具结果自动写入运行时上下文。"""
    tool_name = request.tool_call["name"]
    args = request.tool_call.get("args") or {}
    logger.info(f"[tool monitor]执行工具：{tool_name}")
    logger.info(f"[tool monitor]传入参数：{args}")

    try:
        result = handler(request)
        text = _tool_result_text(result)
        context = request.runtime.context

        if tool_name == "get_user_id":
            _context_setdefault(context, "user_id", text)
        elif tool_name == "get_current_month":
            _context_setdefault(context, "period", text)
        elif tool_name == "get_user_location":
            _context_setdefault(context, "location", text)
        elif tool_name == "get_weather":
            _context_setdefault(context, "weather", text)
        elif tool_name == "fetch_external_data":
            _context_setdefault(context, "usage_data", text)
            _context_setdefault(context, "period", args.get("month", ""))
            _context_setdefault(context, "user_id", args.get("user_id", ""))
        elif tool_name == "rag_summarize":
            _append_knowledge_context(context, text)
        elif tool_name in {"fill_context_for_report", "file_context_for_report"}:
            _context_set(context, "report", True)
            logger.info("[tool monitor]报告场景已激活，下一次模型调用将切换提示词")

        logger.info(f"[tool monitor]工具{tool_name}调用成功")
        return result
    except Exception as e:
        logger.error(f"工具{tool_name}调用失败，原因{str(e)}")
        raise


@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    """模型调用前记录当前消息数量和最后一条消息类型。"""
    messages = state["messages"]
    logger.info(f"[log_before_model]即将调用模型，带有{len(messages)}条消息")

    if messages:
        last_message = messages[-1]
        content = last_message.content
        preview = content if isinstance(content, str) else str(content)
        logger.debug(
            f"[log_before_model]{type(last_message).__name__} | {preview[:200]}"
        )
    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest) -> str:
    """根据 runtime context 在普通客服和报告提示词之间切换。"""
    context = request.runtime.context
    if not _context_value(context, "report", False):
        return load_system_prompts()

    return load_report_prompts().format(
        user_id=_context_value(context, "user_id", "未提供"),
        period=_context_value(context, "period", "未提供"),
        location=_context_value(context, "location", "未提供"),
        weather=_context_value(context, "weather", "未提供"),
        usage_data=_context_value(context, "usage_data", "未提供"),
        knowledge_context=_context_value(context, "knowledge_context", "未提供"),
    )
