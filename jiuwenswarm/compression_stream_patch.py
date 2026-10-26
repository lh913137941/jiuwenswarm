# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Runtime patch: make context compression use streaming transport with non-streaming semantics.

背景：部分下游模型只支持流式调用（``stream=True``），非流式 ``Model.invoke()`` 不可用，
导致上下文压缩（走 ``Model.invoke()``）在网关侧失败。但压缩业务本身不需要流式——它要的是
"阻塞等待、一次拿到完整摘要"。

本补丁在服务启动时给 ``CompressionExecutor.invoke`` 打一个运行时补丁：内部改用
``Model.stream()`` 逐 chunk 收流并聚合，**只在收完整个流后**构造一条与 ``Model.invoke()``
返回形态完全一致的 ``AssistantMessage`` 并作为结果返回。

这样：
- 传输层是流式（绕开下游不支持非流式的限制）；
- 语义层等价非流式：调用方 ``base._invoke_compression_with_retries`` 看起来仍是一次
  普通 ``await invoke()`` 拿到全量响应，空内容判定 / 溢出缩窗重试 / fallback 切换全部兼容；
- 对前端零改动：仍只在 started 前、completed/noop/failed 后各发一条
  ``context.compression_state``，不下发任何增量摘要。

幂等；类级 patch，单 Agent 与专家团（leader + member 各自压缩）全局生效。

安全边界：补丁"重写"了 ``CompressionExecutor.invoke``，依赖其内部契约（``build_messages``、
``self._model``、``self._last_response``、``CompressionResult``）。为避免与上游实现漂移时
静默失败，``apply_compression_stream_patch`` 在打补丁前会做一次契约校验，契约不成立时告警
并**保留原始 invoke**（降级回非流式），而不是打上有缺陷的补丁。
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

logger = logging.getLogger("jiuwenswarm.compression_stream_patch")

_PATCH_APPLIED = False


def _normalize_finish_reason(value: Any) -> str:
    """把聚合 chunk 的 ``finish_reason`` 归一为合法的 ``str``。

    网关在终止 chunk 不下发 ``finish_reason``（本仓库已有“网关 chunk 非标准”的先例，见
    llm_sse_patch 注释）、或流被截断时，``agg.finish_reason`` 可能是 ``None`` / ``""`` /
    ``"null"``。``AssistantMessage.finish_reason`` 是必填 ``str`` 字段，直接透传 ``None``
    会触发 pydantic 校验失败；因此与空流分支保持一致：没有有效终止信息一律归为 ``"stop"``。
    """
    if not value or value == "null":
        return "stop"
    return value


def _assemble_assistant_message(agg: Any):
    """把聚合后的 ``AssistantMessageChunk`` 装配成与 ``Model.invoke()`` 同形的 ``AssistantMessage``。

    下游只读 ``.content`` / ``.usage_metadata``，这里补齐全部字段以保证形态一致。
    """
    from openjiuwen.core.foundation.llm.schema.message import AssistantMessage

    if agg is None:
        # 空流：没有任何 chunk（正常不会出现；为空则内容为空，交由引擎按 EMPTY_RESPONSE 处理）
        return AssistantMessage(
            role="assistant",
            content="",
            finish_reason="stop",
            reasoning_content=None,
        )
    return AssistantMessage(
        role="assistant",
        content=agg.content or "",
        tool_calls=agg.tool_calls,
        usage_metadata=agg.usage_metadata,
        finish_reason=_normalize_finish_reason(agg.finish_reason),
        reasoning_content=agg.reasoning_content,
        parser_content=agg.parser_content,
    )


def _build_invoke_kwargs(executor: Any, request: Any) -> dict[str, Any]:
    """按原始 ``CompressionExecutor.invoke`` 相同的规则组装模型调用参数，避免参数丢失。"""
    kwargs: dict[str, Any] = {"messages": executor.build_messages(request), "tools": request.tools}
    if request.output_parser is not None:
        kwargs["output_parser"] = request.output_parser
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    return kwargs


async def _streaming_invoke(  # type: ignore[no-untyped-def]
    self,
    request: Any,
):
    """流式传输 + 非流式语义的 ``CompressionExecutor.invoke`` 实现。

    与原始 ``CompressionExecutor.invoke`` 的差异仅在于：把 ``Model.invoke(**kwargs)``
    换成 ``Model.stream(**kwargs)`` 并聚合；错误分类、结果包装、``last_response`` 记录全部复用。
    若模型不支持流式（理论上不会，但防御上游变化），回退到原始非流式 ``invoke``。
    """
    from openjiuwen.core.context_engine.processor.forked.compressor.support.compression_executor import (
        CompressionResult,
        classify_compression_error,
    )

    model = self._model

    kwargs = _build_invoke_kwargs(self, request)

    # 防御性回退：模型不支持 stream 时绝不静默失败，优先复用原始非流式 invoke。
    if not hasattr(model, "stream"):
        original = getattr(type(self), "_compression_original_invoke", None)
        if callable(original):
            return await original(self, request)
        try:
            response = await model.invoke(**kwargs)
        except Exception as exc:  # noqa: BLE001 - 与原 invoke 一致的异常捕获粒度
            raise classify_compression_error(exc) from exc
        self._last_response = response
        return CompressionResult(
            response=response,
            usage=getattr(response, "usage_metadata", None) or getattr(response, "usage", None),
        )

    agg = None
    try:
        # 下游只支持流式，只能走 stream；在收完整个流之前不对外返回任何结果，
        # 从而对压缩引擎/前端保持"非流式"语义。
        async for chunk in model.stream(**kwargs):
            agg = chunk if agg is None else agg + chunk
    except Exception as exc:  # noqa: BLE001 - 与原 invoke 一致的异常捕获粒度
        raise classify_compression_error(exc) from exc

    response = _assemble_assistant_message(agg)
    self._last_response = response
    return CompressionResult(
        response=response,
        usage=getattr(response, "usage_metadata", None) or getattr(response, "usage", None),
    )


def _is_signature_compatible(executor_cls: type) -> tuple[bool, str]:
    """校验补丁依赖的上游契约是否仍成立；不成立返回 ``(False, 原因)`` 以保留原 invoke。

    补丁"重写"了 ``invoke``，假设了 ``build_messages`` / ``self._model`` /
    ``self._last_response`` / ``CompressionResult`` 等内部契约。这里在打补丁前做一次校验，
    一旦上游实现漂移就告警并跳过，避免在每一次压缩时抛 ``AttributeError``/``TypeError``。
    """
    if not callable(getattr(executor_cls, "build_messages", None)):
        return False, "CompressionExecutor.build_messages 缺失或不可调用"
    try:
        src = inspect.getsource(executor_cls.invoke)
    except (OSError, TypeError):  # 源码不可得时不阻塞，保持乐观
        src = None
    if src is not None:
        for marker in ("self._model", "self._last_response", "CompressionResult("):
            if marker not in src:
                return False, f"原始 invoke 已不引用 '{marker}'，上游实现可能已变化"
    return True, ""


def apply_compression_stream_patch() -> None:
    """给 ``CompressionExecutor.invoke`` 打流式传输补丁。幂等。

    在服务启动早期（``server/app_agentserver.py`` 模块级、与 llm_sse_patch 同处）调用，
    即可覆盖单 Agent 与专家团所有压缩调用。
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    try:
        from openjiuwen.core.context_engine.processor.forked.compressor.support.compression_executor import (
            CompressionExecutor,
        )
    except Exception as exc:  # pragma: no cover - openjiuwen 不可用时静默跳过
        logger.warning("[compression_stream_patch] 未能导入 CompressionExecutor，跳过补丁: %s", exc)
        return

    if getattr(CompressionExecutor, "_compression_stream_patch_applied", False):
        _PATCH_APPLIED = True
        return

    ok, reason = _is_signature_compatible(CompressionExecutor)
    if not ok:
        logger.warning("[compression_stream_patch] 上游契约不兼容，保留原 invoke 并跳过补丁: %s", reason)
        _PATCH_APPLIED = True
        return

    original_invoke = CompressionExecutor.invoke
    CompressionExecutor.invoke = _streaming_invoke  # type: ignore[assignment]
    CompressionExecutor._compression_original_invoke = original_invoke  # type: ignore[attr-defined]
    CompressionExecutor._compression_stream_patch_applied = True  # type: ignore[attr-defined]
    _PATCH_APPLIED = True
    logger.info(
        "[compression_stream_patch] CompressionExecutor.invoke 已改为流式传输（非流式语义）"
    )
