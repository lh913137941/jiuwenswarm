# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for the runtime compression-stream patch.

正常聚合、空流、流中异常分类、幂等重复调用、openjiuwen 不可用降级、上游契约不一致时跳过。
"""

import builtins

import pytest

import jiuwenswarm.compression_stream_patch as patch


@pytest.fixture(autouse=True)
def _reset_patch():
    """每个用例前后还原补丁的进程级状态，避免用例间相互污染。"""
    from openjiuwen.core.context_engine.processor.forked.compressor.support.compression_executor import (
        CompressionExecutor,
    )

    original_invoke = CompressionExecutor.invoke

    def _restore() -> None:
        CompressionExecutor.invoke = original_invoke
        for _attr in ("_compression_stream_patch_applied", "_compression_original_invoke"):
            try:
                delattr(CompressionExecutor, _attr)
            except AttributeError:
                pass
        patch._PATCH_APPLIED = False

    _restore()
    yield
    _restore()


# ---------------------------------------------------------------------------
# finish_reason 归一（检视意见 #1）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, "stop"),
        ("", "stop"),
        ("null", "stop"),
        ("stop", "stop"),
        ("length", "length"),
        ("tool_calls", "tool_calls"),
    ],
)
def test_normalize_finish_reason(raw, expected):
    """None / 空串 / "null" 一律归为 "stop"，有效值原样透传。"""
    assert patch._normalize_finish_reason(raw) == expected


def test_assemble_empty_stream_uses_stop_finish():
    """空流（无任何 chunk）返回空内容 + finish_reason="stop"，交由引擎按 EMPTY_RESPONSE 处理。"""
    msg = patch._assemble_assistant_message(None)
    assert msg.content == ""
    assert msg.finish_reason == "stop"


def test_assemble_maps_all_fields():
    """正常聚合结果完整映射到 AssistantMessage，含 tool_calls / usage / reasoning。"""
    from openjiuwen.core.foundation.llm.schema.message import UsageMetadata
    from openjiuwen.core.foundation.llm.schema.message_chunk import AssistantMessageChunk
    from openjiuwen.core.foundation.llm.schema.tool_call import ToolCall

    agg = AssistantMessageChunk(
        role="assistant",
        content="summary",
        finish_reason="length",
        reasoning_content="thinking",
        usage_metadata=UsageMetadata(total_tokens=10),
        tool_calls=[ToolCall(id="c1", type="function", name="f", arguments="{}")],
    )
    msg = patch._assemble_assistant_message(agg)
    assert msg.content == "summary"
    assert msg.finish_reason == "length"
    assert msg.reasoning_content == "thinking"
    assert msg.usage_metadata.total_tokens == 10
    assert msg.tool_calls[0].name == "f"


def test_assemble_tolerates_none_finish_reason_via_stub():
    """即使聚合 chunk 的 finish_reason 为 None（网关截断），也不应因 pydantic str 字段校验而失败。"""
    from openjiuwen.core.foundation.llm.schema.message import AssistantMessage

    class _Stub:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    agg = _Stub(
        content="x",
        tool_calls=None,
        usage_metadata=None,
        finish_reason=None,
        reasoning_content=None,
        parser_content=None,
    )
    msg = patch._assemble_assistant_message(agg)
    assert isinstance(msg, AssistantMessage)
    assert msg.finish_reason == "stop"


# ---------------------------------------------------------------------------
# _streaming_invoke 核心路径
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_invoke_forwards_optional_kwargs():
    """output_parser / max_tokens 与原始 invoke 一样透传给 stream，避免参数被静默丢弃。"""
    from openjiuwen.core.context_engine.processor.forked.compressor.support.compression_executor import (
        CompressionExecutor,
        CompressionRequest,
    )
    from openjiuwen.core.foundation.llm.schema.message_chunk import AssistantMessageChunk

    class _Parser:
        pass

    captured = {}

    class FakeModel:
        async def stream(self, **kwargs):
            captured["kwargs"] = kwargs
            yield AssistantMessageChunk(role="assistant", content="done", finish_reason="stop")

    request = CompressionRequest(
        prompt="p",
        output_parser=_Parser(),
        max_tokens=42,
    )
    result = await patch._streaming_invoke(CompressionExecutor(FakeModel()), request)
    assert result.content == "done"
    assert captured["kwargs"]["output_parser"] is request.output_parser
    assert captured["kwargs"]["max_tokens"] == 42


@pytest.mark.asyncio
async def test_streaming_invoke_aggregates_all_chunks():
    """流式 chunk 应逐条聚合，返回非流式语义的完整 AssistantMessage 并记录 last_response。"""
    from openjiuwen.core.context_engine.processor.forked.compressor.support.compression_executor import (
        CompressionExecutor,
        CompressionRequest,
    )
    from openjiuwen.core.foundation.llm.schema.message import UserMessage, UsageMetadata
    from openjiuwen.core.foundation.llm.schema.message_chunk import AssistantMessageChunk

    captured = {}

    class FakeModel:
        async def stream(self, **kwargs):
            captured["kwargs"] = kwargs
            yield AssistantMessageChunk(role="assistant", content="Hello ")
            yield AssistantMessageChunk(
                role="assistant",
                content="world",
                finish_reason="stop",
                usage_metadata=UsageMetadata(total_tokens=7),
            )

    request = CompressionRequest(
        prompt="compress this",
        context_messages=[UserMessage(content="ctx")],
    )
    executor = CompressionExecutor(FakeModel())
    result = await patch._streaming_invoke(executor, request)

    assert result.content == "Hello world"
    assert result.response.finish_reason == "stop"
    assert result.usage.total_tokens == 7
    # last_response 与返回的 response 是同一对象
    assert executor.last_response is result.response
    # 参数组装与原始 invoke 一致：messages 已构建、tools 为 None、未多传未用参数
    assert captured["kwargs"]["tools"] is None
    assert request.prompt in captured["kwargs"]["messages"][-1].content


@pytest.mark.asyncio
async def test_streaming_invoke_classifies_stream_error():
    """流中异常应按原 invoke 的语义分类为 CompressionError（而非裸抛）。"""
    from openjiuwen.core.context_engine.processor.forked.compressor.support.compression_executor import (
        CompressionError,
        CompressionExecutor,
        CompressionRequest,
    )

    class BoomModel:
        async def stream(self, **kwargs):
            raise RuntimeError("connection reset")
            yield  # pragma: no cover

    executor = CompressionExecutor(BoomModel())
    with pytest.raises(CompressionError) as caught:
        await patch._streaming_invoke(executor, CompressionRequest(prompt="x"))
    assert caught.value.kind.name == "UNKNOWN"
    assert "connection reset" in str(caught.value)


@pytest.mark.asyncio
async def test_streaming_invoke_falls_back_when_no_stream():
    """模型不支持 stream（防御上游变化）时回退到非流式 invoke，不静默失败。"""
    from openjiuwen.core.context_engine.processor.forked.compressor.support.compression_executor import (
        CompressionExecutor,
        CompressionRequest,
    )
    from openjiuwen.core.foundation.llm.schema.message import AssistantMessage

    class NonStreamModel:
        async def invoke(self, **kwargs):
            return AssistantMessage(role="assistant", content="fallback-result", finish_reason="stop")

    executor = CompressionExecutor(NonStreamModel())
    result = await patch._streaming_invoke(executor, CompressionRequest(prompt="x"))
    assert result.content == "fallback-result"
    assert executor.last_response.content == "fallback-result"


@pytest.mark.asyncio
async def test_streaming_invoke_reuses_stored_original_invoke():
    """已保留 `_compression_original_invoke` 时，无 stream 模型优先回退到原始 invoke。"""
    from openjiuwen.core.context_engine.processor.forked.compressor.support.compression_executor import (
        CompressionExecutor,
        CompressionRequest,
    )
    from openjiuwen.core.foundation.llm.schema.message import AssistantMessage

    calls = {"stream": 0, "original": 0}

    class NonStreamModel:
        async def invoke(self, **kwargs):
            calls["original"] += 1
            return AssistantMessage(role="assistant", content="original-answer", finish_reason="stop")

    executor = CompressionExecutor(NonStreamModel())
    # 模拟 apply 时保存的原始 invoke
    executor.__class__._compression_original_invoke = CompressionExecutor.invoke
    result = await patch._streaming_invoke(executor, CompressionRequest(prompt="x"))
    assert result.content == "original-answer"
    assert calls["original"] == 1


# ---------------------------------------------------------------------------
# apply_compression_stream_patch 机制
# ---------------------------------------------------------------------------

def test_apply_patch_is_idempotent():
    """重复 apply 只打一次补丁：invoke 被替换为流式实现且保存原始引用，二次调用不再次包装。"""
    from openjiuwen.core.context_engine.processor.forked.compressor.support.compression_executor import (
        CompressionExecutor,
    )

    original = CompressionExecutor.invoke
    patch._PATCH_APPLIED = False

    patch.apply_compression_stream_patch()
    assert CompressionExecutor.invoke is patch._streaming_invoke
    assert CompressionExecutor._compression_original_invoke is original
    assert CompressionExecutor._compression_stream_patch_applied is True

    # 幂等：二次调用不再改动
    patch.apply_compression_stream_patch()
    assert CompressionExecutor.invoke is patch._streaming_invoke
    assert CompressionExecutor._compression_original_invoke is original


def test_signature_compatible_detects_marker_drift():
    """源码级校验应发现上游 invoke 不再引用补丁依赖的内部属性，从而保留原 invoke。"""
    class _Drifted:
        def build_messages(self, request):
            return []

        def invoke(self, request):
            return "no internal contract references here"

    ok, reason = patch._is_signature_compatible(_Drifted)
    assert ok is False
    assert "self._model" in reason


def test_apply_skips_when_upstream_contract_drifted():
    """上游契约不一致（build_messages 缺失）时告警跳过，保留原 invoke，不打上有缺陷的补丁。"""
    from openjiuwen.core.context_engine.processor.forked.compressor.support.compression_executor import (
        CompressionExecutor,
    )

    original = CompressionExecutor.invoke
    saved_build_messages = CompressionExecutor.build_messages
    patch._PATCH_APPLIED = False

    try:
        CompressionExecutor.build_messages = None  # 模拟上游契约漂移
        patch.apply_compression_stream_patch()
        assert CompressionExecutor.invoke is original  # 未被替换
        assert not hasattr(CompressionExecutor, "_compression_stream_patch_applied")
    finally:
        CompressionExecutor.build_messages = saved_build_messages


def test_apply_skips_when_openjiuwen_unavailable(monkeypatch):
    """openjiuwen 不可导入时静默降级（仅告警），不抛错、不打补丁。"""
    from openjiuwen.core.context_engine.processor.forked.compressor.support.compression_executor import (
        CompressionExecutor,
    )

    original = CompressionExecutor.invoke
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name and name.startswith("openjiuwen.core.context_engine.processor.forked.compressor"):
            raise ImportError("openjiuwen unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    patch._PATCH_APPLIED = False

    patch.apply_compression_stream_patch()  # 不应抛出
    assert patch._PATCH_APPLIED is False
    assert CompressionExecutor.invoke is original  # 未被打补丁
