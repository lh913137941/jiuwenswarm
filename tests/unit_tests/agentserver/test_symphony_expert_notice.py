# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""No-plan text follows real tool events through the facade and disk history."""

import asyncio
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from jiuwenswarm.agents.harness.common.rails.stream_event_rail import JiuSwarmStreamEventRail
from jiuwenswarm.common.schema.agent import AgentRequest, AgentResponseChunk
from jiuwenswarm.server.runtime.agent_adapter import interface as facade
from jiuwenswarm.server.runtime.agent_adapter.interface_deep import JiuWenSwarmDeepAdapter
from jiuwenswarm.server.runtime.session import session_history, session_metadata
from jiuwenswarm.symphony import expert_team
from jiuwenswarm.symphony.expert_notice import ExpertNoticeState, ExpertNoticeStream, NO_PLAN_MESSAGES


async def collect_no_plan_scenario(
    monkeypatch, tmp_path, *, candidate_count=0, dict_chunks=False, interrupt=False, ending="final", event_batches=None,
):
    """Reusable controlled backend fixture for the unchanged desktop UI QA."""
    session_id = f"no-plan-{candidate_count}"
    model = SimpleNamespace(invoke=AsyncMock(return_value={"content": '{"status":"no_plan"}'}))
    experts = {
        f"e{i}": expert_team.LocalExpert(f"e{i}", f"专家{i + 1}", "资料分析", object())
        for i in range(candidate_count)
    }
    monkeypatch.setattr(expert_team, "scan_local_experts", AsyncMock(return_value=experts))
    monkeypatch.setattr(expert_team, "_subagent_config", lambda _agent, expert, _model: expert.id)
    bind = AsyncMock()
    monkeypatch.setattr(expert_team, "apply_extension_hot", bind)
    round_state = expert_team.DynamicExpertRound(object(), "请专家整理资料", model, session_id=session_id)
    await round_state.compose("请专家整理资料")
    tool = expert_team.ExpertPlanTool(lambda sid: round_state if sid == session_id else None)
    result = await tool.invoke({}, session=SimpleNamespace(get_session_id=lambda: session_id))
    emitted = AsyncMock()
    await JiuSwarmStreamEventRail()._emit_tool_result(
        SimpleNamespace(write_stream=emitted),
        SimpleNamespace(name=expert_team.PLAN_TOOL_NAME, id="plan-call"), result,
    )
    planner_result = JiuWenSwarmDeepAdapter._parse_stream_chunk(emitted.await_args.args[0])
    initial = [
        {"event_type": "chat.tool_call", "tool_call": {"name": expert_team.PLAN_TOOL_NAME, "tool_call_id": "plan-call"}},
        planner_result, dict(planner_result),
        {"event_type": "chat.delta", "content": "我先整理已有资料。\n\n"},
        {"event_type": "chat.tool_call", "tool_call": {"name": "read_file", "tool_call_id": "read-1"}},
        {"event_type": "chat.tool_result", "tool_name": "read_file", "tool_call_id": "read-1", "result": "资料内容"},
        {"event_type": "chat.delta", "content": "已读完资料。\n\n"},
        {"event_type": "chat.tool_call", "tool_call": {"name": "read_file", "tool_call_id": "read-2"}},
    ]
    tail = [
        {"event_type": "chat.delta", "content": "用户任务已完成。"},
        {"event_type": "chat.final", "content": "尚待修订的中间稿。"},
        {"event_type": "chat.final", "content": "用户任务已完成。"},
        {"event_type": "chat.final", "content": ""},
    ]
    if event_batches is not None:
        batches = [initial + event_batches[0], *event_batches[1:]]
    elif interrupt:
        initial.append({"event_type": "chat.ask_user_question", "request_id": "ask-1", "source": "ask_user_interrupt",
                        "questions": [{"question": "希望输出什么格式？", "options": [{"label": "表格"}]}]})
        # Core may repeat the original tool event while restoring its output.
        batches = [initial, [dict(planner_result), *tail]]
    elif ending == "error":
        batches = [initial + [{"event_type": "chat.error", "error": "后续模型连接失败"}]]
    elif ending == "cancel":
        batches = [initial]
    else:
        batches = [initial + tail]

    class Adapter:
        def __init__(self):
            self.current = round_state

        def get_dynamic_expert_notice_state(self, sid):
            return self.current.notice_state if self.current is not None and sid == session_id else None

        async def process_message_stream_impl(self, request, _inputs):
            batch = batches.pop(0)
            for payload in batch:
                yield payload if dict_chunks else AgentResponseChunk(request.request_id, request.channel_id, payload)
            if ending == "cancel":
                raise asyncio.CancelledError
            if not batches:
                await round_state.close()
                self.current = None

    adapter = Adapter()
    monkeypatch.setattr(facade.JiuWenSwarm, "_ensure_adapter", lambda *_args, **_kwargs: adapter)
    monkeypatch.setattr(facade, "get_config", lambda: {"preferred_language": "zh", "memory": {"mode": "disabled"}})
    monkeypatch.setattr(facade, "get_memory_mode", lambda _config: "disabled")
    monkeypatch.setattr(facade, "_schedule_symphony_session_feedback", lambda *_args: None)
    monkeypatch.setattr(session_history, "get_agent_sessions_dir", lambda: tmp_path)
    monkeypatch.setattr(session_metadata, "get_agent_sessions_dir", lambda: tmp_path)
    # Metadata is outside this test; history's real async writer/read-side dedup is retained.
    monkeypatch.setattr(session_metadata, "update_session_metadata", lambda **_kwargs: None)
    monkeypatch.setattr(session_metadata, "set_session_delivery_context", lambda **_kwargs: None)
    monkeypatch.setattr(session_metadata, "set_context_snapshot_tail", lambda *_args: None)
    swarm = facade.JiuWenSwarm()
    chunks = []
    for index in range(len(batches)):
        params = {"query": "请专家整理资料", "mode": "agent"}
        if index:
            params.update(source="ask_user_interrupt", request_id="ask-1", answers=[{"selected_options": ["表格"]}])
        request = AgentRequest(f"no-plan-request-{index}", "desktop", session_id, params=params, is_stream=True)
        try:
            async for chunk in swarm.process_message_stream(request):
                chunks.append(asdict(chunk))
        except asyncio.CancelledError:
            assert ending == "cancel"
    session_history.flush_history_writes()
    history = session_history.load_history_records(session_id)
    return SimpleNamespace(chunks=chunks, history=history, result=result, model=model, bind=bind, round=round_state)


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate_count,reason", [(0, "no_available_experts"), (3, "no_suitable_plan")])
@pytest.mark.parametrize("dict_chunks", [False, True])
async def test_no_plan_is_visible_once_survives_snapshots_and_history(
    monkeypatch, tmp_path, candidate_count, reason, dict_chunks,
):
    result = await collect_no_plan_scenario(monkeypatch, tmp_path, candidate_count=candidate_count, dict_chunks=dict_chunks)
    notice = NO_PLAN_MESSAGES[reason]
    assert result.result == {"success": True, "status": "no_plan", "reason": reason}
    assert result.model.invoke.await_count == (1 if candidate_count else 0)
    result.bind.assert_not_awaited()
    payloads = [c["payload"] for c in result.chunks]
    assert not any(p.get("event_type") in {"chat.ask_user_question", "chat.notice"} for p in payloads)
    assert sum(notice in p.get("content", "") for p in payloads if p.get("event_type") == "chat.delta") == 1
    finals = [p["content"] for p in payloads if p.get("event_type") == "chat.final"]
    assert finals[-1].count(notice) == 1
    assert "用户任务已完成。" in finals[-1]
    stored = "".join(r["content"] for r in result.history if r.get("event_type") == "chat.final")
    assert stored == finals[-1]
    assert stored.count("我先整理已有资料。") == stored.count("已读完资料。") == 1
    assert "不要重复提示" in result.round.guidance()


@pytest.mark.asyncio
async def test_other_tool_hitl_resume_preserves_notice_without_duplicate_history(monkeypatch, tmp_path):
    result = await collect_no_plan_scenario(monkeypatch, tmp_path, candidate_count=3, interrupt=True)
    notice = NO_PLAN_MESSAGES["no_suitable_plan"]
    second = [c["payload"] for c in result.chunks if c["request_id"] == "no-plan-request-1"]
    assert not any(notice in p.get("content", "") for p in second if p.get("event_type") == "chat.delta")
    final = [p["content"] for p in second if p.get("event_type") == "chat.final"][-1]
    assert final.count(notice) == 1 and final.endswith("用户任务已完成。")
    persisted = [r for r in result.history if r.get("event_type") == "chat.final"]
    assert "".join(r["content"] for r in persisted) == final
    assert all(notice not in r["content"] for r in persisted if r["request_id"] == "no-plan-request-1")
    result.model.invoke.assert_awaited_once()
    result.bind.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("dict_chunks", [False, True])
async def test_deferred_final_retains_model_and_latest_usage_without_routing_pollution(monkeypatch, tmp_path, dict_chunks):
    result = await collect_no_plan_scenario(monkeypatch, tmp_path, dict_chunks=dict_chunks, event_batches=[[
        {"event_type": "chat.reasoning", "content": "真实流式思考"},
        {"event_type": "chat.final", "content": "任务完成。", "model": "model-a", "model_provider": "provider-a",
         "api_type": "chat_completions", "usage": {"output_tokens": 1}, "reasoning_content": "旧思考快照",
         "source": "final-source", "expert_id": "expert-a", "expert_name": "专家A", "aborted": False,
         "id": "wrong-id", "request_id": "wrong-request", "channel_id": "wrong-channel", "role": "leader",
         "member_name": "", "timestamp": 1},
        {"event_type": "chat.final", "content": "", "usage": {"output_tokens": 2}},
    ]])
    stored = [r for r in result.history if r.get("event_type") == "chat.final"][-1]
    assert (stored["model"], stored["model_provider"], stored["api_type"]) == ("model-a", "provider-a", "chat_completions")
    assert stored["usage"] == {"output_tokens": 2}
    assert stored["source"] == "final-source" and stored["expert_id"] == "expert-a"
    assert stored["reasoning_content"] == "真实流式思考"
    assert (stored["id"], stored["request_id"], stored["channel_id"], stored["role"]) == (
        "no-plan-request-0:assistant", "no-plan-request-0", "desktop", "assistant",
    )
    assert "member_name" not in stored and stored["timestamp"] != 1


@pytest.mark.asyncio
@pytest.mark.parametrize("dict_chunks", [False, True])
async def test_same_text_after_flush_persists_latest_model_usage(monkeypatch, tmp_path, dict_chunks):
    result = await collect_no_plan_scenario(monkeypatch, tmp_path, dict_chunks=dict_chunks, event_batches=[[
        {"event_type": "chat.final", "content": "相同正文。", "model": "first", "usage": {"output_tokens": 1}},
        {"event_type": "chat.tool_call", "tool_call": {"name": "read_file", "tool_call_id": "read-3"}},
        {"event_type": "chat.final", "content": "", "model": "latest", "usage": {"output_tokens": 2}},
    ]])
    stored = [r for r in result.history if r.get("event_type") == "chat.final"][-1]
    assert stored["model"] == "latest" and stored["usage"] == {"output_tokens": 2}
    assert stored["content"].endswith("相同正文。")


@pytest.mark.asyncio
async def test_deferred_metadata_does_not_leak_across_tool_boundaries_or_resume(monkeypatch, tmp_path):
    result = await collect_no_plan_scenario(monkeypatch, tmp_path, event_batches=[[
        {"event_type": "chat.final", "content": "第一段。", "model": "first", "usage": {"output_tokens": 1}},
        {"event_type": "chat.tool_call", "tool_call": {"name": "read_file", "tool_call_id": "read-3"}},
        {"event_type": "chat.delta", "content": "后续独立正文。"},
        {"event_type": "chat.tool_call", "tool_call": {"name": "read_file", "tool_call_id": "read-4"}},
        {"event_type": "chat.ask_user_question", "source": "ask_user_interrupt", "request_id": "ask-1"},
    ], [
        {"event_type": "chat.delta", "content": "恢复后的正文。"},
        {"event_type": "chat.final", "content": "恢复后的正文。"},
    ]])
    raw = session_history.read_session_history_records("no-plan-0")
    finals = [r for r in raw if r.get("event_type") == "chat.final"]
    assert any(r.get("model") == "first" for r in finals)
    later = [r for r in finals if "后续独立正文。" in r["content"] or r["request_id"] == "no-plan-request-1"]
    assert later and all("model" not in r and "usage" not in r for r in later)
    stored = "".join(r["content"] for r in result.history if r.get("event_type") == "chat.final")
    assert stored.count(NO_PLAN_MESSAGES["no_available_experts"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["error", "cancel"])
async def test_no_plan_notice_remains_in_history_when_following_work_fails_or_stops(monkeypatch, tmp_path, ending):
    result = await collect_no_plan_scenario(monkeypatch, tmp_path, ending=ending)
    stored = "".join(r["content"] for r in result.history if r.get("event_type") == "chat.final")
    assert stored.count(NO_PLAN_MESSAGES["no_available_experts"]) == 1
    assert any(c["payload"].get("event_type") == "chat.error" for c in result.chunks)
    if ending == "cancel":
        assert any(r.get("aborted") is True for r in result.history)


@pytest.mark.parametrize("changes", [
    {"tool_name": "another_tool"}, {"member_name": "固定团队成员"}, {"success": False},
    {"raw_output": {"success": False, "status": "error", "reason": "expert_planning_failed"}},
    {"raw_output": {"success": True, "status": "ready"}}, {"tool_call_id": ""},
])
def test_non_planner_member_and_error_events_remain_unchanged(changes):
    stream = ExpertNoticeStream(ExpertNoticeState())
    payload = {
        "event_type": "chat.tool_result", "tool_name": expert_team.PLAN_TOOL_NAME, "tool_call_id": "call",
        "raw_output": {"success": True, "status": "no_plan", "reason": "no_available_experts"}, **changes,
    }
    assert stream.consume(payload) == (payload, "")
    final = {"event_type": "chat.final", "content": "正常回答"}
    assert stream.consume(final) == (final, "")
    assert not stream.active


def test_summon_choice_does_not_insert_continue_notice():
    stream = ExpertNoticeStream(ExpertNoticeState())
    result = {
        "event_type": "chat.tool_result", "tool_name": expert_team.PLAN_TOOL_NAME,
        "tool_call_id": "plan-call",
        "raw_output": {
            "success": True, "status": "no_plan", "reason": "no_suitable_plan",
            "decision": "summon", "confirmation_id": "choice-1", "candidate_count": 1,
        },
    }
    assert stream.consume(result) == (result, "")
    assert not stream.active


def test_adapter_returns_only_current_session_round_state():
    child = object.__new__(JiuWenSwarmDeepAdapter)
    child._is_session_scoped_adapter = True
    state = ExpertNoticeState()
    child._dynamic_expert_rounds = {"session-a": SimpleNamespace(notice_state=state)}
    root = object.__new__(JiuWenSwarmDeepAdapter)
    root._is_session_scoped_adapter = False
    root._session_adapters = {"session-a": child}
    assert root.get_dynamic_expert_notice_state("session-a") is state
    assert root.get_dynamic_expert_notice_state("session-b") is None
    child._dynamic_expert_rounds.clear()
    assert root.get_dynamic_expert_notice_state("session-a") is None
