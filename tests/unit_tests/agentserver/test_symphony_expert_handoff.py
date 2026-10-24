"""单专家接管使用真实扩展装载，原回合失败与取消均回滚会话绑定。"""

import asyncio
import json
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from openjiuwen.core.runner import Runner
from openjiuwen.harness.rails import SkillUseRail

from jiuwenswarm.server.runtime.agent_adapter.expert_capability import ExpertApplyBusyError
from jiuwenswarm.server.runtime.expert import expert_store
from jiuwenswarm.server.runtime.session import session_metadata
from jiuwenswarm.symphony.expert_team import DynamicExpertRound, LocalExpert
from tests.unit_tests.agentserver.test_expert_replay import _make_child_adapter, _make_package

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]


async def _make_adapter_with_skill_rail(session_id, work_mode="work"):
    adapter = _make_child_adapter(session_id, work_mode)
    await adapter._instance.register_rail(SkillUseRail(skills_dir=[], skill_mode="all", include_tools=False))
    return adapter


def _skill_names(adapter):
    rail = adapter._instance.find_rails_by_type((SkillUseRail,))[0]
    return {skill.name for skill in rail.get_skills_for_session()}


async def _pending_handoff(tmp_path, monkeypatch, work_mode="work"):
    session_id = f"handoff-{uuid4().hex}"
    monkeypatch.setattr(session_metadata, "get_agent_sessions_dir", lambda: tmp_path / "sessions")
    from jiuwenswarm.common.mode_profiles import profile_for_work_mode

    session_metadata.update_session_metadata(
        session_id=session_id, expert_id="", mode=profile_for_work_mode(work_mode).single_canonical,
        work_mode=work_mode, project_id="host-project", project_dir=str(tmp_path / "project"), sync=True,
    )
    package = _make_package(tmp_path / "selected-cache", "expert-a", "你是确认选中的专家。")
    skill_dir = package / "skills" / "handoff-private"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: handoff-private\ndescription: 专家私有技能\n---\n技能正文")
    (package / "echo.py").write_text(
        'from openjiuwen.core.foundation.tool import Tool, ToolCard\n'
        'class EchoTool(Tool):\n'
        '    def __init__(self):\n'
        '        super().__init__(ToolCard(id="handoff_echo", name="handoff_echo", description="专家工具"))\n'
        '    async def invoke(self, inputs, **kwargs):\n'
        '        return {"echo": "ok"}\n'
        '    async def stream(self, inputs, **kwargs):\n'
        '        yield {"echo": "ok"}\n'
    )
    manifest = json.loads((package / "manifest.json").read_text())
    manifest.update({
        "tools": [{"file": "echo.py", "class": "EchoTool"}],
        "skills": [{"dir": "skills/handoff-private"}],
    })
    (package / "manifest.json").write_text(json.dumps(manifest))
    expert = LocalExpert("expert-a", "专家阿甲", "专家介绍", object(), package, "cache", "avatar.png")
    adapter = await _make_adapter_with_skill_rail(session_id, work_mode)
    monkeypatch.setattr(adapter, "_is_session_live", lambda _: True)
    round_state = DynamicExpertRound(
        adapter._instance, "请专家整理资料", object(), session_id=session_id,
        handoff_expert=adapter._handoff_confirmed_expert,
    )
    round_state.pending_plan = {"nodes": {expert.id: {"label": expert.name, "task": "整理资料"}}, "edges": []}
    round_state._selected_experts = {expert.id: expert}
    round_state.response = {"success": True, "status": "awaiting_confirmation"}
    adapter._dynamic_expert_rounds = {session_id: round_state}
    metadata = round_state.request_confirmation(session_id, "plan-call")
    return adapter, round_state, expert, metadata


async def _confirm(round_state, metadata):
    nonce = round_state.claim_confirmation_resume(round_state.session_id, "plan-call", metadata)
    return await round_state.confirm_and_bind(
        round_state.session_id, "plan-call", metadata, confirmation_nonce=nonce,
    )


@pytest.mark.parametrize("work_mode", ["work", "code", "design"])
async def test_confirmed_handoff_keeps_real_tools_skills_binding_and_replays_after_rebuild(tmp_path, monkeypatch, work_mode):
    adapter, round_state, expert, metadata = await _pending_handoff(tmp_path, monkeypatch, work_mode)
    original_metadata = session_metadata.get_session_metadata(round_state.session_id).copy()
    original_workspace = adapter._instance.deep_config.workspace
    original_mode = (adapter._session_instance_mode, adapter._session_instance_sub_mode)
    resolver = AsyncMock(side_effect=AssertionError("确认时不得重新选择或下载专家"))
    monkeypatch.setattr(expert_store, "resolve_expert_package_dir", resolver)
    with pytest.raises(ExpertApplyBusyError):
        await adapter.apply_expert(expert.id, package_dir=expert.package_dir)

    result = await _confirm(round_state, metadata)

    assert result["status"] == "handed_off" and result["plan_id"] == metadata["dynamic_expert_plan_id"]
    assert adapter._current_expert_id == expert.id
    assert _skill_names(adapter) == {"handoff-private"}
    tool = Runner.resource_mgr.get_tool(adapter._instance.ability_manager.get("handoff_echo").id)
    assert await tool.invoke({}) == {"echo": "ok"}
    durable = json.loads((tmp_path / "sessions" / round_state.session_id / "metadata.json").read_text())
    assert (durable["expert_id"], durable["expert_type"]) == (expert.id, "agent")
    for key in ("work_mode", "mode", "project_id", "project_dir"):
        assert durable[key] == original_metadata[key]
    assert adapter._instance.deep_config.workspace is original_workspace
    assert (adapter._session_instance_mode, adapter._session_instance_sub_mode) == original_mode
    load_record = adapter._expert_load_record
    await adapter._close_dynamic_expert_round(round_state.session_id)
    await adapter._reconcile_expert_binding_on_reuse()
    assert adapter._expert_load_record is load_record and adapter._current_expert_id == expert.id
    assert _skill_names(adapter) == {"handoff-private"}
    assert not adapter._instance.deep_config.subagents
    resolver.assert_not_awaited()

    rebuilt = await _make_adapter_with_skill_rail(round_state.session_id, work_mode)
    resolver.side_effect = None
    resolver.return_value = expert.package_dir
    await rebuilt._replay_expert_from_metadata()
    assert rebuilt._current_expert_id == expert.id and _skill_names(rebuilt) == {"handoff-private"}
    assert rebuilt._instance.ability_manager.get("handoff_echo") is not None
    assert (rebuilt._session_instance_mode, rebuilt._session_instance_sub_mode) == original_mode
    assert session_metadata.get_session_metadata(round_state.session_id)["work_mode"] == work_mode
    await adapter._apply_expert(None)
    await rebuilt._apply_expert(None)


@pytest.mark.parametrize("failure", ["load", "metadata"])
@pytest.mark.parametrize("work_mode", ["work", "code", "design"])
async def test_handoff_failures_clear_loaded_resources_and_metadata(tmp_path, monkeypatch, failure, work_mode):
    adapter, round_state, expert, metadata = await _pending_handoff(tmp_path, monkeypatch, work_mode)
    if failure == "load":
        monkeypatch.setattr(adapter._instance, "load_agent_template", AsyncMock(side_effect=ValueError("装载失败")))
    else:
        original = session_metadata._write_metadata_sync

        def fail_expert_write(session_id, value, **kwargs):
            if value.get("expert_id") == expert.id:
                raise OSError("写入失败")
            return original(session_id, value, **kwargs)

        monkeypatch.setattr(session_metadata, "_write_metadata_sync", fail_expert_write)

    result = await _confirm(round_state, metadata)

    assert result == {
        "success": False, "status": "handoff_failed",
        "plan_id": metadata["dynamic_expert_plan_id"],
        "reason": "expert_load_failed" if failure == "load" else "expert_metadata_write_failed",
    }
    assert adapter._current_expert_id is None and adapter._expert_load_record is None
    assert adapter._instance.ability_manager.get("handoff_echo") is None
    assert not _skill_names(adapter)
    assert session_metadata.get_session_metadata(round_state.session_id)["expert_id"] == ""
    assert session_metadata.get_session_metadata(round_state.session_id, cache_bust=True)["expert_id"] == ""
    assert round_state.pending_plan is None and not round_state._bindings
    await adapter._reconcile_expert_binding_on_reuse()
    assert adapter._current_expert_id is None


async def test_cancelling_handoff_waits_for_load_record_then_rolls_back(tmp_path, monkeypatch):
    adapter, round_state, _expert, metadata = await _pending_handoff(tmp_path, monkeypatch)
    loaded, release = asyncio.Event(), asyncio.Event()
    original_load = adapter._instance.load_agent_template

    async def load_then_pause(package):
        record = await original_load(package)
        loaded.set()
        await release.wait()
        return record

    monkeypatch.setattr(adapter._instance, "load_agent_template", load_then_pause)
    task = asyncio.create_task(_confirm(round_state, metadata))
    await asyncio.wait_for(loaded.wait(), 5)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert adapter._current_expert_id is None and adapter._expert_load_record is None
    assert adapter._instance.ability_manager.get("handoff_echo") is None and not _skill_names(adapter)
    assert session_metadata.get_session_metadata(round_state.session_id, cache_bust=True)["expert_id"] == ""
    assert not adapter._instance._load_records
    assert round_state.pending_plan is None


async def test_internal_handoff_rejects_call_without_consumed_confirmation(tmp_path, monkeypatch):
    adapter, round_state, expert, _metadata = await _pending_handoff(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="expert_confirmation_expired"):
        await adapter._handoff_confirmed_expert(round_state, expert)
    assert adapter._current_expert_id is None
