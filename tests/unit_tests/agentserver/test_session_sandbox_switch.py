# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""New and cached session runtimes must match the sandbox switch."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openjiuwen.core.sys_operation import OperationMode, SandboxGatewayConfig, SysOperation, SysOperationCard
from openjiuwen.core.sys_operation.config import ContainerScope, PreDeployLauncherConfig, SandboxIsolationConfig
from openjiuwen.core.sys_operation.sandbox.shell_operation import ShellOperation as SandboxShellOperation
from openjiuwen.harness.tools import BashTool

from jiuwenswarm.agents.harness.common import session_ops_service
from jiuwenswarm.server.runtime.agent_adapter import interface_deep
from jiuwenswarm.server.runtime.agent_adapter.interface_code import JiuwenSwarmCodeAdapter
from jiuwenswarm.server.runtime.agent_adapter.interface_deep import JiuWenSwarmDeepAdapter
from jiuwenswarm.server.runtime.agent_adapter.sysop_builder import create_local_sysop_card


@pytest.fixture
def session_runtime(monkeypatch, tmp_path):
    children = []
    runtime = {"enabled": True}
    endpoint = {}
    monkeypatch.setattr(interface_deep, "get_sandbox_runtime", lambda: runtime)
    monkeypatch.setattr(interface_deep, "get_sandbox_endpoint", lambda: endpoint)
    state_path = tmp_path / "runtime.yaml"
    state_path.write_text("mode: agent\n", encoding="utf-8")
    monkeypatch.setattr(interface_deep, "get_runtime_state_path", lambda _: state_path)
    warmup = AsyncMock(return_value="previous-turn")
    refresh = AsyncMock(return_value="previous-turn")
    monkeypatch.setattr(session_ops_service, "warmup_session_context", warmup)
    monkeypatch.setattr(session_ops_service, "refresh_session_context_if_stale", refresh)
    monkeypatch.setattr(interface_deep, "_history_tail_lagged", lambda *_: False)

    def make_child(adapter_type, sid):
        child = adapter_type()
        child.mark_as_session_scoped(sid)
        child._instance = SimpleNamespace(history=["previous-turn"])
        child._sys_operation = SimpleNamespace(mode=OperationMode.SANDBOX)
        child.start_interaction = AsyncMock()
        child.cleanup = AsyncMock(side_effect=child._release_sys_operations)
        monkeypatch.setattr(child, "_install_file_soft_delete", lambda: None)
        children.append(child)

        def sandbox_card(url, sandbox_type, **_kwargs):
            return SysOperationCard(
                mode=OperationMode.SANDBOX,
                gateway_config=SandboxGatewayConfig(
                    isolation=SandboxIsolationConfig(container_scope=ContainerScope.CUSTOM, custom_id=sid),
                    launcher_config=PreDeployLauncherConfig(base_url=url, sandbox_type=sandbox_type),
                ),
            )

        monkeypatch.setattr(child, "_create_sandbox_sys_operation", sandbox_card)

        async def create_instance(config=None, *, mode="agent", sub_mode=None):
            child._session_instance_config = config
            child._session_instance_mode = mode
            child._session_instance_sub_mode = sub_mode
            child._instance = SimpleNamespace(history=[])
            child._sys_operation = child._create_sys_operation()
            child.bash = BashTool(child._sys_operation, agent_id=sid)

        child.create_instance = AsyncMock(side_effect=create_instance)
        return child

    yield SimpleNamespace(
        runtime=runtime, endpoint=endpoint, state_path=state_path, warmup=warmup, refresh=refresh,
        make_child=make_child, workdir=str(tmp_path),
    )
    for child in children:
        child._release_sys_operations()


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type,mode,sub_mode", [
    (JiuWenSwarmDeepAdapter, "agent", None),
    (JiuwenSwarmCodeAdapter, "code", "plan"),
])
async def test_same_session_next_call_executes_locally(session_runtime, monkeypatch, adapter_type, mode, sub_mode):
    s = session_runtime
    root = adapter_type()
    sid = "existing-session"
    old = s.make_child(adapter_type, sid)
    old._session_instance_config = {"project_dir": s.workdir}
    old._session_instance_mode = mode
    old._session_instance_sub_mode = sub_mode
    root._session_adapters[sid] = old
    new = s.make_child(adapter_type, sid)
    monkeypatch.setattr(root, "_new_session_scoped_adapter", lambda _: new)

    assert await root._get_or_create_session_adapter(sid) is old
    s.runtime["enabled"] = False
    current = await root._get_or_create_session_adapter(sid, warmup_exclude_request_id="next-turn")

    assert current is new
    assert current._sys_operation.mode == OperationMode.LOCAL
    assert list(root._session_adapters) == [sid]
    old.cleanup.assert_awaited_once()
    new.create_instance.assert_awaited_once_with(
        {"project_dir": s.workdir}, mode=mode, sub_mode=sub_mode,
    )
    new.start_interaction.assert_awaited_once_with(session_id=sid)
    s.warmup.assert_awaited_once_with(
        deep_agent=new._instance, session_id=sid, exclude_request_id="next-turn",
    )
    assert new._synced_history_tail_request_id == "previous-turn"
    assert s.state_path.exists()
    result = await current.bash.invoke({
        "command": "echo same-session-local", "shell_type": "cmd" if os.name == "nt" else "sh",
        "workdir": s.workdir,
    })
    assert result.success, result
    assert "same-session-local" in str(result.data)
    assert await root._get_or_create_session_adapter(sid) is current
    assert new.create_instance.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_switch_does_not_dispose_an_inflight_session(session_runtime, monkeypatch, enabled):
    s = session_runtime
    root = JiuWenSwarmDeepAdapter()
    sid = "busy-session"
    old = s.make_child(JiuWenSwarmDeepAdapter, sid)
    if enabled:
        old._sys_operation = SysOperation(create_local_sysop_card())
        s.endpoint.update(url="http://127.0.0.1:8321", type="jiuwenbox")
    old._active_session_ids[sid] = 1
    root._session_adapters[sid] = old
    s.runtime["enabled"] = enabled
    new = s.make_child(JiuWenSwarmDeepAdapter, sid)
    monkeypatch.setattr(root, "_new_session_scoped_adapter", lambda _: new)

    assert await root._get_or_create_session_adapter(sid) is old
    old.cleanup.assert_not_awaited()
    old._active_session_ids.clear()
    assert await root._get_or_create_session_adapter(sid) is new


@pytest.mark.asyncio
async def test_disabled_local_session_and_other_cached_sessions_are_unchanged(session_runtime):
    s = session_runtime
    root = JiuWenSwarmDeepAdapter()
    local = s.make_child(JiuWenSwarmDeepAdapter, "local-session")
    local._sys_operation = SysOperation(create_local_sysop_card())
    other = s.make_child(JiuWenSwarmDeepAdapter, "other-session")
    root._session_adapters.update({"local-session": local, "other-session": other})
    s.runtime["enabled"] = False

    assert await root._get_or_create_session_adapter("local-session") is local
    assert root._session_adapters["other-session"] is other
    local.cleanup.assert_not_awaited()
    other.cleanup.assert_not_awaited()
    s.warmup.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_cleanup_failure_is_retryable_without_returning_stale_tools(session_runtime, monkeypatch, enabled):
    s = session_runtime
    root = JiuWenSwarmDeepAdapter()
    sid = "cleanup-retry-session"
    old = s.make_child(JiuWenSwarmDeepAdapter, sid)
    if enabled:
        old._sys_operation = SysOperation(create_local_sysop_card())
        s.endpoint.update(url="http://127.0.0.1:8321", type="jiuwenbox")
    root._session_adapters[sid] = old
    s.runtime["enabled"] = enabled
    old.cleanup.side_effect = RuntimeError("cleanup failed")
    new = s.make_child(JiuWenSwarmDeepAdapter, sid)
    monkeypatch.setattr(root, "_new_session_scoped_adapter", lambda _: new)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await root._get_or_create_session_adapter(sid)
    assert root._session_adapters[sid] is old
    new.create_instance.assert_not_awaited()
    old.cleanup.side_effect = None
    assert await root._get_or_create_session_adapter(sid) is new


@pytest.mark.asyncio
async def test_disable_rebuild_failure_never_returns_stale_tools(session_runtime, monkeypatch):
    s = session_runtime
    root = JiuWenSwarmDeepAdapter()
    sid = "retry-session"
    old = s.make_child(JiuWenSwarmDeepAdapter, sid)
    root._session_adapters[sid] = old
    s.runtime["enabled"] = False
    new = s.make_child(JiuWenSwarmDeepAdapter, sid)
    new.create_instance.side_effect = RuntimeError("local setup failed")
    monkeypatch.setattr(root, "_new_session_scoped_adapter", lambda _: new)

    with pytest.raises(RuntimeError, match="local setup failed"):
        await root._get_or_create_session_adapter(sid)
    assert sid not in root._session_adapters
    assert not root._session_adapter_locks[sid].locked()
    assert s.state_path.exists()
    new.cleanup.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_parallel_next_lookups_rebuild_once(session_runtime, monkeypatch, enabled):
    s = session_runtime
    root = JiuWenSwarmDeepAdapter()
    sid = "parallel-session"
    old = s.make_child(JiuWenSwarmDeepAdapter, sid)
    if enabled:
        old._sys_operation = SysOperation(create_local_sysop_card())
        s.endpoint.update(url="http://127.0.0.1:8321", type="jiuwenbox")
    root._session_adapters[sid] = old
    s.runtime["enabled"] = enabled
    new = s.make_child(JiuWenSwarmDeepAdapter, sid)
    monkeypatch.setattr(root, "_new_session_scoped_adapter", lambda _: new)

    results = await asyncio.gather(*(root._get_or_create_session_adapter(sid) for _ in range(3)))
    assert all(child is new for child in results)
    old.cleanup.assert_awaited_once()
    new.create_instance.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type,mode,sub_mode", [
    (JiuWenSwarmDeepAdapter, "agent", None),
    (JiuwenSwarmCodeAdapter, "code", "plan"),
])
async def test_same_session_switches_on_off_on(session_runtime, monkeypatch, adapter_type, mode, sub_mode):
    s = session_runtime
    root = adapter_type()
    sid = "roundtrip-session"
    s.endpoint.update(url="http://127.0.0.1:8321", type="jiuwenbox")
    old = s.make_child(adapter_type, sid)
    old._session_instance_mode, old._session_instance_sub_mode = mode, sub_mode
    root._session_adapters[sid] = old
    sandbox_calls = []

    async def sandbox_invoke(operation, method, **kwargs):
        # Keep the real SysOperation/BashTool dispatch, replacing only remote IO.
        assert operation.mode == OperationMode.SANDBOX
        sandbox_calls.append((method, kwargs["command"]))
        return {"code": 0, "message": "OK", "data": {
            "command": kwargs["command"], "exit_code": 0, "stdout": "sandbox-route", "stderr": "",
        }}

    monkeypatch.setattr(SandboxShellOperation, "invoke", sandbox_invoke)
    created = []

    def make_child(session_id):
        child = s.make_child(adapter_type, session_id)
        created.append(child)
        return child

    monkeypatch.setattr(root, "_new_session_scoped_adapter", make_child)
    for enabled in (False, True, False, True):
        previous = root._session_adapters[sid]
        s.runtime["enabled"] = enabled
        current = await root._get_or_create_session_adapter(sid)
        assert current is not previous
        assert current._sys_operation.mode == (OperationMode.SANDBOX if enabled else OperationMode.LOCAL)
        previous.cleanup.assert_awaited_once()
        result = await current.bash.invoke({
            "command": "echo local-route", "shell_type": "cmd" if os.name == "nt" else "sh",
            "workdir": s.workdir,
        })
        assert result.success, result
        assert ("sandbox-route" if enabled else "local-route") in str(result.data)
        assert await root._get_or_create_session_adapter(sid) is current
        current.create_instance.assert_awaited_once_with(None, mode=mode, sub_mode=sub_mode)
        assert current._synced_history_tail_request_id == "previous-turn"
        assert list(root._session_adapters) == [sid]

    assert len(created) == 4
    assert sandbox_calls == [("execute_cmd", "echo local-route")] * 2
    assert s.warmup.await_count == 4
    assert s.state_path.exists()


@pytest.mark.asyncio
async def test_enable_without_endpoint_rejects_local_runtime(session_runtime, monkeypatch):
    s = session_runtime
    root = JiuWenSwarmDeepAdapter()
    sid = "missing-endpoint-session"
    old = s.make_child(JiuWenSwarmDeepAdapter, sid)
    old._sys_operation = SysOperation(create_local_sysop_card())
    root._session_adapters[sid] = old
    new = s.make_child(JiuWenSwarmDeepAdapter, sid)
    monkeypatch.setattr(root, "_new_session_scoped_adapter", lambda _: new)

    with pytest.raises(RuntimeError, match="执行器模式未生效.*期望 sandbox / 实际 local"):
        await root._get_or_create_session_adapter(sid)
    assert sid not in root._session_adapters
    new.cleanup.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", [JiuWenSwarmDeepAdapter, JiuwenSwarmCodeAdapter])
async def test_new_session_without_sandbox_endpoint_never_starts_locally(session_runtime, monkeypatch, adapter_type):
    s = session_runtime
    root = adapter_type()
    sid = "new-session-without-endpoint"
    created = []

    def make_child(session_id):
        child = s.make_child(adapter_type, session_id)
        created.append(child)
        return child

    monkeypatch.setattr(root, "_new_session_scoped_adapter", make_child)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="执行器模式未生效.*期望 sandbox / 实际 local"):
            await root._get_or_create_session_adapter(sid)
        assert sid not in root._session_adapters
        created[-1].cleanup.assert_awaited_once()
        created[-1].start_interaction.assert_not_awaited()
        s.warmup.assert_not_awaited()
        assert not root._session_adapter_locks[sid].locked()

    s.endpoint.update(url="http://127.0.0.1:8321", type="jiuwenbox")
    current = await root._get_or_create_session_adapter(sid)
    assert current._sys_operation.mode == OperationMode.SANDBOX
    current.start_interaction.assert_awaited_once_with(session_id=sid)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_new_session_validates_executor_in_both_directions(session_runtime, monkeypatch, enabled):
    s = session_runtime
    root = JiuWenSwarmDeepAdapter()
    sid = "new-mismatched-session"
    s.runtime["enabled"] = enabled
    child = s.make_child(JiuWenSwarmDeepAdapter, sid)

    async def create_wrong_mode(*_args, **_kwargs):
        child._sys_operation = SimpleNamespace(mode=OperationMode.LOCAL if enabled else OperationMode.SANDBOX)

    child.create_instance.side_effect = create_wrong_mode
    monkeypatch.setattr(root, "_new_session_scoped_adapter", lambda _: child)
    expected = "sandbox / 实际 local" if enabled else "local / 实际 sandbox"
    with pytest.raises(RuntimeError, match=f"执行器模式未生效.*期望 {expected}.*请检查"):
        await root._get_or_create_session_adapter(sid)
    child.cleanup.assert_awaited_once()
    child.start_interaction.assert_not_awaited()
    assert sid not in root._session_adapters
