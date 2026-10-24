"""Regression tests for restart-safe session context warmup."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from openjiuwen.core.sys_operation import OperationMode

from jiuwenswarm.agents.harness.common import session_ops_service


def _assert_checkpointer_miss_then_history_restore(context_engine):
    """Warmup prefers checkpointer; empty snapshot falls back to history.jsonl."""
    assert context_engine.create_context.await_count == 2
    first, second = context_engine.create_context.await_args_list
    assert "history_messages" not in first.kwargs
    assert "history_messages" in second.kwargs
    context_engine.clear_context.assert_awaited_once()
    context_engine.save_contexts.assert_not_awaited()


@pytest.mark.asyncio
async def test_warmup_excludes_the_current_request_from_restored_history(monkeypatch):
    """A cold-start turn must not restore its just-persisted user message."""
    context_engine = SimpleNamespace(
        get_context=MagicMock(return_value=None),
        create_context=AsyncMock(),
        clear_context=AsyncMock(),
        save_contexts=AsyncMock(),
    )
    deep_agent = SimpleNamespace(
        react_agent=SimpleNamespace(context_engine=context_engine, _config=None),
        card=None,
    )
    records = [
        {"request_id": "previous", "role": "user", "content": "earlier question"},
        {"request_id": "current", "role": "user", "content": "current question"},
    ]
    restored: list[dict[str, object]] = []

    monkeypatch.setattr(session_ops_service, "history_exists", lambda _session_id: True)
    monkeypatch.setattr(session_ops_service, "load_history_records", lambda _session_id: records)
    monkeypatch.setattr(session_ops_service, "resolve_live_agent_session", lambda *_args: object())

    def build_context_messages(history_records, **_kwargs):
        restored.extend(history_records)
        return [object()], 0

    monkeypatch.setattr(
        session_ops_service,
        "_build_context_messages_from_history",
        build_context_messages,
    )

    result = await session_ops_service.warmup_session_context(
        deep_agent=deep_agent,
        session_id="session-1",
        exclude_request_id="current",
    )

    assert result == "previous"  # 历史重灌路径返回所恢复记录的尾
    assert [record["request_id"] for record in restored] == ["previous"]
    _assert_checkpointer_miss_then_history_restore(context_engine)


@pytest.mark.asyncio
async def test_warmup_keeps_all_history_without_a_current_request(monkeypatch):
    """Non-chat callers retain the original full-history recovery behavior."""
    context_engine = SimpleNamespace(
        get_context=MagicMock(return_value=None),
        create_context=AsyncMock(),
        clear_context=AsyncMock(),
        save_contexts=AsyncMock(),
    )
    deep_agent = SimpleNamespace(
        react_agent=SimpleNamespace(context_engine=context_engine, _config=None),
        card=None,
    )
    records = [
        {"request_id": "previous", "role": "user", "content": "earlier question"},
        {"request_id": "current", "role": "user", "content": "current question"},
    ]
    restored: list[dict[str, object]] = []

    monkeypatch.setattr(session_ops_service, "history_exists", lambda _session_id: True)
    monkeypatch.setattr(session_ops_service, "load_history_records", lambda _session_id: records)
    monkeypatch.setattr(session_ops_service, "resolve_live_agent_session", lambda *_args: object())

    def build_context_messages(history_records, **_kwargs):
        restored.extend(history_records)
        return [object()], 0

    monkeypatch.setattr(
        session_ops_service,
        "_build_context_messages_from_history",
        build_context_messages,
    )

    result = await session_ops_service.warmup_session_context(
        deep_agent=deep_agent,
        session_id="session-1",
    )

    assert result == "current"  # 不排除在途请求时尾为最后一条记录
    assert [record["request_id"] for record in restored] == ["previous", "current"]
    _assert_checkpointer_miss_then_history_restore(context_engine)


def _warmup_checkpointer_hit_stubs(monkeypatch, tmp_path, sid, snapshot_tail):
    """快照命中路径的公共桩：metadata 切到 tmp 目录，checkpointer 恢复成功。"""
    import json as _json

    from jiuwenswarm.server.runtime.session import session_metadata

    monkeypatch.setattr(session_ops_service, "get_agent_sessions_dir", lambda: tmp_path)
    monkeypatch.setattr(session_metadata, "get_agent_sessions_dir", lambda: tmp_path)
    with session_metadata._CACHE_LOCK:
        session_metadata._METADATA_CACHE.clear()

    sdir = tmp_path / sid
    sdir.mkdir(parents=True)
    (sdir / "metadata.json").write_text(
        _json.dumps({"session_id": sid}), encoding="utf-8"
    )
    if snapshot_tail:
        session_metadata.set_context_snapshot_tail(sid, snapshot_tail)

    context_engine = SimpleNamespace(
        get_context=MagicMock(return_value=None),
        create_context=AsyncMock(),
        clear_context=AsyncMock(),
        save_contexts=AsyncMock(),
    )
    deep_agent = SimpleNamespace(
        react_agent=SimpleNamespace(context_engine=context_engine, _config=None),
        card=None,
    )
    monkeypatch.setattr(session_ops_service, "history_exists", lambda _sid: True)
    monkeypatch.setattr(
        session_ops_service, "resolve_live_agent_session", lambda *_a: object()
    )
    monkeypatch.setattr(
        session_ops_service,
        "_try_restore_from_checkpointer",
        AsyncMock(return_value=True),
    )
    # 磁盘尾已推进到快照之后（如团队期记录）
    monkeypatch.setattr(
        session_ops_service,
        "load_history_tail_request_id",
        lambda _sid, exclude_request_id=None: "team-9",
    )
    return deep_agent, context_engine


@pytest.mark.asyncio
async def test_warmup_via_checkpointer_returns_snapshot_tail(monkeypatch, tmp_path):
    """快照恢复返回快照尾指纹而非磁盘尾——团队期增量才能被 refresh 补回。"""
    sid = "desktop_snap_aaaa11111111"
    deep_agent, _context_engine = _warmup_checkpointer_hit_stubs(
        monkeypatch, tmp_path, sid, snapshot_tail="pc-1"
    )

    result = await session_ops_service.warmup_session_context(
        deep_agent=deep_agent, session_id=sid, exclude_request_id="current",
    )

    assert result == "pc-1"  # 快照尾，不是磁盘尾 team-9


@pytest.mark.asyncio
async def test_warmup_via_checkpointer_legacy_falls_back_to_disk_tail(
    monkeypatch, tmp_path
):
    """老会话无快照尾指纹 → 回退磁盘尾（旧行为，不更差）。"""
    sid = "desktop_snap_bbbb22222222"
    deep_agent, _ = _warmup_checkpointer_hit_stubs(
        monkeypatch, tmp_path, sid, snapshot_tail=None
    )

    result = await session_ops_service.warmup_session_context(
        deep_agent=deep_agent, session_id=sid, exclude_request_id="current",
    )

    assert result == "team-9"


def test_history_tail_skips_the_in_flight_request():
    records = [
        {"request_id": "pc-1", "role": "user"},
        {"request_id": "phone-2", "role": "user"},
        {"request_id": "current", "role": "user"},
    ]

    assert session_ops_service.history_tail_request_id(records) == "current"
    assert (
        session_ops_service.history_tail_request_id(
            records, exclude_request_id="current"
        )
        == "phone-2"
    )


def test_slice_history_after_fingerprint_returns_delta_or_none():
    records = [
        {"request_id": "pc-1", "role": "user"},
        {"request_id": "phone-2", "role": "user"},
        {"request_id": "current", "role": "user"},
    ]
    assert [
        record["request_id"]
        for record in session_ops_service._slice_history_after_fingerprint(records, "pc-1")
    ] == ["phone-2", "current"]
    assert session_ops_service._slice_history_after_fingerprint(records, "missing") is None
    assert session_ops_service._slice_history_after_fingerprint(records, None) is None


def _refresh_agent(context_engine):
    return SimpleNamespace(
        react_agent=SimpleNamespace(context_engine=context_engine, _config=None),
        card=None,
        save_state=MagicMock(),
    )


def _live_session():
    session = MagicMock()
    session.update_state = MagicMock()
    session.commit = AsyncMock()
    session.post_run = AsyncMock()
    return session


def _patch_refresh_history(monkeypatch, records, tail):
    monkeypatch.setattr(session_ops_service, "history_exists", lambda _session_id: True)
    monkeypatch.setattr(
        session_ops_service,
        "load_history_tail_request_id",
        lambda _session_id, exclude_request_id=None: tail,
    )
    loaded: list[str] = []

    def load_history_records(_session_id):
        loaded.append(_session_id)
        return records

    monkeypatch.setattr(session_ops_service, "load_history_records", load_history_records)
    return loaded


@pytest.mark.asyncio
async def test_refresh_skips_rebuild_when_disk_tail_matches_fingerprint(monkeypatch):
    context_engine = SimpleNamespace(
        get_context=MagicMock(return_value=object()),
        clear_context=AsyncMock(),
        create_context=AsyncMock(),
    )
    records = [
        {"request_id": "phone-2", "role": "user", "content": "phone turn"},
        {"request_id": "current", "role": "user", "content": "需要"},
    ]
    loaded = _patch_refresh_history(monkeypatch, records, "phone-2")

    tail = await session_ops_service.refresh_session_context_if_stale(
        deep_agent=_refresh_agent(context_engine),
        session_id="session-1",
        exclude_request_id="current",
        synced_tail_request_id="phone-2",
    )

    assert tail == "phone-2"
    assert loaded == []
    context_engine.clear_context.assert_not_awaited()
    context_engine.create_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_appends_when_peer_channel_appended_history(monkeypatch):
    live_ctx = SimpleNamespace(add_messages=AsyncMock())
    context_engine = SimpleNamespace(
        get_context=MagicMock(return_value=live_ctx),
        clear_context=AsyncMock(),
        create_context=AsyncMock(),
        save_contexts=AsyncMock(),
    )
    records = [
        {"request_id": "pc-1", "role": "user", "content": "赛里木湖"},
        {"request_id": "phone-2", "role": "user", "content": "什么时候去"},
        {"request_id": "current", "role": "user", "content": "需要"},
    ]
    restored: list[dict[str, object]] = []
    live_session = _live_session()
    appended = [object()]

    _patch_refresh_history(monkeypatch, records, "phone-2")
    monkeypatch.setattr(
        session_ops_service, "resolve_live_agent_session", lambda *_args: live_session
    )

    def build_context_messages(history_records, **_kwargs):
        restored.extend(history_records)
        return list(appended), 0

    monkeypatch.setattr(
        session_ops_service,
        "_build_context_messages_from_history",
        build_context_messages,
    )

    deep_agent = _refresh_agent(context_engine)
    tail = await session_ops_service.refresh_session_context_if_stale(
        deep_agent=deep_agent,
        session_id="session-1",
        exclude_request_id="current",
        synced_tail_request_id="pc-1",
    )

    assert tail == "phone-2"
    context_engine.clear_context.assert_not_awaited()
    context_engine.create_context.assert_not_awaited()
    live_ctx.add_messages.assert_awaited_once_with(appended[0])
    context_engine.save_contexts.assert_awaited_once_with(live_session)
    live_session.commit.assert_not_awaited()
    live_session.post_run.assert_not_awaited()
    live_session.update_state.assert_not_called()
    deep_agent.save_state.assert_not_called()
    assert [record["request_id"] for record in restored] == ["phone-2"]


@pytest.mark.asyncio
async def test_refresh_without_memory_context_uses_warmup(monkeypatch):
    context_engine = SimpleNamespace(
        get_context=MagicMock(return_value=None),
        clear_context=AsyncMock(),
        create_context=AsyncMock(),
        save_contexts=AsyncMock(),
    )
    records = [
        {"request_id": "phone-2", "role": "user", "content": "phone turn"},
        {"request_id": "current", "role": "user", "content": "需要"},
    ]
    restored: list[dict[str, object]] = []

    loaded = _patch_refresh_history(monkeypatch, records, "phone-2")
    monkeypatch.setattr(session_ops_service, "resolve_live_agent_session", lambda *_args: object())

    def build_context_messages(history_records, **_kwargs):
        restored.extend(history_records)
        return [object()], 0

    monkeypatch.setattr(
        session_ops_service,
        "_build_context_messages_from_history",
        build_context_messages,
    )

    tail = await session_ops_service.refresh_session_context_if_stale(
        deep_agent=_refresh_agent(context_engine),
        session_id="session-1",
        exclude_request_id="current",
        synced_tail_request_id=None,
    )

    assert tail == "phone-2"
    assert loaded == ["session-1"]
    _assert_checkpointer_miss_then_history_restore(context_engine)
    assert [record["request_id"] for record in restored] == ["phone-2"]


@pytest.mark.asyncio
async def test_refresh_persists_appended_context_so_init_context_cannot_reload_stale(
    monkeypatch,
):
    """Live Session.context must hold the appended window, not the pre-refresh snapshot."""
    appended_messages = [object()]
    live_ctx = SimpleNamespace(add_messages=AsyncMock())
    session_state = {
        "context": {"default_context_id": {"messages": ["stale-pc"]}},
        "plan_mode": "keep-me",
    }

    class LiveSession:
        def __init__(self):
            self.commit = AsyncMock()
            self.post_run = AsyncMock()

        def update_state(self, data):
            session_state.update(data)

        def get_state(self, key=None):
            if key is None:
                return dict(session_state)
            return session_state.get(key)

    live_session = LiveSession()

    async def save_contexts(sess, context_ids=None):
        sess.update_state(
            {"context": {"default_context_id": {"messages": ["stale-pc"] + appended_messages}}}
        )

    context_engine = SimpleNamespace(
        get_context=MagicMock(return_value=live_ctx),
        clear_context=AsyncMock(),
        create_context=AsyncMock(),
        save_contexts=AsyncMock(side_effect=save_contexts),
    )
    records = [
        {"request_id": "pc-1", "role": "user", "content": "赛里木湖"},
        {"request_id": "phone-2", "role": "user", "content": "什么时候去"},
        {"request_id": "current", "role": "user", "content": "需要"},
    ]

    _patch_refresh_history(monkeypatch, records, "phone-2")
    monkeypatch.setattr(
        session_ops_service, "resolve_live_agent_session", lambda *_args: live_session
    )
    monkeypatch.setattr(
        session_ops_service,
        "_build_context_messages_from_history",
        lambda _records, **_kwargs: (appended_messages, 0),
    )

    tail = await session_ops_service.refresh_session_context_if_stale(
        deep_agent=_refresh_agent(context_engine),
        session_id="session-1",
        exclude_request_id="current",
        synced_tail_request_id="pc-1",
    )

    assert tail == "phone-2"
    live_ctx.add_messages.assert_awaited_once_with(appended_messages[0])
    context_engine.clear_context.assert_not_awaited()
    context_engine.create_context.assert_not_awaited()
    live_session.commit.assert_not_awaited()
    live_session.post_run.assert_not_awaited()
    loaded = live_session.get_state("context")
    assert loaded["default_context_id"]["messages"] == ["stale-pc"] + appended_messages
    assert session_state["plan_mode"] == "keep-me"


@pytest.mark.asyncio
async def test_refresh_keeps_old_fingerprint_when_save_contexts_fails(monkeypatch):
    live_ctx = SimpleNamespace(add_messages=AsyncMock())
    context_engine = SimpleNamespace(
        get_context=MagicMock(return_value=live_ctx),
        clear_context=AsyncMock(),
        create_context=AsyncMock(),
        save_contexts=AsyncMock(side_effect=OSError("disk full")),
    )
    records = [
        {"request_id": "pc-1", "role": "user", "content": "赛里木湖"},
        {"request_id": "phone-2", "role": "user", "content": "什么时候去"},
        {"request_id": "current", "role": "user", "content": "需要"},
    ]
    live_session = _live_session()

    _patch_refresh_history(monkeypatch, records, "phone-2")
    monkeypatch.setattr(
        session_ops_service, "resolve_live_agent_session", lambda *_args: live_session
    )
    monkeypatch.setattr(
        session_ops_service,
        "_build_context_messages_from_history",
        lambda _records, **_kwargs: ([object()], 0),
    )

    tail = await session_ops_service.refresh_session_context_if_stale(
        deep_agent=_refresh_agent(context_engine),
        session_id="session-1",
        exclude_request_id="current",
        synced_tail_request_id="pc-1",
    )

    assert tail == "pc-1"
    live_ctx.add_messages.assert_awaited_once()
    context_engine.clear_context.assert_not_awaited()
    context_engine.create_context.assert_not_awaited()
    context_engine.save_contexts.assert_awaited_once_with(live_session)
    live_session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_rebuilds_context_when_fingerprint_missing_from_history(monkeypatch):
    """指纹失效（rewind/历史轮换截掉锚点）不再静默停更：清窗全量重建并推进到磁盘尾。

    旧契约是"保留旧指纹、本轮不动"——之后每轮都指纹失效，上下文永久停更；
    新契约改为 fallback 重建。
    """
    live_ctx = SimpleNamespace(add_messages=AsyncMock())
    context_engine = SimpleNamespace(
        get_context=MagicMock(return_value=live_ctx),
        clear_context=AsyncMock(),
        create_context=AsyncMock(),
        save_contexts=AsyncMock(),
    )
    records = [
        {"request_id": "phone-2", "role": "user", "content": "什么时候去"},
        {"request_id": "current", "role": "user", "content": "需要"},
    ]
    live_session = _live_session()

    loaded = _patch_refresh_history(monkeypatch, records, "phone-2")
    monkeypatch.setattr(
        session_ops_service, "resolve_live_agent_session", lambda *_args: live_session
    )

    tail = await session_ops_service.refresh_session_context_if_stale(
        deep_agent=_refresh_agent(context_engine),
        session_id="session-1",
        exclude_request_id="current",
        synced_tail_request_id="pc-1",
    )

    # 指纹推进到磁盘尾；旧窗清空后经 create_context(history_messages=...) 全量重建
    assert tail == "phone-2"
    assert loaded == ["session-1"]
    live_ctx.add_messages.assert_not_awaited()
    context_engine.clear_context.assert_awaited_once()
    context_engine.create_context.assert_awaited_once()
    assert "history_messages" in context_engine.create_context.await_args.kwargs
    context_engine.save_contexts.assert_awaited_once()


@pytest.mark.asyncio
async def test_refresh_keeps_old_fingerprint_when_add_messages_fails(monkeypatch):
    live_ctx = SimpleNamespace(add_messages=AsyncMock(side_effect=RuntimeError("boom")))
    context_engine = SimpleNamespace(
        get_context=MagicMock(return_value=live_ctx),
        clear_context=AsyncMock(),
        create_context=AsyncMock(),
        save_contexts=AsyncMock(),
    )
    records = [
        {"request_id": "pc-1", "role": "user", "content": "赛里木湖"},
        {"request_id": "phone-2", "role": "user", "content": "什么时候去"},
        {"request_id": "current", "role": "user", "content": "需要"},
    ]
    live_session = _live_session()

    _patch_refresh_history(monkeypatch, records, "phone-2")
    monkeypatch.setattr(
        session_ops_service, "resolve_live_agent_session", lambda *_args: live_session
    )
    monkeypatch.setattr(
        session_ops_service,
        "_build_context_messages_from_history",
        lambda _records, **_kwargs: ([object()], 0),
    )

    tail = await session_ops_service.refresh_session_context_if_stale(
        deep_agent=_refresh_agent(context_engine),
        session_id="session-1",
        exclude_request_id="current",
        synced_tail_request_id="pc-1",
    )

    assert tail == "pc-1"
    live_ctx.add_messages.assert_awaited_once()
    context_engine.clear_context.assert_not_awaited()
    context_engine.create_context.assert_not_awaited()
    context_engine.save_contexts.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_or_create_reuses_adapter_and_refreshes_context(monkeypatch):
    from jiuwenswarm.server.runtime.agent_adapter.interface_deep import (
        JiuWenSwarmDeepAdapter,
    )

    root = JiuWenSwarmDeepAdapter()
    child = JiuWenSwarmDeepAdapter()
    child.mark_as_session_scoped("sess-1")
    child._instance = object()
    child._synced_history_tail_request_id = "pc-1"
    root._session_adapters["sess-1"] = child

    monkeypatch.setattr(root, "_reload_session_adapter_if_stale", AsyncMock())
    monkeypatch.setattr(root, "_touch_session_adapter", lambda *_args, **_kwargs: None)

    captured: dict[str, object] = {}

    async def fake_refresh(**kwargs):
        captured.update(kwargs)
        return "phone-2"

    monkeypatch.setattr(
        session_ops_service,
        "refresh_session_context_if_stale",
        fake_refresh,
    )

    got = await root._get_or_create_session_adapter(
        "sess-1",
        warmup_exclude_request_id="current",
    )

    assert got is child
    assert child._synced_history_tail_request_id == "phone-2"
    assert captured["session_id"] == "sess-1"
    assert captured["exclude_request_id"] == "current"
    assert captured["synced_tail_request_id"] == "pc-1"
    assert captured["deep_agent"] is child._instance


@pytest.mark.asyncio
async def test_get_or_create_new_adapter_refreshes_after_warmup(monkeypatch):
    """新建 adapter：warmup 命中旧快照（返回快照尾）后必须立刻补 refresh。

    回归场景：普通→团队→普通，卸团后首问新建 adapter，warmup 只恢复到
    装团前快照；若不当轮补增量，轮末指纹推进会把团队期永久跳过。
    """
    from jiuwenswarm.server.runtime.agent_adapter.interface_deep import (
        JiuWenSwarmDeepAdapter,
    )
    from jiuwenswarm.server.runtime.session import session_history

    monkeypatch.setattr(
        "jiuwenswarm.server.runtime.agent_adapter.interface_deep.get_sandbox_runtime",
        lambda: {"enabled": False},
    )

    root = JiuWenSwarmDeepAdapter()
    monkeypatch.setattr(root, "_reload_session_adapter_if_stale", AsyncMock())
    monkeypatch.setattr(root, "_touch_session_adapter", lambda *_a, **_k: None)

    created: dict[str, object] = {}
    real_new = root._new_session_scoped_adapter

    def _new_adapter(sid):
        child = real_new(sid)
        child.create_instance = AsyncMock()
        child.start_interaction = AsyncMock()
        child._instance = object()
        child._sys_operation = SimpleNamespace(mode=OperationMode.LOCAL)
        created["child"] = child
        return child

    monkeypatch.setattr(root, "_new_session_scoped_adapter", _new_adapter)
    monkeypatch.setattr(
        session_ops_service,
        "warmup_session_context",
        AsyncMock(return_value="snap-tail-before-team"),
    )
    refresh_calls: dict[str, object] = {}

    async def fake_refresh(**kwargs):
        refresh_calls.update(kwargs)
        return "disk-tail-after-team"

    monkeypatch.setattr(
        session_ops_service, "refresh_session_context_if_stale", fake_refresh
    )
    monkeypatch.setattr(
        session_history,
        "load_history_tail_request_id",
        lambda _sid, exclude_request_id=None: "disk-tail-after-team",
    )

    got = await root._get_or_create_session_adapter(
        "sess-new", warmup_exclude_request_id="current"
    )

    child = created["child"]
    assert got is child
    # warmup 先跑（恢复快照），refresh 紧随其后、带着 warmup 自报的覆盖尾
    assert refresh_calls["synced_tail_request_id"] == "snap-tail-before-team"
    assert child._synced_history_tail_request_id == "disk-tail-after-team"
    # refresh 已补齐到磁盘尾 → 不落后 → 轮末可正常推进指纹
    assert child._history_sync_lagged is False


@pytest.mark.asyncio
async def test_get_or_create_marks_lagged_when_refresh_cannot_catch_up(monkeypatch):
    """refresh 失败（保持旧指纹）→ 标记落后，轮末不得推进指纹。"""
    from jiuwenswarm.server.runtime.agent_adapter.interface_deep import (
        JiuWenSwarmDeepAdapter,
    )
    from jiuwenswarm.server.runtime.session import session_history

    monkeypatch.setattr(
        "jiuwenswarm.server.runtime.agent_adapter.interface_deep.get_sandbox_runtime",
        lambda: {"enabled": False},
    )

    root = JiuWenSwarmDeepAdapter()
    monkeypatch.setattr(root, "_reload_session_adapter_if_stale", AsyncMock())
    monkeypatch.setattr(root, "_touch_session_adapter", lambda *_a, **_k: None)

    child_holder: dict[str, object] = {}
    real_new = root._new_session_scoped_adapter

    def _new_adapter(sid):
        child = real_new(sid)
        child.create_instance = AsyncMock()
        child.start_interaction = AsyncMock()
        child._instance = object()
        child._sys_operation = SimpleNamespace(mode=OperationMode.LOCAL)
        child_holder["child"] = child
        return child

    monkeypatch.setattr(root, "_new_session_scoped_adapter", _new_adapter)
    monkeypatch.setattr(
        session_ops_service,
        "warmup_session_context",
        AsyncMock(return_value="snap-tail-before-team"),
    )
    monkeypatch.setattr(
        session_ops_service,
        "refresh_session_context_if_stale",
        AsyncMock(return_value="snap-tail-before-team"),  # 失败：保持旧指纹
    )
    monkeypatch.setattr(
        session_history,
        "load_history_tail_request_id",
        lambda _sid, exclude_request_id=None: "disk-tail-after-team",
    )

    await root._get_or_create_session_adapter("sess-lag", warmup_exclude_request_id="c")

    child = child_holder["child"]
    assert child._synced_history_tail_request_id == "snap-tail-before-team"
    assert child._history_sync_lagged is True
    # 轮末推进被拦截，旧指纹保留（下轮 refresh 继续补缺口）
    child._mark_history_tail_synced("some-new-rid")
    assert child._synced_history_tail_request_id == "snap-tail-before-team"


def test_mark_history_tail_synced_advances_when_not_lagged():
    """未落后时轮末正常推进（存量行为不变）。"""
    from jiuwenswarm.server.runtime.agent_adapter.interface_deep import (
        JiuWenSwarmDeepAdapter,
    )

    adapter = JiuWenSwarmDeepAdapter()
    adapter.mark_as_session_scoped("sess-ok")
    adapter._synced_history_tail_request_id = "r1"
    adapter._mark_history_tail_synced("r2")
    assert adapter._synced_history_tail_request_id == "r2"
    # 空 rid 不推进
    adapter._mark_history_tail_synced("")
    assert adapter._synced_history_tail_request_id == "r2"
