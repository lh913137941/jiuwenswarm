# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""History parsing must not monopolize the loop or overlap within a session."""
import asyncio
import json
import threading

import pytest

from jiuwenswarm.common.e2a.wire_codec import parse_agent_server_wire_chunk, parse_agent_server_wire_unary
from jiuwenswarm.common.schema.agent import AgentRequest, ReqMethod
from jiuwenswarm.server import agent_ws_server as module


class RecordingTransport:
    def __init__(self):
        self.sent = []

    async def send_text(self, data):
        self.sent.append(json.loads(data))


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(module, "history_exists", lambda sid: True)
    monkeypatch.setattr(module, "load_todo_snapshot_for_frontend", lambda sid: [])
    return module.AgentWebSocketServer.__new__(module.AgentWebSocketServer)


def request(session="history-session", stream=False):
    return AgentRequest(
        request_id="history-request", channel_id="desktop", req_method=ReqMethod.HISTORY_GET,
        params={"session_id": session, "page_idx": 1}, is_stream=stream,
    )


def handler(server, stream):
    return server._handle_history_get_stream if stream else server._handle_history_get


def payload(frame, stream=False):
    parse = parse_agent_server_wire_chunk if stream else parse_agent_server_wire_unary
    return parse(frame).payload


async def wait_entered(event):
    assert await asyncio.wait_for(asyncio.to_thread(event.wait, 3), 4)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_stream_can_send_before_history_read_finishes(server, monkeypatch, stream):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def read(sid):
        entered.set()
        release.wait(2)  # A watchdog also lets the old synchronous implementation fail safely.
        finished.set()
        return [{"id": "answer", "role": "assistant", "event_type": "chat.final", "content": "saved answer"}]

    monkeypatch.setattr(module, "load_history_records", read)
    ws, send_lock = RecordingTransport(), asyncio.Lock()
    history = asyncio.create_task(handler(server, stream)(ws, request(stream=stream), send_lock))
    try:
        await wait_entered(entered)
        async with send_lock:
            await module.send_wire_payload(ws, {"event": "stream-progress"})
        assert not finished.is_set(), "history blocked the event loop until its read finished"
        assert ws.sent == [{"event": "stream-progress"}]
    finally:
        release.set()
        await history
    payloads = [payload(frame, stream) for frame in ws.sent[1:]]
    if stream:
        assert payloads[0]["message"]["content"] == "saved answer"
        assert payloads[0]["total_pages"] == 1
        assert payloads[-1]["status"] == "done"
    else:
        assert payloads[0]["messages"][0]["content"] == "saved answer"
        assert payloads[0]["total_pages"] == payloads[0]["page_idx"] == 1


@pytest.mark.asyncio
async def test_cancelled_reader_keeps_same_session_serialized(server, monkeypatch):
    entered, release, next_entered = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def read(sid):
        calls.append(sid)
        if len(calls) == 1:
            entered.set()
            release.wait(3)
        else:
            next_entered.set()
        return []

    monkeypatch.setattr(module, "load_history_records", read)
    first = asyncio.create_task(server._handle_history_get(RecordingTransport(), request(), asyncio.Lock()))
    queued = following = None
    try:
        await wait_entered(entered)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        queued = asyncio.create_task(server._handle_history_get_stream(
            RecordingTransport(), request(" history-session ", stream=True), asyncio.Lock()))
        assert not await asyncio.to_thread(next_entered.wait, 0.1), "cancel released a still-running read"
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        ws = RecordingTransport()
        following = asyncio.create_task(server._handle_history_get(ws, request(), asyncio.Lock()))
        assert not await asyncio.to_thread(next_entered.wait, 0.1)
        release.set()
        await asyncio.wait_for(following, 3)
        assert calls == ["history-session", "history-session"]
        assert payload(ws.sent[0])["messages"] == []
    finally:
        release.set()
        await asyncio.gather(*(t for t in [first, queued, following] if t), return_exceptions=True)


@pytest.mark.asyncio
async def test_different_session_can_read_while_first_is_blocked(server, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def read(sid):
        if sid == "blocked":
            entered.set()
            release.wait(3)
        return []

    monkeypatch.setattr(module, "load_history_records", read)
    first = asyncio.create_task(server._handle_history_get(RecordingTransport(), request("blocked"), asyncio.Lock()))
    try:
        await wait_entered(entered)
        ws = RecordingTransport()
        await asyncio.wait_for(server._handle_history_get(ws, request("other"), asyncio.Lock()), 1)
        assert not first.done()
        assert payload(ws.sent[0])["messages"] == []
    finally:
        release.set()
        await first


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_invalid_history_preserves_error_response(server, stream):
    req = request(stream=stream)
    req.params["page_idx"] = 0
    ws = RecordingTransport()
    await handler(server, stream)(ws, req, asyncio.Lock())
    assert "error" in payload(ws.sent[0], stream)
    if stream:
        assert payload(ws.sent[0], stream)["event_type"] == "chat.error"


@pytest.mark.asyncio
async def test_unexpected_read_error_does_not_block_next_request(server, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("unexpected history failure")

    monkeypatch.setattr(server, "get_conversation_history", fail)
    with pytest.raises(RuntimeError, match="unexpected history failure"):
        await server._handle_history_get(RecordingTransport(), request(), asyncio.Lock())
    monkeypatch.setattr(server, "get_conversation_history", lambda **kw: None)
    ws = RecordingTransport()
    await asyncio.wait_for(server._handle_history_get(ws, request(), asyncio.Lock()), 1)
    assert "error" in payload(ws.sent[0])
