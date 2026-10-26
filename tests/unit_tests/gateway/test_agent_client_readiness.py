# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Late E2A connection ACKs must restore Gateway and cron handshake readiness."""

import asyncio
import json
import logging
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from jiuwenswarm.common.e2a.gateway_normalize import e2a_from_agent_fields
from jiuwenswarm.common.e2a.wire_codec import encode_agent_response_for_wire
from jiuwenswarm.common.schema.agent import AgentResponse
from jiuwenswarm.gateway.channel_manager.web.app_web_handlers import WebHandlersBindParams, _register_web_handlers
from jiuwenswarm.gateway.routing import agent_client


_ACK = {"type": "event", "event": "connection.ack", "payload": {"status": "ready"}}


class _QueuedConnection:
    def __init__(self, first_frame):
        self.first_frame = first_frame
        self.recv_count = 0
        self.incoming = asyncio.Queue()
        self.sent = asyncio.Event()

    async def recv(self):
        self.recv_count += 1
        if self.recv_count == 1:
            if self.first_frame == "timeout":
                # Exercise connect()'s timeout branch without a startup-length sleep.
                raise asyncio.TimeoutError
            if self.first_frame == "wait":
                await asyncio.Event().wait()
            return json.dumps(self.first_frame)
        return json.dumps(await self.incoming.get())

    async def send(self, _data):
        self.sent.set()

    async def close(self):
        pass


async def _connect(monkeypatch, first_frame, transport):
    connection = _QueuedConnection(first_frame)
    client = agent_client.WebSocketAgentServerClient()
    monkeypatch.setattr(agent_client, "resolve_agent_e2a_pipe_path", lambda: "test-pipe" if transport == "pipe" else None)
    if transport == "pipe":
        monkeypatch.setattr(client, "_open_pipe_connection", AsyncMock(return_value=connection))
    else:
        monkeypatch.setattr("websockets.legacy.client.connect", AsyncMock(return_value=connection))
    await client.connect("ws://agent-server/ws")
    return client, connection


async def _roundtrip(client, connection, frames):
    connection.sent.clear()
    request_id = f"readiness-rpc-{uuid.uuid4().hex}"
    envelope = e2a_from_agent_fields(
        request_id=request_id, channel_id="web", session_id="cron-session", params={}, is_stream=False,
    )
    task = asyncio.create_task(client.send_request(envelope))
    try:
        await asyncio.wait_for(connection.sent.wait(), timeout=1)
        for frame in frames:
            await connection.incoming.put(frame)
        await connection.incoming.put(encode_agent_response_for_wire(
            AgentResponse(request_id=request_id, channel_id="web", ok=True, payload={"jobs": []}),
            response_id=request_id,
        ))
        # The response is an ordered barrier: the receiver has processed every earlier frame.
        response = await asyncio.wait_for(task, timeout=1)
        assert response.ok and response.payload == {"jobs": []}
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["pipe", "ws"])
@pytest.mark.parametrize("first_frame", ["timeout", {"type": "event", "event": "connection.heartbeat"}, _ACK])
async def test_ack_restores_readiness_and_cron_handshake(monkeypatch, caplog, transport, first_frame):
    caplog.set_level(logging.WARNING, logger=agent_client.__name__)
    client, connection = await _connect(monkeypatch, first_frame, transport)
    pushes = AsyncMock()
    client.set_server_push_handler(pushes)
    channel = SimpleNamespace(channel_id="web", clients=[], on_connect=Mock(), register_method=Mock(), send=AsyncMock())
    _register_web_handlers(WebHandlersBindParams(channel=channel, agent_client=client))
    on_connect = channel.on_connect.call_args.args[0]
    ws = SimpleNamespace(_jiuwen_initial_sid="cron-session")
    channel.clients.append(ws)
    try:
        if first_frame != _ACK:
            assert not client.server_ready
            await on_connect(ws)
            channel.send.assert_not_awaited()
        else:
            assert client.server_ready
            await on_connect(ws)

        await _roundtrip(client, connection, [_ACK, _ACK])

        assert client.server_ready
        assert "无目标队列" not in caplog.text
        pushes.assert_not_awaited()
        await on_connect(ws)
        channel.send.assert_awaited_once()
        ack = channel.send.await_args.args[0]
        assert ack.event_type.value == "connection.ack"
        assert ack.session_id == "cron-session"
    finally:
        await client.disconnect()
    assert not client.server_ready


@pytest.mark.asyncio
@pytest.mark.parametrize("frame", [
    {"type": "event", "event": "connection.heartbeat"},
    {"type": "res", "event": "connection.ack"},
])
async def test_non_ack_frames_do_not_mark_server_ready(monkeypatch, frame):
    client, connection = await _connect(monkeypatch, "timeout", "pipe")
    try:
        await _roundtrip(client, connection, [frame])
        assert not client.server_ready
    finally:
        await client.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["pipe", "ws"])
@pytest.mark.parametrize("configured,expected", [
    (None, 30.0), ("0.25", 0.25), ("90", 90.0), ("bad", 30.0),
    ("", 30.0), ("0", 30.0), ("-1", 30.0), ("nan", 30.0), ("inf", 30.0),
])
async def test_initial_ack_wait_uses_configured_timeout(monkeypatch, transport, configured, expected):
    if configured is None:
        monkeypatch.delenv("AGENT_CONNECTION_ACK_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("AGENT_CONNECTION_ACK_TIMEOUT_SECONDS", configured)
    observed = []
    original_wait = asyncio.wait_for

    async def wait(awaitable, *, timeout):
        observed.append(timeout)
        return await original_wait(awaitable, timeout=timeout)

    monkeypatch.setattr(agent_client.asyncio, "wait_for", wait)
    client, _ = await _connect(monkeypatch, _ACK, transport)
    try:
        assert observed == [expected]
        assert client.server_ready
    finally:
        await client.disconnect()


class _RecordingClient:
    closed = False

    def __init__(self, sid):
        self._jiuwen_initial_sid = sid
        self.frames = asyncio.Queue()

    async def send(self, data):
        await self.frames.put(json.loads(data))

    async def close(self, **_kwargs):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["pipe", "ws"])
async def test_real_timeout_backfills_existing_clients_without_reconnect(monkeypatch, transport):
    from jiuwenswarm.gateway.channel_manager.web.web_connect import WebChannel, WebChannelConfig
    from jiuwenswarm.gateway.routing.keys import AgentRef, RoutingKey

    monkeypatch.setenv("AGENT_CONNECTION_ACK_TIMEOUT_SECONDS", "0.01")
    client, upstream = await _connect(monkeypatch, "wait", transport)
    channel = WebChannel(WebChannelConfig(enabled=True), SimpleNamespace())
    _register_web_handlers(WebHandlersBindParams(channel=channel, agent_client=client))
    peers = [_RecordingClient("initial-1"), _RecordingClient("initial-2"), _RecordingClient("closed")]
    on_connect = channel._connect_hooks[-1]
    try:
        assert not client.server_ready
        for i, peer in enumerate(peers):
            # Real-session rebinding must not lose the deferred connection ACK.
            key = RoutingKey("user", "web", "default", AgentRef.default(), f"real-{i}")
            await channel.register_ws(peer, key)
            await on_connect(peer)
            assert peer.frames.empty()
        await channel.unregister_ws(peers[-1])
        await peers[-1].close()
        await _roundtrip(client, upstream, [_ACK, _ACK])
        for peer in peers[:2]:
            frame = await asyncio.wait_for(peer.frames.get(), timeout=1)
            assert frame["event"] == "connection.ack"
            assert frame["payload"]["session_id"] == peer._jiuwen_initial_sid
            assert frame["payload"]["protocol_version"] == "1.0"
            await on_connect(peer)
            assert peer.frames.empty()
        assert peers[-1].frames.empty()
        # Upstream reconnection may emit another ACK; acknowledged peers stay quiet.
        await client.disconnect()
        await upstream.incoming.put(_ACK)
        await client.connect("ws://agent-server/ws")
        await _roundtrip(client, upstream, [_ACK])
        assert all(peer.frames.empty() for peer in peers)
    finally:
        await client.disconnect()
        await channel.stop()


@pytest.mark.asyncio
async def test_backfill_isolates_failed_and_closed_clients(monkeypatch):
    client, upstream = await _connect(monkeypatch, "timeout", "pipe")
    peers = [SimpleNamespace(_jiuwen_initial_sid=sid, closed=False) for sid in ("broken", "ok", "closed")]
    peers[-1].closed = True

    async def send(msg):
        if msg.session_id == "broken":
            raise RuntimeError("client disconnected during delivery")

    channel = SimpleNamespace(
        channel_id="web", clients=peers, on_connect=Mock(), register_method=Mock(), send=AsyncMock(side_effect=send),
    )
    _register_web_handlers(WebHandlersBindParams(channel=channel, agent_client=client))
    try:
        await _roundtrip(client, upstream, [_ACK, _ACK])
        assert client.server_ready
        assert [call.args[0].session_id for call in channel.send.await_args_list] == ["broken", "ok"]
        assert not peers[0]._jiuwen_connection_ack_sent
        assert peers[1]._jiuwen_connection_ack_sent
        assert not hasattr(peers[2], "_jiuwen_connection_ack_sent")
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_connect_hook_and_backfill_do_not_duplicate_in_flight_ack(monkeypatch):
    client, upstream = await _connect(monkeypatch, "timeout", "pipe")
    peer = SimpleNamespace(_jiuwen_initial_sid="racing")
    started, release = asyncio.Event(), asyncio.Event()

    async def send(_msg):
        started.set()
        await release.wait()

    channel = SimpleNamespace(
        channel_id="web", clients=[peer], on_connect=Mock(), register_method=Mock(), send=AsyncMock(side_effect=send),
    )
    _register_web_handlers(WebHandlersBindParams(channel=channel, agent_client=client))
    delivery = asyncio.create_task(_roundtrip(client, upstream, [_ACK, _ACK]))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        await channel.on_connect.call_args.args[0](peer)
        release.set()
        await delivery
        channel.send.assert_awaited_once()
    finally:
        release.set()
        await asyncio.gather(delivery, return_exceptions=True)
        await client.disconnect()
