"""跨端产物：桌面续轮不带过期 taskId，工具刷新用本轮请求。"""

from types import SimpleNamespace

import pytest

from jiuwenswarm.agents.harness.common.tools.send_file_to_user import SendFileToolkit
from jiuwenswarm.common.schema.agent import AgentRequest
from jiuwenswarm.server.agent_ws_server import _inject_session_xiaoyi_routing
from jiuwenswarm.server.runtime.agent_adapter.interface_deep import (
    JiuWenSwarmDeepAdapter,
    _CRON_TOOL_BOUND,
    _CRON_TOOL_CHANNEL_ID,
    _CRON_TOOL_METADATA,
)
from jiuwenswarm.server.runtime.session import session_metadata as session_metadata_module


def test_desktop_inject_skips_finished_xiaoyi_task_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        session_metadata_module,
        "get_session_metadata",
        lambda session_id, cache_bust=True: {
            "xiaoyi_session_id": "phone-sess",
            "xiaoyi_task_id": "stale-task",
            "xiaoyi_conversation_id": "conv-1",
            "xiaoyi_root_session_id": "root-1",
        },
    )
    request = AgentRequest(
        request_id="pc-1",
        channel_id="desktop",
        session_id="desktop-session",
        metadata={},
    )
    _inject_session_xiaoyi_routing(request)
    assert request.metadata == {
        "xiaoyi_session_id": "phone-sess",
        "xiaoyi_conversation_id": "conv-1",
        "xiaoyi_root_session_id": "root-1",
        # 注入侧同时打路由标记：计费侧据此判定这些字段只用于回发寻址（计费归桌面主进程）
        "xiaoyi_routing_injected": True,
    }


def test_phone_request_with_session_id_is_not_rewritten(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        session_metadata_module,
        "get_session_metadata",
        lambda session_id, cache_bust=True: {
            "xiaoyi_task_id": "stale-task",
        },
    )
    request = AgentRequest(
        request_id="phone-12",
        channel_id="xiaoyi",
        session_id="phone-sess",
        metadata={
            "xiaoyi_session_id": "phone-sess",
            "xiaoyi_task_id": "phone-task-12",
        },
    )
    _inject_session_xiaoyi_routing(request)
    assert request.metadata["xiaoyi_task_id"] == "phone-task-12"


@pytest.mark.asyncio
async def test_update_session_tools_uses_request_not_stale_contextvar() -> None:
    toolkit = SendFileToolkit(
        request_id="old-req",
        session_id="sess",
        channel_id="desktop",
        metadata={"xiaoyi_task_id": "stale-task", "xiaoyi_session_id": "phone-sess"},
    )
    adapter = object.__new__(JiuWenSwarmDeepAdapter)
    adapter._send_file_toolkit = toolkit
    adapter._send_html_card_toolkit = None
    adapter._append_reference_toolkit = None
    adapter._instance = SimpleNamespace(
        ability_manager=SimpleNamespace(
            list=lambda: [
                SimpleNamespace(name="send_file_to_user"),
                SimpleNamespace(name="send_html_card"),
                SimpleNamespace(name="xiaoyi_append_reference"),
            ]
        )
    )
    adapter._ensure_cron_tools_registered = lambda session_id: None
    adapter._resolve_prompt_channel = lambda session_id: "web"

    tokens = (
        _CRON_TOOL_CHANNEL_ID.set("desktop"),
        _CRON_TOOL_METADATA.set(
            {"xiaoyi_task_id": "stale-task", "xiaoyi_session_id": "phone-sess"}
        ),
        _CRON_TOOL_BOUND.set(True),
    )
    try:
        await adapter._update_session_tools(
            "sess",
            "turn-12",
            channel_id="xiaoyi",
            request_metadata={
                "xiaoyi_session_id": "phone-sess",
                "xiaoyi_task_id": "phone-task-12",
            },
        )
    finally:
        _CRON_TOOL_BOUND.reset(tokens[2])
        _CRON_TOOL_METADATA.reset(tokens[1])
        _CRON_TOOL_CHANNEL_ID.reset(tokens[0])

    assert toolkit.channel_id == "xiaoyi"
    assert toolkit.request_id == "turn-12"
    assert toolkit._request_metadata["xiaoyi_task_id"] == "phone-task-12"
    assert toolkit._request_metadata["xiaoyi_session_id"] == "phone-sess"
