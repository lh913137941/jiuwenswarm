# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Resource usage is attached to existing tool-call frames once per invocation."""

import json
from types import SimpleNamespace

import pytest
from openjiuwen.core.single_agent.rail.base import ToolCallInputs

from jiuwenswarm.agents.harness.common.rails import stream_event_rail
from jiuwenswarm.agents.harness.common.rails.stream_event_rail import JiuSwarmStreamEventRail
from jiuwenswarm.common import connectors
from jiuwenswarm.server.utils.stream_utils import parse_stream_chunk


class _StreamSession:
    def __init__(self):
        self.chunks = []

    async def write_stream(self, chunk):
        self.chunks.append(chunk)


@pytest.fixture(autouse=True)
def isolated_connector_registry(tmp_path, monkeypatch):
    monkeypatch.setenv(connectors.CONNECTORS_DIR_ENV, str(tmp_path / "connectors"))
    monkeypatch.delenv(connectors.CLI_CONFIG_ENV, raising=False)
    monkeypatch.setattr(stream_event_rail, "get_agent_skills_dir", lambda: tmp_path / "skills", raising=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("string_arguments", [False, True])
@pytest.mark.parametrize("skill_key", ["skill_name", "skillName"])
async def test_skill_call_has_unified_metadata_and_legacy_skill_name(string_arguments, skill_key, tmp_path):
    directory = tmp_path / "skills" / "ppt-creation"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\nname: PPT制作\n---\nInstructions", encoding="utf-8")
    (directory / ".claw-origin.json").write_text('{"source":"market"}', encoding="utf-8")
    session = _StreamSession()
    arguments = {skill_key: " ppt-creation ", "relative_file_path": "SKILL.md"}
    raw_arguments = json.dumps(arguments) if string_arguments else arguments
    call = SimpleNamespace(id="skill-call-1", name="skill_tool", arguments=raw_arguments)

    await JiuSwarmStreamEventRail._emit_tool_call(session, call)

    assert len(session.chunks) == 1
    payload = session.chunks[0].payload["tool_call"]
    assert payload["skill_name"] == "ppt-creation"
    assert payload["resource_calls"] == [
        {"resource_type": "skill", "resource_id": "ppt-creation", "resource_name": "PPT制作"}
    ]
    assert payload["arguments"] == raw_arguments
    assert call.arguments == raw_arguments
    assert parse_stream_chunk(session.chunks[0])["tool_call"]["resource_calls"] == payload["resource_calls"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("skill_name", "expected"),
    [(" primary ", "primary"), (" ", "fallback"), (123, "fallback")],
)
async def test_skill_name_prefers_valid_snake_case_and_falls_back_to_camel_case(skill_name, expected):
    arguments = {"skill_name": skill_name, "skillName": " fallback "}
    session = _StreamSession()
    call = SimpleNamespace(id="skill-priority", name="skill_tool", arguments=arguments)

    await JiuSwarmStreamEventRail._emit_tool_call(session, call)

    payload = session.chunks[0].payload["tool_call"]
    assert payload["skill_name"] == expected
    assert payload["resource_calls"][0]["resource_name"] == expected
    assert payload["arguments"] == arguments


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["bash", "powershell", "mcp_exec_command"])
async def test_connector_usage_has_unified_metadata_on_shell_tool_calls(tool_name):
    session = _StreamSession()
    call = SimpleNamespace(
        id="connector-call-1", name=tool_name, arguments={"command": "dws calendar list"}
    )

    await JiuSwarmStreamEventRail._emit_tool_call(session, call)

    payload = session.chunks[0].payload["tool_call"]
    assert payload["resource_calls"] == [
        {"resource_type": "connector", "resource_id": "dingtalk", "resource_name": "dws"}
    ]
    assert "skill_name" not in payload


@pytest.mark.asyncio
async def test_separate_connector_calls_each_emit_their_own_marker():
    session = _StreamSession()
    for call_id, command in [("query", "dws calendar list"), ("send", "dws message send --text hello")]:
        await JiuSwarmStreamEventRail._emit_tool_call(
            session, SimpleNamespace(id=call_id, name="powershell", arguments={"command": command})
        )

    assert [chunk.payload["tool_call"]["tool_call_id"] for chunk in session.chunks] == ["query", "send"]
    assert all(chunk.payload["tool_call"]["resource_calls"][0]["resource_id"] == "dingtalk"
               for chunk in session.chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("read_file", {"path": "skills/example/SKILL.md"}),
        ("search_skill", {"query": "example"}),
        ("install_skill", {"identifier": "example"}),
        ("skill_tool", {"skill_name": " "}),
        ("skill_tool", {"skill_name": 123}),
        ("skill_tool", "invalid json"),
        ("bash", {"command": "echo dws"}),
        ("read_file", {"command": "dws calendar list"}),
    ],
)
async def test_unrelated_or_invalid_calls_keep_the_existing_frame(tool_name, arguments):
    session = _StreamSession()
    call = SimpleNamespace(id="ordinary-call", name=tool_name, arguments=arguments)

    await JiuSwarmStreamEventRail._emit_tool_call(session, call)

    assert len(session.chunks) == 1
    payload = session.chunks[0].payload["tool_call"]
    assert "resource_calls" not in payload
    assert "skill_name" not in payload
    assert payload["arguments"] == arguments


@pytest.mark.asyncio
async def test_connector_registry_failure_does_not_drop_the_tool_frame(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("connector configuration unavailable")

    monkeypatch.setattr(stream_event_rail, "connector_calls_for_command", unavailable, raising=False)
    session = _StreamSession()
    call = SimpleNamespace(id="call-1", name="bash", arguments={"command": "dws calendar list"})

    await JiuSwarmStreamEventRail._emit_tool_call(session, call)

    assert len(session.chunks) == 1
    assert session.chunks[0].payload["tool_call"]["tool_call_id"] == "call-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", [None, "builtin"])
async def test_skill_without_confirmed_market_id_reports_name_with_empty_id(tmp_path, origin):
    directory = tmp_path / "skills" / "local-skill"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\nname: 本地技能\n---\nInstructions", encoding="utf-8")
    if origin:
        (directory / ".claw-origin.json").write_text(json.dumps({"source": origin}), encoding="utf-8")
    session = _StreamSession()

    await JiuSwarmStreamEventRail._emit_tool_call(
        session, SimpleNamespace(id="local-call", name="skill_tool", arguments={"skill_name": "local-skill"})
    )

    assert session.chunks[0].payload["tool_call"]["resource_calls"] == [
        {"resource_type": "skill", "resource_id": "", "resource_name": "本地技能"}
    ]


@pytest.mark.asyncio
async def test_explicit_market_skill_id_and_display_name_are_used(tmp_path):
    directory = tmp_path / "skills" / "renamed-local-directory"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\nskillId: market-skill-123\nname: raw-name\ndisplay_name: 展示名称\n---\nInstructions",
        encoding="utf-8",
    )
    session = _StreamSession()

    await JiuSwarmStreamEventRail._emit_tool_call(
        session, SimpleNamespace(id="explicit-id", name="skill_tool", arguments={"skill_name": directory.name})
    )

    assert session.chunks[0].payload["tool_call"]["resource_calls"] == [
        {"resource_type": "skill", "resource_id": "market-skill-123", "resource_name": "展示名称"}
    ]


@pytest.mark.asyncio
async def test_unreadable_skill_metadata_keeps_the_usage_marker(tmp_path):
    directory = tmp_path / "skills" / "broken-skill"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\nname: [invalid\n---\nInstructions", encoding="utf-8")
    (directory / ".claw-origin.json").write_text("invalid json", encoding="utf-8")
    session = _StreamSession()

    await JiuSwarmStreamEventRail._emit_tool_call(
        session, SimpleNamespace(id="broken-call", name="skill_tool", arguments={"skill_name": "broken-skill"})
    )

    assert session.chunks[0].payload["tool_call"]["resource_calls"] == [
        {"resource_type": "skill", "resource_id": "", "resource_name": "broken-skill"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("skill_name", ["member-skill", " member-skill "])
@pytest.mark.parametrize("skill_key", ["skill_name", "skillName"])
async def test_team_member_skill_uses_its_actual_directory(tmp_path, skill_name, skill_key):
    directory = tmp_path / "member-workspace" / "skills" / "member-skill"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\nskillId: market-member-skill\nname: 成员技能\n---\nInstructions", encoding="utf-8"
    )
    skill_rail = SimpleNamespace(get_skills_for_session=lambda session: [
        SimpleNamespace(name="member-skill", directory=directory)
    ])
    rail = JiuSwarmStreamEventRail(member_name="member-1", role="teammate")
    rail._deep_agent = SimpleNamespace(configured_rails=lambda: [skill_rail])
    session = _StreamSession()
    call = SimpleNamespace(id="member-call", name="skill_tool", arguments={skill_key: skill_name})
    ctx = SimpleNamespace(
        session=session, extra={}, exception=None,
        inputs=ToolCallInputs(tool_call=call, tool_name="skill_tool", tool_args=call.arguments),
    )

    await rail.before_tool_call(ctx)

    calls = [chunk for chunk in session.chunks if chunk.type == "tool_call"]
    assert len(calls) == 1
    assert calls[0].payload["tool_call"]["resource_calls"] == [
        {"resource_type": "skill", "resource_id": "market-member-skill", "resource_name": "成员技能"}
    ]


@pytest.mark.asyncio
async def test_saved_skill_display_name_takes_precedence(tmp_path):
    directory = tmp_path / "skills" / "skill-slug"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\nname: raw-name\n---\nInstructions", encoding="utf-8")
    (directory.parent / "skills_state.json").write_text(
        json.dumps({"local_skills": [{"name": "raw-name", "display_name": "已保存的名称"}]}), encoding="utf-8"
    )
    session = _StreamSession()

    await JiuSwarmStreamEventRail._emit_tool_call(
        session, SimpleNamespace(id="display-call", name="skill_tool", arguments={"skill_name": "skill-slug"})
    )

    assert session.chunks[0].payload["tool_call"]["resource_calls"][0]["resource_name"] == "已保存的名称"


@pytest.mark.asyncio
async def test_resource_marker_survives_existing_adapter_and_web_payload():
    from jiuwenswarm.gateway.channel_manager.web.web_connect import WebChannel
    from jiuwenswarm.server.runtime.agent_adapter.interface_deep import JiuWenSwarmDeepAdapter

    session = _StreamSession()
    await JiuSwarmStreamEventRail._emit_tool_call(
        session, SimpleNamespace(id="transport-call", name="bash", arguments={"command": "dws calendar list"})
    )
    parsed = JiuWenSwarmDeepAdapter._parse_stream_chunk(session.chunks[0])
    payload = WebChannel._build_event_payload(
        SimpleNamespace(payload=parsed, session_id="session-1", id="request-1"), "chat.tool_call"
    )

    assert payload["tool_call"]["resource_calls"] == [
        {"resource_type": "connector", "resource_id": "dingtalk", "resource_name": "dws"}
    ]
    assert payload["tool_call"]["tool_call_id"] == "transport-call"
    assert payload["session_id"] == "session-1"
    assert payload["request_id"] == "request-1"
