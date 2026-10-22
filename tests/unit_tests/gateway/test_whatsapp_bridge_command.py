# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""WhatsApp bridge launch must be a fixed argv, never a shell string."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from jiuwenswarm.gateway.channel_manager.base import RobotMessageRouter
from jiuwenswarm.gateway.channel_manager.im_platforms.whatsapp import bridge_launch
from jiuwenswarm.gateway.channel_manager.im_platforms.whatsapp.bridge_launch import (
    build_bridge_argv,
    child_environment,
    config_enabled,
    find_node_executable,
    official_bridge_script,
    resolve_bridge_workdir,
    validate_whatsapp_channel_config,
)
from jiuwenswarm.gateway.channel_manager.im_platforms.whatsapp.whatsapp_connect import (
    WhatsAppChannel,
    WhatsAppChannelConfig,
)
from jiuwenswarm.gateway.channel_manager.spec import ChannelConfigError
from jiuwenswarm.gateway.channel_manager.web import app_web_handlers
from jiuwenswarm.gateway.channel_manager.web.app_web_handlers import (
    WebHandlersBindParams,
    _register_web_handlers,
)


class _FakeWebChannel:
    def __init__(self):
        self.methods: dict[str, object] = {}
        self.responses: list[dict] = []

    def register_method(self, name, handler):
        self.methods[name] = handler

    def on_connect(self, handler):
        return None

    async def send_response(
        self, ws, req_id, *, ok, payload=None, error=None, code=None
    ):
        self.responses.append(
            {"id": req_id, "ok": ok, "payload": payload, "error": error, "code": code}
        )


class _FakeChannelManager:
    def __init__(self):
        self.configs: dict[str, dict] = {}

    async def set_conf(self, channel_id, new_conf):
        self.configs[channel_id] = dict(new_conf)

    def get_conf(self, channel_id):
        return dict(self.configs.get(channel_id, {}))


def test_official_script_is_the_shipped_file():
    script = official_bridge_script()
    assert script.name == "whatsapp-bridge.js"
    assert script.is_file()


def test_default_command_is_node_plus_shipped_script():
    argv = build_bridge_argv(
        "node scripts/whatsapp-bridge.js", resolve_executable=False
    )
    assert argv == ["node", str(official_bridge_script().resolve())]


def test_quoted_node_path_and_integer_port_are_accepted(tmp_path: Path):
    quoted = r'"C:\Program Files (x86)\nodejs\node.exe" "scripts/whatsapp-bridge.js"'
    argv = build_bridge_argv(quoted, resolve_executable=False)
    assert argv[0].endswith("node.exe")
    assert argv[1] == str(official_bridge_script().resolve())
    validate_whatsapp_channel_config(
        {
            "bridge_command": "node scripts/whatsapp-bridge.js",
            "bridge_workdir": str(tmp_path),
            "bridge_env": {"WA_BRIDGE_PORT": 19600.0, "WA_BRIDGE_HOST": "127.0.0.1"},
            "auto_start_bridge": "false",
        }
    )
    assert config_enabled("false") is False
    assert config_enabled("true") is True


@pytest.mark.parametrize(
    "command",
    [
        "node scripts/whatsapp-bridge.js; id > /tmp/pwned",
        "node scripts/whatsapp-bridge.js && id",
        "node scripts/whatsapp-bridge.js | id",
        "node scripts/whatsapp-bridge.js $(id)",
        "node scripts/whatsapp-bridge.js `id`",
        "node -e process.exit(0)",
        "node scripts/whatsapp-bridge.js --require evil.js",
        "bash scripts/whatsapp-bridge.js",
        "node scripts/not-the-bridge.js",
    ],
)
def test_bridge_command_rejects_shell_and_extra_arguments(command: str):
    with pytest.raises(ChannelConfigError):
        build_bridge_argv(command, resolve_executable=False)


def test_bridge_workdir_must_be_an_existing_absolute_directory(tmp_path: Path):
    assert resolve_bridge_workdir("") == bridge_launch.default_bridge_workdir()
    assert resolve_bridge_workdir(str(tmp_path)) == tmp_path.resolve()
    with pytest.raises(ChannelConfigError, match="绝对路径"):
        resolve_bridge_workdir("scripts")
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(ChannelConfigError, match="实际目录"):
        resolve_bridge_workdir(str(target))


def test_bridge_env_rejects_loader_variables_and_remote_host(tmp_path: Path):
    with pytest.raises(ChannelConfigError, match="不允许"):
        validate_whatsapp_channel_config(
            {
                "bridge_env": {"NODE_OPTIONS": "--require evil.js"},
                "bridge_workdir": str(tmp_path),
            }
        )
    with pytest.raises(ChannelConfigError, match="不允许"):
        validate_whatsapp_channel_config(
            {
                "bridge_env": {"LD_PRELOAD": "/tmp/evil.so"},
                "bridge_workdir": str(tmp_path),
            }
        )
    with pytest.raises(ChannelConfigError, match="本机"):
        validate_whatsapp_channel_config(
            {
                "bridge_env": {"WA_BRIDGE_HOST": "1.2.3.4"},
                "bridge_workdir": str(tmp_path),
            }
        )
    outside = tmp_path.parent / "outside-auth"
    with pytest.raises(ChannelConfigError, match="bridge_workdir"):
        validate_whatsapp_channel_config(
            {
                "bridge_env": {"WA_AUTH_DIR": str(outside)},
                "bridge_workdir": str(tmp_path),
            }
        )


def test_child_environment_strips_inherited_loader_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("NODE_OPTIONS", "--require evil.js")
    monkeypatch.setenv("LD_PRELOAD", "evil.so")
    monkeypatch.setenv("PYTHONPATH", "evil")
    env = child_environment({"WA_PRINT_QR": "0"}, tmp_path)
    assert "NODE_OPTIONS" not in env
    assert "LD_PRELOAD" not in env
    assert "PYTHONPATH" not in env
    assert env["WA_PRINT_QR"] == "0"
    assert "PATH" in env or "Path" in env


def test_windows_cmd_shim_is_not_a_node_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    shim = tmp_path / "node.cmd"
    shim.write_text("@echo off\n", encoding="utf-8")
    monkeypatch.setattr(bridge_launch.os, "name", "nt")

    def which(name: str):
        if name == "node":
            return str(shim)
        return None

    monkeypatch.setattr(bridge_launch.shutil, "which", which)
    assert find_node_executable() is None


def test_spawn_uses_resolved_node_and_rejects_a_different_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    script = official_bridge_script()
    node = tmp_path / ("node.exe" if os.name == "nt" else "node")
    node.write_text("", encoding="utf-8")
    decoy_dir = tmp_path / "decoy"
    decoy_dir.mkdir()
    decoy = decoy_dir / node.name
    decoy.write_text("", encoding="utf-8")
    monkeypatch.setattr(bridge_launch, "find_node_executable", lambda: str(node))

    argv = build_bridge_argv("node scripts/whatsapp-bridge.js", resolve_executable=True)
    assert Path(argv[0]) == node.resolve()
    assert argv[1] == str(script.resolve())

    with pytest.raises(ChannelConfigError):
        build_bridge_argv(f'"{decoy}" "{script}"', resolve_executable=True)


@pytest.mark.asyncio
async def test_start_bridge_execs_official_script_without_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    node = tmp_path / ("node.exe" if os.name == "nt" else "node")
    node.write_text("", encoding="utf-8")
    monkeypatch.setattr(bridge_launch, "find_node_executable", lambda: str(node))
    monkeypatch.setenv("NODE_OPTIONS", "--require evil.js")
    calls: dict[str, object] = {}

    async def fake_exec(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return SimpleNamespace(pid=123, returncode=None)

    async def fake_shell(*args, **kwargs):
        raise AssertionError("shell must not be used")

    monkeypatch.setattr(
        "jiuwenswarm.gateway.channel_manager.im_platforms.whatsapp.whatsapp_connect.asyncio.create_subprocess_exec",
        fake_exec,
    )
    monkeypatch.setattr(
        "jiuwenswarm.gateway.channel_manager.im_platforms.whatsapp.whatsapp_connect.asyncio.create_subprocess_shell",
        fake_shell,
    )
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(
            bridge_command="node scripts/whatsapp-bridge.js",
            bridge_workdir=str(tmp_path),
            bridge_env={"WA_PRINT_QR": "1"},
        ),
        RobotMessageRouter(),
    )

    await channel._start_bridge_process()

    assert calls["args"][0] == str(node.resolve())
    assert calls["args"][1] == str(official_bridge_script().resolve())
    kwargs = calls["kwargs"]
    assert kwargs["cwd"] == str(tmp_path.resolve())
    assert "NODE_OPTIONS" not in kwargs["env"]
    assert kwargs["env"]["WA_PRINT_QR"] == "1"
    assert channel._bridge_process.pid == 123


@pytest.mark.asyncio
async def test_shell_command_does_not_start_a_process(monkeypatch: pytest.MonkeyPatch):
    started = False

    async def fake_exec(*args, **kwargs):
        nonlocal started
        started = True

    monkeypatch.setattr(
        "jiuwenswarm.gateway.channel_manager.im_platforms.whatsapp.whatsapp_connect.asyncio.create_subprocess_exec",
        fake_exec,
    )
    channel = WhatsAppChannel(
        WhatsAppChannelConfig(
            bridge_command="node scripts/whatsapp-bridge.js; id > /tmp/pwned"
        ),
        RobotMessageRouter(),
    )

    await channel._start_bridge_process()

    assert started is False
    assert channel._bridge_process is None


@pytest.mark.asyncio
async def test_set_conf_rejects_shell_command_before_save():
    channel = _FakeWebChannel()
    manager = _FakeChannelManager()
    _register_web_handlers(
        WebHandlersBindParams(channel=channel, channel_manager=manager)
    )

    await channel.methods["channel.whatsapp.set_conf"](
        object(),
        "req-bad",
        {
            "bridge_command": "node scripts/whatsapp-bridge.js; id > /tmp/pwned",
            "auto_start_bridge": True,
            "bridge_env": {"LD_PRELOAD": "evil.so"},
        },
        "sess-1",
    )

    assert channel.responses[-1]["ok"] is False
    assert channel.responses[-1]["code"] == "BAD_REQUEST"
    assert "whatsapp" not in manager.configs


@pytest.mark.asyncio
async def test_set_conf_accepts_the_default_command(monkeypatch: pytest.MonkeyPatch):
    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(app_web_handlers, "_clear_agent_config_cache", _noop)
    channel = _FakeWebChannel()
    manager = _FakeChannelManager()
    _register_web_handlers(
        WebHandlersBindParams(channel=channel, channel_manager=manager)
    )

    params = {
        "enabled": False,
        "bridge_command": "node scripts/whatsapp-bridge.js",
        "bridge_workdir": "",
        "auto_start_bridge": False,
    }
    await channel.methods["channel.whatsapp.set_conf"](
        object(), "req-ok", params, "sess-1"
    )

    assert channel.responses[-1]["ok"] is True
    assert (
        manager.configs["whatsapp"]["bridge_command"]
        == "node scripts/whatsapp-bridge.js"
    )
