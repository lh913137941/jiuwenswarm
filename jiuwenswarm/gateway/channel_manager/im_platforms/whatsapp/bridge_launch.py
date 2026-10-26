# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Start the WhatsApp bridge as a fixed argv, never through a shell."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Mapping

from jiuwenswarm.gateway.channel_manager.spec import ChannelConfigError

_COMMAND_RE = re.compile(r'^(?:"([^"]+)"|(\S+))\s+(?:"([^"]+)"|(\S+))$')
_SHELL_METACHARACTERS = frozenset(";|&<>`$")
_NODE_NAMES = frozenset({"node", "node.exe"})
_ALLOWED_ENV = frozenset(
    {
        "WA_BRIDGE_HOST",
        "WA_BRIDGE_PORT",
        "WA_BRIDGE_PATH",
        "WA_AUTH_DIR",
        "WA_PRINT_QR",
    }
)
_BLOCKED_ENV = frozenset(
    {
        "NODE_OPTIONS",
        "NODE_PATH",
        "NODE_DEBUG",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
        "BASH_ENV",
        "ENV",
        "SHELLOPTS",
        "PS4",
        "PERL5OPT",
        "PERL5LIB",
        "RUBYOPT",
        "RUBYLIB",
        "GIT_EXEC_PATH",
        "CDPATH",
    }
)
_MAX_TEXT = 4096


def official_bridge_script() -> Path:
    """Return the bridge script shipped next to this package."""
    return Path(__file__).resolve().parents[4] / "scripts" / "whatsapp-bridge.js"


def default_bridge_workdir() -> Path:
    """Historical default cwd: the im_platforms directory."""
    return Path(__file__).resolve().parents[1]


def config_enabled(value: Any, default: bool = False) -> bool:
    """Read a config flag. The string ``false`` is off."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def validate_whatsapp_channel_config(config: Mapping[str, Any]) -> None:
    """Reject a bridge command, workdir, or environment that must not be stored."""
    command = str(config.get("bridge_command") or "").strip()
    auto_start = config_enabled(config.get("auto_start_bridge", False))
    if command:
        build_bridge_argv(command, resolve_executable=False)
    elif auto_start:
        raise ChannelConfigError("auto_start_bridge 需要合法的 bridge_command")
    workdir = resolve_bridge_workdir(str(config.get("bridge_workdir") or ""))
    raw_env = config.get("bridge_env") or {}
    if not isinstance(raw_env, dict):
        raise ChannelConfigError("bridge_env 必须是对象")
    checked_bridge_env(raw_env, workdir)


def build_bridge_argv(
    command: str,
    *,
    official_script: Path | None = None,
    resolve_executable: bool = True,
) -> list[str]:
    """Return ``[node, shipped-script]`` or raise if the command is anything else."""
    node_token, script_token = _split_command(command)
    script = _match_script(script_token, official_script or official_bridge_script())
    node = _match_node(node_token, resolve_executable=resolve_executable)
    return [node, str(script)]


def resolve_bridge_workdir(raw: str) -> Path:
    """Use the default cwd, or an existing absolute directory."""
    text = (raw or "").strip()
    if not text:
        return default_bridge_workdir()
    if len(text) > _MAX_TEXT or any(ord(char) < 32 for char in text):
        raise ChannelConfigError("bridge_workdir 无效")
    path = Path(text)
    if not path.is_absolute():
        raise ChannelConfigError("bridge_workdir 必须是绝对路径")
    if path.is_symlink():
        raise ChannelConfigError("bridge_workdir 必须是实际目录")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ChannelConfigError("bridge_workdir 不存在") from exc
    if not resolved.is_dir():
        raise ChannelConfigError("bridge_workdir 必须是实际目录")
    return resolved


def checked_bridge_env(overrides: Mapping[str, Any], workdir: Path) -> dict[str, str]:
    """Keep only the bridge's own settings and reject loader variables."""
    cleaned: dict[str, str] = {}
    canonical = {name.upper(): name for name in _ALLOWED_ENV}
    for key, value in overrides.items():
        if not isinstance(key, str) or key.upper() not in canonical:
            raise ChannelConfigError("bridge_env 包含不允许的变量")
        if (
            isinstance(value, bool)
            or isinstance(value, (dict, list))
            or not isinstance(value, (str, int, float))
        ):
            raise ChannelConfigError("bridge_env 的值必须是字符串")
        if isinstance(value, float):
            if not value.is_integer():
                raise ChannelConfigError("bridge_env 的值必须是字符串")
            text = str(int(value))
        else:
            text = str(value)
        if len(text) > _MAX_TEXT or any(ord(char) < 32 for char in text):
            raise ChannelConfigError("bridge_env 的值无效")
        name = canonical[key.upper()]
        cleaned[name] = _check_env_value(name, text, workdir)
    return cleaned


def child_environment(overrides: Mapping[str, Any], workdir: Path) -> dict[str, str]:
    """Parent environment without loader variables, plus checked bridge settings."""
    env = {key: value for key, value in os.environ.items() if not _dangerous_env(key)}
    env.update(checked_bridge_env(overrides, workdir))
    return env


def _split_command(command: str) -> tuple[str, str]:
    text = (command or "").strip()
    if not text or len(text) > _MAX_TEXT or any(ord(char) < 32 for char in text):
        raise ChannelConfigError("bridge_command 无效")
    if any(char in _SHELL_METACHARACTERS for char in text):
        raise ChannelConfigError("bridge_command 不能包含 shell 元字符")
    match = _COMMAND_RE.fullmatch(text)
    if match is None:
        raise ChannelConfigError("bridge_command 只能是 node 与官方 whatsapp-bridge.js")
    node = match.group(1) or match.group(2)
    script = match.group(3) or match.group(4)
    if not node or not script:
        raise ChannelConfigError("bridge_command 只能是 node 与官方 whatsapp-bridge.js")
    return node, script


def _match_script(token: str, official: Path) -> Path:
    try:
        official_real = official.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ChannelConfigError("官方 whatsapp-bridge.js 不存在") from exc
    candidate = Path(token)
    if not candidate.is_absolute():
        candidate = official_real.parents[1] / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ChannelConfigError(
            "bridge_command 必须使用官方 whatsapp-bridge.js"
        ) from exc
    if resolved != official_real or not resolved.is_file():
        raise ChannelConfigError("bridge_command 必须使用官方 whatsapp-bridge.js")
    return resolved


def _match_node(token: str, *, resolve_executable: bool) -> str:
    if Path(token).name.lower() not in _NODE_NAMES:
        raise ChannelConfigError("bridge_command 只能启动 node")
    if not resolve_executable:
        return token
    found = find_node_executable()
    if found is None:
        raise ChannelConfigError("找不到 node 可执行文件")
    found_real = Path(found).resolve()
    token_path = Path(token)
    if token_path.is_absolute() or token_path.parent != Path("."):
        try:
            given = token_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ChannelConfigError("bridge_command 只能启动 node") from exc
        if given != found_real:
            raise ChannelConfigError("bridge_command 只能启动 node")
    return str(found_real)


def find_node_executable() -> str | None:
    """Locate the Node binary. Windows cmd shims are not accepted."""
    names = ("node.exe", "node") if os.name == "nt" else ("node",)
    for name in names:
        found = shutil.which(name)
        if not found:
            continue
        real = Path(found).resolve()
        if real.name.lower() in _NODE_NAMES and real.is_file():
            if os.name == "nt" and real.suffix.lower() != ".exe":
                continue
            return str(real)
    return None


def _check_env_value(name: str, value: str, workdir: Path) -> str:
    if name == "WA_BRIDGE_HOST":
        from jiuwenswarm.common.security.web_identity import is_loopback

        if not is_loopback(value):
            raise ChannelConfigError("WA_BRIDGE_HOST 只能是本机地址")
        return value
    if name == "WA_BRIDGE_PORT":
        if not value.isdigit() or not 1 <= int(value) <= 65535:
            raise ChannelConfigError("WA_BRIDGE_PORT 无效")
        return value
    if name == "WA_BRIDGE_PATH":
        if (
            not value.startswith("/")
            or ".." in value.split("/")
            or len(value) > 128
            or any(char not in "/._~-" and not char.isalnum() for char in value)
        ):
            raise ChannelConfigError("WA_BRIDGE_PATH 无效")
        return value
    if name == "WA_PRINT_QR":
        if value not in {"0", "1"}:
            raise ChannelConfigError("WA_PRINT_QR 无效")
        return value
    if name == "WA_AUTH_DIR":
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = workdir / candidate
        try:
            resolved = candidate.resolve(strict=False)
            root = workdir.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ChannelConfigError("WA_AUTH_DIR 无效") from exc
        if resolved == root or not resolved.is_relative_to(root):
            raise ChannelConfigError("WA_AUTH_DIR 必须位于 bridge_workdir 内")
        return str(resolved)
    raise ChannelConfigError("bridge_env 包含不允许的变量")


def _dangerous_env(key: str) -> bool:
    name = key.upper()
    return name in _BLOCKED_ENV or name.startswith(("LD_", "DYLD_"))
