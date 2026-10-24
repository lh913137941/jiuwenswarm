# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A hidden Windows box-server must stop through lifespan without a console."""

import ctypes
import builtins
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from jiuwenswarm.server.sandbox import jiuwenbox_runner as runner_module


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["import", "stream"])
async def test_stdin_failure_still_stops_process_and_retires_after_exit(monkeypatch, failure):
    from openjiuwen.extensions.sys_operation.sandbox.providers import jiuwenbox as box

    events = []
    writer = Mock()
    writer.write.side_effect = ValueError("closed control stream") if failure == "stream" else None
    proc = SimpleNamespace(pid=123, returncode=None, stdin=writer, kill=Mock())
    proc.terminate = lambda: events.append("terminate")

    async def wait():
        events.append("exit")
        proc.returncode = 0

    proc.wait = wait
    runner = runner_module.JiuwenBoxRunner()
    runner.process, runner.owns_process = proc, True
    monkeypatch.setattr(runner_module, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_a, **_k: SimpleNamespace(
        GenerateConsoleCtrlEvent=Mock(return_value=False),
    ), raising=False)
    real_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if failure == "import" and name == "jiuwenbox.server.parent_control":
            raise ImportError("parent control missing from package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    monkeypatch.setattr(runner, "_release_win_job", lambda: events.append("job"))
    monkeypatch.setattr(box, "retire_jiuwenbox_service", AsyncMock(side_effect=lambda _: events.append("retire")))
    await runner.stop()
    assert events == ["terminate", "exit", "job", "retire"]
    proc.kill.assert_not_called()


def test_no_stdin_on_non_windows_does_not_use_win32_handles(monkeypatch):
    from jiuwenbox.server import parent_control

    monkeypatch.setattr(parent_control, "sys", SimpleNamespace(stdin=None, platform="linux"))
    win_api = Mock(side_effect=AssertionError("Win32 must not be used"))
    monkeypatch.setattr(ctypes, "WinDLL", win_api, raising=False)
    assert parent_control._read_parent_byte() == b""
    win_api.assert_not_called()


@pytest.mark.asyncio
async def test_parent_stdin_requests_shutdown_before_job_and_cache_retirement(monkeypatch):
    from openjiuwen.extensions.sys_operation.sandbox.providers import jiuwenbox as box

    events = []
    control = SimpleNamespace(
        write=lambda data: events.append(("stdin", data)), drain=AsyncMock(),
        close=lambda: events.append("stdin-close"),
    )
    proc = SimpleNamespace(pid=123, returncode=None, stdin=control, terminate=Mock(), kill=Mock())

    async def wait():
        events.append("lifespan-cleanup")
        proc.returncode = 0
        return 0

    proc.wait = wait
    runner = runner_module.JiuwenBoxRunner()
    runner.process, runner.owns_process = proc, True
    monkeypatch.setattr(runner_module, "sys", SimpleNamespace(platform="win32"))
    console_signal = Mock(side_effect=AssertionError("must not rely on a console"))
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_a, **_k: SimpleNamespace(
        GenerateConsoleCtrlEvent=console_signal,
    ), raising=False)
    monkeypatch.setattr(runner, "_release_win_job", lambda: events.append("job"))
    monkeypatch.setattr(box, "retire_jiuwenbox_service", AsyncMock(side_effect=lambda _: events.append("retire")))
    await runner.stop()
    assert events == [("stdin", b"\x01"), "stdin-close", "lifespan-cleanup", "job", "retire"]
    console_signal.assert_not_called()
    proc.terminate.assert_not_called()
    proc.kill.assert_not_called()


@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "win32", reason="Windows CREATE_NO_WINDOW regression")
@pytest.mark.parametrize("early,windowed_stdin,busy", [
    (False, False, False), (True, False, False), (False, True, False), (False, False, True),
])
def test_real_consoleless_server_runs_full_lifespan_cleanup(tmp_path, unused_tcp_port, early, windowed_stdin, busy):
    marker = tmp_path / "events.txt"
    script = tmp_path / "server.py"
    script.write_text(
        "import asyncio, ctypes, sys\n"
        "from contextlib import asynccontextmanager\n"
        "from pathlib import Path\n"
        "from fastapi import FastAPI\n"
        "from jiuwenbox.server.parent_control import run_server\n"
        + ("sys.stdin = None\n" if windowed_stdin else "")
        +
        f"marker = Path({str(marker)!r})\n"
        "@asynccontextmanager\n"
        "async def lifespan(app):\n"
        "    marker.write_text('started', encoding='utf-8')\n"
        "    print('READY ' + str(bool(ctypes.windll.kernel32.GetConsoleWindow())), flush=True)\n"
        "    try:\n"
        "        yield\n"
        "    finally:\n"
        "        await asyncio.sleep(0.05)\n"
        "        marker.write_text('cleanup-complete', encoding='utf-8')\n"
        "app = FastAPI(lifespan=lifespan)\n"
        "@app.get('/wait')\n"
        "async def wait():\n"
        "    print('REQUEST_READY', flush=True)\n"
        "    await asyncio.Event().wait()\n"
        f"run_server(app, host='127.0.0.1', port={unused_tcp_port})\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3] / "jiuwenbox" / "src")}
    proc = subprocess.Popen(
        [sys.executable, str(script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    connection = None
    try:
        if not early:
            # This child has no console; a console control event cannot reach it.
            assert proc.stdout.readline().decode().strip() == "READY False"
            ctypes.windll.kernel32.GenerateConsoleCtrlEvent(1, proc.pid)
            time.sleep(0.25)
            assert proc.poll() is None
        if busy:
            connection = socket.create_connection(("127.0.0.1", unused_tcp_port), timeout=5)
            connection.sendall(b"GET /wait HTTP/1.1\r\nHost: localhost\r\n\r\n")
            assert proc.stdout.readline().decode().strip() == "REQUEST_READY"
        stdout, stderr = proc.communicate(input=b"\x01", timeout=15)
        assert proc.returncode == 0, stderr.decode(errors="replace")
        assert marker.read_text(encoding="utf-8") == "cleanup-complete"
        assert "Application shutdown complete" in stderr.decode(errors="replace")
        if busy:
            assert "timeout graceful shutdown exceeded" in stderr.decode(errors="replace")
    finally:
        if connection is not None:
            connection.close()
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=5)
