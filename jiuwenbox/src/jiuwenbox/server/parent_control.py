# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Private stdin shutdown for a parent-owned, console-less box-server."""

from __future__ import annotations

import argparse
import sys
import threading
from typing import Any

import uvicorn


SHUTDOWN_BYTE = b"\x01"


def _read_parent_byte() -> bytes:
    if sys.stdin is not None:
        return sys.stdin.buffer.read(1)
    if sys.platform != "win32":
        return b""
    # PyInstaller's windowed executable has no sys.stdin even when its parent
    # supplies an inherited pipe. Read the OS handle without requiring fd 0.
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
    kernel32.GetStdHandle.restype = wintypes.HANDLE
    kernel32.ReadFile.argtypes = [
        wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
    ]
    kernel32.ReadFile.restype = wintypes.BOOL
    data, size = ctypes.create_string_buffer(1), wintypes.DWORD()
    if not kernel32.ReadFile(kernel32.GetStdHandle(-10), data, 1, ctypes.byref(size), None):
        raise ctypes.WinError(ctypes.get_last_error())
    return data.raw[:size.value]


def run_server(app: Any, *, host: str, port: int) -> None:
    """Stop through Uvicorn's normal lifespan, without console signals or an API."""
    requested = threading.Event()

    class ParentControlledServer(uvicorn.Server):
        async def on_tick(self, counter: int) -> bool:
            # Only act after startup; setting should_exit during startup can
            # make Uvicorn return before entering its lifespan shutdown path.
            if requested.is_set():
                self.should_exit = True
            return await super().on_tick(counter)

    def read_parent() -> None:
        try:
            if _read_parent_byte() == SHUTDOWN_BYTE:
                requested.set()
        except (OSError, ValueError):
            pass

    threading.Thread(target=read_parent, name="box-parent-control", daemon=True).start()
    server = ParentControlledServer(uvicorn.Config(
        app, host=host, port=port, log_level="info", timeout_graceful_shutdown=5.0,
    ))
    # The HTTP drain is bounded; lifespan shutdown and its sandbox/ACL cleanup
    # still run to completion before the parent can retire the service.
    server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8321)
    args, _unknown = parser.parse_known_args()
    run_server("jiuwenbox.server.app:app", host=args.host, port=args.port)
