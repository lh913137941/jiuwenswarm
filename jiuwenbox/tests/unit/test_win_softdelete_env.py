# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from jiuwenbox.supervisor import win_softdelete


@pytest.mark.skipif(sys.platform != "win32", reason="Windows soft-delete env")
def test_merge_host_log_layout_env_copies_from_process(monkeypatch):
    monkeypatch.setenv("JIUWENSWARM_LOG_DIR", r"C:\host\logs\agent")
    monkeypatch.setenv("JIUWENSWARM_LOG_DATE_ROOT", r"C:\host\logs")
    env: dict[str, str] = {"USERPROFILE": r"C:\Users\jbx-sandbox"}
    win_softdelete.merge_host_log_layout_env(env)
    assert env["JIUWENSWARM_LOG_DIR"] == r"C:\host\logs\agent"
    assert env["JIUWENSWARM_LOG_DATE_ROOT"] == r"C:\host\logs"
    assert env["USERPROFILE"] == r"C:\Users\jbx-sandbox"


def test_merge_soft_delete_env_includes_log_layout(monkeypatch):
    monkeypatch.setenv("JIUWENBOX_SOFT_DELETE", "1")
    monkeypatch.setenv("JIUWENSWARM_LOG_DIR", "/tmp/logs")
    env: dict[str, str] = {}
    win_softdelete.merge_soft_delete_env(env)
    assert env["JIUWENBOX_SOFT_DELETE"] == "1"
    assert env["JIUWENSWARM_LOG_DIR"] == "/tmp/logs"


def _fake_dacl(aces):
    return SimpleNamespace(GetAceCount=lambda: len(aces), GetAce=lambda index: aces[index])


def _fake_win32security(aces, monkeypatch):
    dacl = _fake_dacl(aces)
    sd = SimpleNamespace(GetSecurityDescriptorDacl=lambda: dacl)
    fake = SimpleNamespace(
        ConvertStringSidToSid=lambda sid: sid,
        GetNamedSecurityInfo=lambda *args: sd,
        SE_FILE_OBJECT=1,
        DACL_SECURITY_INFORMATION=4,
        INHERIT_ONLY_ACE=0x8,
    )
    monkeypatch.setitem(sys.modules, "win32security", fake)


def test_sandbox_can_load_dll_accepts_users_group_rx(monkeypatch):
    """安装器只授 Users:(RX), 沙箱用户靠组成员资格拿到读执行权."""
    from jiuwenbox.supervisor import win_constants as const

    _fake_win32security(
        [((const.ACCESS_ALLOWED_ACE_TYPE, 0), const.ALLOW_READ_EXECUTE_RIGHTS, "S-1-5-32-545")],
        monkeypatch,
    )
    assert win_softdelete.sandbox_can_load_dll(Path("jiuwen_softdelete.dll"), "S-1-5-21-sandbox") is True


def test_sandbox_can_load_dll_rejects_deny(monkeypatch):
    from jiuwenbox.supervisor import win_constants as const

    _fake_win32security(
        [
            ((const.ACCESS_DENIED_ACE_TYPE, 0), const.ALLOW_READ_EXECUTE_RIGHTS, "S-1-5-32-545"),
            ((const.ACCESS_ALLOWED_ACE_TYPE, 0), const.ALLOW_READ_EXECUTE_RIGHTS, "S-1-5-32-545"),
        ],
        monkeypatch,
    )
    assert win_softdelete.sandbox_can_load_dll(Path("jiuwen_softdelete.dll"), "S-1-5-21-sandbox") is False


def test_sandbox_can_load_dll_ignores_inherit_only(monkeypatch):
    from jiuwenbox.supervisor import win_constants as const

    _fake_win32security(
        [((const.ACCESS_ALLOWED_ACE_TYPE, 0x8), const.ALLOW_READ_EXECUTE_RIGHTS, "S-1-5-32-545")],
        monkeypatch,
    )
    assert win_softdelete.sandbox_can_load_dll(Path("jiuwen_softdelete.dll"), "S-1-5-21-sandbox") is False
