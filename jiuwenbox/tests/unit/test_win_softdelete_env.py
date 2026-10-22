# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import sys

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
