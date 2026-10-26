# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Runner Python dirs owned by the current user are authorized without elevation."""

import json
import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows sandbox setup")


def test_owned_new_tool_dir_is_granted_without_reinstall(monkeypatch, tmp_path):
    from jiuwenbox.supervisor import win_acl, win_setup
    from jiuwenbox.supervisor import win_constants as const

    owned, external = tmp_path / "uv-python", tmp_path / "program-files-python"
    owned.mkdir()
    external.mkdir()
    registry = {const.REG_VALUE_INSTALLED: "1", const.REG_VALUE_PREINSTALLED_PATHS: json.dumps([])}
    monkeypatch.setattr(win_setup, "reg_get_str", registry.get)
    monkeypatch.setattr(win_setup, "_reg_set_str", registry.__setitem__)
    monkeypatch.setattr(win_setup, "get_sandbox_user_sid", lambda: "S-1-5-21-sandbox")
    monkeypatch.setattr(win_setup, "_path_owned_by_current_user", lambda p: p == str(owned))
    monkeypatch.setattr(win_setup, "_is_admin", lambda: False)
    monkeypatch.setattr(win_setup, "_verify_or_reset_sandbox_user_password", lambda: None)
    monkeypatch.setattr(win_setup, "_resolve_install_log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(win_setup, "soft_delete_dll_acl_missing", lambda: False)
    def record_install(**kwargs):
        recorded = win_setup.get_preinstalled_read_paths() | {
            p.rstrip("\\/").lower() for p in kwargs["preinstall_paths"]
        }
        registry[const.REG_VALUE_PREINSTALLED_PATHS] = json.dumps(sorted(recorded))
    install = MagicMock(side_effect=record_install)
    monkeypatch.setattr(win_setup, "install", install)
    grant = MagicMock()
    monkeypatch.setattr(win_acl, "grant_ace", grant)

    win_setup.ensure_windows_setup(preinstall_paths=[str(owned), str(external)])

    grant.assert_called_once_with(
        str(owned), "S-1-5-21-sandbox", rights=const.ALLOW_READ_EXECUTE_RIGHTS, mode="ALLOW", recursive=True,
    )
    assert set(json.loads(registry[const.REG_VALUE_PREINSTALLED_PATHS])) == {str(owned).lower(), str(external).lower()}
    install.assert_called_once()
    assert install.call_args.kwargs["force"] is True

    grant.reset_mock()
    install.reset_mock()
    win_setup.ensure_windows_setup(preinstall_paths=[str(owned), str(external)])
    grant.assert_not_called()
    install.assert_not_called()
