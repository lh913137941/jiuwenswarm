"""Startup must validate setup without modifying the machine during tests."""

import ctypes
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jiuwenbox.supervisor import win_setup as setup
from jiuwenbox.supervisor import win_constants as const


@pytest.fixture
def state(monkeypatch, tmp_path):
    registry = {const.REG_VALUE_INSTALLED: "1", const.REG_VALUE_PREINSTALLED_PATHS: "[]"}
    monkeypatch.setattr(setup, "_require_windows", lambda: None)
    monkeypatch.setattr(setup, "reg_get_str", registry.get)
    monkeypatch.setattr(setup, "_resolve_install_log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(setup, "_lookup_current_user_sid", lambda: "S-1-host")
    monkeypatch.setattr(setup, "_sandbox_user_exists", lambda: True)
    monkeypatch.setattr(setup, "get_sandbox_user_password", lambda: "test-credential")
    monkeypatch.setattr(setup, "_logon_sandbox_user", lambda password, **kwargs: True)
    monkeypatch.setattr(setup, "_is_admin", lambda: False)
    monkeypatch.setattr(setup, "_can_write_dac", lambda path: True)
    # 软删除 DLL 的补授权单独覆盖, 默认视为已授权, 避免每个用例都触发 install.
    monkeypatch.setattr(setup, "soft_delete_dll_acl_missing", lambda: False)
    return registry


@pytest.mark.parametrize("wait,exit_code,error", [
    (0, 0, None), (0, 1, "退出码 1"), (0x102, 0, "超时"),
])
def test_elevated_process_exit_controls_success(monkeypatch, state, wait, exit_code, error):
    def launch(ptr):
        ptr._obj.hProcess = 123
        return True

    def get_exit(handle, ptr):
        ptr._obj.value = exit_code
        return True

    kernel = SimpleNamespace(WaitForSingleObject=Mock(return_value=wait),
                             GetExitCodeProcess=get_exit, CloseHandle=Mock())
    monkeypatch.setattr(setup, "get_kernel32", lambda: kernel)
    monkeypatch.setattr(setup, "_get_shell32", lambda: SimpleNamespace(ShellExecuteExW=launch))
    monkeypatch.setattr(setup, "_uac_shell_execute_target", lambda *args: ("python.exe", "mock"))
    if error:
        with pytest.raises(RuntimeError, match=error):
            setup._elevate_and_run_install()
    else:
        assert setup._elevate_and_run_install() == 0
    kernel.CloseHandle.assert_called_once_with(123)


@pytest.mark.parametrize("code,message", [(1223, "已取消"), (5, "WinError 5")])
def test_uac_rejection_preserves_error(monkeypatch, state, code, message):
    monkeypatch.setattr(ctypes, "get_last_error", lambda: code, raising=False)
    monkeypatch.setattr(ctypes, "FormatError", lambda err: "access denied", raising=False)
    monkeypatch.setattr(setup, "_get_shell32", lambda: SimpleNamespace(ShellExecuteExW=lambda ptr: False))
    monkeypatch.setattr(setup, "_uac_shell_execute_target", lambda *args: ("python.exe", "mock"))
    with pytest.raises(RuntimeError, match=message):
        setup._elevate_and_run_install()


def test_ready_environment_does_not_install(monkeypatch, state):
    install = Mock()
    monkeypatch.setattr(setup, "install", install)
    setup.ensure_windows_setup()
    install.assert_not_called()


@pytest.mark.parametrize("problem", ["uninstalled", "missing-account", "missing-password", "wrong-password"])
def test_repair_is_requested_and_result_verified(monkeypatch, state, problem):
    broken = [True]
    if problem == "uninstalled":
        state[const.REG_VALUE_INSTALLED] = ""
    elif problem == "missing-account":
        monkeypatch.setattr(setup, "_sandbox_user_exists", lambda: not broken[0])
    elif problem == "missing-password":
        monkeypatch.setattr(setup, "get_sandbox_user_password", lambda: None if broken[0] else "repaired")
    else:
        monkeypatch.setattr(setup, "_logon_sandbox_user", lambda password, **kwargs: not broken[0])

    def repair(**kwargs):
        assert kwargs.get("recreate_user", False) is False
        state[const.REG_VALUE_INSTALLED] = "1"
        broken[0] = False

    install = Mock(side_effect=repair)
    monkeypatch.setattr(setup, "install", install)
    setup.ensure_windows_setup()
    install.assert_called_once()
    assert install.call_args.kwargs["authorize_only"] is False


def test_box_validation_never_elevates(monkeypatch, state):
    state[const.REG_VALUE_INSTALLED] = ""
    install = Mock()
    monkeypatch.setattr(setup, "install", install)
    with pytest.raises(RuntimeError, match="需要管理员"):
        setup.ensure_windows_setup(allow_elevation=False)
    install.assert_not_called()


def test_failed_install_and_stale_marker_cannot_succeed(monkeypatch, state):
    monkeypatch.setattr(setup, "get_sandbox_user_password", lambda: None)
    monkeypatch.setattr(setup, "install", Mock())
    with pytest.raises(RuntimeError, match="凭据"):
        setup.ensure_windows_setup()


def test_new_acl_requests_elevation_for_original_user(monkeypatch, state, tmp_path):
    path = tmp_path / "external"
    path.mkdir()
    authorized = [False]
    monkeypatch.setattr(setup, "_can_write_dac", lambda path: authorized[0])
    install = Mock(side_effect=lambda **kwargs: authorized.__setitem__(0, True))
    monkeypatch.setattr(setup, "install", install)
    setup.ensure_windows_setup(acl_paths=[str(path)])
    assert install.call_args.kwargs["acl_paths"] == [setup._normalize_acl_path(str(path))]
    assert install.call_args.kwargs["authorize_only"] is True
    install.reset_mock()
    setup.ensure_windows_setup(acl_paths=[str(path)])
    install.assert_not_called()


def test_authorization_only_does_not_rebuild_account_or_network(monkeypatch, state, tmp_path):
    from jiuwenbox.supervisor import win_acl, win_wfp
    path = tmp_path / "external"
    path.mkdir()
    monkeypatch.setattr(setup, "_is_admin", lambda: True)
    grant = Mock()
    monkeypatch.setattr(win_acl, "grant_ace", grant)
    create_user = Mock(side_effect=AssertionError("account recreation"))
    network = Mock(side_effect=AssertionError("network rebuild"))
    monkeypatch.setattr(setup, "_create_sandbox_user", create_user)
    monkeypatch.setattr(win_wfp, "install_wfp_filters", network)
    monkeypatch.setattr(setup, "get_sandbox_user_sid", lambda: "S-1-sandbox")
    monkeypatch.setattr(setup, "_reg_set_str", state.__setitem__)
    monkeypatch.setattr(setup, "_record_acl_policy_paths", Mock())
    monkeypatch.setattr(setup, "_ensure_soft_delete_dll_acl", Mock())
    setup.install(authorize_only=True, acl_paths=[str(path)], host_user_sid="S-1-original-user")
    assert grant.call_args.args[1] == "S-1-original-user"
    create_user.assert_not_called()
    network.assert_not_called()


def test_ensure_cli_uses_explicit_merged_policy(monkeypatch, state, tmp_path):
    from jiuwenbox.server import policy_reader, shared_paths
    fs = SimpleNamespace(allow_read=[str(tmp_path)], deny_read=[], allow_write=[], deny_write=[])
    policy = SimpleNamespace(windows=SimpleNamespace(filesystem=fs, proxy=SimpleNamespace(
        port_range_start=60080, port_range_end=60089)))
    path = tmp_path / "runtime.yaml"
    reader = Mock(return_value=SimpleNamespace(load_policy=lambda: policy, policy_path=path))
    monkeypatch.setattr(policy_reader, "PolicyReader", reader)
    monkeypatch.setattr(setup, "collect_preinstall_paths", lambda policy: [])
    ensure = Mock()
    monkeypatch.setattr(setup, "ensure_windows_setup", ensure)
    # The CLI prepares shared ACLs with the same policy after setup. Keep this
    # unit test independent of the installed account and persistent host cache.
    prepare_shared = Mock()
    monkeypatch.setattr(shared_paths, "prepare_shared_tool_acl", prepare_shared)
    assert setup._main(["--ensure", "--policy-path", str(path)]) == 0
    reader.assert_called_once_with(policy_path=path)
    assert ensure.call_args.kwargs["acl_paths"] == [str(tmp_path)]
    prepare_shared.assert_called_once_with(policy)


def test_owned_tool_dir_uses_user_cache_without_hklm(monkeypatch, state, tmp_path):
    from jiuwenbox.supervisor import win_acl
    path = tmp_path / "python"
    path.mkdir()
    monkeypatch.setattr(setup, "_path_owned_by_current_user", lambda path: True)
    monkeypatch.setattr(setup, "get_sandbox_user_sid", lambda: "S-1-sandbox")
    monkeypatch.setattr(win_acl, "grant_ace", Mock())
    monkeypatch.setattr(setup, "_reg_set_str", Mock(side_effect=AssertionError("HKLM write")))
    install = Mock()
    monkeypatch.setattr(setup, "install", install)
    setup.ensure_windows_setup(preinstall_paths=[str(path)])
    setup.ensure_windows_setup(preinstall_paths=[str(path)])
    win_acl.grant_ace.assert_called_once()
    install.assert_not_called()
    assert json.loads((tmp_path / "preinstalled_paths.json").read_text()) == [str(path)]


def test_runtime_acl_missing_permission_fails_closed(monkeypatch, state, tmp_path):
    path = tmp_path / "external"
    path.mkdir()
    monkeypatch.setattr(setup, "_can_write_dac", lambda path: False)
    monkeypatch.setattr(setup, "_get_acl_policy_paths_recorded", lambda: {setup._normalize_acl_path(str(path))})
    install = Mock()
    monkeypatch.setattr(setup, "install", install)
    with pytest.raises(RuntimeError, match="补授权"):
        setup.ensure_acl_policy_paths_authorized([str(path)], allow_elevation=False)
    install.assert_not_called()


def test_missing_soft_delete_dll_acl_triggers_authorize_only(monkeypatch, state, tmp_path):
    path = tmp_path / "external"
    path.mkdir()
    monkeypatch.setattr(setup, "_is_admin", lambda: True)
    monkeypatch.setattr(setup, "soft_delete_dll_acl_missing", lambda: True)
    install = Mock()
    monkeypatch.setattr(setup, "install", install)
    setup.ensure_windows_setup(acl_paths=[str(path)])
    install.assert_called_once()
    assert install.call_args.kwargs["authorize_only"] is True
    assert install.call_args.kwargs["force"] is False


def test_soft_delete_acl_missing_without_elevation_only_warns(monkeypatch, state):
    """软删除补授权是 best-effort, 不能因此让沙箱起不来."""
    monkeypatch.setattr(setup, "soft_delete_dll_acl_missing", lambda: True)
    install = Mock()
    monkeypatch.setattr(setup, "install", install)
    setup.ensure_windows_setup(allow_elevation=False)
    install.assert_not_called()


def test_soft_delete_acl_elevation_failure_only_warns(monkeypatch, state):
    """仅因软删除触发提权时, 用户取消 UAC 不应中止沙箱启动."""
    monkeypatch.setattr(setup, "_is_admin", lambda: True)
    monkeypatch.setattr(setup, "soft_delete_dll_acl_missing", lambda: True)
    install = Mock(side_effect=RuntimeError("elevation cancelled"))
    monkeypatch.setattr(setup, "install", install)
    setup.ensure_windows_setup()
    install.assert_called_once()
