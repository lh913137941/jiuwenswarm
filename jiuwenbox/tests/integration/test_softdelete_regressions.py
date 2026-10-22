# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Real DLL/IPC regressions, isolated in child processes; no sandbox server needed."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from jiuwenbox.supervisor import win_recycle_broker as recycle

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows native DLL")


@pytest.fixture
def dll():
    path = Path(os.environ.get("JIUWENBOX_SOFT_DELETE_DLL") or (
        Path(recycle.__file__).resolve().parents[1] / "native/jiuwen_softdelete.dll"
    ))
    assert path.is_file(), "build the soft-delete DLL before testing"
    return path.resolve()


@pytest.fixture(scope="module")
def native_host(tmp_path_factory):
    supplied = os.environ.get("JIUWENBOX_SOFT_DELETE_TEST_HOST")
    if supplied:
        assert Path(supplied).is_file()
        return supplied
    if not shutil.which("cl"):
        pytest.skip("requires cl or JIUWENBOX_SOFT_DELETE_TEST_HOST (softdelete_host.c)")
    folder = tmp_path_factory.mktemp("softdelete-native")
    result = subprocess.run(
        ["cl", "/nologo", "/utf-8", "/std:c17", str(Path(__file__).with_name("softdelete_host.c")),
         f"/Fe:{folder / 'softdelete_host.exe'}", f"/Fo:{folder / 'softdelete_host.obj'}"],
        cwd=folder, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return str(folder / "softdelete_host.exe")


@pytest.fixture
def broker(tmp_path, monkeypatch, dll):
    archive = tmp_path / "archive"
    monkeypatch.setenv(recycle.ENV_ARCHIVE, str(archive))
    requests = []

    def archive_request(path):
        requests.append(path)
        # Force an archive for short-path env tests, without altering native IPC.
        return recycle._move_to_archive(path)

    server = recycle.RecycleBroker(archive_request)
    server.start()
    token = server.register([str(tmp_path)])
    env = {**os.environ, "JIUWENBOX_SOFT_DELETE": "1",
           "JIUWENBOX_SOFT_DELETE_DLL": str(dll),
           "JIUWENBOX_SOFT_DELETE_RECYCLE_PORT": str(server.port),
           "JIUWENBOX_SOFT_DELETE_RECYCLE_TOKEN": token,
           "SOFTDELETE_PARENT_ONLY": "must not leak"}
    try:
        yield server, env, archive, requests
    finally:
        server.stop()


def run_child(args, env):
    return subprocess.run(args, env=env, capture_output=True, text=True, timeout=30,
                          creationflags=subprocess.CREATE_NO_WINDOW)


@pytest.mark.parametrize("hooked", [False, True], ids=["native-control", "soft-delete"])
@pytest.mark.parametrize("delegated", [True, False], ids=["broker", "local"])
def test_long_path_disposition_without_long_path_manifest(
    tmp_path, dll, native_host, broker, hooked, delegated,
):
    server, env, archive, requests = broker
    if not delegated:
        env.pop("JIUWENBOX_SOFT_DELETE_RECYCLE_PORT", None)
        env.pop("JIUWENBOX_SOFT_DELETE_RECYCLE_TOKEN", None)
    # Exercise the real long-path Shell skip and archive fallback.
    server._recycler = recycle.recycle_with_shell
    deep = tmp_path / ("a" * 90) / ("b" * 90) / ("c" * 90)
    extended = "\\\\?\\" + str(deep)
    os.makedirs(extended)
    victim = extended + "\\payload.txt"
    Path(victim).write_text("long-path payload")
    result = run_child([native_host, str(dll) if hooked else "none", victim], env)
    assert "bare_attrs=4294967295" in result.stdout, "test host must not be longPathAware"
    assert result.returncode == 0, result.stdout + result.stderr
    assert not os.path.exists(victim)
    files = list(archive.glob("payload.txt_*"))
    if hooked:
        assert len(files) == 1
        assert files[0].read_text() == "long-path payload"
    else:
        assert not files


@pytest.mark.parametrize("route", ["direct", "bin", "usr/bin"])
@pytest.mark.parametrize("mode", ["inherit", "minimal", "empty", "explicit"])
def test_custom_environment_delete(tmp_path, dll, broker, route, mode):
    _server, env, archive, requests = broker
    victim = tmp_path / "payload.txt"
    victim.write_text("environment payload")
    observed = tmp_path / "environment.json"
    inner = tmp_path / "inner.py"
    inner.write_text(
        "import json,os,pathlib\n"
        f"pathlib.Path({str(observed)!r}).write_text(json.dumps(dict(os.environ)))\n"
        f"os.remove({str(victim)!r})\n", encoding="utf-8",
    )
    custom = "{'FOO':'bar'}"
    if mode == "inherit":
        custom = "None"
    elif mode == "empty":
        custom = "{}"
    elif mode == "explicit":
        custom = "{'FOO':'bar','sYsTeMrOoT':os.environ['SystemRoot'].lower()}"
    outer = tmp_path / "outer.py"
    code = f"import ctypes,os,subprocess,sys\nctypes.WinDLL({str(dll)!r})\n"
    code += (f"sys.exit(subprocess.run([sys.executable,{str(inner)!r}],"
             f"env={custom},timeout=20).returncode)\n")
    outer.write_text(code, encoding="utf-8")
    if route == "direct":
        args = [sys.executable, str(outer)]
    else:
        git = shutil.which("git")
        if not git:
            pytest.skip("requires Git Bash")
        bash = Path(git).parent.parent / route / "bash.exe"
        if not bash.is_file():
            pytest.skip(f"requires {bash}")
        command = f"{shlex.quote(Path(sys.executable).as_posix())} {shlex.quote(outer.as_posix())}"
        host = tmp_path / "host.py"
        host.write_text(
            f"import ctypes,subprocess,sys\nctypes.WinDLL({str(dll)!r})\n"
            f"sys.exit(subprocess.run([{str(bash)!r},'-c',{command!r}],timeout=25).returncode)\n",
            encoding="utf-8",
        )
        args = [sys.executable, str(host)]
    result = run_child(args, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not victim.exists()
    assert requests == [str(victim)]
    files = list(archive.glob("payload.txt_*"))
    assert len(files) == 1 and files[0].read_text() == "environment payload"
    child_env = json.loads(observed.read_text())
    if mode != "inherit":
        assert "SOFTDELETE_PARENT_ONLY" not in child_env
        assert child_env.get("FOO") == (None if mode == "empty" else "bar")
    assert child_env["SYSTEMROOT"].casefold() == os.environ["SystemRoot"].casefold()
    if mode == "explicit":
        assert child_env["SYSTEMROOT"] == os.environ["SystemRoot"].lower()


@pytest.mark.parametrize("entry", ["os.remove", "DeleteFileW"])
def test_long_path_python_delete(tmp_path, dll, broker, entry):
    server, env, archive, _requests = broker
    server._recycler = recycle.recycle_with_shell
    deep = tmp_path.joinpath(*["d" * 60] * 5)
    victim = "\\\\?\\" + str(deep / ("f" * 200 + ".txt"))
    os.makedirs("\\\\?\\" + str(deep))
    Path(victim).write_text("extended-path payload")
    code = f"import ctypes,os\nctypes.WinDLL({str(dll)!r})\n"
    if entry == "os.remove":
        code += f"os.remove({victim!r})\n"
    else:
        code += ("kernel=ctypes.WinDLL('kernel32',use_last_error=True)\n"
                 "kernel.DeleteFileW.argtypes=[ctypes.c_wchar_p]\n"
                 f"if not kernel.DeleteFileW({victim!r}): raise ctypes.WinError(ctypes.get_last_error())\n")
    python = os.environ.get("JIUWENBOX_SOFT_DELETE_TEST_PYTHON", sys.executable)
    result = run_child([python, "-c", code], env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not os.path.exists(victim)
    files = list(archive.iterdir())
    assert len(files) == 1 and files[0].read_text() == "extended-path payload"


@pytest.mark.parametrize("mode", ["ansi", "ansi-empty"])
def test_ansi_environment_delete(tmp_path, dll, native_host, broker, mode):
    _server, env, archive, requests = broker
    victim = tmp_path / "ansi.txt"
    victim.write_text("ANSI payload")
    script = tmp_path / "delete.py"
    script.write_text(f"import os\nos.remove({str(victim)!r})\n", encoding="utf-8")
    command = subprocess.list2cmdline([sys.executable, str(script)])
    result = run_child([native_host, str(dll), mode, command], env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not victim.exists() and requests == [str(victim)]
    files = list(archive.glob("ansi.txt_*"))
    assert len(files) == 1 and files[0].read_text() == "ANSI payload"


@pytest.mark.parametrize("child_env", ["None", "dict(FOO='bar')", "dict(os.environ)"],
                         ids=["inherited", "minimal-env", "complete-env"])
def test_local_long_path_delete_without_broker(tmp_path, dll, child_env):
    archive = tmp_path / "archive"
    deep = tmp_path.joinpath(*["d" * 60] * 5)
    victim = "\\\\?\\" + str(deep / ("f" * 200 + ".txt"))
    os.makedirs("\\\\?\\" + str(deep))
    Path(victim).write_text("local extended-path payload")
    env = {**os.environ, "JIUWENBOX_SOFT_DELETE": "1",
           "JIUWENBOX_SOFT_DELETE_ARCHIVE": str(archive),
           "JIUWENBOX_SOFT_DELETE_DLL": str(dll)}
    env.pop("JIUWENBOX_SOFT_DELETE_RECYCLE_PORT", None)
    env.pop("JIUWENBOX_SOFT_DELETE_RECYCLE_TOKEN", None)
    python = os.environ.get("JIUWENBOX_SOFT_DELETE_TEST_PYTHON", sys.executable)
    inner = ("import os\n"
             "assert os.environ.get('SystemRoot')\n"
             "assert os.environ.get('JIUWENBOX_SOFT_DELETE_ARCHIVE')\n"
             f"os.remove({victim!r})\n")
    code = ("import ctypes,os,subprocess,sys\n"
            f"ctypes.WinDLL({str(dll)!r})\n"
            f"result=subprocess.run([sys.executable,'-c',{inner!r}],"
            f"env={child_env},timeout=20)\n"
            "sys.exit(result.returncode)\n")
    result = run_child([python, "-c", code], env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not os.path.exists(victim)
    files = list(archive.iterdir())
    assert len(files) == 1 and files[0].read_text() == "local extended-path payload"


def test_invalid_token_keeps_source(tmp_path, dll, broker):
    _server, env, archive, requests = broker
    victim = tmp_path / "keep.txt"
    victim.write_text("keep me")
    env["JIUWENBOX_SOFT_DELETE_RECYCLE_TOKEN"] = "invalid-test-token"
    code = f"import ctypes,os\nctypes.WinDLL({str(dll)!r})\nos.remove({str(victim)!r})"
    result = run_child([sys.executable, "-c", code], env)
    assert result.returncode != 0 and "PermissionError" in result.stderr
    assert victim.read_text() == "keep me"
    assert not requests and not archive.exists()


def _win_to_msys(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = str(resolved)[len(resolved.drive):].replace("\\", "/")
    return f"/{drive}{rest}"


@pytest.mark.xfail(
    strict=False,
    reason="Git Bash rm spawns rm.exe but debug inject does not delegate to broker yet",
)
def test_bash_rm_delegates_to_recycle_broker(tmp_path, dll, broker):
    """Regression guard: bash rm must hit the same broker path as python os.remove."""
    _server, env, archive, requests = broker
    victim = tmp_path / "bash-rm.txt"
    victim.write_text("bash rm payload")
    msys_path = _win_to_msys(victim)
    git = shutil.which("git")
    if not git:
        pytest.skip("requires Git Bash")
    bash = Path(git).parent.parent / "usr" / "bin" / "bash.exe"
    if not bash.is_file():
        pytest.skip(f"requires {bash}")
    host = tmp_path / "host.py"
    host.write_text(
        "import ctypes,subprocess,sys\n"
        f"ctypes.WinDLL({str(dll)!r})\n"
        f"bash={str(bash)!r}\n"
        f"cmd=f'rm -f {msys_path!r}'\n"
        "sys.exit(subprocess.run([bash,'-lc',cmd],timeout=30).returncode)\n",
        encoding="utf-8",
    )
    result = run_child([sys.executable, str(host)], env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not victim.exists()
    assert requests == [str(victim)]
    files = list(archive.glob("bash-rm.txt_*"))
    assert len(files) == 1 and files[0].read_text() == "bash rm payload"
