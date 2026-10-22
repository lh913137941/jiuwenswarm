# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Rail import must not execute a caller-selected path."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from jiuwenswarm.agents.harness.common.plugins import rail_manager as rail_mod
from jiuwenswarm.agents.harness.common.plugins.rail_source_policy import (
    assert_local_extension_peer,
    validate_source,
)

_VALID_RAIL = '''
from openjiuwen.harness.rails import DeepAgentRail

class SampleRail(DeepAgentRail):
    """audit imported rail"""
    priority: int = 60
'''

_PACKAGE_INIT = ""
_PACKAGE_HELPER = "VALUE = 7\n"
_PACKAGE_RAIL = """
from openjiuwen.harness.rails import DeepAgentRail
from .helper import VALUE

class SampleRail(DeepAgentRail):
    marker = VALUE
"""


@pytest.fixture
def manager(monkeypatch, tmp_path):
    monkeypatch.delenv("JIUWENSWARM_RAIL_IMPORT_ROOTS", raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(rail_mod, "get_agent_workspace_dir", lambda: workspace)
    rail_mod.RailManager._instance = None
    instance = rail_mod.RailManager()
    try:
        yield instance
    finally:
        for name in list(getattr(instance, "_rail_finders", {})):
            instance._drop_loaded(name)
        rail_mod.RailManager._instance = None


def _write_rail(folder: Path, source: str, filename: str = "rail.py") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / filename).write_text(source, encoding="utf-8")


def test_import_rejects_path_outside_trusted_root(manager, tmp_path):
    marker = tmp_path / "executed.txt"
    outside = tmp_path / "evil_ext"
    _write_rail(
        outside,
        "from openjiuwen.harness.rails import DeepAgentRail\n"
        "class EvilRail(DeepAgentRail):\n"
        "    pass\n"
        f"open({str(marker)!r}, 'w', encoding='utf-8').write('executed')\n",
    )

    with pytest.raises(ValueError, match="受信任导入目录"):
        manager.import_extension(str(outside))

    assert not marker.exists()
    assert not (manager._extensions_dir / "evil_ext").exists()


def test_import_rejects_nested_folder_and_relative_path(manager, tmp_path):
    nested = tmp_path / "workspace" / "rail_imports" / "group" / "child"
    _write_rail(nested, _VALID_RAIL)
    with pytest.raises(ValueError, match="受信任导入目录"):
        manager.import_extension(str(nested))
    with pytest.raises(ValueError, match="受信任导入目录"):
        manager.import_extension("rail_imports/child")


def test_import_rejects_comment_only_and_process_call(manager, tmp_path):
    root = tmp_path / "workspace" / "rail_imports"
    comment = root / "comment_ext"
    _write_rail(comment, "# DeepAgentRail\n# AgentRail\ndef run():\n    return 1\n")
    with pytest.raises(ValueError, match="验证失败"):
        manager.import_extension(str(comment))
    assert not (manager._extensions_dir / "comment_ext").exists()

    marker = tmp_path / "executed.txt"
    dangerous = root / "evil_ext"
    _write_rail(
        dangerous,
        "from openjiuwen.harness.rails import DeepAgentRail\n"
        "import os\n"
        "class EvilRail(DeepAgentRail):\n"
        "    pass\n"
        f"open({str(marker)!r}, 'w', encoding='utf-8').write('executed')\n"
        "os.system('echo rejected')\n",
    )
    with pytest.raises(ValueError, match="验证失败"):
        manager.import_extension(str(dangerous))
    assert not marker.exists()
    assert not (manager._extensions_dir / "evil_ext").exists()


def test_validate_source_rejects_imported_process_helper():
    source = (
        "from openjiuwen.harness.rails import DeepAgentRail\n"
        "from os import system\n"
        "class EvilRail(DeepAgentRail):\n"
        "    pass\n"
        "system('echo rejected')\n"
    )
    with pytest.raises(ValueError, match="禁止的动态执行或进程调用"):
        validate_source(source, require_rail_class=True)


def test_import_accepts_direct_child_without_executing_it(manager, tmp_path):
    folder = tmp_path / "workspace" / "rail_imports" / "sample_ext"
    _write_rail(folder, _VALID_RAIL)
    imported = manager.import_extension(str(folder))
    assert imported["name"] == "sample_ext"
    assert imported["class_name"] == "SampleRail"
    assert imported["enabled"] is False
    assert imported["priority"] == 60
    installed = manager._extensions_dir / "sample_ext" / "rail.py"
    assert installed.is_file()
    assert "rail_extension_sample_ext" not in sys.modules


def test_load_executes_checked_source_and_rejects_later_tamper(manager, tmp_path):
    pytest.importorskip("openjiuwen.harness.rails")
    folder = tmp_path / "workspace" / "rail_imports" / "sample_ext"
    _write_rail(folder, _VALID_RAIL)
    manager.import_extension(str(folder))

    rail = manager.load_rail_instance_without_enabled_check("sample_ext")
    from openjiuwen.harness.rails import DeepAgentRail

    assert isinstance(rail, DeepAgentRail)

    marker = tmp_path / "executed.txt"
    installed = manager._extensions_dir / "sample_ext" / "rail.py"
    installed.write_text(
        "from openjiuwen.harness.rails import DeepAgentRail\n"
        "import os\n"
        "class SampleRail(DeepAgentRail):\n"
        "    pass\n"
        f"open({str(marker)!r}, 'w', encoding='utf-8').write('executed')\n"
        "os.system('echo rejected')\n",
        encoding="utf-8",
    )
    manager._rail_instances.pop("sample_ext", None)
    with pytest.raises(ValueError, match="验证失败"):
        manager.load_rail_instance_without_enabled_check("sample_ext")
    assert not marker.exists()


def test_package_import_uses_scanned_siblings(manager, tmp_path):
    pytest.importorskip("openjiuwen.harness.rails")
    folder = tmp_path / "workspace" / "rail_imports" / "pack_ext"
    _write_rail(folder, _PACKAGE_INIT, "__init__.py")
    _write_rail(folder, _PACKAGE_HELPER, "helper.py")
    _write_rail(folder, _PACKAGE_RAIL, "rail.py")
    manager.import_extension(str(folder))
    rail = manager.load_rail_instance_without_enabled_check("pack_ext")
    assert rail.marker == 7


def test_symlink_source_is_rejected(manager, tmp_path):
    outside = tmp_path / "outside"
    _write_rail(outside, _VALID_RAIL)
    link = tmp_path / "workspace" / "rail_imports" / "linked_ext"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        if sys.platform != "win32":
            pytest.skip("symlinks require privilege")
        import subprocess

        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            capture_output=True,
            text=True,
        )
        if created.returncode != 0:
            pytest.skip("symlinks require privilege")
    with pytest.raises(ValueError, match="受信任导入目录"):
        manager.import_extension(str(link))
    assert not (manager._extensions_dir / "linked_ext").exists()


def test_config_cannot_enable_or_escape_with_a_bad_name(manager, tmp_path):
    canary = tmp_path / "canary"
    canary.mkdir()
    (canary / "keep.txt").write_text("keep", encoding="utf-8")
    config = {
        "../evil": {
            "name": "../evil",
            "class_name": "EvilRail",
            "enabled": True,
        },
        "sample_ext": {
            "name": "sample_ext",
            "class_name": "SampleRail",
            "enabled": "true",
        },
    }
    manager._config_file.write_text(json.dumps(config), encoding="utf-8")
    manager._load_config()
    assert "../evil" not in manager._extensions
    assert manager._extensions["sample_ext"].enabled is False
    assert (canary / "keep.txt").read_text(encoding="utf-8") == "keep"
    with pytest.raises(ValueError, match="扩展名称无效"):
        manager.delete_extension("../evil")
    assert (canary / "keep.txt").exists()
    with pytest.raises(ValueError, match="布尔值"):
        manager.toggle_extension("sample_ext", "true")


def test_extension_mutation_peer_is_local_only():
    with pytest.raises(PermissionError, match="本机连接"):
        assert_local_extension_peer(SimpleNamespace(remote_address=("10.1.2.3", 9)))
    browser = SimpleNamespace(
        remote_address=("127.0.0.1", 9),
        request_headers={"Origin": "https://attacker.example"},
    )
    with pytest.raises(PermissionError, match="本机连接"):
        assert_local_extension_peer(browser)
    assert_local_extension_peer(SimpleNamespace(remote_address=("127.0.0.1", 9)))
    gateway = SimpleNamespace(
        remote_address=("127.0.0.1", 9),
        request_headers={"Origin": "http://127.0.0.1:18592"},
    )
    assert_local_extension_peer(gateway)
    assert_local_extension_peer(SimpleNamespace(remote_address="stdio:desktop"))
