# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Identify connector usage from command positions, never argument text."""

import json
from pathlib import Path

import pytest

from jiuwenswarm.common import connectors


@pytest.fixture
def connectors_dir(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.delenv(connectors.CLI_CONFIG_ENV, raising=False)
    (tmp_path / connectors.CLI_CONFIG_FILENAME).write_text(
        json.dumps({"connectors": [{"id": "dingtalk", "commands": ["dws-cli"]}]}),
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("dws calendar list", ["dingtalk"]),
        ("dws message send --text hello", ["dingtalk"]),
        ("DWS.EXE calendar list", ["dingtalk"]),
        ('& "C:\\Program Files\\connectors\\dws.exe" calendar list', ["dingtalk"]),
        ('"/opt/connectors/dws" calendar list', ["dingtalk"]),
        ("dws-cli calendar list", ["dingtalk"]),
        ("lark-cli docs list", ["feishu"]),
        ("dws calendar list && dws message list; lark-cli docs list", ["dingtalk", "feishu"]),
        ("echo dws; lark-cli docs list", ["feishu"]),
        ("echo hello;\nlark-cli docs list", ["feishu"]),
        ("echo hello # dws; lark-cli docs list\nwecom-cli contacts list", ["wecom"]),
        ("echo dws", []),
        ('echo "dws; lark-cli docs list"', []),
        ('echo ";" dws', []),
        ("echo hello > dws", []),
        ('python -c "print(\'dws\')"', []),
        ("python custom_script.py", []),
        ('dws --title "unterminated', []),
        ("", []),
    ],
)
def test_connector_calls_only_include_explicit_command_invocations(connectors_dir, command, expected):
    calls = connectors.connector_calls_for_command(command, connectors_dir=connectors_dir)
    assert [call["resource_id"] for call in calls] == expected
    assert all(call["resource_type"] == "connector" and call["resource_name"] for call in calls)


def test_connector_calls_use_new_registration_without_python_changes(connectors_dir):
    config = connectors_dir / connectors.CLI_CONFIG_FILENAME
    config.write_text(
        json.dumps({"connectors": [{"id": "new-service", "commands": ["new-cli"]}]}),
        encoding="utf-8",
    )

    assert connectors.connector_calls_for_command(
        "new-cli search hello", connectors_dir=connectors_dir
    ) == [{"resource_type": "connector", "resource_id": "new-service", "resource_name": "new-cli"}]


def test_cli_allowlist_without_connector_id_still_reports_its_name(connectors_dir):
    (connectors_dir / connectors.CLI_CONFIG_FILENAME).write_text(
        json.dumps({"commands": ["loose-cli"]}), encoding="utf-8"
    )

    assert connectors.connector_calls_for_command(
        "loose-cli search hello", connectors_dir=connectors_dir
    ) == [{"resource_type": "connector", "resource_id": "", "resource_name": "loose-cli"}]


def test_cli_names_file_without_id_still_reports_usage(connectors_dir):
    directory = connectors_dir / "custom"
    directory.mkdir()
    (directory / connectors.CLI_NAMES_FILENAME).write_text('["custom-cli"]', encoding="utf-8")

    assert connectors.connector_calls_for_command(
        "custom-cli search", connectors_dir=connectors_dir
    ) == [{"resource_type": "connector", "resource_id": "", "resource_name": "custom-cli"}]


def test_aliases_share_identity_but_keep_the_invoked_cli_name(connectors_dir):
    assert connectors.connector_calls_for_command(
        "dws calendar list; dws-cli message send", connectors_dir=connectors_dir
    ) == [{"resource_type": "connector", "resource_id": "dingtalk", "resource_name": "dws"}]
