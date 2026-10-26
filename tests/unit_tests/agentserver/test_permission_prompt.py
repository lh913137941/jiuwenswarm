# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Approval copy preserves command meaning and one approval for the full chain."""

from dataclasses import replace

import pytest

from openjiuwen.harness.security.permission_engine.approve.ask_presentation import (
    build_permission_ask_presentation,
)
from openjiuwen.harness.security.permission_engine.models import (
    PermissionLevel,
    PermissionResult,
)

from jiuwenswarm.agents.harness.common.rails.interrupt.permission_prompt import (
    _command_lines,
    build_permission_prompt,
)


@pytest.mark.parametrize(
    ("command", "shell_type", "expected"),
    [
        ("cd .. && dir", "cmd", ["cd .. &&", "dir"]),
        ("echo a&&echo b||echo c", "cmd", ["echo a&&", "echo b||", "echo c"]),
        ('echo "a && b" && dir', "cmd", ['echo "a && b" &&', "dir"]),
        ("echo 'a; b' ; pwd", "bash", ["echo 'a; b' ;", "pwd"]),
        ("echo a\\;b; pwd", "bash", ["echo a\\;b;", "pwd"]),
        ("echo a ^& b & dir", "cmd", ["echo a ^& b &", "dir"]),
        (
            "Write-Output a`;b; Get-Date",
            "powershell",
            ["Write-Output a`;b;", "Get-Date"],
        ),
        (
            "Write-Output 'it''s; fine'; Get-Date",
            "powershell",
            ["Write-Output 'it''s; fine';", "Get-Date"],
        ),
        ("dir 2>&1 | findstr txt", "cmd", ["dir 2>&1 |", "findstr txt"]),
        (
            "echo a &> out.txt && cat out.txt",
            "bash",
            ["echo a &> out.txt &&", "cat out.txt"],
        ),
        ("& python test.py; Get-Date", "powershell", ["& python test.py;", "Get-Date"]),
        ("echo a\r\necho b", "cmd", ["echo a", "echo b"]),
        ("echo a;b && dir", "cmd", ["echo a;b &&", "dir"]),
        ("echo a ^\r\nb && dir", "cmd", ["echo a ^\r\nb &&", "dir"]),
    ],
)
def test_command_display_keeps_operators_and_quoted_arguments(
    command, shell_type, expected
):
    assert _command_lines(command, shell_type) == expected


@pytest.mark.parametrize(
    "command",
    [
        'echo "unterminated && dir',
        "if test -f a; then cat a; fi",
        "echo $(echo a && echo b) && pwd",
        "cat <<EOF\na && b\nEOF",
        "echo a # comment && pwd",
        "Get-ChildItem | ForEach-Object { Write-Output $_; Get-Date }",
    ],
)
def test_complex_scripts_keep_complete_original_text(command):
    assert _command_lines(command, "bash") == [command]


def test_shell_prompt_retains_risk_and_hint_without_mutating_arguments():
    args = {
        "command": "cd .. && dir",
        "workdir": "C:/工作目录",
        "shell_type": "cmd",
        "description": "查看上级目录",
    }
    message = "read C:/\n\n> 默认仅授权当前对象；命令始终按原始命令授权。"
    title, body = build_permission_prompt(
        "mcp_exec_command",
        args,
        message,
        {
            "ask_title": "检测到受保护的文件路径访问，需要确认后才能执行",
        },
    )

    assert title == "助手想要在你的设备上运行程序指令"
    assert "2 条指令，本次授权适用于整组指令" in body
    assert "cd .. &&\ndir" in body
    assert "工具 `mcp_exec_command` 需要授权才能执行：查看上级目录" in body
    assert "C:/工作目录" in body
    assert "读取 C:/" in body
    assert "命令始终按原始命令授权" in body
    assert "授权原因：检测到受保护的文件路径访问" in body
    assert args["command"] == "cd .. && dir"


def test_command_markdown_fences_remain_literal():
    command = 'echo "```" && echo done'
    _, body = build_permission_prompt(
        "bash", {"command": command, "shell_type": "bash"}, ""
    )
    assert '````text\necho "```" &&\necho done\n````' in body


def test_bash_named_tool_honors_explicit_cmd_shell():
    _, body = build_permission_prompt(
        "bash", {"command": "echo a;b && dir", "shell_type": "cmd"}, ""
    )
    assert "2 条指令" in body
    assert "echo a;b &&\ndir" in body


def test_shell_prompt_uses_captured_workdir_without_changing_tool_arguments():
    args = {"command": "chmod -R a-w relative.txt && echo done", "cwd": "ignored-model-cwd"}
    original = dict(args)
    _, body = build_permission_prompt("bash", args, "需要授权", {"workdir": "C:/审批目录"})
    assert body.count("工作目录：") == 1
    assert "C:/审批目录" in body
    assert "ignored-model-cwd" not in body
    assert args == original


@pytest.mark.parametrize(
    ("tool", "title"),
    [
        ("read_file", "助手想要读取你设备上的文件"),
        ("write_file", "助手想要修改你设备上的文件"),
        ("core.powershell", "助手想要在你的设备上运行程序指令"),
        ("send_file_to_user", "助手想要给你发送一个文件"),
        ("memory_get", "助手想要读取你的记忆内容"),
        ("mcp_free_search", "助手想要检索网络或本地信息"),
        ("fetch_webpage", "助手想要获取网页内容"),
        ("unknown_tool", "助手想要执行一项需要你授权的操作"),
    ],
)
def test_permission_intent_matches_tool_action(tool, title):
    assert build_permission_prompt(tool, {}, "需要授权")[0] == title


def test_old_message_intent_and_risk_level_are_preserved():
    message = "**助手想要读取指定的报告**\n\n**风险等级：高风险**"
    title, body = build_permission_prompt("read_file", {}, message)
    assert title == "助手想要读取指定的报告"
    assert "**风险等级：高风险**" in body
    assert "工具 `read_file` 需要授权才能执行" in body


def test_core_captured_file_list_and_risk_are_used_once():
    message = (
        "**助手想要在你的设备上运行程序指令**\n\n"
        "工具 `mcp_exec_command` 需要授权才能执行：复制报告\n\n"
        "涉及的文件操作：\n\n- 读取 C:/资料/报告.txt\n- 写入 D:/备份/报告.txt\n\n"
        "**风险等级：高风险**\n\n"
        "> 默认仅授权当前对象；会话内记住仅对本会话有效，永久记住写入配置。"
    )
    metadata = {
        "ask_title": "检测到受保护的文件路径访问，需要确认后才能执行",
        "ask_risk_level": "高",
        "file_operations": [
            {"path": "C:/资料/报告.txt", "action": "read", "source": "shlex"},
            {"path": "D:/备份/报告.txt", "action": "write", "source": "shlex"},
        ],
    }
    args = {
        "command": "copy ./报告.txt ./备份.txt && echo done",
        "shell_type": "cmd",
        "description": "复制报告",
    }
    title, body = build_permission_prompt("mcp_exec_command", args, message, metadata)
    assert title == "助手想要在你的设备上运行程序指令"
    assert (
        body.index("助手想要")
        < body.index("工具 `")
        < body.index("copy ./")
        < body.index("涉及的文件操作")
        < body.index("风险等级")
    )
    assert body.count("涉及的文件操作") == 1
    assert body.count("风险等级") == 1
    assert "- 读取 C:/资料/报告.txt" in body and "- 写入 D:/备份/报告.txt" in body
    assert "**风险等级：高风险**" in body
    assert "默认仅授权当前对象" in body
    assert args["command"] == "copy ./报告.txt ./备份.txt && echo done"


@pytest.mark.parametrize(
    ("action", "label"),
    [
        ("read", "读取"),
        ("write", "写入"),
        ("exec", "执行"),
        ("access", "访问"),
        ("edit", "写入"),
        ("list", "读取"),
        ("delete", "删除"),
    ],
)
def test_file_actions_have_identical_display_with_and_without_metadata(
    monkeypatch, action, label
):
    presentation = replace(
        build_permission_ask_presentation(
            "custom_tool", {}, PermissionResult(PermissionLevel.ASK)
        ),
        file_operations=(("C:/资料/报告.txt", action, "tool_arg"),),
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.common.rails.interrupt.permission_prompt."
        "build_permission_ask_presentation",
        lambda *args, **kwargs: presentation,
    )
    uncaptured = build_permission_prompt("custom_tool", {}, "")
    captured = build_permission_prompt(
        "custom_tool",
        {},
        "",
        {
            "file_operations": [
                {"path": "C:/资料/报告.txt", "action": action, "source": "tool_arg"}
            ]
        },
    )
    assert captured == uncaptured
    assert f"- {label} C:/资料/报告.txt" in captured[1]


def test_legacy_access_summary_uses_chinese_label():
    _, body = build_permission_prompt("custom_tool", {}, "access C:/资料/报告.txt")
    assert "访问 C:/资料/报告.txt" in body


def test_batch_delete_card_shows_count_reason_and_delete_action():
    message = (
        "**助手想要在你的设备上运行程序指令**\n\n"
        "工具 `bash` 需要授权才能执行\n\n"
        "**风险等级：高风险**\n\n"
        "> 默认仅授权当前对象；会话内记住仅对本会话有效，永久记住写入配置。"
    )
    metadata = {
        "matched_rule": "tools.bash|del_guard:batch_delete",
        "ask_title": "检测到批量删除，需要确认后才能执行",
        "ask_reason": "本次操作将删除 5 个文件，达到阈值 2，需要确认",
        "ask_risk_level": "高",
        "file_operations": [
            {"path": "C:/tmp/bd10_0.txt", "action": "delete", "source": "shlex"},
            {"path": "C:/tmp/bd10_1.txt", "action": "delete", "source": "shlex"},
        ],
    }
    title, body = build_permission_prompt(
        "bash",
        {"command": "rm -f C:/tmp/bd10_0.txt C:/tmp/bd10_1.txt", "shell_type": "bash"},
        message,
        metadata,
    )
    assert title == "检测到批量删除，需要确认后才能执行"
    assert body.startswith("**检测到批量删除，需要确认后才能执行**")
    assert "授权原因：本次操作将删除 5 个文件，达到阈值 2，需要确认" in body
    assert "- 删除 C:/tmp/bd10\\_0.txt" in body
    assert "- 删除 C:/tmp/bd10\\_1.txt" in body
    assert "- 写入 " not in body
    assert "授权说明" not in body
    assert "默认仅授权当前对象" in body


def test_batch_delete_card_keeps_write_actions():
    metadata = {
        "matched_rule": "tools.bash|del_guard:batch_delete",
        "ask_title": "检测到批量删除，需要确认后才能执行",
        "ask_reason": "本次操作将删除 1 个文件，达到阈值 1，需要确认",
        "ask_risk_level": "高",
        "file_operations": [
            {"path": "C:/tmp/a.txt", "action": "delete", "source": "shlex"},
            {"path": "C:/tmp/b.txt", "action": "write", "source": "shlex"},
        ],
    }
    _, body = build_permission_prompt(
        "bash",
        {"command": "rm -f C:/tmp/a.txt && echo x > C:/tmp/b.txt", "shell_type": "bash"},
        "",
        metadata,
    )
    assert "- 删除 C:/tmp/a.txt" in body
    assert "- 写入 C:/tmp/b.txt" in body


def test_batch_delete_card_shows_reason_without_keyword_match():
    metadata = {
        "matched_rule": "tools.bash|del_guard:batch_delete",
        "ask_title": "检测到批量删除，需要确认后才能执行",
        "ask_reason": "批量删除文件数量已达上限，需要确认",
        "ask_risk_level": "高",
        "file_operations": [
            {"path": "C:/tmp/a.txt", "action": "delete", "source": "shlex"},
        ],
    }
    title, body = build_permission_prompt(
        "bash",
        {"command": "rm -f C:/tmp/a.txt", "shell_type": "bash"},
        "",
        metadata,
    )
    assert title == "检测到批量删除，需要确认后才能执行"
    assert "授权原因：批量删除文件数量已达上限，需要确认" in body
    assert "授权说明" not in body
