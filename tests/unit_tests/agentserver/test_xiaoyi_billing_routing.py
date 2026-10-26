# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""桌面续轮的回发寻址注入与计费归属单测。

背景：桌面端在手机发起的话题里续发消息时，agent_ws_server 会从会话级持久化
恢复 xiaoyi_* 回发寻址字段注入 request.metadata。这些字段只用于渠道回发，
不能当作“本轮来自 xiaoyi 渠道”的计费信号——否则后端会与桌面主进程重复上报
同一 interaction：先到的终态完成结算，后到的终态返回 consumedPoints=0，
表现为对话气泡的点数消耗与账单金额不一致。注入侧因此打上
ROUTING_INJECTED_METADATA_KEY，计费侧据此跳过。
"""

from __future__ import annotations

from types import SimpleNamespace

from jiuwenswarm.common.billing_client import ROUTING_INJECTED_METADATA_KEY
from jiuwenswarm.common.e2a.constants import (
    E2A_INTERNAL_ROUTING_INJECTED_KEY,
    E2A_WIRE_INTERNAL_METADATA_KEYS,
)
from jiuwenswarm.server import agent_ws_server
from jiuwenswarm.server.runtime.agent_adapter import interface_deep


def _invocation(core: str = "sess-1&abcd1234"):
    return SimpleNamespace(
        trace=SimpleNamespace(
            trace_id=core,
            conversation_id="sess-1",
            interaction_id="abcd1234",
        )
    )


class _RecordingReport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, query: str, trace_id: str) -> bool:
        self.calls.append((query, trace_id))
        return True


class TestBillingSkipsInjectedRouting:
    def test_injected_routing_metadata_skips_report(self, monkeypatch) -> None:
        """桌面续轮（注入标记）：后端不得再报 NEW（计费归桌面主进程）。"""
        report = _RecordingReport()
        monkeypatch.setattr(interface_deep, "billing_report_new", report)

        request = SimpleNamespace(
            channel_id="desktop",
            metadata={
                "xiaoyi_session_id": "conv_x",
                "xiaoyi_conversation_id": "conv_x",
                "xiaoyi_root_session_id": "root_x",
                ROUTING_INJECTED_METADATA_KEY: True,
            },
        )
        interface_deep._maybe_report_xiaoyi_billing_new(request, "你好", _invocation())

        assert report.calls == []

    def test_real_channel_still_reports(self, monkeypatch) -> None:
        """真实 xiaoyi 渠道（channel_id=xiaoyi）不受标记影响。"""
        report = _RecordingReport()
        monkeypatch.setattr(interface_deep, "billing_report_new", report)

        request = SimpleNamespace(channel_id="xiaoyi", metadata={})
        interface_deep._maybe_report_xiaoyi_billing_new(request, "你好", _invocation())

        assert report.calls == [("你好", "sess-1&abcd1234")]

    def test_real_inbound_with_task_id_still_reports(self, monkeypatch) -> None:
        """手机真实入站（自带 xiaoyi_task_id、无注入标记）仍由后端计费。"""
        report = _RecordingReport()
        monkeypatch.setattr(interface_deep, "billing_report_new", report)

        request = SimpleNamespace(
            channel_id="desktop",
            metadata={"xiaoyi_task_id": "task-1&2&53cc&0"},
        )
        interface_deep._maybe_report_xiaoyi_billing_new(request, "你好", _invocation())

        assert report.calls == [("你好", "sess-1&abcd1234")]


class TestRoutingMarkerTrustBoundary:
    def test_marker_is_wire_internal_and_matches_billing_key(self) -> None:
        """标记键必须被 E2A 入站过滤：客户端不能自带该键跳过后端计费。"""
        assert E2A_INTERNAL_ROUTING_INJECTED_KEY == ROUTING_INJECTED_METADATA_KEY
        assert ROUTING_INJECTED_METADATA_KEY in E2A_WIRE_INTERNAL_METADATA_KEYS


class TestInjectRoutingMarker:
    def test_existing_empty_keys_are_not_marked_as_injected(self, monkeypatch) -> None:
        """请求里已有同名键（空串）时不算注入，也不打计费跳过标记。"""
        from jiuwenswarm.server.runtime.session import session_metadata

        monkeypatch.setattr(
            session_metadata,
            "get_session_metadata",
            lambda _sid, cache_bust=False: {
                "xiaoyi_session_id": "conv_x",
                "xiaoyi_conversation_id": "conv_x",
                "xiaoyi_root_session_id": "root_x",
            },
        )
        request = SimpleNamespace(
            metadata={
                "xiaoyi_session_id": "",
                "xiaoyi_conversation_id": "",
                "xiaoyi_root_session_id": "",
            },
            session_id="desktop_1",
        )
        agent_ws_server._inject_session_xiaoyi_routing(request)

        assert request.metadata == {
            "xiaoyi_session_id": "",
            "xiaoyi_conversation_id": "",
            "xiaoyi_root_session_id": "",
        }

    def test_injection_sets_marker_and_skips_task_id(self, monkeypatch) -> None:
        """真正注入回发字段时打标记；xiaoyi_task_id 仍不注入。"""
        from jiuwenswarm.server.runtime.session import session_metadata

        monkeypatch.setattr(
            session_metadata,
            "get_session_metadata",
            lambda _sid, cache_bust=False: {
                "xiaoyi_session_id": "conv_x",
                "xiaoyi_conversation_id": "conv_x",
                "xiaoyi_root_session_id": "root_x",
                "xiaoyi_task_id": "task-1&2&53cc&0",
            },
        )
        request = SimpleNamespace(metadata={}, session_id="desktop_1")
        agent_ws_server._inject_session_xiaoyi_routing(request)

        assert request.metadata["xiaoyi_session_id"] == "conv_x"
        assert request.metadata["xiaoyi_conversation_id"] == "conv_x"
        assert request.metadata["xiaoyi_root_session_id"] == "root_x"
        assert "xiaoyi_task_id" not in request.metadata
        assert request.metadata[ROUTING_INJECTED_METADATA_KEY] is True

    def test_real_inbound_with_session_routing_not_marked(self, monkeypatch) -> None:
        """请求已带真实入站 xiaoyi_task_id：补路由字段但不打计费跳过标记（仍由后端计费）。"""
        from jiuwenswarm.server.runtime.session import session_metadata

        monkeypatch.setattr(
            session_metadata,
            "get_session_metadata",
            lambda _sid, cache_bust=False: {
                "xiaoyi_session_id": "conv_x",
                "xiaoyi_root_session_id": "root_x",
            },
        )
        request = SimpleNamespace(
            metadata={"xiaoyi_task_id": "task-1&2&53cc&0"},
            session_id="desktop_1",
        )
        agent_ws_server._inject_session_xiaoyi_routing(request)

        assert request.metadata["xiaoyi_session_id"] == "conv_x"
        assert ROUTING_INJECTED_METADATA_KEY not in request.metadata

    def test_request_owning_metadata_not_marked(self, monkeypatch) -> None:
        """请求自带 xiaoyi_session_id（真实入站）直接返回，不打标记。"""
        request = SimpleNamespace(
            metadata={"xiaoyi_session_id": "conv_own"},
            session_id="desktop_1",
        )
        agent_ws_server._inject_session_xiaoyi_routing(request)

        assert request.metadata == {"xiaoyi_session_id": "conv_own"}

    def test_no_session_metadata_not_marked(self, monkeypatch) -> None:
        """会话无持久化回发字段：不注入、不打标记。"""
        from jiuwenswarm.server.runtime.session import session_metadata

        monkeypatch.setattr(
            session_metadata,
            "get_session_metadata",
            lambda _sid, cache_bust=False: {"title": "t"},
        )
        request = SimpleNamespace(metadata={}, session_id="desktop_1")
        agent_ws_server._inject_session_xiaoyi_routing(request)

        assert request.metadata == {}


class TestTerminalStatus:
    def test_cancel_maps_to_failed(self, monkeypatch) -> None:
        """调用方按 failed=True 收口（用户取消）时终态为 FAILED，并透传本地会话/轮次 id。"""
        recorded: list[dict] = []

        def fake_terminal(
            trace_id,
            *,
            session_id,
            interaction_id,
            ok,
            local_session_id=None,
            local_run_id=None,
        ):
            recorded.append(
                {
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "interaction_id": interaction_id,
                    "ok": ok,
                    "local_session_id": local_session_id,
                    "local_run_id": local_run_id,
                }
            )
            return True

        monkeypatch.setattr(interface_deep, "billing_report_terminal", fake_terminal)

        interface_deep._report_xiaoyi_billing_terminal(
            _invocation(),
            failed=True,
            hitl_pending=False,
            local_session_id="desktop_1a176",
            local_run_id="50ec86ba&12&ae6e&0",
        )

        assert recorded == [
            {
                "trace_id": "sess-1&abcd1234",
                "session_id": "sess-1",
                "interaction_id": "abcd1234",
                "ok": False,
                "local_session_id": "desktop_1a176",
                "local_run_id": "50ec86ba&12&ae6e&0",
            }
        ]

    def test_hitl_pending_skips_terminal(self, monkeypatch) -> None:
        """HITL 挂起不算终态（续跑收口时才发）。"""
        report = _RecordingReport()
        monkeypatch.setattr(interface_deep, "billing_report_terminal", report)

        interface_deep._report_xiaoyi_billing_terminal(
            _invocation(), failed=True, hitl_pending=True
        )

        assert report.calls == []

    def test_terminal_failed_helper_covers_cancel_wiring(self) -> None:
        """收口判定纯函数：异常 / 取消 / 终端错误帧任一命中即 FAILED。"""
        assert interface_deep._billing_terminal_failed(billing_failed=False) is False
        assert (
            interface_deep._billing_terminal_failed(billing_failed=False, cancelled=True)
            is True
        )
        assert (
            interface_deep._billing_terminal_failed(billing_failed=False, error_text=True)
            is True
        )
        assert interface_deep._billing_terminal_failed(billing_failed=True) is True