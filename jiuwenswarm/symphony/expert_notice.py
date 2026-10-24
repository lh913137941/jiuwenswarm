# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Keep an expert no-plan notice in ordinary chat text, including HITL resumes."""

from dataclasses import dataclass, field
from typing import Any


NO_PLAN_MESSAGES = {
    "no_available_experts": (
        "现在还没有可邀请的专家。小艺Work先接着做；"
        "你也可以去「专家·专家团」看看。"
    ),
    "no_suitable_plan": (
        "这次没找到合适的专家。小艺Work先接着做；"
        "你也可以补充要求，或去「专家·专家团」看看。"
    ),
}


@dataclass
class ExpertNoticeState:
    """Owned by one DynamicExpertRound; never shared across sessions or turns."""

    text: str = ""
    tool_calls: set[str] = field(default_factory=set)


class ExpertNoticeStream:
    """Protect wire snapshots while persisting only this stream's new text.

    A tool/HITL boundary commits the visible prefix. Later model snapshots
    replace only the current segment. The desktop keeps one UI run on resume,
    whereas history uses a new request ID, so its carried prefix must not be
    written again under that ID.
    """

    def __init__(self, state: ExpertNoticeState) -> None:
        self.state = state
        self._history_prefix = state.text
        self._prefix = state.text
        self._segment = ""

    @property
    def active(self) -> bool:
        return bool(self.state.tool_calls)

    @property
    def text(self) -> str:
        return self._prefix + self._segment

    def history_content(self, content: str) -> str:
        if self._history_prefix and content.startswith(self._history_prefix):
            return content[len(self._history_prefix):]
        return content

    def consume(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Return a protected payload and, once per real call, a notice delta."""
        if payload.get("member_name"):
            return payload, ""
        event_type = payload.get("event_type")
        notice = self._notice(payload) if event_type == "chat.tool_result" else ""
        if notice:
            delta = ("\n\n" if self.text else "") + notice + "\n\n"
            self._prefix = self.text + delta
            self._segment = ""
            self.state.text = self._prefix
            return payload, delta
        if not self.active:
            return payload, ""
        content = payload.get("content")
        if event_type == "chat.delta" and isinstance(content, str):
            self._segment += content
        elif event_type == "chat.final":
            if isinstance(content, str) and content:
                self._segment = content.removeprefix(self._prefix)
            payload = {**payload, "content": self._prefix + self._segment}
        elif event_type in {"chat.tool_call", "chat.ask_user_question"}:
            self._prefix += self._segment
            self._segment = ""
        else:
            return payload, ""
        self.state.text = self.text
        return payload, ""

    def _notice(self, payload: dict[str, Any]) -> str:
        if (payload.get("tool_name") != "symphony_plan_experts"
                or payload.get("success") is False or payload.get("is_error") is True
                or payload.get("status") == "error"):
            return ""
        # StreamEventRail supplies the actual structured tool output. Do not
        # parse display strings or treat an exception as an empty plan.
        result = payload.get("raw_output")
        if not isinstance(result, dict) or result.get("success") is not True or result.get("status") != "no_plan":
            return ""
        if result.get("decision") == "summon":
            return ""
        call_id = payload.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id or call_id in self.state.tool_calls:
            return ""
        reason = result.get("reason")
        message = NO_PLAN_MESSAGES.get(reason) if isinstance(reason, str) else None
        if message:
            self.state.tool_calls.add(call_id)
        return message or ""
