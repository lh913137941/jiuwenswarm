"""Explicit, task-scoped confirmation for dynamic expert plans and no-plan choices."""

from __future__ import annotations

from typing import Any, Callable

from openjiuwen.core.foundation.llm.schema.tool_call import ToolCall
from openjiuwen.core.session.interaction.interactive_input import InteractiveInput
from openjiuwen.core.single_agent.interrupt.response import InterruptRequest
from openjiuwen.core.single_agent.interrupt.state import INTERRUPTION_KEY, RESUME_USER_INPUT_KEY
from openjiuwen.core.single_agent.rail.base import AgentCallbackContext
from openjiuwen.harness.rails.interrupt.confirm_rail import ConfirmInterruptRail, ConfirmPayload
from openjiuwen.harness.rails.interrupt.interrupt_base import InterruptDecision, RejectResult

from jiuwenswarm.symphony.expert_team import (
    CONFIRMATION_NONCE_KEY, CONFIRMATION_STATE_KEY, EXPERT_PLAN_ACTION_KEY,
    DynamicExpertRail, DynamicExpertRound, PLAN_TOOL_NAME, _in_plan_mode,
)

_PLAN_REQUEST_PREFIX = "symphony-plan:"
_NO_PLAN_REQUEST_PREFIX = "symphony-no-plan:"
_NO_PLAN_TITLE = "这次没找到合适的专家"
_NO_PLAN_MESSAGE = "当前可用的专家还不多，去「专家·专家团」召唤更多专家吧。"


def confirmation_request_id(metadata: dict[str, Any]) -> str:
    """A provider may reuse call IDs; each displayed plan needs its own answer ID."""
    plan_id = metadata.get("dynamic_expert_plan_id")
    if isinstance(plan_id, str) and plan_id:
        return f"{_PLAN_REQUEST_PREFIX}{plan_id}"
    no_plan_id = metadata.get("dynamic_expert_no_plan_id")
    return f"{_NO_PLAN_REQUEST_PREFIX}{no_plan_id}" if isinstance(no_plan_id, str) and no_plan_id else ""


def pending_confirmation(session: Any, request_id: str) -> tuple[str, str, dict[str, Any]] | None:
    """Resolve a displayed confirmation to the still-pending Core interrupt ID."""
    state = session.get_state(INTERRUPTION_KEY) if session is not None else None
    entries = getattr(state, "interrupted_tools", {})
    if not isinstance(entries, dict):
        return None
    for entry in entries.values():
        for core_request_id, request in entry.interrupt_requests.items():
            tool_name = entry.tool_call.name
            metadata = dict(request.metadata)
            displayed_id = (
                confirmation_request_id(metadata) if tool_name == PLAN_TOOL_NAME else core_request_id
            )
            if displayed_id and displayed_id == request_id:
                return tool_name, core_request_id, metadata
    return None


def build_confirmation_message(round_state: DynamicExpertRound) -> str:
    """Use the same display data for clients without expert_plan UI."""
    expert_plan = round_state.confirmation_display("")
    display = expert_plan["display"]
    lines = [display["title"], display["description"], ""]
    lines.extend(member["name"] for member in expert_plan["members"])
    if display.get("candidate_hint"):
        lines.extend(["", display["candidate_hint"]])
    return "\n".join(lines)


class DynamicExpertConfirmationRail(ConfirmInterruptRail):
    """Use Core's real tool-call resume while never consulting permission grants."""

    priority = 82

    def __init__(self, round_provider: Callable[[str], DynamicExpertRound | None]) -> None:
        super().__init__(tool_names=[PLAN_TOOL_NAME])
        self._round_provider = round_provider

    async def before_tool_call(self, ctx: AgentCallbackContext) -> None:
        if not ctx.extra.get("_skip_tool"):
            await super().before_tool_call(ctx)

    def _interrupt_plan(
        self, round_state: DynamicExpertRound, metadata: dict[str, str],
    ) -> InterruptDecision:
        expert_plan = round_state.confirmation_display(metadata["dynamic_expert_plan_id"])
        return self.interrupt(InterruptRequest(
            message=build_confirmation_message(round_state),
            payload_schema=self.request.payload_schema,
            # Empty key prevents Core from remembering approval for this tool.
            auto_confirm_key="",
            metadata={
                "source": "confirm_interrupt", **metadata,
                "expert_plan": expert_plan,
            },
            ui_options=[
                {"label": "交给这位专家" if len(expert_plan["members"]) == 1 else "请他们一起做",
                 "value": "approve"},
                {"label": "还是小艺Work来做", "value": "decline"},
            ],
        ))

    def _interrupt_no_plan(
        self, round_state: DynamicExpertRound, metadata: dict[str, str],
    ) -> InterruptDecision:
        display = {"title": _NO_PLAN_TITLE, "description": _NO_PLAN_MESSAGE}
        return self.interrupt(InterruptRequest(
            message=f"{display['title']}\n{display['description']}",
            payload_schema=self.request.payload_schema,
            auto_confirm_key="",
            metadata={
                "source": "confirm_interrupt", **metadata,
                "expert_no_plan": {
                    "confirmation_id": metadata["dynamic_expert_no_plan_id"],
                    "candidate_count": round_state.candidate_count,
                    "display": display,
                },
            },
            ui_options=[
                {"label": "小艺Work继续", "value": "continue"},
                {"label": "去召唤专家", "value": "summon"},
            ],
        ))

    async def resolve_interrupt(
        self, ctx: AgentCallbackContext, tool_call: ToolCall | None,
        user_input: Any, auto_confirm_config: dict | None = None,
    ) -> InterruptDecision:
        session_id = ctx.session.get_session_id() if ctx.session is not None else ""
        call_id = str(getattr(tool_call, "id", "") or "")
        round_state = self._round_provider(session_id)
        if round_state is None or round_state.closed or not call_id or round_state.session_id != session_id:
            return self._expired(ctx)
        if _in_plan_mode(round_state.agent, ctx.session):
            return self.reject(tool_result={
                "success": False, "status": "error", "reason": "expert_plan_unavailable",
            })

        if user_input is None:
            resume_input = ctx.extra.get(RESUME_USER_INPUT_KEY)
            if resume_input is not None:
                # Core retries every tool in the interrupted batch. An answer
                # for another tool leaves this exact saved plan waiting.
                metadata = ctx.session.get_state(CONFIRMATION_STATE_KEY)
                other_answers = (
                    resume_input.user_inputs
                    if isinstance(resume_input, InteractiveInput) and resume_input.raw_inputs is None else {}
                )
                if (not other_answers or call_id in other_answers
                        or any(key.startswith((_PLAN_REQUEST_PREFIX, _NO_PLAN_REQUEST_PREFIX))
                               for key in other_answers)):
                    return self._expired(ctx)
                if (not isinstance(metadata, dict)
                        or not (round_state.pending_plan is not None
                                or (round_state.response or {}).get("status") == "no_plan")
                        or not round_state.matches_confirmation(session_id, call_id, metadata)):
                    return self._expired(ctx)
            else:
                args = DynamicExpertRail._arguments(tool_call.arguments)
                query = args.get("query")
                await round_state.compose(query if isinstance(query, str) else "")
                if round_state.pending_plan is None:
                    if ((round_state.response or {}).get("status") != "no_plan"
                            or (round_state.response or {}).get("decision") is not None
                            or not round_state.supports_user_interaction):
                        return self.approve()
                    metadata = round_state.request_no_plan_confirmation(
                        session_id, call_id, session=ctx.session,
                    )
                else:
                    metadata = round_state.request_confirmation(session_id, call_id, session=ctx.session)
            if metadata is None:
                return self._expired(ctx)
            if "dynamic_expert_no_plan_id" in metadata:
                return self._interrupt_no_plan(round_state, metadata)
            return self._interrupt_plan(round_state, metadata)

        # ReAct clears INTERRUPTION_KEY before invoking resumed tools. Keep
        # our server-owned identity separately until this plan is consumed.
        metadata = ctx.session.get_state(CONFIRMATION_STATE_KEY)
        # Other HITL sources may carry the same Core call ID. Only the input
        # admitted for this displayed plan receives the server-only credential.
        nonce = user_input.get(CONFIRMATION_NONCE_KEY) if isinstance(user_input, dict) else None
        if (not isinstance(metadata, dict)
                or not round_state.matches_confirmation_resume(session_id, call_id, metadata, nonce)):
            return self._expired(ctx)
        try:
            payload = ConfirmPayload.model_validate(user_input)
        except (TypeError, ValueError):
            payload = ConfirmPayload(approved=False, feedback=str(user_input or ""))
        action = user_input.get(EXPERT_PLAN_ACTION_KEY) if isinstance(user_input, dict) else None
        if "dynamic_expert_no_plan_id" in metadata:
            if (action not in {"continue", "summon"} or not payload.approved
                    or payload.auto_confirm or payload.persist_allow):
                return self._expired(ctx)
            result = await round_state.resolve_no_plan_choice(
                session_id, call_id, metadata, nonce, action,
            )
            if not result.get("success"):
                return self._expired(ctx)
            if action == "summon":
                ctx.request_force_finish({
                    "output": "这次先停在这里。召唤更多专家后，你可以发起新任务。",
                    "result_type": "answer",
                })
            return self.reject(tool_result=result)
        # Session/permanent permission grants are not an approval of this plan.
        if action == "approve" and payload.approved and not payload.auto_confirm and not payload.persist_allow:
            result = await round_state.confirm_and_bind(
                session_id, call_id, metadata, confirmation_nonce=nonce,
            )
            if result.get("status") == "handoff_failed":
                ctx.request_force_finish({
                    "output": "这位专家暂时没能接手，这次任务没有继续执行。请再试一次。",
                    "result_type": "answer",
                })
            return self.approve() if result.get("success") else self.reject(tool_result=result)

        if action == "revise" and not payload.approved and payload.feedback.strip():
            result = await round_state.revise_plan(
                session_id, call_id, metadata, nonce, payload.feedback,
            )
            if round_state.pending_plan is not None and result.get("status") == "awaiting_confirmation":
                revised = round_state.request_confirmation(session_id, call_id, session=ctx.session)
                if revised is not None:
                    return self._interrupt_plan(round_state, revised)
            if result.get("status") == "no_plan":
                revised = round_state.request_no_plan_confirmation(
                    session_id, call_id, session=ctx.session,
                )
                if revised is not None:
                    return self._interrupt_no_plan(round_state, revised)
            return self.reject(tool_result=result)

        if action == "decline" and not payload.approved:
            result = {
                "success": True, "status": "declined",
                "message": "用户选择不使用专家。请由小艺Work继续完成原任务，本轮不要重新规划专家。",
            }
            round_state.response = result
            await round_state.close()
            return self.reject(tool_result=result)

        return self._expired(ctx)

    def _expired(self, ctx: AgentCallbackContext) -> RejectResult:
        message = "这份专家协作方案已失效，请重新发起任务。"
        ctx.request_force_finish({"output": message, "result_type": "answer"})
        return self.reject(tool_result={
            "success": False, "status": "error", "reason": "expert_confirmation_expired",
        })
