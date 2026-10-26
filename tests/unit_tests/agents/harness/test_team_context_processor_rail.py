"""Tests for team-side ContextProcessorRail construction (auto_compression merge).

``team_runtime_inheritance._build_context_processor_rail`` must merge
``react.context_engine_config.auto_compression`` into all three passive
compressor configs, mirroring ``interface_deep._build_context_processor_rail``.
"""

from __future__ import annotations

from typing import Any

import jiuwenswarm.agents.harness.team.team_runtime_inheritance as team_runtime_module
from jiuwenswarm.agents.harness.team.team_runtime_inheritance import (
    _build_context_processor_rail,
)

_AUTO = {"model_name": "LLM_CompressModel", "fallback_model_name": "compact-backup"}


class FakeContextProcessorRail:
    def __init__(self, *, processors=None, preset=None, session_memory=None):
        self.processors = processors
        self.preset = preset
        self.session_memory = session_memory


def _build(config: dict[str, Any] | None):
    return _build_context_processor_rail(config)


def _processor_map(rail) -> dict:
    return {key: cfg for key, cfg in rail.processors}


def test_auto_compression_merged_into_all_three_compressors(monkeypatch):
    monkeypatch.setattr(
        team_runtime_module, "ContextProcessorRail", FakeContextProcessorRail
    )

    rail = _build(
        {
            "context_engine_config": {
                "auto_compression": _AUTO,
                "dialogue_compressor_config": {"trigger_context_ratio": 0.8},
                "current_round_compressor_config": {"keep_recent_messages": 4},
                "round_level_compressor_config": {"keep_recent_messages": 4},
            }
        }
    )

    assert rail.preset is True
    processors = _processor_map(rail)
    for key in (
        "DialogueCompressor",
        "CurrentRoundCompressor",
        "RoundLevelCompressor",
    ):
        assert key in processors
        assert processors[key]["auto_compression"] == _AUTO
    # Thresholds survive the merge.
    assert processors["DialogueCompressor"]["trigger_context_ratio"] == 0.8
    assert processors["CurrentRoundCompressor"]["keep_recent_messages"] == 4


def test_auto_compression_only_still_registers_compressors(monkeypatch):
    """auto_compression alone (empty compressor configs) must register all three."""
    monkeypatch.setattr(
        team_runtime_module, "ContextProcessorRail", FakeContextProcessorRail
    )

    rail = _build(
        {"context_engine_config": {"auto_compression": _AUTO}}
    )

    assert [key for key, _ in rail.processors] == [
        "DialogueCompressor",
        "CurrentRoundCompressor",
        "RoundLevelCompressor",
    ]
    for _, cfg in rail.processors:
        assert cfg == {"auto_compression": _AUTO}


def test_invalid_auto_compression_value_is_ignored(monkeypatch):
    """A non-dict auto_compression must not leak into compressor configs."""
    monkeypatch.setattr(
        team_runtime_module, "ContextProcessorRail", FakeContextProcessorRail
    )

    rail = _build(
        {
            "context_engine_config": {
                "auto_compression": "LLM_CompressModel",
                "dialogue_compressor_config": {"trigger_context_ratio": 0.8},
            }
        }
    )

    assert _processor_map(rail)["DialogueCompressor"] == {
        "trigger_context_ratio": 0.8
    }


def test_no_auto_compression_keeps_legacy_behavior(monkeypatch):
    """Without auto_compression the configs pass through untouched."""
    monkeypatch.setattr(
        team_runtime_module, "ContextProcessorRail", FakeContextProcessorRail
    )

    rail = _build(
        {
            "context_engine_config": {
                "dialogue_compressor_config": {"trigger_context_ratio": 0.8},
                "message_summary_offloader_config": {"ttl_seconds": 300},
            }
        }
    )

    assert _processor_map(rail) == {
        "MessageSummaryOffloader": {"ttl_seconds": 300},
        "DialogueCompressor": {"trigger_context_ratio": 0.8},
    }


def test_empty_auto_compression_dict_follows_main_model(monkeypatch):
    """``auto_compression: {}`` is a valid explicit follow-main-model choice."""
    monkeypatch.setattr(
        team_runtime_module, "ContextProcessorRail", FakeContextProcessorRail
    )

    rail = _build(
        {
            "context_engine_config": {
                "auto_compression": {},
                "dialogue_compressor_config": {"trigger_context_ratio": 0.8},
            }
        }
    )

    assert _processor_map(rail)["DialogueCompressor"] == {
        "trigger_context_ratio": 0.8,
        "auto_compression": {},
    }


def test_empty_context_engine_config_registers_no_processors(monkeypatch):
    """Empty context_engine_config → rail with no user processors (preset only)."""
    monkeypatch.setattr(
        team_runtime_module, "ContextProcessorRail", FakeContextProcessorRail
    )

    rail = _build({"context_engine_config": {}})

    assert rail.processors is None
