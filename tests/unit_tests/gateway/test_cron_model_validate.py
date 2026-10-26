# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""validate_cron_model 单元测试.

重点回归：get_model_config 返回原始 entry（不解析环境变量），当 config.yaml
模板里 model_client_config.model_name 为 "${MODEL_NAME}" 占位符时，落库的
canonical 值必须经过 resolve_env_vars 解析，否则 cron job 记录会存进占位符
原文（cron_job_create_action_002_full_params 长红根因）。
"""

from __future__ import annotations

import pytest

from jiuwenswarm.gateway.cron.models import validate_cron_model


def _entry(model_name: str, alias: str = "") -> dict:
    entry: dict = {"model_client_config": {"model_name": model_name}}
    if alias:
        entry["alias"] = alias
    return entry


@pytest.fixture
def patch_config(monkeypatch):
    """按真实语义打桩：命中时返回原始 entry（含未解析占位符）."""

    def _patch(entries: list[dict], available: list[str] | None = None) -> None:
        import jiuwenswarm.common.config as cfg

        def fake_get_model_config(name: str, index=None):
            for entry in entries:
                mcc = entry.get("model_client_config") or {}
                resolved = str(cfg.resolve_env_vars(mcc.get("model_name", "")) or "")
                alias = str(entry.get("alias", "") or "")
                if resolved == name or (alias and alias == name):
                    return entry
            return None

        monkeypatch.setattr(cfg, "get_model_config", fake_get_model_config)
        monkeypatch.setattr(cfg, "get_model_names", lambda: available or [])

    return _patch


def test_placeholder_model_name_resolved_to_env_value(monkeypatch, patch_config):
    """${MODEL_NAME} 占位符必须解析成环境变量值后落库（长红根因回归）."""
    monkeypatch.setenv("MODEL_NAME", "DeepSeek-V4-Flash-0731")
    patch_config([_entry("${MODEL_NAME}")], available=["DeepSeek-V4-Flash-0731"])
    assert (
        validate_cron_model("DeepSeek-V4-Flash-0731") == "DeepSeek-V4-Flash-0731"
    )


def test_placeholder_with_default_syntax(monkeypatch, patch_config):
    """${VAR:-default} 语法在 env 未设时取 default."""
    monkeypatch.delenv("MODEL_NAME", raising=False)
    patch_config([_entry("${MODEL_NAME:-fallback-model}")], available=["fallback-model"])
    assert validate_cron_model("fallback-model") == "fallback-model"


def test_plain_model_name_passthrough(patch_config):
    """无占位符的普通名字原样返回."""
    patch_config([_entry("plain-model")], available=["plain-model"])
    assert validate_cron_model("plain-model") == "plain-model"


def test_alias_match_resolves_canonical(monkeypatch, patch_config):
    """别名命中时 canonical 同样要解析占位符."""
    monkeypatch.setenv("MODEL_NAME", "DeepSeek-V4-Flash-0731")
    patch_config(
        [_entry("${MODEL_NAME}", alias="alias-x")],
        available=["DeepSeek-V4-Flash-0731"],
    )
    assert validate_cron_model("alias-x") == "DeepSeek-V4-Flash-0731"


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_empty_input_returns_none(raw):
    assert validate_cron_model(raw) is None


def test_unknown_model_raises_with_hint(patch_config):
    patch_config([], available=["model-a", "model-b"])
    with pytest.raises(ValueError, match="model-a"):
        validate_cron_model("no-such-model")
