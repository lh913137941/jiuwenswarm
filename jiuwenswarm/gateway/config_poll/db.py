# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

POLL_TABLES: tuple[str, ...] = (
    "channel_config",
    "logging_config",
    "log_masking_rule",
    "audit_log_config",
)
_POLL_TABLES = frozenset(POLL_TABLES)
# 本进程 DB handler 未注册或数据库中尚未建表的表：只告警一次，避免每轮 ERROR 刷屏
_skipped_uninitialized: set[str] = set()

# 错误消息中指示"表缺失"的关键词（覆盖 PersistentStore 的 "not initialized"
# 以及 MySQL/SQLAlchemy 的 "doesn't exist" / "does not exist"）
_MISSING_TABLE_MARKERS = ("not initialized", "doesn't exist", "does not exist")


def _is_missing_table_error(exc: Exception) -> bool:
    """判断异常是否表示表未创建或未注册。"""
    text = str(exc).lower()
    return any(marker in text for marker in _MISSING_TABLE_MARKERS)


def _skip_missing_table(table: str, exc: Exception) -> None:
    """将表标记为缺失并记录一次性告警。"""
    if table not in _skipped_uninitialized:
        _skipped_uninitialized.add(table)
        logger.warning(
            "[ConfigPoll] skip table=%s (%s); other poll tables unaffected",
            table,
            exc,
        )


def _normalize_updated_at(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def row_key(table: str, row: dict[str, Any]) -> str:
    if table == "channel_config":
        return str(row.get("channel_id") or "").strip()
    if table == "log_masking_rule":
        return str(row.get("rule_id") or "").strip()
    # logging_config / audit_log_config：单文档表
    return str(row.get("id") or "default").strip()


def row_stamp(row: dict[str, Any]) -> str:
    ts = _normalize_updated_at(row.get("updated_at"))
    return ts.isoformat() if ts is not None else ""


def row_snapshot(table: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for row in rows:
        key = row_key(table, row)
        if not key:
            continue
        snapshot[key] = row_stamp(row)
    return snapshot


async def list_table_records(table: str) -> list[dict[str, Any]]:
    """拉取 poll 表全行：优先 ``PersistentStore``，否则 ``db_queries`` reader。"""
    if table not in _POLL_TABLES:
        logger.warning("[ConfigPoll] unsupported table: %s", table)
        return []

    # 跳过此前已判定为缺失的表，避免每轮重复查询产生 ERROR 刷屏
    if table in _skipped_uninitialized:
        return []

    from jiuwenswarm.gateway.storage.access import get_persistent_store

    store = get_persistent_store()
    if store is not None:
        await store.ensure_ready()
        try:
            rows = await store.list(table)
        except Exception as exc:
            # 表未注册（"not initialized"）或数据库中未建表（"doesn't exist"）：
            # 标记为缺失并跳过，其它 poll 表不受影响。
            if _is_missing_table_error(exc):
                _skip_missing_table(table, exc)
                return []
            raise
        return [dict(row) for row in rows or []]

    from jiuwenswarm.server.runtime.enterprise_config import db_queries

    try:
        return await db_queries.list_records(table)
    except Exception as exc:
        if _is_missing_table_error(exc):
            _skip_missing_table(table, exc)
            return []
        raise
