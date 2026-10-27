# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Readers must observe complete writes without holding the file lock to parse."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
import threading

import pytest

from jiuwenswarm.server.runtime.session import session_history as history
from jiuwenswarm.server.runtime.session import session_metadata


@pytest.fixture(params=[False, True], ids=["jsonl", "legacy-json"])
def session(tmp_path, monkeypatch, request):
    history.flush_history_writes()
    monkeypatch.setattr(history, "get_agent_sessions_dir", lambda: tmp_path)
    monkeypatch.setenv("JIUWENSWARM_USE_LEGACY_HISTORY_JSON", "1" if request.param else "0")
    sid = "read-write-session"
    seed = [{"id": "seed", "role": "user", "content": "original history"}]
    path = history.write_history_records(sid, seed)
    return sid, path, seed


def gate_write(monkeypatch, path, mode, occurrence=1):
    """Pause a real file open while the real writer owns its storage lock."""
    entered, release = threading.Event(), threading.Event()
    original_open = Path.open
    count = 0

    def open_file(self, *args, **kwargs):
        nonlocal count
        stream = original_open(self, *args, **kwargs)
        actual_mode = kwargs.get("mode", args[0] if args else "r")
        if self == path and actual_mode == mode:
            count += 1
            if count == occurrence:
                entered.set()
                if not release.wait(5):
                    stream.close()
                    raise TimeoutError("write gate watchdog expired")
        return stream

    monkeypatch.setattr(Path, "open", open_file)
    return entered, release


def start_read(pool, sid):
    started = threading.Event()

    def read():
        started.set()
        return history.load_history_records(sid)

    future = pool.submit(read)
    assert started.wait(2)
    return future


def test_reader_waits_for_rewrite_to_finish(session, monkeypatch):
    sid, path, _ = session
    replacement = [{"id": "new", "role": "user", "content": "complete replacement"}]
    entered, release = gate_write(monkeypatch, path, "w")
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(history.write_history_records, sid, replacement)
        try:
            assert entered.wait(2)
            assert path.stat().st_size == 0  # Actual truncate-before-write window.
            reader = start_read(pool, sid)
            with pytest.raises(TimeoutError):
                reader.result(timeout=0.1)
        finally:
            release.set()
        writer.result(timeout=3)
        assert reader.result(timeout=3) == replacement


def test_reader_waits_for_queued_compact_summary(session, monkeypatch):
    sid, path, _ = session
    # Metadata persistence is independent of the actual history writer under test.
    monkeypatch.setattr(session_metadata, "update_session_metadata", lambda **kwargs: None)
    mode = "a" if path.suffix == ".jsonl" else "w"
    entered, release = gate_write(monkeypatch, path, mode, occurrence=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            history.append_compact_history_records(
                session_id=sid, request_id="compact", channel_id="desktop",
                summary="completed compact summary", timestamp=10.0,
            )
            assert entered.wait(2)
            reader = start_read(pool, sid)
            with pytest.raises(TimeoutError):
                reader.result(timeout=0.1)
        finally:
            release.set()
            pool.submit(history.flush_history_writes).result(timeout=3)
        records = reader.result(timeout=3)
    assert len(records) == 3
    assert records[0]["id"] == "seed"
    assert [record["event_type"] for record in records[1:]] == [
        "context.compact_boundary", "context.compact_summary",
    ]
    assert records[-1]["content"] == "completed compact summary"


def test_parsing_snapshot_does_not_block_next_write(session, monkeypatch):
    sid, _, seed = session
    entered, release = threading.Event(), threading.Event()
    original_loads = history.json.loads

    def parse(text, *args, **kwargs):
        if "original history" in text and not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original_loads(text, *args, **kwargs)

    monkeypatch.setattr(history.json, "loads", parse)
    appended = {"id": "next", "role": "user", "content": "next message"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        reader = pool.submit(history.load_history_records, sid)
        try:
            assert entered.wait(2)
            # The reader already owns a text snapshot; writers need not wait for parsing.
            pool.submit(history._write_item, sid, appended).result(timeout=2)
        finally:
            release.set()
        assert reader.result(timeout=3) == seed
    assert history.load_history_records(sid) == seed + [appended]


def test_truncate_can_read_while_owning_write_lock(session):
    sid, _, _ = session
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(history.truncate_history_records, session_id=sid, cut_index=0).result(timeout=3)
    assert result == {"remaining_records": 0, "removed_records": 1}
    assert history.load_history_records(sid) == []
