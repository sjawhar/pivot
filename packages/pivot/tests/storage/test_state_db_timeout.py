from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import pytest

from pivot import exceptions
from pivot.storage import state

if TYPE_CHECKING:
    import pathlib


def test_write_transaction_succeeds_within_timeout(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "state.db"
    db = state.StateDB(db_path, readonly=False, write_timeout=5.0)

    try:
        with db._write_transaction("test_operation") as txn:
            txn.put(b"test_key", b"test_value")

        with db._env.begin() as txn:
            assert txn.get(b"test_key") == b"test_value"
    finally:
        db.close()


def test_write_transaction_timeout_on_contention(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "state.db"
    db = state.StateDB(db_path, readonly=False, write_timeout=0.5)

    holder_ready = threading.Event()

    def hold_write_lock():
        with db._write_transaction("holder"):
            holder_ready.set()
            time.sleep(2)

    try:
        thread = threading.Thread(target=hold_write_lock, daemon=False)
        thread.start()

        holder_ready.wait(timeout=5)

        with (
            pytest.raises(exceptions.PivotDBWriteTimeoutError) as exc_info,
            db._write_transaction("contended_operation") as txn,
        ):
            txn.put(b"should_fail", b"value")

        assert "timed out" in str(exc_info.value).lower()
        assert "contended_operation" in str(exc_info.value)

        thread.join(timeout=5)
    finally:
        db.close()


def test_write_timeout_configurable(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "state.db"
    db = state.StateDB(db_path, readonly=False, write_timeout=10.0)

    assert db._write_timeout == 10.0

    db2 = state.StateDB(db_path, readonly=False, write_timeout=2.5)
    assert db2._write_timeout == 2.5

    db.close()
    db2.close()


def test_readonly_db_skips_timeout_logic(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "state.db"
    db_write = state.StateDB(db_path, readonly=False, write_timeout=5.0)

    with db_write._write_transaction("setup") as txn:
        txn.put(b"test_key", b"test_value")

    db_write.close()

    db_read = state.StateDB(db_path, readonly=True, write_timeout=5.0)

    assert db_read.readonly is True

    with (
        pytest.raises(RuntimeError, match="readonly"),
        db_read._write_transaction("should_fail") as txn,
    ):  # noqa: E501
        pass

    db_read.close()


def test_write_transaction_commits_on_success(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "state.db"
    db = state.StateDB(db_path, readonly=False, write_timeout=5.0)

    try:
        with db._write_transaction("test_commit") as txn:
            txn.put(b"key1", b"value1")
            txn.put(b"key2", b"value2")

        with db._env.begin() as txn:
            assert txn.get(b"key1") == b"value1"
            assert txn.get(b"key2") == b"value2"
    finally:
        db.close()
