from __future__ import annotations

import multiprocessing
import pathlib
import time
from typing import TYPE_CHECKING

import pytest

from pivot import exceptions
from pivot.storage import state

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event


def _helper_hold_write_lock(db_path: str, ready: Event, hold_seconds: float) -> None:
    db = state.StateDB(pathlib.Path(db_path))
    with db, db._write_transaction(timeout=10.0):
        ready.set()
        time.sleep(hold_seconds)


def test_write_succeeds_within_timeout(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "state.db"
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    with state.StateDB(db_path, write_timeout=0.1) as db:
        db.save(test_file, file_stat, "hash")


def test_write_timeout_raises_error(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "state.db"
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    process = ctx.Process(
        target=_helper_hold_write_lock,
        args=(str(db_path), ready, 0.5),
    )
    process.start()
    try:
        assert ready.wait(5), "Writer did not acquire lock in time"
        with (
            state.StateDB(db_path, write_timeout=0.05) as db,
            pytest.raises(exceptions.PivotDBWriteTimeoutError),
        ):
            db.save(test_file, file_stat, "hash")
    finally:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def test_timeout_is_configurable(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "state.db"
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    process = ctx.Process(
        target=_helper_hold_write_lock,
        args=(str(db_path), ready, 0.2),
    )
    process.start()
    try:
        assert ready.wait(5), "Writer did not acquire lock in time"
        with state.StateDB(db_path, write_timeout=1.0) as db:
            db.save(test_file, file_stat, "hash")
    finally:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
