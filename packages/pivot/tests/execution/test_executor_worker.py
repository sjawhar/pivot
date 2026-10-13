"""Worker primitives retained after the store-based execution rewrite.

The deleted annotation/injection worker tests targeted removed ``StageDefinition``
and cache helpers. Store-backed execution itself is covered by test_worker_store,
test_execution_modes, test_worker_artifact_lock and test_skip_detection_integration.
This module retains the worker primitives those suites depend on.
"""

from __future__ import annotations

import queue
import subprocess
import threading
from typing import TYPE_CHECKING, cast

from pivot.executor import worker
from pivot.types import OutputMessage

if TYPE_CHECKING:
    import multiprocessing as mp
    import pathlib

    import pytest


def _queue_lines(output_queue: mp.Queue[OutputMessage]) -> list[tuple[str, bool]]:
    messages = list[OutputMessage]()
    while True:
        try:
            messages.append(output_queue.get(timeout=0.5))
        except queue.Empty:
            break
    lines = list[tuple[str, bool]]()
    for message in messages:
        raw = cast("dict[str, object]", cast("object", message))
        line = raw.get("line")
        is_stderr = raw.get("is_stderr")
        if isinstance(line, str) and isinstance(is_stderr, bool):
            lines.append((line, is_stderr))
    return lines


def test_queue_writer_splits_on_newlines(output_queue: mp.Queue[OutputMessage]) -> None:
    """Queue writer emits one message per complete line."""
    ring = worker._OutputRingBuffer()
    with worker._QueueWriter("stage", output_queue, is_stderr=False, ring_buffer=ring) as writer:
        assert writer.write("one\ntwo\n") == 8

    assert _queue_lines(output_queue) == [("one", False), ("two", False)]
    assert ring.snapshot() == [("one", False), ("two", False)]


def test_queue_writer_flushes_partial_line(output_queue: mp.Queue[OutputMessage]) -> None:
    """Queue writer flushes a final partial line at context exit."""
    ring = worker._OutputRingBuffer()
    with worker._QueueWriter("stage", output_queue, is_stderr=False, ring_buffer=ring) as writer:
        writer.write("partial")

    assert _queue_lines(output_queue) == [("partial", False)]


def test_queue_writer_marks_stderr(output_queue: mp.Queue[OutputMessage]) -> None:
    """Queue writer preserves stream identity for stderr."""
    ring = worker._OutputRingBuffer()
    with worker._QueueWriter("stage", output_queue, is_stderr=True, ring_buffer=ring) as writer:
        writer.write("problem\n")

    assert _queue_lines(output_queue) == [("problem", True)]


def test_queue_writer_fileno_captures_subprocess_output(
    output_queue: mp.Queue[OutputMessage],
) -> None:
    """A subprocess inheriting QueueWriter.fileno() is captured."""
    ring = worker._OutputRingBuffer()
    with worker._QueueWriter("stage", output_queue, is_stderr=False, ring_buffer=ring) as writer:
        subprocess.run(
            ["/bin/sh", "-c", "printf from-subprocess"],
            stdout=writer.fileno(),
            check=True,
        )

    assert ("from-subprocess", False) in _queue_lines(output_queue)


def test_queue_writer_is_not_tty(output_queue: mp.Queue[OutputMessage]) -> None:
    """The redirected stream reports non-interactive behavior."""
    ring = worker._OutputRingBuffer()
    with worker._QueueWriter("stage", output_queue, is_stderr=False, ring_buffer=ring) as writer:
        assert not writer.isatty()


def test_ring_buffer_evicts_and_marks_truncation() -> None:
    """Bounded ring buffer evicts oldest lines and reports truncation."""
    ring = worker._OutputRingBuffer(max_lines=2)
    ring.append("one", False)
    ring.append("two", True)
    ring.append("three", False)

    assert ring.snapshot() == [
        ("[1 earlier lines truncated]", False),
        ("two", True),
        ("three", False),
    ]


def test_ring_buffer_snapshot_is_copy() -> None:
    """Mutating a snapshot cannot mutate ring-buffer state."""
    ring = worker._OutputRingBuffer(max_lines=3)
    ring.append("one", False)
    snapshot = ring.snapshot()
    snapshot.clear()

    assert ring.snapshot() == [("one", False)]


def test_ring_buffer_thread_safe() -> None:
    """Concurrent appends preserve whole lines without race exceptions."""
    ring = worker._OutputRingBuffer(max_lines=1000)

    def append(prefix: str) -> None:
        for index in range(100):
            ring.append(f"{prefix}-{index}", False)

    threads = [threading.Thread(target=append, args=(str(index),)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(ring.snapshot()) == 400


def test_hash_dependencies_files_and_missing(tmp_path: pathlib.Path) -> None:
    """Legacy external path list hashes files and reports only missing entries."""
    existing = tmp_path / "existing.txt"
    existing.write_text("content")
    missing = tmp_path / "missing.txt"

    hashes, missing_paths, unreadable, entries = worker.hash_dependencies(
        [str(existing), str(missing)]
    )

    assert str(existing) in hashes
    assert missing_paths == [str(missing)]
    assert unreadable == []
    assert entries[0][0] == str(existing)


def test_hash_dependencies_directory_has_manifest(tmp_path: pathlib.Path) -> None:
    """Directory hashing returns a tree hash with every file in its manifest."""
    directory = tmp_path / "data"
    directory.mkdir()
    (directory / "a.txt").write_text("a")
    (directory / "b.txt").write_text("b")

    hashes, missing, unreadable, _ = worker.hash_dependencies([str(directory)])

    assert missing == []
    assert unreadable == []
    hash_info = hashes[str(directory)]
    assert "manifest" in hash_info
    assert {entry["relpath"] for entry in hash_info["manifest"]} == {"a.txt", "b.txt"}


def _double(value: int) -> int:
    return value * 2


def _identity(value: int) -> int:
    return value


def test_execute_with_joblib_protection_uses_stage_result() -> None:
    """Joblib protection returns the stage result under the installed backend."""
    assert worker._execute_with_joblib_protection(_double, {"value": 21}) == 42


def test_execute_with_joblib_processes_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process opt-in path still invokes the function successfully."""
    monkeypatch.setenv("PIVOT_NESTED_PARALLELISM", "processes")
    assert worker._execute_with_joblib_protection(_identity, {"value": 7}) == 7


# DirectoryOut, IncrementalOut, and old annotation injection tests are exercised
# by the dedicated red suites and store-based worker coverage listed above.
