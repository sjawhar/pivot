"""The fingerprint caches are saved best-effort, so a read command never waits a pivot write's
30 s for another process's state-database write lock (Dispatch BET-263)."""

from __future__ import annotations

import logging
import multiprocessing
import pathlib
import time
from typing import TYPE_CHECKING

import pytest
from tests import helpers

from pivot import fingerprint
from pivot.storage import state

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_AST_ENTRY: fingerprint.AstHashEntry = ("stage.py", "0" * 16, "my_stage", "3.13", 1, "a" * 16)
_MANIFEST_ENTRY = (b"sm:my_stage", b'{"m": {}, "s": {}}')


@pytest.fixture
def state_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[pathlib.Path]:
    """A state directory with an empty database, which the flushes write to, and no pending
    write left behind for another test."""
    path = tmp_path / ".pivot"
    with state.StateDB(path):
        pass
    monkeypatch.setattr(
        fingerprint, "_pending_ast_writes", dict[pathlib.Path, list[fingerprint.AstHashEntry]]()
    )
    monkeypatch.setattr(
        fingerprint, "_pending_manifest_writes", dict[pathlib.Path, list[tuple[bytes, bytes]]]()
    )
    yield path


def _queue_ast(path: pathlib.Path) -> None:
    fingerprint._pending_ast_writes.setdefault(path, []).append(_AST_ENTRY)


def _queue_manifest(path: pathlib.Path) -> None:
    fingerprint._pending_manifest_writes.setdefault(path, []).append(_MANIFEST_ENTRY)


def _helper_keys(path: pathlib.Path, prefix: bytes) -> list[bytes]:
    with state.StateDB(path, readonly=True) as state_db:
        return [key for key, _ in state_db.iter_prefix(prefix)]


@pytest.mark.parametrize(
    ("flush", "queue", "prefix", "message"),
    [
        pytest.param(
            fingerprint.flush_ast_hash_cache,
            _queue_ast,
            b"fp:",
            "Not saving 1 AST hashes",
            id="ast-hashes",
        ),
        pytest.param(
            fingerprint.flush_manifest_cache,
            _queue_manifest,
            b"sm:",
            "Not saving 1 manifest entries",
            id="manifests",
        ),
    ],
)
def test_a_flush_gives_up_after_2_s_of_a_held_write_lock_and_retries_later(
    state_dir: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    flush: Callable[[], None],
    queue: Callable[[pathlib.Path], None],
    prefix: bytes,
    message: str,
) -> None:
    """While another process holds the write lock, a fingerprint flush gives up after 2 s, well
    inside a pivot write's own 30 s, writes nothing and says so in a debug log. The entries stay
    pending, so the next flush saves them once the lock is free."""
    queue(state_dir)
    ctx = multiprocessing.get_context("spawn")
    held, release = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=helpers.hold_write_lock, args=(str(state_dir), held, release))
    holder.start()
    try:
        assert held.wait(30), "The holder took the write lock"
        started = time.monotonic()
        with caplog.at_level(logging.DEBUG, logger="pivot.storage.state"):
            flush()
        budget = time.monotonic() - started
    finally:
        release.set()
        holder.join(30)

    assert 2.0 <= budget < 3.0
    assert message in caplog.text
    assert _helper_keys(state_dir, prefix) == [], "A locked database is left as it was"

    flush()
    assert _helper_keys(state_dir, prefix) != [], "With the lock free, the flush saves"


@pytest.mark.parametrize(
    ("flush", "queue", "prefix"),
    [
        pytest.param(
            fingerprint.flush_ast_hash_cache,
            _queue_ast,
            b"fp:",
            id="ast-hashes",
        ),
        pytest.param(
            fingerprint.flush_manifest_cache,
            _queue_manifest,
            b"sm:",
            id="manifests",
        ),
    ],
)
def test_a_flush_with_a_free_lock_saves_its_entries_and_leaves_none_pending(
    state_dir: pathlib.Path,
    flush: Callable[[], None],
    queue: Callable[[pathlib.Path], None],
    prefix: bytes,
) -> None:
    """The ordinary path is unchanged: the entries are written and nothing stays pending."""
    queue(state_dir)

    flush()

    assert _helper_keys(state_dir, prefix) != []
    assert fingerprint._pending_ast_writes == {}
    assert fingerprint._pending_manifest_writes == {}
