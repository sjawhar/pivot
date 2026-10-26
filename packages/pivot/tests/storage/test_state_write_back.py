"""A read-only check saves the file hashes it read best-effort, and a state database this user
cannot write still serves reads (Dispatch BET-263)."""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import pathlib
import time
from typing import TYPE_CHECKING

import pytest

from pivot.storage import cache, state

if TYPE_CHECKING:
    from collections.abc import Generator
    from multiprocessing.synchronize import Event

    import lmdb

    from pivot.types import FileHashEntry


def _helper_entries(path: pathlib.Path) -> list[FileHashEntry]:
    file_hash, file_stat = cache.hash_file(path)
    return cache.deferred_file_hash_entries([(path, file_stat, file_hash)])


def _helper_new_state_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    state_dir = tmp_path / ".pivot"
    with state.StateDB(state_dir):
        pass
    return state_dir


def test_write_back_saves_entries_and_their_identity(tmp_path: pathlib.Path) -> None:
    """Saved entries are found by their own path and, for a hard link, by identity."""
    data = tmp_path / "data.txt"
    data.write_text("data")
    link = tmp_path / "link.txt"
    os.link(data, link)
    with state.StateDB(tmp_path / ".pivot"):
        pass

    state.write_back_file_hashes(tmp_path / ".pivot", _helper_entries(data))

    with state.StateDB(tmp_path / ".pivot", readonly=True) as state_db:
        expected = cache.hash_file(data)[0]
        assert state_db.get(data, data.stat()) == expected
        assert state_db.get(link, link.stat()) == expected


def test_write_back_never_creates_a_state_database(tmp_path: pathlib.Path) -> None:
    """A state directory with no database is left without one."""
    data = tmp_path / "data.txt"
    data.write_text("data")
    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()

    state.write_back_file_hashes(state_dir, _helper_entries(data))

    assert not (state_dir / "state.lmdb").exists()


def test_write_back_keeps_the_stat_taken_when_the_file_was_read(tmp_path: pathlib.Path) -> None:
    """An entry pairs the hash with the stat of the read that produced it, so a file changed
    between the read and the save is not vouched for."""
    data = tmp_path / "data.txt"
    data.write_text("data")
    state_dir = _helper_new_state_dir(tmp_path)
    entries = _helper_entries(data)
    data.write_text("changed, longer")

    state.write_back_file_hashes(state_dir, entries)

    with state.StateDB(state_dir, readonly=True) as state_db:
        assert state_db.get(data, data.stat()) is None


def test_write_back_leaves_out_entries_the_cache_cannot_hold(tmp_path: pathlib.Path) -> None:
    """A path too long for an LMDB key, or an mtime before 1970, is left out; the rest of the
    batch is saved."""
    long_dir = tmp_path / ("d" * 200) / ("e" * 200)
    long_dir.mkdir(parents=True)
    long_path = long_dir / ("f" * 200)
    long_path.write_text("long")
    old = tmp_path / "old.txt"
    old.write_text("old")
    os.utime(old, ns=(-1_000_000_000, -1_000_000_000))
    kept = tmp_path / "kept.txt"
    kept.write_text("kept")
    state_dir = _helper_new_state_dir(tmp_path)

    state.write_back_file_hashes(
        state_dir, _helper_entries(long_path) + _helper_entries(old) + _helper_entries(kept)
    )

    with state.StateDB(state_dir, readonly=True) as state_db:
        assert state_db.get(kept, kept.stat()) == cache.hash_file(kept)[0]
        assert state_db.get(old, old.stat()) is None


def test_write_back_leaves_out_an_mtime_after_2554(tmp_path: pathlib.Path) -> None:
    """An mtime the unsigned field cannot hold is left out at either end: before 1970, and
    after 2554, which a filesystem with 64-bit seconds (tmpfs, btrfs) reaches. The write-back
    is a cache, so it may never fail its caller."""
    kept = tmp_path / "kept.txt"
    kept.write_text("kept")
    state_dir = _helper_new_state_dir(tmp_path)
    path, _, size, dev, inode, file_hash = _helper_entries(kept)[0]
    far_future = (f"{path}.future", 32_000_000_000_000_000_000, size, dev, inode + 1, file_hash)

    state.write_back_file_hashes(state_dir, [far_future, *_helper_entries(kept)])

    with state.StateDB(state_dir, readonly=True) as state_db:
        assert state_db.get(kept, kept.stat()) == cache.hash_file(kept)[0]


class _RecordingTxn:
    """A write transaction that notes the key of each put, in order."""

    _txn: lmdb.Transaction
    _puts: list[bytes]

    def __init__(self, txn: lmdb.Transaction, puts: list[bytes]) -> None:
        self._txn = txn
        self._puts = puts

    def get(self, key: bytes) -> bytes | None:
        return self._txn.get(key)

    def put(self, key: bytes, value: bytes) -> bool:
        self._puts.append(key)
        return self._txn.put(key, value)

    def delete(self, key: bytes) -> bool:
        return self._txn.delete(key)


def _helper_record_write_transactions(monkeypatch: pytest.MonkeyPatch) -> list[list[bytes]]:
    """Make every StateDB write transaction note its puts; return the keys, one list per
    transaction."""
    transactions = list[list[bytes]]()
    real = state.StateDB._write_transaction

    @contextlib.contextmanager
    def _recording(self: state.StateDB, timeout: float = 30.0) -> Generator[_RecordingTxn]:
        with real(self, timeout) as txn:
            puts = list[bytes]()
            transactions.append(puts)
            yield _RecordingTxn(txn, puts)

    monkeypatch.setattr(state.StateDB, "_write_transaction", _recording)
    return transactions


def _helper_three_files(tmp_path: pathlib.Path) -> list[pathlib.Path]:
    paths = [tmp_path / f"{name}.txt" for name in ("a", "b", "c")]
    for path in paths:
        path.write_text(path.stem)
    return paths


def test_write_back_saves_its_batch_in_one_transaction(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write-back takes the write lock once for its whole batch, not once per file."""
    paths = _helper_three_files(tmp_path)
    state_dir = _helper_new_state_dir(tmp_path)
    transactions = _helper_record_write_transactions(monkeypatch)

    state.write_back_file_hashes(state_dir, [e for p in paths for e in _helper_entries(p)])

    assert len(transactions) == 1
    assert sorted(key for key in transactions[0] if key.startswith(b"hash:")) == sorted(
        b"hash:" + os.fsencode(path) for path in paths
    )


def test_a_write_puts_every_hash_entry_before_the_first_index_entry(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batch's index entries are written after all its hash entries, so the pages of a
    large write hold one kind of entry each, and a status whose lookups all hit by path reads
    no index page from disk. Written the other way round, the index pages fall between the
    hash pages, and a cold status reads them too."""
    paths = _helper_three_files(tmp_path)
    state_dir = _helper_new_state_dir(tmp_path)
    transactions = _helper_record_write_transactions(monkeypatch)

    state.write_back_file_hashes(state_dir, [e for p in paths for e in _helper_entries(p)])

    puts = [key for transaction in transactions for key in transaction]
    assert [key.split(b":", 1)[0] for key in puts] == [b"hash"] * 3 + [b"inode"] * 3


def _helper_read_only(state_dir: pathlib.Path, *, lock_writable: bool) -> dict[pathlib.Path, int]:
    """Take the write bits off the database's files (and its lock file unless
    ``lock_writable``); return the modes to put back."""
    lmdb_dir = state_dir / "state.lmdb"
    modes = {
        path: path.stat().st_mode
        for path in [lmdb_dir, *lmdb_dir.iterdir()]
        if path.name != "lock.mdb" or not lock_writable
    }
    for path, mode in modes.items():
        path.chmod(mode & ~0o222)
    return modes


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any file mode")
def test_a_read_only_state_database_serves_reads_and_skips_the_write_back(
    tmp_path: pathlib.Path,
) -> None:
    """A database whose data file this user cannot write, with a lock file it can, opens for
    reading with its reader registered; a write-back into it is skipped, not raised, and a
    direct write still fails as a write to it must."""
    recorded = tmp_path / "recorded.txt"
    recorded.write_text("recorded")
    unrecorded = tmp_path / "unrecorded.txt"
    unrecorded.write_text("unrecorded")
    state_dir = tmp_path / ".pivot"
    with state.StateDB(state_dir) as state_db:
        recorded_hash, _ = cache.hash_file(recorded, state_db)
    modes = _helper_read_only(state_dir, lock_writable=True)
    try:
        with state.open_readonly(state_dir) as state_db:
            assert state_db is not None
            assert state_db.get(recorded, recorded.stat()) == recorded_hash
        state.write_back_file_hashes(state_dir, _helper_entries(unrecorded))
        with pytest.raises(state.StateDBError, match="cannot be written"):
            state.StateDB(state_dir)
        with state.StateDB(state_dir, readonly=True) as state_db:
            assert state_db.get(unrecorded, unrecorded.stat()) is None
    finally:
        for path, mode in modes.items():
            path.chmod(mode)


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any file mode")
def test_a_state_database_whose_lock_file_is_read_only_is_not_used(
    tmp_path: pathlib.Path,
) -> None:
    """A reader that cannot register in the lock file is not protected from a writer, so the
    database is not opened: the caller reads files, and a write-back is skipped."""
    unrecorded = tmp_path / "unrecorded.txt"
    unrecorded.write_text("unrecorded")
    state_dir = tmp_path / ".pivot"
    with state.StateDB(state_dir):
        pass
    modes = _helper_read_only(state_dir, lock_writable=False)
    try:
        with state.open_readonly(state_dir) as state_db:
            assert state_db is None
        state.write_back_file_hashes(state_dir, _helper_entries(unrecorded))
    finally:
        for path, mode in modes.items():
            path.chmod(mode)
    with state.StateDB(state_dir, readonly=True) as state_db:
        assert state_db.get(unrecorded, unrecorded.stat()) is None


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any file mode")
def test_a_state_database_on_a_read_only_filesystem_is_not_used(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a read-only filesystem LMDB would open without its lock file, so no writer through
    another mount could see the reader: the lock file reports not writable there, and the
    database is not opened."""
    state_dir = _helper_new_state_dir(tmp_path)
    modes = _helper_read_only(state_dir, lock_writable=True)
    real_access = os.access

    def _access(path: str | os.PathLike[str], mode: int) -> bool:
        if os.fspath(path).endswith("lock.mdb"):
            return False
        return real_access(path, mode)

    monkeypatch.setattr(os, "access", _access)
    try:
        with state.open_readonly(state_dir) as state_db:
            assert state_db is None
    finally:
        for path, mode in modes.items():
            path.chmod(mode)


def _helper_hold_write_lock(state_dir: str, held: Event, release: Event) -> None:
    db = state.StateDB(pathlib.Path(state_dir))
    with db, db._write_transaction(timeout=10.0):
        held.set()
        release.wait(60)


def test_write_back_waits_no_longer_than_its_budget_for_a_held_lock(
    tmp_path: pathlib.Path,
) -> None:
    """While another process holds the write lock, a write-back gives up after 2 s, well inside
    a pivot write's own 30 s, and saves nothing; with no entries, or with a timeout of 0, it
    does not wait at all."""
    data = tmp_path / "data.txt"
    data.write_text("data")
    state_dir = _helper_new_state_dir(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    held, release = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_helper_hold_write_lock, args=(str(state_dir), held, release))
    holder.start()
    try:
        assert held.wait(30), "The holder took the write lock"
        started = time.monotonic()
        state.write_back_file_hashes(state_dir, _helper_entries(data))
        budget = time.monotonic() - started
        started = time.monotonic()
        state.write_back_file_hashes(state_dir, [])
        state.write_back_file_hashes(state_dir, _helper_entries(data), timeout=0)
        no_wait = time.monotonic() - started
    finally:
        release.set()
        holder.join(30)

    assert 2.0 <= budget < 3.0
    assert no_wait < 0.5
    with state.StateDB(state_dir, readonly=True) as state_db:
        assert state_db.get(data, data.stat()) is None
