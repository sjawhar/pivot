"""A read-only check saves the file hashes it read best-effort, and a state database this user
cannot write still serves reads (Dispatch BET-263)."""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
import time
from typing import TYPE_CHECKING

import lmdb
import pytest

from pivot.storage import cache, state
from tests import helpers

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable, Generator

    from pytest_mock import MockerFixture

    from pivot.types import FileHashEntry


def _helper_entries(path: pathlib.Path) -> list[FileHashEntry]:
    file_hash, file_stat = cache.hash_file(path)
    return cache.deferred_file_hash_entries([(path, file_stat, file_hash)])


def _helper_new_state_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    state_dir = tmp_path / ".pivot"
    with state.StateDB(state_dir):
        pass
    return state_dir


def test_write_back_saves_entries_and_their_identity(
    tmp_path: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saved entries are found by their own path and, for a hard link on a filesystem that
    keeps one identity per live file (mountinfo says ext4), by identity."""
    mountinfo = tmp_path_factory.mktemp("mounts") / "mountinfo"
    helpers.mount_table(monkeypatch, mountinfo, [(os.stat(tmp_path).st_dev, "/", "ext4")])
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
def test_an_error_in_the_callers_body_carries_no_context_from_the_open(
    tmp_path: pathlib.Path,
) -> None:
    """When the database is not used, `open_readonly` yields None after it has handled its own
    PermissionError, so an error the caller's `with` body raises does not show that error as
    its context ("During handling of the above exception")."""
    state_dir = _helper_new_state_dir(tmp_path)
    modes = _helper_read_only(state_dir, lock_writable=False)
    try:
        with (
            pytest.raises(ValueError, match="the caller's own error") as raised,
            state.open_readonly(state_dir) as state_db,
        ):
            assert state_db is None
            raise ValueError("the caller's own error")
    finally:
        for path, mode in modes.items():
            path.chmod(mode)
    assert raised.value.__context__ is None


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


def test_write_back_waits_no_longer_than_its_budget_for_a_held_lock(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """While another process holds the write lock, a write-back gives up after 2 s, well inside
    a pivot write's own 30 s, saves nothing and says so in a debug log; with no entries, or with
    a timeout of 0, it does not wait at all."""
    data = tmp_path / "data.txt"
    data.write_text("data")
    state_dir = _helper_new_state_dir(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    held, release = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=helpers.hold_write_lock, args=(str(state_dir), held, release))
    holder.start()
    try:
        assert held.wait(30), "The holder took the write lock"
        started = time.monotonic()
        with caplog.at_level(logging.DEBUG, logger="pivot.storage.state"):
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
    assert "Not saving 1 file hashes" in caplog.text
    with state.StateDB(state_dir, readonly=True) as state_db:
        assert state_db.get(data, data.stat()) is None


def _helper_damage(state_dir: pathlib.Path) -> pathlib.Path:
    """Zero the database file's first 8 KiB, both LMDB meta pages, as a torn copy can leave
    it; returns the file."""
    data_file = state_dir / "state.lmdb" / "data.mdb"
    with data_file.open("r+b") as f:
        f.write(b"\0" * 8192)
    return data_file


def _helper_hasher(
    path: pathlib.Path, seen: list[state.StateDB | None]
) -> Callable[[state.StateDB | None, list[tuple[pathlib.Path, os.stat_result, str]]], str]:
    """A ``hash_read_only`` body that hashes ``path`` and records each database it is given."""

    def _hash(
        state_db: state.StateDB | None, pending: list[tuple[pathlib.Path, os.stat_result, str]]
    ) -> str:
        seen.append(state_db)
        return cache.hash_file(path, state_db, pending=pending)[0]

    return _hash


def test_hash_read_only_saves_what_its_body_read_in_one_write_back(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """The body gets the database read-only and a list for ``pending``; what it read is saved
    afterwards, once, so the next call reads nothing."""
    data = tmp_path / "data.txt"
    data.write_text("data")
    state_dir = _helper_new_state_dir(tmp_path)
    write_back = mocker.spy(state, "write_back_file_hashes")
    seen = list[state.StateDB | None]()

    assert cache.hash_read_only(state_dir, _helper_hasher(data, seen)) == cache.hash_file(data)[0]
    assert seen[0] is not None and seen[0].readonly
    assert write_back.call_count == 1
    with state.StateDB(state_dir, readonly=True) as state_db:
        assert state_db.get(data, data.stat()) == cache.hash_file(data)[0]


def test_hash_read_only_reads_the_files_past_a_database_lmdb_cannot_open_when_asked(
    tmp_path: pathlib.Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    """With ``read_files_if_damaged`` (`pivot diff`, `pivot plots diff`) a damaged database is
    not used: the body runs with none, every file is read, and nothing is written to the
    database. Without it (status) the error propagates, as it always has."""
    data = tmp_path / "data.txt"
    data.write_text("data")
    state_dir = _helper_new_state_dir(tmp_path)
    data_file = _helper_damage(state_dir)
    before = data_file.read_bytes()
    write_back = mocker.spy(state, "write_back_file_hashes")
    seen = list[state.StateDB | None]()

    with caplog.at_level(logging.DEBUG, logger="pivot.storage.cache"):
        result = cache.hash_read_only(
            state_dir, _helper_hasher(data, seen), read_files_if_damaged=True
        )
    assert result == cache.hash_file(data)[0]
    assert seen == [None]
    assert write_back.call_count == 0
    assert data_file.read_bytes() == before
    assert "Reading files without the state database" in caplog.text
    with pytest.raises(lmdb.Error):
        cache.hash_read_only(state_dir, _helper_hasher(data, seen))


def test_hash_read_only_reads_the_files_again_after_a_lookup_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """A database that opens but fails a lookup (a damaged page) is not used either: with
    ``read_files_if_damaged`` the body runs again with no database, and nothing is saved."""
    data = tmp_path / "data.txt"
    data.write_text("data")
    state_dir = _helper_new_state_dir(tmp_path)

    def _corrupted(
        self: state.StateDB, items: list[tuple[pathlib.Path, os.stat_result]]
    ) -> dict[pathlib.Path, str | None]:
        raise lmdb.Error("MDB_CORRUPTED: Located page was wrong type")

    monkeypatch.setattr(state.StateDB, "get_many", _corrupted)
    write_back = mocker.spy(state, "write_back_file_hashes")
    seen = list[state.StateDB | None]()

    result = cache.hash_read_only(state_dir, _helper_hasher(data, seen), read_files_if_damaged=True)
    assert result == cache.hash_file(data)[0]
    assert [state_db is None for state_db in seen] == [False, True]
    assert write_back.call_count == 0
    with pytest.raises(lmdb.Error):
        cache.hash_read_only(state_dir, _helper_hasher(data, seen))
