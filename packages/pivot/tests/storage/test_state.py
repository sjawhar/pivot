from __future__ import annotations

import json
import multiprocessing
import os
import pathlib
import stat
import struct
import time
from typing import TYPE_CHECKING, Any, cast

import lmdb
import pytest

from pivot import run_history
from pivot.storage import cache, state

if TYPE_CHECKING:
    from threading import Barrier

    from pivot.types import DeferredWrites


def test_state_cache_hit(tmp_path: pathlib.Path) -> None:
    """Unchanged mtime/size/inode returns cached hash."""
    db_path = tmp_path
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    with state.StateDB(db_path) as db:
        db.save(test_file, file_stat, "abc123")
        result = db.get(test_file, file_stat)

    assert result == "abc123"


def test_state_cache_miss_mtime(tmp_path: pathlib.Path) -> None:
    """Changed mtime triggers cache miss."""
    db_path = tmp_path
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    old_stat = test_file.stat()

    with state.StateDB(db_path) as db:
        db.save(test_file, old_stat, "abc123")

        time.sleep(0.01)
        test_file.write_text("content")
        new_stat = test_file.stat()

        result = db.get(test_file, new_stat)

    assert result is None


def test_state_cache_miss_size(tmp_path: pathlib.Path) -> None:
    """Changed size triggers cache miss."""
    db_path = tmp_path
    test_file = tmp_path / "file.txt"
    test_file.write_text("short")
    old_stat = test_file.stat()

    with state.StateDB(db_path) as db:
        db.save(test_file, old_stat, "abc123")

        test_file.write_text("much longer content")
        os.utime(test_file, (old_stat.st_mtime, old_stat.st_mtime))
        new_stat = test_file.stat()

        result = db.get(test_file, new_stat)

    assert result is None


def test_state_cache_miss_inode(tmp_path: pathlib.Path) -> None:
    """Changed inode triggers cache miss."""
    db_path = tmp_path
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    old_stat = test_file.stat()

    with state.StateDB(db_path) as db:
        db.save(test_file, old_stat, "abc123")

        # Force inode change by writing to temp file and renaming
        # (unlink + write may reuse the same inode on some filesystems)
        temp_file = tmp_path / "file.txt.tmp"
        temp_file.write_text("content")
        test_file.unlink()
        temp_file.rename(test_file)
        new_stat = test_file.stat()

        # Verify inode actually changed (skip if filesystem reuses inodes)
        if new_stat.st_ino == old_stat.st_ino:
            pytest.skip("Filesystem reused inode - cannot test inode change detection")

        result = db.get(test_file, new_stat)

    assert result is None


def test_state_save_many(tmp_path: pathlib.Path) -> None:
    """Batch save works correctly."""
    db_path = tmp_path
    files = list[tuple[pathlib.Path, os.stat_result]]()
    entries = list[tuple[pathlib.Path, os.stat_result, str]]()

    for i in range(5):
        f = tmp_path / f"file_{i}.txt"
        f.write_text(f"content {i}")
        file_stat = f.stat()
        files.append((f, file_stat))
        entries.append((f, file_stat, f"hash_{i}"))

    with state.StateDB(db_path) as db:
        db.save_many(entries)

        for i, (f, file_stat) in enumerate(files):
            result = db.get(f, file_stat)
            assert result == f"hash_{i}"


def test_state_db_persistence(tmp_path: pathlib.Path) -> None:
    """State survives process restart (new instance)."""
    db_path = tmp_path
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    with state.StateDB(db_path) as db1:
        db1.save(test_file, file_stat, "persistent_hash")

    with state.StateDB(db_path) as db2:
        result = db2.get(test_file, file_stat)

    assert result == "persistent_hash"


def test_state_get_missing_path(tmp_path: pathlib.Path) -> None:
    """Getting uncached path returns None."""
    db_path = tmp_path
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    with state.StateDB(db_path) as db:
        result = db.get(test_file, file_stat)

    assert result is None


def test_state_update_existing(tmp_path: pathlib.Path) -> None:
    """Saving same path updates the cached hash."""
    db_path = tmp_path
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    with state.StateDB(db_path) as db:
        db.save(test_file, file_stat, "old_hash")
        db.save(test_file, file_stat, "new_hash")
        result = db.get(test_file, file_stat)

    assert result == "new_hash"


def test_state_db_creates_parent_dirs(tmp_path: pathlib.Path) -> None:
    """StateDB creates parent directories if needed."""
    state_dir = tmp_path / "nested" / "deep"
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    with state.StateDB(state_dir) as db:
        db.save(test_file, file_stat, "hash")

    lmdb_path = state_dir / "state.lmdb"
    assert lmdb_path.is_dir()


def test_state_close(tmp_path: pathlib.Path) -> None:
    """StateDB can be closed and reopened."""
    db_path = tmp_path
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    db = state.StateDB(db_path)
    db.save(test_file, file_stat, "hash")
    db.close()

    db2 = state.StateDB(db_path)
    result = db2.get(test_file, file_stat)

    assert result == "hash"


def test_state_context_manager(tmp_path: pathlib.Path) -> None:
    """StateDB works as context manager."""
    db_path = tmp_path
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    with state.StateDB(db_path) as db:
        db.save(test_file, file_stat, "hash")

    with state.StateDB(db_path) as db:
        result = db.get(test_file, file_stat)

    assert result == "hash"


def test_state_absolute_paths(tmp_path: pathlib.Path) -> None:
    """Paths are stored as absolute for consistency."""
    db_path = tmp_path
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    with state.StateDB(db_path) as db:
        db.save(test_file.resolve(), file_stat, "hash")
        result = db.get(test_file.resolve(), file_stat)

    assert result == "hash"


def test_state_get_many(tmp_path: pathlib.Path) -> None:
    """Batch get returns correct hashes for multiple files."""
    db_path = tmp_path
    files = list[tuple[pathlib.Path, os.stat_result]]()
    entries = list[tuple[pathlib.Path, os.stat_result, str]]()

    for i in range(5):
        f = tmp_path / f"file_{i}.txt"
        f.write_text(f"content {i}")
        file_stat = f.stat()
        files.append((f, file_stat))
        entries.append((f, file_stat, f"hash_{i}"))

    with state.StateDB(db_path) as db:
        db.save_many(entries)
        results = db.get_many(files)

    for i, (f, _) in enumerate(files):
        assert results[f] == f"hash_{i}"


def test_state_get_many_mixed(tmp_path: pathlib.Path) -> None:
    """Batch get handles mix of cached and uncached files."""
    db_path = tmp_path

    # Create cached file
    cached = tmp_path / "cached.txt"
    cached.write_text("cached content")
    cached_stat = cached.stat()

    # Create uncached file
    uncached = tmp_path / "uncached.txt"
    uncached.write_text("uncached content")
    uncached_stat = uncached.stat()

    with state.StateDB(db_path) as db:
        db.save(cached, cached_stat, "cached_hash")
        results = db.get_many([(cached, cached_stat), (uncached, uncached_stat)])

    assert results[cached] == "cached_hash"
    assert results[uncached] is None


def test_state_get_many_empty(tmp_path: pathlib.Path) -> None:
    """Batch get with empty list returns empty dict."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        results = db.get_many([])

    assert results == {}


def test_state_save_skips_a_path_too_long_for_a_key(tmp_path: pathlib.Path) -> None:
    """save leaves out an entry whose key exceeds LMDB's limit, as save_many does: this cache
    only saves a later hash, so a path too long for it must not fail the caller
    (``pivot commit`` and ``pivot track`` hash through a writable StateDB)."""
    db_path = tmp_path
    nested = tmp_path
    for i in range(12):
        nested = nested / ("d" * 50 + str(i))
    nested.mkdir(parents=True)
    long_path = nested / "file.txt"
    long_path.write_text("content")
    file_stat = long_path.stat()

    with state.StateDB(db_path) as db:
        db.save(long_path, file_stat, "hash123")
        assert db.get(long_path, file_stat) is None, "Too long to cache"
        file_hash, _ = cache.hash_file(long_path, db)

    assert file_hash == cache.hash_file(long_path)[0], "Hashing through the StateDB succeeds"


def test_state_save_many_skips_a_path_too_long_for_a_key(tmp_path: pathlib.Path) -> None:
    """save_many leaves out an entry whose key exceeds LMDB's limit and saves the rest."""
    db_path = tmp_path
    nested = tmp_path
    for i in range(12):
        nested = nested / ("e" * 50 + str(i))
    nested.mkdir(parents=True)
    long_path = nested / "file.txt"
    long_path.write_text("content")
    short_path = tmp_path / "short.txt"
    short_path.write_text("short")

    with state.StateDB(db_path) as db:
        db.save_many(
            [(long_path, long_path.stat(), "hash123"), (short_path, short_path.stat(), "hash456")]
        )
        assert db.get(long_path, long_path.stat()) is None, "Too long to cache"
        assert db.get(short_path, short_path.stat()) == "hash456", "The rest is saved"


def test_state_raises_after_close(tmp_path: pathlib.Path) -> None:
    """Operations on closed StateDB raise RuntimeError."""
    db_path = tmp_path
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    db = state.StateDB(db_path)
    db.close()

    with pytest.raises(RuntimeError, match="closed StateDB"):
        db.get(test_file, file_stat)


# -----------------------------------------------------------------------------
# Generation tracking tests
# -----------------------------------------------------------------------------


def test_generation_get_nonexistent(tmp_path: pathlib.Path) -> None:
    """Getting generation for untracked identity returns None."""
    db_path = tmp_path
    identity = "stage:output"

    with state.StateDB(db_path) as db:
        result = db.get_generation(identity)

    assert result is None


def test_generation_increment_creates_new(tmp_path: pathlib.Path) -> None:
    """Incrementing untracked identity creates it with generation 1."""
    db_path = tmp_path
    identity = "stage:output"

    with state.StateDB(db_path) as db:
        gen = db.increment_generation(identity)
        assert gen == 1
        assert db.get_generation(identity) == 1


def test_generation_increment_existing(tmp_path: pathlib.Path) -> None:
    """Incrementing tracked identity increases generation."""
    db_path = tmp_path
    identity = "stage:output"

    with state.StateDB(db_path) as db:
        db.increment_generation(identity)
        db.increment_generation(identity)
        gen = db.increment_generation(identity)

    assert gen == 3


def test_generation_persistence(tmp_path: pathlib.Path) -> None:
    """Generations persist across DB instances."""
    db_path = tmp_path
    identity = "stage:output"

    with state.StateDB(db_path) as db:
        db.increment_generation(identity)
        db.increment_generation(identity)

    with state.StateDB(db_path) as db:
        assert db.get_generation(identity) == 2
        gen = db.increment_generation(identity)
        assert gen == 3


def test_generation_get_many(tmp_path: pathlib.Path) -> None:
    """Batch query returns generations for multiple identities."""
    db_path = tmp_path
    identities = [f"stage:out_{i}" for i in range(3)]

    with state.StateDB(db_path) as db:
        db.increment_generation(identities[0])
        db.increment_generation(identities[0])
        db.increment_generation(identities[1])
        # identities[2] not tracked

        results = db.get_many_generations(identities)

    assert results[identities[0]] == 2
    assert results[identities[1]] == 1
    assert results[identities[2]] is None


def test_generation_get_many_empty(tmp_path: pathlib.Path) -> None:
    """Batch query with empty list returns empty dict."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        results = db.get_many_generations([])

    assert results == {}


def test_dep_generations_get_nonexistent(tmp_path: pathlib.Path) -> None:
    """Getting dep generations for unknown stage returns None."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        result = db.get_dep_generations("unknown_stage")

    assert result is None


def test_dep_generations_record_and_get(tmp_path: pathlib.Path) -> None:
    """Record and retrieve dependency generations."""
    db_path = tmp_path
    deps = {"upstream:out1": 5, "upstream:out2": 3}

    with state.StateDB(db_path) as db:
        db.record_dep_generations("my_stage", deps)
        result = db.get_dep_generations("my_stage")

    assert result == deps


def test_dep_generations_update_replaces(tmp_path: pathlib.Path) -> None:
    """Recording dep generations replaces previous values."""
    db_path = tmp_path
    old_deps = {"upstream:out1": 1, "upstream:out2": 2}
    new_deps = {"upstream:out1": 5, "upstream:out3": 1}

    with state.StateDB(db_path) as db:
        db.record_dep_generations("my_stage", old_deps)
        db.record_dep_generations("my_stage", new_deps)
        result = db.get_dep_generations("my_stage")

    assert result == new_deps, "Old deps should be replaced, not merged"


def test_dep_generations_multiple_stages(tmp_path: pathlib.Path) -> None:
    """Different stages have independent dep generations."""
    db_path = tmp_path
    stage1_deps = {"upstream:out1": 1}
    stage2_deps = {"upstream:out2": 2, "upstream:out3": 3}

    with state.StateDB(db_path) as db:
        db.record_dep_generations("stage1", stage1_deps)
        db.record_dep_generations("stage2", stage2_deps)

        result1 = db.get_dep_generations("stage1")
        result2 = db.get_dep_generations("stage2")

    assert result1 == stage1_deps
    assert result2 == stage2_deps


def test_dep_generations_persistence(tmp_path: pathlib.Path) -> None:
    """Dep generations persist across DB instances."""
    db_path = tmp_path
    deps = {"upstream:out": 42}

    with state.StateDB(db_path) as db:
        db.record_dep_generations("my_stage", deps)

    with state.StateDB(db_path) as db:
        result = db.get_dep_generations("my_stage")

    assert result == deps


# -----------------------------------------------------------------------------
# Symlink handling tests
# -----------------------------------------------------------------------------


def test_generation_identity_is_keyed_by_producer_and_key(tmp_path: pathlib.Path) -> None:
    """Generation tracking is keyed by artifact identity, not filesystem paths."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        gen1 = db.increment_generation("stage:out1")
        assert gen1 == 1

        assert db.get_generation("stage:out2") is None

        gen2 = db.increment_generation("stage:out2")
        assert gen2 == 1

        assert db.get_generation("stage:out1") == 1


# -----------------------------------------------------------------------------
# Remote index tracking tests
# -----------------------------------------------------------------------------


def test_remote_hash_exists_false(tmp_path: pathlib.Path) -> None:
    """Unknown hash returns False."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        result = db.remote_hash_exists("origin", "abc123def456")

    assert result is False


def test_remote_hash_exists_true(tmp_path: pathlib.Path) -> None:
    """Added hash returns True."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", ["abc123def456"])
        result = db.remote_hash_exists("origin", "abc123def456")

    assert result is True


def test_remote_hashes_add_multiple(tmp_path: pathlib.Path) -> None:
    """Multiple hashes can be added at once."""
    db_path = tmp_path
    hashes = ["hash1", "hash2", "hash3"]

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", hashes)

        for h in hashes:
            assert db.remote_hash_exists("origin", h)


def test_remote_hashes_intersection(tmp_path: pathlib.Path) -> None:
    """Intersection returns only hashes known to exist on remote."""
    db_path = tmp_path
    known = {"hash1", "hash2", "hash3"}
    query = {"hash1", "hash3", "hash4", "hash5"}

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", known)
        result = db.remote_hashes_intersection("origin", query)

    assert result == {"hash1", "hash3"}


def test_remote_hashes_intersection_empty_query(tmp_path: pathlib.Path) -> None:
    """Empty query returns empty set."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", ["hash1", "hash2"])
        result = db.remote_hashes_intersection("origin", set())

    assert result == set()


def test_remote_hashes_intersection_no_matches(tmp_path: pathlib.Path) -> None:
    """No matches returns empty set."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", ["hash1", "hash2"])
        result = db.remote_hashes_intersection("origin", {"hash3", "hash4"})

    assert result == set()


def test_remote_hashes_remove(tmp_path: pathlib.Path) -> None:
    """Removed hashes no longer exist."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", ["hash1", "hash2", "hash3"])
        db.remote_hashes_remove("origin", ["hash2"])

        assert db.remote_hash_exists("origin", "hash1")
        assert not db.remote_hash_exists("origin", "hash2")
        assert db.remote_hash_exists("origin", "hash3")


def test_remote_index_clear(tmp_path: pathlib.Path) -> None:
    """Clear removes all hashes for a remote."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", ["hash1", "hash2"])
        db.remote_hashes_add("backup", ["hash3", "hash4"])

        db.remote_index_clear("origin")

        assert not db.remote_hash_exists("origin", "hash1")
        assert not db.remote_hash_exists("origin", "hash2")
        assert db.remote_hash_exists("backup", "hash3")
        assert db.remote_hash_exists("backup", "hash4")


def test_remote_hashes_different_remotes_independent(tmp_path: pathlib.Path) -> None:
    """Different remotes have independent hash indexes."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", ["hash1"])
        db.remote_hashes_add("backup", ["hash2"])

        assert db.remote_hash_exists("origin", "hash1")
        assert not db.remote_hash_exists("origin", "hash2")
        assert not db.remote_hash_exists("backup", "hash1")
        assert db.remote_hash_exists("backup", "hash2")


def test_remote_hashes_persistence(tmp_path: pathlib.Path) -> None:
    """Remote hashes persist across DB instances."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", ["persistent_hash"])

    with state.StateDB(db_path) as db:
        assert db.remote_hash_exists("origin", "persistent_hash")


def test_remote_get_url_returns_none_for_unknown(tmp_path: pathlib.Path) -> None:
    """remote_get_url returns None for untracked remote."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        result = db.remote_get_url("unknown")

    assert result is None


def test_remote_set_url_and_get_url_roundtrip(tmp_path: pathlib.Path) -> None:
    """remote_set_url stores URL and remote_get_url retrieves it."""
    db_path = tmp_path
    test_url = "s3://my-bucket/prefix"

    with state.StateDB(db_path) as db:
        db.remote_set_url("origin", test_url)
        result = db.remote_get_url("origin")

    assert result == test_url


def test_remote_set_url_overwrites_existing(tmp_path: pathlib.Path) -> None:
    """remote_set_url overwrites previously stored URL."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        db.remote_set_url("origin", "s3://old-bucket")
        db.remote_set_url("origin", "s3://new-bucket")
        result = db.remote_get_url("origin")

    assert result == "s3://new-bucket"


def test_remote_url_persistence(tmp_path: pathlib.Path) -> None:
    """Remote URLs persist across DB instances."""
    db_path = tmp_path
    test_url = "s3://persistent-bucket/path"

    with state.StateDB(db_path) as db:
        db.remote_set_url("origin", test_url)

    with state.StateDB(db_path) as db:
        assert db.remote_get_url("origin") == test_url


def test_remote_index_clear_also_deletes_url(tmp_path: pathlib.Path) -> None:
    """remote_index_clear also deletes the remote URL entry."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        db.remote_set_url("origin", "s3://bucket")
        db.remote_hashes_add("origin", ["hash1"])
        db.remote_index_clear("origin")

        assert db.remote_get_url("origin") is None
        assert not db.remote_hash_exists("origin", "hash1")


def test_remote_urls_different_remotes_independent(tmp_path: pathlib.Path) -> None:
    """Different remotes have independent URL storage."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        db.remote_set_url("origin", "s3://origin-bucket")
        db.remote_set_url("backup", "s3://backup-bucket")

        assert db.remote_get_url("origin") == "s3://origin-bucket"
        assert db.remote_get_url("backup") == "s3://backup-bucket"


# -----------------------------------------------------------------------------
# Readonly mode tests
# -----------------------------------------------------------------------------


def test_readonly_allows_reads(tmp_path: pathlib.Path) -> None:
    """Readonly mode allows all read operations."""
    db_path = tmp_path
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    # Create data in write mode
    with state.StateDB(db_path) as db:
        db.save(test_file, file_stat, "hash123")
        db.increment_generation("stage:output")
        db.record_dep_generations("stage", {"upstream:out": 1})
        db.remote_hashes_add("origin", ["remote_hash"])

    # Verify reads work in readonly mode
    with state.StateDB(db_path, readonly=True) as db:
        assert db.get(test_file, file_stat) == "hash123"
        assert db.get_generation("stage:output") == 1
        assert db.get_dep_generations("stage") == {"upstream:out": 1}
        assert db.remote_hash_exists("origin", "remote_hash")


def test_open_readonly_uses_the_database_mount_to_classify_an_lmdb_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 7-1: LMDB reports an existing database on a read-only filesystem as plain
    `lmdb.Error`, not `OSError(EROFS)`. `open_readonly` calls statvfs on `state.lmdb` itself
    (its parent can be another, writable mount) and yields None only when that mount is read-only;
    the same error on a writable mount is a damaged database and raises."""
    with state.StateDB(tmp_path):
        pass
    lmdb_path = tmp_path / "state.lmdb"
    statvfs_paths = list[object]()

    def lmdb_error(*_: object, **__: object) -> object:
        raise lmdb.Error("Permission denied")

    class _StatVfs:
        f_flag: int

        def __init__(self, flag: int) -> None:
            self.f_flag = flag

    def readonly_statvfs(path: object) -> _StatVfs:
        statvfs_paths.append(path)
        return _StatVfs(os.ST_RDONLY)

    monkeypatch.setattr(state, "_acquire_shared_env", lmdb_error)
    monkeypatch.setattr(os, "statvfs", cast("Any", readonly_statvfs))

    with state.open_readonly(tmp_path, create=False) as state_db:
        assert state_db is None

    assert statvfs_paths == [lmdb_path]

    def writable_statvfs(_: object) -> _StatVfs:
        return _StatVfs(0)

    monkeypatch.setattr(os, "statvfs", cast("Any", writable_statvfs))
    with (
        pytest.raises(lmdb.Error, match="Permission denied"),
        state.open_readonly(tmp_path, create=False),
    ):
        pass


def test_readonly_blocks_save(tmp_path: pathlib.Path) -> None:
    """Readonly mode blocks save operation."""
    db_path = tmp_path
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    with state.StateDB(db_path) as db:
        db.save(test_file, file_stat, "initial")

    with (
        state.StateDB(db_path, readonly=True) as db,
        pytest.raises(RuntimeError, match="readonly StateDB"),
    ):
        db.save(test_file, file_stat, "new_hash")


def test_readonly_blocks_save_many(tmp_path: pathlib.Path) -> None:
    """Readonly mode blocks save_many operation."""
    db_path = tmp_path
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    with state.StateDB(db_path) as db:
        pass  # Just create

    with (
        state.StateDB(db_path, readonly=True) as db,
        pytest.raises(RuntimeError, match="readonly StateDB"),
    ):
        db.save_many([(test_file, file_stat, "hash")])


def test_readonly_blocks_increment_generation(tmp_path: pathlib.Path) -> None:
    """Readonly mode blocks increment_generation operation."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        pass  # Just create

    with (
        state.StateDB(db_path, readonly=True) as db,
        pytest.raises(RuntimeError, match="readonly StateDB"),
    ):
        db.increment_generation("stage:output")


def test_readonly_blocks_record_dep_generations(tmp_path: pathlib.Path) -> None:
    """Readonly mode blocks record_dep_generations operation."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        pass  # Just create

    with (
        state.StateDB(db_path, readonly=True) as db,
        pytest.raises(RuntimeError, match="readonly StateDB"),
    ):
        db.record_dep_generations("stage", {"upstream:out": 1})


def test_readonly_blocks_remote_hashes_add(tmp_path: pathlib.Path) -> None:
    """Readonly mode blocks remote_hashes_add operation."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        pass  # Just create

    with (
        state.StateDB(db_path, readonly=True) as db,
        pytest.raises(RuntimeError, match="readonly StateDB"),
    ):
        db.remote_hashes_add("origin", ["hash1"])


def test_readonly_blocks_remote_hashes_remove(tmp_path: pathlib.Path) -> None:
    """Readonly mode blocks remote_hashes_remove operation."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", ["hash1"])

    with (
        state.StateDB(db_path, readonly=True) as db,
        pytest.raises(RuntimeError, match="readonly StateDB"),
    ):
        db.remote_hashes_remove("origin", ["hash1"])


def test_readonly_blocks_remote_index_clear(tmp_path: pathlib.Path) -> None:
    """Readonly mode blocks remote_index_clear operation."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", ["hash1"])

    with (
        state.StateDB(db_path, readonly=True) as db,
        pytest.raises(RuntimeError, match="readonly StateDB"),
    ):
        db.remote_index_clear("origin")


def test_readonly_blocks_remote_set_url(tmp_path: pathlib.Path) -> None:
    """Readonly mode blocks remote_set_url operation."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        pass  # Just create

    with (
        state.StateDB(db_path, readonly=True) as db,
        pytest.raises(RuntimeError, match="readonly StateDB"),
    ):
        db.remote_set_url("origin", "s3://bucket")


@pytest.mark.parametrize(
    "files",
    [
        pytest.param({}, id="empty-dir"),
        pytest.param({"lock.mdb": b""}, id="lock-only"),
        pytest.param({"data.mdb": b"", "lock.mdb": b""}, id="empty-data"),
    ],
)
def test_readonly_open_bootstraps_incomplete_lmdb_dir(
    tmp_path: pathlib.Path, files: dict[str, bytes]
) -> None:
    """Readonly open succeeds when state.lmdb/ exists without a usable data.mdb (#454)."""
    lmdb_dir = tmp_path / "state.lmdb"
    lmdb_dir.mkdir()
    for name, content in files.items():
        (lmdb_dir / name).write_bytes(content)

    with state.StateDB(tmp_path, readonly=True) as db:
        assert db.get_generation("stage:output") is None


def _helper_open_readonly_at_barrier(state_dir: str, barrier: Barrier) -> str:
    barrier.wait()
    try:
        with state.StateDB(pathlib.Path(state_dir), readonly=True):
            return "ok"
    except lmdb.Error as e:
        return f"{type(e).__name__}: {e}"


def test_concurrent_readonly_opens_of_new_state_dir(tmp_path: pathlib.Path) -> None:
    """Processes that readonly-open a fresh state dir at the same instant all succeed.

    The first opener creates state.lmdb/ before data.mdb, so a check on the directory
    alone sends the others into a readonly open of an incomplete environment (#228, #454).
    """
    n_procs = 8
    ctx = multiprocessing.get_context("spawn")
    failures = list[str]()
    with ctx.Manager() as manager, ctx.Pool(n_procs) as pool:
        for i in range(10):
            barrier = manager.Barrier(n_procs, timeout=60)
            state_dir = str(tmp_path / f"round{i}" / ".pivot")
            results = pool.starmap(
                _helper_open_readonly_at_barrier, [(state_dir, barrier)] * n_procs
            )
            failures.extend(r for r in results if r != "ok")

    assert failures == []


# -----------------------------------------------------------------------------
# apply_deferred_writes tests
# -----------------------------------------------------------------------------


def test_apply_deferred_writes_dep_generations(tmp_path: pathlib.Path) -> None:
    """apply_deferred_writes records dependency generations."""
    db_path = tmp_path
    deferred: DeferredWrites = {"dep_generations": {"upstream:out1": 5, "upstream:out2": 3}}

    with state.StateDB(db_path) as db:
        db.apply_deferred_writes("my_stage", [], deferred)
        result = db.get_dep_generations("my_stage")

    assert result == {"upstream:out1": 5, "upstream:out2": 3}


def test_apply_deferred_writes_output_generations(tmp_path: pathlib.Path) -> None:
    """apply_deferred_writes increments output generations."""
    db_path = tmp_path
    output1 = "stage:output1"
    output2 = "stage:output2"
    deferred: DeferredWrites = {"increment_outputs": True}

    with state.StateDB(db_path) as db:
        # First apply - outputs should be at generation 1
        db.apply_deferred_writes("stage", [output1, output2], deferred)
        assert db.get_generation(output1) == 1
        assert db.get_generation(output2) == 1

        # Second apply - outputs should increment to 2
        db.apply_deferred_writes("stage", [output1, output2], deferred)
        assert db.get_generation(output1) == 2
        assert db.get_generation(output2) == 2


def test_apply_deferred_writes_skips_output_increment_when_flag_absent(
    tmp_path: pathlib.Path,
) -> None:
    """Output generations should NOT be incremented when increment_outputs is absent."""
    db_path = tmp_path
    output1 = "stage:output1"
    deferred: DeferredWrites = {"dep_generations": {"upstream:out": 5}}

    with state.StateDB(db_path) as db:
        db.apply_deferred_writes("stage", [output1], deferred)
        assert db.get_generation(output1) is None
        assert db.get_dep_generations("stage") == {"upstream:out": 5}


def test_apply_deferred_writes_file_hash_entries(tmp_path: pathlib.Path) -> None:
    """apply_deferred_writes stores file hash entries."""
    db_path = tmp_path
    file_path = tmp_path / "input.txt"
    file_path.write_text("data")
    file_stat = file_path.stat()
    file_hash, _ = cache.hash_file(file_path)
    deferred: DeferredWrites = {
        "file_hash_entries": [
            (
                str(file_path),
                file_stat.st_mtime_ns,
                file_stat.st_size,
                file_stat.st_dev,
                file_stat.st_ino,
                file_hash,
            )
        ]
    }

    with state.StateDB(db_path) as db:
        db.apply_deferred_writes("stage", [], deferred)
        result = db.get(file_path, file_stat)

    assert result == file_hash


def _helper_stat_on_device(fs_stat: os.stat_result, device: int) -> os.stat_result:
    """``fs_stat`` with only its device changed: the same inode number, size and mtime_ns, as
    a file on another filesystem can have."""
    visible = list(fs_stat)
    visible[stat.ST_DEV] = device
    hidden = {
        name: getattr(fs_stat, name)
        for name in (
            "st_atime",
            "st_mtime",
            "st_ctime",
            "st_atime_ns",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    }
    return os.stat_result(visible, hidden)


def test_state_cache_miss_on_another_device(tmp_path: pathlib.Path) -> None:
    """A file on another filesystem can repeat the inode number, size and mtime_ns of the file
    an entry recorded (a directory symlink re-pointed at another mount). The entry records the
    device too, so it does not match and the file is read again."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()
    elsewhere = _helper_stat_on_device(file_stat, file_stat.st_dev + 1)
    assert (elsewhere.st_ino, elsewhere.st_size, elsewhere.st_mtime_ns) == (
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
    )

    with state.StateDB(tmp_path / ".pivot") as db:
        db.save(test_file, file_stat, "abc123")
        db.save_many([(tmp_path / "other.txt", file_stat, "def456")])
        db.apply_deferred_writes(
            "stage",
            [],
            {
                "file_hash_entries": [
                    (
                        str(tmp_path / "deferred.txt"),
                        file_stat.st_mtime_ns,
                        file_stat.st_size,
                        file_stat.st_dev,
                        file_stat.st_ino,
                        "0123456789abcdef",
                    )
                ]
            },
        )
        assert db.get(test_file, file_stat) == "abc123", "The same device still matches"
        assert db.get(test_file, elsewhere) is None, "save records the device"
        assert db.get(tmp_path / "other.txt", elsewhere) is None, "save_many records the device"
        assert db.get_many([(tmp_path / "deferred.txt", elsewhere)]) == {
            tmp_path / "deferred.txt": None
        }, "apply_deferred_writes records the device"


def test_state_entry_written_before_the_device_field_never_matches(
    tmp_path: pathlib.Path,
) -> None:
    """An entry an earlier pivot wrote holds (mtime_ns, size, inode) and the hash, with no
    device. Read with the device field it never matches a file, so the file is read once again
    and recorded anew."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()
    old_value = (
        struct.pack(">QQQ", file_stat.st_mtime_ns, file_stat.st_size, file_stat.st_ino)
        + b"0123456789abcdef"
    )

    with state.StateDB(tmp_path / ".pivot") as db:
        db.put_raw(b"hash:" + os.fsencode(test_file.absolute()), old_value)
        assert db.get(test_file, file_stat) is None


def test_apply_deferred_writes_run_cache(tmp_path: pathlib.Path) -> None:
    """apply_deferred_writes writes run cache entry."""
    db_path = tmp_path
    run_cache_entry = run_history.RunCacheEntry(
        run_id="test_run_123",
        output_hashes=[run_history.OutputHashEntry(path="/output.csv", hash="abc123")],
    )
    deferred: DeferredWrites = {
        "run_cache_input_hash": "input_hash_xyz",
        "run_cache_entry": run_cache_entry,
    }

    with state.StateDB(db_path) as db:
        db.apply_deferred_writes("my_stage", [], deferred)
        result = db.lookup_run_cache("my_stage", "input_hash_xyz")

    assert result is not None
    assert result["run_id"] == "test_run_123"
    assert len(result["output_hashes"]) == 1
    assert result["output_hashes"][0]["hash"] == "abc123"


def test_apply_deferred_writes_all_fields(tmp_path: pathlib.Path) -> None:
    """apply_deferred_writes handles all fields atomically."""
    db_path = tmp_path
    output_path = "stage:output"
    input_path = tmp_path / "input.csv"
    input_path.write_text("data")
    input_stat = input_path.stat()
    input_hash, _ = cache.hash_file(input_path)
    run_cache_entry = run_history.RunCacheEntry(
        run_id="run_456",
        output_hashes=[run_history.OutputHashEntry(path=str(output_path), hash="def456")],
    )
    deferred: DeferredWrites = {
        "dep_generations": {"upstream:out": 10},
        "run_cache_input_hash": "input_abc",
        "run_cache_entry": run_cache_entry,
        "increment_outputs": True,
        "file_hash_entries": [
            (
                str(input_path),
                input_stat.st_mtime_ns,
                input_stat.st_size,
                input_stat.st_dev,
                input_stat.st_ino,
                input_hash,
            )
        ],
    }

    with state.StateDB(db_path) as db:
        db.apply_deferred_writes("stage", [output_path], deferred)

        # Verify all writes applied
        assert db.get_dep_generations("stage") == {"upstream:out": 10}
        assert db.get_generation(output_path) == 1
        result = db.lookup_run_cache("stage", "input_abc")
        assert result is not None
        assert result["run_id"] == "run_456"
        assert db.get(input_path, input_stat) == input_hash


def test_apply_deferred_writes_empty(tmp_path: pathlib.Path) -> None:
    """apply_deferred_writes handles empty deferred dict."""
    db_path = tmp_path
    deferred: DeferredWrites = {}

    with state.StateDB(db_path) as db:
        # Should not raise
        db.apply_deferred_writes("stage", [], deferred)


def test_apply_deferred_writes_readonly_blocked(tmp_path: pathlib.Path) -> None:
    """apply_deferred_writes blocked in readonly mode."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        pass  # Create database

    deferred: DeferredWrites = {"dep_generations": {"/dep.csv": 1}}

    with (
        state.StateDB(db_path, readonly=True) as db,
        pytest.raises(RuntimeError, match="readonly StateDB"),
    ):
        db.apply_deferred_writes("stage", [], deferred)


def test_apply_deferred_writes_path_too_long_dep(tmp_path: pathlib.Path) -> None:
    """apply_deferred_writes raises PathTooLongError for long dep paths."""
    db_path = tmp_path
    long_path = "/" + "d" * 600  # Exceeds 511 byte limit
    deferred: DeferredWrites = {"dep_generations": {long_path: 1}}

    with state.StateDB(db_path) as db, pytest.raises(state.PathTooLongError):
        db.apply_deferred_writes("stage", [], deferred)


def test_apply_deferred_writes_path_too_long_output(tmp_path: pathlib.Path) -> None:
    """apply_deferred_writes raises PathTooLongError for long output paths."""
    db_path = tmp_path
    long_output = "stage:" + "o" * 600
    deferred: DeferredWrites = {"increment_outputs": True}

    with state.StateDB(db_path) as db, pytest.raises(state.PathTooLongError):
        db.apply_deferred_writes("stage", [long_output], deferred)


def test_no_commit_record_is_kept_until_a_lock_clears_it(tmp_path: pathlib.Path) -> None:
    """The stage's record stays through deferred writes that write no lock for it, until a
    deferred write for its written lock clears it; other stages have none."""
    cleared: DeferredWrites = {"clear_no_commit_record": True}

    with state.StateDB(tmp_path) as db:
        assert db.get_no_commit_record("s") is None
        db.add_no_commit_pvt_members("s", {"data/s/a.pvt": "aaa"})
        assert db.get_no_commit_record("s") == {"pvt_members": {"data/s/a.pvt": "aaa"}}
        assert db.get_no_commit_record("other") is None
        db.apply_deferred_writes("s", [], {"dep_generations": {"dep": 1}})
        assert db.get_no_commit_record("s") == {"pvt_members": {"data/s/a.pvt": "aaa"}}
        db.apply_deferred_writes("other", [], cleared)
        assert db.get_no_commit_record("s") == {"pvt_members": {"data/s/a.pvt": "aaa"}}
        db.apply_deferred_writes("s", [], cleared)
        assert db.get_no_commit_record("s") is None
        db.apply_deferred_writes("s", [], cleared)
        assert db.get_no_commit_record("s") is None


@pytest.mark.parametrize(
    "stored",
    [
        pytest.param(b"\xff not json", id="not-json"),
        pytest.param(b"[1, 2]", id="not-an-object"),
    ],
)
def test_unreadable_no_commit_record_reads_as_none(tmp_path: pathlib.Path, stored: bytes) -> None:
    """A record this Pivot cannot read at all lists no member, so nothing is claimed through it."""
    _helper_store_no_commit_record(tmp_path, stored)

    with state.StateDB(tmp_path, readonly=True) as db:
        assert db.get_no_commit_record("s") is None


_RUN_FIELDS = {"code_manifest": {"func:tree": "a1b2"}, "output_hashes": {"b/tree": "c3d4"}}
_MEMBERS = {"pvt_members": {"data/s/a.pvt": "aaa"}}


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        pytest.param({"pvt_members": {"data/s/a.pvt": 1}}, {}, id="hash-not-a-string"),
        pytest.param({"pvt_members": ["data/s/a.pvt"]}, {}, id="members-not-a-mapping"),
        pytest.param(
            {**_RUN_FIELDS, "pvt_members": ["data/s/a.pvt"]},
            _RUN_FIELDS,
            id="members-malformed-run-fields-valid",
        ),
        pytest.param(
            {"code_manifest": "a1b2", "output_hashes": {"b/tree": 3}, **_MEMBERS},
            _MEMBERS,
            id="run-fields-malformed-members-valid",
        ),
        pytest.param(
            {"code_manifest": ["func:tree"], "output_hashes": _RUN_FIELDS["output_hashes"]},
            {"output_hashes": _RUN_FIELDS["output_hashes"]},
            id="code-manifest-malformed",
        ),
        pytest.param(
            {"code_manifest": _RUN_FIELDS["code_manifest"], "output_hashes": None, **_MEMBERS},
            {"code_manifest": _RUN_FIELDS["code_manifest"], **_MEMBERS},
            id="output-hashes-malformed",
        ),
        pytest.param(
            {"later_key": [1], "pvt_members": 5},
            {"later_key": [1]},
            id="other-key-kept-as-stored",
        ),
    ],
)
def test_each_malformed_field_is_left_out_and_hides_no_other(
    tmp_path: pathlib.Path, stored: dict[str, object], expected: dict[str, object]
) -> None:
    """The reader checks `code_manifest`, `output_hashes` and `pvt_members` each on its own: a
    malformed one is left out (a malformed `pvt_members` lists no member, so nothing is claimed
    through it), every valid one still reads, and any other key is returned as stored."""
    _helper_store_no_commit_record(tmp_path, json.dumps(stored).encode())

    with state.StateDB(tmp_path, readonly=True) as db:
        assert db.get_no_commit_record("s") == expected


def test_record_holding_only_run_fields_reads_as_present_and_survives_adding_members(
    tmp_path: pathlib.Path,
) -> None:
    """The `nocommit:` record is shared with a `--no-commit` run's own writer, which keeps
    `code_manifest` and `output_hashes` in it. A record holding only those reads as present with
    them unchanged, and recording `*.pvt` members keeps them as stored."""
    _helper_store_no_commit_record(tmp_path, json.dumps(_RUN_FIELDS).encode())
    members = {"pvt_members": {"data/s/a.pvt": "aaa"}}

    with state.StateDB(tmp_path) as db:
        assert db.get_no_commit_record("s") == _RUN_FIELDS
        db.add_no_commit_pvt_members("s", {"data/s/a.pvt": "aaa"})
        assert db.get_no_commit_record("s") == {**_RUN_FIELDS, **members}
    env = lmdb.open(str(tmp_path / "state.lmdb"))
    with env.begin() as txn:
        stored = txn.get(b"nocommit:s")
    env.close()
    assert stored is not None
    assert json.loads(stored) == {**_RUN_FIELDS, **members}


def _helper_store_no_commit_record(tmp_path: pathlib.Path, stored: bytes) -> None:
    with state.StateDB(tmp_path):
        pass
    env = lmdb.open(str(tmp_path / "state.lmdb"))
    with env.begin(write=True) as txn:
        txn.put(b"nocommit:s", stored)
    env.close()


def test_adding_pvt_members_keeps_the_record_and_its_other_fields(tmp_path: pathlib.Path) -> None:
    """Members Pivot records before it writes them join the stage's record: earlier members stay,
    a member written again takes its new hash, a field another writer keeps in the record (the
    shared `nocommit:` record) stays, and a written lock still clears the whole record."""
    with state.StateDB(tmp_path) as db:
        db.add_no_commit_pvt_members("s", {"data/s/b.pvt": "bbb"})
        assert db.get_no_commit_record("s") == {"pvt_members": {"data/s/b.pvt": "bbb"}}
        db.add_no_commit_pvt_members("s", {"data/s/a.pvt": "aaa", "data/s/b.pvt": "ccc"})
        assert db.get_no_commit_record("s") == {
            "pvt_members": {"data/s/a.pvt": "aaa", "data/s/b.pvt": "ccc"}
        }
        assert db.get_no_commit_record("other") is None
    env = lmdb.open(str(tmp_path / "state.lmdb"))
    with env.begin(write=True) as txn:
        txn.put(b"nocommit:s", b'{"output_hashes": {"s": "h"}, "pvt_members": {"x.pvt": "x"}}')
    env.close()

    with state.StateDB(tmp_path) as db:
        db.add_no_commit_pvt_members("s", {"data/s/a.pvt": "aaa"})
    env = lmdb.open(str(tmp_path / "state.lmdb"))
    with env.begin() as txn:
        stored = txn.get(b"nocommit:s")
    env.close()
    assert stored is not None
    assert json.loads(stored) == {
        "output_hashes": {"s": "h"},
        "pvt_members": {"data/s/a.pvt": "aaa", "x.pvt": "x"},
    }

    with state.StateDB(tmp_path) as db:
        db.apply_deferred_writes("s", [], {"clear_no_commit_record": True})
        assert db.get_no_commit_record("s") is None


def test_adding_pvt_members_replaces_an_unreadable_record(tmp_path: pathlib.Path) -> None:
    """A record this Pivot cannot read claims nothing; recording members starts a new one."""
    with state.StateDB(tmp_path):
        pass
    env = lmdb.open(str(tmp_path / "state.lmdb"))
    with env.begin(write=True) as txn:
        txn.put(b"nocommit:s", b'{"pvt_members": ["data/s/a.pvt"]}')
    env.close()

    with state.StateDB(tmp_path) as db:
        db.add_no_commit_pvt_members("s", {"data/s/b.pvt": "bbb"})
        assert db.get_no_commit_record("s") == {"pvt_members": {"data/s/b.pvt": "bbb"}}


def test_readonly_statedb_refuses_to_add_pvt_members(tmp_path: pathlib.Path) -> None:
    with state.StateDB(tmp_path):
        pass
    with (
        state.StateDB(tmp_path, readonly=True) as db,
        pytest.raises(RuntimeError, match="readonly"),
    ):
        db.add_no_commit_pvt_members("s", {"data/s/b.pvt": "bbb"})


# -----------------------------------------------------------------------------
# AST hash cache tests
# -----------------------------------------------------------------------------

# Test constants for AST hash cache tests
_TEST_PY_VERSION = "3.13"
_TEST_SCHEMA_VERSION = 1
_TEST_DIGEST = "0123456789abcdef"


def _helper_ast_entry(
    hash_hex: str,
    *,
    rel_path: str = "src/stages.py",
    source_digest: str = _TEST_DIGEST,
    qualname: str = "my_func",
    py_version: str = _TEST_PY_VERSION,
    schema_version: int = _TEST_SCHEMA_VERSION,
) -> tuple[str, str, str, str, int, str]:
    return (rel_path, source_digest, qualname, py_version, schema_version, hash_hex)


def test_ast_hash_cache_roundtrip(tmp_path: pathlib.Path) -> None:
    """Save and retrieve AST hash."""
    with state.StateDB(tmp_path) as db:
        db.save_ast_hash_many([_helper_ast_entry("abc123def456")])
        result = db.get_ast_hash(
            "src/stages.py", _TEST_DIGEST, "my_func", _TEST_PY_VERSION, _TEST_SCHEMA_VERSION
        )

    assert result == "abc123def456"


@pytest.mark.parametrize(
    ("field", "lookup"),
    [
        pytest.param(
            "source_digest",
            ("src/stages.py", "fedcba9876543210", "my_func", "3.13", 1),
            id="source_digest",
        ),
        pytest.param(
            "qualname", ("src/stages.py", _TEST_DIGEST, "other_func", "3.13", 1), id="qualname"
        ),
        pytest.param(
            "py_version", ("src/stages.py", _TEST_DIGEST, "my_func", "3.12", 1), id="py_version"
        ),
        pytest.param(
            "schema_version",
            ("src/stages.py", _TEST_DIGEST, "my_func", "3.13", 2),
            id="schema_version",
        ),
    ],
)
def test_ast_hash_cache_misses_when_a_key_field_changes(
    tmp_path: pathlib.Path, field: str, lookup: tuple[str, str, str, str, int]
) -> None:
    """A different source content digest, qualname, Python version or schema version is a
    different entry: a file edited in place is a miss whatever its mtime, size and inode."""
    with state.StateDB(tmp_path) as db:
        db.save_ast_hash_many([_helper_ast_entry("abc123def456")])
        result = db.get_ast_hash(*lookup)

    assert result is None, field


def test_ast_hash_cache_persistence(tmp_path: pathlib.Path) -> None:
    """AST hashes persist across DB instances."""
    with state.StateDB(tmp_path) as db:
        db.save_ast_hash_many([_helper_ast_entry("persistent_hash")])

    with state.StateDB(tmp_path) as db:
        result = db.get_ast_hash(
            "src/stages.py", _TEST_DIGEST, "my_func", _TEST_PY_VERSION, _TEST_SCHEMA_VERSION
        )

    assert result == "persistent_hash"


def test_ast_hash_cache_multiple_functions(tmp_path: pathlib.Path) -> None:
    """Multiple functions in same file have independent hashes."""
    qualnames = {"func_a": "hash_a", "func_b": "hash_b", "MyClass.method": "hash_c"}
    with state.StateDB(tmp_path) as db:
        db.save_ast_hash_many(
            [_helper_ast_entry(hash_hex, qualname=name) for name, hash_hex in qualnames.items()]
        )

        found = {
            name: db.get_ast_hash(
                "src/stages.py", _TEST_DIGEST, name, _TEST_PY_VERSION, _TEST_SCHEMA_VERSION
            )
            for name in qualnames
        }

    assert found == qualnames


def test_ast_hash_cache_update_existing(tmp_path: pathlib.Path) -> None:
    """Saving same key updates the cached hash."""
    with state.StateDB(tmp_path) as db:
        db.save_ast_hash_many([_helper_ast_entry("old_hash")])
        db.save_ast_hash_many([_helper_ast_entry("new_hash")])
        result = db.get_ast_hash(
            "src/stages.py", _TEST_DIGEST, "my_func", _TEST_PY_VERSION, _TEST_SCHEMA_VERSION
        )

    assert result == "new_hash"


def test_ast_hash_cache_many(tmp_path: pathlib.Path) -> None:
    """Batch save works correctly."""
    files = {"src/a.py": "hash_a", "src/b.py": "hash_b", "src/c.py": "hash_c"}
    with state.StateDB(tmp_path) as db:
        db.save_ast_hash_many(
            [_helper_ast_entry(hash_hex, rel_path=rel_path) for rel_path, hash_hex in files.items()]
        )

        found = {
            rel_path: db.get_ast_hash(
                rel_path, _TEST_DIGEST, "my_func", _TEST_PY_VERSION, _TEST_SCHEMA_VERSION
            )
            for rel_path in files
        }

    assert found == files


def test_ast_hash_cache_many_empty(tmp_path: pathlib.Path) -> None:
    """Batch save with empty list doesn't error."""
    with state.StateDB(tmp_path) as db:
        db.save_ast_hash_many([])  # Should not raise


@pytest.mark.parametrize(
    ("rel_path", "qualname"),
    [
        pytest.param("a" * 600, "func", id="long-key"),
        pytest.param("", "func", id="empty-rel-path"),
        pytest.param("src/stages.py", "", id="empty-qualname"),
        pytest.param("src/sta\x00ges.py", "func", id="null-byte-in-path"),
        pytest.param("src/stages.py", "my\x00func", id="null-byte-in-qualname"),
    ],
)
def test_ast_hash_cache_skips_invalid_keys(
    tmp_path: pathlib.Path, rel_path: str, qualname: str
) -> None:
    """An entry whose key is too long, empty or holds a null byte is silently skipped, and a
    lookup of it finds nothing."""
    with state.StateDB(tmp_path) as db:
        db.save_ast_hash_many([_helper_ast_entry("hash", rel_path=rel_path, qualname=qualname)])
        result = db.get_ast_hash(
            rel_path, _TEST_DIGEST, qualname, _TEST_PY_VERSION, _TEST_SCHEMA_VERSION
        )

    assert result is None


def test_ast_hash_cache_readonly_blocked(tmp_path: pathlib.Path) -> None:
    """Readonly mode blocks save operations."""
    with state.StateDB(tmp_path) as db:
        pass  # Create database

    with (
        state.StateDB(tmp_path, readonly=True) as db,
        pytest.raises(RuntimeError, match="readonly StateDB"),
    ):
        db.save_ast_hash_many([_helper_ast_entry("hash")])


def test_ast_hash_cache_readonly_allows_reads(tmp_path: pathlib.Path) -> None:
    """Readonly mode allows reading AST hashes."""
    with state.StateDB(tmp_path) as db:
        db.save_ast_hash_many([_helper_ast_entry("hash123")])

    with state.StateDB(tmp_path, readonly=True) as db:
        result = db.get_ast_hash(
            "src/stages.py", _TEST_DIGEST, "my_func", _TEST_PY_VERSION, _TEST_SCHEMA_VERSION
        )

    assert result == "hash123"


def test_clear_fingerprint_caches_deletes_all_ast_hash_entries(tmp_path: pathlib.Path) -> None:
    """clear_fingerprint_caches removes all fp: prefixed entries."""
    files = {"src/a.py": "hash_a", "src/b.py": "hash_b", "src/c.py": "hash_c"}

    def lookup(db: state.StateDB) -> dict[str, str | None]:
        return {
            rel_path: db.get_ast_hash(
                rel_path, _TEST_DIGEST, "my_func", _TEST_PY_VERSION, _TEST_SCHEMA_VERSION
            )
            for rel_path in files
        }

    with state.StateDB(tmp_path) as db:
        db.save_ast_hash_many(
            [_helper_ast_entry(hash_hex, rel_path=rel_path) for rel_path, hash_hex in files.items()]
        )
        assert lookup(db) == files

        deleted = db.clear_fingerprint_caches()

        assert deleted == 3
        assert lookup(db) == dict.fromkeys(files)


def test_clear_fingerprint_caches_returns_zero_when_empty(tmp_path: pathlib.Path) -> None:
    """clear_fingerprint_caches returns 0 when no entries exist."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        deleted = db.clear_fingerprint_caches()

    assert deleted == 0


def test_clear_fingerprint_caches_keeps_other_state(tmp_path: pathlib.Path) -> None:
    """clear_fingerprint_caches deletes fp: and sm: entries but no other data."""
    db_path = tmp_path
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()
    output_identity = "stage:output"

    with state.StateDB(db_path) as db:
        # Add various entry types
        db.save_ast_hash_many([_helper_ast_entry("hash_a")])
        db.save(test_file, file_stat, "file_hash")
        db.increment_generation(output_identity)
        db.record_dep_generations("my_stage", {"upstream:out": 5})
        db.put_raw(b"sm:my_stage\x003.13\x007", b"{}")

        deleted = db.clear_fingerprint_caches()
        assert deleted == 2
        assert db.get_raw(b"sm:my_stage\x003.13\x007") is None

        # Verify other entries are untouched
        assert db.get(test_file, file_stat) == "file_hash"
        assert db.get_generation(output_identity) == 1
        assert db.get_dep_generations("my_stage") == {"upstream:out": 5}


def test_clear_fingerprint_caches_readonly_blocked(tmp_path: pathlib.Path) -> None:
    """clear_fingerprint_caches blocked in readonly mode."""
    with state.StateDB(tmp_path) as db:
        db.save_ast_hash_many([_helper_ast_entry("hash_a")])

    with (
        state.StateDB(tmp_path, readonly=True) as db,
        pytest.raises(RuntimeError, match="readonly StateDB"),
    ):
        db.clear_fingerprint_caches()


# -----------------------------------------------------------------------------
# Raw key-value access tests (stage manifest cache)
# -----------------------------------------------------------------------------


def test_stage_manifest_roundtrip(tmp_path: pathlib.Path) -> None:
    """Save and retrieve a stage manifest."""
    db_path = tmp_path
    key = "sm:my_stage\x003.13\x001"
    manifest = {"self:train": "aabb", "func:helper": "ccdd"}
    sources = {"src/train.py": "0123456789abcdef", "src/helper.py": "fedcba9876543210"}
    value = json.dumps({"m": manifest, "s": sources}, separators=(",", ":"))

    with state.StateDB(db_path) as db:
        db.put_raw(key.encode(), value.encode())
        result = db.get_raw(key.encode())

    assert result is not None
    assert json.loads(result.decode()) == {"m": manifest, "s": sources}


def test_stage_manifest_not_found(tmp_path: pathlib.Path) -> None:
    """Returns None for unknown key."""
    db_path = tmp_path
    with state.StateDB(db_path) as db:
        result = db.get_raw(b"sm:nonexistent\x003.13\x001")
    assert result is None


def test_put_raw_readonly_blocked(tmp_path: pathlib.Path) -> None:
    """put_raw blocked in readonly mode."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        pass  # Create database

    with (
        state.StateDB(db_path, readonly=True) as db,
        pytest.raises(RuntimeError, match="readonly StateDB"),
    ):
        db.put_raw(b"sm:test\x003.13\x001", b"value")


def test_put_raw_many_readonly_blocked(tmp_path: pathlib.Path) -> None:
    """put_raw_many blocked in readonly mode."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        pass  # Create database

    with (
        state.StateDB(db_path, readonly=True) as db,
        pytest.raises(RuntimeError, match="readonly StateDB"),
    ):
        db.put_raw_many([(b"sm:test\x003.13\x001", b"value")])


def test_put_raw_key_too_long(tmp_path: pathlib.Path) -> None:
    """put_raw raises PathTooLongError for oversized keys."""
    db_path = tmp_path
    long_key = b"k" * 512  # Exceeds 511 byte limit

    with state.StateDB(db_path) as db, pytest.raises(state.PathTooLongError):
        db.put_raw(long_key, b"value")


def test_put_raw_many_skips_oversized_keys(tmp_path: pathlib.Path) -> None:
    """put_raw_many silently skips entries with oversized keys."""
    db_path = tmp_path
    long_key = b"k" * 512  # Exceeds 511 byte limit
    normal_key = b"sm:normal\x003.13\x001"

    with state.StateDB(db_path) as db:
        db.put_raw_many(
            [
                (long_key, b"should_be_skipped"),
                (normal_key, b"should_be_saved"),
            ]
        )
        # Oversized key was skipped
        assert db.get_raw(long_key) is None
        # Normal key was saved
        assert db.get_raw(normal_key) == b"should_be_saved"


def test_put_raw_many_empty(tmp_path: pathlib.Path) -> None:
    """put_raw_many with empty list does not error."""
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        db.put_raw_many([])  # Should not raise


def test_put_raw_many_persistence(tmp_path: pathlib.Path) -> None:
    """put_raw_many entries persist across DB instances."""
    db_path = tmp_path
    entries = [
        (b"sm:stage_a\x003.13\x001", b'{"m":{"self:a":"hash_a"},"s":{}}'),
        (b"sm:stage_b\x003.13\x001", b'{"m":{"self:b":"hash_b"},"s":{}}'),
    ]

    with state.StateDB(db_path) as db:
        db.put_raw_many(entries)

    with state.StateDB(db_path, readonly=True) as db:
        for key, expected_value in entries:
            assert db.get_raw(key) == expected_value


def test_get_raw_after_close_raises(tmp_path: pathlib.Path) -> None:
    """get_raw on closed StateDB raises RuntimeError."""
    db_path = tmp_path

    db = state.StateDB(db_path)
    db.close()

    with pytest.raises(RuntimeError, match="closed StateDB"):
        db.get_raw(b"sm:test\x003.13\x001")


def test_put_raw_overwrites_existing(tmp_path: pathlib.Path) -> None:
    """put_raw with same key overwrites the previous value."""
    db_path = tmp_path
    key = b"sm:overwrite_test\x003.13\x001"

    with state.StateDB(db_path) as db:
        db.put_raw(key, b"old_value")
        db.put_raw(key, b"new_value")
        assert db.get_raw(key) == b"new_value"


def test_apply_deferred_writes_skips_output_increment_when_flag_false(
    tmp_path: pathlib.Path,
) -> None:
    """Output generations should NOT be incremented when increment_outputs is explicitly False.

    This covers the case where the worker explicitly sets increment_outputs=False
    (as opposed to the key being absent entirely). Both should skip incrementing.
    """
    db_path = tmp_path
    output1 = "stage:output1"
    deferred: DeferredWrites = {"increment_outputs": False, "dep_generations": {"upstream:out": 5}}

    with state.StateDB(db_path) as db:
        db.apply_deferred_writes("stage", [output1], deferred)
        assert db.get_generation(output1) is None, (
            "Output generation should not be incremented when flag is explicitly False"
        )
        assert db.get_dep_generations("stage") == {"upstream:out": 5}


def test_apply_deferred_writes_empty_output_paths_with_flag_true(
    tmp_path: pathlib.Path,
) -> None:
    """increment_outputs=True with empty output_paths should not error.

    A stage with no declared outputs but increment_outputs=True should
    simply be a no-op for output generation incrementing.
    """
    db_path = tmp_path
    deferred: DeferredWrites = {"increment_outputs": True}

    with state.StateDB(db_path) as db:
        # Should not raise
        db.apply_deferred_writes("stage", [], deferred)


_RAN = {"code_manifest": {"func:helpers.plain": "1f2e"}, "output_hashes": {"p/use_plain": "9a8b"}}
# The key another feature writes into the same record: `*.pvt` directory members.
_PVT = {"pvt_members": {"data/out/a.pvt": "77aa"}}


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        pytest.param(_RAN, _RAN, id="this-pivots-fields"),
        pytest.param(_PVT, _PVT, id="another-field-only"),
        pytest.param({**_RAN, **_PVT}, {**_RAN, **_PVT}, id="another-field-too"),
        pytest.param(
            {"code_manifest": _RAN["code_manifest"], **_PVT},
            {"code_manifest": _RAN["code_manifest"], **_PVT},
            id="manifest-without-outputs",
        ),
        pytest.param(
            {**_RAN, "code_manifest": {"func:helpers.plain": 1}},
            {"output_hashes": _RAN["output_hashes"]},
            id="malformed-manifest",
        ),
        pytest.param(
            {**_RAN, "output_hashes": ["p/use_plain"], **_PVT},
            {"code_manifest": _RAN["code_manifest"], **_PVT},
            id="malformed-outputs-and-another-field",
        ),
        pytest.param({**_RAN, "pvt_members": ["data/out/a.pvt"]}, _RAN, id="malformed-pvt-members"),
        pytest.param({**_RAN, "pvt_members": 7}, _RAN, id="pvt-members-not-an-object"),
        pytest.param(
            {"code_manifest": 7, "output_hashes": {"p/use_plain": 9}, **_PVT},
            _PVT,
            id="malformed-run-keys-and-valid-pvt-members",
        ),
        pytest.param(
            {**_RAN, "later_field": [1]}, {**_RAN, "later_field": [1]}, id="unknown-key-as-stored"
        ),
    ],
)
def test_a_no_commit_record_reads_each_valid_field_it_knows(
    tmp_path: pathlib.Path, stored: object, expected: object
) -> None:
    """Each key the record declares (the run's code manifest and output hashes, and the
    `pvt_members` another feature writes) is validated on its own: a malformed one is dropped
    and hides no other. A key the record does not declare reads as stored, and a record that
    holds only another feature's key reads as present, never as absent."""
    with state.StateDB(tmp_path) as db:
        db.put_raw(b"nocommit:p/use_plain", json.dumps(stored).encode())

        assert db.get_no_commit_record("p/use_plain") == expected


_OLDER_RUN = {
    "code_manifest": {"func:helpers.plain": "0000"},
    "output_hashes": {"p/use_plain": "11"},
}


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        pytest.param(None, _RAN, id="no-record"),
        pytest.param(json.dumps(_PVT).encode(), {**_RAN, **_PVT}, id="another-field-only"),
        pytest.param(
            json.dumps({**_OLDER_RUN, **_PVT}).encode(),
            {**_RAN, **_PVT},
            id="an-older-run-and-another-field",
        ),
        pytest.param(b"not json", _RAN, id="unreadable-record"),
    ],
)
def test_applying_a_no_commit_record_replaces_its_keys_and_keeps_every_other(
    tmp_path: pathlib.Path, stored: bytes | None, expected: dict[str, object]
) -> None:
    """Another feature writes its keys into the stage's `nocommit:` record during the run, before
    the coordinator applies the run's record (the worker adds each `*.pvt` member before it writes
    the file). Applying the run's record replaces the code manifest and output hashes, an older
    run's included, and keeps every other key."""
    deferred: DeferredWrites = {
        "no_commit_record": {
            "code_manifest": _RAN["code_manifest"],
            "output_hashes": _RAN["output_hashes"],
        }
    }
    with state.StateDB(tmp_path) as db:
        if stored is not None:
            db.put_raw(b"nocommit:p/use_plain", stored)

        db.apply_deferred_writes("p/use_plain", [], deferred)

        raw = db.get_raw(b"nocommit:p/use_plain")
        assert raw is not None and json.loads(raw) == expected
        assert db.get_no_commit_record("p/use_plain") == expected


def test_a_written_lock_removes_the_whole_no_commit_record(tmp_path: pathlib.Path) -> None:
    """A lock written for the stage supersedes the record, every feature's keys included."""
    with state.StateDB(tmp_path) as db:
        db.put_raw(b"nocommit:p/use_plain", json.dumps({**_RAN, **_PVT}).encode())

        db.apply_deferred_writes("p/use_plain", [], {"clear_no_commit_record": True})

        assert db.get_no_commit_record("p/use_plain") is None


@pytest.mark.parametrize(
    "value", [None, b"[]", b"not json"], ids=["absent", "not-an-object", "not-json"]
)
def test_a_no_commit_record_is_none_when_absent_or_not_a_json_object(
    tmp_path: pathlib.Path, value: bytes | None
) -> None:
    with state.StateDB(tmp_path) as db:
        if value is not None:
            db.put_raw(b"nocommit:p/use_plain", value)

        assert db.get_no_commit_record("p/use_plain") is None
