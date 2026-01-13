from __future__ import annotations

import os
import pathlib
import time
from typing import TYPE_CHECKING

import pytest

from pivot import run_history
from pivot.storage import state

if TYPE_CHECKING:
    from pivot.types import DeferredWrites


def test_state_cache_hit(tmp_path: pathlib.Path) -> None:
    """Unchanged mtime/size/inode returns cached hash."""
    db_path = tmp_path / "state.db"
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    with state.StateDB(db_path) as db:
        db.save(test_file, file_stat, "abc123")
        result = db.get(test_file, file_stat)

    assert result == "abc123"


def test_state_cache_miss_mtime(tmp_path: pathlib.Path) -> None:
    """Changed mtime triggers cache miss."""
    db_path = tmp_path / "state.db"
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
    db_path = tmp_path / "state.db"
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
    db_path = tmp_path / "state.db"
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
    db_path = tmp_path / "state.db"
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
    db_path = tmp_path / "state.db"
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
    db_path = tmp_path / "state.db"
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    with state.StateDB(db_path) as db:
        result = db.get(test_file, file_stat)

    assert result is None


def test_state_update_existing(tmp_path: pathlib.Path) -> None:
    """Saving same path updates the cached hash."""
    db_path = tmp_path / "state.db"
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
    db_path = tmp_path / "nested" / "deep" / "state.db"
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    with state.StateDB(db_path) as db:
        db.save(test_file, file_stat, "hash")

    # LMDB creates a directory (state.lmdb/) not a file (state.db)
    lmdb_path = db_path.parent / "state.lmdb"
    assert lmdb_path.is_dir()


def test_state_close(tmp_path: pathlib.Path) -> None:
    """StateDB can be closed and reopened."""
    db_path = tmp_path / "state.db"
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
    db_path = tmp_path / "state.db"
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
    db_path = tmp_path / "state.db"
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    with state.StateDB(db_path) as db:
        db.save(test_file.resolve(), file_stat, "hash")
        result = db.get(test_file.resolve(), file_stat)

    assert result == "hash"


def test_state_get_many(tmp_path: pathlib.Path) -> None:
    """Batch get returns correct hashes for multiple files."""
    db_path = tmp_path / "state.db"
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
    db_path = tmp_path / "state.db"

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
    db_path = tmp_path / "state.db"

    with state.StateDB(db_path) as db:
        results = db.get_many([])

    assert results == {}


def test_state_path_too_long_error(tmp_path: pathlib.Path) -> None:
    """PathTooLongError raised for paths exceeding LMDB key limit."""
    db_path = tmp_path / "state.db"
    # Create a deeply nested path that exceeds 511 bytes when encoded
    # Each segment is 50 chars, need ~10 segments to exceed limit
    nested = tmp_path
    for i in range(12):
        nested = nested / ("d" * 50 + str(i))
    nested.mkdir(parents=True)
    long_path = nested / "file.txt"
    long_path.write_text("content")
    file_stat = long_path.stat()

    with state.StateDB(db_path) as db, pytest.raises(state.PathTooLongError) as exc_info:
        db.save(long_path, file_stat, "hash123")

    assert "Path too long" in str(exc_info.value)
    assert "511" in str(exc_info.value)


def test_state_path_too_long_error_save_many(tmp_path: pathlib.Path) -> None:
    """PathTooLongError raised in save_many for paths exceeding limit."""
    db_path = tmp_path / "state.db"
    nested = tmp_path
    for i in range(12):
        nested = nested / ("e" * 50 + str(i))
    nested.mkdir(parents=True)
    long_path = nested / "file.txt"
    long_path.write_text("content")
    file_stat = long_path.stat()

    with state.StateDB(db_path) as db, pytest.raises(state.PathTooLongError):
        db.save_many([(long_path, file_stat, "hash123")])


def test_state_raises_after_close(tmp_path: pathlib.Path) -> None:
    """Operations on closed StateDB raise RuntimeError."""
    db_path = tmp_path / "state.db"
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
    """Getting generation for untracked path returns None."""
    db_path = tmp_path / "state.db"
    test_file = tmp_path / "output.txt"

    with state.StateDB(db_path) as db:
        result = db.get_generation(test_file)

    assert result is None


def test_generation_increment_creates_new(tmp_path: pathlib.Path) -> None:
    """Incrementing untracked path creates it with generation 1."""
    db_path = tmp_path / "state.db"
    test_file = tmp_path / "output.txt"

    with state.StateDB(db_path) as db:
        gen = db.increment_generation(test_file)
        assert gen == 1
        assert db.get_generation(test_file) == 1


def test_generation_increment_existing(tmp_path: pathlib.Path) -> None:
    """Incrementing tracked path increases generation."""
    db_path = tmp_path / "state.db"
    test_file = tmp_path / "output.txt"

    with state.StateDB(db_path) as db:
        db.increment_generation(test_file)
        db.increment_generation(test_file)
        gen = db.increment_generation(test_file)

    assert gen == 3


def test_generation_persistence(tmp_path: pathlib.Path) -> None:
    """Generations persist across DB instances."""
    db_path = tmp_path / "state.db"
    test_file = tmp_path / "output.txt"

    with state.StateDB(db_path) as db:
        db.increment_generation(test_file)
        db.increment_generation(test_file)

    with state.StateDB(db_path) as db:
        assert db.get_generation(test_file) == 2
        gen = db.increment_generation(test_file)
        assert gen == 3


def test_generation_get_many(tmp_path: pathlib.Path) -> None:
    """Batch query returns generations for multiple paths."""
    db_path = tmp_path / "state.db"
    files = [tmp_path / f"file_{i}.txt" for i in range(3)]

    with state.StateDB(db_path) as db:
        db.increment_generation(files[0])
        db.increment_generation(files[0])
        db.increment_generation(files[1])
        # files[2] not tracked

        results = db.get_many_generations(files)

    assert results[files[0]] == 2
    assert results[files[1]] == 1
    assert results[files[2]] is None


def test_generation_get_many_empty(tmp_path: pathlib.Path) -> None:
    """Batch query with empty list returns empty dict."""
    db_path = tmp_path / "state.db"

    with state.StateDB(db_path) as db:
        results = db.get_many_generations([])

    assert results == {}


def test_dep_generations_get_nonexistent(tmp_path: pathlib.Path) -> None:
    """Getting dep generations for unknown stage returns None."""
    db_path = tmp_path / "state.db"

    with state.StateDB(db_path) as db:
        result = db.get_dep_generations("unknown_stage")

    assert result is None


def test_dep_generations_record_and_get(tmp_path: pathlib.Path) -> None:
    """Record and retrieve dependency generations."""
    db_path = tmp_path / "state.db"
    deps = {"/path/to/dep1.csv": 5, "/path/to/dep2.csv": 3}

    with state.StateDB(db_path) as db:
        db.record_dep_generations("my_stage", deps)
        result = db.get_dep_generations("my_stage")

    assert result == deps


def test_dep_generations_update_replaces(tmp_path: pathlib.Path) -> None:
    """Recording dep generations replaces previous values."""
    db_path = tmp_path / "state.db"
    old_deps = {"/path/to/dep1.csv": 1, "/path/to/dep2.csv": 2}
    new_deps = {"/path/to/dep1.csv": 5, "/path/to/dep3.csv": 1}

    with state.StateDB(db_path) as db:
        db.record_dep_generations("my_stage", old_deps)
        db.record_dep_generations("my_stage", new_deps)
        result = db.get_dep_generations("my_stage")

    assert result == new_deps, "Old deps should be replaced, not merged"


def test_dep_generations_multiple_stages(tmp_path: pathlib.Path) -> None:
    """Different stages have independent dep generations."""
    db_path = tmp_path / "state.db"
    stage1_deps = {"/dep1.csv": 1}
    stage2_deps = {"/dep2.csv": 2, "/dep3.csv": 3}

    with state.StateDB(db_path) as db:
        db.record_dep_generations("stage1", stage1_deps)
        db.record_dep_generations("stage2", stage2_deps)

        result1 = db.get_dep_generations("stage1")
        result2 = db.get_dep_generations("stage2")

    assert result1 == stage1_deps
    assert result2 == stage2_deps


def test_dep_generations_persistence(tmp_path: pathlib.Path) -> None:
    """Dep generations persist across DB instances."""
    db_path = tmp_path / "state.db"
    deps = {"/path/to/dep.csv": 42}

    with state.StateDB(db_path) as db:
        db.record_dep_generations("my_stage", deps)

    with state.StateDB(db_path) as db:
        result = db.get_dep_generations("my_stage")

    assert result == deps


# -----------------------------------------------------------------------------
# Symlink handling tests
# -----------------------------------------------------------------------------


def test_generation_tracks_logical_path_not_symlink_target(tmp_path: pathlib.Path) -> None:
    """Generation tracking uses logical paths, not resolved symlink targets.

    This is critical because Pivot outputs become symlinks to cache after execution.
    If we resolved symlinks, the generation key would change every time the file's
    hash changes (different cache path). We need to track the DECLARED path.
    """
    db_path = tmp_path / "state.db"
    real_dir = tmp_path / "real_data"
    real_dir.mkdir()
    output_file = real_dir / "output.csv"

    symlink_dir = tmp_path / "data"
    symlink_dir.symlink_to(real_dir)
    symlinked_path = symlink_dir / "output.csv"

    with state.StateDB(db_path) as db:
        # Increment via symlink path
        gen1 = db.increment_generation(symlinked_path)
        assert gen1 == 1

        # Get via real path - should be INDEPENDENT (different logical path)
        gen_via_real = db.get_generation(output_file)
        assert gen_via_real is None, "Different logical paths should have independent generations"

        # Increment via real path - starts fresh
        gen2 = db.increment_generation(output_file)
        assert gen2 == 1, "Real path should start at generation 1"

        # Symlink path still has its own generation
        gen_via_symlink = db.get_generation(symlinked_path)
        assert gen_via_symlink == 1, "Symlink path generation unchanged"


# -----------------------------------------------------------------------------
# Remote index tracking tests
# -----------------------------------------------------------------------------


def test_remote_hash_exists_false(tmp_path: pathlib.Path) -> None:
    """Unknown hash returns False."""
    db_path = tmp_path / "state.db"

    with state.StateDB(db_path) as db:
        result = db.remote_hash_exists("origin", "abc123def456")

    assert result is False


def test_remote_hash_exists_true(tmp_path: pathlib.Path) -> None:
    """Added hash returns True."""
    db_path = tmp_path / "state.db"

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", ["abc123def456"])
        result = db.remote_hash_exists("origin", "abc123def456")

    assert result is True


def test_remote_hashes_add_multiple(tmp_path: pathlib.Path) -> None:
    """Multiple hashes can be added at once."""
    db_path = tmp_path / "state.db"
    hashes = ["hash1", "hash2", "hash3"]

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", hashes)

        for h in hashes:
            assert db.remote_hash_exists("origin", h)


def test_remote_hashes_intersection(tmp_path: pathlib.Path) -> None:
    """Intersection returns only hashes known to exist on remote."""
    db_path = tmp_path / "state.db"
    known = {"hash1", "hash2", "hash3"}
    query = {"hash1", "hash3", "hash4", "hash5"}

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", known)
        result = db.remote_hashes_intersection("origin", query)

    assert result == {"hash1", "hash3"}


def test_remote_hashes_intersection_empty_query(tmp_path: pathlib.Path) -> None:
    """Empty query returns empty set."""
    db_path = tmp_path / "state.db"

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", ["hash1", "hash2"])
        result = db.remote_hashes_intersection("origin", set())

    assert result == set()


def test_remote_hashes_intersection_no_matches(tmp_path: pathlib.Path) -> None:
    """No matches returns empty set."""
    db_path = tmp_path / "state.db"

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", ["hash1", "hash2"])
        result = db.remote_hashes_intersection("origin", {"hash3", "hash4"})

    assert result == set()


def test_remote_hashes_remove(tmp_path: pathlib.Path) -> None:
    """Removed hashes no longer exist."""
    db_path = tmp_path / "state.db"

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", ["hash1", "hash2", "hash3"])
        db.remote_hashes_remove("origin", ["hash2"])

        assert db.remote_hash_exists("origin", "hash1")
        assert not db.remote_hash_exists("origin", "hash2")
        assert db.remote_hash_exists("origin", "hash3")


def test_remote_index_clear(tmp_path: pathlib.Path) -> None:
    """Clear removes all hashes for a remote."""
    db_path = tmp_path / "state.db"

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
    db_path = tmp_path / "state.db"

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", ["hash1"])
        db.remote_hashes_add("backup", ["hash2"])

        assert db.remote_hash_exists("origin", "hash1")
        assert not db.remote_hash_exists("origin", "hash2")
        assert not db.remote_hash_exists("backup", "hash1")
        assert db.remote_hash_exists("backup", "hash2")


def test_remote_hashes_persistence(tmp_path: pathlib.Path) -> None:
    """Remote hashes persist across DB instances."""
    db_path = tmp_path / "state.db"

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", ["persistent_hash"])

    with state.StateDB(db_path) as db:
        assert db.remote_hash_exists("origin", "persistent_hash")


# -----------------------------------------------------------------------------
# Readonly mode tests
# -----------------------------------------------------------------------------


def test_readonly_allows_reads(tmp_path: pathlib.Path) -> None:
    """Readonly mode allows all read operations."""
    db_path = tmp_path / "state.db"
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    file_stat = test_file.stat()

    # Create data in write mode
    with state.StateDB(db_path) as db:
        db.save(test_file, file_stat, "hash123")
        db.increment_generation(test_file)
        db.record_dep_generations("stage", {"/dep.csv": 1})
        db.remote_hashes_add("origin", ["remote_hash"])

    # Verify reads work in readonly mode
    with state.StateDB(db_path, readonly=True) as db:
        assert db.get(test_file, file_stat) == "hash123"
        assert db.get_generation(test_file) == 1
        assert db.get_dep_generations("stage") == {"/dep.csv": 1}
        assert db.remote_hash_exists("origin", "remote_hash")


def test_readonly_blocks_save(tmp_path: pathlib.Path) -> None:
    """Readonly mode blocks save operation."""
    db_path = tmp_path / "state.db"
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
    db_path = tmp_path / "state.db"
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
    db_path = tmp_path / "state.db"
    test_file = tmp_path / "output.txt"

    with state.StateDB(db_path) as db:
        pass  # Just create

    with (
        state.StateDB(db_path, readonly=True) as db,
        pytest.raises(RuntimeError, match="readonly StateDB"),
    ):
        db.increment_generation(test_file)


def test_readonly_blocks_record_dep_generations(tmp_path: pathlib.Path) -> None:
    """Readonly mode blocks record_dep_generations operation."""
    db_path = tmp_path / "state.db"

    with state.StateDB(db_path) as db:
        pass  # Just create

    with (
        state.StateDB(db_path, readonly=True) as db,
        pytest.raises(RuntimeError, match="readonly StateDB"),
    ):
        db.record_dep_generations("stage", {"/dep.csv": 1})


def test_readonly_blocks_remote_hashes_add(tmp_path: pathlib.Path) -> None:
    """Readonly mode blocks remote_hashes_add operation."""
    db_path = tmp_path / "state.db"

    with state.StateDB(db_path) as db:
        pass  # Just create

    with (
        state.StateDB(db_path, readonly=True) as db,
        pytest.raises(RuntimeError, match="readonly StateDB"),
    ):
        db.remote_hashes_add("origin", ["hash1"])


def test_readonly_blocks_remote_hashes_remove(tmp_path: pathlib.Path) -> None:
    """Readonly mode blocks remote_hashes_remove operation."""
    db_path = tmp_path / "state.db"

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", ["hash1"])

    with (
        state.StateDB(db_path, readonly=True) as db,
        pytest.raises(RuntimeError, match="readonly StateDB"),
    ):
        db.remote_hashes_remove("origin", ["hash1"])


def test_readonly_blocks_remote_index_clear(tmp_path: pathlib.Path) -> None:
    """Readonly mode blocks remote_index_clear operation."""
    db_path = tmp_path / "state.db"

    with state.StateDB(db_path) as db:
        db.remote_hashes_add("origin", ["hash1"])

    with (
        state.StateDB(db_path, readonly=True) as db,
        pytest.raises(RuntimeError, match="readonly StateDB"),
    ):
        db.remote_index_clear("origin")


# -----------------------------------------------------------------------------
# apply_deferred_writes tests
# -----------------------------------------------------------------------------


def test_apply_deferred_writes_dep_generations(tmp_path: pathlib.Path) -> None:
    """apply_deferred_writes records dependency generations."""
    db_path = tmp_path / "state.db"
    deferred: DeferredWrites = {"dep_generations": {"/path/dep1.csv": 5, "/path/dep2.csv": 3}}

    with state.StateDB(db_path) as db:
        db.apply_deferred_writes("my_stage", [], deferred)
        result = db.get_dep_generations("my_stage")

    assert result == {"/path/dep1.csv": 5, "/path/dep2.csv": 3}


def test_apply_deferred_writes_output_generations(tmp_path: pathlib.Path) -> None:
    """apply_deferred_writes increments output generations."""
    db_path = tmp_path / "state.db"
    output1 = tmp_path / "output1.csv"
    output2 = tmp_path / "output2.csv"
    deferred: DeferredWrites = {}

    with state.StateDB(db_path) as db:
        # First apply - outputs should be at generation 1
        db.apply_deferred_writes("stage", [str(output1), str(output2)], deferred)
        assert db.get_generation(output1) == 1
        assert db.get_generation(output2) == 1

        # Second apply - outputs should increment to 2
        db.apply_deferred_writes("stage", [str(output1), str(output2)], deferred)
        assert db.get_generation(output1) == 2
        assert db.get_generation(output2) == 2


def test_apply_deferred_writes_run_cache(tmp_path: pathlib.Path) -> None:
    """apply_deferred_writes writes run cache entry."""
    db_path = tmp_path / "state.db"
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
    db_path = tmp_path / "state.db"
    output_path = tmp_path / "output.csv"
    run_cache_entry = run_history.RunCacheEntry(
        run_id="run_456",
        output_hashes=[run_history.OutputHashEntry(path=str(output_path), hash="def456")],
    )
    deferred: DeferredWrites = {
        "dep_generations": {"/dep.csv": 10},
        "run_cache_input_hash": "input_abc",
        "run_cache_entry": run_cache_entry,
    }

    with state.StateDB(db_path) as db:
        db.apply_deferred_writes("stage", [str(output_path)], deferred)

        # Verify all writes applied
        assert db.get_dep_generations("stage") == {"/dep.csv": 10}
        assert db.get_generation(output_path) == 1
        result = db.lookup_run_cache("stage", "input_abc")
        assert result is not None
        assert result["run_id"] == "run_456"


def test_apply_deferred_writes_empty(tmp_path: pathlib.Path) -> None:
    """apply_deferred_writes handles empty deferred dict."""
    db_path = tmp_path / "state.db"
    deferred: DeferredWrites = {}

    with state.StateDB(db_path) as db:
        # Should not raise
        db.apply_deferred_writes("stage", [], deferred)


def test_apply_deferred_writes_readonly_blocked(tmp_path: pathlib.Path) -> None:
    """apply_deferred_writes blocked in readonly mode."""
    db_path = tmp_path / "state.db"

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
    db_path = tmp_path / "state.db"
    long_path = "/" + "d" * 600  # Exceeds 511 byte limit
    deferred: DeferredWrites = {"dep_generations": {long_path: 1}}

    with state.StateDB(db_path) as db, pytest.raises(state.PathTooLongError):
        db.apply_deferred_writes("stage", [], deferred)


def test_apply_deferred_writes_path_too_long_output(tmp_path: pathlib.Path) -> None:
    """apply_deferred_writes raises PathTooLongError for long output paths."""
    db_path = tmp_path / "state.db"
    # Create a deeply nested path that exceeds 511 bytes
    nested = tmp_path
    for i in range(12):
        nested = nested / ("o" * 50 + str(i))
    long_output = str(nested / "output.csv")
    deferred: DeferredWrites = {}

    with state.StateDB(db_path) as db, pytest.raises(state.PathTooLongError):
        db.apply_deferred_writes("stage", [long_output], deferred)
