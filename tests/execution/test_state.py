import os
import pathlib
import time

import pytest

from pivot import state


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

        test_file.unlink()
        test_file.write_text("content")
        new_stat = test_file.stat()

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
