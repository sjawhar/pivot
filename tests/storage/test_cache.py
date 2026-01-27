from __future__ import annotations

import mmap
import multiprocessing
import os
import pathlib
import stat
import time
from typing import TYPE_CHECKING, cast

import pytest

from pivot.storage import cache, state

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _helper_assert_no_temp_dirs(
    parent: pathlib.Path,
    *,
    allow_lock_files: bool = True,
) -> None:
    """Assert no restore/backup temp directories exist (lock files allowed by default)."""
    unexpected = list[pathlib.Path]()
    for p in parent.iterdir():
        # Check lock files first (lock prefix is a superset of temp prefix)
        if p.name.startswith(cache._RESTORE_LOCK_PREFIX):
            if not allow_lock_files:
                unexpected.append(p)
            continue
        # Check for temp/backup artifacts
        if p.name.startswith(cache._RESTORE_TEMP_PREFIX) or p.name.startswith(
            cache._BACKUP_TEMP_PREFIX
        ):
            unexpected.append(p)
    assert len(unexpected) == 0, f"Found unexpected temp dirs: {unexpected}"


if TYPE_CHECKING:
    from collections.abc import Generator
    from multiprocessing.managers import SyncManager

    from pivot.types import DirHash, FileHash


@pytest.fixture
def mp_manager() -> Generator[SyncManager]:
    """Provide a multiprocessing Manager with automatic cleanup."""
    manager = multiprocessing.Manager()
    yield manager
    manager.shutdown()


# === Hash File Tests ===


def test_hash_file(tmp_path: pathlib.Path) -> None:
    """hash_file returns consistent xxhash64 hash."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("hello world")

    hash1 = cache.hash_file(test_file)
    hash2 = cache.hash_file(test_file)

    assert hash1 == hash2
    assert len(hash1) == 16  # xxhash64 hex is 16 chars


def test_hash_file_different_content(tmp_path: pathlib.Path) -> None:
    """Different content produces different hash."""
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file1.write_text("hello")
    file2.write_text("world")

    assert cache.hash_file(file1) != cache.hash_file(file2)


def test_hash_file_uses_state_cache(tmp_path: pathlib.Path) -> None:
    """hash_file uses state cache to skip rehashing."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    db_path = tmp_path / "state.db"

    with state.StateDB(db_path) as db:
        cache.hash_file(test_file, state_db=db)  # First hash populates cache
        db.save(test_file, test_file.stat(), "cached_hash")
        hash2 = cache.hash_file(test_file, state_db=db)

    assert hash2 == "cached_hash"


def test_hash_file_binary(tmp_path: pathlib.Path) -> None:
    """hash_file works with binary content."""
    test_file = tmp_path / "binary.bin"
    test_file.write_bytes(b"\x00\x01\x02\xff\xfe")

    file_hash = cache.hash_file(test_file)

    assert len(file_hash) == 16


def test_hash_file_large_uses_mmap(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Large files (>=MMAP_THRESHOLD) use mmap and produce valid hashes."""
    # Monkeypatch threshold to 100 bytes to avoid writing large files
    monkeypatch.setattr(cache, "MMAP_THRESHOLD", 100)

    small_file = tmp_path / "small.bin"
    small_file.write_bytes(b"x" * 50)  # Below threshold

    large_file = tmp_path / "large.bin"
    large_file.write_bytes(b"x" * 150)  # Above threshold

    # Both should produce valid hashes
    small_hash = cache.hash_file(small_file)
    large_hash = cache.hash_file(large_file)

    assert len(small_hash) == 16
    assert len(large_hash) == 16
    # Hashes should be different (different content sizes)
    assert small_hash != large_hash


def test_hash_file_mmap_consistent(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mmap and buffered read produce identical hashes for same content."""
    # Use small content but monkeypatch threshold to test both paths
    content = b"test content for mmap consistency check"
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(content)

    # Set threshold below content size to force mmap path
    monkeypatch.setattr(cache, "MMAP_THRESHOLD", 10)
    mmap_hash = cache.hash_file(test_file)

    # Set threshold above content size to force buffered path
    monkeypatch.setattr(cache, "MMAP_THRESHOLD", len(content) + 100)
    buffered_hash = cache.hash_file(test_file)

    # Both methods must produce identical hash
    assert mmap_hash == buffered_hash


def test_hash_file_mmap_fallback(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Falls back to buffered read when mmap fails."""
    # Use small content with low threshold to trigger mmap path
    content = b"test content for mmap fallback testing"
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(content)

    # Set threshold below content size to trigger mmap path
    monkeypatch.setattr(cache, "MMAP_THRESHOLD", 10)

    # Get expected hash via normal path first (mmap succeeds)
    expected_hash = cache.hash_file(test_file)

    # Now make mmap fail
    def failing_mmap(*args: object, **kwargs: object) -> mmap.mmap:
        raise OSError("mmap failed")

    monkeypatch.setattr(mmap, "mmap", failing_mmap)

    # Should fall back to buffered read and produce same hash
    fallback_hash = cache.hash_file(test_file)
    assert fallback_hash == expected_hash


# === Hash Directory Tests ===


def test_hash_directory(tmp_path: pathlib.Path) -> None:
    """hash_directory returns hash and manifest."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "a.txt").write_text("content a")
    (test_dir / "b.txt").write_text("content b")

    tree_hash, manifest = cache.hash_directory(test_dir)

    assert len(tree_hash) == 16
    assert len(manifest) == 2


def test_hash_directory_relative_paths(tmp_path: pathlib.Path) -> None:
    """Manifest contains relative paths only."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "file.txt").write_text("content")
    subdir = test_dir / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("nested")

    _, manifest = cache.hash_directory(test_dir)

    relpaths = [e["relpath"] for e in manifest]
    assert "file.txt" in relpaths
    assert "subdir/nested.txt" in relpaths
    assert not any(str(tmp_path) in p for p in relpaths)


def test_hash_directory_sorted_manifest(tmp_path: pathlib.Path) -> None:
    """Manifest is sorted by relpath."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "z.txt").write_text("z")
    (test_dir / "a.txt").write_text("a")
    (test_dir / "m.txt").write_text("m")

    _, manifest = cache.hash_directory(test_dir)

    relpaths = [e["relpath"] for e in manifest]
    assert relpaths == sorted(relpaths)


def test_hash_directory_deterministic(tmp_path: pathlib.Path) -> None:
    """Same directory content produces same hash."""
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    for d in [dir1, dir2]:
        d.mkdir()
        (d / "file.txt").write_text("same content")

    hash1, _ = cache.hash_directory(dir1)
    hash2, _ = cache.hash_directory(dir2)

    assert hash1 == hash2


def test_hash_directory_includes_size(tmp_path: pathlib.Path) -> None:
    """Manifest entries include file size."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "file.txt").write_text("hello")

    _, manifest = cache.hash_directory(test_dir)

    assert manifest[0]["size"] == 5


def test_hash_directory_skips_symlinks(tmp_path: pathlib.Path) -> None:
    """Symlinks in directory are skipped."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "real.txt").write_text("content")
    (test_dir / "link.txt").symlink_to(test_dir / "real.txt")

    _, manifest = cache.hash_directory(test_dir)

    relpaths = [e["relpath"] for e in manifest]
    assert "real.txt" in relpaths
    assert "link.txt" not in relpaths, "Symlinks should be skipped"


def test_hash_directory_marks_executable(tmp_path: pathlib.Path) -> None:
    """Executable files are marked in manifest."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    regular = test_dir / "regular.txt"
    regular.write_text("content")
    executable = test_dir / "script.sh"
    executable.write_text("#!/bin/bash")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

    _, manifest = cache.hash_directory(test_dir)

    manifest_dict = {e["relpath"]: e for e in manifest}
    assert manifest_dict["regular.txt"]["isexec"] is False
    assert manifest_dict["script.sh"]["isexec"] is True


def test_hash_directory_raises_on_unreadable_subdirs(tmp_path: pathlib.Path) -> None:
    """Unreadable subdirectories raise PermissionError (fail-fast, no partial hashes)."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "readable.txt").write_text("content")
    unreadable = test_dir / "unreadable"
    unreadable.mkdir()
    (unreadable / "hidden.txt").write_text("hidden")
    unreadable.chmod(0o000)

    try:
        with pytest.raises(PermissionError):
            cache.hash_directory(test_dir)
    finally:
        unreadable.chmod(0o755)


def test_hash_directory_handles_deleted_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Files deleted between scan and hash are gracefully skipped."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "keep.txt").write_text("keep")
    to_delete = test_dir / "delete.txt"
    to_delete.write_text("delete")

    original_hash_file = cache.hash_file
    call_count = 0

    def hash_file_with_delete(path: pathlib.Path, state_db: state.StateDB | None = None) -> str:
        nonlocal call_count
        call_count += 1
        if path.name == "delete.txt":
            to_delete.unlink()
            raise FileNotFoundError()
        return original_hash_file(path, state_db)

    monkeypatch.setattr(cache, "hash_file", hash_file_with_delete)

    _, manifest = cache.hash_directory(test_dir)
    assert len(manifest) == 1
    assert manifest[0]["relpath"] == "keep.txt"


# === Save to Cache Tests ===


def test_save_to_cache_creates_cache_file(tmp_path: pathlib.Path) -> None:
    """save_to_cache creates file in cache directory."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(test_file, cache_dir)
    assert output_hash is not None

    cache_path = cache_dir / output_hash["hash"][:2] / output_hash["hash"][2:]
    assert cache_path.exists()


def test_save_to_cache_read_only(tmp_path: pathlib.Path) -> None:
    """Cached files are read-only (mode 0o444)."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(test_file, cache_dir)
    assert output_hash is not None

    cache_path = cache_dir / output_hash["hash"][:2] / output_hash["hash"][2:]
    mode = cache_path.stat().st_mode & 0o777
    assert mode == 0o444


def test_save_to_cache_creates_symlink(tmp_path: pathlib.Path) -> None:
    """Original file is replaced with symlink to cache when SYMLINK mode used."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    cache.save_to_cache(test_file, cache_dir, checkout_mode=cache.CheckoutMode.SYMLINK)

    assert test_file.is_symlink()


def test_save_to_cache_directory(tmp_path: pathlib.Path) -> None:
    """save_to_cache handles directories."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "file.txt").write_text("content")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(test_dir, cache_dir)
    assert output_hash is not None

    assert "manifest" in output_hash
    assert test_dir.is_symlink() or test_dir.is_dir()


def test_save_to_cache_deduplicates(tmp_path: pathlib.Path) -> None:
    """Identical files share cache entry."""
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file1.write_text("same content")
    file2.write_text("same content")
    cache_dir = tmp_path / "cache"

    hash1 = cache.save_to_cache(file1, cache_dir)
    hash2 = cache.save_to_cache(file2, cache_dir)
    assert hash1 is not None and hash2 is not None

    assert hash1["hash"] == hash2["hash"]


def test_save_atomic_no_partial(tmp_path: pathlib.Path) -> None:
    """No partial files on failure."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    cache.save_to_cache(test_file, cache_dir)

    tmp_files = list(cache_dir.rglob("*.tmp"))
    assert len(tmp_files) == 0


# === Restore from Cache Tests ===


def test_restore_from_cache_creates_link(tmp_path: pathlib.Path) -> None:
    """restore_from_cache creates symlink to cached file when SYMLINK mode used."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(
        test_file, cache_dir, checkout_mode=cache.CheckoutMode.SYMLINK
    )
    test_file.unlink()

    restored = cache.restore_from_cache(
        test_file, output_hash, cache_dir, checkout_mode=cache.CheckoutMode.SYMLINK
    )

    assert restored is True
    assert test_file.is_symlink()
    assert test_file.read_text() == "content"


def test_restore_from_cache_missing(tmp_path: pathlib.Path) -> None:
    """restore_from_cache returns False if cache entry missing."""
    test_file = tmp_path / "file.txt"
    cache_dir = tmp_path / "cache"
    missing_hash: FileHash = {"hash": "0" * 16}

    restored = cache.restore_from_cache(test_file, missing_hash, cache_dir)

    assert restored is False


def test_restore_directory_from_cache(tmp_path: pathlib.Path) -> None:
    """restore_from_cache restores directories."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "file.txt").write_text("content")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(test_dir, cache_dir)
    cache.remove_output(test_dir)  # Use remove_output to handle symlinks

    restored = cache.restore_from_cache(test_dir, output_hash, cache_dir)

    assert restored is True
    assert (test_dir / "file.txt").exists()


def test_restore_directory_hardlink_mode(tmp_path: pathlib.Path) -> None:
    """restore_from_cache restores directories with hardlink mode."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    subdir = test_dir / "subdir"
    subdir.mkdir()
    (test_dir / "file.txt").write_text("content")
    (subdir / "nested.txt").write_text("nested")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(
        test_dir, cache_dir, checkout_mode=cache.CheckoutMode.HARDLINK
    )
    cache.remove_output(test_dir)

    restored = cache.restore_from_cache(
        test_dir, output_hash, cache_dir, checkout_mode=cache.CheckoutMode.HARDLINK
    )

    assert restored is True
    assert (test_dir / "file.txt").read_text() == "content"
    assert (subdir / "nested.txt").read_text() == "nested"


# === Atomic Directory Restore Tests ===


def test_restore_directory_atomic_no_temp_dirs_on_success(tmp_path: pathlib.Path) -> None:
    """Atomic restore leaves no temp directories on success."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "file.txt").write_text("content")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(
        test_dir, cache_dir, checkout_mode=cache.CheckoutMode.HARDLINK
    )
    cache.remove_output(test_dir)

    restored = cache.restore_from_cache(
        test_dir, output_hash, cache_dir, checkout_mode=cache.CheckoutMode.HARDLINK
    )

    assert restored is True
    assert (test_dir / "file.txt").read_text() == "content"
    _helper_assert_no_temp_dirs(tmp_path)


def test_restore_directory_atomic_cleans_up_on_cache_miss(
    tmp_path: pathlib.Path,
) -> None:
    """Atomic restore returns False without creating temp dirs when cache missing."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    target = tmp_path / "mydir"

    # Create hash with non-existent cache entry
    missing_hash: DirHash = {
        "hash": "1234567890abcdef",
        "manifest": [{"relpath": "file.txt", "hash": "0" * 16, "size": 7, "isexec": False}],
    }

    result = cache.restore_from_cache(
        target, missing_hash, cache_dir, checkout_mode=cache.CheckoutMode.HARDLINK
    )

    assert result is False
    assert not target.exists()
    _helper_assert_no_temp_dirs(tmp_path, allow_lock_files=False)


def test_restore_directory_atomic_cleans_up_on_exception(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atomic restore cleans up temp directory on exception."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "file.txt").write_text("content")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(
        test_dir, cache_dir, checkout_mode=cache.CheckoutMode.HARDLINK
    )
    cache.remove_output(test_dir)

    # Make checkout fail after temp dir is created
    def failing_checkout(*args: object, **kwargs: object) -> None:
        raise OSError("simulated failure")

    monkeypatch.setattr(cache, "_checkout_with_fallback", failing_checkout)

    with pytest.raises(OSError, match="simulated failure"):
        cache.restore_from_cache(
            test_dir, output_hash, cache_dir, checkout_mode=cache.CheckoutMode.HARDLINK
        )

    assert not test_dir.exists()
    _helper_assert_no_temp_dirs(tmp_path)


def test_restore_directory_preserves_original_on_rename_failure(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two-phase rename preserves original if swap fails."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "original.txt").write_text("original content")
    cache_dir = tmp_path / "cache"

    # Save a version to cache
    output_hash = cache.save_to_cache(test_dir, cache_dir, checkout_mode=cache.CheckoutMode.COPY)

    # Modify the original
    (test_dir / "original.txt").write_text("modified content")

    # Make the final replace fail
    original_replace = pathlib.Path.replace

    def failing_replace(self: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
        if self.name.startswith(cache._RESTORE_TEMP_PREFIX):
            raise OSError("simulated replace failure")
        return original_replace(self, target)

    monkeypatch.setattr(pathlib.Path, "replace", failing_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        cache.restore_from_cache(
            test_dir, output_hash, cache_dir, checkout_mode=cache.CheckoutMode.COPY
        )

    # Original should be restored (or still present)
    assert test_dir.exists()
    # The test_dir should have original.txt (though content may vary based on when failure occurred)


def test_restore_directory_validates_all_entries_before_writing(
    tmp_path: pathlib.Path,
) -> None:
    """All manifest entries validated before any files written."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    target = tmp_path / "mydir"

    # Create a real cached file for the first entry
    first_file = tmp_path / "first.txt"
    first_file.write_text("first")
    first_hash = cache.hash_file(first_file)
    cache_path = cache.get_cache_path(cache_dir, first_hash)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    first_file.rename(cache_path)

    # Create manifest with first entry valid, second entry missing
    partial_hash: DirHash = {
        "hash": "abcd1234567890ef",
        "manifest": [
            {"relpath": "first.txt", "hash": first_hash, "size": 5, "isexec": False},
            {"relpath": "second.txt", "hash": "deadbeef12345678", "size": 6, "isexec": False},
        ],
    }

    result = cache.restore_from_cache(
        target, partial_hash, cache_dir, checkout_mode=cache.CheckoutMode.COPY
    )

    assert result is False
    # No files should have been written (validation happens before any writes)
    assert not target.exists()


def test_cleanup_removes_old_temps(tmp_path: pathlib.Path) -> None:
    """Temps older than max age are cleaned."""
    # Create an old temp directory
    old_temp = tmp_path / f"{cache._RESTORE_TEMP_PREFIX}old_test"
    old_temp.mkdir()
    (old_temp / "file.txt").write_text("old")

    # Set mtime to 2 hours ago
    old_mtime = time.time() - 7200
    os.utime(old_temp, (old_mtime, old_mtime))

    cache._cleanup_stale_restore_temps(tmp_path)

    assert not old_temp.exists()


def test_cleanup_preserves_recent_temps(tmp_path: pathlib.Path) -> None:
    """Recent temps are preserved."""
    # Create a recent temp directory
    recent_temp = tmp_path / f"{cache._RESTORE_TEMP_PREFIX}recent_test"
    recent_temp.mkdir()
    (recent_temp / "file.txt").write_text("recent")
    # mtime is now, which is recent

    cache._cleanup_stale_restore_temps(tmp_path)

    assert recent_temp.exists()
    # Cleanup
    cache._clear_path(recent_temp)


def test_cleanup_skips_symlinks_via_is_dir(tmp_path: pathlib.Path) -> None:
    """is_dir(follow_symlinks=False) skips symlinks safely."""
    # Create a real directory to link to
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    (real_dir / "important.txt").write_text("do not delete")

    # Create a symlink with our temp prefix pointing to it
    symlink = tmp_path / f"{cache._RESTORE_TEMP_PREFIX}symlink_attack"
    symlink.symlink_to(real_dir)

    cache._cleanup_stale_restore_temps(tmp_path)

    # The real directory should still exist
    assert real_dir.exists()
    assert (real_dir / "important.txt").exists()
    # Cleanup
    symlink.unlink()


def test_get_lock_filename_short_name_passthrough() -> None:
    """Short names are passed through unchanged."""
    result = cache._get_lock_filename("mydir")
    assert result == f"{cache._RESTORE_LOCK_PREFIX}mydir"


def test_get_lock_filename_long_name_hashed() -> None:
    """Long names are hashed to fit filesystem limit."""
    long_name = "a" * 300  # Exceeds 255 - prefix length
    result = cache._get_lock_filename(long_name)

    assert result.startswith(cache._RESTORE_LOCK_PREFIX)
    assert len(result) <= 255, "Lock filename must fit filesystem limit"
    # Hash portion should be 32 hex chars
    hash_portion = result[len(cache._RESTORE_LOCK_PREFIX) :]
    assert len(hash_portion) == 32
    assert all(c in "0123456789abcdef" for c in hash_portion)


def test_get_lock_filename_boundary() -> None:
    """Names exactly at the boundary are not hashed."""
    max_name_len = 255 - len(cache._RESTORE_LOCK_PREFIX)
    boundary_name = "x" * max_name_len

    result = cache._get_lock_filename(boundary_name)

    assert result == f"{cache._RESTORE_LOCK_PREFIX}{boundary_name}"
    assert len(result) == 255


def test_get_lock_filename_one_over_boundary() -> None:
    """Names one char over the boundary are hashed."""
    max_name_len = 255 - len(cache._RESTORE_LOCK_PREFIX)
    over_boundary_name = "x" * (max_name_len + 1)

    result = cache._get_lock_filename(over_boundary_name)

    assert len(result) < 255, "Hashed result should be shorter than limit"
    hash_portion = result[len(cache._RESTORE_LOCK_PREFIX) :]
    assert len(hash_portion) == 32


def _restore_worker(
    test_dir: pathlib.Path,
    output_hash: DirHash,
    cache_dir: pathlib.Path,
    result_queue: multiprocessing.Queue[tuple[str, str | bool]],
) -> None:
    """Worker function for concurrent restore test (must be module-level for pickling)."""
    try:
        result = cache.restore_from_cache(
            test_dir, output_hash, cache_dir, checkout_mode=cache.CheckoutMode.HARDLINK
        )
        result_queue.put(("success", result))
    except Exception as e:
        result_queue.put(("error", str(e)))


def test_concurrent_restore_uses_locking(
    tmp_path: pathlib.Path,
    mp_manager: SyncManager,
) -> None:
    """Concurrent restores to same path are serialized by lock."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "file.txt").write_text("content")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(
        test_dir, cache_dir, checkout_mode=cache.CheckoutMode.HARDLINK
    )
    cache.remove_output(test_dir)

    # Use Manager().Queue() for cross-process communication (cast via object for type safety)
    result_queue = cast(
        "multiprocessing.Queue[tuple[str, str | bool]]",
        cast("object", mp_manager.Queue()),
    )
    p1 = multiprocessing.Process(
        target=_restore_worker, args=(test_dir, output_hash, cache_dir, result_queue)
    )
    p2 = multiprocessing.Process(
        target=_restore_worker, args=(test_dir, output_hash, cache_dir, result_queue)
    )
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    # Both should succeed (second waits for first due to lock)
    results = [result_queue.get(timeout=60), result_queue.get(timeout=60)]
    assert all(r[0] == "success" for r in results), f"Got errors: {results}"
    assert (test_dir / "file.txt").read_text() == "content"


# === Remove Output Tests ===


def test_remove_output_file(tmp_path: pathlib.Path) -> None:
    """remove_output deletes regular file."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")

    cache.remove_output(test_file)

    assert not test_file.exists()


def test_remove_output_directory(tmp_path: pathlib.Path) -> None:
    """remove_output deletes directory recursively."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "file.txt").write_text("content")

    cache.remove_output(test_dir)

    assert not test_dir.exists()


def test_remove_output_symlink(tmp_path: pathlib.Path) -> None:
    """remove_output removes symlink without following."""
    target = tmp_path / "target.txt"
    target.write_text("content")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    cache.remove_output(link)

    assert not link.exists()
    assert target.exists()


def test_remove_output_missing_ok(tmp_path: pathlib.Path) -> None:
    """remove_output does nothing if path doesn't exist."""
    missing = tmp_path / "missing.txt"

    cache.remove_output(missing)


def test_remove_output_readonly_file(tmp_path: pathlib.Path) -> None:
    """remove_output deletes read-only file (as in hardlinked cache).

    When outputs are hardlinked to the cache, they become read-only (0o444).
    remove_output must be able to delete these files so stages can re-run.
    """
    test_file = tmp_path / "readonly.txt"
    test_file.write_text("content")
    os.chmod(test_file, 0o444)

    cache.remove_output(test_file)

    assert not test_file.exists()


def test_remove_output_readonly_directory(tmp_path: pathlib.Path) -> None:
    """remove_output deletes directory containing read-only files.

    Directory outputs with HARDLINK mode have individual files hardlinked
    to read-only cache. remove_output must handle this recursively.
    """
    test_dir = tmp_path / "readonly_dir"
    test_dir.mkdir()
    (test_dir / "file1.txt").write_text("content1")
    (test_dir / "file2.txt").write_text("content2")

    # Make all files read-only (simulating hardlinks to cache)
    os.chmod(test_dir / "file1.txt", 0o444)
    os.chmod(test_dir / "file2.txt", 0o444)

    cache.remove_output(test_dir)

    assert not test_dir.exists()


def test_remove_output_readonly_nested_directory(tmp_path: pathlib.Path) -> None:
    """remove_output handles nested directories with read-only files.

    Tests the scenario from the user bug report where outputs were in
    nested directories like plots/bar_chart_weighted_scores/headline.png.
    """
    test_dir = tmp_path / "plots"
    nested_dir = test_dir / "bar_chart_weighted_scores"
    nested_dir.mkdir(parents=True)
    (nested_dir / "headline.png").write_bytes(b"PNG")
    (nested_dir / "other.png").write_bytes(b"PNG")

    # Make all files read-only
    os.chmod(nested_dir / "headline.png", 0o444)
    os.chmod(nested_dir / "other.png", 0o444)

    cache.remove_output(test_dir)

    assert not test_dir.exists()


def test_remove_output_hardlinked_file_preserves_cache_permissions(tmp_path: pathlib.Path) -> None:
    """Removing hardlinked output file does not corrupt cache permissions.

    When outputs are hardlinked to the cache, they share the same inode.
    chmod on the output would also change the cache file's permissions.
    This test verifies that remove_output does not chmod the file.
    """
    # Create cache file (read-only)
    cache_dir = tmp_path / "cache"
    cache_file = cache_dir / "ab" / "content_hash"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("cached content")
    os.chmod(cache_file, 0o444)

    # Create output as hardlink to cache
    output_file = tmp_path / "output.txt"
    os.link(cache_file, output_file)

    # Verify setup
    assert cache_file.stat().st_ino == output_file.stat().st_ino, "Should be hardlinked"
    assert cache_file.stat().st_mode & 0o777 == 0o444

    # Remove output
    cache.remove_output(output_file)

    # Cache should still exist and be read-only
    assert not output_file.exists()
    assert cache_file.exists()
    assert cache_file.stat().st_mode & 0o777 == 0o444, "Cache should remain read-only"


def test_remove_output_hardlinked_directory_preserves_cache_permissions(
    tmp_path: pathlib.Path,
) -> None:
    """Removing directory with hardlinked files does not corrupt cache permissions.

    When directory outputs are cached with HARDLINK mode, individual files are
    hardlinked to the cache. This test verifies that removing the directory
    does not change permissions on the cached files.
    """
    # Create cache files (read-only)
    cache_dir = tmp_path / "cache"
    cache_file1 = cache_dir / "ab" / "hash1"
    cache_file2 = cache_dir / "cd" / "hash2"
    cache_file1.parent.mkdir(parents=True)
    cache_file2.parent.mkdir(parents=True)
    cache_file1.write_text("content1")
    cache_file2.write_text("content2")
    os.chmod(cache_file1, 0o444)
    os.chmod(cache_file2, 0o444)

    # Create output directory with hardlinked files
    output_dir = tmp_path / "output_dir"
    output_dir.mkdir()
    os.link(cache_file1, output_dir / "file1.txt")
    os.link(cache_file2, output_dir / "file2.txt")

    # Make directory read-only to force onexc handler
    os.chmod(output_dir, 0o555)

    # Verify setup
    assert cache_file1.stat().st_mode & 0o777 == 0o444
    assert cache_file2.stat().st_mode & 0o777 == 0o444

    # Remove output directory
    cache.remove_output(output_dir)

    # Cache files should still exist and be read-only
    assert not output_dir.exists()
    assert cache_file1.exists()
    assert cache_file2.exists()
    assert cache_file1.stat().st_mode & 0o777 == 0o444, "Cache file 1 should remain read-only"
    assert cache_file2.stat().st_mode & 0o777 == 0o444, "Cache file 2 should remain read-only"


# === Protection Tests ===


def test_protect(tmp_path: pathlib.Path) -> None:
    """protect makes file read-only."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")

    cache.protect(test_file)

    mode = test_file.stat().st_mode & 0o777
    assert mode == 0o444


def test_unprotect(tmp_path: pathlib.Path) -> None:
    """unprotect restores write permission."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache.protect(test_file)

    cache.unprotect(test_file)

    mode = test_file.stat().st_mode & 0o777
    assert mode & stat.S_IWUSR


# === Checkout Mode Tests ===


def test_checkout_mode_symlink(tmp_path: pathlib.Path) -> None:
    """SYMLINK mode creates symlinks."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    cache.save_to_cache(test_file, cache_dir, checkout_mode=cache.CheckoutMode.SYMLINK)

    assert test_file.is_symlink()


def test_checkout_mode_hardlink(tmp_path: pathlib.Path) -> None:
    """HARDLINK mode creates hardlinks."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(
        test_file, cache_dir, checkout_mode=cache.CheckoutMode.HARDLINK
    )
    assert output_hash is not None

    cache_path = cache_dir / output_hash["hash"][:2] / output_hash["hash"][2:]
    assert test_file.stat().st_ino == cache_path.stat().st_ino


def test_checkout_mode_copy(tmp_path: pathlib.Path) -> None:
    """COPY mode creates separate copies."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(test_file, cache_dir, checkout_mode=cache.CheckoutMode.COPY)
    assert output_hash is not None

    cache_path = cache_dir / output_hash["hash"][:2] / output_hash["hash"][2:]
    assert not test_file.is_symlink()
    assert test_file.stat().st_ino != cache_path.stat().st_ino
    assert test_file.read_text() == "content"


# === Idempotency Tests (BUG-006) ===


def test_save_to_cache_idempotent_file(tmp_path: pathlib.Path) -> None:
    """Second save_to_cache on symlinked file is idempotent."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    h1 = cache.save_to_cache(test_file, cache_dir, checkout_mode=cache.CheckoutMode.SYMLINK)
    assert test_file.is_symlink()
    h2 = cache.save_to_cache(
        test_file, cache_dir, checkout_mode=cache.CheckoutMode.SYMLINK
    )  # Should NOT raise ELOOP

    assert h1 == h2
    assert test_file.read_text() == "content"


def test_save_to_cache_idempotent_directory(tmp_path: pathlib.Path) -> None:
    """Second save_to_cache on symlinked directory is idempotent."""
    test_dir = tmp_path / "dir"
    test_dir.mkdir()
    (test_dir / "a.txt").write_text("a")
    cache_dir = tmp_path / "cache"

    h1 = cache.save_to_cache(test_dir, cache_dir, checkout_mode=cache.CheckoutMode.SYMLINK)
    assert h1 is not None
    assert test_dir.is_symlink()
    h2 = cache.save_to_cache(
        test_dir, cache_dir, checkout_mode=cache.CheckoutMode.SYMLINK
    )  # Should NOT raise ELOOP
    assert h2 is not None

    assert h1["hash"] == h2["hash"]
    assert (test_dir / "a.txt").read_text() == "a"


def test_checkout_from_cache_idempotent_symlink(tmp_path: pathlib.Path) -> None:
    """_checkout_from_cache skips if already correctly symlinked."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(test_file, cache_dir)
    assert output_hash is not None
    cache_path = cache.get_cache_path(cache_dir, output_hash["hash"])

    # Call _checkout_from_cache again - should be idempotent
    cache._checkout_from_cache(test_file, cache_path, cache.CheckoutMode.SYMLINK)

    assert test_file.is_symlink()
    assert test_file.read_text() == "content"


def test_checkout_from_cache_idempotent_hardlink(tmp_path: pathlib.Path) -> None:
    """_checkout_from_cache skips if already correctly hardlinked."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(
        test_file, cache_dir, checkout_mode=cache.CheckoutMode.HARDLINK
    )
    assert output_hash is not None
    cache_path = cache.get_cache_path(cache_dir, output_hash["hash"])
    original_inode = test_file.stat().st_ino

    # Call _checkout_from_cache again - should be idempotent
    cache._checkout_from_cache(test_file, cache_path, cache.CheckoutMode.HARDLINK)

    assert test_file.stat().st_ino == original_inode


def test_save_to_cache_broken_symlink(tmp_path: pathlib.Path) -> None:
    """Broken symlink triggers re-cache (not idempotent skip)."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    h1 = cache.save_to_cache(test_file, cache_dir, checkout_mode=cache.CheckoutMode.SYMLINK)
    assert h1 is not None
    cache_path = cache.get_cache_path(cache_dir, h1["hash"])

    # Break the symlink by removing cache entry
    cache_path.unlink()

    # Re-create the original file content
    test_file.unlink()
    test_file.write_text("content")

    # Save again - should re-cache since symlink is broken
    h2 = cache.save_to_cache(test_file, cache_dir, checkout_mode=cache.CheckoutMode.SYMLINK)
    assert h2 is not None
    assert h2["hash"] == h1["hash"]
    assert test_file.is_symlink()
    assert cache_path.exists()


def test_save_to_cache_symlink_wrong_hash(tmp_path: pathlib.Path) -> None:
    """Symlink to wrong cache location triggers re-cache."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    h1 = cache.save_to_cache(test_file, cache_dir)
    assert h1 is not None

    # Replace with symlink to wrong location (fake hash)
    test_file.unlink()
    wrong_cache = cache_dir / "aa" / "bbccddee11223344"
    wrong_cache.parent.mkdir(parents=True, exist_ok=True)
    wrong_cache.write_text("wrong content")
    test_file.symlink_to(wrong_cache)

    # Create fresh file with original content
    test_file.unlink()
    test_file.write_text("content")

    # Save again - should use correct hash
    h2 = cache.save_to_cache(test_file, cache_dir)
    assert h2 is not None
    assert h2["hash"] == h1["hash"]


def test_get_symlink_cache_hash_extracts_hash(tmp_path: pathlib.Path) -> None:
    """_get_symlink_cache_hash extracts hash from valid symlink."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(
        test_file, cache_dir, checkout_mode=cache.CheckoutMode.SYMLINK
    )
    assert output_hash is not None

    extracted = cache._get_symlink_cache_hash(test_file, cache_dir)
    assert extracted == output_hash["hash"]


def test_get_symlink_cache_hash_returns_none_for_regular_file(tmp_path: pathlib.Path) -> None:
    """_get_symlink_cache_hash returns None for non-symlink."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    result = cache._get_symlink_cache_hash(test_file, cache_dir)
    assert result is None


def test_get_symlink_cache_hash_returns_none_for_outside_cache(tmp_path: pathlib.Path) -> None:
    """_get_symlink_cache_hash returns None for symlink outside cache."""
    target = tmp_path / "target.txt"
    target.write_text("content")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    cache_dir = tmp_path / "cache"

    result = cache._get_symlink_cache_hash(link, cache_dir)
    assert result is None


# === Link Mode Fallback Tests ===


def test_checkout_with_fallback_uses_first_successful_mode(tmp_path: pathlib.Path) -> None:
    """_checkout_with_fallback uses first mode that succeeds."""
    cache_dir = tmp_path / "cache"
    cache_path = cache_dir / "ab" / "cdef0123456789"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("content")

    target = tmp_path / "target.txt"

    cache._checkout_with_fallback(
        target, cache_path, [cache.CheckoutMode.HARDLINK, cache.CheckoutMode.COPY]
    )

    assert target.exists()
    assert not target.is_symlink()
    # Hardlink should share inode with cache
    assert target.stat().st_ino == cache_path.stat().st_ino


def test_checkout_with_fallback_falls_back_on_exdev(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_checkout_with_fallback falls back to next mode on EXDEV error."""
    import errno
    import os

    cache_dir = tmp_path / "cache"
    cache_path = cache_dir / "ab" / "cdef0123456789"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("content")

    target = tmp_path / "target.txt"

    # Mock os.link to raise EXDEV
    def mock_link(src: str, dst: str) -> None:
        raise OSError(errno.EXDEV, "Cross-device link")

    monkeypatch.setattr(os, "link", mock_link)

    cache._checkout_with_fallback(
        target, cache_path, [cache.CheckoutMode.HARDLINK, cache.CheckoutMode.COPY]
    )

    assert target.exists()
    # Should have fallen back to COPY, so different inode
    assert target.stat().st_ino != cache_path.stat().st_ino
    assert target.read_text() == "content"


def test_checkout_with_fallback_raises_on_last_mode_failure(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_checkout_with_fallback raises error when all modes fail."""
    import errno
    import os
    import shutil

    cache_dir = tmp_path / "cache"
    cache_path = cache_dir / "ab" / "cdef0123456789"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("content")

    target = tmp_path / "target.txt"

    # Mock both os.link and shutil.copy2 to fail
    def mock_link(src: str, dst: str) -> None:
        raise OSError(errno.EPERM, "Permission denied")

    def mock_copy2(src: str, dst: str) -> None:
        raise OSError(errno.EACCES, "Access denied")

    monkeypatch.setattr(os, "link", mock_link)
    monkeypatch.setattr(shutil, "copy2", mock_copy2)

    with pytest.raises(OSError) as exc_info:
        cache._checkout_with_fallback(
            target, cache_path, [cache.CheckoutMode.HARDLINK, cache.CheckoutMode.COPY]
        )

    assert exc_info.value.errno == errno.EACCES


def test_restore_from_cache_with_checkout_modes_list(tmp_path: pathlib.Path) -> None:
    """restore_from_cache accepts checkout_modes list."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(test_file, cache_dir)

    # Remove original
    test_file.unlink()

    # Restore with modes list
    result = cache.restore_from_cache(
        test_file,
        output_hash,
        cache_dir,
        checkout_modes=[
            cache.CheckoutMode.HARDLINK,
            cache.CheckoutMode.SYMLINK,
            cache.CheckoutMode.COPY,
        ],
    )

    assert result is True
    assert test_file.exists()


def test_restore_from_cache_single_mode_raises_on_non_recoverable_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """restore_from_cache raises error for non-recoverable failures."""
    import errno
    import os

    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(test_file, cache_dir)
    test_file.unlink()

    # Mock os.link to fail with EPERM (not in fallback list for _checkout_from_cache)
    def mock_link(src: str, dst: str) -> None:
        raise OSError(errno.EPERM, "Permission denied")

    monkeypatch.setattr(os, "link", mock_link)

    # EPERM is not handled internally by _checkout_from_cache for HARDLINK
    with pytest.raises(OSError) as exc_info:
        cache.restore_from_cache(
            test_file,
            output_hash,
            cache_dir,
            checkout_mode=cache.CheckoutMode.HARDLINK,
        )

    assert exc_info.value.errno == errno.EPERM


def test_save_to_cache_with_checkout_modes_list(tmp_path: pathlib.Path) -> None:
    """save_to_cache accepts checkout_modes list."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(
        test_file,
        cache_dir,
        checkout_modes=[cache.CheckoutMode.HARDLINK, cache.CheckoutMode.SYMLINK],
    )

    assert output_hash is not None
    assert "hash" in output_hash


# === Scandir Skip Tests (Hot Path Ignore) ===


def test_scandir_recursive_skips_pycache(tmp_path: pathlib.Path) -> None:
    """_scandir_recursive should skip __pycache__ directories."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "real.py").write_text("# code")
    pycache = test_dir / "__pycache__"
    pycache.mkdir()
    (pycache / "real.cpython-313.pyc").write_text("bytecode")

    _, manifest = cache.hash_directory(test_dir)

    relpaths = [e["relpath"] for e in manifest]
    assert "real.py" in relpaths
    assert "__pycache__/real.cpython-313.pyc" not in relpaths


def test_scandir_recursive_skips_venv(tmp_path: pathlib.Path) -> None:
    """_scandir_recursive should skip .venv and venv directories."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "app.py").write_text("# code")

    # Both .venv and venv should be skipped
    for venv_name in [".venv", "venv"]:
        venv_dir = test_dir / venv_name
        venv_dir.mkdir()
        (venv_dir / "pyvenv.cfg").write_text("home = /usr/bin")

    _, manifest = cache.hash_directory(test_dir)

    relpaths = [e["relpath"] for e in manifest]
    assert "app.py" in relpaths
    assert ".venv/pyvenv.cfg" not in relpaths
    assert "venv/pyvenv.cfg" not in relpaths


def test_scandir_recursive_skips_git(tmp_path: pathlib.Path) -> None:
    """_scandir_recursive should skip .git directories."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "code.py").write_text("# code")
    git_dir = test_dir / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]")
    objects_dir = git_dir / "objects"
    objects_dir.mkdir()
    (objects_dir / "pack").mkdir()

    _, manifest = cache.hash_directory(test_dir)

    relpaths = [e["relpath"] for e in manifest]
    assert "code.py" in relpaths
    assert ".git/config" not in relpaths
    assert ".git/objects/pack" not in relpaths


def test_scandir_recursive_skips_pyc_files(tmp_path: pathlib.Path) -> None:
    """_scandir_recursive should skip .pyc and .pyo files."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "module.py").write_text("# code")
    (test_dir / "module.pyc").write_text("bytecode")
    (test_dir / "module.pyo").write_text("optimized")

    _, manifest = cache.hash_directory(test_dir)

    relpaths = [e["relpath"] for e in manifest]
    assert "module.py" in relpaths
    assert "module.pyc" not in relpaths
    assert "module.pyo" not in relpaths


def test_scandir_recursive_skips_swap_files(tmp_path: pathlib.Path) -> None:
    """_scandir_recursive should skip vim swap files (.swp, .swo, ~)."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "file.txt").write_text("content")
    (test_dir / "file.txt.swp").write_text("swap")
    (test_dir / "file.txt.swo").write_text("swap2")
    (test_dir / "file.txt~").write_text("backup")
    (test_dir / ".#file.txt").write_text("emacs lock")

    _, manifest = cache.hash_directory(test_dir)

    relpaths = [e["relpath"] for e in manifest]
    assert "file.txt" in relpaths
    assert "file.txt.swp" not in relpaths
    assert "file.txt.swo" not in relpaths
    assert "file.txt~" not in relpaths
    assert ".#file.txt" not in relpaths


def test_scandir_recursive_skips_ide_dirs(tmp_path: pathlib.Path) -> None:
    """_scandir_recursive should skip IDE directories (.idea, .vscode)."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "main.py").write_text("# code")

    for ide_dir in [".idea", ".vscode"]:
        d = test_dir / ide_dir
        d.mkdir()
        (d / "settings.json").write_text("{}")

    _, manifest = cache.hash_directory(test_dir)

    relpaths = [e["relpath"] for e in manifest]
    assert "main.py" in relpaths
    assert ".idea/settings.json" not in relpaths
    assert ".vscode/settings.json" not in relpaths


def test_scandir_recursive_skips_build_dirs(tmp_path: pathlib.Path) -> None:
    """_scandir_recursive should skip build output directories."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "setup.py").write_text("# setup")

    for build_dir in ["dist", "build", "node_modules"]:
        d = test_dir / build_dir
        d.mkdir()
        (d / "artifact").write_text("build output")

    _, manifest = cache.hash_directory(test_dir)

    relpaths = [e["relpath"] for e in manifest]
    assert "setup.py" in relpaths
    assert "dist/artifact" not in relpaths
    assert "build/artifact" not in relpaths
    assert "node_modules/artifact" not in relpaths


def test_scandir_recursive_skips_pivot_internal(tmp_path: pathlib.Path) -> None:
    """_scandir_recursive should skip .pivot internal directory."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "pipeline.py").write_text("# pipeline")
    pivot_dir = test_dir / ".pivot"
    pivot_dir.mkdir()
    (pivot_dir / "state.lmdb").write_text("database")

    _, manifest = cache.hash_directory(test_dir)

    relpaths = [e["relpath"] for e in manifest]
    assert "pipeline.py" in relpaths
    assert ".pivot/state.lmdb" not in relpaths


def test_scandir_recursive_does_not_skip_regular_dirs(tmp_path: pathlib.Path) -> None:
    """_scandir_recursive should not skip regular directories."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "main.py").write_text("# main")
    src_dir = test_dir / "src"
    src_dir.mkdir()
    (src_dir / "module.py").write_text("# module")
    data_dir = test_dir / "data"
    data_dir.mkdir()
    (data_dir / "input.csv").write_text("a,b,c")

    _, manifest = cache.hash_directory(test_dir)

    relpaths = [e["relpath"] for e in manifest]
    assert "main.py" in relpaths
    assert "src/module.py" in relpaths
    assert "data/input.csv" in relpaths


# === _resolve_checkout_modes Tests ===


def test_resolve_checkout_modes_single_mode() -> None:
    """Single checkout_mode returns list with that mode."""
    result = cache._resolve_checkout_modes(cache.CheckoutMode.COPY, None)
    assert result == [cache.CheckoutMode.COPY]


def test_resolve_checkout_modes_list_returned_as_is() -> None:
    """checkout_modes list is returned unchanged."""
    modes = [cache.CheckoutMode.HARDLINK, cache.CheckoutMode.COPY]
    result = cache._resolve_checkout_modes(None, modes)
    assert result == modes


def test_resolve_checkout_modes_empty_list_raises() -> None:
    """Empty checkout_modes list raises ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        cache._resolve_checkout_modes(None, [])


def test_resolve_checkout_modes_both_none_uses_default() -> None:
    """Both None uses default order."""
    result = cache._resolve_checkout_modes(None, None)
    assert result == cache.DEFAULT_CHECKOUT_MODE_ORDER


# === get_cache_path Validation Tests ===


def test_get_cache_path_wrong_length_raises(tmp_path: pathlib.Path) -> None:
    """Hash with wrong length raises SecurityValidationError."""
    from pivot import exceptions

    # Too short
    with pytest.raises(exceptions.SecurityValidationError, match="exactly"):
        cache.get_cache_path(tmp_path, "abc123")

    # Too long
    with pytest.raises(exceptions.SecurityValidationError, match="exactly"):
        cache.get_cache_path(tmp_path, "a" * 20)


def test_get_cache_path_invalid_hex_raises(tmp_path: pathlib.Path) -> None:
    """Hash with invalid hex chars raises SecurityValidationError."""
    from pivot import exceptions

    # 16 chars but not hex (contains 'g')
    with pytest.raises(exceptions.SecurityValidationError, match="invalid characters"):
        cache.get_cache_path(tmp_path, "0123456789abcdeg")


def test_get_cache_path_valid_returns_path(tmp_path: pathlib.Path) -> None:
    """Valid hash returns correct cache path structure."""
    result = cache.get_cache_path(tmp_path, "0123456789abcdef")
    assert result == tmp_path / "01" / "23456789abcdef"


# === _should_skip_entry Tests ===


def test_should_skip_entry_pycache_dir(tmp_path: pathlib.Path) -> None:
    """_should_skip_entry skips __pycache__ directories."""
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()

    with os.scandir(tmp_path) as entries:
        for entry in entries:
            if entry.name == "__pycache__":
                assert cache._should_skip_entry(entry) is True


def test_should_skip_entry_pyc_files(tmp_path: pathlib.Path) -> None:
    """_should_skip_entry skips .pyc, .pyo files."""
    (tmp_path / "module.pyc").write_text("bytecode")
    (tmp_path / "module.pyo").write_text("optimized")

    with os.scandir(tmp_path) as entries:
        for entry in entries:
            if entry.name.endswith((".pyc", ".pyo")):
                assert cache._should_skip_entry(entry) is True


def test_should_skip_entry_swap_files(tmp_path: pathlib.Path) -> None:
    """_should_skip_entry skips .swp and .swo vim swap files."""
    (tmp_path / "file.swp").write_text("swap")
    (tmp_path / "file.swo").write_text("swap2")

    with os.scandir(tmp_path) as entries:
        for entry in entries:
            if entry.name.endswith((".swp", ".swo")):
                assert cache._should_skip_entry(entry) is True


def test_should_skip_entry_backup_files(tmp_path: pathlib.Path) -> None:
    """_should_skip_entry skips ~ backup files."""
    (tmp_path / "file.txt~").write_text("backup")

    with os.scandir(tmp_path) as entries:
        for entry in entries:
            if entry.name.endswith("~"):
                assert cache._should_skip_entry(entry) is True


def test_should_skip_entry_emacs_lock_files(tmp_path: pathlib.Path) -> None:
    """_should_skip_entry skips .# Emacs lock files."""
    (tmp_path / ".#file.txt").write_text("emacs lock")

    with os.scandir(tmp_path) as entries:
        for entry in entries:
            if entry.name.startswith(".#"):
                assert cache._should_skip_entry(entry) is True


def test_should_skip_entry_regular_file_not_skipped(tmp_path: pathlib.Path) -> None:
    """_should_skip_entry does not skip regular files."""
    (tmp_path / "module.py").write_text("# code")
    (tmp_path / "data.csv").write_text("a,b")

    with os.scandir(tmp_path) as entries:
        for entry in entries:
            if entry.name in ("module.py", "data.csv"):
                assert cache._should_skip_entry(entry) is False


# === atomic_write_file Tests ===


def test_atomic_write_file_normal_write(tmp_path: pathlib.Path) -> None:
    """atomic_write_file writes file atomically."""
    dest = tmp_path / "output.txt"

    def write_fn(fd: int) -> None:
        with os.fdopen(fd, "w") as f:
            f.write("test content")

    cache.atomic_write_file(dest, write_fn)

    assert dest.exists()
    assert dest.read_text() == "test content"
    _helper_assert_no_temp_dirs(tmp_path)


def test_atomic_write_file_cleans_up_on_exception(tmp_path: pathlib.Path) -> None:
    """atomic_write_file cleans up temp file on exception."""
    dest = tmp_path / "output.txt"

    def failing_write_fn(fd: int) -> None:
        # Don't close fd, let exception propagate
        raise RuntimeError("write failed")

    with pytest.raises(RuntimeError, match="write failed"):
        cache.atomic_write_file(dest, failing_write_fn)

    assert not dest.exists()
    # No temp files should remain
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0


def test_atomic_write_file_fd_closed_on_exception(tmp_path: pathlib.Path) -> None:
    """File descriptor is closed even on exception before fdopen."""
    dest = tmp_path / "output.txt"

    def write_fn_raises_before_fdopen(fd: int) -> None:
        _ = fd  # Mark as intentionally unused
        # Simulate exception before taking ownership via fdopen
        raise RuntimeError("early failure")

    # The fd should be closed in the finally block
    with pytest.raises(RuntimeError, match="early failure"):
        cache.atomic_write_file(dest, write_fn_raises_before_fdopen)

    # No temp files should remain
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0


# === copy_to_cache Tests ===


def test_copy_to_cache_creates_read_only_file(tmp_path: pathlib.Path) -> None:
    """copy_to_cache creates file with mode 0o444."""
    src = tmp_path / "source.txt"
    src.write_text("content")
    cache_path = tmp_path / "cache" / "ab" / "cdef"
    cache_path.parent.mkdir(parents=True)

    cache.copy_to_cache(src, cache_path)

    assert cache_path.exists()
    mode = cache_path.stat().st_mode & 0o777
    assert mode == 0o444


def test_copy_to_cache_skips_if_exists(tmp_path: pathlib.Path) -> None:
    """copy_to_cache is idempotent - skips if cache path exists."""
    src = tmp_path / "source.txt"
    src.write_text("content")
    cache_path = tmp_path / "cache" / "ab" / "cdef"
    cache_path.parent.mkdir(parents=True)

    # Create existing cache entry
    cache_path.write_text("existing content")

    # Should skip, not overwrite
    cache.copy_to_cache(src, cache_path)

    assert cache_path.read_text() == "existing content"


# === _make_writable_and_retry Tests ===


def test_make_writable_and_retry_makes_parent_writable(tmp_path: pathlib.Path) -> None:
    """_make_writable_and_retry makes parent directory writable before retrying."""
    # Create a read-only directory with a file
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()
    test_file = ro_dir / "file.txt"
    test_file.write_text("content")

    # Make directory read-only (can't delete files inside)
    os.chmod(ro_dir, 0o555)

    try:
        # Should make parent writable and allow deletion
        def mock_unlink(path: str) -> None:
            pathlib.Path(path).unlink()

        # The retry should succeed after chmod
        cache._make_writable_and_retry(mock_unlink, str(test_file), OSError("test"))
        assert not test_file.exists()
    finally:
        # Cleanup
        os.chmod(ro_dir, 0o755)


def test_make_writable_and_retry_handles_directory(tmp_path: pathlib.Path) -> None:
    """_make_writable_and_retry handles read-only directories."""
    # Create a read-only directory
    ro_dir = tmp_path / "readonly_dir"
    ro_dir.mkdir()
    os.chmod(ro_dir, 0o555)

    try:
        call_count = 0

        def mock_rmdir(path: str) -> None:
            nonlocal call_count
            call_count += 1
            pathlib.Path(path).rmdir()

        # Should make directory writable and allow rmdir
        cache._make_writable_and_retry(mock_rmdir, str(ro_dir), OSError("test"))
        assert call_count == 1
        assert not ro_dir.exists()
    finally:
        if ro_dir.exists():
            os.chmod(ro_dir, 0o755)


# === Path Traversal Security Tests ===


# === Symlink Fast Path Tests ===


def test_restore_directory_symlink_fast_path(tmp_path: pathlib.Path) -> None:
    """_restore_directory_from_cache uses symlink fast path when cache dir exists."""

    # Create source directory with files
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    (src_dir / "file1.txt").write_text("content1")
    (src_dir / "sub").mkdir()
    (src_dir / "sub" / "file2.txt").write_text("content2")

    # Create cache directory
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Save to cache with SYMLINK mode - this creates the cache dir entry
    dir_hash_info = cache.save_to_cache(
        src_dir, cache_dir, state_db=None, checkout_mode=cache.CheckoutMode.SYMLINK
    )
    assert isinstance(dir_hash_info, dict) and "manifest" in dir_hash_info

    # Now src_dir is a symlink - remove it to test restore
    src_dir.unlink()

    # Restore to a different location using SYMLINK mode (should use fast path)
    output_dir = tmp_path / "restored"
    result = cache._restore_directory_from_cache(
        output_dir, dir_hash_info, cache_dir, [cache.CheckoutMode.SYMLINK]
    )

    assert result is True
    # Output should be a symlink to cache dir
    assert output_dir.is_symlink()
    # Contents should be accessible
    assert (output_dir / "file1.txt").read_text() == "content1"
    assert (output_dir / "sub" / "file2.txt").read_text() == "content2"


def test_restore_directory_symlink_fallback_on_failure(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """_restore_directory_from_cache falls back to file-by-file when symlink fails."""
    # Create source directory with a file
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    (src_dir / "file.txt").write_text("content")

    # Create cache directory
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Save to cache with SYMLINK mode - this creates the cache dir entry
    output_hash = cache.save_to_cache(
        src_dir, cache_dir, state_db=None, checkout_mode=cache.CheckoutMode.SYMLINK
    )
    # Narrow type - save_to_cache on directory returns DirHash
    assert output_hash is not None and "manifest" in output_hash

    # Now src_dir is a symlink - remove it to test restore
    src_dir.unlink()

    # Restore to a different location, but make symlink creation fail
    output_dir = tmp_path / "restored"

    # Mock Path.symlink_to to fail, forcing fallback to file-by-file
    original_symlink_to = pathlib.Path.symlink_to

    def failing_symlink_to(self: pathlib.Path, target: pathlib.Path) -> None:
        if "symlink_" in str(self):  # Only fail for temp symlinks
            raise OSError("Simulated symlink failure")
        original_symlink_to(self, target)

    mocker.patch.object(pathlib.Path, "symlink_to", failing_symlink_to)
    result = cache._restore_directory_from_cache(
        output_dir, output_hash, cache_dir, [cache.CheckoutMode.SYMLINK, cache.CheckoutMode.COPY]
    )

    assert result is True
    # Output should NOT be a symlink (fell back to COPY)
    assert not output_dir.is_symlink()
    # Contents should still be accessible
    assert (output_dir / "file.txt").read_text() == "content"


def test_restore_directory_copy_mode(tmp_path: pathlib.Path) -> None:
    """_restore_directory_from_cache works with COPY mode."""
    import shutil

    from pivot.types import DirHash

    # Create source directory with a file
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    (src_dir / "file.txt").write_text("content")

    # Create cache directory
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Hash and cache individual files (not using SYMLINK mode)
    dir_hash, manifest = cache.hash_directory(src_dir)

    # Cache each file individually
    for entry in manifest:
        file_path = src_dir / entry["relpath"]
        cache_path = cache.get_cache_path(cache_dir, entry["hash"])
        cache.copy_to_cache(file_path, cache_path)

    # Remove the source directory
    shutil.rmtree(src_dir)

    # Create the DirHash for restoration
    output_hash = DirHash(hash=dir_hash, manifest=manifest)

    # Restore using COPY mode
    output_dir = tmp_path / "restored"
    result = cache._restore_directory_from_cache(
        output_dir, output_hash, cache_dir, [cache.CheckoutMode.COPY]
    )

    assert result is True
    # Output should NOT be a symlink
    assert not output_dir.is_symlink()
    # Contents should be accessible
    assert (output_dir / "file.txt").read_text() == "content"


def test_restore_directory_rejects_path_traversal(tmp_path: pathlib.Path) -> None:
    """_restore_directory_from_cache rejects manifests with path traversal."""
    from pivot import exceptions
    from pivot.types import DirHash, DirManifestEntry

    # Create fake cache dir
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create a directory to restore to
    output_dir = tmp_path / "output"

    # Create a malicious DirHash with path traversal in manifest
    manifest_entry = DirManifestEntry(
        relpath="../../../etc/passwd",  # Path traversal attempt
        hash="b" * cache.XXHASH64_HEX_LENGTH,
        size=100,
        isexec=False,
    )
    malicious_hash = DirHash(
        hash="a" * cache.XXHASH64_HEX_LENGTH,  # Fake hash
        manifest=[manifest_entry],
    )

    # First, create the cache files so we get past the existence check
    cache_path = cache.get_cache_path(cache_dir, "b" * cache.XXHASH64_HEX_LENGTH)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("fake content")
    os.chmod(cache_path, 0o444)

    # Should raise SecurityValidationError
    with pytest.raises(exceptions.SecurityValidationError, match="path traversal"):
        cache._restore_directory_from_cache(
            output_dir, malicious_hash, cache_dir, [cache.CheckoutMode.COPY]
        )
