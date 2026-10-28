from __future__ import annotations

import concurrent.futures
import datetime
import errno
import fcntl
import logging
import mmap
import multiprocessing
import os
import pathlib
import queue
import shutil
import stat
import time
from typing import TYPE_CHECKING, Any, cast

import pytest

from pivot import exceptions
from pivot.storage import cache, state
from pivot.types import DirHash, DirManifestEntry, FileHash, HashInfo, is_dir_hash
from tests import xxh64_forge

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _helper_entries(directory: pathlib.Path) -> list[str]:
    """The sorted names in a cache prefix directory, less its lock files, which persist."""
    return sorted(
        p.name
        for p in directory.iterdir()
        if p.name != cache._PREFIX_LOCK and not p.name.endswith(cache._TREE_LOCK_SUFFIX)
    )


def _helper_assert_no_temp_dirs(
    parent: pathlib.Path,
    *,
    allow_lock_files: bool = True,
    state_dir: pathlib.Path | None = None,
) -> None:
    """Assert no restore/backup temp directories exist.

    Lock files live in state_dir/locks/, not next to data files.
    When allow_lock_files=False, also checks the locks dir for leftover lock files.
    """
    unexpected = list[pathlib.Path]()
    for p in parent.iterdir():
        if p.name.startswith(cache._RESTORE_TEMP_PREFIX) or p.name.startswith(
            cache._BACKUP_TEMP_PREFIX
        ):
            unexpected.append(p)
    if not allow_lock_files and state_dir is not None:
        lock_dir = state_dir / "locks"
        if lock_dir.exists():
            for p in lock_dir.iterdir():
                if p.name.startswith(cache._RESTORE_LOCK_PREFIX):
                    unexpected.append(p)
    assert len(unexpected) == 0, f"Found unexpected temp artifacts: {unexpected}"


if TYPE_CHECKING:
    from collections.abc import Generator
    from multiprocessing.managers import SyncManager
    from threading import Barrier, Event


@pytest.fixture
def mp_manager() -> Generator[SyncManager]:
    """Provide a multiprocessing Manager with automatic cleanup."""
    manager = multiprocessing.get_context("spawn").Manager()
    yield manager
    manager.shutdown()


# === Hash File Tests ===


def test_hash_file(tmp_path: pathlib.Path) -> None:
    """hash_file returns consistent xxhash64 hash."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("hello world")

    hash1, stat1 = cache.hash_file(test_file)
    hash2, stat2 = cache.hash_file(test_file)

    assert hash1 == hash2
    assert len(hash1) == 16  # xxhash64 hex is 16 chars
    assert stat1.st_size == stat2.st_size


def test_hash_file_different_content(tmp_path: pathlib.Path) -> None:
    """Different content produces different hash."""
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file1.write_text("hello")
    file2.write_text("world")

    hash1, _ = cache.hash_file(file1)
    hash2, _ = cache.hash_file(file2)
    assert hash1 != hash2


def test_hash_file_uses_state_cache(tmp_path: pathlib.Path) -> None:
    """hash_file uses state cache to skip rehashing."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        cache.hash_file(test_file, state_db=db)  # First hash populates cache
        db.save(test_file, test_file.stat(), "cached_hash")
        hash2, _ = cache.hash_file(test_file, state_db=db)

    assert hash2 == "cached_hash"


def test_hash_file_binary(tmp_path: pathlib.Path) -> None:
    """hash_file works with binary content."""
    test_file = tmp_path / "binary.bin"
    test_file.write_bytes(b"\x00\x01\x02\xff\xfe")

    file_hash, _ = cache.hash_file(test_file)

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
    small_hash, _ = cache.hash_file(small_file)
    large_hash, _ = cache.hash_file(large_file)

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
    mmap_hash, _ = cache.hash_file(test_file)

    # Set threshold above content size to force buffered path
    monkeypatch.setattr(cache, "MMAP_THRESHOLD", len(content) + 100)
    buffered_hash, _ = cache.hash_file(test_file)

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
    expected_hash, _ = cache.hash_file(test_file)

    # Now make mmap fail
    def failing_mmap(*args: object, **kwargs: object) -> mmap.mmap:
        raise OSError("mmap failed")

    monkeypatch.setattr(mmap, "mmap", failing_mmap)

    # Should fall back to buffered read and produce same hash
    fallback_hash, _ = cache.hash_file(test_file)
    assert fallback_hash == expected_hash


# === Hash Directory Tests ===


def test_hash_directory(tmp_path: pathlib.Path) -> None:
    """hash_directory returns the tree hash and manifest."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "a.txt").write_text("content a")
    (test_dir / "b.txt").write_text("content b")

    result = cache.hash_directory(test_dir)

    assert len(result["hash"]) == 16
    assert len(result["manifest"]) == 2


def test_hash_directory_warns_about_left_out_symlinks(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A recorded directory names each user symlink it leaves out, in one warning."""
    test_dir = tmp_path / "artifact"
    (test_dir / "nested").mkdir(parents=True)
    target = test_dir / "target.txt"
    target.write_text("target")
    (test_dir / "link.txt").symlink_to(target)
    (test_dir / "nested" / "other.txt").symlink_to(target)

    with caplog.at_level(logging.WARNING, logger="pivot.storage.cache"):
        manifest = cache.hash_directory(test_dir, warn_left_out_symlinks=True)["manifest"]

    assert [entry["relpath"] for entry in manifest] == ["target.txt"]
    assert [record.getMessage() for record in caplog.records] == [
        f"Directory artifact {test_dir} leaves out symlinks link.txt, nested/other.txt "
        + "(a symlink inside a directory artifact is not stored); "
        + "replace each with a real file or directory"
    ]


def test_hash_directory_rehash_is_silent_about_left_out_symlinks(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Status and skip checks re-hash a directory without repeating the warning."""
    test_dir = tmp_path / "artifact"
    test_dir.mkdir()
    (test_dir / "target.txt").write_text("target")
    (test_dir / "link.txt").symlink_to(test_dir / "target.txt")

    with caplog.at_level(logging.WARNING, logger="pivot.storage.cache"):
        cache.hash_directory(test_dir)

    assert caplog.records == []


def test_hash_directory_hashes_cache_object_links_by_target(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A symlink checkout's link to a cache object counts as the file it points to."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "keep.txt").write_text("keep")
    cache_dir = tmp_path / "cache"
    expected = cache.hash_directory(source)
    cache.save_to_cache(source, cache_dir, checkout_mode=cache.CheckoutMode.COPY)
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "keep.txt").symlink_to(
        cache.get_cache_path(cache_dir, expected["manifest"][0]["hash"])
    )

    with caplog.at_level(logging.WARNING, logger="pivot.storage.cache"):
        result = cache.hash_directory(linked, cache_dir=cache_dir, warn_left_out_symlinks=True)

    assert result == expected
    assert caplog.records == []


def test_hash_directory_leaves_out_links_to_hash_named_files_outside_the_cache(
    tmp_path: pathlib.Path,
) -> None:
    """Only a link into the given cache root is content, however its target is named."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "keep.txt").write_text("keep")
    cache_dir = tmp_path / "cache"
    manifest = cache.hash_directory(source)["manifest"]
    cache.save_to_cache(source, cache_dir, checkout_mode=cache.CheckoutMode.COPY)
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "keep.txt").symlink_to(cache.get_cache_path(cache_dir, manifest[0]["hash"]))

    assert cache.hash_directory(linked, cache_dir=tmp_path / "other-cache")["manifest"] == []
    assert cache.hash_directory(linked)["manifest"] == []


def test_hash_directory_relative_paths(tmp_path: pathlib.Path) -> None:
    """Manifest contains relative paths only."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "file.txt").write_text("content")
    subdir = test_dir / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("nested")

    manifest = cache.hash_directory(test_dir)["manifest"]

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

    manifest = cache.hash_directory(test_dir)["manifest"]

    relpaths = [e["relpath"] for e in manifest]
    assert relpaths == sorted(relpaths)


def test_hash_directory_deterministic(tmp_path: pathlib.Path) -> None:
    """Same directory content produces same hash."""
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    for d in [dir1, dir2]:
        d.mkdir()
        (d / "file.txt").write_text("same content")

    hash1 = cache.hash_directory(dir1)["hash"]
    hash2 = cache.hash_directory(dir2)["hash"]

    assert hash1 == hash2


def test_hash_directory_includes_size(tmp_path: pathlib.Path) -> None:
    """Manifest entries include file size."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "file.txt").write_text("hello")

    manifest = cache.hash_directory(test_dir)["manifest"]

    assert manifest[0]["size"] == 5


def test_hash_directory_skips_symlinks(tmp_path: pathlib.Path) -> None:
    """Symlinks in directory are skipped."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "real.txt").write_text("content")
    (test_dir / "link.txt").symlink_to(test_dir / "real.txt")

    manifest = cache.hash_directory(test_dir)["manifest"]

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

    manifest = cache.hash_directory(test_dir)["manifest"]

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

    original_hash_contents = cache._hash_contents

    def hash_contents_after_delete(path: str | os.PathLike[str], size: int) -> str:
        if os.fspath(path) == os.fspath(to_delete):
            to_delete.unlink()
        return original_hash_contents(path, size)

    monkeypatch.setattr(cache, "_hash_contents", hash_contents_after_delete)

    manifest = cache.hash_directory(test_dir)["manifest"]
    assert len(manifest) == 1
    assert manifest[0]["relpath"] == "keep.txt"


def test_hash_file_permission_error(tmp_path: pathlib.Path) -> None:
    """hash_file raises PermissionError for unreadable files (fail-fast)."""
    test_file = tmp_path / "unreadable.txt"
    test_file.write_text("content")
    test_file.chmod(0o000)

    try:
        with pytest.raises(PermissionError):
            cache.hash_file(test_file)
    finally:
        # Cleanup
        test_file.chmod(0o644)


def test_hash_file_state_cache_invalidation(tmp_path: pathlib.Path) -> None:
    """State cache correctly invalidates when file mtime or size changes."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("original")
    db_path = tmp_path

    with state.StateDB(db_path) as db:
        # First hash - cache miss
        hash1, _ = cache.hash_file(test_file, state_db=db)

        # Second hash - cache hit (should return cached value)
        cached_hash, _ = cache.hash_file(test_file, state_db=db)
        assert cached_hash == hash1

        # Modify file content (changes both mtime and size)
        import time

        time.sleep(0.01)  # Ensure mtime changes
        test_file.write_text("modified content")

        # Third hash - cache should be invalidated
        hash2, _ = cache.hash_file(test_file, state_db=db)
        assert hash2 != hash1, "Hash should change when file content changes"


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

    tmp_files = [*cache_dir.rglob("*.tmp"), *cache_dir.rglob(f"{cache._CACHE_TEMP_PREFIX}*")]
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
    _helper_assert_no_temp_dirs(tmp_path, allow_lock_files=False, state_dir=tmp_path / ".pivot")


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
    first_hash, _ = cache.hash_file(first_file)
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


def test_cleanup_removes_stale_prefixed_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale file staged for restore is cleaned up like a stale directory."""
    stale_file = tmp_path / f"{cache._RESTORE_TEMP_PREFIX}old.file"
    stale_file.write_text("orphan")
    monkeypatch.setattr(cache, "_RESTORE_TEMP_MAX_AGE", -1)

    cache._cleanup_stale_restore_temps(tmp_path)

    assert not stale_file.exists()


def _crash_save_after_staging(path: pathlib.Path, cache_dir: pathlib.Path) -> None:
    """Exit after save staged its workspace replacement, before its atomic rename."""

    def _replace_and_crash(self: pathlib.Path, destination: pathlib.Path) -> pathlib.Path:
        os._exit(73)

    pathlib.Path.replace = _replace_and_crash  # pyright: ignore[reportAttributeAccessIssue]
    _ = cache.save_to_cache(path, cache_dir, checkout_mode=cache.CheckoutMode.COPY)


def test_save_file_cleans_staging_left_by_killed_save(tmp_path: pathlib.Path) -> None:
    """A later save clears the deterministic file staging path an interrupted save left."""
    target = tmp_path / "data" / "out.txt"
    target.parent.mkdir(parents=True)
    target.write_text("stage output\n")
    cache_dir = tmp_path / "cache"
    _ = cache.save_to_cache(target, cache_dir, checkout_mode=cache.CheckoutMode.COPY)
    target.unlink()
    target.write_text("stage output\n")

    process = multiprocessing.get_context("spawn").Process(
        target=_crash_save_after_staging, args=(target, cache_dir)
    )
    process.start()
    process.join(timeout=60)

    staging_path = target.parent / (
        f"{cache._RESTORE_TEMP_PREFIX}{cache._restore_target_token(target)}.file"
    )
    assert process.exitcode == 73
    assert staging_path.exists()

    _ = cache.save_to_cache(target, cache_dir, checkout_mode=cache.CheckoutMode.COPY)

    assert not staging_path.exists()
    assert target.read_text() == "stage output\n"


def test_save_directory_uses_target_tagged_staging_paths(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Directory saves use deterministic staging names for each workspace file."""
    target = tmp_path / "data" / "out"
    target.mkdir(parents=True)
    (target / "root.txt").write_text("root")
    (target / "sub").mkdir()
    (target / "sub" / "nested.txt").write_text("nested")
    staging_paths = list[pathlib.Path | None]()
    original_checkout = cache._checkout_with_fallback

    def _record_staging(
        path: pathlib.Path,
        cache_path: pathlib.Path,
        checkout_modes: list[cache.CheckoutMode],
        *,
        executable: bool = False,
        staging_path: pathlib.Path | None = None,
        keep_existing_copy: bool = False,
    ) -> None:
        staging_paths.append(staging_path)
        original_checkout(
            path,
            cache_path,
            checkout_modes,
            executable=executable,
            staging_path=staging_path,
            keep_existing_copy=keep_existing_copy,
        )

    monkeypatch.setattr(cache, "_checkout_with_fallback", _record_staging)

    _ = cache.save_to_cache(target, tmp_path / "cache", checkout_mode=cache.CheckoutMode.COPY)

    workspace_files = [target / "root.txt", target / "sub" / "nested.txt"]
    assert staging_paths == [
        path.parent / f"{cache._RESTORE_TEMP_PREFIX}{cache._restore_target_token(path)}.file"
        for path in workspace_files
    ]


def _save_file_while_restore_lock_is_held(
    path: pathlib.Path,
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
    result_queue: multiprocessing.Queue[str],
) -> None:
    """Save one workspace file under the target's restore lock."""
    _ = cache.save_to_cache(
        path,
        cache_dir,
        checkout_mode=cache.CheckoutMode.COPY,
        state_dir=state_dir,
    )
    result_queue.put("saved")


def test_save_waits_for_target_restore_lock(tmp_path: pathlib.Path) -> None:
    """A save cannot clear a checkout's staging file while that checkout holds the lock."""
    target = tmp_path / "data" / "out.txt"
    target.parent.mkdir(parents=True)
    target.write_text("stage output\n")
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / ".pivot"
    results = multiprocessing.get_context("spawn").Queue()
    process = multiprocessing.get_context("spawn").Process(
        target=_save_file_while_restore_lock_is_held,
        args=(target, cache_dir, state_dir, results),
    )

    with cache.restore_lock(target, state_dir):
        process.start()
        with pytest.raises(queue.Empty):
            results.get(timeout=0.5)

    process.join(timeout=10)
    assert process.exitcode == 0
    assert results.get(timeout=1) == "saved"


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


def test_get_lock_path_lives_under_state_dir_locks(tmp_path: pathlib.Path) -> None:
    """Lock files are created under state_dir/locks/, not next to data."""
    state_dir = tmp_path / ".pivot"
    state_dir.mkdir(parents=True)
    target = tmp_path / "data" / "mydir"

    result = cache._get_lock_path(target, state_dir)

    assert result.parent == state_dir / "locks"
    assert result.name.startswith(cache._RESTORE_LOCK_PREFIX)


def test_get_lock_path_uses_hash_of_resolved_path(tmp_path: pathlib.Path) -> None:
    """Lock filename is a hash so any target path length is safe."""
    state_dir = tmp_path / ".pivot"
    state_dir.mkdir(parents=True)
    target = tmp_path / ("a" * 300)

    result = cache._get_lock_path(target, state_dir)

    assert len(result.name) <= 255, "Lock filename must fit filesystem limit"
    hash_portion = result.name[len(cache._RESTORE_LOCK_PREFIX) :]
    assert len(hash_portion) == 32
    assert all(c in "0123456789abcdef" for c in hash_portion)


def test_get_lock_path_is_stable_when_target_becomes_symlink(tmp_path: pathlib.Path) -> None:
    """The lock identifies the workspace path, not the cache target it currently points to."""
    state_dir = tmp_path / ".pivot"
    target = tmp_path / "data" / "output.txt"
    target.parent.mkdir(parents=True)
    before = cache._get_lock_path(target, state_dir)

    cache_target = tmp_path / "cache" / "output.txt"
    cache_target.parent.mkdir()
    cache_target.write_text("cached")
    target.symlink_to(cache_target)

    assert cache._get_lock_path(target, state_dir) == before


def test_directory_restore_keeps_lock_file(tmp_path: pathlib.Path) -> None:
    """A waiting restorer must continue to contend on the same lock inode."""
    target = tmp_path / "data" / "out"
    target.mkdir(parents=True)
    (target / "result.txt").write_text("content")
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / ".pivot"
    output_hash = cache.save_to_cache(target, cache_dir, checkout_mode=cache.CheckoutMode.COPY)

    assert cache.restore_from_cache(
        target,
        output_hash,
        cache_dir,
        checkout_mode=cache.CheckoutMode.COPY,
        state_dir=state_dir,
    )
    assert cache._get_lock_path(target, state_dir).exists()


def test_get_lock_path_different_targets_different_locks(tmp_path: pathlib.Path) -> None:
    """Different target paths produce different lock files."""
    state_dir = tmp_path / ".pivot"
    state_dir.mkdir(parents=True)

    lock_a = cache._get_lock_path(tmp_path / "dir_a", state_dir)
    lock_b = cache._get_lock_path(tmp_path / "dir_b", state_dir)

    assert lock_a != lock_b


def _restore_worker(
    test_dir: pathlib.Path,
    output_hash: DirHash,
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
    result_queue: multiprocessing.Queue[tuple[str, str | bool]],
) -> None:
    """Worker function for concurrent restore test (must be module-level for pickling)."""
    try:
        result = cache.restore_from_cache(
            test_dir,
            output_hash,
            cache_dir,
            checkout_mode=cache.CheckoutMode.HARDLINK,
            state_dir=state_dir,
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
    state_dir = tmp_path / ".pivot"
    ctx = multiprocessing.get_context("spawn")
    p1 = ctx.Process(
        target=_restore_worker, args=(test_dir, output_hash, cache_dir, state_dir, result_queue)
    )
    p2 = ctx.Process(
        target=_restore_worker, args=(test_dir, output_hash, cache_dir, state_dir, result_queue)
    )
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    # Both should succeed (second waits for first due to lock)
    results = [result_queue.get(timeout=60), result_queue.get(timeout=60)]
    assert all(r[0] == "success" for r in results), f"Got errors: {results}"
    assert (test_dir / "file.txt").read_text() == "content"


_SHARED_PARENT_FILES = 6
_SHARED_PARENT_DIRS = 4
_FILES_PER_DIR = 10


def _helper_cache_shared_parent_outputs(root: pathlib.Path) -> dict[str, HashInfo]:
    """Cache file and directory outputs that all live in one parent directory."""
    cache_dir = root / "cache"
    parent = root / "data" / "shared"
    parent.mkdir(parents=True)
    hashes = dict[str, HashInfo]()
    for i in range(_SHARED_PARENT_FILES):
        path = parent / f"file{i}.txt"
        path.write_text(f"file {i}\n")
        hashes[str(path)] = cache.save_to_cache(
            path, cache_dir, checkout_mode=cache.CheckoutMode.COPY
        )
    for i in range(_SHARED_PARENT_DIRS):
        path = parent / f"dir{i}"
        (path / "sub").mkdir(parents=True)
        for j in range(_FILES_PER_DIR):
            (path / "sub" / f"f{j}.txt").write_text(f"dir {i} file {j}\n")
        hashes[str(path)] = cache.save_to_cache(
            path, cache_dir, checkout_mode=cache.CheckoutMode.COPY
        )
    return hashes


def _helper_assert_shared_parent_outputs(parent: pathlib.Path) -> None:
    for i in range(_SHARED_PARENT_FILES):
        assert (parent / f"file{i}.txt").read_text() == f"file {i}\n"
    for i in range(_SHARED_PARENT_DIRS):
        sub = parent / f"dir{i}" / "sub"
        assert sorted(p.name for p in sub.iterdir()) == sorted(
            f"f{j}.txt" for j in range(_FILES_PER_DIR)
        )
        for j in range(_FILES_PER_DIR):
            assert (sub / f"f{j}.txt").read_text() == f"dir {i} file {j}\n"
    _helper_assert_no_temp_dirs(parent)


def _helper_restore_all_at_barrier(
    hashes: dict[str, HashInfo],
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    barrier: Barrier,
) -> list[str]:
    """Restore every output from 8 threads at once; return one line per failed restore."""
    barrier.wait()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(
                cache.restore_from_cache,
                pathlib.Path(path),
                output_hash,
                cache_dir,
                checkout_modes=checkout_modes,
                state_dir=state_dir,
            )
            for path, output_hash in hashes.items()
        ]
    failures = list[str]()
    for future in futures:
        exc = future.exception()
        if exc is not None:
            failures.append(f"{type(exc).__name__}: {exc}")
        elif future.result() is not True:
            failures.append("restore_from_cache returned False")
    return failures


@pytest.mark.parametrize(
    "checkout_modes",
    [
        pytest.param([cache.CheckoutMode.COPY], id="copy"),
        pytest.param(cache.DEFAULT_CHECKOUT_MODE_ORDER, id="default-order"),
        pytest.param([cache.CheckoutMode.SYMLINK], id="symlink"),
    ],
)
def test_concurrent_restores_into_shared_parent(
    tmp_path: pathlib.Path, checkout_modes: list[cache.CheckoutMode]
) -> None:
    """Processes restoring the same outputs into one parent directory all succeed (#461)."""
    hashes = _helper_cache_shared_parent_outputs(tmp_path)
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / ".pivot"
    parent = tmp_path / "data" / "shared"
    n_procs = 4
    ctx = multiprocessing.get_context("spawn")
    failures = list[str]()
    with ctx.Manager() as manager, ctx.Pool(n_procs) as pool:
        barrier = manager.Barrier(n_procs, timeout=60)
        for _ in range(5):
            for path in hashes:
                cache.remove_output(pathlib.Path(path))
            results = pool.starmap(
                _helper_restore_all_at_barrier,
                [(hashes, cache_dir, state_dir, checkout_modes, barrier)] * n_procs,
            )
            failures.extend(f for result in results for f in result)
            _helper_assert_shared_parent_outputs(parent)

    assert failures == []


def _helper_restore_repeatedly_at_barrier(
    path: pathlib.Path,
    output_hash: HashInfo,
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
    times: int,
    barrier: Barrier,
) -> list[str]:
    barrier.wait()
    failures = list[str]()
    for _ in range(times):
        try:
            restored = cache.restore_from_cache(
                path,
                output_hash,
                cache_dir,
                checkout_mode=cache.CheckoutMode.COPY,
                state_dir=state_dir,
            )
        except OSError as e:
            failures.append(f"{type(e).__name__}: {e}")
        else:
            if not restored:
                failures.append("restore_from_cache returned False")
    return failures


def test_concurrent_repeated_restores_of_one_directory(tmp_path: pathlib.Path) -> None:
    """The per-path restore lock excludes every concurrent restorer, not just the first (#461).

    Deleting the lock file on release lets a third process lock a fresh file while a
    waiter still holds the old one; both then swap the directory at once.
    """
    target = tmp_path / "data" / "out"
    target.mkdir(parents=True)
    for j in range(5):
        (target / f"f{j}.txt").write_text(f"{j}\n")
    cache_dir = tmp_path / "cache"
    output_hash = cache.save_to_cache(target, cache_dir, checkout_mode=cache.CheckoutMode.COPY)
    n_procs = 8
    ctx = multiprocessing.get_context("spawn")
    with ctx.Manager() as manager, ctx.Pool(n_procs) as pool:
        barrier = manager.Barrier(n_procs, timeout=60)
        results = pool.starmap(
            _helper_restore_repeatedly_at_barrier,
            [(target, output_hash, cache_dir, tmp_path / ".pivot", 40, barrier)] * n_procs,
        )

    assert [f for result in results for f in result] == []
    assert sorted(p.name for p in target.iterdir()) == [f"f{j}.txt" for j in range(5)]
    _helper_assert_no_temp_dirs(target.parent)


def _helper_commit_same_new_outputs_at_barrier(
    root: pathlib.Path,
    worker: int,
    kind: str,
    mode: cache.CheckoutMode,
    rounds: int,
    barrier: Barrier,
) -> list[str]:
    """Each round, write an output every worker writes byte for byte, none of it cached yet,
    then commit it as a stage does; return one line per failed commit."""
    cache_dir = root / "cache"
    failures = list[str]()
    for r in range(rounds):
        content = f"round {r}\n"
        out = root / f"worker{worker}" / f"out{r}"
        output_hash: HashInfo
        if kind == "file":
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content)
            output_hash = FileHash(hash=cache.hash_file(out)[0])
        else:
            (out / "sub").mkdir(parents=True)
            (out / "sub" / "tool").write_text(content)
            executable = kind == "dir-mixed-exec" and worker % 2 == 0
            (out / "sub" / "tool").chmod(0o755 if executable else 0o644)
            output_hash = cache.hash_directory(out)
        barrier.wait()
        try:
            cache.store_hashed_output(
                out,
                output_hash,
                cache_dir,
                [mode, cache.CheckoutMode.COPY],
                state_dir=root / ".pivot",
            )
        except OSError as e:
            failures.append(f"round {r}: {type(e).__name__}: {e}")
    return failures


@pytest.mark.parametrize(
    ("kind", "mode"),
    [
        pytest.param("file", cache.CheckoutMode.HARDLINK, id="file-hardlink"),
        pytest.param("dir-mixed-exec", cache.CheckoutMode.HARDLINK, id="dir-hardlink-mixed-exec"),
        pytest.param("dir", cache.CheckoutMode.SYMLINK, id="dir-symlink"),
    ],
)
def test_concurrent_commits_of_the_same_new_output(
    tmp_path: pathlib.Path, kind: str, mode: cache.CheckoutMode
) -> None:
    """Parallel stages whose outputs hold the same new bytes all commit them.

    Each finds no cache object and publishes one. A writer that renamed its copy over an
    object another had just published took that object's last name while the other was
    hardlinking it, and link() failed with ENOENT; two writers copying one directory tree to
    its cache name collided with EEXIST.
    """
    n_procs = 8
    rounds = 50
    ctx = multiprocessing.get_context("spawn")
    with ctx.Manager() as manager, ctx.Pool(n_procs) as pool:
        barrier = manager.Barrier(n_procs, timeout=60)
        results = pool.starmap(
            _helper_commit_same_new_outputs_at_barrier,
            [(tmp_path, worker, kind, mode, rounds, barrier) for worker in range(n_procs)],
        )

    assert [f for result in results for f in result] == []
    for worker in range(n_procs):
        for r in range(rounds):
            out = tmp_path / f"worker{worker}" / f"out{r}"
            read = out if kind == "file" else out / "sub" / "tool"
            assert read.read_text() == f"round {r}\n"
            executable = kind == "dir-mixed-exec" and worker % 2 == 0
            assert os.access(read, os.X_OK) is executable, f"worker {worker} round {r}"


def _helper_crash_after_backup_rename(
    path: pathlib.Path,
    output_hash: HashInfo,
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
) -> None:
    """Simulate a kill after the original output moves to the backup location."""
    original_rename = pathlib.Path.rename

    def _rename_and_crash(source: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
        result = original_rename(source, target)
        if source == path and target.name.startswith(cache._BACKUP_TEMP_PREFIX):
            os._exit(73)
        return result

    pathlib.Path.rename = _rename_and_crash  # pyright: ignore[reportAttributeAccessIssue]
    _ = cache.restore_from_cache(
        path,
        output_hash,
        cache_dir,
        checkout_mode=cache.CheckoutMode.COPY,
        state_dir=state_dir,
    )


def test_directory_restore_recovers_fresh_crash_orphans(tmp_path: pathlib.Path) -> None:
    """The next restore removes its fresh temp and backup dirs after a process crash (#228)."""
    target = tmp_path / "data" / "out"
    target.mkdir(parents=True)
    (target / "result.txt").write_text("cached content")
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / ".pivot"
    output_hash = cache.save_to_cache(target, cache_dir, checkout_mode=cache.CheckoutMode.COPY)
    (target / "result.txt").write_text("old workspace content")

    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(
        target=_helper_crash_after_backup_rename,
        args=(target, output_hash, cache_dir, state_dir),
    )
    process.start()
    process.join(timeout=60)

    assert process.exitcode == 73
    assert not target.exists()

    assert cache.restore_from_cache(
        target,
        output_hash,
        cache_dir,
        checkout_mode=cache.CheckoutMode.COPY,
        state_dir=state_dir,
    )

    assert (target / "result.txt").read_text() == "cached content"
    _helper_assert_no_temp_dirs(target.parent)


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


@pytest.mark.parametrize("mode", [cache.CheckoutMode.COPY, cache.CheckoutMode.HARDLINK])
def test_save_directory_with_read_only_subdirectory(
    tmp_path: pathlib.Path, mode: cache.CheckoutMode
) -> None:
    """A directory output holding a read-only subdirectory saves and checks out.

    File checkouts stage a temp file beside the target; the stage must not fail on a
    parent the stage's own code made read-only.
    """
    out = tmp_path / "out"
    (out / "sub").mkdir(parents=True)
    (out / "sub" / "f.txt").write_text("content\n")
    (out / "top.txt").write_text("top\n")
    os.chmod(out / "sub", 0o555)
    try:
        output_hash = cache.save_to_cache(out, tmp_path / "cache", checkout_mode=mode)

        assert "manifest" in output_hash
        assert (out / "sub" / "f.txt").read_text() == "content\n"
        _helper_assert_no_temp_dirs(out / "sub")
    finally:
        os.chmod(out / "sub", 0o755)


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


# === Directory artifacts include every file ===

# Names a code-fingerprinting or watch ignore list would skip; a declared directory artifact
# holds them like any other file.
_TOOLING_RELPATHS = [
    "__pycache__/x.pyc",
    "build/y",
    "dist/artifact",
    "node_modules/artifact",
    ".git/config",
    ".hg/store",
    ".svn/entries",
    ".venv/pyvenv.cfg",
    "venv/pyvenv.cfg",
    ".idea/settings.json",
    ".vscode/settings.json",
    ".pivot/state.lmdb",
    "module.pyc",
    "module.pyo",
    "file.txt.swp",
    "file.txt.swo",
    "file.txt~",
    ".#file.txt",
]


def _helper_write_tree(root: pathlib.Path) -> None:
    (root / "keep.txt").write_text("keep")
    for relpath in _TOOLING_RELPATHS:
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"content of {relpath}")


def _helper_tree_contents(root: pathlib.Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): path.read_text()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_hash_directory_includes_every_file(tmp_path: pathlib.Path) -> None:
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    _helper_write_tree(test_dir)

    manifest = cache.hash_directory(test_dir)["manifest"]

    relpaths = {entry["relpath"] for entry in manifest}
    assert relpaths == {"keep.txt", *_TOOLING_RELPATHS}


@pytest.mark.parametrize("relpath", _TOOLING_RELPATHS)
def test_editing_any_file_changes_directory_hash(tmp_path: pathlib.Path, relpath: str) -> None:
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    _helper_write_tree(test_dir)
    before = cache.hash_directory(test_dir)["hash"]

    (test_dir / relpath).write_text("edited")
    after = cache.hash_directory(test_dir)["hash"]

    assert after != before, f"editing {relpath} must change the directory hash"


@pytest.mark.parametrize(
    "mode",
    [cache.CheckoutMode.COPY, cache.CheckoutMode.HARDLINK, cache.CheckoutMode.SYMLINK],
)
def test_directory_output_keeps_every_file_through_cache_and_restore(
    tmp_path: pathlib.Path, mode: cache.CheckoutMode
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    _helper_write_tree(output)
    expected = _helper_tree_contents(output)
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(output, cache_dir, checkout_modes=[mode])
    assert _helper_tree_contents(output) == expected, "saving must keep every file in place"

    cache.remove_output(output)
    assert cache.restore_from_cache(output, output_hash, cache_dir, checkout_mode=mode)

    assert _helper_tree_contents(output) == expected


@pytest.mark.parametrize(
    "mode",
    [cache.CheckoutMode.HARDLINK, cache.CheckoutMode.SYMLINK, cache.CheckoutMode.COPY],
)
def test_directory_output_executable_survives_cache_and_restore(
    tmp_path: pathlib.Path, mode: cache.CheckoutMode
) -> None:
    """An executable file in a directory output keeps its executable bit through
    save_to_cache and restore_from_cache, under every checkout mode.

    Regression: HARDLINK and SYMLINK checkout read their mode straight from the cache
    object, which ``copy_to_cache`` used to chmod to 0o444 unconditionally. A consumer
    stage then re-hashed the checked-out file as non-executable, so its recorded
    dependency hash never matched the producer's recorded output hash and `pivot status`
    reported it stale forever.
    """
    output = tmp_path / "out"
    output.mkdir()
    script = output / "run.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o755)
    (output / "data.txt").write_text("data\n")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(output, cache_dir, checkout_modes=[mode])
    assert os.access(output / "run.sh", os.X_OK), "checkout must keep the executable bit"
    assert not os.access(output / "data.txt", os.X_OK)

    cache.remove_output(output)
    assert cache.restore_from_cache(output, output_hash, cache_dir, checkout_mode=mode)

    assert os.access(output / "run.sh", os.X_OK), "restore must keep the executable bit"
    assert not os.access(output / "data.txt", os.X_OK)


def test_directory_output_isexec_collision_falls_back_to_copy(tmp_path: pathlib.Path) -> None:
    """Two files with identical content but different executable bits hash the same and
    share one cache object. HARDLINK/SYMLINK checkout has no mode of its own, so it can only
    ever reproduce whatever executable bit the shared object already carries; checkout for
    whichever file disagrees with that falls back to COPY instead of silently serving the
    wrong permissions, and the shared object is never changed to suit it.
    """
    cache_dir = tmp_path / "cache"
    modes = [cache.CheckoutMode.HARDLINK, cache.CheckoutMode.SYMLINK, cache.CheckoutMode.COPY]

    non_exec_dir = tmp_path / "non_exec"
    non_exec_dir.mkdir()
    (non_exec_dir / "shared.sh").write_text("#!/bin/sh\necho hi\n")

    exec_dir = tmp_path / "exec"
    exec_dir.mkdir()
    shared_in_exec = exec_dir / "shared.sh"
    shared_in_exec.write_text("#!/bin/sh\necho hi\n")
    shared_in_exec.chmod(0o755)

    cache.save_to_cache(non_exec_dir, cache_dir, checkout_modes=modes)
    assert not os.access(non_exec_dir / "shared.sh", os.X_OK)

    cache.save_to_cache(exec_dir, cache_dir, checkout_modes=modes)
    assert os.access(exec_dir / "shared.sh", os.X_OK), (
        "the second save shares a cache object with the first, stored non-executable; "
        "the collision must not leave exec_dir's own file non-executable"
    )
    assert not os.access(non_exec_dir / "shared.sh", os.X_OK), "the first save is unaffected"


def test_restore_from_cache_repairs_legacy_symlink_tree_isexec(tmp_path: pathlib.Path) -> None:
    """A whole-directory SYMLINK cache tree an older pivot built (every file 0o444
    regardless of isexec) must not keep serving a non-executable file forever: its tree hash
    already incorporates isexec, so the tree directory is never rebuilt, but a restore through
    the symlink fast path must still see the manifest's executable bit.
    """
    cache_dir = tmp_path / "cache"
    modes = [cache.CheckoutMode.SYMLINK]

    output = tmp_path / "out"
    output.mkdir()
    (output / "run.sh").write_text("#!/bin/sh\necho hi\n")
    (output / "run.sh").chmod(0o755)

    output_hash = cache.save_to_cache(output, cache_dir, checkout_modes=modes)
    assert isinstance(output_hash, dict) and "manifest" in output_hash

    # Simulate a tree an older pivot built: force every file in the tree cache dir back to
    # 0o444, as pivot 0.3.2 did regardless of isexec, and drop the mark this pivot gives a tree
    # it built. The tree hash (which already includes isexec) is unchanged.
    tree_path = cache.get_cache_path(cache_dir, output_hash["hash"])
    for f in tree_path.rglob("*"):
        if f.is_file() and not f.is_symlink():
            os.chmod(f, 0o444)
    cache._tree_layout_marker(tree_path).unlink()

    cache.remove_output(output)
    assert cache.restore_from_cache(output, output_hash, cache_dir, checkout_modes=modes)

    assert os.access(output / "run.sh", os.X_OK), (
        "restoring from a legacy tree cache must still see the manifest's executable bit"
    )


def test_save_to_cache_symlink_repairs_legacy_tree_isexec(tmp_path: pathlib.Path) -> None:
    """`_store_directory`'s SYMLINK branch must repair a legacy tree directory's isexec bits
    even when it already exists (so the ``not cache_dir_path.exists()`` build step is
    skipped)."""
    cache_dir = tmp_path / "cache"
    modes = [cache.CheckoutMode.SYMLINK]

    first_output = tmp_path / "first"
    first_output.mkdir()
    (first_output / "run.sh").write_text("#!/bin/sh\necho hi\n")
    (first_output / "run.sh").chmod(0o755)
    output_hash = cache.save_to_cache(first_output, cache_dir, checkout_modes=modes)
    assert isinstance(output_hash, dict) and "manifest" in output_hash

    tree_path = cache.get_cache_path(cache_dir, output_hash["hash"])
    for f in tree_path.rglob("*"):
        if f.is_file() and not f.is_symlink():
            os.chmod(f, 0o444)
    cache._tree_layout_marker(tree_path).unlink()

    second_output = tmp_path / "second"
    second_output.mkdir()
    (second_output / "run.sh").write_text("#!/bin/sh\necho hi\n")
    (second_output / "run.sh").chmod(0o755)
    cache.save_to_cache(second_output, cache_dir, checkout_modes=modes)

    assert os.access(second_output / "run.sh", os.X_OK), (
        "saving against an existing legacy tree must still produce an executable checkout"
    )


def test_repair_tree_dir_isexec_rejects_path_traversal(tmp_path: pathlib.Path) -> None:
    """_repair_tree_dir_isexec must not chmod outside the tree cache directory: a manifest
    entry can come straight from a lock file on the restore path, so a relpath pointing
    outside the tree marked isexec must raise instead of reaching through to a file outside
    the tree, exactly like the file-by-file restore branch already does for the same
    manifest shape."""
    from pivot import exceptions
    from pivot.types import DirManifestEntry

    cache_dir = tmp_path / "cache"
    modes = [cache.CheckoutMode.SYMLINK]

    output = tmp_path / "out"
    output.mkdir()
    (output / "run.sh").write_text("#!/bin/sh\necho hi\n")
    output_hash = cache.save_to_cache(output, cache_dir, checkout_modes=modes)
    assert isinstance(output_hash, dict) and "manifest" in output_hash

    victim = tmp_path / "victim.txt"
    victim.write_text("do not touch")
    victim.chmod(0o644)

    tree_path = cache.get_cache_path(cache_dir, output_hash["hash"])
    relpath = os.path.relpath(victim, tree_path)
    malicious_manifest = [
        DirManifestEntry(
            relpath=relpath, hash="a" * cache.XXHASH64_HEX_LENGTH, size=100, isexec=True
        )
    ]

    with pytest.raises(exceptions.SecurityValidationError, match="path traversal"):
        cache._repair_tree_dir_isexec(tree_path, malicious_manifest)

    assert victim.stat().st_mode & 0o777 == 0o644, "victim file must be untouched"


def test_restore_from_cache_symlink_fast_path_rejects_manifest_traversal(
    tmp_path: pathlib.Path,
) -> None:
    """The symlink fast path in `_restore_directory_from_cache` must reject a lock-supplied
    manifest with a traversal relpath before it checks or repairs a tree it did not mark. The
    manifest does not hash to the tree's name, so it is refused before any check or repair, and
    the file-by-file restore finds no object for the traversal entry: nothing outside the output
    is touched and the tree is left unmarked."""
    from pivot.types import DirHash, DirManifestEntry

    cache_dir = tmp_path / "cache"
    modes = [cache.CheckoutMode.SYMLINK]

    output = tmp_path / "out"
    output.mkdir()
    (output / "run.sh").write_text("#!/bin/sh\necho hi\n")
    output_hash = cache.save_to_cache(output, cache_dir, checkout_modes=modes)
    assert isinstance(output_hash, dict) and "manifest" in output_hash

    victim = tmp_path / "victim.txt"
    victim.write_text("do not touch")
    victim.chmod(0o644)

    tree_path = cache.get_cache_path(cache_dir, output_hash["hash"])
    cache._tree_layout_marker(tree_path).unlink()
    relpath = os.path.relpath(victim, tree_path)
    tampered_hash = DirHash(
        hash=output_hash["hash"],
        manifest=[
            *output_hash["manifest"],
            DirManifestEntry(
                relpath=relpath, hash="a" * cache.XXHASH64_HEX_LENGTH, size=100, isexec=True
            ),
        ],
    )

    cache.remove_output(output)
    assert not cache.restore_from_cache(output, tampered_hash, cache_dir, checkout_modes=modes)

    assert victim.stat().st_mode & 0o777 == 0o644, "victim file must be untouched"
    assert not cache._tree_layout_marker(tree_path).exists(), "a forged manifest marks nothing"


def test_directory_output_restore_isexec_mismatch_without_copy_fallback_raises(
    tmp_path: pathlib.Path,
) -> None:
    """A pure restore (no local bytes to repair the cache object from — the situation after
    `pivot pull`, or a tree an older pivot cached) with no COPY in checkout_modes has nowhere
    safe to fall back to when the cache object's mode disagrees with the manifest: checkout
    raises a clear error instead of silently producing the wrong executable bit."""
    from pivot import exceptions
    from pivot.types import DirHash, DirManifestEntry

    cache_dir = tmp_path / "cache"
    content = b"#!/bin/sh\necho hi\n"
    file_hash = "a1b2c3d4e5f6a7b8"
    cache_path = cache.get_cache_path(cache_dir, file_hash)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(content)
    cache_path.chmod(0o444)  # as a cold `pivot pull` would leave it: no isexec in scope

    output_hash = DirHash(
        hash="deadbeefdeadbeef",
        manifest=[
            DirManifestEntry(relpath="run.sh", hash=file_hash, size=len(content), isexec=True)
        ],
    )

    with pytest.raises(exceptions.CacheModeCollisionError, match="isexec"):
        cache.restore_from_cache(
            tmp_path / "out",
            output_hash,
            cache_dir,
            checkout_modes=[cache.CheckoutMode.HARDLINK],
        )


def test_single_file_output_unaffected_by_directory_entry_sharing_its_bytes(
    tmp_path: pathlib.Path,
) -> None:
    """A single-file output (``FileHash``, no ``isexec`` field) must not be forced through
    the directory exec-bit guard just because another file with identical bytes is tracked
    as executable in some directory manifest. A single file's mode is never part of any hash,
    so there is no wrong result for the guard to prevent: under a restrictive checkout_modes
    with no COPY, it must still hardlink rather than raise ``CacheModeCollisionError``."""
    cache_dir = tmp_path / "cache"
    content = "#!/bin/sh\necho hi\n"

    exec_dir = tmp_path / "exec_dir"
    exec_dir.mkdir()
    (exec_dir / "shared.sh").write_text(content)
    (exec_dir / "shared.sh").chmod(0o755)
    cache.save_to_cache(exec_dir, cache_dir, checkout_modes=[cache.CheckoutMode.HARDLINK])

    single_file = tmp_path / "shared.sh"
    single_file.write_text(content)
    cache.save_to_cache(single_file, cache_dir, checkout_modes=[cache.CheckoutMode.HARDLINK])

    assert single_file.stat().st_ino == (exec_dir / "shared.sh").stat().st_ino, (
        "a single-file output sharing bytes with an executable directory entry must still "
        "hardlink to the shared cache object, not fall back to a private copy"
    )


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


def test_copy_to_cache_keeps_a_non_executable_object_when_executable_requested(
    tmp_path: pathlib.Path,
) -> None:
    """An object cached non-executable (by an older pivot, by a cold `pivot pull`, or by a
    non-executable file with the same bytes) is kept when an executable copy of the same bytes
    is cached: a symlink checkout of that other file reads its mode through the object's name,
    and a hardlink checkout may be linking the object right then."""
    cache_path = tmp_path / "cache" / "ab" / "cdef"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"content")
    cache_path.chmod(0o444)
    inode = cache_path.stat().st_ino

    src = tmp_path / "source.txt"
    src.write_text("content")

    cache.copy_to_cache(src, cache_path, executable=True)

    assert cache_path.stat().st_ino == inode
    assert cache_path.stat().st_mode & 0o777 == 0o444
    assert _helper_entries(cache_path.parent) == [cache_path.name], "no temp file left"


def test_copy_to_cache_leaves_executable_object_alone_when_non_executable_requested(
    tmp_path: pathlib.Path,
) -> None:
    """An object already cached executable is never demoted by a later non-executable
    request."""
    cache_path = tmp_path / "cache" / "ab" / "cdef"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"content")
    cache_path.chmod(0o555)

    src = tmp_path / "source.txt"
    src.write_text("content")

    cache.copy_to_cache(src, cache_path, executable=False)

    assert cache_path.stat().st_mode & 0o777 == 0o555


def test_copy_to_cache_keeps_an_object_another_writer_published_first(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two writers caching the same new bytes both find no object, and both copy them. The
    one that finishes second leaves the first one's object in place instead of renaming its
    own copy over it: a checkout may be hardlinking that object right then, and link() fails
    with ENOENT once a rename has taken the object's last name."""
    src = tmp_path / "out.txt"
    src.write_text("same bytes")
    cache_path = tmp_path / "cache" / "ab" / "cdef0123456789"
    real_copy2 = shutil.copy2
    published = list[int]()

    def _copy2_while_another_writer_publishes(source: str, dest: str) -> object:
        if not published:
            published.append(-1)
            cache.copy_to_cache(src, cache_path)
            published[0] = cache_path.stat().st_ino
        return real_copy2(source, dest)

    monkeypatch.setattr(shutil, "copy2", _copy2_while_another_writer_publishes)

    cache.copy_to_cache(src, cache_path)

    assert cache_path.stat().st_ino == published[0]
    assert _helper_entries(cache_path.parent) == [cache_path.name], "no temp file left"


@pytest.mark.parametrize("link_errno", [errno.EPERM, errno.ENOTSUP, errno.ENOSYS])
@pytest.mark.parametrize(
    ("existing_mode", "executable", "expected_mode"),
    [
        pytest.param(None, False, 0o444, id="absent"),
        pytest.param(None, True, 0o555, id="absent-executable"),
        pytest.param(0o444, True, 0o444, id="present"),
    ],
)
def test_copy_to_cache_without_hard_links(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    link_errno: int,
    existing_mode: int | None,
    executable: bool,
    expected_mode: int,
) -> None:
    """On a filesystem that refuses link() (vfat, many SMB and FUSE mounts; a FUSE mount with
    no link handler returns ENOSYS before Linux 6.15), a new object is renamed into place and
    an existing one is kept."""
    src = tmp_path / "out.txt"
    src.write_text("content")
    cache_path = tmp_path / "cache" / "ab" / "cdef0123456789"
    inode = None
    if existing_mode is not None:
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text("content")
        cache_path.chmod(existing_mode)
        inode = cache_path.stat().st_ino

    def _refuse_link(source: object, dest: object) -> None:
        raise OSError(link_errno, os.strerror(link_errno))

    monkeypatch.setattr(os, "link", _refuse_link)

    cache.copy_to_cache(src, cache_path, executable=executable)

    assert cache_path.read_text() == "content"
    assert cache_path.stat().st_mode & 0o777 == expected_mode
    if inode is not None:
        assert cache_path.stat().st_ino == inode
    assert _helper_entries(cache_path.parent) == [cache_path.name], "no temp file left"


@pytest.mark.parametrize("link_errno", [errno.EIO, errno.ENOSPC])
def test_copy_to_cache_raises_a_failed_link(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, link_errno: int
) -> None:
    """A link() that fails for any reason but missing hard-link support is an error, and
    publishes nothing."""
    src = tmp_path / "out.txt"
    src.write_text("content")
    cache_path = tmp_path / "cache" / "ab" / "cdef0123456789"

    def _fail_link(source: object, dest: object) -> None:
        raise OSError(link_errno, os.strerror(link_errno))

    monkeypatch.setattr(os, "link", _fail_link)

    with pytest.raises(OSError) as raised:
        cache.copy_to_cache(src, cache_path)

    assert raised.value.errno == link_errno
    assert _helper_entries(cache_path.parent) == [], "nothing published, no temp file left"


@pytest.mark.parametrize("link_errno", [errno.ENOSYS, errno.ENOTSUP])
def test_hardlink_checkout_falls_through_a_link_the_filesystem_cannot_make(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, link_errno: int
) -> None:
    """A filesystem without hard-link support (ENOSYS, ENOTSUP) moves checkout on to the next
    mode, as EXDEV does."""
    cache_path = tmp_path / "cache" / "ab" / "cdef0123456789"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("content")
    target = tmp_path / "target.txt"

    def _refuse_link(source: object, dest: object) -> None:
        raise OSError(link_errno, os.strerror(link_errno))

    monkeypatch.setattr(os, "link", _refuse_link)

    cache._checkout_with_fallback(
        target, cache_path, [cache.CheckoutMode.HARDLINK, cache.CheckoutMode.COPY]
    )

    assert target.read_text() == "content"
    assert target.stat().st_ino != cache_path.stat().st_ino


def _helper_refuse_links_to(
    monkeypatch: pytest.MonkeyPatch, cache_path: pathlib.Path, link_errno: int
) -> None:
    """Make os.link refuse, with link_errno, every new link to the cache object at cache_path."""
    real_link = os.link

    def _link(source: str, dest: str, **kwargs: Any) -> None:
        if pathlib.Path(source) == cache_path:
            raise OSError(link_errno, os.strerror(link_errno))
        real_link(source, dest, **kwargs)

    monkeypatch.setattr(os, "link", _link)


_FULL_OBJECT_CONTENT = "#!/bin/sh\necho full\n"


def _helper_restore_with_one_link_refused(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    link_errno: int,
    modes: list[cache.CheckoutMode] | None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Cache a file output, or a directory output holding an executable full.sh and other.txt,
    remove it, then restore it with every link to full.sh's object refused with link_errno.

    Returns the restored full.sh entry and its cache object.
    """
    cache_dir = tmp_path / "cache"
    if kind == "file":
        output = full = tmp_path / "full.sh"
        full.write_text(_FULL_OBJECT_CONTENT)
    else:
        output = tmp_path / "out"
        output.mkdir()
        full = output / "full.sh"
        full.write_text(_FULL_OBJECT_CONTENT)
        full.chmod(0o755)
        (output / "other.txt").write_text("other\n")
    full_hash, _ = cache.hash_file(full)
    output_hash = cache.save_to_cache(output, cache_dir, checkout_mode=cache.CheckoutMode.COPY)
    cache.remove_output(output)
    full_object = cache.get_cache_path(cache_dir, full_hash)
    _helper_refuse_links_to(monkeypatch, full_object, link_errno)

    assert cache.restore_from_cache(
        output, output_hash, cache_dir, checkout_modes=modes, state_dir=tmp_path / ".pivot"
    )

    if kind == "directory":
        other = output / "other.txt"
        other_object = cache.get_cache_path(cache_dir, cache.hash_file(other)[0])
        assert other.stat().st_ino == other_object.stat().st_ino, "other entries still hardlink"
    return full, full_object


@pytest.mark.parametrize(
    "modes",
    [
        pytest.param(None, id="default-order"),
        pytest.param([cache.CheckoutMode.HARDLINK], id="hardlink-only"),
        pytest.param([cache.CheckoutMode.HARDLINK, cache.CheckoutMode.COPY], id="hardlink-copy"),
    ],
)
@pytest.mark.parametrize("kind", ["file", "directory"])
def test_restore_copies_a_cache_object_at_the_hard_link_limit(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    modes: list[cache.CheckoutMode] | None,
) -> None:
    """A hard link refused with EMLINK means that one object already has as many links as the
    filesystem allows (65,000 on ext4). That entry is checked out as a copy: a regular file
    with the object's bytes and its mode, never a symlink into the cache, whether symlink,
    copy or nothing comes next in the order."""
    full, full_object = _helper_restore_with_one_link_refused(
        tmp_path, monkeypatch, kind, errno.EMLINK, modes
    )

    entry = os.lstat(full)
    assert stat.S_ISREG(entry.st_mode), "an object at the link limit is copied, not symlinked"
    assert entry.st_ino != full_object.stat().st_ino
    assert full.read_text() == _FULL_OBJECT_CONTENT
    # A directory entry keeps its recorded executable bit; a single file's mode is not hashed.
    assert stat.S_IMODE(entry.st_mode) == (0o755 if kind == "directory" else 0o644)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_restore_symlinks_when_a_hard_link_crosses_devices(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """EXDEV says the workspace cannot hardlink into the cache at all, so the default order
    still moves on to symlink."""
    full, full_object = _helper_restore_with_one_link_refused(
        tmp_path, monkeypatch, kind, errno.EXDEV, None
    )

    assert full.is_symlink()
    assert full.resolve() == full_object.resolve()


@pytest.mark.parametrize("operation", ["restore", "save"])
def test_copy_at_the_hard_link_limit_is_staged_under_the_targets_restore_name(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """The copy made at the link limit is built under the target's deterministic restore name,
    which the next restore or save of the target clears if this one is killed mid-copy, and
    then renamed over the target."""
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / ".pivot"
    target = tmp_path / "data" / "out.txt"
    target.parent.mkdir(parents=True)
    target.write_text("content\n")
    output_hash = cache.save_to_cache(target, cache_dir, checkout_mode=cache.CheckoutMode.COPY)
    if operation == "restore":
        target.unlink()
    _helper_refuse_links_to(
        monkeypatch, cache.get_cache_path(cache_dir, output_hash["hash"]), errno.EMLINK
    )
    copied = list[pathlib.Path]()
    renamed = list[tuple[pathlib.Path, pathlib.Path]]()
    real_copy2 = shutil.copy2
    real_replace = pathlib.Path.replace

    def _copy2(source: str, dest: str, **kwargs: Any) -> object:
        copied.append(pathlib.Path(dest))
        return real_copy2(source, dest, **kwargs)

    def _replace(self: pathlib.Path, dest: pathlib.Path) -> pathlib.Path:
        renamed.append((self, pathlib.Path(dest)))
        return real_replace(self, dest)

    monkeypatch.setattr(shutil, "copy2", _copy2)
    monkeypatch.setattr(pathlib.Path, "replace", _replace)

    if operation == "restore":
        assert cache.restore_from_cache(target, output_hash, cache_dir, state_dir=state_dir)
    else:
        cache.save_to_cache(target, cache_dir, state_dir=state_dir)

    staging = cache.restore_staging_path(target)
    assert [p for p in copied if p.parent == target.parent] == [staging]
    assert renamed == [(staging, target)]
    assert stat.S_ISREG(os.lstat(target).st_mode)
    assert target.read_text() == "content\n"
    assert not staging.exists()


def test_commit_keeps_a_written_output_whose_cache_object_is_at_the_hard_link_limit(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage output whose cache object is at the hard-link limit stays the regular file the
    stage wrote, with the default order, instead of becoming a symlink into the cache."""
    cache_dir = tmp_path / "cache"
    output = tmp_path / "out.txt"
    output.write_text("")
    output_hash, written = cache.hash_file(output)
    _helper_refuse_links_to(monkeypatch, cache.get_cache_path(cache_dir, output_hash), errno.EMLINK)

    cache.store_hashed_output(
        output, FileHash(hash=output_hash), cache_dir, state_dir=tmp_path / ".pivot"
    )

    kept = os.lstat(output)
    assert stat.S_ISREG(kept.st_mode)
    assert kept.st_ino == written.st_ino, "the file the stage wrote is kept, not replaced"
    assert output.read_text() == ""


def test_symlinked_entry_keeps_its_mode_when_the_same_bytes_are_cached_executable(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the cache on another device, the default checkout order symlinks each directory
    entry, and the entry reads its mode through the cache object's name. A later directory
    output holding the same bytes as an executable leaves that object alone: the earlier
    entry stays non-executable and its directory keeps its hash, and the executable gets a
    copy of its own."""
    cache_dir = tmp_path / "cache"
    real_link = os.link

    def _cross_device_link(source: str, dest: str, **kwargs: Any) -> None:
        if not pathlib.Path(dest).is_relative_to(cache_dir):
            raise OSError(errno.EXDEV, os.strerror(errno.EXDEV))
        real_link(source, dest, **kwargs)

    monkeypatch.setattr(os, "link", _cross_device_link)
    content = "#!/bin/sh\necho hi\n"

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "run.sh").write_text(content)
    plain_hash = cache.save_to_cache(plain, cache_dir)
    assert (plain / "run.sh").is_symlink()

    tool = tmp_path / "tool"
    tool.mkdir()
    (tool / "run.sh").write_text(content)
    (tool / "run.sh").chmod(0o755)
    cache.save_to_cache(tool, cache_dir)

    assert not os.access(plain / "run.sh", os.X_OK)
    assert cache.hash_directory(plain, cache_dir=cache_dir)["hash"] == plain_hash["hash"]
    assert os.access(tool / "run.sh", os.X_OK)
    assert not (tool / "run.sh").is_symlink()


def _helper_same_tree_twice(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    first, second = tmp_path / "first", tmp_path / "second"
    for out in (first, second):
        (out / "sub").mkdir(parents=True)
        (out / "sub" / "f.txt").write_text("same")
        (out / "run.sh").write_text("#!/bin/sh\n")
        (out / "run.sh").chmod(0o755)
    return first, second


def test_symlink_tree_keeps_a_tree_another_writer_published_first(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two outputs with the same directory content, saved in SYMLINK mode at once, both find
    no cache tree and both copy one. The one that finishes second neither fails on nor
    touches the tree the first published."""
    cache_dir = tmp_path / "cache"
    first, second = _helper_same_tree_twice(tmp_path)
    tree = cache.get_cache_path(cache_dir, cache.hash_directory(second)["hash"])
    real_copytree = shutil.copytree
    published = list[int]()

    def _copytree_while_another_writer_publishes(
        source: str, dest: str, *args: Any, **kwargs: Any
    ) -> object:
        if not published:
            published.append(-1)
            cache.save_to_cache(first, cache_dir, checkout_mode=cache.CheckoutMode.SYMLINK)
            published[0] = tree.stat().st_ino
        return real_copytree(source, dest, *args, **kwargs)

    monkeypatch.setattr(shutil, "copytree", _copytree_while_another_writer_publishes)

    cache.save_to_cache(second, cache_dir, checkout_mode=cache.CheckoutMode.SYMLINK)

    assert tree.stat().st_ino == published[0]
    assert first.resolve() == second.resolve() == tree.resolve()
    assert (second / "sub" / "f.txt").read_text() == "same"
    assert os.access(second / "run.sh", os.X_OK)
    assert _helper_entries(tree.parent) == sorted(
        [tree.name, cache._tree_layout_marker(tree).name]
    ), "no temporary tree left"


def test_symlink_tree_appears_only_once_complete(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restore that finds a tree's cache name symlinks to it at once, so the name exists
    only once every file is in place with its final mode."""
    cache_dir = tmp_path / "cache"
    out, _ = _helper_same_tree_twice(tmp_path)
    tree = cache.get_cache_path(cache_dir, cache.hash_directory(out)["hash"])
    real_copytree = shutil.copytree
    tree_named_after_copy = list[bool]()

    def _copytree_then_look(source: str, dest: str, *args: Any, **kwargs: Any) -> object:
        result = real_copytree(source, dest, *args, **kwargs)
        if pathlib.Path(source) == out:  # not copytree's own call for a subdirectory
            tree_named_after_copy.append(tree.exists())
        return result

    monkeypatch.setattr(shutil, "copytree", _copytree_then_look)

    cache.save_to_cache(out, cache_dir, checkout_mode=cache.CheckoutMode.SYMLINK)

    assert tree_named_after_copy == [False]
    assert out.resolve() == tree.resolve()
    assert (tree / "run.sh").stat().st_mode & 0o777 == 0o555
    assert (tree / "sub" / "f.txt").stat().st_mode & 0o777 == 0o444


def _helper_crash_during_tree_copy(out: pathlib.Path, cache_dir: pathlib.Path) -> None:
    """Simulate a kill while copying a directory output into the cache as a SYMLINK tree."""
    real_copytree = shutil.copytree

    def _copy_and_crash(source: str, dest: str, *args: Any, **kwargs: Any) -> object:
        real_copytree(source, dest, *args, **kwargs)
        os._exit(73)

    shutil.copytree = _copy_and_crash
    _ = cache.save_to_cache(out, cache_dir, checkout_mode=cache.CheckoutMode.SYMLINK)


def test_symlink_tree_publish_removes_a_killed_publishers_copy(tmp_path: pathlib.Path) -> None:
    """A process killed while copying a tree into the cache leaves its copy beside the tree's
    cache name, never under it; the next publish into that directory removes the copy."""
    cache_dir = tmp_path / "cache"
    out, _ = _helper_same_tree_twice(tmp_path)
    tree = cache.get_cache_path(cache_dir, cache.hash_directory(out)["hash"])
    process = multiprocessing.get_context("spawn").Process(
        target=_helper_crash_during_tree_copy, args=(out, cache_dir)
    )
    process.start()
    process.join(timeout=60)

    assert process.exitcode == 73
    assert not tree.exists()
    left = _helper_entries(tree.parent)
    assert len(left) == 1 and left[0].startswith(cache._CACHE_TEMP_PREFIX), "the killed copy"
    # The sweep of one prefix directory runs at most once per _CACHE_SWEEP_INTERVAL.
    os.utime(tree.parent / cache._PREFIX_LOCK, (0, 0))

    cache.save_to_cache(out, cache_dir, checkout_mode=cache.CheckoutMode.SYMLINK)

    assert out.resolve() == tree.resolve()
    assert _helper_entries(tree.parent) == sorted([tree.name, cache._tree_layout_marker(tree).name])


def _helper_publish_tree_unseen(
    src: pathlib.Path,
    tree: pathlib.Path,
    output_hash: DirHash,
    cache_dir: pathlib.Path,
    unseen_pid: int,
    copied: Event,
    release: Event,
) -> None:
    """Publish ``tree`` as a process the caller cannot see: one whose PID, from the caller's PID
    namespace or host, belongs to no running process. It pauses once its copy is complete,
    before renaming it into place, until ``release`` is set."""
    os.getpid = lambda: unseen_pid
    real_copytree = shutil.copytree

    def _copy_then_pause(source: str, dest: str, *args: Any, **kwargs: Any) -> object:
        result = real_copytree(source, dest, *args, **kwargs)
        if pathlib.Path(source) == src:
            copied.set()
            assert release.wait(60)
        return result

    shutil.copytree = _copy_then_pause
    cache._publish_tree(src, tree, output_hash, cache_dir)


def test_symlink_tree_publish_keeps_a_live_publishers_copy_it_cannot_see(
    tmp_path: pathlib.Path,
) -> None:
    """A publisher in another PID namespace (a bwrap or `unshare --pid` sandbox), or on another
    host sharing the cache, has a PID this process sees as dead. Another tree published into the
    same prefix directory meanwhile leaves its copy alone, and its tree is published whole."""
    cache_dir = tmp_path / "cache"
    prefix = cache_dir / "ab"
    prefix.mkdir(parents=True)
    src = tmp_path / "theirs"
    for i in range(200):
        (src / f"d{i % 10}").mkdir(parents=True, exist_ok=True)
        (src / f"d{i % 10}" / f"f{i}.txt").write_text(f"{i}\n")
    theirs = cache.hash_directory(src)
    ours_src, _ = _helper_same_tree_twice(tmp_path)
    ours = cache.hash_directory(ours_src)
    ctx = multiprocessing.get_context("spawn")
    exited = ctx.Process(target=int)
    exited.start()
    exited.join(timeout=60)
    with ctx.Manager() as manager:
        copied, release = manager.Event(), manager.Event()
        publisher = ctx.Process(
            target=_helper_publish_tree_unseen,
            args=(src, prefix / "cdef0123456789", theirs, cache_dir, exited.pid)
            + (copied, release),
        )
        publisher.start()
        assert copied.wait(60)
        cache._publish_tree(ours_src, prefix / "99990123456789", ours, cache_dir)
        release.set()
        publisher.join(timeout=60)

    assert publisher.exitcode == 0
    assert cache.hash_directory(prefix / "cdef0123456789") == theirs
    assert cache.hash_directory(prefix / "99990123456789") == ours
    assert not [p for p in prefix.iterdir() if p.name.startswith(cache._CACHE_TEMP_PREFIX)]


def _helper_tree_with_files_missing(
    tmp_path: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, DirHash]:
    """A SYMLINK cache tree as an older pivot left it when killed while copying it straight to its
    cache name: one file missing, one cut short, no layout marker. Its file objects are cached."""
    cache_dir = tmp_path / "cache"
    out = tmp_path / "out"
    out.mkdir()
    for name in ("a.txt", "b.txt", "c.txt"):
        (out / name).write_text(f"{name} " * 100)
    output_hash = cache.save_to_cache(out, cache_dir, checkout_mode=cache.CheckoutMode.COPY)
    assert is_dir_hash(output_hash)
    tree = cache.get_cache_path(cache_dir, output_hash["hash"])
    tree.mkdir(parents=True)
    (tree / "a.txt").write_text("a.txt " * 100)
    (tree / "b.txt").write_text("b.txt ")
    shutil.rmtree(out)
    return cache_dir, tree, output_hash


def test_symlink_restore_repairs_a_tree_left_with_files_missing(tmp_path: pathlib.Path) -> None:
    """A tree whose name exists but which lacks files is not trusted: the restore repairs it from
    the file objects, so the output hashes to the hash it was restored for."""
    cache_dir, tree, output_hash = _helper_tree_with_files_missing(tmp_path)
    out = tmp_path / "out"

    assert cache.restore_from_cache(
        out, output_hash, cache_dir, checkout_modes=[cache.CheckoutMode.SYMLINK]
    )

    assert out.resolve() == tree.resolve()
    assert cache.hash_directory(out, cache_dir=cache_dir)["hash"] == output_hash["hash"]
    assert (tree / "c.txt").stat().st_mode & 0o777 == 0o444
    assert cache._tree_layout_marker(tree).exists()


def test_symlink_restore_refuses_a_tree_it_cannot_repair(tmp_path: pathlib.Path) -> None:
    """With a missing file's object gone too, the tree can restore nothing; the restore reports a
    cache miss instead of symlinking to it."""
    cache_dir, tree, output_hash = _helper_tree_with_files_missing(tmp_path)
    c_hash = next(e["hash"] for e in output_hash["manifest"] if e["relpath"] == "c.txt")
    cache.get_cache_path(cache_dir, c_hash).unlink()
    out = tmp_path / "out"

    assert not cache.restore_from_cache(
        out, output_hash, cache_dir, checkout_modes=[cache.CheckoutMode.SYMLINK]
    )

    assert not out.exists()
    assert not cache._tree_layout_marker(tree).exists()


@pytest.mark.parametrize(
    "old_suffix",
    [
        pytest.param(".layout-checked", id="pivot-0.3.4-layout-checked"),
        pytest.param(".verified", id="pivot-0.5.11-size-checked"),
    ],
)
def test_symlink_restore_rechecks_a_partial_tree_an_older_pivot_marked(
    tmp_path: pathlib.Path, old_suffix: str
) -> None:
    """Pivot 0.3.4 marked a tree after checking only its layout, so its mark sits beside partial
    trees too; pivot 0.3.5 to 0.5.11 marked one after checking only sizes. The restore checks such
    a tree again, repairs it, and replaces the old mark."""
    cache_dir, tree, output_hash = _helper_tree_with_files_missing(tmp_path)
    old_mark = tree.with_name(tree.name + old_suffix)
    old_mark.touch()
    out = tmp_path / "out"

    assert cache.restore_from_cache(
        out, output_hash, cache_dir, checkout_modes=[cache.CheckoutMode.SYMLINK]
    )

    assert cache.hash_directory(out, cache_dir=cache_dir)["hash"] == output_hash["hash"]
    assert cache._tree_layout_marker(tree).exists()
    assert not old_mark.exists()


def test_symlink_restore_repairs_stale_bytes_in_a_complete_tree_pivot_0_5_11_marked(
    tmp_path: pathlib.Path,
) -> None:
    """Pivot 0.3.5 to 0.5.11 marked a tree `.verified` after checking its files' sizes only, so a
    tree holding every file at its recorded size, one with stale bytes, can carry that mark. The
    restore checks it by content, repairs the stale file, links the tree and marks it anew."""
    cache_dir, tree, output_hash = _helper_tree_with_files_missing(tmp_path)
    (tree / "b.txt").write_text("b.txt " * 100)
    (tree / "c.txt").write_text("C.TXT " * 100)  # stale bytes of the recorded size
    old_mark = tree.with_name(tree.name + ".verified")
    old_mark.touch()
    out = tmp_path / "out"

    assert cache.restore_from_cache(
        out, output_hash, cache_dir, checkout_modes=[cache.CheckoutMode.SYMLINK]
    )

    assert out.resolve() == tree.resolve()
    assert (out / "c.txt").read_text() == "c.txt " * 100
    assert cache._tree_layout_marker(tree).exists()
    assert not old_mark.exists()


def test_symlink_restore_records_repaired_bytes_not_stale_same_size_bytes(
    tmp_path: pathlib.Path,
) -> None:
    """Two-guarantee probe (Dispatch BET-237 records every output hash; #522 content-checks a
    tree): restoring through an unmarked SYMLINK cache tree that holds every file at its recorded
    size, one with stale bytes, content-checks and repairs the tree before it records anything.
    So every hash it records vouches for the bytes its file now holds, never the stale bytes a
    size-only check would have waved through and recorded as the output's hash."""
    cache_dir, tree, output_hash = _helper_tree_with_files_missing(tmp_path)
    (tree / "b.txt").write_text("b.txt " * 100)
    (tree / "c.txt").write_text("C.TXT " * 100)  # stale bytes of the recorded size, tree unmarked
    assert not cache._tree_layout_marker(tree).exists()
    out = tmp_path / "out"
    recorded = list[tuple[pathlib.Path, os.stat_result, str]]()

    assert cache.restore_from_cache(
        out,
        output_hash,
        cache_dir,
        checkout_modes=[cache.CheckoutMode.SYMLINK],
        recorded=recorded,
    )

    assert (out / "c.txt").read_text() == "c.txt " * 100
    assert cache.hash_directory(out, cache_dir=cache_dir)["hash"] == output_hash["hash"]
    recorded_names = {path.name for path, _, _ in recorded}
    assert recorded_names == {"a.txt", "b.txt", "c.txt"}, recorded_names
    with state.StateDB(tmp_path / "state") as state_db:
        state_db.save_many(recorded)
    with state.StateDB(tmp_path / "state", readonly=True) as state_db:
        for name in ("a.txt", "b.txt", "c.txt"):
            member = out / name
            assert state_db.get(member, member.stat()) == cache.hash_file(member)[0], (
                f"{name}'s recorded hash vouches for the bytes it now holds, not stale bytes"
            )


_FORGED_HASH = "0123456789abcdef"


def _helper_marked_tree(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, DirHash]:
    """A SYMLINK cache tree this pivot built and marked, holding ``a.txt`` = ``AAAA``."""
    cache_dir = tmp_path / "cache"
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_bytes(b"AAAA")
    output_hash = cache.save_to_cache(src, cache_dir, checkout_modes=[cache.CheckoutMode.SYMLINK])
    assert is_dir_hash(output_hash)
    tree = cache.get_cache_path(cache_dir, output_hash["hash"])
    assert cache._tree_layout_marker(tree).exists(), "the tree is marked"
    return cache_dir, tree, output_hash


def _helper_digest(output_hash: DirHash) -> str:
    """The digest a tree's marker records for ``output_hash``'s manifest."""
    canonical = cache._canonical_manifest(output_hash)
    assert canonical is not None
    return cache._manifest_digest(canonical)


def _helper_forge(
    honest: DirHash, entries: list[dict[str, object]], forgery: str, cache_dir: pathlib.Path
) -> DirHash:
    """``entries`` made to hash to ``honest``'s tree name (XXH64 is not second-preimage
    resistant): ``padded`` adds a key pivot never writes to the last entry (the round-5
    reviewer's forgery); ``canonical`` adds a last entry holding only pivot's four fields, its
    relpath steering the hash, its content a cached object, so even a comparison of only those
    fields sees a manifest that names the tree."""
    if forgery == "padded":
        manifest = xxh64_forge.pad_to(entries, honest["hash"])
    else:
        filler = cache_dir.parent / "filler.bin"
        filler.write_bytes(b"F")
        filler_hash = cache.hash_file(filler)[0]
        cache.copy_to_cache(filler, cache.get_cache_path(cache_dir, filler_hash))
        extra: dict[str, object] = {"hash": filler_hash, "size": 1, "isexec": False}
        manifest = xxh64_forge.steer_with_relpath(entries, extra, honest["hash"])
    return cast("DirHash", cast("object", {"hash": honest["hash"], "manifest": manifest}))


@pytest.mark.parametrize("forgery", ["padded", "canonical"])
@pytest.mark.parametrize("target", ["hash-inside-the-tree", "absolute-path"])
def test_symlink_restore_records_no_hash_a_forged_manifest_chose(
    tmp_path: pathlib.Path, target: str, forgery: str
) -> None:
    """B1: a forged .pvt or lock manifest under a marked tree's name made the symlink fast path
    record the hashes it supplied, for a file inside the output or, by absolute path, anywhere.
    The forgery is made to hash to the tree's name, so the name decides nothing: the tree is
    trusted only for the manifest it was checked to hold, which its marker records by SHA-256.

    The trust decision is asserted first. The ``canonical`` forgery steers with an extra entry
    naming a file the tree lacks; at 0664ea83 the fast path's stat of that file failed and the
    restore fell back to file by file, so the restore alone records nothing there even though
    the tree was trusted. Neither forgery is trusted, takes the fast path or is vouched for, and
    the tree is left as it was."""
    cache_dir, tree, honest = _helper_marked_tree(tmp_path)
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"victim bytes")
    entries: list[dict[str, object]] = [dict(entry) for entry in honest["manifest"]]
    if target == "hash-inside-the-tree":
        entries[0]["hash"] = _FORGED_HASH
    else:
        entries.append({"relpath": str(victim), "hash": _FORGED_HASH, "size": 12, "isexec": False})
    forged = _helper_forge(honest, entries, forgery, cache_dir)
    out = tmp_path / "out"
    recorded = list[tuple[pathlib.Path, os.stat_result, str]]()

    assert not cache._ensure_tree_complete(tree, forged, cache_dir), "the forgery is not trusted"

    _ = cache.restore_from_cache(
        out, forged, cache_dir, checkout_modes=[cache.CheckoutMode.SYMLINK], recorded=recorded
    )

    assert [str(path) for path, _, file_hash in recorded if file_hash == _FORGED_HASH] == []
    assert [str(path) for path, _, _ in recorded if not path.is_relative_to(out)] == []
    assert not out.is_symlink(), "the forged manifest did not take the fast path"
    assert (tree / "a.txt").read_bytes() == b"AAAA", "the tree is untouched"
    assert cache._read_marker(tree) == _helper_digest(honest), "still marked for its manifest"


@pytest.mark.parametrize("forgery", ["padded", "canonical"])
def test_a_forged_manifest_cannot_make_an_honest_restore_serve_its_bytes(
    tmp_path: pathlib.Path, forgery: str
) -> None:
    """Defect on main (7fbdfa31, #522's tree check), closed here: a forged manifest under an
    unmarked honest tree's name made `_ensure_tree_complete` repair the tree to the forger's
    bytes and mark it, so every later honest restore of that hash served them. A padded forgery
    is refused before the tree is touched. One steered with only pivot's fields repairs the tree
    to the bytes it names and marks it for that manifest alone, so an honest restore does not
    trust the tree and restores the honest bytes file by file."""
    cache_dir, tree, honest = _helper_marked_tree(tmp_path)
    cache._tree_layout_marker(tree).unlink()
    other = tmp_path / "other.txt"
    other.write_bytes(b"BBBB")
    other_hash = cache.hash_file(other)[0]
    cache.copy_to_cache(other, cache.get_cache_path(cache_dir, other_hash))
    entries: list[dict[str, object]] = [
        {"relpath": "a.txt", "hash": other_hash, "size": 4, "isexec": False}
    ]
    forged = _helper_forge(honest, entries, forgery, cache_dir)
    _ = cache.restore_from_cache(
        tmp_path / "evil", forged, cache_dir, checkout_modes=[cache.CheckoutMode.SYMLINK]
    )

    legit = tmp_path / "legit"
    assert cache.restore_from_cache(
        legit, honest, cache_dir, checkout_modes=[cache.CheckoutMode.SYMLINK]
    )

    assert (legit / "a.txt").read_bytes() == b"AAAA"
    assert cache.hash_directory(legit, cache_dir=cache_dir)["hash"] == honest["hash"]
    if forgery == "padded":
        assert (tree / "a.txt").read_bytes() == b"AAAA", "refused before the tree was touched"


@pytest.mark.parametrize("field", ["unread", "size"])
def test_a_manifest_value_json_cannot_encode_still_restores(
    tmp_path: pathlib.Path, field: str
) -> None:
    """B2 (round 6): a hand-written .pvt or lock can hold a value JSON cannot encode (YAML reads
    `2026-01-01` as a date); round 5 encoded the whole manifest and failed the restore. A field
    pivot does not write is left out of the comparison, so the tree is trusted; a field pivot
    writes that cannot be encoded leaves no tree trusted, and the restore goes file by file."""
    cache_dir, _, honest = _helper_marked_tree(tmp_path)
    entry: dict[str, object] = dict(honest["manifest"][0])
    entry["when" if field == "unread" else "size"] = datetime.date(2026, 1, 1)
    out = tmp_path / "out"
    recorded = list[tuple[pathlib.Path, os.stat_result, str]]()

    assert cache.restore_from_cache(
        out,
        cast("DirHash", cast("object", {"hash": honest["hash"], "manifest": [entry]})),
        cache_dir,
        checkout_modes=[cache.CheckoutMode.SYMLINK],
        recorded=recorded,
    )

    assert (out / "a.txt").read_bytes() == b"AAAA"
    assert [(path, h) for path, _, h in recorded] == [
        (out / "a.txt", honest["manifest"][0]["hash"])
    ]
    assert out.is_symlink() == (field == "unread")


def test_a_repair_whose_result_differs_from_the_manifest_is_not_marked(
    tmp_path: pathlib.Path,
) -> None:
    """After repairing an unmarked tree, its manifest is recomputed and must equal the caller's,
    entry for entry, before the tree is marked. A cache object holding other bytes than its name
    (a corrupt object) is copied in by the repair; the recomputed manifest then differs, and the
    tree is left unmarked rather than vouched for."""
    cache_dir, tree, output_hash = _helper_tree_with_files_missing(tmp_path)
    c_hash = next(e["hash"] for e in output_hash["manifest"] if e["relpath"] == "c.txt")
    c_object = cache.get_cache_path(cache_dir, c_hash)
    c_object.chmod(0o644)
    c_object.write_text("not the bytes its name says")

    assert not cache._ensure_tree_complete(tree, output_hash, cache_dir)

    assert not cache._tree_layout_marker(tree).exists()


@pytest.mark.parametrize("stale", [False, True], ids=["matching", "stale"])
def test_a_tree_an_older_pivot_marked_is_checked_not_repaired(
    tmp_path: pathlib.Path, stale: bool
) -> None:
    """An older pivot marked a checked tree with an empty `.content-checked`. Such a tree is
    checked against the manifest: re-marked for it when it matches; when it differs, not trusted
    and not repaired (an output may symlink to it), so the restore goes file by file."""
    cache_dir, tree, honest = _helper_marked_tree(tmp_path)
    marker = cache._tree_layout_marker(tree)
    marker.unlink()
    marker.touch()
    if stale:
        (tree / "a.txt").chmod(0o644)
        (tree / "a.txt").write_bytes(b"ZZZZ")
    out = tmp_path / "out"

    assert cache.restore_from_cache(
        out, honest, cache_dir, checkout_modes=[cache.CheckoutMode.SYMLINK]
    )

    assert (out / "a.txt").read_bytes() == b"AAAA"
    assert out.is_symlink() == (not stale)
    assert (tree / "a.txt").read_bytes() == (b"ZZZZ" if stale else b"AAAA"), "not repaired"
    assert cache._read_marker(tree) == ("" if stale else _helper_digest(honest))


@pytest.mark.parametrize("guard", ["check", "repair"])
def test_tree_check_and_repair_refuse_a_manifest_path_leading_outside_the_tree(
    tmp_path: pathlib.Path, guard: str
) -> None:
    """The tree check and the repair resolve each recorded path inside the tree and raise on one
    leading out of it, since a manifest may come straight from a .pvt or lock. A forged manifest
    must first be the one a tree was checked to hold to reach them, so they are checked directly."""
    cache_dir, tree, honest = _helper_marked_tree(tmp_path)
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"victim")
    outside = DirManifestEntry(
        relpath=os.path.relpath(victim, tree), hash=_FORGED_HASH, size=6, isexec=False
    )
    traversal = DirHash(hash=honest["hash"], manifest=[*honest["manifest"], outside])

    with pytest.raises(exceptions.SecurityValidationError, match="path traversal"):
        if guard == "check":
            cache._tree_matches(tree, traversal, cache._scan_entries(tree))
        else:
            cache._repair_tree_dir_files(tree, traversal["manifest"], cache_dir)

    assert victim.read_bytes() == b"victim"


@pytest.mark.parametrize("what", ["file", "directory"])
def test_hashing_with_a_writable_state_database_never_fails_on_its_save(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, what: str
) -> None:
    """`hash_file` and `hash_directory` save what they read to a writable StateDB at once; that
    save is best-effort, so a write that fails (here a full map) leaves the hash to return."""
    target = tmp_path / "target"
    if what == "file":
        target.write_bytes(b"content")
    else:
        target.mkdir()
        (target / "a.txt").write_bytes(b"content")
    expected = (
        cache.hash_file(target)[0] if what == "file" else cache.hash_directory(target)["hash"]
    )

    def full_save_many(*_: object) -> None:
        raise state.DatabaseFullError("state database is full")

    monkeypatch.setattr(state.StateDB, "save_many", full_save_many)
    with state.StateDB(tmp_path / "state") as state_db:
        found = (
            cache.hash_file(target, state_db)[0]
            if what == "file"
            else cache.hash_directory(target, state_db)["hash"]
        )

    assert found == expected


def test_hardlink_no_op_check_compares_the_device_too(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 5, B2: a workspace file holding other bytes but whose inode number equals the cache
    object's, on another device, is a different file. The HARDLINK no-op check compared the inode
    alone, kept the stale bytes and recorded the cached hash for them; it is replaced."""
    content = b"cached bytes"
    cache_dir = tmp_path / "cache"
    src = tmp_path / "src.bin"
    src.write_bytes(content)
    file_hash = cache.hash_file(src)[0]
    cache_object = cache.get_cache_path(cache_dir, file_hash)
    cache.copy_to_cache(src, cache_object)
    out = tmp_path / "out.csv"
    out.write_bytes(b"stale bytes!")
    object_inode = cache_object.stat().st_ino
    real_stat = pathlib.Path.stat

    def same_inode_on_another_device(
        self: pathlib.Path, *, follow_symlinks: bool = True
    ) -> os.stat_result:
        found = real_stat(self, follow_symlinks=follow_symlinks)
        if self != out:
            return found
        fields = list(found)
        fields[stat.ST_INO] = object_inode
        fields[stat.ST_DEV] = found.st_dev + 1
        return os.stat_result(fields)

    monkeypatch.setattr(pathlib.Path, "stat", same_inode_on_another_device)
    recorded = list[tuple[pathlib.Path, os.stat_result, str]]()
    assert cache.restore_from_cache(
        out,
        FileHash(hash=file_hash),
        cache_dir,
        checkout_modes=[cache.CheckoutMode.HARDLINK, cache.CheckoutMode.COPY],
        recorded=recorded,
    )
    monkeypatch.undo()

    assert out.read_bytes() == content, "the stale file was replaced"
    assert [(path, recorded_hash) for path, _, recorded_hash in recorded] == [(out, file_hash)]


def test_symlink_restore_repairs_a_tree_with_a_file_cut_short(tmp_path: pathlib.Path) -> None:
    """A tree holding every file it records, one of them cut short, is not trusted either."""
    cache_dir, tree, output_hash = _helper_tree_with_files_missing(tmp_path)
    (tree / "c.txt").write_text("c.txt " * 100)
    out = tmp_path / "out"

    assert cache.restore_from_cache(
        out, output_hash, cache_dir, checkout_modes=[cache.CheckoutMode.SYMLINK]
    )

    assert out.resolve() == tree.resolve()
    assert cache.hash_directory(out, cache_dir=cache_dir)["hash"] == output_hash["hash"]


def test_symlink_restore_goes_file_by_file_past_a_damaged_tree_it_cannot_lock(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A damaged tree this user can neither lock nor so repair (another user's, in a shared
    cache) is not used: the output is restored file by file from the cached objects, and the
    tree is left unmarked."""
    cache_dir, tree, output_hash = _helper_tree_with_files_missing(tmp_path)
    real_open = cache._open_cache_lock

    def _tree_lock_not_writable(path: pathlib.Path) -> int:
        if path.name.endswith(cache._TREE_LOCK_SUFFIX) and path.name != cache._PREFIX_LOCK:
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path))
        return real_open(path)

    monkeypatch.setattr(cache, "_open_cache_lock", _tree_lock_not_writable)
    out = tmp_path / "out"

    assert cache.restore_from_cache(
        out, output_hash, cache_dir, checkout_modes=[cache.CheckoutMode.SYMLINK]
    )

    assert not out.is_symlink()
    assert cache.hash_directory(out, cache_dir=cache_dir)["hash"] == output_hash["hash"]
    assert not (tree / "c.txt").exists()
    assert not cache._tree_layout_marker(tree).exists()


def test_symlink_commit_repairs_a_tree_that_lost_a_file_while_it_was_copied(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tree copy that loses a file before it is renamed into place (a sweep on another host
    whose flock is not coherent with this one's) is not marked complete; the commit then checks
    the tree and repairs it from the file objects before it symlinks the output to it."""
    out, _ = _helper_same_tree_twice(tmp_path)
    cache_dir = tmp_path / "cache"
    real_copytree = shutil.copytree

    def _copy_then_lose_a_file(source: str, dest: str, *args: Any, **kwargs: Any) -> object:
        result = real_copytree(source, dest, *args, **kwargs)
        if pathlib.Path(source) == out:  # not copytree's own call for a subdirectory
            (pathlib.Path(dest) / "sub" / "f.txt").unlink()
        return result

    monkeypatch.setattr(shutil, "copytree", _copy_then_lose_a_file)

    output_hash = cache.save_to_cache(out, cache_dir, checkout_mode=cache.CheckoutMode.SYMLINK)

    tree = cache.get_cache_path(cache_dir, output_hash["hash"])
    assert out.resolve() == tree.resolve()
    assert (tree / "sub" / "f.txt").read_text() == "same"
    assert cache.hash_directory(out, cache_dir=cache_dir)["hash"] == output_hash["hash"]
    assert cache._tree_layout_marker(tree).exists()


def _helper_damaged_tree(root: pathlib.Path, files: int) -> tuple[pathlib.Path, DirHash]:
    """A flat tree of ``files`` cached files as an older pivot left it: every file but one
    missing, a read-only root, no mark. Return the cache dir and the output's hash."""
    cache_dir = root / "cache"
    src = root / "src"
    src.mkdir()
    for i in range(files):
        (src / f"f{i:03d}.txt").write_text(f"file {i}\n" * 50)
    output_hash = cache.save_to_cache(src, cache_dir, checkout_mode=cache.CheckoutMode.COPY)
    assert is_dir_hash(output_hash)
    tree = cache.get_cache_path(cache_dir, output_hash["hash"])
    tree.mkdir(parents=True)
    shutil.copy2(src / "f000.txt", tree / "f000.txt")
    (tree / "f000.txt").chmod(0o444)
    tree.chmod(0o555)
    return cache_dir, output_hash


def _helper_restore_at_barrier(
    out: pathlib.Path, output_hash: DirHash, cache_dir: pathlib.Path, barrier: Barrier
) -> str:
    barrier.wait()
    try:
        restored = cache.restore_from_cache(
            out, output_hash, cache_dir, checkout_modes=[cache.CheckoutMode.SYMLINK]
        )
    except OSError as e:
        return f"{type(e).__name__}: {e}"
    return "ok" if restored else "not restored"


def test_concurrent_restores_repair_one_damaged_tree_once(tmp_path: pathlib.Path) -> None:
    """Several processes restoring through one damaged tree with a read-only root repair it one
    at a time: each restore succeeds, the tree ends exactly what its hash records, its root
    read-only again, with no repair's temporary file left inside it."""
    n_procs = 8
    ctx = multiprocessing.get_context("spawn")
    with ctx.Manager() as manager, ctx.Pool(n_procs) as pool:
        for r in range(3):
            root = tmp_path / f"round{r}"
            root.mkdir()
            cache_dir, output_hash = _helper_damaged_tree(root, 400)
            tree = cache.get_cache_path(cache_dir, output_hash["hash"])
            barrier = manager.Barrier(n_procs, timeout=60)
            results = pool.starmap(
                _helper_restore_at_barrier,
                [(root / f"out{i}", output_hash, cache_dir, barrier) for i in range(n_procs)],
            )

            assert results == ["ok"] * n_procs
            assert sorted(p.name for p in tree.iterdir()) == sorted(
                e["relpath"] for e in output_hash["manifest"]
            )
            assert cache.hash_directory(tree)["hash"] == output_hash["hash"]
            assert tree.stat().st_mode & 0o777 == 0o555
            for i in range(n_procs):
                assert cache.hash_directory(root / f"out{i}", cache_dir=cache_dir) == output_hash


def _helper_tree_hash(src: pathlib.Path) -> DirHash:
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "f.txt").write_text(f"{src.name}\n")
    return cache.hash_directory(src)


def test_tree_publish_uses_a_prefix_lock_this_user_cannot_write(tmp_path: pathlib.Path) -> None:
    """A shared cache directory can hold lock files another user created, which this user can
    read but not write. A tree publish into that directory still works: a lock file this user
    cannot write is locked read-only."""
    cache_dir = tmp_path / "cache"
    prefix = cache_dir / "ab"
    first, second = tmp_path / "first", tmp_path / "second"
    cache._publish_tree(first, prefix / "00000000000001", _helper_tree_hash(first), cache_dir)
    for lock in prefix.glob("*.lock"):
        lock.chmod(0o444)

    second_hash = _helper_tree_hash(second)
    cache._publish_tree(second, prefix / "00000000000002", second_hash, cache_dir)

    assert cache.hash_directory(prefix / "00000000000002") == second_hash


def test_prefix_lock_is_writable_by_whoever_can_write_its_directory(
    tmp_path: pathlib.Path,
) -> None:
    """The lock file a publish creates carries its directory's write bits, so every user who
    can create entries in a shared prefix directory can also take its exclusive lock (NFS locks
    exclusively only a file opened for writing)."""
    cache_dir = tmp_path / "cache"
    prefix = cache_dir / "ab"
    prefix.mkdir(parents=True)
    prefix.chmod(0o777)
    src = tmp_path / "src"

    cache._publish_tree(src, prefix / "00000000000001", _helper_tree_hash(src), cache_dir)

    (lock,) = prefix.glob("*.lock")
    assert lock.stat().st_mode & 0o777 == 0o666


def test_sweep_reports_and_skips_a_temporary_entry_it_cannot_remove(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A temporary tree another user's killed pivot left cannot be removed; the sweep reports it
    and the publish goes on."""
    cache_dir = tmp_path / "cache"
    prefix = cache_dir / "ab"
    stuck = prefix / f"{cache._CACHE_TEMP_PREFIX}theirs"
    (stuck / "sub").mkdir(parents=True)
    real_clear = cache._clear_path

    def _clear_unless_theirs(path: pathlib.Path) -> None:
        if path == stuck:
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path / "sub"))
        real_clear(path)

    monkeypatch.setattr(cache, "_clear_path", _clear_unless_theirs)
    src = tmp_path / "src"
    src_hash = _helper_tree_hash(src)

    with caplog.at_level(logging.WARNING, logger="pivot.storage.cache"):
        cache._publish_tree(src, prefix / "00000000000001", src_hash, cache_dir)

    assert cache.hash_directory(prefix / "00000000000001") == src_hash
    assert stuck.is_dir()
    assert str(stuck) in caplog.text


def _helper_crash_during_object_copy(src: pathlib.Path, cache_path: pathlib.Path) -> None:
    """Simulate a kill while copying a file into the cache."""
    real_copy2 = shutil.copy2

    def _copy_and_crash(source: str, dest: str) -> None:
        real_copy2(source, dest)
        os._exit(73)

    shutil.copy2 = _copy_and_crash
    cache.copy_to_cache(src, cache_path)


def test_copy_to_cache_removes_a_killed_copy_beside_its_objects(tmp_path: pathlib.Path) -> None:
    """A process killed while copying a file into the cache leaves its temporary file in the
    prefix directory; a later copy into that directory removes it."""
    src = tmp_path / "out.txt"
    src.write_text("content")
    prefix = tmp_path / "cache" / "ab"
    process = multiprocessing.get_context("spawn").Process(
        target=_helper_crash_during_object_copy, args=(src, prefix / "00000000000001")
    )
    process.start()
    process.join(timeout=60)
    assert process.exitcode == 73
    assert [n[: len(cache._CACHE_TEMP_PREFIX)] for n in _helper_entries(prefix)] == [
        cache._CACHE_TEMP_PREFIX
    ]
    # The sweep of one prefix directory runs at most once per _CACHE_SWEEP_INTERVAL.
    os.utime(prefix / cache._PREFIX_LOCK, (0, 0))

    cache.copy_to_cache(src, prefix / "00000000000002")

    assert _helper_entries(prefix) == ["00000000000002"]


def test_sweep_removes_an_older_pivots_temporary_files_once_untouched_for_an_hour(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An older pivot made its temporaries without the lock, so one may belong to a copy still
    running: it is removed only once untouched for the sweep interval."""
    src = tmp_path / "out.txt"
    src.write_text("content")
    prefix = tmp_path / "cache" / "ab"
    prefix.mkdir(parents=True)
    legacy = ["tmpk3j2h1g0.tmp", ".pivot_download_x9y8z7"]
    for name in legacy:
        (prefix / name).write_text("partial")

    cache.copy_to_cache(src, prefix / "00000000000001")
    assert _helper_entries(prefix) == sorted([*legacy, "00000000000001"])

    os.utime(prefix / cache._PREFIX_LOCK, (0, 0))
    later = time.time() + 2 * cache._CACHE_SWEEP_INTERVAL
    monkeypatch.setattr(time, "time", lambda: later)
    cache.copy_to_cache(src, prefix / "00000000000002")

    assert _helper_entries(prefix) == ["00000000000001", "00000000000002"]


def test_sweep_skips_an_older_pivots_temporary_file_that_vanishes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An older pivot sharing the cache renames its temporary file into place whenever its copy
    is done, so the file can vanish between the sweep listing it and looking at it; the sweep
    moves on without asking anyone to remove it, and the copy is published."""
    src = tmp_path / "out.txt"
    src.write_text("content")
    prefix = tmp_path / "cache" / "ab"
    prefix.mkdir(parents=True)
    legacy = prefix / "tmpk3j2h1g0.tmp"
    legacy.write_text("partial")
    real_scandir = os.scandir
    renamed = list[bool]()

    def _list_then_rename_away(path: Any) -> Any:
        if path != prefix or renamed:
            return real_scandir(path)
        with real_scandir(path) as listing:
            entries = list(listing)
        legacy.rename(prefix / "0000000000000f")
        renamed.append(True)
        return iter(entries)

    monkeypatch.setattr(os, "scandir", _list_then_rename_away)

    with caplog.at_level(logging.WARNING, logger="pivot.storage.cache"):
        cache.copy_to_cache(src, prefix / "00000000000001")

    assert renamed == [True], "the sweep listed the directory"
    assert _helper_entries(prefix) == ["00000000000001", "0000000000000f"]
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.parametrize("flock_errno", [errno.ENOLCK, errno.ENOSYS, errno.EOPNOTSUPP])
def test_cache_writes_go_on_where_the_filesystem_refuses_flock(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, flock_errno: int
) -> None:
    """A cache on a filesystem that refuses flock() (Lustre without `flock`, an NFS mount whose
    lock manager does not answer) still takes new file objects and SYMLINK trees: nothing there
    can take the exclusive lock to sweep, so staging goes on without the shared one."""

    def _refuse(fd: int, operation: int) -> None:
        raise OSError(flock_errno, os.strerror(flock_errno))

    monkeypatch.setattr(fcntl, "flock", _refuse)
    src = tmp_path / "out.txt"
    src.write_text("content")
    cache_path = tmp_path / "cache" / "ab" / "00000000000001"
    out, _ = _helper_same_tree_twice(tmp_path)

    cache.copy_to_cache(src, cache_path)
    output_hash = cache.save_to_cache(
        out, tmp_path / "cache", checkout_mode=cache.CheckoutMode.SYMLINK
    )

    assert cache_path.read_text() == "content"
    tree = cache.get_cache_path(tmp_path / "cache", output_hash["hash"])
    assert out.resolve() == tree.resolve()
    assert cache.hash_directory(out, cache_dir=tmp_path / "cache")["hash"] == output_hash["hash"]


def test_cache_write_raises_a_flock_failure_that_is_not_a_refusal(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a filesystem's refusal of flock() is passed over; any other failure of the shared
    lock is raised, and nothing is published."""
    real_flock = fcntl.flock

    def _shared_fails(fd: int, operation: int) -> None:
        if operation == fcntl.LOCK_SH:
            raise OSError(errno.EIO, os.strerror(errno.EIO))
        real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", _shared_fails)
    src = tmp_path / "out.txt"
    src.write_text("content")
    cache_path = tmp_path / "cache" / "ab" / "00000000000001"

    with pytest.raises(OSError, match="Input/output error"):
        cache.copy_to_cache(src, cache_path)

    assert not cache_path.exists()


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

    # Create source directory with a file
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    (src_dir / "file.txt").write_text("content")

    # Create cache directory
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Hash and cache individual files (not using SYMLINK mode)
    output_hash = cache.hash_directory(src_dir)

    # Cache each file individually
    for entry in output_hash["manifest"]:
        file_path = src_dir / entry["relpath"]
        cache_path = cache.get_cache_path(cache_dir, entry["hash"])
        cache.copy_to_cache(file_path, cache_path)

    # Remove the source directory
    shutil.rmtree(src_dir)

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


# === Empty directories ===

# Hashes of _helper_write_pinned_tree and of an empty directory, recorded with pivot 0.3.3,
# before empty directories were part of a directory's hash.
_PINNED_TREE_HASH = "3b717a0258af2654"
_EMPTY_ROOT_HASH = "ccab0b28617f1f56"
_ALL_MODES = [cache.CheckoutMode.HARDLINK, cache.CheckoutMode.SYMLINK, cache.CheckoutMode.COPY]


def _helper_write_pinned_tree(root: pathlib.Path) -> None:
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("alpha\n")
    (root / "sub" / "b.bin").write_bytes(b"\x00\x01beta")
    os.chmod(root / "sub" / "b.bin", 0o755)


def _helper_add_empty_dirs(root: pathlib.Path) -> None:
    (root / "bundle" / "classifier_source").mkdir(parents=True)
    (root / "sub" / "x" / "y").mkdir(parents=True)


def _helper_dir_set(root: pathlib.Path) -> set[str]:
    """Every directory under root (following root itself if it is a symlink, nothing below)."""
    return {
        str(pathlib.Path(dirpath, name).relative_to(root))
        for dirpath, dirnames, _ in os.walk(root)
        for name in dirnames
        if not pathlib.Path(dirpath, name).is_symlink()
    }


def test_hash_directory_without_empty_dirs_keeps_its_hash(tmp_path: pathlib.Path) -> None:
    """A directory holding no empty directory hashes exactly as it did before empty
    directories were recorded, so no existing lock entry for it changes."""
    tree = tmp_path / "out"
    _helper_write_pinned_tree(tree)
    empty = tmp_path / "empty"
    empty.mkdir()

    result = cache.hash_directory(tree)

    assert result["hash"] == _PINNED_TREE_HASH
    assert "empty_dirs" not in result
    assert cache.hash_directory(empty) == DirHash(hash=_EMPTY_ROOT_HASH, manifest=[])


def test_hash_directory_records_deepest_empty_dirs(tmp_path: pathlib.Path) -> None:
    """An empty directory is part of a directory's content: it is recorded (only the deepest
    of a chain; its parents are implied) and it changes the hash."""
    tree = tmp_path / "out"
    _helper_write_pinned_tree(tree)
    _helper_add_empty_dirs(tree)

    result = cache.hash_directory(tree)

    assert "empty_dirs" in result
    assert result["empty_dirs"] == ["bundle/classifier_source", "sub/x/y"]
    assert [entry["relpath"] for entry in result["manifest"]] == ["a.txt", "sub/b.bin"]
    assert result["hash"] != _PINNED_TREE_HASH


def test_hash_directory_counts_directory_holding_only_left_out_symlink_as_empty(
    tmp_path: pathlib.Path,
) -> None:
    """A user symlink is not stored, so a directory holding only one restores empty; its hash
    records it as the empty directory it restores as."""
    tree = tmp_path / "out"
    (tree / "links").mkdir(parents=True)
    (tree / "keep.txt").write_text("keep")
    (tree / "links" / "keep.txt").symlink_to(tree / "keep.txt")

    result = cache.hash_directory(tree)

    assert "empty_dirs" in result
    assert result["empty_dirs"] == ["links"]


@pytest.mark.parametrize("mode", _ALL_MODES)
@pytest.mark.parametrize("drop_tree", [False, True], ids=["cached-tree", "file-objects-only"])
def test_directory_output_keeps_empty_dirs_through_cache_and_restore(
    tmp_path: pathlib.Path, mode: cache.CheckoutMode, drop_tree: bool
) -> None:
    """A directory output restored from the cache is the tree that was saved: its empty
    directories come back and it hashes to the hash recorded for it.

    ``file-objects-only`` restores without a whole-directory SYMLINK cache tree, as after
    `pivot pull`, which fetches file objects only.
    """
    output = tmp_path / "out"
    _helper_write_pinned_tree(output)
    _helper_add_empty_dirs(output)
    expected_dirs = _helper_dir_set(output)
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(output, cache_dir, checkout_modes=[mode])
    assert _helper_dir_set(output) == expected_dirs, "saving must keep every directory in place"

    cache.remove_output(output)
    if drop_tree:
        cache.remove_output(cache.get_cache_path(cache_dir, output_hash["hash"]))
    assert cache.restore_from_cache(output, output_hash, cache_dir, checkout_mode=mode)

    assert _helper_dir_set(output) == expected_dirs
    assert cache.hash_directory(output, cache_dir=cache_dir) == output_hash


def test_restore_directory_rejects_empty_dir_path_traversal(tmp_path: pathlib.Path) -> None:
    """A recorded empty directory may come straight from a lock file; one pointing outside
    the output must raise before anything is created there."""
    cache_dir = tmp_path / "cache"
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    saved = cache.save_to_cache(output, cache_dir, checkout_mode=cache.CheckoutMode.COPY)
    assert is_dir_hash(saved)
    cache.remove_output(output)
    malicious = DirHash(hash=saved["hash"], manifest=saved["manifest"], empty_dirs=["../escaped"])

    with pytest.raises(exceptions.SecurityValidationError, match="path traversal"):
        cache.restore_from_cache(
            output, malicious, cache_dir, checkout_mode=cache.CheckoutMode.COPY
        )

    assert not (tmp_path / "escaped").exists()


def _helper_add_stale_dir_to_tree(tree_path: pathlib.Path) -> None:
    """Make a whole-directory SYMLINK cache tree look built by an older pivot: an empty
    directory its hash does not record (it hashed without them but copied them into the tree),
    and no record that this pivot checked the tree."""
    tree_path.chmod(0o755)
    (tree_path / "stale" / "deep").mkdir(parents=True)
    tree_path.chmod(0o555)
    cache._tree_layout_marker(tree_path).unlink()


def test_restore_from_cache_repairs_legacy_symlink_tree_empty_dirs(tmp_path: pathlib.Path) -> None:
    """Restoring through a legacy SYMLINK cache tree gives the tree its hash records, not one
    with extra empty directories that would hash differently from the lock."""
    cache_dir = tmp_path / "cache"
    modes = [cache.CheckoutMode.SYMLINK]
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    output_hash = cache.save_to_cache(output, cache_dir, checkout_modes=modes)
    _helper_add_stale_dir_to_tree(cache.get_cache_path(cache_dir, output_hash["hash"]))

    cache.remove_output(output)
    assert cache.restore_from_cache(output, output_hash, cache_dir, checkout_modes=modes)

    assert _helper_dir_set(output) == set()
    assert cache.hash_directory(output, cache_dir=cache_dir) == output_hash


def test_save_to_cache_symlink_repairs_legacy_tree_empty_dirs(tmp_path: pathlib.Path) -> None:
    """Saving an output whose hash names an existing legacy tree repairs that tree too."""
    cache_dir = tmp_path / "cache"
    modes = [cache.CheckoutMode.SYMLINK]
    first = tmp_path / "first"
    first.mkdir()
    (first / "keep.txt").write_text("keep")
    output_hash = cache.save_to_cache(first, cache_dir, checkout_modes=modes)
    _helper_add_stale_dir_to_tree(cache.get_cache_path(cache_dir, output_hash["hash"]))

    second = tmp_path / "second"
    second.mkdir()
    (second / "keep.txt").write_text("keep")
    assert cache.save_to_cache(second, cache_dir, checkout_modes=modes) == output_hash

    assert _helper_dir_set(second) == set()
    assert cache.hash_directory(second, cache_dir=cache_dir) == output_hash


def _helper_tree_listing(root: pathlib.Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_symlink_tree_leaves_out_user_symlinks(tmp_path: pathlib.Path) -> None:
    """A whole-directory SYMLINK cache tree holds what the hash records: a user symlink, which
    is not stored, does not come back as a copy of its target, and a directory holding only
    one comes back empty, as it does from a file-by-file restore."""
    cache_dir = tmp_path / "cache"
    modes = [cache.CheckoutMode.SYMLINK]
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "big.bin").write_text("big")
    output = tmp_path / "out"
    (output / "links").mkdir(parents=True)
    (output / "keep.txt").write_text("keep")
    (output / "link.txt").symlink_to(output / "keep.txt")
    (output / "links" / "elsewhere").symlink_to(elsewhere)

    output_hash = cache.save_to_cache(output, cache_dir, checkout_modes=modes)
    cache.remove_output(output)
    assert cache.restore_from_cache(output, output_hash, cache_dir, checkout_modes=modes)

    assert _helper_tree_listing(output) == ["keep.txt", "links"]
    assert cache.hash_directory(output, cache_dir=cache_dir) == output_hash


def test_restore_from_cache_repairs_legacy_symlink_tree_unlisted_files(
    tmp_path: pathlib.Path,
) -> None:
    """An older pivot copied a user symlink's target into the SYMLINK cache tree; restoring
    through that tree removes what the hash does not record."""
    cache_dir = tmp_path / "cache"
    modes = [cache.CheckoutMode.SYMLINK]
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    output_hash = cache.save_to_cache(output, cache_dir, checkout_modes=modes)
    tree_path = cache.get_cache_path(cache_dir, output_hash["hash"])
    tree_path.chmod(0o755)
    (tree_path / "link.txt").write_text("keep")
    (tree_path / "link.txt").chmod(0o444)
    (tree_path / "linked").mkdir()
    (tree_path / "linked" / "big.bin").write_text("big")
    tree_path.chmod(0o555)
    cache._tree_layout_marker(tree_path).unlink()

    cache.remove_output(output)
    assert cache.restore_from_cache(output, output_hash, cache_dir, checkout_modes=modes)

    assert _helper_tree_listing(output) == ["keep.txt"]
    assert cache.hash_directory(output, cache_dir=cache_dir) == output_hash


@pytest.mark.parametrize("with_empty_dir", [False, True], ids=["no-empty-dirs", "empty-dirs"])
def test_symlink_restore_does_not_walk_a_tree_this_pivot_built(
    tmp_path: pathlib.Path, mocker: MockerFixture, with_empty_dir: bool
) -> None:
    """The SYMLINK fast path stays a symlink swap: a tree this pivot built already holds what
    its hash records, so restoring through it does not walk it (that walk took about 0.5 s on
    20k files)."""
    cache_dir = tmp_path / "cache"
    modes = [cache.CheckoutMode.SYMLINK]
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    if with_empty_dir:
        (output / "empty").mkdir()
    output_hash = cache.save_to_cache(output, cache_dir, checkout_modes=modes)
    scan = mocker.spy(cache, "_scan_entries")

    for _ in range(2):
        cache.remove_output(output)
        assert cache.restore_from_cache(output, output_hash, cache_dir, checkout_modes=modes)

    assert scan.call_count == 0


def test_symlink_restore_repairs_a_legacy_tree_once(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    cache_dir = tmp_path / "cache"
    modes = [cache.CheckoutMode.SYMLINK]
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    output_hash = cache.save_to_cache(output, cache_dir, checkout_modes=modes)
    _helper_add_stale_dir_to_tree(cache.get_cache_path(cache_dir, output_hash["hash"]))
    scan = mocker.spy(cache, "_scan_entries")

    for _ in range(2):
        cache.remove_output(output)
        assert cache.restore_from_cache(output, output_hash, cache_dir, checkout_modes=modes)

    assert scan.call_count == 1
    assert _helper_dir_set(output) == set()


def test_restore_missing_in_directory_recreates_missing_empty_dir(tmp_path: pathlib.Path) -> None:
    cache_dir = tmp_path / "cache"
    output = tmp_path / "out"
    (output / "empty").mkdir(parents=True)
    (output / "keep.txt").write_text("keep")
    output_hash = cache.save_to_cache(output, cache_dir, checkout_mode=cache.CheckoutMode.COPY)
    assert is_dir_hash(output_hash)
    (output / "empty").rmdir()

    outcome = cache.restore_missing_in_directory(
        output, output_hash, cache_dir, [cache.CheckoutMode.COPY]
    )

    assert outcome == cache.MissingRestore(restored=0, missing=0, local_changes=[], replace=False)
    assert (output / "empty").is_dir()
    assert cache.hash_directory(output) == output_hash


def test_restore_missing_in_directory_replaces_directory_holding_extra_empty_dir(
    tmp_path: pathlib.Path,
) -> None:
    """An extra empty directory holds nothing the swap could lose, so the directory is
    replaced whole to reach the target."""
    cache_dir = tmp_path / "cache"
    output = tmp_path / "out"
    (output / "recorded").mkdir(parents=True)
    (output / "keep.txt").write_text("keep")
    output_hash = cache.save_to_cache(output, cache_dir, checkout_mode=cache.CheckoutMode.COPY)
    assert is_dir_hash(output_hash)
    (output / "extra" / "deep").mkdir(parents=True)

    outcome = cache.restore_missing_in_directory(
        output, output_hash, cache_dir, [cache.CheckoutMode.COPY]
    )

    assert outcome == cache.MissingRestore(restored=0, missing=0, local_changes=[], replace=True)


def test_restore_missing_in_directory_keeps_empty_dirs_a_0_3_3_hash_does_not_list(
    tmp_path: pathlib.Path,
) -> None:
    """A hash pivot 0.3.3 recorded lists no empty directories, so one on disk is not extra:
    the directory is left as it is rather than replaced, which would drop it."""
    cache_dir = tmp_path / "cache"
    output = tmp_path / "out"
    (output / "cls" / "src").mkdir(parents=True)
    (output / "keep.txt").write_text("keep")
    saved = cache.save_to_cache(output, cache_dir, checkout_mode=cache.CheckoutMode.COPY)
    assert is_dir_hash(saved)
    legacy = DirHash(hash=saved["hash"], manifest=saved["manifest"])

    outcome = cache.restore_missing_in_directory(
        output, legacy, cache_dir, [cache.CheckoutMode.COPY]
    )

    assert outcome == cache.MissingRestore(restored=0, missing=0, local_changes=[], replace=False)
    assert (output / "cls" / "src").is_dir()


def test_symlink_restore_succeeds_when_layout_marker_cannot_be_written(
    tmp_path: pathlib.Path,
) -> None:
    """The layout marker only saves a walk: a cache directory the restoring user cannot write
    (another user's cache, a read-only mount) still restores, as it did before the marker."""
    cache_dir = tmp_path / "cache"
    modes = [cache.CheckoutMode.SYMLINK]
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    output_hash = cache.save_to_cache(output, cache_dir, checkout_modes=modes)
    tree_path = cache.get_cache_path(cache_dir, output_hash["hash"])
    cache._tree_layout_marker(tree_path).unlink()
    shard = tree_path.parent
    shard.chmod(0o555)
    try:
        cache.remove_output(output)
        assert cache.restore_from_cache(output, output_hash, cache_dir, checkout_modes=modes)
    finally:
        shard.chmod(0o755)

    assert (output / "keep.txt").read_text() == "keep"
