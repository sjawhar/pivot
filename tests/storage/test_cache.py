import mmap
import os
import pathlib
import stat
import time
from typing import TYPE_CHECKING

import pytest

from pivot.storage import cache, state

if TYPE_CHECKING:
    from pivot.types import DirHash, FileHash


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


def test_hash_file_large_uses_mmap(tmp_path: pathlib.Path) -> None:
    """Large files (>=MMAP_THRESHOLD) use mmap and produce valid hashes."""
    pattern = b"x" * 1024  # 1KB pattern

    small_file = tmp_path / "small.bin"
    small_file.write_bytes(pattern)

    # Create file just over MMAP_THRESHOLD to ensure mmap path is used
    large_file = tmp_path / "large.bin"
    large_file.write_bytes(b"x" * (cache.MMAP_THRESHOLD + 1))

    # Both should produce valid hashes
    small_hash = cache.hash_file(small_file)
    large_hash = cache.hash_file(large_file)

    assert len(small_hash) == 16
    assert len(large_hash) == 16
    # Hashes should be different (different content sizes)
    assert small_hash != large_hash


def test_hash_file_mmap_consistent(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mmap and buffered read produce identical hashes for same content."""
    content = b"test content" * 1000000  # ~12MB, above threshold
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(content)

    # Get hash via mmap path (file > threshold)
    mmap_hash = cache.hash_file(test_file)

    # Force buffered path by raising threshold above file size
    monkeypatch.setattr(cache, "MMAP_THRESHOLD", len(content) + 1)
    buffered_hash = cache.hash_file(test_file)

    # Both methods must produce identical hash
    assert mmap_hash == buffered_hash


def test_hash_file_mmap_fallback(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Falls back to buffered read when mmap fails."""
    content = b"test content" * 1000000  # ~12MB
    large_file = tmp_path / "large.bin"
    large_file.write_bytes(content)

    # Get expected hash via normal path first
    expected_hash = cache.hash_file(large_file)

    # Now make mmap fail
    def failing_mmap(*args: object, **kwargs: object) -> mmap.mmap:
        raise OSError("mmap failed")

    monkeypatch.setattr(mmap, "mmap", failing_mmap)

    # Should fall back to buffered read and produce same hash
    fallback_hash = cache.hash_file(large_file)
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


def test_hash_directory_skips_unreadable_subdirs(tmp_path: pathlib.Path) -> None:
    """Unreadable subdirectories are skipped rather than causing failure."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()
    (test_dir / "readable.txt").write_text("content")
    unreadable = test_dir / "unreadable"
    unreadable.mkdir()
    (unreadable / "hidden.txt").write_text("hidden")
    unreadable.chmod(0o000)

    try:
        _, manifest = cache.hash_directory(test_dir)
        relpaths = [e["relpath"] for e in manifest]
        assert "readable.txt" in relpaths
        assert "unreadable/hidden.txt" not in relpaths
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
    # No temp directories left behind (lock files are expected and cleaned up by age)
    temp_dirs = [
        p
        for p in tmp_path.iterdir()
        if (
            p.name.startswith(cache._RESTORE_TEMP_PREFIX)
            or p.name.startswith(cache._BACKUP_TEMP_PREFIX)
        )
        and not p.name.startswith(cache._RESTORE_LOCK_PREFIX)
    ]
    assert len(temp_dirs) == 0


def test_restore_directory_atomic_cleans_up_on_cache_miss(
    tmp_path: pathlib.Path,
) -> None:
    """Atomic restore returns False without creating temp dirs when cache missing."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    target = tmp_path / "mydir"

    # Create hash with non-existent cache entry
    missing_hash: DirHash = {
        "hash": "missing123456789",
        "manifest": [{"relpath": "file.txt", "hash": "0" * 16, "size": 7, "isexec": False}],
    }

    result = cache.restore_from_cache(
        target, missing_hash, cache_dir, checkout_mode=cache.CheckoutMode.HARDLINK
    )

    assert result is False
    assert not target.exists()
    # No temp directories left behind (validation happens before temp creation)
    temp_dirs = [
        p
        for p in tmp_path.iterdir()
        if p.name.startswith(cache._RESTORE_TEMP_PREFIX)
        or p.name.startswith(cache._BACKUP_TEMP_PREFIX)
    ]
    assert len(temp_dirs) == 0


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
    # No temp directories left behind (lock files are expected and cleaned up by age)
    temp_dirs = [
        p
        for p in tmp_path.iterdir()
        if (
            p.name.startswith(cache._RESTORE_TEMP_PREFIX)
            or p.name.startswith(cache._BACKUP_TEMP_PREFIX)
        )
        and not p.name.startswith(cache._RESTORE_LOCK_PREFIX)
    ]
    assert len(temp_dirs) == 0


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
        "hash": "test123456789012",
        "manifest": [
            {"relpath": "first.txt", "hash": first_hash, "size": 5, "isexec": False},
            {"relpath": "second.txt", "hash": "missing123456789", "size": 6, "isexec": False},
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
