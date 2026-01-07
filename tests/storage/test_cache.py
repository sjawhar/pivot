"""Tests for content-addressable cache module."""

import pathlib
import stat
from typing import TYPE_CHECKING

from pivot import cache, state

if TYPE_CHECKING:
    from pivot.types import FileHash


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
    assert "isexec" not in manifest_dict["regular.txt"]
    assert manifest_dict["script.sh"].get("isexec") is True


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
    """Original file is replaced with symlink to cache."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    cache.save_to_cache(test_file, cache_dir)

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
    """restore_from_cache creates symlink to cached file."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(test_file, cache_dir)
    test_file.unlink()

    restored = cache.restore_from_cache(test_file, output_hash, cache_dir)

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

    output_hash = cache.save_to_cache(test_dir, cache_dir, link_mode=cache.LinkMode.HARDLINK)
    cache.remove_output(test_dir)

    restored = cache.restore_from_cache(
        test_dir, output_hash, cache_dir, link_mode=cache.LinkMode.HARDLINK
    )

    assert restored is True
    assert (test_dir / "file.txt").read_text() == "content"
    assert (subdir / "nested.txt").read_text() == "nested"


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


# === Link Mode Tests ===


def test_link_mode_symlink(tmp_path: pathlib.Path) -> None:
    """SYMLINK mode creates symlinks."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    cache.save_to_cache(test_file, cache_dir, link_mode=cache.LinkMode.SYMLINK)

    assert test_file.is_symlink()


def test_link_mode_hardlink(tmp_path: pathlib.Path) -> None:
    """HARDLINK mode creates hardlinks."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(test_file, cache_dir, link_mode=cache.LinkMode.HARDLINK)
    assert output_hash is not None

    cache_path = cache_dir / output_hash["hash"][:2] / output_hash["hash"][2:]
    assert test_file.stat().st_ino == cache_path.stat().st_ino


def test_link_mode_copy(tmp_path: pathlib.Path) -> None:
    """COPY mode creates separate copies."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    cache_dir = tmp_path / "cache"

    output_hash = cache.save_to_cache(test_file, cache_dir, link_mode=cache.LinkMode.COPY)
    assert output_hash is not None

    cache_path = cache_dir / output_hash["hash"][:2] / output_hash["hash"][2:]
    assert not test_file.is_symlink()
    assert test_file.stat().st_ino != cache_path.stat().st_ino
    assert test_file.read_text() == "content"
