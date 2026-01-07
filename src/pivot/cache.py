from __future__ import annotations

import contextlib
import enum
import errno
import json
import os
import pathlib
import shutil
import stat
import tempfile
from typing import TYPE_CHECKING

import xxhash

from pivot import exceptions
from pivot.types import DirHash, DirManifestEntry, FileHash, OutputHash

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from pivot import state as state_mod

CHUNK_SIZE = 1024 * 1024  # 1MB chunks for hashing
XXHASH64_HEX_LENGTH = 16  # xxhash64 produces 64-bit hash = 16 hex characters


def atomic_write_file(
    dest: pathlib.Path,
    write_fn: Callable[[int], None],
    mode: int = 0o644,
) -> None:
    """Atomically write to dest using temp file + rename pattern.

    Args:
        dest: Target file path.
        write_fn: Function that receives the file descriptor and writes content.
                  MUST close fd (typically via os.fdopen which takes ownership).
        mode: File permissions (default 0o644).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    tmp = pathlib.Path(tmp_path)
    fd_closed = False
    try:
        write_fn(fd)
        fd_closed = True  # write_fn took ownership via os.fdopen
        os.chmod(tmp_path, mode)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        # Only close fd if write_fn didn't (e.g., exception before os.fdopen)
        if not fd_closed:
            with contextlib.suppress(OSError):
                os.close(fd)


class LinkMode(enum.StrEnum):
    """Strategy for linking workspace files to cache."""

    SYMLINK = "symlink"
    HARDLINK = "hardlink"
    COPY = "copy"


def hash_file(path: pathlib.Path, state_db: state_mod.StateDB | None = None) -> str:
    """Compute xxhash64 of file contents, using state cache if available."""
    file_stat = path.stat()

    if state_db is not None:
        cached = state_db.get(path, file_stat)
        if cached is not None:
            return cached

    hasher = xxhash.xxh64()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            hasher.update(chunk)
    file_hash = hasher.hexdigest()

    if state_db is not None:
        state_db.save(path, file_stat, file_hash)

    return file_hash


def _scandir_recursive(path: pathlib.Path) -> Generator[os.DirEntry[str]]:
    """Yield all files recursively using os.scandir() for efficiency.

    DirEntry objects cache stat results, avoiding redundant syscalls.
    Symlinks are skipped to prevent loops.
    """
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_file():
                    yield entry
                elif entry.is_dir():
                    yield from _scandir_recursive(pathlib.Path(entry.path))
    except PermissionError:
        pass  # Skip unreadable directories rather than failing the entire walk


def hash_directory(
    path: pathlib.Path, state_db: state_mod.StateDB | None = None
) -> tuple[str, list[DirManifestEntry]]:
    """Compute tree hash of directory, returning hash and manifest.

    Symlink handling:
    - Symlinks INSIDE directories are skipped (prevents infinite loops)
    - Base path may be symlinked (resolved for consistency)
    - Content-based hashing only (symlink metadata excluded from fingerprints)

    Note: For portability, paths are stored as normalized (symlinks preserved)
    in lock files, but resolved here for consistent hashing.
    """
    manifest = list[DirManifestEntry]()
    resolved_base = path.resolve()

    for entry in sorted(_scandir_recursive(path), key=lambda e: e.path):
        file_path = pathlib.Path(entry.path)
        # Verify file is still within the directory (paranoid check)
        if not file_path.resolve().is_relative_to(resolved_base):
            continue
        try:
            rel = file_path.relative_to(path)
            file_stat = entry.stat(follow_symlinks=True)
            manifest_entry: DirManifestEntry = {
                "relpath": str(rel),
                "hash": hash_file(file_path, state_db),
                "size": file_stat.st_size,
            }
            if file_stat.st_mode & stat.S_IXUSR:
                manifest_entry["isexec"] = True
            manifest.append(manifest_entry)
        except FileNotFoundError:
            continue  # File deleted between scan and hash

    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    tree_hash = xxhash.xxh64(manifest_json.encode()).hexdigest()

    return tree_hash, manifest


def get_cache_path(cache_dir: pathlib.Path, file_hash: str) -> pathlib.Path:
    """Get cache path for a hash (XX/XXXX... structure)."""
    if len(file_hash) < 3:
        raise ValueError(f"Hash too short for cache path structure: {len(file_hash)} chars")
    return cache_dir / file_hash[:2] / file_hash[2:]


def _clear_path(path: pathlib.Path) -> None:
    """Remove file, symlink, or directory at path if it exists."""
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _get_symlink_cache_hash(path: pathlib.Path, cache_dir: pathlib.Path) -> str | None:
    """Extract hash from symlink target if it points to cache, else None."""
    if not path.is_symlink():
        return None
    try:
        target = path.resolve()
        cache_resolved = cache_dir.resolve()
        if not target.is_relative_to(cache_resolved):
            return None
        rel = target.relative_to(cache_resolved)
        if len(rel.parts) != 2:
            return None
        reconstructed = rel.parts[0] + rel.parts[1]
        if len(reconstructed) != XXHASH64_HEX_LENGTH:
            return None
        return reconstructed
    except (OSError, ValueError):
        return None  # Includes ELOOP for circular symlinks


def copy_to_cache(src: pathlib.Path, cache_path: pathlib.Path) -> None:
    """Atomically copy file to cache with read-only permissions."""
    if cache_path.exists():
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=cache_path.parent, suffix=".tmp")
    tmp = pathlib.Path(tmp_path)
    try:
        os.close(fd)
        shutil.copy2(src, tmp_path)
        os.chmod(tmp_path, 0o444)
        tmp.replace(cache_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _link_from_cache(
    path: pathlib.Path,
    cache_path: pathlib.Path,
    link_mode: LinkMode,
    *,
    executable: bool = False,
) -> None:
    """Create link from workspace path to cache."""
    # Idempotency: skip if already correctly linked
    if link_mode == LinkMode.SYMLINK and path.is_symlink():
        with contextlib.suppress(OSError):
            if path.resolve() == cache_path.resolve():
                return
    elif link_mode == LinkMode.HARDLINK and path.exists() and not path.is_symlink():
        with contextlib.suppress(OSError):
            if path.stat().st_ino == cache_path.stat().st_ino:
                return

    _clear_path(path)

    if link_mode == LinkMode.SYMLINK:
        path.symlink_to(cache_path.resolve())
    elif link_mode == LinkMode.HARDLINK:
        try:
            os.link(cache_path, path)
        except OSError as e:
            # Fall back to copy if hardlink fails (e.g., cross-device)
            if e.errno == errno.EXDEV:
                shutil.copy2(cache_path, path)
                os.chmod(path, 0o644)
            else:
                raise
    else:
        shutil.copy2(cache_path, path)
        os.chmod(path, 0o755 if executable else 0o644)


def save_to_cache(
    path: pathlib.Path,
    cache_dir: pathlib.Path,
    state_db: state_mod.StateDB | None = None,
    link_mode: LinkMode = LinkMode.SYMLINK,
) -> OutputHash:
    """Save file or directory to cache, replace with link, return hash info."""
    if path.is_dir():
        return _save_directory_to_cache(path, cache_dir, state_db, link_mode)
    return _save_file_to_cache(path, cache_dir, state_db, link_mode)


def _save_file_to_cache(
    path: pathlib.Path,
    cache_dir: pathlib.Path,
    state_db: state_mod.StateDB | None,
    link_mode: LinkMode,
) -> FileHash:
    """Save single file to cache."""
    # Idempotency: check if already a valid cache symlink (cheap check first)
    if link_mode == LinkMode.SYMLINK:
        existing_hash = _get_symlink_cache_hash(path, cache_dir)
        if existing_hash is not None:
            cache_path = get_cache_path(cache_dir, existing_hash)
            if cache_path.exists():
                return FileHash(hash=existing_hash)

    file_hash = hash_file(path, state_db)
    cache_path = get_cache_path(cache_dir, file_hash)

    copy_to_cache(path, cache_path)
    _link_from_cache(path, cache_path, link_mode)

    return FileHash(hash=file_hash)


def _save_directory_to_cache(
    path: pathlib.Path,
    cache_dir: pathlib.Path,
    state_db: state_mod.StateDB | None,
    link_mode: LinkMode,
) -> DirHash:
    """Save directory to cache."""
    # Idempotency check for SYMLINK mode
    if link_mode == LinkMode.SYMLINK and path.is_symlink():
        existing_hash = _get_symlink_cache_hash(path, cache_dir)
        if existing_hash is not None:
            cache_dir_path = get_cache_path(cache_dir, existing_hash)
            if cache_dir_path.exists():
                # Already correctly linked - compute manifest from actual content
                _, manifest = hash_directory(path, state_db)
                return DirHash(hash=existing_hash, manifest=manifest)

    tree_hash, manifest = hash_directory(path, state_db)

    for entry in manifest:
        file_path = path / entry["relpath"]
        cache_path = get_cache_path(cache_dir, entry["hash"])
        copy_to_cache(file_path, cache_path)

    if link_mode == LinkMode.SYMLINK:
        cache_dir_path = get_cache_path(cache_dir, tree_hash)
        if not cache_dir_path.exists():
            cache_dir_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(path, cache_dir_path)
            for f in cache_dir_path.rglob("*"):
                # Skip symlinks to avoid changing permissions on target files outside cache
                if f.is_file() and not f.is_symlink():
                    os.chmod(f, 0o444)
            os.chmod(cache_dir_path, 0o555)

        _clear_path(path)
        path.symlink_to(cache_dir_path.resolve())

    return DirHash(hash=tree_hash, manifest=manifest)


def restore_from_cache(
    path: pathlib.Path,
    output_hash: OutputHash,
    cache_dir: pathlib.Path,
    link_mode: LinkMode = LinkMode.SYMLINK,
) -> bool:
    """Restore file or directory from cache. Returns True if successful."""
    if output_hash is None:
        return False

    if "manifest" in output_hash:
        return _restore_directory_from_cache(path, output_hash, cache_dir, link_mode)
    return _restore_file_from_cache(path, output_hash, cache_dir, link_mode)


def _restore_file_from_cache(
    path: pathlib.Path,
    output_hash: FileHash,
    cache_dir: pathlib.Path,
    link_mode: LinkMode,
) -> bool:
    """Restore single file from cache."""
    cache_path = get_cache_path(cache_dir, output_hash["hash"])
    if not cache_path.exists():
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    _link_from_cache(path, cache_path, link_mode)
    return True


def _restore_directory_from_cache(
    path: pathlib.Path,
    output_hash: DirHash,
    cache_dir: pathlib.Path,
    link_mode: LinkMode,
) -> bool:
    """Restore directory from cache."""
    cache_dir_path = get_cache_path(cache_dir, output_hash["hash"])

    if link_mode == LinkMode.SYMLINK and cache_dir_path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _clear_path(path)
        path.symlink_to(cache_dir_path.resolve())
        return True

    # For COPY/HARDLINK modes, restore file-by-file
    _clear_path(path)
    path.mkdir(parents=True, exist_ok=True)
    resolved_base = path.resolve()
    for entry in output_hash["manifest"]:
        file_cache_path = get_cache_path(cache_dir, entry["hash"])
        if not file_cache_path.exists():
            return False
        file_path = path / entry["relpath"]
        # Validate no path traversal (e.g., "../../../etc/passwd")
        if not file_path.resolve().is_relative_to(resolved_base):
            raise exceptions.SecurityValidationError(
                f"Manifest contains path traversal: {entry['relpath']!r}"
            )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        _link_from_cache(
            file_path, file_cache_path, link_mode, executable=entry.get("isexec", False)
        )

    # Ensure all directories are writable for COPY mode (including root)
    if link_mode == LinkMode.COPY:
        os.chmod(path, 0o755)
        for dir_path in path.rglob("*"):
            if dir_path.is_dir():
                os.chmod(dir_path, 0o755)

    return True


def remove_output(path: pathlib.Path) -> None:
    """Remove output file or directory before execution."""
    _clear_path(path)


def protect(path: pathlib.Path) -> None:
    """Make file read-only (mode 0o444)."""
    os.chmod(path, 0o444)


def unprotect(path: pathlib.Path) -> None:
    """Restore write permission to file."""
    current = path.stat().st_mode
    os.chmod(path, current | stat.S_IWUSR)
