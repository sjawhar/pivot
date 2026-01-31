from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import logging
import mmap
import os
import pathlib
import secrets
import shutil
import stat
import tempfile
import time
from typing import TYPE_CHECKING

import xxhash

from pivot import exceptions, metrics
from pivot.config import CheckoutMode as CheckoutMode
from pivot.types import DirHash, DirManifestEntry, FileHash, OutputHash

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from pivot import ignore as ignore_mod
    from pivot.storage import state as state_mod

CHUNK_SIZE = 1024 * 1024  # 1MB chunks for hashing
MMAP_THRESHOLD = 10 * 1024 * 1024  # 10MB - use mmap for files larger than this
XXHASH64_HEX_LENGTH = 16  # xxhash64 produces 64-bit hash = 16 hex characters

_RESTORE_TEMP_PREFIX = ".pivot_restore_"
_BACKUP_TEMP_PREFIX = ".pivot_backup_"
_RESTORE_LOCK_PREFIX = ".pivot_restore_lock_"
_RESTORE_TEMP_MAX_AGE = 3600  # 1 hour


def _get_lock_filename(path_name: str) -> str:
    """Generate lock filename, hashing if too long for filesystem."""
    max_name_len = 255 - len(_RESTORE_LOCK_PREFIX)
    if len(path_name) <= max_name_len:
        return f"{_RESTORE_LOCK_PREFIX}{path_name}"
    # Hash long names to fit filesystem limit (first 32 hex chars of SHA-256 = 128 bits)
    name_hash = hashlib.sha256(path_name.encode()).hexdigest()[:32]
    return f"{_RESTORE_LOCK_PREFIX}{name_hash}"


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


def hash_file(path: pathlib.Path, state_db: state_mod.StateDB | None = None) -> str:
    """Compute xxhash64 of file contents, using state cache if available."""
    file_stat = path.stat()

    if state_db is not None:
        cached = state_db.get(path, file_stat)
        if cached is not None:
            return cached

    with metrics.timed("cache.hash_file"):
        hasher = xxhash.xxh64()
        with open(path, "rb") as f:
            if file_stat.st_size >= MMAP_THRESHOLD:
                try:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        hasher.update(mm)
                except (ValueError, OSError):
                    # Fall back to buffered read if mmap fails (empty file, network FS, etc.)
                    while chunk := f.read(CHUNK_SIZE):
                        hasher.update(chunk)
            else:
                while chunk := f.read(CHUNK_SIZE):
                    hasher.update(chunk)
        file_hash = hasher.hexdigest()

    if state_db is not None and not state_db.readonly:
        state_db.save(path, file_stat, file_hash)

    return file_hash


# Hardcoded ignore patterns for hot path - O(1) lookups only.
# For full .pivotignore support, see pivot.ignore module.
_IGNORE_DIRS: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".venv",
        "venv",
        ".git",
        ".hg",
        ".svn",
        ".idea",
        ".vscode",
        "node_modules",
        ".pivot",
        "dist",
        "build",
    }
)
_IGNORE_SUFFIXES: tuple[str, ...] = (".pyc", ".pyo", ".swp", ".swo")


def _should_skip_entry(entry: os.DirEntry[str]) -> bool:
    """Fast ignore check for hot path. O(1) lookups only."""
    name = entry.name
    if entry.is_dir(follow_symlinks=False):
        return name in _IGNORE_DIRS
    # File checks: suffix and special patterns
    return name.endswith(_IGNORE_SUFFIXES) or name.endswith("~") or name.startswith(".#")


def _scandir_recursive(
    path: pathlib.Path,
    ignore_filter: ignore_mod.IgnoreFilter | None = None,
    base_path: pathlib.Path | None = None,
) -> Generator[os.DirEntry[str]]:
    """Yield all files recursively using os.scandir() for efficiency.

    DirEntry objects cache stat results, avoiding redundant syscalls.
    Symlinks are skipped to prevent loops.
    PermissionError propagates to caller - partial hashes would be incorrect.

    Args:
        path: Directory to scan
        ignore_filter: Optional IgnoreFilter for .pivotignore patterns
        base_path: Base path for computing relative paths (for ignore matching).
            Defaults to the initial path if not provided.
    """
    if base_path is None:
        base_path = path

    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_symlink():
                continue
            # Fast path: hardcoded patterns (O(1) lookup)
            if _should_skip_entry(entry):
                continue
            # Slow path: .pivotignore patterns (if provided)
            if ignore_filter is not None:
                # Pass absolute path so is_ignored() can relativize to project root.
                # This ensures patterns like "data/raw/*.csv" match when hashing
                # a subdirectory like "data/".
                entry_path = pathlib.Path(entry.path)
                is_dir = entry.is_dir(follow_symlinks=False)
                if ignore_filter.is_ignored(str(entry_path), is_dir=is_dir):
                    continue
            if entry.is_file():
                yield entry
            elif entry.is_dir():
                yield from _scandir_recursive(pathlib.Path(entry.path), ignore_filter, base_path)


def hash_directory(
    path: pathlib.Path,
    state_db: state_mod.StateDB | None = None,
    ignore_filter: ignore_mod.IgnoreFilter | None = None,
) -> tuple[str, list[DirManifestEntry]]:
    """Compute tree hash of directory, returning hash and manifest.

    Symlink handling:
    - Symlinks INSIDE directories are skipped (prevents infinite loops)
    - Base path may be symlinked (resolved for consistency)
    - Content-based hashing only (symlink metadata excluded from fingerprints)

    Note: For portability, paths are stored as normalized (symlinks preserved)
    in lock files, but resolved here for consistent hashing.

    Args:
        path: Directory to hash
        state_db: Optional StateDB for caching file hashes
        ignore_filter: Optional IgnoreFilter for .pivotignore patterns.
            When provided, files matching .pivotignore patterns are excluded
            from the hash (in addition to hardcoded patterns like __pycache__).
    """
    with metrics.timed("cache.hash_directory"):
        manifest = list[DirManifestEntry]()
        resolved_base = path.resolve()

        for entry in sorted(_scandir_recursive(path, ignore_filter), key=lambda e: e.path):
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
                    "isexec": bool(file_stat.st_mode & stat.S_IXUSR),
                }
                manifest.append(manifest_entry)
            except FileNotFoundError:
                continue  # File deleted between scan and hash

        manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        tree_hash = xxhash.xxh64(manifest_json.encode()).hexdigest()

        return tree_hash, manifest


_VALID_HASH_CHARS = frozenset("0123456789abcdef")


def get_cache_path(cache_dir: pathlib.Path, file_hash: str) -> pathlib.Path:
    """Get cache path for a hash (XX/XXXX... structure)."""
    if len(file_hash) != XXHASH64_HEX_LENGTH:
        raise exceptions.SecurityValidationError(
            f"Hash must be exactly {XXHASH64_HEX_LENGTH} characters, got {len(file_hash)}"
        )
    # Prevents path traversal via malicious hashes
    if not all(c in _VALID_HASH_CHARS for c in file_hash):
        raise exceptions.SecurityValidationError(f"Hash contains invalid characters: {file_hash!r}")
    return cache_dir / file_hash[:2] / file_hash[2:]


def _make_writable_and_retry(func: Callable[[str], object], path: str, exc: BaseException) -> None:
    """onexc handler for rmtree: make parent directory writable before retrying.

    IMPORTANT: We only chmod directories, never files. Files may be hardlinks to
    the cache, and chmod would corrupt the cache's read-only permissions (hardlinks
    share the same inode, so chmod affects both the output file and cache file).

    To delete a file, you only need write permission on the parent directory.
    To delete a directory, you need write permission on the parent directory.
    """
    # Make parent directory writable so we can modify its contents
    parent = os.path.dirname(path)
    if parent:
        try:
            parent_perm = os.lstat(parent).st_mode
            if not (parent_perm & stat.S_IWUSR):
                os.chmod(parent, parent_perm | stat.S_IWUSR)
        except OSError:
            pass  # Best effort - may not own parent

    # For directories only: make writable so we can delete contents
    # Never chmod files - they may be hardlinks to read-only cache
    try:
        st = os.lstat(path)
        if stat.S_ISDIR(st.st_mode) and not (st.st_mode & stat.S_IWUSR):
            os.chmod(path, st.st_mode | stat.S_IWUSR)
    except OSError as chmod_exc:
        if chmod_exc.errno not in (errno.ENOENT, errno.EPERM):
            raise exc from chmod_exc

    func(path)


def _clear_path(path: pathlib.Path) -> None:
    """Remove file, symlink, or directory at path if it exists.

    IMPORTANT: Never chmod files before deletion - they may be hardlinks to the
    cache, and chmod would corrupt the cache's read-only permissions.
    """
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, onexc=_make_writable_and_retry)
    else:
        try:
            path.unlink()
        except PermissionError as perm_err:
            # Make parent directory writable, not the file itself.
            # Files may be hardlinks to read-only cache - chmod would corrupt cache.
            parent = path.parent
            try:
                parent_perm = parent.stat().st_mode
            except FileNotFoundError:
                # Parent was deleted concurrently - re-raise original error
                raise perm_err from None
            if not (parent_perm & stat.S_IWUSR):
                os.chmod(parent, parent_perm | stat.S_IWUSR)
            path.unlink()


def _cleanup_stale_restore_temps(parent_dir: pathlib.Path) -> None:
    """Remove restore temp directories and lock files older than max age."""
    if not parent_dir.exists():
        return

    cutoff = time.time() - _RESTORE_TEMP_MAX_AGE
    for entry in os.scandir(parent_dir):
        # Handle lock files (regular files)
        if entry.is_file(follow_symlinks=False) and entry.name.startswith(_RESTORE_LOCK_PREFIX):
            try:
                if entry.stat(follow_symlinks=False).st_mtime < cutoff:
                    logger.debug(f"Cleaning up stale lock file: {entry.path}")
                    os.unlink(entry.path)
            except OSError as e:
                logger.debug(f"Failed to clean up lock file {entry.path}: {e}")
            continue

        # Handle temp/backup directories
        # is_dir(follow_symlinks=False) returns False for symlinks, preventing attacks
        if not entry.is_dir(follow_symlinks=False):
            continue
        if not (
            entry.name.startswith(_RESTORE_TEMP_PREFIX)
            or entry.name.startswith(_BACKUP_TEMP_PREFIX)
        ):
            continue
        # Use mtime for age check - simpler and more reliable than parsing
        try:
            if entry.stat(follow_symlinks=False).st_mtime < cutoff:
                logger.debug(f"Cleaning up stale temp: {entry.path}")
                _clear_path(pathlib.Path(entry.path))
        except OSError as e:
            logger.debug(f"Failed to clean up temp {entry.path}: {e}")


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


FALLBACK_ERRNO = frozenset({errno.EXDEV, errno.EPERM, errno.EACCES})


def _checkout_from_cache(
    path: pathlib.Path,
    cache_path: pathlib.Path,
    checkout_mode: CheckoutMode,
    *,
    executable: bool = False,
) -> None:
    """Create link from workspace path to cache."""
    # Idempotency: skip if already correctly linked
    if checkout_mode == CheckoutMode.SYMLINK and path.is_symlink():
        with contextlib.suppress(OSError):
            if path.resolve() == cache_path.resolve():
                return
    elif checkout_mode == CheckoutMode.HARDLINK and path.exists() and not path.is_symlink():
        with contextlib.suppress(OSError):
            if path.stat().st_ino == cache_path.stat().st_ino:
                return

    _clear_path(path)

    if checkout_mode == CheckoutMode.SYMLINK:
        path.symlink_to(cache_path.resolve())
    elif checkout_mode == CheckoutMode.HARDLINK:
        os.link(cache_path, path)
    else:
        shutil.copy2(cache_path, path)
        os.chmod(path, 0o755 if executable else 0o644)


def _checkout_with_fallback(
    path: pathlib.Path,
    cache_path: pathlib.Path,
    checkout_modes: list[CheckoutMode],
    *,
    executable: bool = False,
) -> None:
    """Try each link mode in order until one succeeds."""
    if not checkout_modes:
        raise ValueError("checkout_modes cannot be empty")
    for i, mode in enumerate(checkout_modes):
        try:
            _checkout_from_cache(path, cache_path, mode, executable=executable)
            return
        except OSError as e:
            if e.errno not in FALLBACK_ERRNO or i == len(checkout_modes) - 1:
                raise
            logger.debug(f"Checkout mode {mode.value} failed ({e}), trying next mode")


DEFAULT_CHECKOUT_MODE_ORDER = [CheckoutMode.HARDLINK, CheckoutMode.SYMLINK, CheckoutMode.COPY]


def _resolve_checkout_modes(
    checkout_mode: CheckoutMode | None,
    checkout_modes: list[CheckoutMode] | None,
) -> list[CheckoutMode]:
    """Resolve effective checkout modes from single mode or list."""
    if checkout_mode is not None:
        return [checkout_mode]
    if checkout_modes is not None:
        if not checkout_modes:
            raise ValueError("checkout_modes cannot be empty")
        return checkout_modes
    return DEFAULT_CHECKOUT_MODE_ORDER.copy()


def save_to_cache(
    path: pathlib.Path,
    cache_dir: pathlib.Path,
    state_db: state_mod.StateDB | None = None,
    checkout_mode: CheckoutMode | None = None,
    checkout_modes: list[CheckoutMode] | None = None,
) -> OutputHash:
    """Save file or directory to cache, replace with link, return hash info.

    Args:
        path: File or directory to save
        cache_dir: Cache directory
        state_db: Optional state database for hash caching
        checkout_mode: Single link mode (no fallback). Takes precedence over checkout_modes.
        checkout_modes: Ordered list of link modes to try with fallback on failure.
    """
    effective_modes = _resolve_checkout_modes(checkout_mode, checkout_modes)

    if path.is_dir():
        return _save_directory_to_cache(path, cache_dir, state_db, effective_modes)
    return _save_file_to_cache(path, cache_dir, state_db, effective_modes)


def _save_file_to_cache(
    path: pathlib.Path,
    cache_dir: pathlib.Path,
    state_db: state_mod.StateDB | None,
    checkout_modes: list[CheckoutMode],
) -> FileHash:
    """Save single file to cache."""
    # Idempotency: check if already a valid cache symlink (cheap check first)
    if checkout_modes and checkout_modes[0] == CheckoutMode.SYMLINK:
        existing_hash = _get_symlink_cache_hash(path, cache_dir)
        if existing_hash is not None:
            cache_path = get_cache_path(cache_dir, existing_hash)
            if cache_path.exists():
                return FileHash(hash=existing_hash)

    file_hash = hash_file(path, state_db)
    cache_path = get_cache_path(cache_dir, file_hash)

    copy_to_cache(path, cache_path)
    _checkout_with_fallback(path, cache_path, checkout_modes)

    return FileHash(hash=file_hash)


def _save_directory_to_cache(
    path: pathlib.Path,
    cache_dir: pathlib.Path,
    state_db: state_mod.StateDB | None,
    checkout_modes: list[CheckoutMode],
) -> DirHash:
    """Save directory to cache."""
    # Idempotency check for SYMLINK mode
    if checkout_modes and checkout_modes[0] == CheckoutMode.SYMLINK and path.is_symlink():
        existing_hash = _get_symlink_cache_hash(path, cache_dir)
        if existing_hash is not None:
            cache_dir_path = get_cache_path(cache_dir, existing_hash)
            if cache_dir_path.exists():
                # Already correctly linked - compute manifest from actual content
                _, manifest = hash_directory(path, state_db)
                return DirHash(hash=existing_hash, manifest=manifest)

    tree_hash, manifest = hash_directory(path, state_db)

    # Cache individual files first
    for entry in manifest:
        file_path = path / entry["relpath"]
        cache_path = get_cache_path(cache_dir, entry["hash"])
        copy_to_cache(file_path, cache_path)

    if checkout_modes and checkout_modes[0] == CheckoutMode.SYMLINK:
        # SYMLINK mode: cache entire directory, symlink to it
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
    else:
        # HARDLINK/COPY modes: replace each file with link/copy from cache
        for entry in manifest:
            file_path = path / entry["relpath"]
            cache_path = get_cache_path(cache_dir, entry["hash"])
            _checkout_with_fallback(
                file_path, cache_path, checkout_modes, executable=entry["isexec"]
            )

    return DirHash(hash=tree_hash, manifest=manifest)


def restore_from_cache(
    path: pathlib.Path,
    output_hash: OutputHash,
    cache_dir: pathlib.Path,
    checkout_mode: CheckoutMode | None = None,
    checkout_modes: list[CheckoutMode] | None = None,
) -> bool:
    """Restore file or directory from cache. Returns True if successful.

    Args:
        path: Target path to restore to
        output_hash: Hash info for the cached output
        cache_dir: Cache directory
        checkout_mode: Single link mode (no fallback). Takes precedence over checkout_modes.
        checkout_modes: Ordered list of link modes to try with fallback on failure.
    """
    if output_hash is None:
        return False

    effective_modes = _resolve_checkout_modes(checkout_mode, checkout_modes)

    if "manifest" in output_hash:
        return _restore_directory_from_cache(path, output_hash, cache_dir, effective_modes)
    return _restore_file_from_cache(path, output_hash, cache_dir, effective_modes)


def _restore_file_from_cache(
    path: pathlib.Path,
    output_hash: FileHash,
    cache_dir: pathlib.Path,
    checkout_modes: list[CheckoutMode],
) -> bool:
    """Restore single file from cache."""
    cache_path = get_cache_path(cache_dir, output_hash["hash"])
    if not cache_path.exists():
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    _checkout_with_fallback(path, cache_path, checkout_modes)
    return True


def _restore_directory_from_cache(
    path: pathlib.Path,
    output_hash: DirHash,
    cache_dir: pathlib.Path,
    checkout_modes: list[CheckoutMode],
) -> bool:
    """Restore directory from cache atomically with file locking."""
    cache_dir_path = get_cache_path(cache_dir, output_hash["hash"])

    # Validate ALL manifest entries exist before writing anything (security)
    # This also handles the SYMLINK fast path - if cache_dir_path doesn't exist,
    # we need to check individual files
    symlink_fast_path = checkout_modes[0] == CheckoutMode.SYMLINK and cache_dir_path.exists()
    if not symlink_fast_path:
        for entry in output_hash["manifest"]:
            file_cache_path = get_cache_path(cache_dir, entry["hash"])
            if not file_cache_path.exists():
                return False

    # Ensure parent exists and clean up stale temps
    path.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_restore_temps(path.parent)

    # Acquire lock for this specific path to prevent concurrent restore races
    lock_path = path.parent / _get_lock_filename(path.name)
    lock_fd: int | None = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        # SYMLINK fast path: create symlink at temp location, then atomic rename
        if symlink_fast_path:
            temp_symlink = path.parent / f"{_RESTORE_TEMP_PREFIX}symlink_{secrets.token_hex(4)}"
            try:
                # Test symlink creation at temp location first
                temp_symlink.symlink_to(cache_dir_path.resolve())
                # Atomic swap: clear target then rename temp symlink
                _clear_path(path)
                temp_symlink.replace(path)
                return True
            except OSError as e:
                logger.debug(f"Symlink creation failed, will try file-by-file: {e}")
                if temp_symlink.is_symlink():
                    temp_symlink.unlink()
                # Re-validate file entries before falling back to file-by-file
                for entry in output_hash["manifest"]:
                    file_cache_path = get_cache_path(cache_dir, entry["hash"])
                    if not file_cache_path.exists():
                        return False

        temp_path = pathlib.Path(tempfile.mkdtemp(prefix=_RESTORE_TEMP_PREFIX, dir=path.parent))
        backup_path = (
            path.parent / f"{_BACKUP_TEMP_PREFIX}{path.name}.{os.getpid()}.{secrets.token_hex(4)}"
        )

        # Track state for proper cleanup
        swap_completed = False
        backup_restore_failed = False

        try:
            resolved_temp = temp_path.resolve()

            # Restore all files to temp directory
            for file_count, entry in enumerate(output_hash["manifest"], start=1):
                file_cache_path = get_cache_path(cache_dir, entry["hash"])
                file_path = temp_path / entry["relpath"]

                # Validate no path traversal (e.g., "../../../etc/passwd")
                if not file_path.resolve().is_relative_to(resolved_temp):
                    raise exceptions.SecurityValidationError(
                        f"Manifest contains path traversal: {entry['relpath']!r}"
                    )
                file_path.parent.mkdir(parents=True, exist_ok=True)
                _checkout_with_fallback(
                    file_path, file_cache_path, checkout_modes, executable=entry["isexec"]
                )
                # Keep mtime fresh to prevent cleanup during long restores
                if file_count % 100 == 0:
                    try:
                        os.utime(temp_path, None)
                    except OSError as e:
                        logger.debug(f"mtime refresh failed (may be NFS with root_squash): {e}")

            # Final mtime touch after restore completes
            try:
                os.utime(temp_path, None)
            except OSError as e:
                logger.debug(f"mtime refresh failed (may be NFS with root_squash): {e}")

            # Ensure directories writable if COPY mode was used
            # Use is_dir with follow_symlinks=False to avoid traversing symlinks
            if CheckoutMode.COPY in checkout_modes:
                try:
                    os.chmod(temp_path, 0o755)
                    for item in temp_path.rglob("*"):
                        if item.is_dir() and not item.is_symlink():
                            os.chmod(item, 0o755)
                except OSError as e:
                    logger.debug(f"chmod failed (may be NFS with root_squash): {e}")

            # Atomic swap with rollback capability:
            # - Backup original first so we can restore on failure
            # - Replace atomically so path always has valid content
            # - Clean backup only after successful swap
            try:
                if path.exists() or path.is_symlink():
                    path.rename(backup_path)
            except FileNotFoundError:
                logger.debug("Path disappeared between check and rename, proceeding")

            temp_path.replace(path)
            swap_completed = True

            try:
                _clear_path(backup_path)
            except OSError as e:
                logger.debug(f"Backup cleanup failed (non-critical): {e}")

            return True

        except BaseException:
            # Restore original if we had one and swap didn't complete
            if backup_path.exists() and not path.exists():
                try:
                    backup_path.rename(path)
                except OSError as e:
                    logger.debug(f"Failed to restore backup: {e}")
                    backup_restore_failed = True
            raise
        finally:
            # Clean up temp directory on any exit
            if temp_path.exists():
                _clear_path(temp_path)
            # Only clean up backup if swap completed successfully
            # Don't delete backup if restoration failed - that's the user's data!
            if backup_path.exists() and swap_completed and not backup_restore_failed:
                _clear_path(backup_path)

    finally:
        # Release lock and clean up lock file
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError as e:
                logger.debug(f"Failed to unlock: {e}")
            try:
                os.close(lock_fd)
            except OSError as e:
                logger.debug(f"Failed to close lock fd: {e}")
            # Don't delete lock file here - let cleanup handle stale ones


def remove_output(path: pathlib.Path) -> None:
    """Remove output file or directory before execution."""
    _clear_path(path)


def protect(path: pathlib.Path) -> None:
    """Make file read-only (mode 0o444)."""
    os.chmod(path, 0o444)


def unprotect(path: pathlib.Path) -> None:
    """Restore write permission to file.

    WARNING: Do not use on files that might be hardlinks to the cache.
    Hardlinks share the same inode, so chmod would also change the cache
    file's permissions, corrupting its read-only invariant.
    """
    current = path.stat().st_mode
    os.chmod(path, current | stat.S_IWUSR)
