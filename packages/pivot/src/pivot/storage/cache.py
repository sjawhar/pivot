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
import posixpath
import secrets
import shutil
import socket
import stat
import tempfile
import time
from typing import TYPE_CHECKING, TypedDict

import xxhash

from pivot import exceptions, metrics
from pivot.config import CheckoutMode as CheckoutMode
from pivot.types import DirHash, DirManifestEntry, FileHash, HashInfo, is_dir_hash

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    from pivot.storage import state as state_mod

CHUNK_SIZE = 1024 * 1024  # 1MB chunks for hashing
MMAP_THRESHOLD = 10 * 1024 * 1024  # 10MB - use mmap for files larger than this
XXHASH64_HEX_LENGTH = 16  # xxhash64 produces 64-bit hash = 16 hex characters

_RESTORE_TEMP_PREFIX = ".pivot_restore_"
_BACKUP_TEMP_PREFIX = ".pivot_backup_"
_RESTORE_LOCK_PREFIX = ".pivot_restore_lock_"
_RESTORE_TEMP_MAX_AGE = 3600  # 1 hour


def _restore_target_token(target_path: pathlib.Path) -> str:
    """Return a stable token identifying the logical workspace target."""
    logical_path = target_path.parent.resolve() / target_path.name
    return hashlib.sha256(str(logical_path).encode()).hexdigest()[:32]


def restore_staging_path(path: pathlib.Path, token: str | None = None) -> pathlib.Path:
    """Return the deterministic staging path for a file restore."""
    return path.parent / f"{_RESTORE_TEMP_PREFIX}{token or _restore_target_token(path)}.file"


def _get_lock_path_from_token(token: str, state_dir: pathlib.Path) -> pathlib.Path:
    """Get a restore lock path from a previously resolved target token."""
    lock_dir = state_dir / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / f"{_RESTORE_LOCK_PREFIX}{token}"


def _get_lock_path(
    target_path: pathlib.Path, state_dir: pathlib.Path, *, token: str | None = None
) -> pathlib.Path:
    """Get lock file path under state_dir/locks/ for a logical restore target.

    The final path component is intentionally not resolved: a restored symlink must
    retain the same lock as the missing workspace path it replaced.
    """
    return _get_lock_path_from_token(token or _restore_target_token(target_path), state_dir)


@contextlib.contextmanager
def restore_lock(path: pathlib.Path, state_dir: pathlib.Path | None) -> Generator[str | None]:
    """Hold the exclusive restore lock for path and yield its target token.

    No lock is taken when state_dir is None. The lock file is never deleted. Unlinking it
    would let a new restorer lock a fresh file while a waiter still holds the old one.
    """
    if state_dir is None:
        yield None
        return
    token = _restore_target_token(path)
    lock_fd = os.open(_get_lock_path(path, state_dir, token=token), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield token
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def atomic_write_file(
    dest: pathlib.Path,
    write_fn: Callable[[int], None],
    mode: int = 0o644,
    *,
    staging_path: pathlib.Path | None = None,
) -> None:
    """Atomically write to dest using a temporary file and replacement.

    A caller holding a restore lock can supply its target-tagged staging path so a
    later restore of the same target cleans up a process killed before replacement.

    Args:
        dest: Target file path.
        write_fn: Function that receives the file descriptor and writes content.
                  MUST close fd (typically via os.fdopen which takes ownership).
        mode: File permissions (default 0o644).
        staging_path: Optional deterministic temporary path beside ``dest``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if staging_path is None:
        fd, tmp_path = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
        tmp = pathlib.Path(tmp_path)
    else:
        tmp = staging_path
        _clear_path(tmp)
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    fd_closed = False
    try:
        write_fn(fd)
        fd_closed = True  # write_fn took ownership via os.fdopen
        os.chmod(tmp, mode)
        tmp.replace(dest)
    except BaseException:
        _clear_path(tmp)
        raise
    finally:
        # Only close fd if write_fn didn't (e.g., exception before os.fdopen)
        if not fd_closed:
            with contextlib.suppress(OSError):
                os.close(fd)


def hash_file(
    path: pathlib.Path, state_db: state_mod.StateDB | None = None
) -> tuple[str, os.stat_result]:
    """Compute xxhash64 of file contents, using state cache if available.

    Returns:
        Tuple of (file_hash, file_stat) where file_hash is the xxhash64 hex digest
        and file_stat is the os.stat_result for the file.
    """
    file_stat = path.stat()

    if state_db is not None:
        cached = state_db.get(path, file_stat)
        if cached is not None:
            return cached, file_stat

    _t = metrics.start()
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
    metrics.end("cache.hash_file", _t)

    if state_db is not None and not state_db.readonly:
        state_db.save(path, file_stat, file_hash)

    return file_hash, file_stat


def _cache_object_key(path: pathlib.Path, cache_dir: pathlib.Path | None) -> str | None:
    """Return the cache key a symlink resolves to, if its target is an object in ``cache_dir``.

    Cache objects live at ``<cache_dir>/XX/YYYY...`` named by their content hash; a file-by-file
    symlink checkout restores each directory member as a link to one. A link anywhere else,
    even to a hash-named file, is a user symlink.
    """
    if cache_dir is None:
        return None
    key = _get_symlink_cache_hash(path, cache_dir)
    if key is None or not is_valid_hash(key) or not path.is_file():
        return None
    return key


def _scandir_recursive(
    path: pathlib.Path,
    cache_dir: pathlib.Path | None,
    left_out: list[pathlib.Path],
    dirs: list[pathlib.Path],
) -> Generator[os.DirEntry[str]]:
    """Yield every file under `path` recursively, using os.scandir() for efficiency.

    A declared directory artifact is hashed and cached whole: no name is ignored, so a change
    to any file in a directory input reruns its consumers, and every file in a directory output
    survives cache and restore. Ignore rules belong to code fingerprinting and watch mode.
    A symlink to an object in ``cache_dir`` (left by a symlink checkout) counts as the file it
    points to; any other symlink is not stored and is appended to ``left_out``.
    DirEntry objects cache stat results, avoiding redundant syscalls.
    PermissionError propagates to caller - partial hashes would be incorrect.

    Args:
        path: Directory to scan
        cache_dir: Cache object root whose links count as content, or None for no cache
        left_out: Collects the symlinks that are not part of the artifact
        dirs: Collects every directory under ``path`` (symlinked directories are not followed)
    """
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_symlink():
                if _cache_object_key(pathlib.Path(entry.path), cache_dir) is not None:
                    yield entry
                else:
                    left_out.append(pathlib.Path(entry.path))
            elif entry.is_file():
                yield entry
            elif entry.is_dir():
                dirs.append(pathlib.Path(entry.path))
                yield from _scandir_recursive(pathlib.Path(entry.path), cache_dir, left_out, dirs)


def _parent_dirs(relpaths: Iterable[str]) -> set[str]:
    """Every directory above each relpath, short of the root, as relpaths."""
    return {
        str(parent) for relpath in relpaths for parent in pathlib.PurePath(relpath).parents[:-1]
    }


def _recorded_empty_dirs(output_hash: DirHash) -> list[str]:
    return output_hash["empty_dirs"] if "empty_dirs" in output_hash else []


def _restored_dirs(output_hash: DirHash) -> set[str]:
    """Every directory a restore of ``output_hash`` creates under its root, as relpaths."""
    empty_dirs = _recorded_empty_dirs(output_hash)
    file_parents = _parent_dirs(entry["relpath"] for entry in output_hash["manifest"])
    return file_parents | _parent_dirs(empty_dirs) | set(empty_dirs)


def _empty_dir_path(root: pathlib.Path, relpath: str) -> pathlib.Path:
    """Return where the recorded empty directory ``relpath`` belongs under ``root``.

    ``relpath`` may come straight from a lock file, so one leading outside ``root`` is refused,
    as the file-by-file restore refuses a manifest entry's.
    """
    target = root / relpath
    if not target.resolve().is_relative_to(root.resolve()):
        raise exceptions.SecurityValidationError(f"Manifest contains path traversal: {relpath!r}")
    return target


def _warn_left_out_symlinks(path: pathlib.Path, left_out: list[pathlib.Path]) -> None:
    """Log one warning naming every symlink a directory artifact does not store."""
    absolute = path.absolute()
    cwd = pathlib.Path.cwd()
    shown = absolute.relative_to(cwd) if absolute.is_relative_to(cwd) else path
    names = ", ".join(sorted(str(link.relative_to(path)) for link in left_out))
    one = len(left_out) == 1
    logger.warning(
        "Directory artifact %s leaves out %s %s (a symlink inside a directory artifact is not "
        + "stored); replace %s with a real file or directory",
        shown,
        "symlink" if one else "symlinks",
        names,
        "it" if one else "each",
    )


def hash_directory(
    path: pathlib.Path,
    state_db: state_mod.StateDB | None = None,
    *,
    cache_dir: pathlib.Path | None = None,
    warn_left_out_symlinks: bool = False,
) -> DirHash:
    """Compute the tree hash of a directory, returning it with the manifest.

    The hash covers the manifest and the directory's empty directories (see ``DirHash``). With
    no empty directories it hashes the manifest alone, so recording them changed no hash of a
    directory that holds none.

    Symlink handling:
    - A symlink to an object in ``cache_dir`` is hashed by its target's content under its own
      relpath; the content must match the object's name
    - Any other symlink inside the directory is left out of the manifest
    - Base path may be symlinked (resolved for consistency)

    Note: For portability, paths are stored as normalized (symlinks preserved)
    in lock files, but resolved here for consistent hashing.

    Args:
        path: Directory to hash
        state_db: Optional StateDB for caching file hashes
        cache_dir: The configured cache's object root (``<cache.dir>/files``). Pass it
            wherever one exists, so recording and re-hashing agree on symlink checkouts;
            None means no cache, and then no symlink is content.
        warn_left_out_symlinks: Log one warning naming the symlinks left out. Set where a
            manifest is recorded (a stage output, a tracked directory), not on every re-hash.
    """
    _t = metrics.start()
    manifest = list[DirManifestEntry]()
    left_out = list[pathlib.Path]()
    dirs = list[pathlib.Path]()
    resolved_base = path.resolve()

    for entry in sorted(_scandir_recursive(path, cache_dir, left_out, dirs), key=lambda e: e.path):
        file_path = pathlib.Path(entry.path)
        # Verify a regular file is still within the directory (paranoid check). A cache-object
        # link resolves into the cache by design.
        if not entry.is_symlink() and not file_path.resolve().is_relative_to(resolved_base):
            continue
        try:
            rel = file_path.relative_to(path)
            file_stat = entry.stat(follow_symlinks=True)
            file_hash, _ = hash_file(file_path, state_db)
            if entry.is_symlink() and file_hash != _cache_object_key(file_path, cache_dir):
                left_out.append(file_path)
                continue
            manifest_entry: DirManifestEntry = {
                "relpath": str(rel),
                "hash": file_hash,
                "size": file_stat.st_size,
                "isexec": bool(file_stat.st_mode & stat.S_IXUSR),
            }
            manifest.append(manifest_entry)
        except FileNotFoundError:
            continue  # File deleted between scan and hash

    if left_out and warn_left_out_symlinks:
        _warn_left_out_symlinks(path, left_out)

    # A deepest empty directory holds no stored file and no other directory.
    found_dirs = {str(directory.relative_to(path)) for directory in dirs}
    file_dirs = {posixpath.dirname(entry["relpath"]) for entry in manifest}
    has_child = {posixpath.dirname(directory) for directory in found_dirs}
    empty_dirs = sorted(found_dirs - file_dirs - has_child)

    hashed: object = {"empty_dirs": empty_dirs, "manifest": manifest} if empty_dirs else manifest
    hashed_json = json.dumps(hashed, sort_keys=True, separators=(",", ":"))
    output_hash = DirHash(hash=xxhash.xxh64(hashed_json.encode()).hexdigest(), manifest=manifest)
    if empty_dirs:
        output_hash["empty_dirs"] = empty_dirs

    metrics.end("cache.hash_directory", _t)
    return output_hash


_VALID_HASH_CHARS = frozenset("0123456789abcdef")


def is_valid_hash(value: object) -> bool:
    """Whether ``value`` is a cache key: an xxhash64 digest, 16 lowercase hex characters."""
    return (
        isinstance(value, str)
        and len(value) == XXHASH64_HEX_LENGTH
        and all(c in _VALID_HASH_CHARS for c in value)
    )


def get_cache_path(cache_dir: pathlib.Path, file_hash: str) -> pathlib.Path:
    """Get cache path for a hash (XX/XXXX... structure)."""
    if len(file_hash) != XXHASH64_HEX_LENGTH:
        raise exceptions.SecurityValidationError(
            f"Hash must be exactly {XXHASH64_HEX_LENGTH} characters, got {len(file_hash)}"
        )
    # Prevents path traversal via malicious hashes
    if not is_valid_hash(file_hash):
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
    """Remove stale restore and backup artifacts older than max age."""
    cutoff = time.time() - _RESTORE_TEMP_MAX_AGE

    if not parent_dir.exists():
        return

    for entry in os.scandir(parent_dir):
        if entry.name.startswith(_RESTORE_LOCK_PREFIX):
            continue
        if not (
            entry.name.startswith(_RESTORE_TEMP_PREFIX)
            or entry.name.startswith(_BACKUP_TEMP_PREFIX)
        ):
            continue
        try:
            entry_stat = entry.stat(follow_symlinks=False)
            if entry.is_dir(follow_symlinks=False):
                age = entry_stat.st_mtime
            elif stat.S_ISREG(entry_stat.st_mode):
                age = entry_stat.st_ctime
            else:
                continue
            if age < cutoff:
                logger.debug(f"Cleaning up stale temp: {entry.path}")
                _clear_path(pathlib.Path(entry.path))
        except OSError as e:
            logger.debug(f"Failed to clean up temp {entry.path}: {e}")


def _cleanup_fresh_restore_orphans(path: pathlib.Path, token: str) -> None:
    """Remove fresh temp and backup entries left by an interrupted restore of path.

    The caller holds path's restore lock, so entries tagged with token cannot belong to
    a live restore of the same target.
    """
    prefixes = (
        f"{_RESTORE_TEMP_PREFIX}{token}.",
        f"{_BACKUP_TEMP_PREFIX}{token}.",
    )
    for entry in os.scandir(path.parent):
        if entry.name.startswith(prefixes):
            _clear_path(pathlib.Path(entry.path))


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


# link() errors on filesystems without hard links: vfat, exFAT, many SMB and FUSE mounts
# (EPERM, ENOTSUP), and FUSE mounts without a link handler on Linux before 6.15 (ENOSYS).
_NO_HARDLINK_ERRNO = frozenset({errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS})


def publish_cache_object(
    staged: str | os.PathLike[str], cache_path: str | os.PathLike[str]
) -> bool:
    """Publish the complete file ``staged`` as the cache object ``cache_path`` unless an object
    is already published there; return whether ``staged`` was published.

    Objects are content-addressed, so one already there holds the same bytes. It is never
    replaced: a checkout hardlinking it has looked ``cache_path`` up and is about to link the
    inode it found, and Linux refuses that link with ENOENT once a rename over ``cache_path``
    has taken the inode's last name, though the name itself never went missing. A symlink
    checkout reads the object's executable bit through the name, so replacing it with a copy
    of another mode would change that checkout's file. link() publishes without replacing.
    The caller removes ``staged`` afterwards.
    """
    try:
        os.link(staged, cache_path)
    except FileExistsError:
        return False
    except OSError as exc:
        if exc.errno not in _NO_HARDLINK_ERRNO:
            raise
        # No hard links here, so nothing can be hardlinking the object either:
        # rename unless another process already published it.
        if os.path.exists(cache_path):
            return False
        os.replace(staged, cache_path)
    return True


def copy_to_cache(src: pathlib.Path, cache_path: pathlib.Path, *, executable: bool = False) -> None:
    """Atomically copy file to cache with read-only permissions.

    ``executable`` records the source's executable bit on the cache object itself (0o555
    instead of 0o444): a HARDLINK or SYMLINK checkout has no mode of its own, so it reads
    straight through to whatever mode the cache object carries, and only COPY checkout
    re-chmods afterwards (see ``_stage_checkout``).

    The object is published with ``publish_cache_object`` and never changed afterwards, its
    mode included. An object already there is kept whatever executable bit it carries; a
    directory entry needing the other bit is checked out as a copy (``_checkout_with_fallback``).
    """
    if cache_path.exists():
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=cache_path.parent, suffix=".tmp")
    tmp = pathlib.Path(tmp_path)
    try:
        os.close(fd)
        shutil.copy2(src, tmp_path)
        os.chmod(tmp_path, 0o555 if executable else 0o444)
        _ = publish_cache_object(tmp, cache_path)
    finally:
        tmp.unlink(missing_ok=True)


# A link the filesystem cannot make (across devices, refused, unsupported, or past the
# filesystem's link limit) falls through to the next checkout mode.
FALLBACK_ERRNO = _NO_HARDLINK_ERRNO | {errno.EXDEV, errno.EACCES, errno.EMLINK}


def _stage_checkout(
    tmp_path: pathlib.Path,
    cache_path: pathlib.Path,
    checkout_mode: CheckoutMode,
    *,
    executable: bool | None,
) -> None:
    """Create the link or copy of cache_path at tmp_path."""
    if checkout_mode == CheckoutMode.SYMLINK:
        tmp_path.symlink_to(cache_path.resolve())
    elif checkout_mode == CheckoutMode.HARDLINK:
        os.link(cache_path, tmp_path)
    else:
        shutil.copy2(cache_path, tmp_path)
        os.chmod(tmp_path, 0o755 if executable else 0o644)


def _checkout_from_cache(
    path: pathlib.Path,
    cache_path: pathlib.Path,
    checkout_mode: CheckoutMode,
    *,
    executable: bool | None = None,
    staging_path: pathlib.Path | None = None,
) -> None:
    """Create link or copy at path atomically, so readers never see a partial file.

    Args:
        staging_path: Where to build the link or copy before renaming it over path.
            Defaults to a fresh random name beside path.
    """
    # Idempotency: skip if already correctly linked
    if checkout_mode == CheckoutMode.SYMLINK and path.is_symlink():
        with contextlib.suppress(OSError):
            if path.resolve() == cache_path.resolve():
                return
    elif checkout_mode == CheckoutMode.HARDLINK and path.exists() and not path.is_symlink():
        with contextlib.suppress(OSError):
            if path.stat().st_ino == cache_path.stat().st_ino:
                return

    tmp_path = staging_path or path.parent / f"{_RESTORE_TEMP_PREFIX}{secrets.token_hex(8)}"
    try:
        try:
            _stage_checkout(tmp_path, cache_path, checkout_mode, executable=executable)
        except PermissionError:
            # A read-only parent. Replacing an existing entry needs it writable anyway,
            # so make it writable as _clear_path does (never chmod the file: it may be
            # a hardlink to the read-only cache), then retry.
            parent_mode = path.parent.stat().st_mode
            if parent_mode & stat.S_IWUSR or not (path.exists() or path.is_symlink()):
                raise
            os.chmod(path.parent, parent_mode | stat.S_IWUSR)
            _stage_checkout(tmp_path, cache_path, checkout_mode, executable=executable)
        try:
            tmp_path.replace(path)
        except IsADirectoryError:
            _clear_path(path)
            tmp_path.replace(path)
    finally:
        # Also covers a replace() that did nothing: POSIX rename is a no-op when both
        # names are hardlinks to the same inode, leaving tmp_path behind.
        tmp_path.unlink(missing_ok=True)


def _cache_object_isexec(cache_path: pathlib.Path) -> bool | None:
    """Whether the cache object at cache_path is stored executable, or None if absent.

    HARDLINK and SYMLINK checkout have no mode of their own: the workspace file shares the
    cache object's permissions directly, so they can only ever produce whatever executable
    bit the cache object itself carries.
    """
    try:
        return bool(cache_path.stat().st_mode & stat.S_IXUSR)
    except FileNotFoundError:
        return None


def _repair_tree_dir_isexec(cache_dir_path: pathlib.Path, manifest: list[DirManifestEntry]) -> None:
    """Chmod any exec-marked file in a whole-directory SYMLINK cache tree that is missing
    its executable bit, to 0o555.

    A tree an older pivot built (every file chmod'd 0o444 regardless of isexec) keeps being
    reused across an upgrade: its tree hash already incorporates isexec, so it never changes
    and neither _store_directory's ``not cache_dir_path.exists()`` branch nor the
    symlink_fast_path restore would otherwise ever revisit it. Chmod'ing in place is safe
    here, unlike the single-file cache: a tree directory is a private ``copytree`` copy with
    no other hardlinks sharing its files.

    ``manifest`` may come straight from a lock file on the restore path, so each relpath is
    checked the same way the file-by-file restore branch already does: a manifest claiming
    ``../../../outside/file`` must not chmod a file outside the tree.
    """
    resolved_tree = cache_dir_path.resolve()
    for entry in manifest:
        if not entry["isexec"]:
            continue
        f = cache_dir_path / entry["relpath"]
        if not f.resolve().is_relative_to(resolved_tree):
            raise exceptions.SecurityValidationError(
                f"Manifest contains path traversal: {entry['relpath']!r}"
            )
        if _cache_object_isexec(f) is False:
            os.chmod(f, 0o555)


_TREE_LAYOUT_MARKER_SUFFIX = ".layout-checked"


def _tree_layout_marker(cache_dir_path: pathlib.Path) -> pathlib.Path:
    """Return the file beside a whole-directory SYMLINK cache tree that records its layout
    holds exactly what its hash records. The name is never a cache key, so cache scans skip it.
    """
    return cache_dir_path.with_name(cache_dir_path.name + _TREE_LAYOUT_MARKER_SUFFIX)


def _ensure_tree_dir_layout(cache_dir_path: pathlib.Path, output_hash: DirHash) -> None:
    """Repair a whole-directory SYMLINK cache tree's layout once, not on every restore.

    Only this pivot records ``empty_dirs``, so a tree whose hash records them was built by it
    and needs no check. Any other tree is checked once and then marked.
    """
    if "empty_dirs" in output_hash:
        return
    marker = _tree_layout_marker(cache_dir_path)
    if marker.exists():
        return
    _repair_tree_dir_layout(cache_dir_path, output_hash)
    # The marker only saves a walk; a cache the restoring user cannot write still restores.
    with contextlib.suppress(OSError):
        marker.touch()


def _repair_tree_dir_layout(cache_dir_path: pathlib.Path, output_hash: DirHash) -> None:
    """Give a whole-directory SYMLINK cache tree exactly the files and directories
    ``output_hash`` records.

    A tree an older pivot built can hold entries its hash does not record: directories its hash
    left out because they were empty, and copies of the targets of symlinks its hash left out.
    The tree's name never changes, so neither _store_directory's ``not cache_dir_path.exists()``
    branch nor the symlink_fast_path restore would otherwise revisit it, and an output restored
    through it would not hash to the hash it was restored for. Changing the tree in place is
    safe for the reason ``_repair_tree_dir_isexec`` gives.
    """
    found_files, found = _scan_entries(cache_dir_path)
    found_dirs = set(found)
    unlisted = set(found_files) - {entry["relpath"] for entry in output_hash["manifest"]}
    extra_dirs = found_dirs - _restored_dirs(output_hash)
    missing = [
        _empty_dir_path(cache_dir_path, relpath)
        for relpath in _recorded_empty_dirs(output_hash)
        if relpath not in found_dirs
    ]
    if not unlisted and not extra_dirs and not missing:
        return
    modes = dict[pathlib.Path, int]()

    def _make_writable(directory: pathlib.Path) -> None:
        if directory not in modes:
            modes[directory] = directory.stat().st_mode
            os.chmod(directory, modes[directory] | stat.S_IWUSR)

    try:
        for relpath in unlisted:
            target = cache_dir_path / relpath
            _make_writable(target.parent)
            target.unlink(missing_ok=True)
        # Reverse order reaches a directory's children before the directory itself.
        for relpath in sorted(extra_dirs, reverse=True):
            target = cache_dir_path / relpath
            _make_writable(target.parent)
            modes.pop(target, None)
            with contextlib.suppress(FileNotFoundError):  # a concurrent repair got there first
                target.rmdir()
        for target in missing:
            existing = target.parent
            while not existing.exists():
                existing = existing.parent
            _make_writable(existing)
            target.mkdir(parents=True, exist_ok=True)
    finally:
        for directory, mode in modes.items():
            os.chmod(directory, mode)


def _checkout_with_fallback(
    path: pathlib.Path,
    cache_path: pathlib.Path,
    checkout_modes: list[CheckoutMode],
    *,
    executable: bool | None = None,
    staging_path: pathlib.Path | None = None,
    keep_existing_copy: bool = False,
    display_path: pathlib.Path | None = None,
) -> None:
    """Try each link mode in order until one succeeds.

    With ``keep_existing_copy``, reaching COPY while ``path`` still exists leaves it in place:
    the caller guarantees it already holds the cached bytes. A failed link attempt clears
    ``path`` first, so COPY then copies from the cache as usual.

    ``executable`` is None for a single-file output: ``FileHash`` has no isexec field, so a
    single file's mode is never part of any hash and there is no wrong permission to guard
    against. For a directory entry (``executable`` is a bool), a HARDLINK or SYMLINK whose
    cache object's recorded executable bit does not match it is skipped rather than attempted:
    two files with identical content can disagree on the executable bit (the cache key is
    content-only), and a cache object a remote pull populated by hash alone carries no
    executable bit at all. Either way the shared object cannot represent two permissions at
    once, so checkout falls through to the next mode; only COPY makes a private copy and can
    always re-chmod it correctly. Cache objects never change once published, so the bit read
    here is the bit a link or symlink made after it keeps.

    ``display_path`` names the file in a raised error when ``path`` is a restore-only staging
    location (a caller checking out into a scratch directory before an atomic swap); it
    defaults to ``path`` for every other caller.
    """
    if not checkout_modes:
        raise ValueError("checkout_modes cannot be empty")
    display = display_path if display_path is not None else path
    for i, mode in enumerate(checkout_modes):
        is_last = i == len(checkout_modes) - 1
        if keep_existing_copy and mode == CheckoutMode.COPY and path.exists():
            return
        if executable is not None and mode in (CheckoutMode.HARDLINK, CheckoutMode.SYMLINK):
            cached_isexec = _cache_object_isexec(cache_path)
            if cached_isexec is not None and cached_isexec != executable:
                if is_last:
                    cached_state = "executable" if cached_isexec else "non-executable"
                    raise exceptions.CacheModeCollisionError(
                        f"Cache object {cache_path} is recorded {cached_state}, but {display} "
                        + f"needs isexec={executable}."
                    )
                logger.debug(
                    f"Cache object {cache_path} is executable={cached_isexec}, "
                    + f"need {executable}; skipping {mode.value}"
                )
                continue
        try:
            _checkout_from_cache(
                path, cache_path, mode, executable=executable, staging_path=staging_path
            )
            return
        except OSError as e:
            if e.errno not in FALLBACK_ERRNO or is_last:
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


def missing_cache_objects(
    output_hash: HashInfo, cache_dir: pathlib.Path, checkout_modes: list[CheckoutMode]
) -> list[str]:
    """Return hashes required to restore an output that are absent from the cache."""
    if is_dir_hash(output_hash):
        tree_path = get_cache_path(cache_dir, output_hash["hash"])
        if checkout_modes[0] == CheckoutMode.SYMLINK and tree_path.exists():
            return []
        hashes = [entry["hash"] for entry in output_hash["manifest"]]
    else:
        hashes = [output_hash["hash"]]
    return [file_hash for file_hash in hashes if not get_cache_path(cache_dir, file_hash).exists()]


def save_to_cache(
    path: pathlib.Path,
    cache_dir: pathlib.Path,
    state_db: state_mod.StateDB | None = None,
    checkout_mode: CheckoutMode | None = None,
    checkout_modes: list[CheckoutMode] | None = None,
    state_dir: pathlib.Path | None = None,
) -> HashInfo:
    """Save file or directory to cache, replace with link, return hash info.

    Args:
        path: File or directory to save
        cache_dir: Cache directory
        state_db: Optional state database for hash caching
        checkout_mode: Single link mode (no fallback). Takes precedence over checkout_modes.
        checkout_modes: Ordered list of link modes to try with fallback on failure.
        state_dir: State directory for the workspace target's restore lock.
    """
    effective_modes = _resolve_checkout_modes(checkout_mode, checkout_modes)

    if path.is_dir():
        return _save_directory_to_cache(path, cache_dir, state_db, effective_modes, state_dir)
    return _save_file_to_cache(path, cache_dir, state_db, effective_modes, state_dir)


def _save_file_to_cache(
    path: pathlib.Path,
    cache_dir: pathlib.Path,
    state_db: state_mod.StateDB | None,
    checkout_modes: list[CheckoutMode],
    state_dir: pathlib.Path | None,
) -> FileHash:
    """Save single file to cache."""
    # Idempotency: check if already a valid cache symlink (cheap check first)
    if checkout_modes and checkout_modes[0] == CheckoutMode.SYMLINK:
        existing_hash = _get_symlink_cache_hash(path, cache_dir)
        if existing_hash is not None:
            cache_path = get_cache_path(cache_dir, existing_hash)
            if cache_path.exists():
                return FileHash(hash=existing_hash)

    file_hash, _ = hash_file(path, state_db)
    _store_file(
        path,
        file_hash,
        cache_dir,
        checkout_modes,
        keep_existing_copy=False,
        state_dir=state_dir,
    )
    return FileHash(hash=file_hash)


def _store_file(
    path: pathlib.Path,
    file_hash: str,
    cache_dir: pathlib.Path,
    checkout_modes: list[CheckoutMode],
    *,
    keep_existing_copy: bool,
    state_dir: pathlib.Path | None = None,
) -> None:
    cache_path = get_cache_path(cache_dir, file_hash)
    copy_to_cache(path, cache_path)
    with restore_lock(path, state_dir) as lock_token:
        staging_path = restore_staging_path(path, lock_token)
        _clear_path(staging_path)
        _checkout_with_fallback(
            path,
            cache_path,
            checkout_modes,
            staging_path=staging_path,
            keep_existing_copy=keep_existing_copy,
        )


def _save_directory_to_cache(
    path: pathlib.Path,
    cache_dir: pathlib.Path,
    state_db: state_mod.StateDB | None,
    checkout_modes: list[CheckoutMode],
    state_dir: pathlib.Path | None,
) -> DirHash:
    """Save directory to cache."""
    output_hash = hash_directory(path, state_db, cache_dir=cache_dir)
    # Idempotency check for SYMLINK mode: already linked to the tree its content hashes to
    if (
        checkout_modes
        and checkout_modes[0] == CheckoutMode.SYMLINK
        and path.is_symlink()
        and _get_symlink_cache_hash(path, cache_dir) == output_hash["hash"]
        and get_cache_path(cache_dir, output_hash["hash"]).exists()
    ):
        return output_hash

    _store_directory(
        path,
        output_hash,
        cache_dir,
        checkout_modes,
        keep_existing_copy=False,
        state_dir=state_dir,
    )
    return output_hash


_TREE_TEMP_PREFIX = ".pivot_tree_"


def _host_tag() -> str:
    """A short tag for this host, so a publisher never takes another host's temporary tree in a
    shared cache for one of its own."""
    return hashlib.sha256(socket.gethostname().encode()).hexdigest()[:8]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # running, as another user
    return True


def _remove_abandoned_tree_temps(prefix_dir: pathlib.Path) -> None:
    """Remove the temporary trees that publishers on this host, killed before renaming them into
    place, left in prefix_dir. A tree whose publisher is still running is kept."""
    own_host = f"{_TREE_TEMP_PREFIX}{_host_tag()}_"
    for entry in os.scandir(prefix_dir):
        if not entry.name.startswith(own_host):
            continue
        pid = entry.name.removeprefix(own_host).split("_", 1)[0]
        if not pid.isdigit() or _pid_alive(int(pid)):
            continue
        try:
            _clear_path(pathlib.Path(entry.path))
        except OSError as e:  # another publisher removing it at the same time
            logger.debug(f"Failed to remove abandoned cache tree {entry.path}: {e}")


def _publish_tree(
    path: pathlib.Path,
    cache_dir_path: pathlib.Path,
    manifest: list[DirManifestEntry],
    cache_dir: pathlib.Path,
) -> None:
    """Copy directory output ``path`` into the cache as the whole-directory SYMLINK tree
    ``cache_dir_path``.

    A restore symlinks to a tree as soon as its name exists, so the tree is built under a
    temporary name beside it and renamed into place complete. rename() will not replace a
    directory that holds anything, so a tree another writer published first is kept and this
    copy dropped. The temporary name records this process, so a later publish into the same
    prefix directory removes the copy a killed publisher left.
    """

    def _user_symlinks(directory: str, names: list[str]) -> list[str]:
        # Not stored (see _scandir_recursive); copytree would copy each target.
        return [
            name
            for name in names
            if (link := pathlib.Path(directory, name)).is_symlink()
            and _cache_object_key(link, cache_dir) is None
        ]

    cache_dir_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_abandoned_tree_temps(cache_dir_path.parent)
    owner = f"{_TREE_TEMP_PREFIX}{_host_tag()}_{os.getpid()}_"
    staged = pathlib.Path(tempfile.mkdtemp(prefix=owner, dir=cache_dir_path.parent))
    try:
        shutil.copytree(path, staged, ignore=_user_symlinks, dirs_exist_ok=True)
        isexec_by_relpath = {entry["relpath"]: entry["isexec"] for entry in manifest}
        for f in staged.rglob("*"):
            # Skip symlinks to avoid changing permissions on target files outside cache
            if f.is_file() and not f.is_symlink():
                executable = isexec_by_relpath.get(str(f.relative_to(staged)), False)
                os.chmod(f, 0o555 if executable else 0o444)
        os.chmod(staged, 0o555)
        try:
            staged.rename(cache_dir_path)
        except OSError as e:
            if e.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise
            return
        with contextlib.suppress(OSError):
            _tree_layout_marker(cache_dir_path).touch()
    finally:
        _clear_path(staged)


def _store_directory(
    path: pathlib.Path,
    output_hash: DirHash,
    cache_dir: pathlib.Path,
    checkout_modes: list[CheckoutMode],
    *,
    keep_existing_copy: bool,
    state_dir: pathlib.Path | None = None,
) -> None:
    manifest = output_hash["manifest"]
    # Cache individual files first
    for entry in manifest:
        file_path = path / entry["relpath"]
        cache_path = get_cache_path(cache_dir, entry["hash"])
        copy_to_cache(file_path, cache_path, executable=entry["isexec"])

    with restore_lock(path, state_dir):
        if checkout_modes and checkout_modes[0] == CheckoutMode.SYMLINK:
            # SYMLINK mode: cache entire directory, symlink to it
            cache_dir_path = get_cache_path(cache_dir, output_hash["hash"])
            if not cache_dir_path.exists():
                _publish_tree(path, cache_dir_path, manifest, cache_dir)

            # A tree an older pivot built keeps its 0o444-everywhere files, and entries its hash
            # does not record, forever otherwise: its name never changes, so this branch never
            # revisits an existing tree directory above.
            _repair_tree_dir_isexec(cache_dir_path, manifest)
            _ensure_tree_dir_layout(cache_dir_path, output_hash)

            _clear_path(path)
            path.symlink_to(cache_dir_path.resolve())
        else:
            # HARDLINK/COPY modes: replace each file with link/copy from cache
            for entry in manifest:
                file_path = path / entry["relpath"]
                cache_path = get_cache_path(cache_dir, entry["hash"])
                staging_path = restore_staging_path(file_path)
                _clear_path(staging_path)
                _checkout_with_fallback(
                    file_path,
                    cache_path,
                    checkout_modes,
                    executable=entry["isexec"],
                    staging_path=staging_path,
                    keep_existing_copy=keep_existing_copy,
                )


def store_hashed_output(
    path: pathlib.Path,
    output_hash: HashInfo,
    cache_dir: pathlib.Path,
    checkout_modes: list[CheckoutMode] | None = None,
    *,
    state_dir: pathlib.Path | None = None,
) -> None:
    """Put an output whose hash is already known into the cache, then link it per mode.

    ``output_hash`` must be the hash of ``path``'s current contents; nothing is hashed again.
    Each file is copied into the cache once and the workspace path is linked to the cache entry
    with the first checkout mode that works. Reaching COPY leaves the workspace file as it is,
    since it already holds the cached bytes.
    """
    effective_modes = _resolve_checkout_modes(None, checkout_modes)
    if is_dir_hash(output_hash):
        _store_directory(
            path,
            output_hash,
            cache_dir,
            effective_modes,
            keep_existing_copy=True,
            state_dir=state_dir,
        )
    else:
        _store_file(
            path,
            output_hash["hash"],
            cache_dir,
            effective_modes,
            keep_existing_copy=True,
            state_dir=state_dir,
        )


def restore_from_cache(
    path: pathlib.Path,
    output_hash: HashInfo,
    cache_dir: pathlib.Path,
    checkout_mode: CheckoutMode | None = None,
    checkout_modes: list[CheckoutMode] | None = None,
    state_dir: pathlib.Path | None = None,
    *,
    lock_token: str | None = None,
) -> bool:
    """Restore file or directory from cache. Returns True if successful.

    Args:
        path: Target path to restore to
        output_hash: Hash info for the cached output
        cache_dir: Cache directory
        checkout_mode: Single link mode (no fallback). Takes precedence over checkout_modes.
        checkout_modes: Ordered list of link modes to try with fallback on failure.
        state_dir: State directory for restore locks and crash cleanup.
        lock_token: Target token yielded by an already-held restore_lock for path.
    """
    effective_modes = _resolve_checkout_modes(checkout_mode, checkout_modes)
    if lock_token is not None and state_dir is None:
        raise ValueError("state_dir is required when lock_token is set")

    if is_dir_hash(output_hash):
        return _restore_directory_from_cache(
            path,
            output_hash,
            cache_dir,
            effective_modes,
            state_dir,
            lock_token=lock_token,
        )
    return _restore_file_from_cache(
        path, output_hash, cache_dir, effective_modes, state_dir, lock_token=lock_token
    )


def _restore_file_from_cache(
    path: pathlib.Path,
    output_hash: FileHash,
    cache_dir: pathlib.Path,
    checkout_modes: list[CheckoutMode],
    state_dir: pathlib.Path | None = None,
    *,
    lock_token: str | None = None,
) -> bool:
    """Restore single file from cache.

    With a state_dir, holds path's restore lock and stages the file under one name
    tagged with path's token, so the next restore removes what a killed one left.
    """
    cache_path = get_cache_path(cache_dir, output_hash["hash"])
    if not cache_path.exists():
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    if state_dir is None:
        _checkout_with_fallback(path, cache_path, checkout_modes)
        return True

    if lock_token is None:
        with restore_lock(path, state_dir) as held_token:
            return _restore_file_from_cache(
                path,
                output_hash,
                cache_dir,
                checkout_modes,
                state_dir,
                lock_token=held_token,
            )

    staging_path = restore_staging_path(path, lock_token)
    _clear_path(staging_path)
    _checkout_with_fallback(path, cache_path, checkout_modes, staging_path=staging_path)
    return True


def _restore_directory_from_cache(
    path: pathlib.Path,
    output_hash: DirHash,
    cache_dir: pathlib.Path,
    checkout_modes: list[CheckoutMode],
    state_dir: pathlib.Path | None = None,
    *,
    lock_token: str | None = None,
) -> bool:
    """Restore a directory atomically."""
    cache_dir_path = get_cache_path(cache_dir, output_hash["hash"])
    symlink_fast_path = checkout_modes[0] == CheckoutMode.SYMLINK and cache_dir_path.exists()
    if not symlink_fast_path:
        for entry in output_hash["manifest"]:
            if not get_cache_path(cache_dir, entry["hash"]).exists():
                return False

    path.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_restore_temps(path.parent)

    if state_dir is not None and lock_token is None:
        with restore_lock(path, state_dir) as held_token:
            return _restore_directory_from_cache(
                path,
                output_hash,
                cache_dir,
                checkout_modes,
                state_dir,
                lock_token=held_token,
            )

    target_token = lock_token or _restore_target_token(path)
    if lock_token is not None:
        _cleanup_fresh_restore_orphans(path, target_token)
    restore_prefix = f"{_RESTORE_TEMP_PREFIX}{target_token}."

    if symlink_fast_path:
        # Same repairs as _store_directory: a tree an older pivot built keeps its
        # 0o444-everywhere files, and entries its hash does not record, forever otherwise,
        # since this path never rebuilds an existing tree directory.
        _repair_tree_dir_isexec(cache_dir_path, output_hash["manifest"])
        _ensure_tree_dir_layout(cache_dir_path, output_hash)
        temp_symlink = path.parent / f"{restore_prefix}symlink_{secrets.token_hex(4)}"
        try:
            temp_symlink.symlink_to(cache_dir_path.resolve())
            _clear_path(path)
            temp_symlink.replace(path)
            return True
        except OSError as e:
            logger.debug(f"Symlink creation failed, will try file-by-file: {e}")
            if temp_symlink.is_symlink():
                temp_symlink.unlink()
            for entry in output_hash["manifest"]:
                if not get_cache_path(cache_dir, entry["hash"]).exists():
                    return False

    temp_path = pathlib.Path(tempfile.mkdtemp(prefix=restore_prefix, dir=path.parent))
    backup_path = path.parent / f"{_BACKUP_TEMP_PREFIX}{target_token}.{secrets.token_hex(4)}"
    swap_completed = False
    backup_restore_failed = False
    try:
        resolved_temp = temp_path.resolve()
        for file_count, entry in enumerate(output_hash["manifest"], start=1):
            file_cache_path = get_cache_path(cache_dir, entry["hash"])
            file_path = temp_path / entry["relpath"]
            if not file_path.resolve().is_relative_to(resolved_temp):
                raise exceptions.SecurityValidationError(
                    f"Manifest contains path traversal: {entry['relpath']!r}"
                )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            _checkout_with_fallback(
                file_path,
                file_cache_path,
                checkout_modes,
                executable=entry["isexec"],
                display_path=path / entry["relpath"],
            )
            if file_count % 100 == 0:
                try:
                    os.utime(temp_path, None)
                except OSError as e:
                    logger.debug(f"mtime refresh failed (may be NFS with root_squash): {e}")

        for relpath in _recorded_empty_dirs(output_hash):
            _empty_dir_path(temp_path, relpath).mkdir(parents=True, exist_ok=True)

        try:
            os.utime(temp_path, None)
        except OSError as e:
            logger.debug(f"mtime refresh failed (may be NFS with root_squash): {e}")

        if CheckoutMode.COPY in checkout_modes:
            try:
                os.chmod(temp_path, 0o755)
                for item in temp_path.rglob("*"):
                    if item.is_dir() and not item.is_symlink():
                        os.chmod(item, 0o755)
            except OSError as e:
                logger.debug(f"chmod failed (may be NFS with root_squash): {e}")

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
        if backup_path.exists() and not path.exists():
            try:
                backup_path.rename(path)
            except OSError as e:
                logger.debug(f"Failed to restore backup: {e}")
                backup_restore_failed = True
        raise
    finally:
        if temp_path.exists():
            _clear_path(temp_path)
        if backup_path.exists() and swap_completed and not backup_restore_failed:
            _clear_path(backup_path)


def remove_output(path: pathlib.Path) -> None:
    """Remove output file or directory before execution."""
    _clear_path(path)


class MissingRestore(TypedDict):
    """Outcome of restore_missing_in_directory."""

    restored: int  # missing files and empty directories restored in place
    unavailable: int  # missing files whose blob is not in the cache
    local_changes: list[str]  # changed or extra files whose content the cache lacks
    replace: bool  # directory holds only cached content but not the target: replace it whole


def _is_cached(cache_dir: pathlib.Path, content_hash: str | None) -> bool:
    return (
        content_hash is not None
        and is_valid_hash(content_hash)
        and get_cache_path(cache_dir, content_hash).exists()
    )


def _scan_entries(path: pathlib.Path) -> tuple[list[str], list[str]]:
    """Return every file and symlink, and every directory, under ``path`` as relative paths.

    Symlinked directories count as symlinks and are not followed.
    """
    files = list[str]()
    dirs = list[str]()

    def _walk(directory: pathlib.Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                entry_path = pathlib.Path(entry.path)
                relpath = str(entry_path.relative_to(path))
                if entry.is_symlink() or entry.is_file():
                    files.append(relpath)
                elif entry.is_dir():
                    dirs.append(relpath)
                    _walk(entry_path)

    _walk(path)
    return files, dirs


def _is_same_file(file_stat: os.stat_result, cache_path: pathlib.Path) -> bool:
    """Whether file_stat is the cache blob itself (a hardlink checkout). Cache blobs are
    read-only and never rewritten, so sharing the inode proves the content without a read."""
    try:
        blob_stat = cache_path.stat()
    except FileNotFoundError:
        return False
    return (file_stat.st_dev, file_stat.st_ino) == (blob_stat.st_dev, blob_stat.st_ino)


def restore_missing_in_directory(
    path: pathlib.Path,
    output_hash: DirHash,
    cache_dir: pathlib.Path,
    checkout_modes: list[CheckoutMode],
    state_db: state_mod.StateDB | None = None,
) -> MissingRestore:
    """Restore missing target files and empty directories without overwriting local changes.

    When every existing file matches the target, missing files and empty directories are
    restored in place. When a file differs from the target or an extra file or directory
    exists, only a whole-directory swap can restore the target. The swap is safe only when
    every file it removes is in the cache (an extra empty directory removes none); otherwise
    the local changes are returned without writing anything.

    A retained file is read only when nothing else proves its content: a symlink to the
    target's cache path, a hardlink to the target's cache blob, or a matching ``state_db``
    entry each settle it without a read.
    """
    lexical_base = pathlib.Path(os.path.abspath(path))
    resolved_base = path.resolve()
    expected_paths = set[str]()
    missing = list[tuple[DirManifestEntry, pathlib.Path, pathlib.Path]]()
    # relpath -> hash of the content on disk (None when it cannot be in the cache)
    changed = dict[str, str | None]()

    for entry in output_hash["manifest"]:
        relpath = entry["relpath"]
        expected_paths.add(relpath)
        file_path = path / relpath
        lexical_path = pathlib.Path(os.path.abspath(file_path))
        if not lexical_path.is_relative_to(lexical_base):
            raise exceptions.SecurityValidationError(
                f"Manifest contains path traversal: {relpath!r}"
            )

        file_cache_path = get_cache_path(cache_dir, entry["hash"])
        if file_path.is_symlink():
            if not file_path.parent.resolve().is_relative_to(resolved_base):
                raise exceptions.SecurityValidationError(
                    f"Manifest contains path traversal: {relpath!r}"
                )
            if file_path.resolve() == file_cache_path.resolve():
                if not file_cache_path.exists():
                    missing.append((entry, file_path, file_cache_path))
            else:
                changed[relpath] = _get_symlink_cache_hash(file_path, cache_dir)
            continue
        if file_path.exists():
            if not file_path.is_file():
                changed[relpath] = None
                continue
            if _is_same_file(file_path.stat(), file_cache_path):
                continue
            file_hash, _ = hash_file(file_path, state_db)
            if file_hash != entry["hash"]:
                changed[relpath] = file_hash
            continue
        if not file_path.parent.resolve().is_relative_to(resolved_base):
            raise exceptions.SecurityValidationError(
                f"Manifest contains path traversal: {relpath!r}"
            )
        missing.append((entry, file_path, file_cache_path))

    missing_dirs = [
        dir_path
        for dir_path in (
            _empty_dir_path(path, relpath) for relpath in _recorded_empty_dirs(output_hash)
        )
        if not dir_path.exists() and not dir_path.is_symlink()
    ]

    files, dirs = _scan_entries(path)
    for relpath in files:
        if relpath in expected_paths:
            continue
        extra_path = path / relpath
        if extra_path.is_symlink():
            changed[relpath] = _get_symlink_cache_hash(extra_path, cache_dir)
        else:
            changed[relpath], _ = hash_file(extra_path, state_db)
    # A hash pivot 0.3.3 recorded lists no empty directories, so on disk none of them is extra.
    extra_dirs = (
        set(dirs) - _restored_dirs(output_hash) if "empty_dirs" in output_hash else set[str]()
    )

    if changed or extra_dirs:
        # Replacing the directory deletes every file it holds; refuse if any is not cached.
        local_changes = sorted(
            rel for rel, content_hash in changed.items() if not _is_cached(cache_dir, content_hash)
        )
        if local_changes:
            return MissingRestore(
                restored=0, unavailable=0, local_changes=local_changes, replace=False
            )
        return MissingRestore(restored=0, unavailable=0, local_changes=[], replace=True)

    unavailable = sum(1 for _, _, file_cache_path in missing if not file_cache_path.exists())
    if unavailable:
        return MissingRestore(restored=0, unavailable=unavailable, local_changes=[], replace=False)

    for entry, file_path, file_cache_path in missing:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        _checkout_with_fallback(
            file_path, file_cache_path, checkout_modes, executable=entry["isexec"]
        )
    for dir_path in missing_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    return MissingRestore(
        restored=len(missing) + len(missing_dirs), unavailable=0, local_changes=[], replace=False
    )


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
