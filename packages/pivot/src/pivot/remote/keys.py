"""Content-addressed object layout and integrity checks shared by every remote backend.

A cache object with hash ``abcdef1234567890`` lives at ``<prefix>files/ab/cdef1234567890`` on
every remote, and every backend verifies downloaded bytes with the same streaming digest before
they reach the cache. Changing the hash algorithm or the key layout changes this module only.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import tempfile
from typing import TYPE_CHECKING

import xxhash

from pivot import exceptions, project
from pivot import state_dir as state_dir_mod
from pivot.storage import cache

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

_HEX_PATTERN = re.compile(r"^[a-f0-9]{16}$")


def validate_hash(cache_hash: str) -> None:
    """Validate hash is exactly 16 lowercase hex characters (xxhash64 format)."""
    if not _HEX_PATTERN.match(cache_hash):
        raise exceptions.RemoteError(
            f"Invalid cache hash '{cache_hash}': expected 16 lowercase hex characters"
        )


def hash_to_key(prefix: str, hash_: str) -> str:
    """Convert cache hash to remote object key (files/XX/YYYYYYYY...)."""
    return f"{prefix}files/{hash_[:2]}/{hash_[2:]}"


def key_to_hash(prefix: str, key: str) -> str | None:
    """Extract valid hash from remote object key, or None if not a valid cache file key.

    Returns None if the key doesn't match the expected structure or if the
    reconstructed hash isn't a valid 16-char lowercase hex string.
    """
    expected_prefix = f"{prefix}files/"
    if not key.startswith(expected_prefix):
        return None
    parts = key[len(expected_prefix) :].split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    hash_ = parts[0] + parts[1]
    if not _HEX_PATTERN.match(hash_):
        return None
    return hash_


async def _write_all_async(fd: int, data: bytes) -> None:
    """Write all bytes to fd asynchronously, handling partial writes."""
    written = 0
    while written < len(data):
        n = await asyncio.to_thread(os.write, fd, data[written:])
        if n == 0:
            raise OSError("os.write returned 0")
        written += n


async def stream_to_fd(read_chunk: Callable[[], Awaitable[bytes]], fd: int) -> str:
    """Write every chunk ``read_chunk`` returns to fd until it returns b"".

    Returns the xxh64 integrity digest of everything written.
    """
    hasher = xxhash.xxh64()
    while chunk := await read_chunk():
        hasher.update(chunk)
        await _write_all_async(fd, chunk)
    return hasher.hexdigest()


def check_digest(expected_hash: str, actual_hash: str, location: str) -> None:
    """Raise RemoteError if the downloaded object at ``location`` is not ``expected_hash``.

    The digest detects accidental corruption during storage or transfer. It does not
    authenticate remote content.
    """
    if actual_hash != expected_hash:
        root = project.get_project_root()
        state_db_path = state_dir_mod.state_dir_or_default(root, warn=False) / "state.lmdb"
        if state_db_path.is_relative_to(root):
            state_db_path = state_db_path.relative_to(root)
        raise exceptions.RemoteError(
            f"Hash mismatch: expected {expected_hash}, got {actual_hash} "
            + f"for {location}. Delete {location} from the remote, then run "
            + "`pivot push` on a machine whose cache has a good copy and has not run "
            + "`pivot push`, `pivot pull`, `pivot fetch`, or `pivot status --remote` "
            + "against this remote for this object. Pivot records hashes those commands "
            + "upload, download, or find, then skips recorded hashes. If every machine "
            + f"with a good copy has recorded it, delete `{state_db_path}/` before "
            + "pushing; this also clears run history and local caches."
        )


async def atomic_download(
    local_path: Path,
    download_verified: Callable[[Path], Awaitable[None]],
    *,
    overwrite: bool,
    readonly: bool,
) -> None:
    """Place a verified download at ``local_path`` atomically.

    ``download_verified`` writes the object into the existing temporary file it is given and
    raises unless the bytes match their expected hash; only then does the file reach
    ``local_path``, so a failed or corrupt download leaves nothing behind.

    Args:
        overwrite: Replace an existing file at local_path. When False, local_path is a cache
            object: an existing one is kept (another process may be hardlinking it), and the
            download is staged so a killed one is swept (``cache.staged_cache_file``).
        readonly: If True, set file permissions to 0o444 (for cache files).
    """
    with contextlib.ExitStack() as stack:
        if overwrite:
            state_dir_mod.make_dirs(local_path.parent)
            fd, name = tempfile.mkstemp(dir=local_path.parent, prefix=".pivot_download_")
            os.close(fd)
            tmp_path = local_path.with_name(os.path.basename(name))
            stack.callback(tmp_path.unlink, missing_ok=True)
        else:
            tmp_path = stack.enter_context(cache.staged_cache_file(local_path))
        await download_verified(tmp_path)
        if readonly:
            os.chmod(tmp_path, 0o444)
        if overwrite:
            os.replace(tmp_path, local_path)
        else:
            _ = cache.publish_cache_object(tmp_path, local_path)
