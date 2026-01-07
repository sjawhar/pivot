from __future__ import annotations

import pathlib
import struct
from typing import TYPE_CHECKING, Self

import lmdb

if TYPE_CHECKING:
    import os

# Key prefix for file hash entries (coordinates with #34's gen: and dep: prefixes)
_KEY_PREFIX = b"hash:"

# Default LMDB map size (10GB virtual - grows as needed)
_MAP_SIZE = 10 * 1024 * 1024 * 1024

# LMDB default max key size
_MAX_KEY_SIZE = 511


class StateDBError(Exception):
    """Base exception for StateDB errors."""


class PathTooLongError(StateDBError):
    """Raised when a file path exceeds LMDB's key size limit."""


class DatabaseFullError(StateDBError):
    """Raised when the state database reaches its size limit."""


def _make_key(path: pathlib.Path) -> bytes:
    """Create LMDB key from path."""
    return _KEY_PREFIX + str(path.resolve()).encode()


def _pack_value(mtime_ns: int, size: int, inode: int, hash_hex: str) -> bytes:
    """Pack metadata and hash into binary value."""
    return struct.pack(">QQQ", mtime_ns, size, inode) + hash_hex.encode("ascii")


def _unpack_value(data: bytes) -> tuple[int, int, int, str]:
    """Unpack binary value into metadata and hash."""
    mtime_ns, size, inode = struct.unpack(">QQQ", data[:24])
    hash_hex = data[24:].decode("ascii")
    return mtime_ns, size, inode, hash_hex


def _match_cached_hash(value: bytes | None, fs_stat: os.stat_result) -> str | None:
    """Return cached hash if stored metadata matches fs_stat, else None."""
    if value is None:
        return None
    mtime_ns, size, inode, hash_hex = _unpack_value(value)
    if fs_stat.st_mtime_ns == mtime_ns and fs_stat.st_size == size and fs_stat.st_ino == inode:
        return hash_hex
    return None


class StateDB:
    """LMDB cache of (path, mtime_ns, size, inode) -> hash to skip rehashing unchanged files."""

    _env: lmdb.Environment
    _closed: bool

    def __init__(self, db_path: pathlib.Path) -> None:
        lmdb_path = db_path.parent / "state.lmdb"
        lmdb_path.parent.mkdir(parents=True, exist_ok=True)
        self._env = lmdb.open(str(lmdb_path), map_size=_MAP_SIZE)
        self._closed = False

    def _check_closed(self) -> None:
        """Raise if database is closed."""
        if self._closed:
            raise RuntimeError("Cannot operate on closed StateDB")

    def get(self, path: pathlib.Path, fs_stat: os.stat_result) -> str | None:
        """Return cached hash if file metadata matches, else None."""
        self._check_closed()
        with self._env.begin() as txn:
            value = txn.get(_make_key(path))
        return _match_cached_hash(value, fs_stat)

    def get_many(
        self, items: list[tuple[pathlib.Path, os.stat_result]]
    ) -> dict[pathlib.Path, str | None]:
        """Batch query for multiple files."""
        self._check_closed()
        if not items:
            return {}
        results = dict[pathlib.Path, str | None]()
        with self._env.begin() as txn:
            for path, fs_stat in items:
                value = txn.get(_make_key(path))
                results[path] = _match_cached_hash(value, fs_stat)
        return results

    def save(self, path: pathlib.Path, fs_stat: os.stat_result, file_hash: str) -> None:
        """Cache file metadata and hash."""
        self._check_closed()
        key = _make_key(path)
        if len(key) > _MAX_KEY_SIZE:
            raise PathTooLongError(
                f"Path too long for state cache ({len(key)} bytes, max {_MAX_KEY_SIZE}): {path}"
            )
        value = _pack_value(fs_stat.st_mtime_ns, fs_stat.st_size, fs_stat.st_ino, file_hash)
        try:
            with self._env.begin(write=True) as txn:
                txn.put(key, value)
        except lmdb.MapFullError as e:
            raise DatabaseFullError(
                f"State cache is full ({_MAP_SIZE // (1024**3)}GB limit). Delete .pivot/state.lmdb/ to reset."
            ) from e

    def save_many(self, entries: list[tuple[pathlib.Path, os.stat_result, str]]) -> None:
        """Batch save multiple entries atomically."""
        self._check_closed()
        try:
            with self._env.begin(write=True) as txn:
                for path, fs_stat, file_hash in entries:
                    key = _make_key(path)
                    if len(key) > _MAX_KEY_SIZE:
                        raise PathTooLongError(
                            f"Path too long for state cache ({len(key)} bytes, max {_MAX_KEY_SIZE}): {path}"
                        )
                    value = _pack_value(
                        fs_stat.st_mtime_ns, fs_stat.st_size, fs_stat.st_ino, file_hash
                    )
                    txn.put(key, value)
        except lmdb.MapFullError as e:
            raise DatabaseFullError(
                f"State cache is full ({_MAP_SIZE // (1024**3)}GB limit). Delete .pivot/state.lmdb/ to reset."
            ) from e

    def close(self) -> None:
        """Close the database."""
        if not self._closed:
            self._env.close()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
