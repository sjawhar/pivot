"""SQLite-backed cache for file metadata to hash mappings."""

from __future__ import annotations

import sqlite3
import threading
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    import os
    import pathlib

# Timeout for acquiring database lock (seconds)
_DB_TIMEOUT = 5.0


class StateDB:
    """SQLite cache of (inode, mtime, size) -> hash to skip rehashing unchanged files.

    Thread-safe: all operations are protected by a lock.
    """

    _conn: sqlite3.Connection
    _lock: threading.Lock
    _closed: bool

    def __init__(self, db_path: pathlib.Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, timeout=_DB_TIMEOUT, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._closed = False
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS state (
                path TEXT PRIMARY KEY,
                mtime_ns INTEGER,
                size INTEGER,
                inode INTEGER,
                hash TEXT
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_state_hash ON state(hash)")
        self._conn.commit()

    def _check_closed(self) -> None:
        """Raise if database is closed."""
        if self._closed:
            raise RuntimeError("Cannot operate on closed StateDB")

    def get(self, path: pathlib.Path, fs_stat: os.stat_result) -> str | None:
        """Return cached hash if file metadata matches, else None."""
        self._check_closed()
        abs_path = str(path.resolve())
        with self._lock:
            cursor = self._conn.execute(
                "SELECT hash FROM state WHERE path = ? AND mtime_ns = ? AND size = ? AND inode = ?",
                (abs_path, fs_stat.st_mtime_ns, fs_stat.st_size, fs_stat.st_ino),
            )
            row = cursor.fetchone()
        return row[0] if row else None

    def save(self, path: pathlib.Path, fs_stat: os.stat_result, file_hash: str) -> None:
        """Cache file metadata and hash."""
        self._check_closed()
        abs_path = str(path.resolve())
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO state (path, mtime_ns, size, inode, hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    abs_path,
                    fs_stat.st_mtime_ns,
                    fs_stat.st_size,
                    fs_stat.st_ino,
                    file_hash,
                ),
            )
            self._conn.commit()

    def save_many(self, entries: list[tuple[pathlib.Path, os.stat_result, str]]) -> None:
        """Batch save multiple entries with atomic transaction."""
        self._check_closed()
        data = [(str(p.resolve()), s.st_mtime_ns, s.st_size, s.st_ino, h) for p, s, h in entries]
        with self._lock, self._conn:  # Atomic transaction - rolls back on error
            self._conn.executemany(
                """
                    INSERT OR REPLACE INTO state (path, mtime_ns, size, inode, hash)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                data,
            )

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
