"""SQLite-backed cache for file metadata to hash mappings."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    import os
    import pathlib


class StateDB:
    """SQLite cache of (inode, mtime, size) -> hash to skip rehashing unchanged files."""

    def __init__(self, db_path: pathlib.Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS state (
                path TEXT PRIMARY KEY,
                mtime REAL,
                size INTEGER,
                inode INTEGER,
                hash TEXT
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_state_hash ON state(hash)")
        self._conn.commit()

    def get(self, path: pathlib.Path, fs_stat: os.stat_result) -> str | None:
        """Return cached hash if file metadata matches, else None."""
        abs_path = str(path.resolve())
        cursor = self._conn.execute(
            "SELECT hash FROM state WHERE path = ? AND mtime = ? AND size = ? AND inode = ?",
            (abs_path, fs_stat.st_mtime, fs_stat.st_size, fs_stat.st_ino),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def save(self, path: pathlib.Path, fs_stat: os.stat_result, file_hash: str) -> None:
        """Cache file metadata and hash."""
        abs_path = str(path.resolve())
        self._conn.execute(
            """
            INSERT OR REPLACE INTO state (path, mtime, size, inode, hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            (abs_path, fs_stat.st_mtime, fs_stat.st_size, fs_stat.st_ino, file_hash),
        )
        self._conn.commit()

    def save_many(self, entries: list[tuple[pathlib.Path, os.stat_result, str]]) -> None:
        """Batch save multiple entries."""
        data = [(str(p.resolve()), s.st_mtime, s.st_size, s.st_ino, h) for p, s, h in entries]
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO state (path, mtime, size, inode, hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            data,
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
