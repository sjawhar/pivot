from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from pivot import exceptions

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class RemoteFetcher(Protocol):
    """Protocol for remote cache fetchers."""

    def fetch(self, file_hash: str) -> bytes | None:
        """Fetch file content by hash from remote. Returns None if not found."""
        ...

    def fetch_many(self, file_hashes: Sequence[str]) -> dict[str, bytes]:
        """Fetch multiple files efficiently. Returns dict mapping hash to content."""
        ...

    def exists(self, file_hash: str) -> bool:
        """Check if file exists in remote without downloading."""
        ...


_default_remote: RemoteFetcher | None = None


def set_default_remote(fetcher: RemoteFetcher | None) -> None:
    """Set the default remote fetcher (called during configuration)."""
    global _default_remote
    _default_remote = fetcher


def get_default_remote() -> RemoteFetcher | None:
    """Get the configured default remote fetcher."""
    return _default_remote


def fetch_from_remote(file_hash: str) -> bytes | None:
    """Fetch file from default remote. Returns None if no remote configured or not found."""
    remote = get_default_remote()
    if remote is None:
        logger.debug("No remote configured, skipping remote fetch")
        return None

    try:
        return remote.fetch(file_hash)
    except exceptions.RemoteFetchError as e:
        logger.warning(f"Remote fetch failed for {file_hash[:8]}...: {e!r}")
        return None
