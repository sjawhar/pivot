from __future__ import annotations

from typing import TYPE_CHECKING

from pivot import remote

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pytest


# =============================================================================
# fetch_from_remote Tests
# =============================================================================


def test_fetch_from_remote_no_remote_configured() -> None:
    """Returns None when no remote is configured."""
    # Ensure no remote is set
    remote.set_default_remote(None)

    result = remote.fetch_from_remote("abc123")

    assert result is None


def test_fetch_from_remote_with_mock_fetcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns content from configured remote fetcher."""

    class MockFetcher:
        def fetch(self, file_hash: str) -> bytes | None:
            if file_hash == "abc123":
                return b"file content"
            return None

        def fetch_many(self, file_hashes: Sequence[str]) -> dict[str, bytes]:
            return {h: b"content" for h in file_hashes if h == "abc123"}

        def exists(self, file_hash: str) -> bool:
            return file_hash == "abc123"

    remote.set_default_remote(MockFetcher())

    try:
        result = remote.fetch_from_remote("abc123")
        assert result == b"file content"

        result_missing = remote.fetch_from_remote("missing")
        assert result_missing is None
    finally:
        remote.set_default_remote(None)


# =============================================================================
# set_default_remote / get_default_remote Tests
# =============================================================================


def test_set_get_default_remote() -> None:
    """Can set and get default remote."""

    class MockFetcher:
        def fetch(self, file_hash: str) -> bytes | None:
            return None

        def fetch_many(self, file_hashes: Sequence[str]) -> dict[str, bytes]:
            return {}

        def exists(self, file_hash: str) -> bool:
            return False

    fetcher = MockFetcher()

    try:
        remote.set_default_remote(fetcher)
        assert remote.get_default_remote() is fetcher

        remote.set_default_remote(None)
        assert remote.get_default_remote() is None
    finally:
        remote.set_default_remote(None)
