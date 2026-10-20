"""The interface every remote cache backend implements, and the factory that picks one."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pivot.remote import config as remote_config
from pivot.remote import storage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path

    from pivot.types import TransferResult


class RemoteStorage(Protocol):
    """Content-addressed cache storage behind an s3:// or gs:// URL."""

    @property
    def url(self) -> str:
        """Normalized URL for identity comparison."""
        ...

    async def exists(self, cache_hash: str) -> bool: ...

    async def bulk_exists(self, hashes: list[str], concurrency: int = ...) -> dict[str, bool]: ...

    def iter_hashes(self) -> AsyncIterator[str]: ...

    async def list_hashes(self) -> set[str]: ...

    async def upload_file(self, local_path: Path, cache_hash: str) -> None: ...

    async def download_file(
        self, cache_hash: str, local_path: Path, *, readonly: bool = False
    ) -> None: ...

    async def upload_batch(
        self,
        items: list[tuple[Path, str]],
        concurrency: int = ...,
        callback: Callable[[int, int, str], None] | None = None,
    ) -> list[TransferResult]: ...

    async def download_batch(
        self,
        items: list[tuple[str, Path]],
        concurrency: int = ...,
        callback: Callable[[int, int, str], None] | None = None,
        *,
        readonly: bool = False,
    ) -> list[TransferResult]: ...


def open_remote(url: str) -> RemoteStorage:
    """Open the remote backend for ``url``'s scheme (s3:// or gs://)."""
    scheme, _, _ = remote_config.validate_remote_url(url)
    match scheme:
        case remote_config.RemoteScheme.S3:
            return storage.S3Remote(url)
        case remote_config.RemoteScheme.GS:
            # Imported here so importing this module loads neither the GCS backend nor aiohttp
            from pivot.remote import gcs

            return gcs.GCSRemote(url)
