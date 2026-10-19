"""Google Cloud Storage remote backend (gs://bucket/prefix), on gcloud-aio-storage.

Credentials are Application Default Credentials (``GOOGLE_APPLICATION_CREDENTIALS``, then
``gcloud auth application-default login``, then the GCE metadata server). ``STORAGE_EMULATOR_HOST``
points the backend at an emulator such as fake-gcs-server, which needs no credentials.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
from collections import defaultdict
from typing import TYPE_CHECKING

import aiohttp

from pivot import config, exceptions, metrics
from pivot.remote import config as remote_config
from pivot.remote import keys
from pivot.types import TransferResult

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
    from pathlib import Path

    from gcloud.aio.storage import Storage, StreamResponse

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 20
STREAM_CHUNK_SIZE = 8 * 1024 * 1024  # Read size for downloads; larger uploads go resumable
STREAM_READ_TIMEOUT = 60  # Seconds to wait for each read from the network
# Threshold for listing by prefix instead of one metadata request per hash in bulk_exists
BULK_EXISTS_LIST_THRESHOLD = 50
LIST_PAGE_SIZE = 1000  # Objects per listing page (the GCS maximum)

# Statuses that may succeed on retry: request timeout, rate limit, and server-side failures
_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_RETRY_BASE_DELAY = 0.5
_RETRY_MAX_DELAY = 20.0

_OCTET_STREAM = "application/octet-stream"

_AUTH_ERROR_MESSAGE = (
    "Google Cloud credentials are missing, expired, or invalid. Run "
    "'gcloud auth application-default login' to refresh Application Default Credentials, "
    "or set GOOGLE_APPLICATION_CREDENTIALS to a service account key file."
)


def _google_message(exc: aiohttp.ClientResponseError) -> str:
    """Google's own error message from a gcloud-aio error ("<reason>: <JSON body>")."""
    _, _, body = exc.message.partition(": ")
    try:
        parsed = json.loads(body)
        message = parsed["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return exc.message
    return str(message)


def _remote_error(
    exc: aiohttp.ClientError | TimeoutError, action: str, location: str
) -> exceptions.RemoteConnectionError:
    """An actionable error for a failed GCS request to ``action`` (read/list/write) ``location``."""
    if isinstance(exc, aiohttp.ClientResponseError):
        if exc.status == 401:
            return exceptions.RemoteConnectionError(_AUTH_ERROR_MESSAGE)
        if exc.status == 403:
            return exceptions.RemoteConnectionError(
                f"Permission denied to {action} {location}: {_google_message(exc)}"
            )
        return exceptions.RemoteConnectionError(
            f"GCS {action} error for {location}: HTTP {exc.status}: {_google_message(exc)}"
        )
    return exceptions.RemoteConnectionError(
        f"Cannot reach Google Cloud Storage to {action} {location}: {exc!r}. "
        + "Check network connectivity."
    )


def _emulated() -> bool:
    """Whether gcloud-aio-storage talks to an emulator (STORAGE_EMULATOR_HOST), not GCS."""
    from gcloud.aio.storage import storage as gcloud_storage_impl

    is_dev, _ = gcloud_storage_impl.init_api_root(None)
    return is_dev


async def _stream_download_to_fd(stream: StreamResponse, fd: int) -> str:
    """Stream a GCS download to fd and return its xxh64 integrity digest."""
    return await keys.stream_to_fd(
        lambda: asyncio.wait_for(stream.read(STREAM_CHUNK_SIZE), timeout=STREAM_READ_TIMEOUT),
        fd,
    )


class GCSRemote:
    """Async Google Cloud Storage remote backend.

    Each method opens one aiohttp session, shared by every request it makes (a batch's
    concurrent transfers included), and closes it when done. Credentials are fetched once per
    session, before the first request, so a missing login fails before any transfer starts, with
    one message saying how to log in.
    """

    _bucket: str
    _prefix: str
    _retries: int
    _timeout: aiohttp.ClientTimeout

    def __init__(self, url: str) -> None:
        """Initialize GCS remote from gs://bucket/prefix URL."""
        try:
            import gcloud.aio.storage  # noqa: F401  # pyright: ignore[reportUnusedImport] - availability check
        except ModuleNotFoundError:
            raise exceptions.RemoteError(
                "gcloud-aio-storage is required for GCS remote storage. "
                + "Install with: pip install pivot[gcs]"
            ) from None

        scheme, bucket, prefix = remote_config.validate_remote_url(url)
        if scheme is not remote_config.RemoteScheme.GS:
            raise exceptions.InvalidRemoteURLError(
                f"Not a GCS URL: {url}. Expected format: gs://bucket/prefix"
            )
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""
        self._retries = config.get_remote_retries()
        # No whole-request deadline: a large object takes as long as it takes, while a stalled
        # connection or read still fails.
        self._timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=config.get_remote_connect_timeout(),
            sock_read=STREAM_READ_TIMEOUT,
        )

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def url(self) -> str:
        """Normalized GCS URL for identity comparison."""
        if self._prefix:
            return f"gs://{self._bucket}/{self._prefix.rstrip('/')}"
        return f"gs://{self._bucket}"

    def _location(self, key: str) -> str:
        return f"gs://{self._bucket}/{key}"

    @contextlib.asynccontextmanager
    async def _client(self) -> AsyncIterator[Storage]:
        """A gcloud-aio-storage client on one aiohttp session, with credentials in hand."""
        from gcloud.aio import storage as gcloud_storage

        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            client = gcloud_storage.Storage(session=session)
            if not _emulated():
                try:
                    await client.token.get()
                except Exception as e:  # any failure to get a token leaves no usable credentials
                    raise exceptions.RemoteConnectionError(
                        f"{_AUTH_ERROR_MESSAGE} Getting a token failed: {e!r}"
                    ) from e
            yield client

    async def _with_retries[T](self, request: Callable[[], Awaitable[T]]) -> T:
        """Run ``request``, retrying transient failures up to ``remote.retries`` times."""
        attempt = 0
        while True:
            try:
                return await request()
            except aiohttp.ClientResponseError as e:
                if e.status not in _RETRYABLE_STATUSES or attempt >= self._retries:
                    raise
            except (aiohttp.ClientConnectionError, TimeoutError):
                if attempt >= self._retries:
                    raise
            delay = random.uniform(0, min(_RETRY_MAX_DELAY, _RETRY_BASE_DELAY * 2**attempt))
            attempt += 1
            logger.debug("Retrying GCS request (attempt %d) in %.2fs", attempt + 1, delay)
            await asyncio.sleep(delay)

    async def _exists(self, client: Storage, hash_: str) -> bool:
        key = keys.hash_to_key(self._prefix, hash_)
        try:
            await self._with_retries(
                lambda: client.download_metadata(
                    self._bucket,
                    key,
                    timeout=self._timeout,  # pyright: ignore[reportArgumentType] - gcloud-aio passes an aiohttp.ClientTimeout through
                )
            )
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                return False
            raise _remote_error(e, "read", self._location(key)) from e
        except (aiohttp.ClientError, TimeoutError) as e:
            raise _remote_error(e, "read", self._location(key)) from e
        return True

    async def _iter_keys(self, client: Storage, prefix: str) -> AsyncGenerator[str]:
        """Every object name under ``prefix``, one listing page at a time."""
        params = {
            "prefix": prefix,
            "maxResults": str(LIST_PAGE_SIZE),
            "fields": "items(name),nextPageToken",
        }
        while True:
            try:
                page = await self._with_retries(
                    lambda: client.list_objects(
                        self._bucket,
                        params=dict(params),
                        timeout=self._timeout,  # pyright: ignore[reportArgumentType] - gcloud-aio passes an aiohttp.ClientTimeout through
                    )
                )
            except (aiohttp.ClientError, TimeoutError) as e:
                raise _remote_error(e, "list", self._location(prefix)) from e
            for item in page.get("items", []):
                yield str(item["name"])
            token = page.get("nextPageToken")
            if not token:
                return
            params["pageToken"] = str(token)

    async def _upload(self, client: Storage, local_path: Path, hash_: str) -> None:
        """Upload one file: a simple upload up to 8 MiB, a resumable upload above it."""
        key = keys.hash_to_key(self._prefix, hash_)
        resumable = local_path.stat().st_size > STREAM_CHUNK_SIZE

        async def attempt() -> None:
            with local_path.open("rb") as f:
                await client.upload(
                    self._bucket,
                    key,
                    f,
                    content_type=_OCTET_STREAM,
                    force_resumable_upload=resumable,
                    timeout=self._timeout,  # pyright: ignore[reportArgumentType] - gcloud-aio passes an aiohttp.ClientTimeout through
                )

        try:
            await self._with_retries(attempt)
        except (aiohttp.ClientError, TimeoutError) as e:
            raise _remote_error(e, "write", self._location(key)) from e

    async def _download_verified(
        self, client: Storage, key: str, dest: Path, expected_hash: str
    ) -> None:
        """Stream ``key`` into the existing file ``dest`` and check its xxh64 digest."""
        try:
            stream = await self._with_retries(
                lambda: client.download_stream(
                    self._bucket,
                    key,
                    timeout=self._timeout,  # pyright: ignore[reportArgumentType] - gcloud-aio passes an aiohttp.ClientTimeout through
                )
            )
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                raise exceptions.RemoteError(
                    f"Remote cache file not found: {self._location(key)}"
                ) from e
            raise _remote_error(e, "read", self._location(key)) from e
        except (aiohttp.ClientError, TimeoutError) as e:
            raise _remote_error(e, "read", self._location(key)) from e
        fd = os.open(dest, os.O_WRONLY | os.O_TRUNC)
        try:
            async with stream:
                actual_hash = await _stream_download_to_fd(stream, fd)
        finally:
            with contextlib.suppress(OSError):
                os.close(fd)
        keys.check_digest(expected_hash, actual_hash, self._location(key))

    async def exists(self, cache_hash: str) -> bool:
        """Check if hash exists on remote via an object metadata request."""
        keys.validate_hash(cache_hash)
        async with self._client() as client:
            return await self._exists(client, cache_hash)

    async def bulk_exists(
        self, hashes: list[str], concurrency: int = DEFAULT_CONCURRENCY
    ) -> dict[str, bool]:
        """Check which hashes exist on remote.

        For small batches (< BULK_EXISTS_LIST_THRESHOLD), asks for each object's metadata in
        parallel. For large batches, lists by 2-char prefix (one page holds 1000 objects).
        """
        _t = metrics.start()
        if not hashes:
            metrics.end("gcs.bulk_exists", _t)
            return {}

        for h in hashes:
            keys.validate_hash(h)

        try:
            async with self._client() as client:
                if len(hashes) >= BULK_EXISTS_LIST_THRESHOLD:
                    return await self._bulk_exists_via_list(client, hashes, concurrency)
                return await self._bulk_exists_via_head(client, hashes, concurrency)
        finally:
            metrics.end("gcs.bulk_exists", _t)

    async def _bulk_exists_via_head(
        self, client: Storage, hashes: list[str], concurrency: int
    ) -> dict[str, bool]:
        semaphore = asyncio.Semaphore(concurrency)

        async def check_one(hash_: str) -> tuple[str, bool]:
            async with semaphore:
                return hash_, await self._exists(client, hash_)

        results = await asyncio.gather(*[check_one(h) for h in hashes], return_exceptions=True)

        output = dict[str, bool]()
        for result in results:
            if isinstance(result, BaseException):
                raise exceptions.RemoteConnectionError(
                    f"GCS bulk_exists failed: {result}"
                ) from result
            hash_, exists = result
            output[hash_] = exists
        return output

    async def _bulk_exists_via_list(
        self, client: Storage, hashes: list[str], concurrency: int
    ) -> dict[str, bool]:
        by_prefix = defaultdict[str, set[str]](set)
        for h in hashes:
            by_prefix[h[:2]].add(h)

        semaphore = asyncio.Semaphore(concurrency)

        async def list_prefix(prefix: str, wanted: set[str]) -> dict[str, bool]:
            names = self._iter_keys(client, f"{self._prefix}files/{prefix}/")
            async with semaphore, contextlib.aclosing(names):
                found = set[str]()
                async for key in names:
                    hash_ = keys.key_to_hash(self._prefix, key)
                    if hash_ in wanted:
                        found.add(hash_)
                        if found == wanted:
                            break
                return {h: h in found for h in wanted}

        tasks = [list_prefix(prefix, wanted) for prefix, wanted in by_prefix.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = dict[str, bool]()
        for result in results:
            if isinstance(result, BaseException):
                raise exceptions.RemoteConnectionError(
                    f"GCS bulk_exists LIST failed: {result}"
                ) from result
            output.update(result)
        return output

    async def iter_hashes(self) -> AsyncIterator[str]:
        """Iterate over all cache hashes on remote (memory-efficient streaming)."""
        async with self._client() as client:
            async for key in self._iter_keys(client, f"{self._prefix}files/"):
                hash_ = keys.key_to_hash(self._prefix, key)
                if hash_ is not None:
                    yield hash_

    async def list_hashes(self) -> set[str]:
        """List all cache hashes on remote (collects into memory)."""
        _t = metrics.start()
        hashes = set[str]()
        async for hash_ in self.iter_hashes():
            hashes.add(hash_)
        metrics.end("gcs.list_hashes", _t)
        return hashes

    async def upload_file(self, local_path: Path, cache_hash: str) -> None:
        """Upload a single file to remote, streamed from disk."""
        keys.validate_hash(cache_hash)
        async with self._client() as client:
            await self._upload(client, local_path, cache_hash)

    async def download_file(
        self, cache_hash: str, local_path: Path, *, readonly: bool = False
    ) -> None:
        """Download a single file from remote (atomic write via temp file, streamed)."""
        keys.validate_hash(cache_hash)
        key = keys.hash_to_key(self._prefix, cache_hash)
        async with self._client() as client:
            await keys.atomic_download(
                local_path,
                lambda dest: self._download_verified(client, key, dest, cache_hash),
                overwrite=True,
                readonly=readonly,
            )

    async def upload_batch(
        self,
        items: list[tuple[Path, str]],
        concurrency: int = DEFAULT_CONCURRENCY,
        callback: Callable[[int, int, str], None] | None = None,
    ) -> list[TransferResult]:
        """Upload multiple files in parallel; a failed file is reported, never raised."""
        _t = metrics.start()
        if not items:
            metrics.end("gcs.upload_batch", _t)
            return []

        for _, h in items:
            keys.validate_hash(h)

        try:
            semaphore = asyncio.Semaphore(concurrency)
            completed = 0

            async with self._client() as client:

                async def upload_one(local_path: Path, hash_: str) -> TransferResult:
                    nonlocal completed
                    async with semaphore:
                        try:
                            await self._upload(client, local_path, hash_)
                            completed += 1
                            if callback:
                                callback(completed, len(items), local_path.name)
                            return TransferResult(hash=hash_, success=True)
                        except Exception as e:
                            return TransferResult(hash=hash_, success=False, error=str(e))

                return await asyncio.gather(*[upload_one(p, h) for p, h in items])
        finally:
            metrics.end("gcs.upload_batch", _t)

    async def download_batch(
        self,
        items: list[tuple[str, Path]],
        concurrency: int = DEFAULT_CONCURRENCY,
        callback: Callable[[int, int, str], None] | None = None,
        *,
        readonly: bool = False,
    ) -> list[TransferResult]:
        """Download multiple cache entries in parallel with atomic writes and streaming.

        A destination that already exists is kept, never replaced: destinations are
        content-addressed cache entries that other processes may be hardlinking.

        Args:
            readonly: If True, set file permissions to 0o444 (for cache files).
        """
        _t = metrics.start()
        if not items:
            metrics.end("gcs.download_batch", _t)
            return []

        for h, _ in items:
            keys.validate_hash(h)

        try:
            semaphore = asyncio.Semaphore(concurrency)
            completed = 0

            async with self._client() as client:

                async def download_one(hash_: str, local_path: Path) -> TransferResult:
                    nonlocal completed
                    async with semaphore:
                        key = keys.hash_to_key(self._prefix, hash_)
                        try:
                            await keys.atomic_download(
                                local_path,
                                lambda dest: self._download_verified(client, key, dest, hash_),
                                overwrite=False,
                                readonly=readonly,
                            )
                            completed += 1
                            if callback:
                                callback(completed, len(items), local_path.name)
                            return TransferResult(hash=hash_, success=True)
                        except Exception as e:
                            return TransferResult(hash=hash_, success=False, error=str(e))

                return await asyncio.gather(*[download_one(h, p) for h, p in items])
        finally:
            metrics.end("gcs.download_batch", _t)
