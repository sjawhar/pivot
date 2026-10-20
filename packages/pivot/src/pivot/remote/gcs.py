"""Google Cloud Storage remote backend (gs://bucket/prefix), on gcloud-aio-storage.

Credentials are Application Default Credentials (``GOOGLE_APPLICATION_CREDENTIALS``, then
``gcloud auth application-default login``, then the GCE metadata server). ``STORAGE_EMULATOR_HOST``
points the backend at an emulator such as fake-gcs-server, which needs no credentials.

No error this module raises or logs contains an HTTP client exception's str or repr, nor chains
one as its cause: aiohttp's repr carries the request headers, including the bearer token. Errors
report the HTTP status and Google's own error message instead (``_describe``).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import os
import random
from typing import TYPE_CHECKING, cast

import aiohttp

from pivot import config, exceptions, project
from pivot.remote import batch, keys
from pivot.remote import config as remote_config

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
    from pathlib import Path

    from gcloud.aio.storage import Storage, StreamResponse

    from pivot.types import TransferResult

logger = logging.getLogger(__name__)

STREAM_CHUNK_SIZE = 8 * 1024 * 1024  # Read size for downloads
STREAM_READ_TIMEOUT = 60  # Seconds to wait for each read from the network
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


class _UploadNotStoredError(Exception):
    """The server's reply to an upload does not show the whole object stored. Retryable."""


def _google_message(exc: aiohttp.ClientResponseError) -> str:
    """Google's own error message from a gcloud-aio error ("<reason>: <JSON body>")."""
    _, _, body = exc.message.partition(": ")
    try:
        parsed = json.loads(body)
        message = parsed["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return exc.message
    return str(message)


def _describe(exc: BaseException) -> str:
    """What failed, for a message: never an HTTP client exception's str or repr."""
    if isinstance(exc, aiohttp.ClientResponseError):
        return f"HTTP {exc.status}: {_google_message(exc)}"
    if isinstance(exc, aiohttp.ClientConnectorError):
        return f"{type(exc).__name__} connecting to {exc.host}:{exc.port}"
    if isinstance(exc, (aiohttp.ClientError, TimeoutError)):
        return type(exc).__name__
    return f"{type(exc).__name__}: {exc}"


def _remote_error(
    exc: aiohttp.ClientError | TimeoutError, action: str, location: str
) -> exceptions.RemoteConnectionError:
    """An actionable error for a failed GCS request to ``action`` (read/list/write) ``location``.

    Raise it ``from None``, so a traceback does not print the client exception either.
    """
    if isinstance(exc, aiohttp.ClientResponseError):
        if exc.status == 401:
            return exceptions.RemoteConnectionError(_AUTH_ERROR_MESSAGE)
        if exc.status == 403:
            return exceptions.RemoteConnectionError(
                f"Permission denied to {action} {location}: {_google_message(exc)}"
            )
        return exceptions.RemoteConnectionError(
            f"GCS {action} error for {location}: {_describe(exc)}"
        )
    return exceptions.RemoteConnectionError(
        f"Cannot reach Google Cloud Storage to {action} {location}: {_describe(exc)}. "
        + "Check network connectivity."
    )


def _emulated() -> bool:
    """Whether gcloud-aio-storage talks to an emulator (STORAGE_EMULATOR_HOST), not GCS."""
    from gcloud.aio.storage import storage as gcloud_storage_impl

    is_dev, _ = gcloud_storage_impl.init_api_root(None)
    return is_dev


def _md5_base64(path: Path) -> str:
    """The file's MD5 as Cloud Storage reports it in ``md5Hash`` (base64 of the digest)."""
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, lambda: hashlib.md5(usedforsecurity=False)).digest()
    return base64.b64encode(digest).decode()


def _not_stored(stored: object, key: str, size: int, md5: str) -> str | None:
    """Why the object resource an upload returned is not ``size`` bytes with ``md5`` at ``key``."""
    if not isinstance(stored, dict):
        return "the server returned no object"
    resource = cast("dict[str, object]", stored)
    if resource.get("name") != key:
        return f"the server stored {resource.get('name')!r}"
    if str(resource.get("size")) != str(size):
        return f"the server stored {resource.get('size')} of {size} bytes"
    if resource.get("md5Hash") != md5:
        return "the stored object's MD5 does not match the local file"
    return None


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
        # connection or read still fails. Passed to every storage request pivot makes; only the
        # token request keeps gcloud-aio's own fixed 10s total, which its API cannot change.
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
            try:
                # Reads GOOGLE_APPLICATION_CREDENTIALS, so a missing or bad key file fails here
                client = gcloud_storage.Storage(session=session)
                # Refresh the token in the request that needs it, never in gcloud-aio's
                # background task: nothing awaits that task, so asyncio would log a failed
                # refresh's exception, whose repr carries the bearer token. At or above
                # force_refresh_after (0.95), the background refresh is unreachable.
                client.token.background_refresh_after = 1.0
                if not _emulated():
                    await client.token.get()
            except Exception as e:  # any failure here leaves no usable credentials
                raise exceptions.RemoteConnectionError(
                    f"{_AUTH_ERROR_MESSAGE} Getting a token failed: {_describe(e)}"
                ) from None
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
            except (aiohttp.ClientConnectionError, TimeoutError, _UploadNotStoredError):
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
            raise _remote_error(e, "read", self._location(key)) from None
        except (aiohttp.ClientError, TimeoutError) as e:
            raise _remote_error(e, "read", self._location(key)) from None
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
                raise _remote_error(e, "list", self._location(prefix)) from None
            for item in page.get("items", []):
                yield str(item["name"])
            token = page.get("nextPageToken")
            if not token:
                return
            params["pageToken"] = str(token)

    async def _upload(self, client: Storage, hash_: str, local_path: Path) -> None:
        """Upload one file in one simple upload (``uploadType=media``), at any size.

        The upload streams the file from disk in a single POST carrying ``self._timeout``. A
        transient failure is retried by ``_with_retries`` as a new upload that reopens the file
        from byte 0; a 4xx other than 408 and 429 fails at once. gcloud-aio's resumable upload
        is never used: with every failure restarting from byte 0 it resumes nothing, and its own
        retry resends the body on the same session, which Cloud Storage refuses with a 400.

        The upload counts only when the object the server returns has this key, the local
        file's size and its MD5; any other reply is retried as a new upload, then fails.
        gcloud-aio returns the JSON of any reply below 400, so a reply without the stored
        object (an empty body, a short object) would otherwise count as uploaded.
        """
        key = keys.hash_to_key(self._prefix, hash_)
        size = local_path.stat().st_size
        md5 = await asyncio.to_thread(_md5_base64, local_path)

        async def attempt() -> None:
            with local_path.open("rb") as f:
                stored: object = await client.upload(
                    self._bucket,
                    key,
                    f,
                    content_type=_OCTET_STREAM,
                    force_resumable_upload=False,
                    timeout=self._timeout,  # pyright: ignore[reportArgumentType] - gcloud-aio passes an aiohttp.ClientTimeout through
                )
            problem = _not_stored(stored, key, size, md5)
            if problem is not None:
                raise _UploadNotStoredError(problem)

        try:
            await self._with_retries(attempt)
        except _UploadNotStoredError as e:
            raise exceptions.RemoteConnectionError(
                f"Upload to {self._location(key)} did not complete: {e}"
            ) from None
        except (aiohttp.ClientError, TimeoutError) as e:
            raise _remote_error(e, "write", self._location(key)) from None

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
                ) from None
            raise _remote_error(e, "read", self._location(key)) from None
        except (aiohttp.ClientError, TimeoutError) as e:
            raise _remote_error(e, "read", self._location(key)) from None
        fd = os.open(dest, os.O_WRONLY | os.O_TRUNC)
        try:
            async with stream:
                actual_hash = await _stream_download_to_fd(stream, fd)
        except aiohttp.ClientError as e:  # the body broke off mid-download
            raise _remote_error(e, "read", self._location(key)) from None
        finally:
            with contextlib.suppress(OSError):
                os.close(fd)
        keys.check_digest(
            expected_hash,
            actual_hash,
            self._location(key),
            state_dir=project.get_state_dir(),
        )

    async def exists(self, cache_hash: str) -> bool:
        """Check if hash exists on remote via an object metadata request."""
        keys.validate_hash(cache_hash)
        async with self._client() as client:
            return await self._exists(client, cache_hash)

    async def bulk_exists(
        self, hashes: list[str], concurrency: int = batch.DEFAULT_CONCURRENCY
    ) -> dict[str, bool]:
        """Check which hashes exist on remote.

        Small batches ask for each object's metadata in parallel; large ones list by 2-char
        prefix, one request per 1000 objects (``batch.bulk_exists``).
        """
        return await batch.bulk_exists(
            hashes,
            concurrency,
            open_client=self._client,
            exists_one=self._exists,
            list_names=self._iter_keys,
            prefix=self._prefix,
            label="GCS",
            metric="gcs.bulk_exists",
        )

    async def iter_hashes(self) -> AsyncIterator[str]:
        """Iterate over all cache hashes on remote (memory-efficient streaming)."""
        async with self._client() as client:
            names = self._iter_keys(client, f"{self._prefix}files/")
            async for hash_ in batch.cache_hashes(names, self._prefix):
                yield hash_

    async def list_hashes(self) -> set[str]:
        """List all cache hashes on remote (collects into memory)."""
        return await batch.collect_hashes(self.iter_hashes(), "gcs.list_hashes")

    async def upload_file(self, local_path: Path, cache_hash: str) -> None:
        """Upload a single file to remote, streamed from disk."""
        keys.validate_hash(cache_hash)
        async with self._client() as client:
            await self._upload(client, cache_hash, local_path)

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
        concurrency: int = batch.DEFAULT_CONCURRENCY,
        callback: Callable[[int, int, str], None] | None = None,
    ) -> list[TransferResult]:
        """Upload multiple files in parallel; a failed file is reported, never raised.

        Missing or unusable credentials are the exception: they raise one RemoteConnectionError
        before any transfer starts, rather than a failed result per file.
        """
        return await batch.transfer_batch(
            [(h, p) for p, h in items],
            concurrency,
            callback,
            open_client=self._client,
            transfer_one=self._upload,
            metric="gcs.upload_batch",
        )

    async def download_batch(
        self,
        items: list[tuple[str, Path]],
        concurrency: int = batch.DEFAULT_CONCURRENCY,
        callback: Callable[[int, int, str], None] | None = None,
        *,
        readonly: bool = False,
    ) -> list[TransferResult]:
        """Download multiple cache entries in parallel with atomic writes and streaming.

        A destination that already exists is kept, never replaced: destinations are
        content-addressed cache entries that other processes may be hardlinking. A failed
        download is reported in its result, never raised, except that missing or unusable
        credentials raise one RemoteConnectionError before any transfer starts.

        Args:
            readonly: If True, set file permissions to 0o444 (for cache files).
        """

        async def download_one(client: Storage, hash_: str, local_path: Path) -> None:
            key = keys.hash_to_key(self._prefix, hash_)
            await keys.atomic_download(
                local_path,
                lambda dest: self._download_verified(client, key, dest, hash_),
                overwrite=False,
                readonly=readonly,
            )

        return await batch.transfer_batch(
            items,
            concurrency,
            callback,
            open_client=self._client,
            transfer_one=download_one,
            metric="gcs.download_batch",
        )
