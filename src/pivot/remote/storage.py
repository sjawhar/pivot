from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
from typing import TYPE_CHECKING, Protocol, TypedDict

from pivot import config, exceptions, metrics
from pivot.remote import config as remote_config
from pivot.types import TransferResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence
    from pathlib import Path

    import aioboto3
    from aiobotocore.config import AioConfig
    from botocore.exceptions import ClientError
    from types_aiobotocore_s3 import S3Client
    from types_aiobotocore_s3.type_defs import GetObjectOutputTypeDef

logger = logging.getLogger(__name__)

# Constants for S3 operations
DEFAULT_CONCURRENCY = 20
STREAM_CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks for streaming
STREAM_READ_TIMEOUT = 60  # Seconds to wait for each chunk read
# Threshold for using LIST vs HEAD in bulk_exists. HEAD is O(n) requests but lower latency;
# LIST is O(prefixes) requests (max 256 for 2-char hex) but returns up to 1000 keys each.
# At 50 hashes, LIST wins unless hashes are spread across 50+ prefixes (unlikely).
BULK_EXISTS_LIST_THRESHOLD = 50

_HEX_PATTERN = re.compile(r"^[a-f0-9]{16}$")

# Cached S3 config to avoid re-parsing config on every call.
# Safe without locking - asyncio is single-threaded, worst case race just
# creates duplicate config (harmless).
_cached_s3_config: AioConfig | None = None


# =============================================================================
# Remote Fetcher Protocol (for pivot get --rev)
# =============================================================================


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


# =============================================================================
# S3 Remote Storage (for push/pull commands)
# =============================================================================


class _MultipartPart(TypedDict):
    """Part info for S3 multipart upload."""

    PartNumber: int
    ETag: str


def _get_s3_config() -> AioConfig:
    """Get standard S3 client config with retries and timeouts (cached)."""
    global _cached_s3_config
    if _cached_s3_config is None:
        from aiobotocore.config import AioConfig

        _cached_s3_config = AioConfig(
            retries={"max_attempts": config.get_remote_retries()},
            connect_timeout=config.get_remote_connect_timeout(),
            read_timeout=STREAM_READ_TIMEOUT,
        )
    return _cached_s3_config


def _validate_hash(cache_hash: str) -> None:
    """Validate hash is exactly 16 lowercase hex characters (xxhash64 format)."""
    if not _HEX_PATTERN.match(cache_hash):
        raise exceptions.RemoteError(
            f"Invalid cache hash '{cache_hash}': expected 16 lowercase hex characters"
        )


def _is_not_found_error(e: ClientError) -> bool:
    """Check if botocore ClientError is a 404 Not Found."""
    return e.response.get("Error", {}).get("Code") == "404"


def _hash_to_key(prefix: str, hash_: str) -> str:
    """Convert cache hash to S3 key (files/XX/YYYYYYYY...)."""
    return f"{prefix}files/{hash_[:2]}/{hash_[2:]}"


def _key_to_hash(prefix: str, key: str) -> str | None:
    """Extract valid hash from S3 key, or None if not a valid cache file key.

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


async def _stream_download_to_fd(
    response: GetObjectOutputTypeDef,
    fd: int,
) -> None:
    """Stream S3 response body to file descriptor in chunks with timeout."""
    stream = response["Body"]
    try:
        while True:
            chunk: bytes = await asyncio.wait_for(
                stream.read(STREAM_CHUNK_SIZE),
                timeout=STREAM_READ_TIMEOUT,
            )
            if not chunk:
                break
            await _write_all_async(fd, chunk)
    finally:
        stream.close()  # type: ignore[reportUnknownMemberType] - StreamingBody proxies to aiohttp.ClientResponse.close() (sync)


async def _atomic_download(
    s3: S3Client,
    bucket: str,
    key: str,
    local_path: Path,
    *,
    readonly: bool = False,
) -> None:
    """Download S3 object to local path atomically via temp file with streaming.

    Args:
        readonly: If True, set file permissions to 0o444 (for cache files).
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=local_path.parent, prefix=".pivot_download_")
    move_succeeded = False
    try:
        response = await s3.get_object(Bucket=bucket, Key=key)
        await _stream_download_to_fd(response, fd)
        os.close(fd)
        fd = -1  # Mark as closed
        if readonly:
            os.chmod(tmp_path, 0o444)
        shutil.move(tmp_path, local_path)
        move_succeeded = True
    finally:
        if fd >= 0:
            os.close(fd)
        if not move_succeeded and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def _stream_upload(
    s3: S3Client,
    bucket: str,
    key: str,
    local_path: Path,
) -> None:
    """Upload file to S3 with streaming to avoid memory exhaustion on large files."""
    file_size = local_path.stat().st_size

    # For small files (<= 8MB), use simple put_object
    if file_size <= STREAM_CHUNK_SIZE:
        with local_path.open("rb") as f:
            await s3.put_object(Bucket=bucket, Key=key, Body=f.read())
        return

    # For large files, use multipart upload
    mpu = await s3.create_multipart_upload(Bucket=bucket, Key=key)
    upload_id: str = mpu["UploadId"]
    parts = list[_MultipartPart]()

    try:
        with local_path.open("rb") as f:
            part_number = 1
            while True:
                chunk = f.read(STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                part_response = await s3.upload_part(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=chunk,
                )
                parts.append(_MultipartPart(PartNumber=part_number, ETag=part_response["ETag"]))
                part_number += 1

        await s3.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},  # pyright: ignore[reportArgumentType] - _MultipartPart is compatible with CompletedPartTypeDef
        )
    except Exception:
        try:
            await s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        except Exception as abort_error:
            logger.warning(f"Failed to abort multipart upload {upload_id}: {abort_error}")
        raise


class S3Remote:
    """Async S3 remote storage backend.

    Creates one aioboto3 session at init time and reuses it across all methods.
    Each method creates its own S3 client via ``async with self._session.client()``,
    which is lightweight (reuses the session's credential chain) but gets a fresh
    connection pool. Batch methods share a single client across concurrent tasks.
    """

    _bucket: str
    _prefix: str
    _session: aioboto3.Session

    def __init__(self, url: str) -> None:
        """Initialize S3 remote from s3://bucket/prefix URL."""
        try:
            import aioboto3
        except ModuleNotFoundError:
            raise exceptions.RemoteError(
                "aioboto3 is required for S3 remote storage. Install with: pip install pivot[s3]"
            ) from None

        bucket, prefix = remote_config.validate_s3_url(url)
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""
        self._session = aioboto3.Session()

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def prefix(self) -> str:
        return self._prefix

    async def exists(self, cache_hash: str) -> bool:
        """Check if hash exists on remote via HEAD request."""
        from botocore import exceptions as botocore_exc

        _validate_hash(cache_hash)
        async with self._session.client("s3", config=_get_s3_config()) as s3:
            try:
                await s3.head_object(
                    Bucket=self._bucket,
                    Key=_hash_to_key(self._prefix, cache_hash),
                )
                return True
            except botocore_exc.ClientError as e:
                if _is_not_found_error(e):
                    return False
                raise exceptions.RemoteConnectionError(f"S3 error: {e}") from e

    async def bulk_exists(
        self, hashes: list[str], concurrency: int = DEFAULT_CONCURRENCY
    ) -> dict[str, bool]:
        """Check which hashes exist on remote.

        For small batches (< BULK_EXISTS_LIST_THRESHOLD), uses parallel HEAD requests.
        For large batches, uses LIST by prefix which is more efficient (1 LIST = up to 1000 keys).
        """
        _t = metrics.start()
        if not hashes:
            metrics.end("storage.bulk_exists", _t)
            return {}

        for h in hashes:
            _validate_hash(h)

        try:
            # Single S3 client for all operations - reuses connection pool
            async with self._session.client("s3", config=_get_s3_config()) as s3:
                # For large batches, LIST by prefix is more efficient than HEAD per hash
                if len(hashes) >= BULK_EXISTS_LIST_THRESHOLD:
                    output = await self._bulk_exists_via_list(s3, hashes, concurrency)
                else:
                    output = await self._bulk_exists_via_head(s3, hashes, concurrency)

            return output
        finally:
            metrics.end("storage.bulk_exists", _t)

    async def _bulk_exists_via_head(
        self, s3: S3Client, hashes: list[str], concurrency: int
    ) -> dict[str, bool]:
        """Check existence using parallel HEAD requests (better for small batches)."""
        from botocore import exceptions as botocore_exc

        semaphore = asyncio.Semaphore(concurrency)

        async def check_one(hash_: str) -> tuple[str, bool]:
            async with semaphore:
                try:
                    await s3.head_object(
                        Bucket=self._bucket,
                        Key=_hash_to_key(self._prefix, hash_),
                    )
                    return (hash_, True)
                except botocore_exc.ClientError as e:
                    if _is_not_found_error(e):
                        return (hash_, False)
                    raise exceptions.RemoteConnectionError(f"S3 HEAD error for {hash_}: {e}") from e

        results = await asyncio.gather(*[check_one(h) for h in hashes], return_exceptions=True)

        output = dict[str, bool]()
        for result in results:
            if isinstance(result, BaseException):
                raise exceptions.RemoteConnectionError(
                    f"S3 bulk_exists failed: {result}"
                ) from result
            hash_, exists = result
            output[hash_] = exists

        return output

    async def _bulk_exists_via_list(
        self, s3: S3Client, hashes: list[str], concurrency: int
    ) -> dict[str, bool]:
        """Check existence using LIST by prefix (better for large batches).

        Groups hashes by their 2-char prefix, then lists each prefix bucket.
        One LIST request can return up to 1000 keys, so for N hashes spread
        across P prefixes, we make P requests instead of N.
        """
        from collections import defaultdict

        # Group hashes by prefix
        by_prefix = defaultdict[str, set[str]](set)
        for h in hashes:
            by_prefix[h[:2]].add(h)

        # List each prefix in parallel
        semaphore = asyncio.Semaphore(concurrency)

        async def list_prefix(prefix: str, wanted: set[str]) -> dict[str, bool]:
            async with semaphore:
                found = set[str]()
                s3_prefix = f"{self._prefix}files/{prefix}/"
                paginator = s3.get_paginator("list_objects_v2")
                async for page in paginator.paginate(Bucket=self._bucket, Prefix=s3_prefix):
                    for obj in page.get("Contents", []):
                        key = obj.get("Key")
                        if key is None:
                            continue
                        hash_ = _key_to_hash(self._prefix, key)
                        if hash_ and hash_ in wanted:
                            found.add(hash_)
                            # Early exit if we found all we're looking for
                            if found == wanted:
                                break
                    if found == wanted:
                        break
                return {h: h in found for h in wanted}

        tasks = [list_prefix(prefix, wanted) for prefix, wanted in by_prefix.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = dict[str, bool]()
        for result in results:
            if isinstance(result, BaseException):
                raise exceptions.RemoteConnectionError(
                    f"S3 bulk_exists LIST failed: {result}"
                ) from result
            output.update(result)

        return output

    async def iter_hashes(self) -> AsyncIterator[str]:
        """Iterate over all cache hashes on remote (memory-efficient streaming)."""
        prefix = f"{self._prefix}files/"

        async with self._session.client("s3", config=_get_s3_config()) as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj.get("Key")
                    if key is None:
                        continue
                    hash_ = _key_to_hash(self._prefix, key)
                    if hash_ is not None:
                        yield hash_

    async def list_hashes(self) -> set[str]:
        """List all cache hashes on remote (collects into memory)."""
        _t = metrics.start()
        hashes = set[str]()
        async for hash_ in self.iter_hashes():
            hashes.add(hash_)
        metrics.end("storage.list_hashes", _t)
        return hashes

    async def upload_file(self, local_path: Path, cache_hash: str) -> None:
        """Upload a single file to remote with streaming for large files."""
        _validate_hash(cache_hash)
        async with self._session.client("s3", config=_get_s3_config()) as s3:
            await _stream_upload(
                s3, self._bucket, _hash_to_key(self._prefix, cache_hash), local_path
            )

    async def download_file(
        self, cache_hash: str, local_path: Path, *, readonly: bool = False
    ) -> None:
        """Download a single file from remote (atomic write via temp file, streamed)."""
        _validate_hash(cache_hash)
        async with self._session.client("s3", config=_get_s3_config()) as s3:
            await _atomic_download(
                s3,
                self._bucket,
                _hash_to_key(self._prefix, cache_hash),
                local_path,
                readonly=readonly,
            )

    async def upload_batch(
        self,
        items: list[tuple[Path, str]],
        concurrency: int = DEFAULT_CONCURRENCY,
        callback: Callable[[int], None] | None = None,
    ) -> list[TransferResult]:
        """Upload multiple files in parallel with streaming for large files."""
        _t = metrics.start()
        if not items:
            metrics.end("storage.upload_batch", _t)
            return []

        for _, h in items:
            _validate_hash(h)

        try:
            semaphore = asyncio.Semaphore(concurrency)
            completed = 0

            # Single S3 client for all uploads - reuses connection pool
            async with self._session.client("s3", config=_get_s3_config()) as s3:

                async def upload_one(local_path: Path, hash_: str) -> TransferResult:
                    nonlocal completed
                    async with semaphore:
                        try:
                            await _stream_upload(
                                s3, self._bucket, _hash_to_key(self._prefix, hash_), local_path
                            )
                            completed += 1
                            if callback:
                                callback(completed)
                            return TransferResult(hash=hash_, success=True)
                        except Exception as e:
                            return TransferResult(hash=hash_, success=False, error=str(e))

                results = await asyncio.gather(*[upload_one(p, h) for p, h in items])

            return results
        finally:
            metrics.end("storage.upload_batch", _t)

    async def download_batch(
        self,
        items: list[tuple[str, Path]],
        concurrency: int = DEFAULT_CONCURRENCY,
        callback: Callable[[int], None] | None = None,
        *,
        readonly: bool = False,
    ) -> list[TransferResult]:
        """Download multiple files in parallel with atomic writes and streaming.

        Args:
            readonly: If True, set file permissions to 0o444 (for cache files).
        """
        _t = metrics.start()
        if not items:
            metrics.end("storage.download_batch", _t)
            return []

        for h, _ in items:
            _validate_hash(h)

        try:
            semaphore = asyncio.Semaphore(concurrency)
            completed = 0

            # Single S3 client for all downloads - reuses connection pool
            async with self._session.client("s3", config=_get_s3_config()) as s3:

                async def download_one(hash_: str, local_path: Path) -> TransferResult:
                    nonlocal completed
                    async with semaphore:
                        try:
                            await _atomic_download(
                                s3,
                                self._bucket,
                                _hash_to_key(self._prefix, hash_),
                                local_path,
                                readonly=readonly,
                            )
                            completed += 1
                            if callback:
                                callback(completed)
                            return TransferResult(hash=hash_, success=True)
                        except Exception as e:
                            return TransferResult(hash=hash_, success=False, error=str(e))

                results = await asyncio.gather(*[download_one(h, p) for h, p in items])

            return results
        finally:
            metrics.end("storage.download_batch", _t)
