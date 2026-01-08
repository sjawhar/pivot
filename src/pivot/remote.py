from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from pivot import exceptions, remote_config
from pivot.types import TransferResult

if TYPE_CHECKING:
    import pathlib
    from collections.abc import AsyncIterator, Callable, Sequence

logger = logging.getLogger(__name__)

# Constants for S3 operations
DEFAULT_CONCURRENCY = 20
MAX_RETRIES = 10
STREAM_CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks for streaming
STREAM_READ_TIMEOUT = 60  # Seconds to wait for each chunk read
CONNECT_TIMEOUT = 30  # Seconds to wait for connection
MIN_HASH_LENGTH = 3  # Minimum hash length (2-char prefix + at least 1 char)

_HEX_PATTERN = re.compile(r"^[a-f0-9]+$", re.IGNORECASE)


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


def _get_s3_config() -> Any:
    """Get standard S3 client config with retries and timeouts."""
    from aiobotocore.config import AioConfig

    return AioConfig(
        retries={"max_attempts": MAX_RETRIES},
        connect_timeout=CONNECT_TIMEOUT,
        read_timeout=STREAM_READ_TIMEOUT,
    )


def _validate_hash(cache_hash: str) -> None:
    """Validate hash has minimum length and is hexadecimal."""
    if len(cache_hash) < MIN_HASH_LENGTH:
        raise exceptions.RemoteError(
            f"Invalid cache hash '{cache_hash}': must be at least {MIN_HASH_LENGTH} characters"
        )
    if not _HEX_PATTERN.match(cache_hash):
        raise exceptions.RemoteError(f"Invalid cache hash '{cache_hash}': must be hexadecimal")


def _is_not_found_error(e: Any) -> bool:
    """Check if botocore ClientError is a 404 Not Found."""
    return e.response.get("Error", {}).get("Code") == "404"


def _hash_to_key(prefix: str, hash_: str) -> str:
    """Convert cache hash to S3 key (files/XX/YYYYYYYY...)."""
    return f"{prefix}files/{hash_[:2]}/{hash_[2:]}"


def _key_to_hash(prefix: str, key: str) -> str | None:
    """Extract hash from S3 key, or None if not a cache file key."""
    expected_prefix = f"{prefix}files/"
    if not key.startswith(expected_prefix):
        return None
    parts = key[len(expected_prefix) :].split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0] + parts[1]


async def _write_all_async(fd: int, data: bytes) -> None:
    """Write all bytes to fd asynchronously, handling partial writes."""
    written = 0
    while written < len(data):
        n = await asyncio.to_thread(os.write, fd, data[written:])
        if n == 0:
            raise OSError("os.write returned 0")
        written += n


async def _stream_download_to_fd(
    response: Any,  # S3 get_object response
    fd: int,
) -> None:
    """Stream S3 response body to file descriptor in chunks with timeout."""
    async with response["Body"] as stream:
        while True:
            chunk: bytes = await asyncio.wait_for(
                stream.read(STREAM_CHUNK_SIZE),
                timeout=STREAM_READ_TIMEOUT,
            )
            if not chunk:
                break
            await _write_all_async(fd, chunk)


async def _atomic_download(
    s3: Any,  # S3 client
    bucket: str,
    key: str,
    local_path: pathlib.Path,
) -> None:
    """Download S3 object to local path atomically via temp file with streaming."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=local_path.parent, prefix=".pivot_download_")
    move_succeeded = False
    try:
        response = await s3.get_object(Bucket=bucket, Key=key)
        await _stream_download_to_fd(response, fd)
        os.close(fd)
        fd = -1  # Mark as closed
        shutil.move(tmp_path, local_path)
        move_succeeded = True
    finally:
        if fd >= 0:
            os.close(fd)
        if not move_succeeded and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def _stream_upload(
    s3: Any,  # S3 client
    bucket: str,
    key: str,
    local_path: pathlib.Path,
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
    upload_id = mpu["UploadId"]
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
            MultipartUpload={"Parts": parts},
        )
    except Exception:
        try:
            await s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        except Exception as abort_error:
            logger.warning(f"Failed to abort multipart upload {upload_id}: {abort_error}")
        raise


class S3Remote:
    """Async S3 remote storage backend."""

    _bucket: str
    _prefix: str

    def __init__(self, url: str) -> None:
        """Initialize S3 remote from s3://bucket/prefix URL."""
        bucket, prefix = remote_config.validate_s3_url(url)
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def prefix(self) -> str:
        return self._prefix

    async def exists(self, cache_hash: str) -> bool:
        """Check if hash exists on remote via HEAD request."""
        import aioboto3
        from botocore import exceptions as botocore_exc

        _validate_hash(cache_hash)
        session = aioboto3.Session()
        async with session.client("s3", config=_get_s3_config()) as s3:
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
        """Check which hashes exist on remote using parallel HEAD requests."""
        import aioboto3
        from botocore import exceptions as botocore_exc

        if not hashes:
            return {}

        for h in hashes:
            _validate_hash(h)

        semaphore = asyncio.Semaphore(concurrency)
        session = aioboto3.Session()

        async with session.client("s3", config=_get_s3_config()) as s3:

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
                        raise exceptions.RemoteConnectionError(
                            f"S3 HEAD error for {hash_}: {e}"
                        ) from e

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

    async def iter_hashes(self) -> AsyncIterator[str]:
        """Iterate over all cache hashes on remote (memory-efficient streaming)."""
        import aioboto3

        session = aioboto3.Session()
        prefix = f"{self._prefix}files/"

        async with session.client("s3", config=_get_s3_config()) as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj.get("Key")
                    if key is None:
                        continue
                    hash_ = _key_to_hash(self._prefix, key)
                    if hash_:
                        yield hash_

    async def list_hashes(self) -> set[str]:
        """List all cache hashes on remote (collects into memory)."""
        hashes = set[str]()
        async for hash_ in self.iter_hashes():
            hashes.add(hash_)
        return hashes

    async def upload_file(self, local_path: pathlib.Path, cache_hash: str) -> None:
        """Upload a single file to remote with streaming for large files."""
        import aioboto3

        _validate_hash(cache_hash)
        session = aioboto3.Session()
        async with session.client("s3", config=_get_s3_config()) as s3:
            await _stream_upload(
                s3, self._bucket, _hash_to_key(self._prefix, cache_hash), local_path
            )

    async def download_file(self, cache_hash: str, local_path: pathlib.Path) -> None:
        """Download a single file from remote (atomic write via temp file, streamed)."""
        import aioboto3

        _validate_hash(cache_hash)
        session = aioboto3.Session()
        async with session.client("s3", config=_get_s3_config()) as s3:
            await _atomic_download(
                s3, self._bucket, _hash_to_key(self._prefix, cache_hash), local_path
            )

    async def upload_batch(
        self,
        items: list[tuple[pathlib.Path, str]],
        concurrency: int = DEFAULT_CONCURRENCY,
        callback: Callable[[int], None] | None = None,
    ) -> list[TransferResult]:
        """Upload multiple files in parallel with streaming for large files."""
        import aioboto3

        if not items:
            return []

        for _, h in items:
            _validate_hash(h)

        semaphore = asyncio.Semaphore(concurrency)
        session = aioboto3.Session()
        completed = 0

        async def upload_one(local_path: pathlib.Path, hash_: str) -> TransferResult:
            nonlocal completed
            async with semaphore:
                try:
                    async with session.client("s3", config=_get_s3_config()) as s3:
                        await _stream_upload(
                            s3, self._bucket, _hash_to_key(self._prefix, hash_), local_path
                        )
                    completed += 1
                    if callback:
                        callback(completed)
                    return TransferResult(hash=hash_, success=True)
                except Exception as e:
                    return TransferResult(hash=hash_, success=False, error=str(e))

        return await asyncio.gather(*[upload_one(p, h) for p, h in items])

    async def download_batch(
        self,
        items: list[tuple[str, pathlib.Path]],
        concurrency: int = DEFAULT_CONCURRENCY,
        callback: Callable[[int], None] | None = None,
    ) -> list[TransferResult]:
        """Download multiple files in parallel with atomic writes and streaming."""
        import aioboto3

        if not items:
            return []

        for h, _ in items:
            _validate_hash(h)

        semaphore = asyncio.Semaphore(concurrency)
        session = aioboto3.Session()
        completed = 0

        async def download_one(hash_: str, local_path: pathlib.Path) -> TransferResult:
            nonlocal completed
            async with semaphore:
                try:
                    async with session.client("s3", config=_get_s3_config()) as s3:
                        await _atomic_download(
                            s3, self._bucket, _hash_to_key(self._prefix, hash_), local_path
                        )
                    completed += 1
                    if callback:
                        callback(completed)
                    return TransferResult(hash=hash_, success=True)
                except Exception as e:
                    return TransferResult(hash=hash_, success=False, error=str(e))

        return await asyncio.gather(*[download_one(h, p) for h, p in items])
