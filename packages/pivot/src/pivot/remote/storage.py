from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import TYPE_CHECKING, TypedDict

from pivot import config, exceptions
from pivot.remote import batch, keys
from pivot.remote import config as remote_config

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable
    from pathlib import Path

    import aioboto3
    from aiobotocore.config import AioConfig
    from botocore.exceptions import ClientError
    from types_aiobotocore_s3 import S3Client
    from types_aiobotocore_s3.type_defs import GetObjectOutputTypeDef

    from pivot.types import TransferResult

logger = logging.getLogger(__name__)

# Constants for S3 operations
STREAM_CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks for streaming
STREAM_READ_TIMEOUT = 60  # Seconds to wait for each chunk read

# Cached S3 config to avoid re-parsing config on every call.
# Safe without locking - asyncio is single-threaded, worst case race just
# creates duplicate config (harmless).
_cached_s3_config: AioConfig | None = None


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


def _is_not_found_error(e: ClientError) -> bool:
    """Check if botocore ClientError is a 404 Not Found."""
    return e.response.get("Error", {}).get("Code") == "404"


_AUTH_ERROR_CODES = frozenset(
    {
        "ExpiredTokenException",
        "ExpiredToken",
        "InvalidIdentityToken",
        "InvalidGrantException",
        "UnrecognizedClientException",
        "InvalidClientTokenId",
        "SignatureDoesNotMatch",
    }
)

_AUTH_ERROR_MESSAGE = (
    "AWS credentials are expired or invalid. "
    "Run 'aws sso login' to refresh your credentials, or check your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
)


def _is_auth_error(e: ClientError) -> bool:
    """Check if botocore ClientError is an authentication error."""
    return e.response.get("Error", {}).get("Code") in _AUTH_ERROR_CODES


async def _stream_download_to_fd(
    response: GetObjectOutputTypeDef,
    fd: int,
) -> str:
    """Stream S3 response body to fd and return its xxh64 integrity digest."""
    stream = response["Body"]
    try:
        return await keys.stream_to_fd(
            lambda: asyncio.wait_for(stream.read(STREAM_CHUNK_SIZE), timeout=STREAM_READ_TIMEOUT),
            fd,
        )
    finally:
        with contextlib.suppress(Exception):
            stream.close()  # pyright: ignore[reportUnknownMemberType]


async def _download_verified(
    s3: S3Client, bucket: str, key: str, dest: Path, expected_hash: str
) -> None:
    """Stream bucket/key into the existing file ``dest`` and check its xxh64 digest."""
    try:
        response = await s3.get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        error_str = str(exc)
        if "NoSuchKey" in error_str or "404" in error_str:
            raise exceptions.RemoteError(f"Remote cache file not found: {bucket}/{key}") from exc
        if "AccessDenied" in error_str or "403" in error_str:
            raise exceptions.RemoteError(
                f"Access denied to {bucket}/{key}. Check AWS credentials."
            ) from exc
        raise exceptions.RemoteError(f"Failed to download from S3: {exc}") from exc
    fd = os.open(dest, os.O_WRONLY | os.O_TRUNC)
    try:
        actual_hash = await _stream_download_to_fd(response, fd)
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
    keys.check_digest(expected_hash, actual_hash, f"{bucket}/{key}")


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
    Each method creates its own S3 client via ``async with self._client()``,
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

        scheme, bucket, prefix = remote_config.validate_remote_url(url)
        if scheme is not remote_config.RemoteScheme.S3:
            raise exceptions.InvalidRemoteURLError(
                f"Not an S3 URL: {url}. Expected format: s3://bucket/prefix"
            )
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""
        try:
            # Reads the AWS profile, so AWS_PROFILE naming a missing profile fails here
            self._session = aioboto3.Session()
        except Exception as e:  # no session means no S3 access at all
            raise exceptions.RemoteConnectionError(
                f"Cannot set up S3 access for {self.url}: {e}"
            ) from e

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def url(self) -> str:
        """Normalized S3 URL for identity comparison."""
        if self._prefix:
            return f"s3://{self._bucket}/{self._prefix.rstrip('/')}"
        return f"s3://{self._bucket}"

    @contextlib.asynccontextmanager
    async def _client(self) -> AsyncIterator[S3Client]:
        """An S3 client from the instance's session, for one method call.

        A client that cannot start (an invalid AWS_ENDPOINT_URL, for example) raises
        ``RemoteConnectionError``, as the GCS backend does, so a caller that reports each file's
        result keeps reporting them.
        """
        client_config = _get_s3_config()
        async with contextlib.AsyncExitStack() as stack:
            try:
                s3 = await stack.enter_async_context(
                    self._session.client("s3", config=client_config)
                )
            except Exception as e:  # no client means no S3 access at all
                raise exceptions.RemoteConnectionError(
                    f"Cannot set up S3 access for {self.url}: {e}"
                ) from e
            yield s3

    async def _head_exists(self, s3: S3Client, hash_: str, error: str) -> bool:
        """HEAD the cache object ``hash_``; other failures raise ``S3 <error>: <cause>``."""
        from botocore import exceptions as botocore_exc

        try:
            await s3.head_object(
                Bucket=self._bucket,
                Key=keys.hash_to_key(self._prefix, hash_),
            )
            return True
        except botocore_exc.ClientError as e:
            if _is_not_found_error(e):
                return False
            if _is_auth_error(e):
                raise exceptions.RemoteConnectionError(_AUTH_ERROR_MESSAGE) from e
            raise exceptions.RemoteConnectionError(f"S3 {error}: {e}") from e
        except botocore_exc.NoCredentialsError as e:
            raise exceptions.RemoteConnectionError(_AUTH_ERROR_MESSAGE) from e

    async def _list_names(self, s3: S3Client, key_prefix: str, error: str) -> AsyncGenerator[str]:
        """Every key under ``key_prefix``; a failure raises ``S3 <error>: <cause>``."""
        from botocore import exceptions as botocore_exc

        try:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self._bucket, Prefix=key_prefix):
                for obj in page.get("Contents", []):
                    key = obj.get("Key")
                    if key is not None:
                        yield key
        except botocore_exc.ClientError as e:
            if _is_auth_error(e):
                raise exceptions.RemoteConnectionError(_AUTH_ERROR_MESSAGE) from e
            raise exceptions.RemoteConnectionError(f"S3 {error}: {e}") from e
        except botocore_exc.NoCredentialsError as e:
            raise exceptions.RemoteConnectionError(_AUTH_ERROR_MESSAGE) from e

    async def exists(self, cache_hash: str) -> bool:
        """Check if hash exists on remote via HEAD request."""
        keys.validate_hash(cache_hash)
        async with self._client() as s3:
            return await self._head_exists(s3, cache_hash, "error")

    async def bulk_exists(
        self, hashes: list[str], concurrency: int = batch.DEFAULT_CONCURRENCY
    ) -> dict[str, bool]:
        """Check which hashes exist on remote.

        Small batches send parallel HEAD requests; large ones LIST by 2-char prefix, one
        request per 1000 keys (``batch.bulk_exists``).
        """
        return await batch.bulk_exists(
            hashes,
            concurrency,
            open_client=self._client,
            exists_one=lambda s3, h: self._head_exists(s3, h, f"HEAD error for {h}"),
            list_names=lambda s3, key_prefix: self._list_names(s3, key_prefix, "LIST error"),
            prefix=self._prefix,
            label="S3",
            metric="storage.bulk_exists",
        )

    async def iter_hashes(self) -> AsyncIterator[str]:
        """Iterate over all cache hashes on remote (memory-efficient streaming)."""
        async with self._client() as s3:
            names = self._list_names(s3, f"{self._prefix}files/", "error")
            async for hash_ in batch.cache_hashes(names, self._prefix):
                yield hash_

    async def list_hashes(self) -> set[str]:
        """List all cache hashes on remote (collects into memory)."""
        return await batch.collect_hashes(self.iter_hashes(), "storage.list_hashes")

    async def upload_file(self, local_path: Path, cache_hash: str) -> None:
        """Upload a single file to remote with streaming for large files."""
        from botocore import exceptions as botocore_exc

        keys.validate_hash(cache_hash)
        try:
            async with self._client() as s3:
                await _stream_upload(
                    s3, self._bucket, keys.hash_to_key(self._prefix, cache_hash), local_path
                )
        except botocore_exc.ClientError as e:
            if _is_auth_error(e):
                raise exceptions.RemoteConnectionError(_AUTH_ERROR_MESSAGE) from e
            raise exceptions.RemoteConnectionError(f"S3 upload error: {e}") from e
        except botocore_exc.NoCredentialsError as e:
            raise exceptions.RemoteConnectionError(_AUTH_ERROR_MESSAGE) from e

    async def download_file(
        self, cache_hash: str, local_path: Path, *, readonly: bool = False
    ) -> None:
        """Download a single file from remote (atomic write via temp file, streamed)."""
        from botocore import exceptions as botocore_exc

        keys.validate_hash(cache_hash)
        key = keys.hash_to_key(self._prefix, cache_hash)
        try:
            async with self._client() as s3:
                await keys.atomic_download(
                    local_path,
                    lambda dest: _download_verified(s3, self._bucket, key, dest, cache_hash),
                    overwrite=True,
                    readonly=readonly,
                )
        except botocore_exc.ClientError as e:
            if _is_auth_error(e):
                raise exceptions.RemoteConnectionError(_AUTH_ERROR_MESSAGE) from e
            raise exceptions.RemoteConnectionError(f"S3 download error: {e}") from e
        except botocore_exc.NoCredentialsError as e:
            raise exceptions.RemoteConnectionError(_AUTH_ERROR_MESSAGE) from e

    async def _upload_one(self, s3: S3Client, hash_: str, local_path: Path) -> None:
        await _stream_upload(s3, self._bucket, keys.hash_to_key(self._prefix, hash_), local_path)

    async def upload_batch(
        self,
        items: list[tuple[Path, str]],
        concurrency: int = batch.DEFAULT_CONCURRENCY,
        callback: Callable[[int, int, str], None] | None = None,
    ) -> list[TransferResult]:
        """Upload multiple files in parallel with streaming for large files."""
        return await batch.transfer_batch(
            [(h, p) for p, h in items],
            concurrency,
            callback,
            open_client=self._client,
            transfer_one=self._upload_one,
            metric="storage.upload_batch",
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
        content-addressed cache entries that other processes may be hardlinking.

        Args:
            readonly: If True, set file permissions to 0o444 (for cache files).
        """

        async def download_one(s3: S3Client, hash_: str, local_path: Path) -> None:
            key = keys.hash_to_key(self._prefix, hash_)
            await keys.atomic_download(
                local_path,
                lambda dest: _download_verified(s3, self._bucket, key, dest, hash_),
                overwrite=False,
                readonly=readonly,
            )

        return await batch.transfer_batch(
            items,
            concurrency,
            callback,
            open_client=self._client,
            transfer_one=download_one,
            metric="storage.download_batch",
        )
