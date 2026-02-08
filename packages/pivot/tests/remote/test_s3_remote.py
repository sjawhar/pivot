from __future__ import annotations

import asyncio
import os
import pathlib
from typing import TYPE_CHECKING, Any

import pytest
from botocore import exceptions as botocore_exc

from pivot import exceptions
from pivot.remote import storage as remote_mod

if TYPE_CHECKING:
    import types

    from pytest_mock import MockerFixture
    from types_aiobotocore_s3 import S3Client


def _helper_patch_s3_client(
    mocker: MockerFixture,
    s3_remote: remote_mod.S3Remote,
    client: Any,
) -> None:
    mock_client_cm = mocker.AsyncMock()
    mock_client_cm.__aenter__.return_value = client
    mocker.patch.object(s3_remote._session, "client", return_value=mock_client_cm)


# -----------------------------------------------------------------------------
# RemoteFetcher Protocol Tests (for pivot get --rev)
# -----------------------------------------------------------------------------


def test_set_and_get_default_remote(mocker: MockerFixture) -> None:
    """set_default_remote sets and get_default_remote retrieves it."""
    mock_fetcher = mocker.Mock()

    old_remote = remote_mod.get_default_remote()
    try:
        remote_mod.set_default_remote(mock_fetcher)
        assert remote_mod.get_default_remote() is mock_fetcher

        remote_mod.set_default_remote(None)
        assert remote_mod.get_default_remote() is None
    finally:
        remote_mod.set_default_remote(old_remote)


def test_fetch_from_remote_no_remote_configured() -> None:
    """fetch_from_remote returns None when no remote configured."""
    old_remote = remote_mod.get_default_remote()
    try:
        remote_mod.set_default_remote(None)
        result = remote_mod.fetch_from_remote("abc123def4567890")
        assert result is None
    finally:
        remote_mod.set_default_remote(old_remote)


def test_fetch_from_remote_success(mocker: MockerFixture) -> None:
    """fetch_from_remote returns content when remote fetch succeeds."""
    mock_fetcher = mocker.Mock()
    mock_fetcher.fetch.return_value = b"file content"

    old_remote = remote_mod.get_default_remote()
    try:
        remote_mod.set_default_remote(mock_fetcher)
        result = remote_mod.fetch_from_remote("abc123def4567890")
        assert result == b"file content"
        mock_fetcher.fetch.assert_called_once_with("abc123def4567890")
    finally:
        remote_mod.set_default_remote(old_remote)


def test_fetch_from_remote_fetch_error(mocker: MockerFixture) -> None:
    """fetch_from_remote returns None when remote raises RemoteFetchError."""
    mock_fetcher = mocker.Mock()
    mock_fetcher.fetch.side_effect = exceptions.RemoteFetchError("Network error")

    old_remote = remote_mod.get_default_remote()
    try:
        remote_mod.set_default_remote(mock_fetcher)
        result = remote_mod.fetch_from_remote("abc123def4567890")
        assert result is None
    finally:
        remote_mod.set_default_remote(old_remote)


# -----------------------------------------------------------------------------
# S3Remote Initialization Tests
# -----------------------------------------------------------------------------


def test_s3_remote_init_basic() -> None:
    """S3Remote parses bucket and prefix from URL."""
    r = remote_mod.S3Remote("s3://my-bucket/my-prefix")
    assert r.bucket == "my-bucket"
    assert r.prefix == "my-prefix/"


def test_s3_remote_init_no_prefix() -> None:
    """S3Remote handles URL without prefix."""
    r = remote_mod.S3Remote("s3://my-bucket")
    assert r.bucket == "my-bucket"
    assert r.prefix == ""


def test_s3_remote_init_nested_prefix() -> None:
    """S3Remote handles nested prefix path."""
    r = remote_mod.S3Remote("s3://bucket/path/to/cache")
    assert r.bucket == "bucket"
    assert r.prefix == "path/to/cache/"


def test_s3_remote_init_invalid_url() -> None:
    """S3Remote raises on invalid URL."""
    with pytest.raises(exceptions.InvalidRemoteURLError):
        remote_mod.S3Remote("not-an-s3-url")


# -----------------------------------------------------------------------------
# Hash to Key Conversion Tests
# -----------------------------------------------------------------------------


def test_hash_to_key_with_prefix() -> None:
    """Hash converts to key with prefix."""
    key = remote_mod._hash_to_key("cache/", "abcdef1234567890")
    assert key == "cache/files/ab/cdef1234567890"


def test_hash_to_key_no_prefix() -> None:
    """Hash converts to key without prefix."""
    key = remote_mod._hash_to_key("", "abcdef1234567890")
    assert key == "files/ab/cdef1234567890"


def test_key_to_hash_with_prefix() -> None:
    """Key converts back to hash."""
    hash_ = remote_mod._key_to_hash("cache/", "cache/files/ab/cdef1234567890")
    assert hash_ == "abcdef1234567890"


def test_key_to_hash_no_prefix() -> None:
    """Key without prefix converts to hash."""
    hash_ = remote_mod._key_to_hash("", "files/ab/cdef1234567890")
    assert hash_ == "abcdef1234567890"


def test_key_to_hash_wrong_prefix() -> None:
    """Key with wrong prefix returns None."""
    hash_ = remote_mod._key_to_hash("cache/", "other/files/ab/cdef1234567890")
    assert hash_ is None


def test_key_to_hash_not_cache_file() -> None:
    """Non-cache key returns None."""
    hash_ = remote_mod._key_to_hash("cache/", "cache/stages/my_stage.lock")
    assert hash_ is None


def test_key_to_hash_malformed() -> None:
    """Malformed key returns None."""
    hash_ = remote_mod._key_to_hash("", "files/abcdef1234567890")  # Missing split
    assert hash_ is None


def test_key_to_hash_empty_parts() -> None:
    """Key with empty parts returns None."""
    assert remote_mod._key_to_hash("", "files//abcdef") is None
    assert remote_mod._key_to_hash("", "files/ab/") is None


def test_key_to_hash_rejects_invalid_hex() -> None:
    """Key producing non-hex or wrong-length hash returns None."""
    assert remote_mod._key_to_hash("", "files/AB/CDEF1234567890") is None  # Uppercase
    assert remote_mod._key_to_hash("", "files/ab/c") is None  # Too short
    assert remote_mod._key_to_hash("", "files/ab/cdef12345678901234") is None  # Too long


def test_validate_hash_short_raises() -> None:
    """Hash with wrong length raises RemoteError."""
    with pytest.raises(exceptions.RemoteError, match="16 lowercase hex"):
        remote_mod._validate_hash("")
    with pytest.raises(exceptions.RemoteError, match="16 lowercase hex"):
        remote_mod._validate_hash("ab")
    with pytest.raises(exceptions.RemoteError, match="16 lowercase hex"):
        remote_mod._validate_hash("abc")  # Was valid before, now too short
    with pytest.raises(exceptions.RemoteError, match="16 lowercase hex"):
        remote_mod._validate_hash("abcdef123456789")  # 15 chars


def test_validate_hash_non_hex_raises() -> None:
    """Non-lowercase-hex hash raises RemoteError."""
    with pytest.raises(exceptions.RemoteError, match="16 lowercase hex"):
        remote_mod._validate_hash("ghijklmnopqrstuv")  # Non-hex chars
    with pytest.raises(exceptions.RemoteError, match="16 lowercase hex"):
        remote_mod._validate_hash("ABCDEF1234567890")  # Uppercase
    with pytest.raises(exceptions.RemoteError, match="16 lowercase hex"):
        remote_mod._validate_hash("abc/def123456789")  # Path separator


def test_validate_hash_valid() -> None:
    """Valid 16-char lowercase hex hash passes validation."""
    remote_mod._validate_hash("abcdef1234567890")
    remote_mod._validate_hash("0123456789abcdef")
    remote_mod._validate_hash("0000000000000000")
    remote_mod._validate_hash("ffffffffffffffff")


def test_is_not_found_error_true() -> None:
    """_is_not_found_error returns True for 404."""
    error = botocore_exc.ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
    )
    assert remote_mod._is_not_found_error(error) is True


def test_is_not_found_error_false() -> None:
    """_is_not_found_error returns False for non-404."""
    error_403 = botocore_exc.ClientError(
        {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject"
    )
    assert remote_mod._is_not_found_error(error_403) is False

    error_empty = botocore_exc.ClientError({}, "HeadObject")
    assert remote_mod._is_not_found_error(error_empty) is False


# -----------------------------------------------------------------------------
# Async Method Tests (Moto)
# -----------------------------------------------------------------------------


async def test_exists_true(s3_remote: remote_mod.S3Remote, aioboto3_s3_client: S3Client) -> None:
    """exists returns True when object exists."""
    cache_hash = "abcdef1234567890"
    await aioboto3_s3_client.put_object(
        Bucket=s3_remote.bucket,
        Key=remote_mod._hash_to_key(s3_remote.prefix, cache_hash),
        Body=b"test",
    )

    result = await s3_remote.exists(cache_hash)

    assert result is True


async def test_exists_false_404(s3_remote: remote_mod.S3Remote) -> None:
    """exists returns False on 404 error."""
    result = await s3_remote.exists("abcdef1234567890")

    assert result is False


async def test_exists_raises_on_other_error(
    s3_remote: remote_mod.S3Remote, mocker: MockerFixture
) -> None:
    """exists raises RemoteConnectionError on non-404 errors."""
    mock_client = mocker.AsyncMock()
    error = botocore_exc.ClientError(
        {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject"
    )
    mock_client.head_object = mocker.AsyncMock(side_effect=error)
    _helper_patch_s3_client(mocker, s3_remote, mock_client)

    with pytest.raises(exceptions.RemoteConnectionError, match="S3 error"):
        await s3_remote.exists("abcdef1234567890")


async def test_upload_file(
    s3_remote: remote_mod.S3Remote,
    tmp_path: pathlib.Path,
    aioboto3_s3_client: S3Client,
) -> None:
    """upload_file puts object to S3."""
    cache_hash = "abc123def4567890"
    test_file = tmp_path / "test.txt"
    test_file.write_bytes(b"test content")

    await s3_remote.upload_file(test_file, cache_hash)

    response = await aioboto3_s3_client.get_object(
        Bucket=s3_remote.bucket,
        Key=remote_mod._hash_to_key(s3_remote.prefix, cache_hash),
    )
    content = await response["Body"].read()

    assert content == b"test content"


async def test_download_file(
    s3_remote: remote_mod.S3Remote,
    tmp_path: pathlib.Path,
    aioboto3_s3_client: S3Client,
) -> None:
    """download_file gets object from S3."""
    cache_hash = "abc123def4567890"
    await aioboto3_s3_client.put_object(
        Bucket=s3_remote.bucket,
        Key=remote_mod._hash_to_key(s3_remote.prefix, cache_hash),
        Body=b"downloaded content",
    )

    dest_file = tmp_path / "dest.txt"

    await s3_remote.download_file(cache_hash, dest_file)

    assert dest_file.read_bytes() == b"downloaded content"


async def test_download_file_streaming_body_api_regression(
    s3_remote: remote_mod.S3Remote,
    tmp_path: pathlib.Path,
    aioboto3_s3_client: S3Client,
) -> None:
    """Regression test: StreamingBody.read() works without async-with."""
    cache_hash = "abc123def4567890"
    await aioboto3_s3_client.put_object(
        Bucket=s3_remote.bucket,
        Key=remote_mod._hash_to_key(s3_remote.prefix, cache_hash),
        Body=b"test content for regression",
    )

    file_path = tmp_path / "downloaded.txt"

    await s3_remote.download_file(cache_hash, file_path)

    assert file_path.read_bytes() == b"test content for regression"


async def test_download_file_readonly(
    s3_remote: remote_mod.S3Remote,
    tmp_path: pathlib.Path,
    aioboto3_s3_client: S3Client,
) -> None:
    """download_file with readonly=True sets 0o444 permissions (for cache files)."""
    cache_hash = "abc123def4567890"
    await aioboto3_s3_client.put_object(
        Bucket=s3_remote.bucket,
        Key=remote_mod._hash_to_key(s3_remote.prefix, cache_hash),
        Body=b"cached content",
    )

    dest_file = tmp_path / "cached.txt"

    await s3_remote.download_file(cache_hash, dest_file, readonly=True)

    assert dest_file.read_bytes() == b"cached content"
    # Verify read-only permissions (0o444)
    mode = dest_file.stat().st_mode & 0o777
    assert mode == 0o444, f"Expected 0o444, got {oct(mode)}"


async def test_bulk_exists(s3_remote: remote_mod.S3Remote, aioboto3_s3_client: S3Client) -> None:
    """bulk_exists checks multiple hashes in parallel."""
    hashes = ["a1b2c3d4e5f6789a", "b2c3d4e5f6789ab1", "c3d4e5f6789ab1c2"]
    existing = {hashes[0], hashes[2]}

    for cache_hash in existing:
        await aioboto3_s3_client.put_object(
            Bucket=s3_remote.bucket,
            Key=remote_mod._hash_to_key(s3_remote.prefix, cache_hash),
            Body=b"content",
        )

    result = await s3_remote.bulk_exists(hashes)

    assert result[hashes[0]] is True
    assert result[hashes[1]] is False
    assert result[hashes[2]] is True


async def test_bulk_exists_raises_on_non_404_error(
    s3_remote: remote_mod.S3Remote, mocker: MockerFixture
) -> None:
    """bulk_exists raises RemoteConnectionError on non-404 errors."""
    mock_client = mocker.AsyncMock()
    error = botocore_exc.ClientError(
        {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject"
    )
    mock_client.head_object = mocker.AsyncMock(side_effect=error)
    _helper_patch_s3_client(mocker, s3_remote, mock_client)

    with pytest.raises(exceptions.RemoteConnectionError, match="bulk_exists failed"):
        await s3_remote.bulk_exists(["a1b2c3d4e5f6789a", "b2c3d4e5f6789ab1", "c3d4e5f6789ab1c2"])


async def test_bulk_exists_uses_list_for_large_batches(
    s3_remote: remote_mod.S3Remote,
    aioboto3_s3_client: S3Client,
) -> None:
    """bulk_exists returns expected results for large batches (>= 50 hashes)."""
    hashes = [f"a{i:015x}" for i in range(60)]
    existing_hashes = set(hashes[::2])

    for cache_hash in existing_hashes:
        await aioboto3_s3_client.put_object(
            Bucket=s3_remote.bucket,
            Key=remote_mod._hash_to_key(s3_remote.prefix, cache_hash),
            Body=b"content",
        )

    result = await s3_remote.bulk_exists(hashes)

    for cache_hash in existing_hashes:
        assert result[cache_hash] is True, f"Hash {cache_hash} should exist"
    for cache_hash in set(hashes) - existing_hashes:
        assert result[cache_hash] is False, f"Hash {cache_hash} should not exist"


async def test_list_hashes(s3_remote: remote_mod.S3Remote, aioboto3_s3_client: S3Client) -> None:
    """list_hashes returns all cache hashes from S3."""
    hashes = {"abc123def4567890", "def456abc1234567", "123456789abcdef0"}
    for cache_hash in hashes:
        await aioboto3_s3_client.put_object(
            Bucket=s3_remote.bucket,
            Key=remote_mod._hash_to_key(s3_remote.prefix, cache_hash),
            Body=b"content",
        )

    result = await s3_remote.list_hashes()

    assert result == hashes


async def test_iter_hashes(s3_remote: remote_mod.S3Remote, aioboto3_s3_client: S3Client) -> None:
    """iter_hashes yields hashes without collecting into memory."""
    expected = {"abc123def4567890", "def456abc1234567"}

    for cache_hash in expected:
        await aioboto3_s3_client.put_object(
            Bucket=s3_remote.bucket,
            Key=remote_mod._hash_to_key(s3_remote.prefix, cache_hash),
            Body=b"content",
        )

    hashes = [h async for h in s3_remote.iter_hashes()]

    assert set(hashes) == expected


async def test_iter_hashes_skips_invalid_keys(
    s3_remote: remote_mod.S3Remote, aioboto3_s3_client: S3Client
) -> None:
    """iter_hashes skips keys that don't produce valid 16-char lowercase hex hashes."""
    keys = [
        f"{s3_remote.prefix}files/ab/cdef1234567890",  # Valid: 16 lowercase hex
        f"{s3_remote.prefix}files/AB/CDEF1234567890",  # Invalid: uppercase
        f"{s3_remote.prefix}files/ab/c",  # Invalid: too short
        f"{s3_remote.prefix}files/ab/cdef12345678901234",  # Invalid: too long
        f"{s3_remote.prefix}files/stages/my_stage.lock",  # Invalid: not a cache file
    ]

    for key in keys:
        await aioboto3_s3_client.put_object(
            Bucket=s3_remote.bucket,
            Key=key,
            Body=b"content",
        )

    hashes = [h async for h in s3_remote.iter_hashes()]

    assert hashes == ["abcdef1234567890"]


async def test_upload_batch(
    s3_remote: remote_mod.S3Remote,
    tmp_path: pathlib.Path,
    aioboto3_s3_client: S3Client,
) -> None:
    """upload_batch uploads multiple files in parallel."""
    files = list[tuple[pathlib.Path, str]]()
    contents: dict[str, bytes] = {}
    for i in range(3):
        cache_hash = f"a{i}b2c3d4e5f6789a"
        f = tmp_path / f"file{i}.txt"
        data = f"content {i}".encode()
        f.write_bytes(data)
        files.append((f, cache_hash))
        contents[cache_hash] = data

    results = await s3_remote.upload_batch(files, concurrency=10)

    assert len(results) == 3
    assert all(r["success"] for r in results)

    for cache_hash, expected in contents.items():
        response = await aioboto3_s3_client.get_object(
            Bucket=s3_remote.bucket,
            Key=remote_mod._hash_to_key(s3_remote.prefix, cache_hash),
        )
        assert await response["Body"].read() == expected


async def test_upload_batch_with_callback(
    s3_remote: remote_mod.S3Remote, tmp_path: pathlib.Path
) -> None:
    """upload_batch calls callback for each completed upload."""
    files = list[tuple[pathlib.Path, str]]()
    for i in range(3):
        f = tmp_path / f"file{i}.txt"
        f.write_text(f"content {i}")
        files.append((f, f"a{i}b2c3d4e5f6789a"))

    callback_values = list[int]()

    def callback(n: int) -> None:
        callback_values.append(n)

    await s3_remote.upload_batch(files, concurrency=10, callback=callback)

    assert len(callback_values) == 3
    assert set(callback_values) == {1, 2, 3}


async def test_upload_batch_empty() -> None:
    """upload_batch with empty list returns empty results."""
    r = remote_mod.S3Remote("s3://bucket/prefix")
    results = await r.upload_batch([])

    assert results == []


async def test_download_batch(
    s3_remote: remote_mod.S3Remote,
    tmp_path: pathlib.Path,
    aioboto3_s3_client: S3Client,
) -> None:
    """download_batch downloads multiple files in parallel."""
    items = [(f"a{i}b2c3d4e5f6789a", tmp_path / f"dest{i}.txt") for i in range(3)]

    for cache_hash, _ in items:
        await aioboto3_s3_client.put_object(
            Bucket=s3_remote.bucket,
            Key=remote_mod._hash_to_key(s3_remote.prefix, cache_hash),
            Body=b"content",
        )

    results = await s3_remote.download_batch(items, concurrency=10)

    assert len(results) == 3
    assert all(r["success"] for r in results)
    for _, path in items:
        assert path.exists()
        assert path.read_bytes() == b"content"


async def test_download_batch_empty() -> None:
    """download_batch with empty list returns empty results."""
    r = remote_mod.S3Remote("s3://bucket/prefix")
    results = await r.download_batch([])

    assert results == []


# -----------------------------------------------------------------------------
# Multipart Upload Tests (Large Files)
# -----------------------------------------------------------------------------


async def test_upload_file_large_uses_multipart(
    s3_remote: remote_mod.S3Remote,
    tmp_path: pathlib.Path,
    aioboto3_s3_client: S3Client,
) -> None:
    """upload_file uses multipart upload for large files."""
    # moto enforces S3's real 5 MiB minimum multipart part size, so we cannot
    # monkeypatch STREAM_CHUNK_SIZE to a small value — the payload must exceed
    # the real STREAM_CHUNK_SIZE to trigger the multipart code path.
    test_file = tmp_path / "large_file.bin"
    large_payload = b"x" * (remote_mod.STREAM_CHUNK_SIZE + 1)
    test_file.write_bytes(large_payload)

    cache_hash = "abc123def4567890"

    await s3_remote.upload_file(test_file, cache_hash)

    response = await aioboto3_s3_client.get_object(
        Bucket=s3_remote.bucket,
        Key=remote_mod._hash_to_key(s3_remote.prefix, cache_hash),
    )
    assert await response["Body"].read() == large_payload


async def test_upload_file_small_uses_put_object(
    s3_remote: remote_mod.S3Remote,
    tmp_path: pathlib.Path,
    aioboto3_s3_client: S3Client,
) -> None:
    """upload_file uses simple put_object for small files."""
    test_file = tmp_path / "small_file.txt"
    test_file.write_text("small content")
    cache_hash = "abc123def4567890"

    await s3_remote.upload_file(test_file, cache_hash)

    response = await aioboto3_s3_client.get_object(
        Bucket=s3_remote.bucket,
        Key=remote_mod._hash_to_key(s3_remote.prefix, cache_hash),
    )
    assert await response["Body"].read() == b"small content"


async def test_upload_file_multipart_aborts_on_error(
    s3_remote: remote_mod.S3Remote, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """upload_file aborts multipart upload on error."""
    # Monkeypatch chunk size to 100 bytes to avoid writing large files
    mocker.patch.object(remote_mod, "STREAM_CHUNK_SIZE", 100)

    mock_client = mocker.AsyncMock()
    mock_client.create_multipart_upload = mocker.AsyncMock(
        return_value={"UploadId": "test-upload-id"}
    )
    mock_client.upload_part = mocker.AsyncMock(side_effect=Exception("Upload failed"))
    mock_client.abort_multipart_upload = mocker.AsyncMock()
    _helper_patch_s3_client(mocker, s3_remote, mock_client)

    test_file = tmp_path / "large_file.bin"
    test_file.write_bytes(b"x" * 150)  # Above 100 byte threshold

    with pytest.raises(Exception, match="Upload failed"):
        await s3_remote.upload_file(test_file, "abc123def4567890")

    mock_client.abort_multipart_upload.assert_called_once()


# -----------------------------------------------------------------------------
# Atomic Download Tests
# -----------------------------------------------------------------------------


async def test_download_file_cleans_up_on_error(
    s3_remote: remote_mod.S3Remote, tmp_path: pathlib.Path
) -> None:
    """download_file cleans up temp file on error."""
    dest_file = tmp_path / "dest.txt"

    with pytest.raises(botocore_exc.ClientError):
        await s3_remote.download_file("abc123def4567890", dest_file)

    # Verify no temp files left behind
    temp_files = [f for f in os.listdir(tmp_path) if f.startswith(".pivot_download_")]
    assert len(temp_files) == 0, f"Temp files not cleaned up: {temp_files}"


async def test_download_file_mid_stream_error(
    s3_remote: remote_mod.S3Remote,
    tmp_path: pathlib.Path,
    aioboto3_s3_client: S3Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """download_file cleans up when stream.read() raises mid-transfer."""
    cache_hash = "abc123def4567890"
    await aioboto3_s3_client.put_object(
        Bucket=s3_remote.bucket,
        Key=remote_mod._hash_to_key(s3_remote.prefix, cache_hash),
        Body=b"x" * 1024,
    )

    call_count = 0
    real_stream_download = remote_mod._stream_download_to_fd

    async def _inject_read_failure(response: Any, fd: int) -> None:
        """Wrap stream.read to fail on second call, then delegate to real impl."""
        nonlocal call_count
        body = response["Body"]
        original_read = body.read

        async def _failing_read(amt: int) -> bytes:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise ConnectionError("network interrupted")
            return await original_read(amt)

        body.read = _failing_read
        await real_stream_download(response, fd)

    monkeypatch.setattr(remote_mod, "_stream_download_to_fd", _inject_read_failure)

    dest_file = tmp_path / "dest.txt"
    with pytest.raises(ConnectionError, match="network interrupted"):
        await s3_remote.download_file(cache_hash, dest_file)

    temp_files = [f for f in os.listdir(tmp_path) if f.startswith(".pivot_download_")]
    assert len(temp_files) == 0, f"Temp files not cleaned up: {temp_files}"
    assert not dest_file.exists(), "Dest file should not exist after mid-stream error"


async def test_download_file_stream_read_timeout(
    s3_remote: remote_mod.S3Remote,
    tmp_path: pathlib.Path,
    aioboto3_s3_client: S3Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """download_file cleans up when stream.read() times out."""
    cache_hash = "abc123def4567890"
    await aioboto3_s3_client.put_object(
        Bucket=s3_remote.bucket,
        Key=remote_mod._hash_to_key(s3_remote.prefix, cache_hash),
        Body=b"x" * 1024,
    )

    # Use a short timeout so the test doesn't wait 60s
    monkeypatch.setattr(remote_mod, "STREAM_READ_TIMEOUT", 0.1)

    real_stream_download = remote_mod._stream_download_to_fd
    hang_event = asyncio.Event()

    async def _inject_hanging_read(response: Any, fd: int) -> None:
        body = response["Body"]
        original_read = body.read
        call_count = 0

        async def _hanging_read(amt: int) -> bytes:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                await hang_event.wait()  # Hangs forever (until timeout)
            return await original_read(amt)

        body.read = _hanging_read
        await real_stream_download(response, fd)

    monkeypatch.setattr(remote_mod, "_stream_download_to_fd", _inject_hanging_read)

    dest_file = tmp_path / "dest.txt"
    with pytest.raises(asyncio.TimeoutError):
        await s3_remote.download_file(cache_hash, dest_file)

    temp_files = [f for f in os.listdir(tmp_path) if f.startswith(".pivot_download_")]
    assert len(temp_files) == 0, f"Temp files not cleaned up: {temp_files}"
    assert not dest_file.exists(), "Dest file should not exist after timeout"


async def test_download_batch_with_callback(
    s3_remote: remote_mod.S3Remote,
    tmp_path: pathlib.Path,
    aioboto3_s3_client: S3Client,
) -> None:
    """download_batch calls callback for each completed download."""
    items = [(f"a{i}b2c3d4e5f6789a", tmp_path / f"dest{i}.txt") for i in range(3)]

    for cache_hash, _ in items:
        await aioboto3_s3_client.put_object(
            Bucket=s3_remote.bucket,
            Key=remote_mod._hash_to_key(s3_remote.prefix, cache_hash),
            Body=b"content",
        )

    callback_values = list[int]()

    def callback(n: int) -> None:
        callback_values.append(n)

    await s3_remote.download_batch(items, concurrency=10, callback=callback)

    assert len(callback_values) == 3
    assert set(callback_values) == {1, 2, 3}


async def test_download_file_default_permissions(
    s3_remote: remote_mod.S3Remote,
    tmp_path: pathlib.Path,
    aioboto3_s3_client: S3Client,
) -> None:
    """download_file without readonly flag creates file with default permissions."""
    cache_hash = "abc123def4567890"
    await aioboto3_s3_client.put_object(
        Bucket=s3_remote.bucket,
        Key=remote_mod._hash_to_key(s3_remote.prefix, cache_hash),
        Body=b"writable content",
    )

    dest_file = tmp_path / "writable.txt"

    await s3_remote.download_file(cache_hash, dest_file, readonly=False)

    assert dest_file.read_bytes() == b"writable content"
    # Verify file is writable (not 0o444)
    mode = dest_file.stat().st_mode & 0o777
    assert mode != 0o444, f"Expected writable permissions, got {oct(mode)}"
    # Default umask typically results in 0o644 or similar
    assert mode & 0o200, f"Expected owner write permission, got {oct(mode)}"


async def test_download_batch_readonly(
    s3_remote: remote_mod.S3Remote,
    tmp_path: pathlib.Path,
    aioboto3_s3_client: S3Client,
) -> None:
    """download_batch with readonly=True sets 0o444 permissions on all files."""
    items = [(f"a{i}b2c3d4e5f6789a", tmp_path / f"cached{i}.txt") for i in range(3)]

    for cache_hash, _ in items:
        await aioboto3_s3_client.put_object(
            Bucket=s3_remote.bucket,
            Key=remote_mod._hash_to_key(s3_remote.prefix, cache_hash),
            Body=b"content",
        )

    results = await s3_remote.download_batch(items, concurrency=10, readonly=True)

    assert len(results) == 3
    assert all(r["success"] for r in results)

    # Verify all files have read-only permissions
    for _, path in items:
        assert path.exists()
        mode = path.stat().st_mode & 0o777
        assert mode == 0o444, f"Expected 0o444 for {path}, got {oct(mode)}"


# -----------------------------------------------------------------------------
# Metrics / Finally Tests
# -----------------------------------------------------------------------------


async def test_bulk_exists_calls_metrics_end_on_error(
    s3_remote: remote_mod.S3Remote, mocker: MockerFixture
) -> None:
    """bulk_exists calls metrics.end even when an error occurs."""
    from pivot import metrics

    error = botocore_exc.ClientError(
        {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject"
    )
    mock_client = mocker.AsyncMock()
    mock_client.head_object = mocker.AsyncMock(side_effect=error)
    _helper_patch_s3_client(mocker, s3_remote, mock_client)

    spy_end = mocker.spy(metrics, "end")

    with pytest.raises(exceptions.RemoteConnectionError):
        await s3_remote.bulk_exists(["a1b2c3d4e5f6789a"])

    spy_end.assert_any_call("storage.bulk_exists", mocker.ANY)


async def test_upload_batch_calls_metrics_end_on_error(
    s3_remote: remote_mod.S3Remote, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """upload_batch calls metrics.end even when uploads fail."""
    from pivot import metrics

    mock_client = mocker.AsyncMock()
    mock_client.put_object = mocker.AsyncMock(side_effect=Exception("upload boom"))
    _helper_patch_s3_client(mocker, s3_remote, mock_client)

    spy_end = mocker.spy(metrics, "end")

    f = tmp_path / "file.txt"
    f.write_text("content")

    # upload_batch catches per-item errors, so it completes without raising
    await s3_remote.upload_batch([(f, "a1b2c3d4e5f6789a")])

    spy_end.assert_any_call("storage.upload_batch", mocker.ANY)


async def test_download_batch_calls_metrics_end_on_error(
    s3_remote: remote_mod.S3Remote, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """download_batch calls metrics.end even when downloads fail."""
    from pivot import metrics

    mock_client = mocker.AsyncMock()
    mock_client.get_object = mocker.AsyncMock(side_effect=Exception("download boom"))
    _helper_patch_s3_client(mocker, s3_remote, mock_client)

    spy_end = mocker.spy(metrics, "end")

    await s3_remote.download_batch([("a1b2c3d4e5f6789a", tmp_path / "dest.txt")])

    spy_end.assert_any_call("storage.download_batch", mocker.ANY)


# -----------------------------------------------------------------------------
# Session Reuse Tests
# -----------------------------------------------------------------------------


async def test_s3_remote_reuses_session_across_methods(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """S3Remote reuses the same aioboto3 session across multiple method calls."""
    mock_session_class = mocker.patch("aioboto3.Session", autospec=True)
    mock_session = mocker.MagicMock()
    mock_session_class.return_value = mock_session

    mock_client = mocker.AsyncMock()
    mock_client.head_object = mocker.AsyncMock(return_value={})
    mock_client.put_object = mocker.AsyncMock()
    mock_client_cm = mocker.AsyncMock()
    mock_client_cm.__aenter__.return_value = mock_client
    mock_session.client.return_value = mock_client_cm

    r = remote_mod.S3Remote("s3://bucket/prefix")
    await r.exists("abc123def4567890")

    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
    await r.upload_file(test_file, "abc123def4567890")

    # Session created once in __init__; both methods share it via self._session.client()
    assert mock_session_class.call_count == 1


# -----------------------------------------------------------------------------
# Init Error Tests
# -----------------------------------------------------------------------------


def test_s3_remote_init_raises_on_missing_aioboto3(mocker: MockerFixture) -> None:
    """S3Remote raises RemoteError when aioboto3 is not installed."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> types.ModuleType:  # noqa: ANN401 - wraps builtins.__import__ which requires Any for forwarded args
        if name == "aioboto3":
            raise ModuleNotFoundError("No module named 'aioboto3'")
        return original_import(name, *args, **kwargs)

    mocker.patch("builtins.__import__", side_effect=mock_import)

    with pytest.raises(exceptions.RemoteError, match="pip install pivot\\[s3\\]"):
        remote_mod.S3Remote("s3://bucket/prefix")
