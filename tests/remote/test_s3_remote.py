from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING, Any

import pytest
from botocore import exceptions as botocore_exc

from pivot import exceptions
from pivot.remote import storage as remote_mod

if TYPE_CHECKING:
    import types
    from collections.abc import AsyncIterator
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture


class MockBody:
    """Mock S3 StreamingBody for testing.

    Supports async context manager (``async with response["Body"] as stream``)
    and ``stream.read(size)`` matching the aiobotocore StreamingBody stub API.
    """

    _data: bytes
    _read_called: bool

    def __init__(self, data: bytes = b"content") -> None:
        self._data = data
        self._read_called = False

    async def __aenter__(self) -> MockBody:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    async def read(self, size: int = -1) -> bytes:
        if self._read_called:
            return b""
        self._read_called = True
        return self._data


def _make_mock_get_object_response(**_kwargs: object) -> dict[str, MockBody]:
    """Create a mock S3 get_object response for testing."""
    return {"Body": MockBody()}


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
# Async Method Tests (Mocked)
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_s3_session(mocker: MockerFixture) -> MagicMock:
    """Mock aioboto3 session for testing."""
    mock_session_class = mocker.patch("aioboto3.Session", autospec=True)
    mock_session = mocker.MagicMock()
    mock_session_class.return_value = mock_session
    return mock_session


async def test_exists_true(mock_s3_session: MagicMock, mocker: MockerFixture) -> None:
    """exists returns True when object exists."""
    mock_client = mocker.AsyncMock()
    mock_client.head_object = mocker.AsyncMock(return_value={})
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    r = remote_mod.S3Remote("s3://bucket/prefix")
    result = await r.exists("abc123def4567890")

    assert result is True
    mock_client.head_object.assert_called_once()


async def test_exists_false_404(mock_s3_session: MagicMock, mocker: MockerFixture) -> None:
    """exists returns False on 404 error."""
    mock_client = mocker.AsyncMock()
    error = botocore_exc.ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
    )
    mock_client.head_object = mocker.AsyncMock(side_effect=error)
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    r = remote_mod.S3Remote("s3://bucket/prefix")
    result = await r.exists("abc123def4567890")

    assert result is False


async def test_exists_raises_on_other_error(
    mock_s3_session: MagicMock, mocker: MockerFixture
) -> None:
    """exists raises RemoteConnectionError on non-404 errors."""
    mock_client = mocker.AsyncMock()
    error = botocore_exc.ClientError(
        {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject"
    )
    mock_client.head_object = mocker.AsyncMock(side_effect=error)
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    r = remote_mod.S3Remote("s3://bucket/prefix")

    with pytest.raises(exceptions.RemoteConnectionError, match="S3 error"):
        await r.exists("abc123def4567890")


async def test_upload_file(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """upload_file puts object to S3."""
    mock_client = mocker.AsyncMock()
    mock_client.put_object = mocker.AsyncMock()
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    r = remote_mod.S3Remote("s3://bucket/prefix")
    await r.upload_file(test_file, "abc123def4567890")

    mock_client.put_object.assert_called_once()
    call_kwargs = mock_client.put_object.call_args[1]
    assert call_kwargs["Bucket"] == "bucket"
    assert call_kwargs["Key"] == "prefix/files/ab/c123def4567890"


async def test_download_file(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """download_file gets object from S3."""
    mock_client = mocker.AsyncMock()
    mock_client.get_object = mocker.AsyncMock(
        return_value={"Body": MockBody(b"downloaded content")}
    )
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    dest_file = tmp_path / "dest.txt"

    r = remote_mod.S3Remote("s3://bucket/prefix")
    await r.download_file("abc123def4567890", dest_file)

    assert dest_file.read_bytes() == b"downloaded content"


async def test_download_file_readonly(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """download_file with readonly=True sets 0o444 permissions (for cache files)."""
    mock_client = mocker.AsyncMock()
    mock_client.get_object = mocker.AsyncMock(return_value={"Body": MockBody(b"cached content")})
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    dest_file = tmp_path / "cached.txt"

    r = remote_mod.S3Remote("s3://bucket/prefix")
    await r.download_file("abc123def4567890", dest_file, readonly=True)

    assert dest_file.read_bytes() == b"cached content"
    # Verify read-only permissions (0o444)
    mode = dest_file.stat().st_mode & 0o777
    assert mode == 0o444, f"Expected 0o444, got {oct(mode)}"


async def test_bulk_exists(mock_s3_session: MagicMock, mocker: MockerFixture) -> None:
    """bulk_exists checks multiple hashes in parallel."""

    def mock_head_side_effect(**kwargs: str) -> dict[str, str]:
        key = kwargs.get("Key", "")
        # Key format is prefix/files/XX/YYYYYY...
        # b2c3d4e5f6789ab1 becomes prefix/files/b2/c3d4e5f6789ab1
        if "/b2/" in key:
            raise botocore_exc.ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
            )
        return {}

    mock_client = mocker.AsyncMock()
    mock_client.head_object = mocker.AsyncMock(side_effect=mock_head_side_effect)
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    r = remote_mod.S3Remote("s3://bucket/prefix")
    result = await r.bulk_exists(["a1b2c3d4e5f6789a", "b2c3d4e5f6789ab1", "c3d4e5f6789ab1c2"])

    assert result["a1b2c3d4e5f6789a"] is True
    assert result["b2c3d4e5f6789ab1"] is False
    assert result["c3d4e5f6789ab1c2"] is True
    assert mock_client.head_object.call_count == 3


async def test_bulk_exists_raises_on_non_404_error(
    mock_s3_session: MagicMock, mocker: MockerFixture
) -> None:
    """bulk_exists raises RemoteConnectionError on non-404 errors."""

    def mock_head_side_effect(**kwargs: str) -> dict[str, str]:
        key = kwargs.get("Key", "")
        # Simulate 403 Forbidden for b2c3d4e5f6789ab1 (key has /b2/)
        if "/b2/" in key:
            raise botocore_exc.ClientError(
                {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject"
            )
        return {}

    mock_client = mocker.AsyncMock()
    mock_client.head_object = mocker.AsyncMock(side_effect=mock_head_side_effect)
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    r = remote_mod.S3Remote("s3://bucket/prefix")

    with pytest.raises(exceptions.RemoteConnectionError, match="bulk_exists failed"):
        await r.bulk_exists(["a1b2c3d4e5f6789a", "b2c3d4e5f6789ab1", "c3d4e5f6789ab1c2"])


async def test_bulk_exists_uses_list_for_large_batches(
    mock_s3_session: MagicMock, mocker: MockerFixture
) -> None:
    """bulk_exists uses LIST instead of HEAD for large batches (>= 50 hashes)."""

    # Generate 60 hashes to exceed the LIST threshold (50)
    hashes = [f"a{i:015x}" for i in range(60)]
    # Mark some as existing on remote (every other one)
    existing_hashes = set(hashes[::2])

    class MockPaginator:
        async def paginate(
            self,
            Bucket: str,  # noqa: N803
            Prefix: str,  # noqa: N803
        ) -> AsyncIterator[dict[str, list[dict[str, str]]]]:
            # Extract the prefix part (e.g., "a0" from "prefix/files/a0/")
            hash_prefix = Prefix.split("/")[-2] if Prefix.endswith("/") else ""
            # Return only existing hashes that match this prefix
            contents = [
                {"Key": f"prefix/files/{h[:2]}/{h[2:]}"}
                for h in existing_hashes
                if h.startswith(hash_prefix)
            ]
            yield {"Contents": contents}

    mock_client = mocker.AsyncMock()
    mock_client.get_paginator = mocker.Mock(return_value=MockPaginator())
    # HEAD should NOT be called (we use LIST for large batches)
    mock_client.head_object = mocker.AsyncMock()
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    r = remote_mod.S3Remote("s3://bucket/prefix")
    result = await r.bulk_exists(hashes)

    # Verify results
    for h in existing_hashes:
        assert result[h] is True, f"Hash {h} should exist"
    for h in set(hashes) - existing_hashes:
        assert result[h] is False, f"Hash {h} should not exist"

    # HEAD should NOT have been called (we use LIST for large batches)
    assert mock_client.head_object.call_count == 0
    # get_paginator should have been called
    assert mock_client.get_paginator.call_count > 0


async def test_list_hashes(mock_s3_session: MagicMock, mocker: MockerFixture) -> None:
    """list_hashes returns all cache hashes from S3."""

    class MockPaginator:
        async def paginate(
            self, **kwargs: object
        ) -> AsyncIterator[dict[str, list[dict[str, str]]]]:
            pages = [
                {
                    "Contents": [
                        {"Key": "prefix/files/ab/c123def4567890"},
                        {"Key": "prefix/files/de/f456abc1234567"},
                    ]
                },
                {
                    "Contents": [
                        {"Key": "prefix/files/12/3456789abcdef0"},
                    ]
                },
            ]
            for page in pages:
                yield page

    mock_client = mocker.AsyncMock()
    mock_client.get_paginator = mocker.MagicMock(return_value=MockPaginator())
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    r = remote_mod.S3Remote("s3://bucket/prefix")
    result = await r.list_hashes()

    assert result == {"abc123def4567890", "def456abc1234567", "123456789abcdef0"}


async def test_iter_hashes(mock_s3_session: MagicMock, mocker: MockerFixture) -> None:
    """iter_hashes yields hashes without collecting into memory."""

    class MockPaginator:
        async def paginate(
            self, **kwargs: object
        ) -> AsyncIterator[dict[str, list[dict[str, str]]]]:
            pages = [
                {"Contents": [{"Key": "prefix/files/ab/c123def4567890"}]},
                {"Contents": [{"Key": "prefix/files/de/f456abc1234567"}]},
            ]
            for page in pages:
                yield page

    mock_client = mocker.AsyncMock()
    mock_client.get_paginator = mocker.MagicMock(return_value=MockPaginator())
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    r = remote_mod.S3Remote("s3://bucket/prefix")
    hashes = []
    async for h in r.iter_hashes():
        hashes.append(h)

    assert hashes == ["abc123def4567890", "def456abc1234567"]


async def test_iter_hashes_skips_invalid_keys(
    mock_s3_session: MagicMock, mocker: MockerFixture
) -> None:
    """iter_hashes skips keys that don't produce valid 16-char lowercase hex hashes."""

    class MockPaginator:
        async def paginate(
            self, **kwargs: object
        ) -> AsyncIterator[dict[str, list[dict[str, str]]]]:
            yield {
                "Contents": [
                    {"Key": "prefix/files/ab/cdef1234567890"},  # Valid: 16 lowercase hex
                    {"Key": "prefix/files/AB/CDEF1234567890"},  # Invalid: uppercase
                    {"Key": "prefix/files/ab/c"},  # Invalid: too short
                    {"Key": "prefix/files/ab/cdef12345678901234"},  # Invalid: too long
                    {"Key": "prefix/stages/my_stage.lock"},  # Invalid: not a cache file
                ]
            }

    mock_client = mocker.AsyncMock()
    mock_client.get_paginator = mocker.MagicMock(return_value=MockPaginator())
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    r = remote_mod.S3Remote("s3://bucket/prefix")
    hashes = []
    async for h in r.iter_hashes():
        hashes.append(h)

    assert hashes == ["abcdef1234567890"]


async def test_upload_batch(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """upload_batch uploads multiple files in parallel."""
    mock_client = mocker.AsyncMock()
    mock_client.put_object = mocker.AsyncMock()
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    files = list[tuple[pathlib.Path, str]]()
    for i in range(3):
        f = tmp_path / f"file{i}.txt"
        f.write_text(f"content {i}")
        files.append((f, f"a{i}b2c3d4e5f6789a"))

    r = remote_mod.S3Remote("s3://bucket/prefix")
    results = await r.upload_batch(files, concurrency=10)

    assert len(results) == 3
    assert all(r["success"] for r in results)


async def test_upload_batch_with_callback(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """upload_batch calls callback for each completed upload."""
    mock_client = mocker.AsyncMock()
    mock_client.put_object = mocker.AsyncMock()
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    files = list[tuple[pathlib.Path, str]]()
    for i in range(3):
        f = tmp_path / f"file{i}.txt"
        f.write_text(f"content {i}")
        files.append((f, f"a{i}b2c3d4e5f6789a"))

    callback_values = list[int]()

    def callback(n: int) -> None:
        callback_values.append(n)

    r = remote_mod.S3Remote("s3://bucket/prefix")
    await r.upload_batch(files, concurrency=10, callback=callback)

    assert len(callback_values) == 3
    assert set(callback_values) == {1, 2, 3}


async def test_upload_batch_empty() -> None:
    """upload_batch with empty list returns empty results."""
    r = remote_mod.S3Remote("s3://bucket/prefix")
    results = await r.upload_batch([])

    assert results == []


async def test_download_batch(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """download_batch downloads multiple files in parallel."""
    mock_client = mocker.AsyncMock()
    mock_client.get_object = mocker.AsyncMock(side_effect=_make_mock_get_object_response)
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    items = [(f"a{i}b2c3d4e5f6789a", tmp_path / f"dest{i}.txt") for i in range(3)]

    r = remote_mod.S3Remote("s3://bucket/prefix")
    results = await r.download_batch(items, concurrency=10)

    assert len(results) == 3
    assert all(r["success"] for r in results)
    for _, path in items:
        assert path.exists()


async def test_download_batch_empty() -> None:
    """download_batch with empty list returns empty results."""
    r = remote_mod.S3Remote("s3://bucket/prefix")
    results = await r.download_batch([])

    assert results == []


# -----------------------------------------------------------------------------
# Multipart Upload Tests (Large Files)
# -----------------------------------------------------------------------------


async def test_upload_file_large_uses_multipart(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """upload_file uses multipart upload for large files."""
    # Monkeypatch chunk size to 100 bytes to avoid writing large files
    mocker.patch.object(remote_mod, "STREAM_CHUNK_SIZE", 100)

    mock_client = mocker.AsyncMock()
    mock_client.create_multipart_upload = mocker.AsyncMock(
        return_value={"UploadId": "test-upload-id"}
    )
    mock_client.upload_part = mocker.AsyncMock(return_value={"ETag": "test-etag"})
    mock_client.complete_multipart_upload = mocker.AsyncMock()
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    # Create file larger than patched STREAM_CHUNK_SIZE (100 bytes)
    test_file = tmp_path / "large_file.bin"
    test_file.write_bytes(b"x" * 250)  # 250 bytes = 3 parts (100 + 100 + 50)

    r = remote_mod.S3Remote("s3://bucket/prefix")
    await r.upload_file(test_file, "abc123def4567890")

    mock_client.create_multipart_upload.assert_called_once()
    assert mock_client.upload_part.call_count == 3  # 250 bytes = 3 parts at 100 byte chunks
    mock_client.complete_multipart_upload.assert_called_once()


async def test_upload_file_small_uses_put_object(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """upload_file uses simple put_object for small files."""
    mock_client = mocker.AsyncMock()
    mock_client.put_object = mocker.AsyncMock()
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    test_file = tmp_path / "small_file.txt"
    test_file.write_text("small content")

    r = remote_mod.S3Remote("s3://bucket/prefix")
    await r.upload_file(test_file, "abc123def4567890")

    mock_client.put_object.assert_called_once()


async def test_upload_file_multipart_aborts_on_error(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
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
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    test_file = tmp_path / "large_file.bin"
    test_file.write_bytes(b"x" * 150)  # Above 100 byte threshold

    r = remote_mod.S3Remote("s3://bucket/prefix")
    with pytest.raises(Exception, match="Upload failed"):
        await r.upload_file(test_file, "abc123def4567890")

    mock_client.abort_multipart_upload.assert_called_once()


# -----------------------------------------------------------------------------
# Atomic Download Tests
# -----------------------------------------------------------------------------


async def test_download_file_cleans_up_on_error(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """download_file cleans up temp file on error."""

    mock_client = mocker.AsyncMock()
    mock_client.get_object = mocker.AsyncMock(side_effect=Exception("Download failed"))
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    dest_file = tmp_path / "dest.txt"

    r = remote_mod.S3Remote("s3://bucket/prefix")
    with pytest.raises(Exception, match="Download failed"):
        await r.download_file("abc123def4567890", dest_file)

    # Verify no temp files left behind
    temp_files = [f for f in os.listdir(tmp_path) if f.startswith(".pivot_download_")]
    assert len(temp_files) == 0, f"Temp files not cleaned up: {temp_files}"


async def test_download_batch_with_callback(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """download_batch calls callback for each completed download."""
    mock_client = mocker.AsyncMock()
    mock_client.get_object = mocker.AsyncMock(side_effect=_make_mock_get_object_response)
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    items = [(f"a{i}b2c3d4e5f6789a", tmp_path / f"dest{i}.txt") for i in range(3)]

    callback_values = list[int]()

    def callback(n: int) -> None:
        callback_values.append(n)

    r = remote_mod.S3Remote("s3://bucket/prefix")
    await r.download_batch(items, concurrency=10, callback=callback)

    assert len(callback_values) == 3
    assert set(callback_values) == {1, 2, 3}


async def test_download_file_default_permissions(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """download_file without readonly flag creates file with default permissions."""
    mock_client = mocker.AsyncMock()
    mock_client.get_object = mocker.AsyncMock(return_value={"Body": MockBody(b"writable content")})
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    dest_file = tmp_path / "writable.txt"

    r = remote_mod.S3Remote("s3://bucket/prefix")
    await r.download_file("abc123def4567890", dest_file, readonly=False)

    assert dest_file.read_bytes() == b"writable content"
    # Verify file is writable (not 0o444)
    mode = dest_file.stat().st_mode & 0o777
    assert mode != 0o444, f"Expected writable permissions, got {oct(mode)}"
    # Default umask typically results in 0o644 or similar
    assert mode & 0o200, f"Expected owner write permission, got {oct(mode)}"


async def test_download_batch_readonly(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """download_batch with readonly=True sets 0o444 permissions on all files."""
    mock_client = mocker.AsyncMock()
    mock_client.get_object = mocker.AsyncMock(side_effect=_make_mock_get_object_response)
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    items = [(f"a{i}b2c3d4e5f6789a", tmp_path / f"cached{i}.txt") for i in range(3)]

    r = remote_mod.S3Remote("s3://bucket/prefix")
    results = await r.download_batch(items, concurrency=10, readonly=True)

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
    mock_s3_session: MagicMock, mocker: MockerFixture
) -> None:
    """bulk_exists calls metrics.end even when an error occurs."""
    from pivot import metrics

    error = botocore_exc.ClientError(
        {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject"
    )
    mock_client = mocker.AsyncMock()
    mock_client.head_object = mocker.AsyncMock(side_effect=error)
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    spy_end = mocker.spy(metrics, "end")

    r = remote_mod.S3Remote("s3://bucket/prefix")

    with pytest.raises(exceptions.RemoteConnectionError):
        await r.bulk_exists(["a1b2c3d4e5f6789a"])

    spy_end.assert_any_call("storage.bulk_exists", mocker.ANY)


async def test_upload_batch_calls_metrics_end_on_error(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """upload_batch calls metrics.end even when uploads fail."""
    from pivot import metrics

    mock_client = mocker.AsyncMock()
    mock_client.put_object = mocker.AsyncMock(side_effect=Exception("upload boom"))
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    spy_end = mocker.spy(metrics, "end")

    f = tmp_path / "file.txt"
    f.write_text("content")

    r = remote_mod.S3Remote("s3://bucket/prefix")
    # upload_batch catches per-item errors, so it completes without raising
    await r.upload_batch([(f, "a1b2c3d4e5f6789a")])

    spy_end.assert_any_call("storage.upload_batch", mocker.ANY)


async def test_download_batch_calls_metrics_end_on_error(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """download_batch calls metrics.end even when downloads fail."""
    from pivot import metrics

    mock_client = mocker.AsyncMock()
    mock_client.get_object = mocker.AsyncMock(side_effect=Exception("download boom"))
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    spy_end = mocker.spy(metrics, "end")

    r = remote_mod.S3Remote("s3://bucket/prefix")
    await r.download_batch([("a1b2c3d4e5f6789a", tmp_path / "dest.txt")])

    spy_end.assert_any_call("storage.download_batch", mocker.ANY)


# -----------------------------------------------------------------------------
# Session Reuse Tests
# -----------------------------------------------------------------------------


async def test_s3_remote_reuses_session_across_methods(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """S3Remote reuses the same aioboto3 session across multiple method calls."""
    mock_client = mocker.AsyncMock()
    mock_client.head_object = mocker.AsyncMock(return_value={})
    mock_client.put_object = mocker.AsyncMock()
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    r = remote_mod.S3Remote("s3://bucket/prefix")
    await r.exists("abc123def4567890")

    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
    await r.upload_file(test_file, "abc123def4567890")

    # Session created once in __init__; both methods share it via self._session.client()
    # The mock_s3_session fixture patches aioboto3.Session — verify it was called once
    import aioboto3

    session_cls = aioboto3.Session  # type: ignore[attr-defined] - mocked by fixture
    assert session_cls.call_count == 1  # pyright: ignore[reportAttributeAccessIssue] - mock attribute


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
