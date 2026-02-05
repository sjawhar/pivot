from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any

import pytest
from botocore import exceptions as botocore_exc

from pivot import exceptions
from pivot.remote import storage as remote_mod

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture


class MockStreamContent:
    _data: bytes
    _read_called: bool

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._read_called = False

    async def read(self, size: int = -1) -> bytes:
        if self._read_called:
            return b""
        self._read_called = True
        return self._data


class MockBody:
    content: MockStreamContent

    def __init__(self, data: bytes = b"content") -> None:
        self.content = MockStreamContent(data)

    async def __aenter__(self) -> MockBody:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


def _make_mock_body(data: bytes = b"content") -> MockBody:
    return MockBody(data)


def _make_mock_body_factory() -> Any:
    def factory(**_: Any) -> dict[str, Any]:
        return {"Body": MockBody()}

    return factory


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


def test_validate_hash_short_raises() -> None:
    """Short hash raises RemoteError."""
    with pytest.raises(exceptions.RemoteError, match="must be at least"):
        remote_mod._validate_hash("")
    with pytest.raises(exceptions.RemoteError, match="must be at least"):
        remote_mod._validate_hash("ab")


def test_validate_hash_non_hex_raises() -> None:
    """Non-hexadecimal hash raises RemoteError."""
    with pytest.raises(exceptions.RemoteError, match="must be hexadecimal"):
        remote_mod._validate_hash("ghijkl")  # g-z are not hex
    with pytest.raises(exceptions.RemoteError, match="must be hexadecimal"):
        remote_mod._validate_hash("abc/def")  # Contains path separator
    with pytest.raises(exceptions.RemoteError, match="must be hexadecimal"):
        remote_mod._validate_hash("abc def")  # Contains space


def test_validate_hash_valid() -> None:
    """Valid hash passes validation."""
    remote_mod._validate_hash("abc")  # Minimum length
    remote_mod._validate_hash("abcdef1234567890")  # Typical length
    remote_mod._validate_hash("ABCDEF")  # Uppercase hex is valid
    remote_mod._validate_hash("0123456789abcdef")  # All hex chars


def test_is_not_found_error_true() -> None:
    """_is_not_found_error returns True for 404."""
    error = type("MockError", (), {"response": {"Error": {"Code": "404"}}})()
    assert remote_mod._is_not_found_error(error) is True


def test_is_not_found_error_false() -> None:
    """_is_not_found_error returns False for non-404."""
    error_403 = type("MockError", (), {"response": {"Error": {"Code": "403"}}})()
    assert remote_mod._is_not_found_error(error_403) is False

    error_empty = type("MockError", (), {"response": {}})()
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


@pytest.mark.asyncio
async def test_exists_true(mock_s3_session: MagicMock, mocker: MockerFixture) -> None:
    """exists returns True when object exists."""
    mock_client = mocker.AsyncMock()
    mock_client.head_object = mocker.AsyncMock(return_value={})
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    r = remote_mod.S3Remote("s3://bucket/prefix")
    result = await r.exists("abc123def45678")

    assert result is True
    mock_client.head_object.assert_called_once()


@pytest.mark.asyncio
async def test_exists_false_404(mock_s3_session: MagicMock, mocker: MockerFixture) -> None:
    """exists returns False on 404 error."""
    mock_client = mocker.AsyncMock()
    error = botocore_exc.ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
    )
    mock_client.head_object = mocker.AsyncMock(side_effect=error)
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    r = remote_mod.S3Remote("s3://bucket/prefix")
    result = await r.exists("abc123def45678")

    assert result is False


@pytest.mark.asyncio
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
        await r.exists("abc123def45678")


@pytest.mark.asyncio
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
    await r.upload_file(test_file, "abc123def45678")

    mock_client.put_object.assert_called_once()
    call_kwargs = mock_client.put_object.call_args[1]
    assert call_kwargs["Bucket"] == "bucket"
    assert call_kwargs["Key"] == "prefix/files/ab/c123def45678"


@pytest.mark.asyncio
async def test_download_file(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """download_file gets object from S3."""
    mock_client = mocker.AsyncMock()
    mock_client.get_object = mocker.AsyncMock(
        return_value={"Body": _make_mock_body(b"downloaded content")}
    )
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    dest_file = tmp_path / "dest.txt"

    r = remote_mod.S3Remote("s3://bucket/prefix")
    await r.download_file("abc123def45678", dest_file)

    assert dest_file.read_bytes() == b"downloaded content"


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_bulk_exists_uses_list_for_large_batches(
    mock_s3_session: MagicMock, mocker: MockerFixture
) -> None:
    """bulk_exists uses LIST instead of HEAD for large batches (>= 50 hashes)."""
    from collections.abc import AsyncIterator  # noqa: TC003

    # Generate 60 hashes to exceed the LIST threshold (50)
    hashes = [f"a{i:015x}" for i in range(60)]
    # Mark some as existing on remote (every other one)
    existing_hashes = set(hashes[::2])

    class MockPaginator:
        async def paginate(
            self,
            Bucket: str,  # noqa: N803
            Prefix: str,  # noqa: N803
        ) -> AsyncIterator[dict[str, Any]]:
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


@pytest.mark.asyncio
async def test_list_hashes(mock_s3_session: MagicMock, mocker: MockerFixture) -> None:
    """list_hashes returns all cache hashes from S3."""
    from collections.abc import AsyncIterator  # noqa: TC003

    class MockPaginator:
        async def paginate(
            self, **kwargs: object
        ) -> AsyncIterator[dict[str, list[dict[str, str]]]]:
            pages = [
                {
                    "Contents": [
                        {"Key": "prefix/files/ab/c123def45678"},
                        {"Key": "prefix/files/de/f456abc12345"},
                    ]
                },
                {
                    "Contents": [
                        {"Key": "prefix/files/12/3456789abcde"},
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

    assert result == {"abc123def45678", "def456abc12345", "123456789abcde"}


@pytest.mark.asyncio
async def test_iter_hashes(mock_s3_session: MagicMock, mocker: MockerFixture) -> None:
    """iter_hashes yields hashes without collecting into memory."""
    from collections.abc import AsyncIterator  # noqa: TC003

    class MockPaginator:
        async def paginate(
            self, **kwargs: object
        ) -> AsyncIterator[dict[str, list[dict[str, str]]]]:
            pages = [
                {"Contents": [{"Key": "prefix/files/ab/c123def45678"}]},
                {"Contents": [{"Key": "prefix/files/de/f456abc12345"}]},
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

    assert hashes == ["abc123def45678", "def456abc12345"]


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_upload_batch_empty() -> None:
    """upload_batch with empty list returns empty results."""
    r = remote_mod.S3Remote("s3://bucket/prefix")
    results = await r.upload_batch([])

    assert results == []


@pytest.mark.asyncio
async def test_download_batch(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """download_batch downloads multiple files in parallel."""
    mock_client = mocker.AsyncMock()
    mock_client.get_object = mocker.AsyncMock(side_effect=_make_mock_body_factory())
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    items = [(f"a{i}b2c3d4e5f6789a", tmp_path / f"dest{i}.txt") for i in range(3)]

    r = remote_mod.S3Remote("s3://bucket/prefix")
    results = await r.download_batch(items, concurrency=10)

    assert len(results) == 3
    assert all(r["success"] for r in results)
    for _, path in items:
        assert path.exists()


@pytest.mark.asyncio
async def test_download_batch_empty() -> None:
    """download_batch with empty list returns empty results."""
    r = remote_mod.S3Remote("s3://bucket/prefix")
    results = await r.download_batch([])

    assert results == []


# -----------------------------------------------------------------------------
# Multipart Upload Tests (Large Files)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
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
    await r.upload_file(test_file, "abc123def45678")

    mock_client.create_multipart_upload.assert_called_once()
    assert mock_client.upload_part.call_count == 3  # 250 bytes = 3 parts at 100 byte chunks
    mock_client.complete_multipart_upload.assert_called_once()


@pytest.mark.asyncio
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
    await r.upload_file(test_file, "abc123def45678")

    mock_client.put_object.assert_called_once()


@pytest.mark.asyncio
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
        await r.upload_file(test_file, "abc123def45678")

    mock_client.abort_multipart_upload.assert_called_once()


# -----------------------------------------------------------------------------
# Atomic Download Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_file_cleans_up_on_error(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """download_file cleans up temp file on error."""
    import os

    mock_client = mocker.AsyncMock()
    mock_client.get_object = mocker.AsyncMock(side_effect=Exception("Download failed"))
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    dest_file = tmp_path / "dest.txt"

    r = remote_mod.S3Remote("s3://bucket/prefix")
    with pytest.raises(Exception, match="Download failed"):
        await r.download_file("abc123def45678", dest_file)

    # Verify no temp files left behind
    temp_files = [f for f in os.listdir(tmp_path) if f.startswith(".pivot_download_")]
    assert len(temp_files) == 0, f"Temp files not cleaned up: {temp_files}"


@pytest.mark.asyncio
async def test_download_batch_with_callback(
    mock_s3_session: MagicMock, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """download_batch calls callback for each completed download."""
    mock_client = mocker.AsyncMock()
    mock_client.get_object = mocker.AsyncMock(side_effect=_make_mock_body_factory())
    mock_s3_session.client.return_value.__aenter__.return_value = mock_client

    items = [(f"a{i}b2c3d4e5f6789a", tmp_path / f"dest{i}.txt") for i in range(3)]

    callback_values = list[int]()

    def callback(n: int) -> None:
        callback_values.append(n)

    r = remote_mod.S3Remote("s3://bucket/prefix")
    await r.download_batch(items, concurrency=10, callback=callback)

    assert len(callback_values) == 3
    assert set(callback_values) == {1, 2, 3}
