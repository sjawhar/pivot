"""GCSRemote against fake-gcs-server, case for case with test_s3_remote.py."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import pathlib
from typing import TYPE_CHECKING, Any

import aiohttp
import multidict
import pytest
import xxhash
import yarl
from gcloud.aio import storage as gcloud_storage

from pivot import config, exceptions
from pivot.remote import gcs, keys
from pivot.storage import cache

if TYPE_CHECKING:
    import types

    from pytest_mock import MockerFixture

_OCTET_STREAM = "application/octet-stream"


def _helper_hash(content: bytes) -> str:
    """Content-addressed key for content, as pivot's cache computes it."""
    return xxhash.xxh64(content).hexdigest()


def _response_error(status: int, message: str) -> aiohttp.ClientResponseError:
    """The error gcloud-aio raises for an HTTP error status, carrying Google's JSON body."""
    url = yarl.URL("https://storage.googleapis.com/storage/v1/b/bucket/o/key")
    request_info = aiohttp.RequestInfo(
        url=url,
        method="GET",
        headers=multidict.CIMultiDictProxy(multidict.CIMultiDict[str]()),
        real_url=url,
    )
    body = json.dumps({"error": {"code": status, "message": message}})
    return aiohttp.ClientResponseError(request_info, (), status=status, message=f"Reason: {body}")


async def _put(
    client: gcloud_storage.Storage, remote: gcs.GCSRemote, hash_: str, body: bytes
) -> None:
    """Store body on fake-gcs-server where ``remote`` keeps the cache object ``hash_``."""
    key = keys.hash_to_key(remote.prefix, hash_)
    await client.upload(remote.bucket, key, body, content_type=_OCTET_STREAM)


async def _get(client: gcloud_storage.Storage, remote: gcs.GCSRemote, hash_: str) -> bytes:
    return await client.download(remote.bucket, keys.hash_to_key(remote.prefix, hash_))


async def _object_names(client: gcloud_storage.Storage, bucket: str) -> list[str]:
    listing = await client.list_objects(bucket)
    return sorted(item["name"] for item in listing.get("items", []))


@pytest.fixture
def no_retries(mocker: MockerFixture) -> None:
    """remote.retries = 0, read when a GCSRemote is constructed."""
    mocker.patch.object(config, "get_remote_retries", return_value=0)


# -----------------------------------------------------------------------------
# GCSRemote Initialization Tests
# -----------------------------------------------------------------------------


def test_gcs_remote_init_basic() -> None:
    """GCSRemote parses bucket and prefix from URL."""
    r = gcs.GCSRemote("gs://my-bucket/my-prefix")
    assert r.bucket == "my-bucket"
    assert r.prefix == "my-prefix/"


def test_gcs_remote_init_no_prefix() -> None:
    """GCSRemote handles URL without prefix."""
    r = gcs.GCSRemote("gs://my-bucket")
    assert r.bucket == "my-bucket"
    assert r.prefix == ""


def test_gcs_remote_init_nested_prefix() -> None:
    """GCSRemote handles nested prefix path."""
    r = gcs.GCSRemote("gs://bucket/path/to/cache")
    assert r.bucket == "bucket"
    assert r.prefix == "path/to/cache/"


def test_gcs_remote_init_invalid_url() -> None:
    """GCSRemote raises on invalid URL."""
    with pytest.raises(exceptions.InvalidRemoteURLError):
        gcs.GCSRemote("not-a-gs-url")


def test_gcs_remote_init_rejects_s3_url() -> None:
    """GCSRemote refuses an s3:// URL rather than treating its bucket as a GCS bucket."""
    with pytest.raises(exceptions.InvalidRemoteURLError, match="gs://bucket/prefix"):
        gcs.GCSRemote("s3://bucket/prefix")


@pytest.mark.parametrize(
    "input_url,expected_url",
    [
        pytest.param("gs://bucket/prefix", "gs://bucket/prefix", id="basic_prefix"),
        pytest.param("gs://bucket", "gs://bucket", id="no_prefix"),
        pytest.param("gs://bucket/path/to/cache", "gs://bucket/path/to/cache", id="nested_prefix"),
        pytest.param("gs://bucket/prefix/", "gs://bucket/prefix", id="trailing_slash_stripped"),
    ],
)
def test_gcs_remote_url_normalization(input_url: str, expected_url: str) -> None:
    """GCSRemote.url property normalizes URLs consistently."""
    r = gcs.GCSRemote(input_url)
    assert r.url == expected_url, f"URL normalization failed for {input_url}"


def test_gcs_remote_reads_retries_and_connect_timeout_from_config(mocker: MockerFixture) -> None:
    """remote.retries and remote.connect_timeout configure every GCS request."""
    mocker.patch.object(config, "get_remote_retries", return_value=3)
    mocker.patch.object(config, "get_remote_connect_timeout", return_value=7)

    r = gcs.GCSRemote("gs://bucket/prefix")

    assert r._retries == 3
    assert r._timeout.sock_connect == 7
    assert r._timeout.sock_read == gcs.STREAM_READ_TIMEOUT
    assert r._timeout.total is None, "a large transfer must not hit a whole-request deadline"


def test_gcs_remote_init_raises_on_missing_gcloud_aio_storage(mocker: MockerFixture) -> None:
    """GCSRemote raises RemoteError naming the extra when gcloud-aio-storage is not installed."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> types.ModuleType:  # noqa: ANN401 - wraps builtins.__import__ which requires Any for forwarded args
        if name == "gcloud.aio" or name.startswith("gcloud.aio.storage"):
            raise ModuleNotFoundError("No module named 'gcloud.aio.storage'")
        return original_import(name, *args, **kwargs)

    mocker.patch("builtins.__import__", side_effect=mock_import)

    with pytest.raises(exceptions.RemoteError, match="pip install pivot\\[gcs\\]"):
        gcs.GCSRemote("gs://bucket/prefix")


# -----------------------------------------------------------------------------
# Object Key Layout
# -----------------------------------------------------------------------------


async def test_object_key_is_prefix_files_two_char_dir_and_rest(
    gcs_remote: gcs.GCSRemote, gcs_client: gcloud_storage.Storage, tmp_path: pathlib.Path
) -> None:
    """A pushed object lands at <prefix>files/<hash[:2]>/<hash[2:]>, the layout S3 uses."""
    content = b"layout"
    cache_hash = _helper_hash(content)
    local = tmp_path / "f"
    local.write_bytes(content)

    await gcs_remote.upload_file(local, cache_hash)

    assert await _object_names(gcs_client, gcs_remote.bucket) == [
        f"test-prefix/files/{cache_hash[:2]}/{cache_hash[2:]}"
    ]


# -----------------------------------------------------------------------------
# exists
# -----------------------------------------------------------------------------


async def test_exists_true(gcs_remote: gcs.GCSRemote, gcs_client: gcloud_storage.Storage) -> None:
    """exists returns True when object exists."""
    cache_hash = "abcdef1234567890"
    await _put(gcs_client, gcs_remote, cache_hash, b"test")

    assert await gcs_remote.exists(cache_hash) is True


async def test_exists_false_404(gcs_remote: gcs.GCSRemote) -> None:
    """exists returns False on 404."""
    assert await gcs_remote.exists("abcdef1234567890") is False


async def test_exists_permission_denied_names_object_and_permission(
    gcs_remote: gcs.GCSRemote, mocker: MockerFixture
) -> None:
    """exists raises RemoteConnectionError carrying Google's permission message on 403."""
    error = _response_error(
        403, "u@x.dev does not have storage.objects.get access to the Google Cloud Storage object."
    )
    mocker.patch.object(
        gcloud_storage.Storage, "download_metadata", new=mocker.AsyncMock(side_effect=error)
    )

    with pytest.raises(exceptions.RemoteConnectionError) as exc_info:
        await gcs_remote.exists("abcdef1234567890")

    message = str(exc_info.value)
    assert f"gs://{gcs_remote.bucket}/test-prefix/files/ab/cdef1234567890" in message
    assert "storage.objects.get" in message


async def test_gcs_exists_auth_error_expired_credentials(
    gcs_remote: gcs.GCSRemote, mocker: MockerFixture
) -> None:
    """exists raises RemoteConnectionError with the ADC refresh command on 401."""
    error = _response_error(401, "Invalid Credentials")
    mocker.patch.object(
        gcloud_storage.Storage, "download_metadata", new=mocker.AsyncMock(side_effect=error)
    )

    with pytest.raises(
        exceptions.RemoteConnectionError, match="gcloud auth application-default login"
    ):
        await gcs_remote.exists("abcdef1234567890")


async def test_gcs_exists_auth_error_no_credentials(
    monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """With no usable Application Default Credentials, exists fails before any request."""
    monkeypatch.delenv("STORAGE_EMULATOR_HOST", raising=False)
    from gcloud.aio import auth as gcloud_auth

    mocker.patch.object(
        gcloud_auth.Token, "get", new=mocker.AsyncMock(side_effect=OSError("no metadata server"))
    )
    metadata = mocker.patch.object(gcloud_storage.Storage, "download_metadata")
    r = gcs.GCSRemote("gs://bucket/prefix")

    with pytest.raises(
        exceptions.RemoteConnectionError, match="gcloud auth application-default login"
    ):
        await r.exists("abcdef1234567890")

    metadata.assert_not_called()


async def test_gcs_exists_non_auth_error_still_generic(
    no_retries: None, gcs_remote: gcs.GCSRemote, mocker: MockerFixture
) -> None:
    """exists raises RemoteConnectionError with a generic GCS message for other errors."""
    error = _response_error(500, "Internal server error")
    mocker.patch.object(
        gcloud_storage.Storage, "download_metadata", new=mocker.AsyncMock(side_effect=error)
    )

    with pytest.raises(exceptions.RemoteConnectionError, match="GCS read error.*HTTP 500"):
        await gcs_remote.exists("abcdef1234567890")


# -----------------------------------------------------------------------------
# Retries (remote.retries)
# -----------------------------------------------------------------------------


@pytest.fixture
def instant_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gcs, "_RETRY_BASE_DELAY", 0)


async def test_transient_error_is_retried(
    instant_backoff: None,
    gcs_bucket: str,
    gcs_client: gcloud_storage.Storage,
    mocker: MockerFixture,
) -> None:
    """A 503 is retried, and the retry's answer is the result."""
    mocker.patch.object(config, "get_remote_retries", return_value=2)
    remote = gcs.GCSRemote(f"gs://{gcs_bucket}/test-prefix/")
    cache_hash = "abcdef1234567890"
    await _put(gcs_client, remote, cache_hash, b"content")
    real = gcloud_storage.Storage.download_metadata
    calls = 0

    async def flaky(self: gcloud_storage.Storage, *args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401 - forwards download_metadata's arguments
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _response_error(503, "Backend Error")
        return await real(self, *args, **kwargs)

    mocker.patch.object(gcloud_storage.Storage, "download_metadata", new=flaky)

    assert await remote.exists(cache_hash) is True
    assert calls == 2


@pytest.mark.parametrize("retries", [0, 2])
async def test_retries_stop_after_remote_retries(
    instant_backoff: None, retries: int, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request is tried 1 + remote.retries times, then its error is raised."""
    monkeypatch.setenv("STORAGE_EMULATOR_HOST", "http://127.0.0.1:1")  # no credentials needed
    mocker.patch.object(config, "get_remote_retries", return_value=retries)
    metadata = mocker.patch.object(
        gcloud_storage.Storage,
        "download_metadata",
        new=mocker.AsyncMock(side_effect=_response_error(503, "Backend Error")),
    )
    r = gcs.GCSRemote("gs://bucket/prefix")

    with pytest.raises(exceptions.RemoteConnectionError, match="HTTP 503"):
        await r.exists("abcdef1234567890")

    assert metadata.await_count == retries + 1


async def test_permanent_error_is_not_retried(
    instant_backoff: None, gcs_remote: gcs.GCSRemote, mocker: MockerFixture
) -> None:
    """A 403 will not change on retry, so it is raised at once."""
    metadata = mocker.patch.object(
        gcloud_storage.Storage,
        "download_metadata",
        new=mocker.AsyncMock(side_effect=_response_error(403, "denied")),
    )

    with pytest.raises(exceptions.RemoteConnectionError):
        await gcs_remote.exists("abcdef1234567890")

    assert metadata.await_count == 1


async def test_requests_use_the_configured_timeouts(
    gcs_remote: gcs.GCSRemote, mocker: MockerFixture
) -> None:
    """Every request gets remote.connect_timeout and the read timeout, not gcloud-aio's 10s total."""
    metadata = mocker.patch.object(
        gcloud_storage.Storage,
        "download_metadata",
        new=mocker.AsyncMock(side_effect=_response_error(404, "No such object")),
    )

    await gcs_remote.exists("abcdef1234567890")

    assert metadata.await_args is not None
    assert metadata.await_args.kwargs["timeout"] is gcs_remote._timeout


# -----------------------------------------------------------------------------
# iter_hashes / list_hashes
# -----------------------------------------------------------------------------


async def test_gcs_iter_hashes_auth_error(gcs_remote: gcs.GCSRemote, mocker: MockerFixture) -> None:
    """iter_hashes raises RemoteConnectionError with the ADC refresh command on 401."""
    mocker.patch.object(
        gcloud_storage.Storage,
        "list_objects",
        new=mocker.AsyncMock(side_effect=_response_error(401, "Invalid Credentials")),
    )

    with pytest.raises(
        exceptions.RemoteConnectionError, match="gcloud auth application-default login"
    ):
        async for _ in gcs_remote.iter_hashes():
            pass


async def test_list_hashes(gcs_remote: gcs.GCSRemote, gcs_client: gcloud_storage.Storage) -> None:
    """list_hashes returns all cache hashes from GCS."""
    hashes = {"abc123def4567890", "def456abc1234567", "123456789abcdef0"}
    for cache_hash in hashes:
        await _put(gcs_client, gcs_remote, cache_hash, b"content")

    assert await gcs_remote.list_hashes() == hashes


async def test_list_hashes_over_several_listing_pages(
    gcs_remote: gcs.GCSRemote,
    gcs_client: gcloud_storage.Storage,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """list_hashes follows nextPageToken past the first listing page."""
    monkeypatch.setattr(gcs, "LIST_PAGE_SIZE", 2)
    hashes = {f"{i:02x}{'0' * 14}" for i in range(5)}
    for cache_hash in hashes:
        await _put(gcs_client, gcs_remote, cache_hash, b"content")
    list_objects = mocker.spy(gcloud_storage.Storage, "list_objects")

    assert await gcs_remote.list_hashes() == hashes
    assert list_objects.call_count == 3


async def test_iter_hashes(gcs_remote: gcs.GCSRemote, gcs_client: gcloud_storage.Storage) -> None:
    """iter_hashes yields hashes without collecting into memory."""
    expected = {"abc123def4567890", "def456abc1234567"}
    for cache_hash in expected:
        await _put(gcs_client, gcs_remote, cache_hash, b"content")

    hashes = [h async for h in gcs_remote.iter_hashes()]

    assert set(hashes) == expected


async def test_iter_hashes_skips_invalid_keys(
    gcs_remote: gcs.GCSRemote, gcs_client: gcloud_storage.Storage
) -> None:
    """iter_hashes skips keys that don't produce valid 16-char lowercase hex hashes."""
    names = [
        f"{gcs_remote.prefix}files/ab/cdef1234567890",  # Valid: 16 lowercase hex
        f"{gcs_remote.prefix}files/AB/CDEF1234567890",  # Invalid: uppercase
        f"{gcs_remote.prefix}files/ab/c",  # Invalid: too short
        f"{gcs_remote.prefix}files/ab/cdef12345678901234",  # Invalid: too long
        f"{gcs_remote.prefix}files/stages/my_stage.lock",  # Invalid: not a cache file
        "other-prefix/files/12/3456789abcdef0",  # Outside the remote's prefix
    ]
    for name in names:
        await gcs_client.upload(gcs_remote.bucket, name, b"x", content_type=_OCTET_STREAM)

    hashes = [h async for h in gcs_remote.iter_hashes()]

    assert hashes == ["abcdef1234567890"]


# -----------------------------------------------------------------------------
# bulk_exists
# -----------------------------------------------------------------------------


async def test_bulk_exists(
    gcs_remote: gcs.GCSRemote, gcs_client: gcloud_storage.Storage, mocker: MockerFixture
) -> None:
    """Below the threshold, bulk_exists asks per object."""
    hashes = ["a1b2c3d4e5f6789a", "b2c3d4e5f6789ab1", "c3d4e5f6789ab1c2"]
    for cache_hash in (hashes[0], hashes[2]):
        await _put(gcs_client, gcs_remote, cache_hash, b"content")
    metadata = mocker.spy(gcloud_storage.Storage, "download_metadata")
    list_objects = mocker.spy(gcloud_storage.Storage, "list_objects")

    result = await gcs_remote.bulk_exists(hashes)

    assert result == {hashes[0]: True, hashes[1]: False, hashes[2]: True}
    assert metadata.call_count == 3
    assert list_objects.call_count == 0


async def test_bulk_exists_raises_on_non_404_error(
    gcs_remote: gcs.GCSRemote, mocker: MockerFixture
) -> None:
    """bulk_exists raises RemoteConnectionError on non-404 errors."""
    mocker.patch.object(
        gcloud_storage.Storage,
        "download_metadata",
        new=mocker.AsyncMock(side_effect=_response_error(403, "Forbidden")),
    )

    with pytest.raises(exceptions.RemoteConnectionError, match="bulk_exists failed"):
        await gcs_remote.bulk_exists(["a1b2c3d4e5f6789a", "b2c3d4e5f6789ab1", "c3d4e5f6789ab1c2"])


async def test_bulk_exists_uses_list_for_large_batches(
    gcs_remote: gcs.GCSRemote, gcs_client: gcloud_storage.Storage, mocker: MockerFixture
) -> None:
    """At or above the threshold (50), bulk_exists lists by prefix instead of asking per object."""
    hashes = [f"a{i:015x}" for i in range(60)]
    existing_hashes = set(hashes[::2])
    for cache_hash in existing_hashes:
        await _put(gcs_client, gcs_remote, cache_hash, b"content")
    metadata = mocker.spy(gcloud_storage.Storage, "download_metadata")

    result = await gcs_remote.bulk_exists(hashes)

    assert result == {h: h in existing_hashes for h in hashes}
    assert metadata.call_count == 0


async def test_bulk_exists_list_raises_on_error(
    gcs_remote: gcs.GCSRemote, mocker: MockerFixture
) -> None:
    """A LIST failure in a large bulk_exists raises RemoteConnectionError."""
    mocker.patch.object(
        gcloud_storage.Storage,
        "list_objects",
        new=mocker.AsyncMock(side_effect=_response_error(403, "Forbidden")),
    )

    with pytest.raises(exceptions.RemoteConnectionError, match="bulk_exists LIST failed"):
        await gcs_remote.bulk_exists([f"a{i:015x}" for i in range(60)])


async def test_bulk_exists_empty(gcs_remote: gcs.GCSRemote) -> None:
    assert await gcs_remote.bulk_exists([]) == {}


# -----------------------------------------------------------------------------
# upload_file
# -----------------------------------------------------------------------------


async def test_gcs_upload_file_auth_error(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """upload_file raises RemoteConnectionError with the ADC refresh command on 401."""
    mocker.patch.object(
        gcloud_storage.Storage,
        "upload",
        new=mocker.AsyncMock(side_effect=_response_error(401, "Invalid Credentials")),
    )
    test_file = tmp_path / "test.txt"
    test_file.write_bytes(b"test content")

    with pytest.raises(
        exceptions.RemoteConnectionError, match="gcloud auth application-default login"
    ):
        await gcs_remote.upload_file(test_file, "abc123def4567890")


async def test_upload_file_permission_denied_names_bucket_and_permission(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """A 403 on upload names the object written and the permission Google says is missing."""
    mocker.patch.object(
        gcloud_storage.Storage,
        "upload",
        new=mocker.AsyncMock(
            side_effect=_response_error(
                403,
                "u@x.dev does not have storage.objects.create access to the Google Cloud "
                + "Storage object. Permission 'storage.objects.create' denied on resource.",
            )
        ),
    )
    test_file = tmp_path / "test.txt"
    test_file.write_bytes(b"test content")

    with pytest.raises(exceptions.RemoteConnectionError) as exc_info:
        await gcs_remote.upload_file(test_file, "abc123def4567890")

    message = str(exc_info.value)
    assert message.startswith(
        f"Permission denied to write gs://{gcs_remote.bucket}/test-prefix/files/ab/c123def4567890"
    )
    assert "Permission 'storage.objects.create' denied" in message


async def test_upload_file(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path, gcs_client: gcloud_storage.Storage
) -> None:
    """upload_file puts object to GCS."""
    cache_hash = "abc123def4567890"
    test_file = tmp_path / "test.txt"
    test_file.write_bytes(b"test content")

    await gcs_remote.upload_file(test_file, cache_hash)

    assert await _get(gcs_client, gcs_remote, cache_hash) == b"test content"


async def test_upload_file_large_uses_resumable(
    gcs_remote: gcs.GCSRemote,
    tmp_path: pathlib.Path,
    gcs_client: gcloud_storage.Storage,
    mocker: MockerFixture,
) -> None:
    """A file above 8 MiB goes up through a resumable upload session."""
    test_file = tmp_path / "large_file.bin"
    large_payload = b"x" * (gcs.STREAM_CHUNK_SIZE + 1)
    test_file.write_bytes(large_payload)
    cache_hash = "abc123def4567890"
    resumable = mocker.spy(gcloud_storage.Storage, "_upload_resumable")
    simple = mocker.spy(gcloud_storage.Storage, "_upload_simple")

    await gcs_remote.upload_file(test_file, cache_hash)

    assert resumable.call_count == 1
    assert simple.call_count == 0
    assert await _get(gcs_client, gcs_remote, cache_hash) == large_payload


async def test_upload_file_small_uses_simple_upload(
    gcs_remote: gcs.GCSRemote,
    tmp_path: pathlib.Path,
    gcs_client: gcloud_storage.Storage,
    mocker: MockerFixture,
) -> None:
    """A file of at most 8 MiB goes up in one simple upload request."""
    test_file = tmp_path / "file.bin"
    payload = b"y" * gcs.STREAM_CHUNK_SIZE
    test_file.write_bytes(payload)
    cache_hash = "abc123def4567890"
    resumable = mocker.spy(gcloud_storage.Storage, "_upload_resumable")
    simple = mocker.spy(gcloud_storage.Storage, "_upload_simple")

    await gcs_remote.upload_file(test_file, cache_hash)

    assert simple.call_count == 1
    assert resumable.call_count == 0
    assert await _get(gcs_client, gcs_remote, cache_hash) == payload


# -----------------------------------------------------------------------------
# upload_batch
# -----------------------------------------------------------------------------


async def test_upload_batch(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path, gcs_client: gcloud_storage.Storage
) -> None:
    """upload_batch uploads multiple files in parallel."""
    files = list[tuple[pathlib.Path, str]]()
    contents = dict[str, bytes]()
    for i in range(3):
        data = f"content {i}".encode()
        cache_hash = _helper_hash(data)
        f = tmp_path / f"file{i}.txt"
        f.write_bytes(data)
        files.append((f, cache_hash))
        contents[cache_hash] = data

    results = await gcs_remote.upload_batch(files, concurrency=1)

    assert len(results) == 3
    assert all(r["success"] for r in results)
    for cache_hash, data in contents.items():
        assert await _get(gcs_client, gcs_remote, cache_hash) == data


async def test_upload_batch_with_callback(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path
) -> None:
    """upload_batch calls callback for each completed upload."""
    files = list[tuple[pathlib.Path, str]]()
    for i in range(3):
        f = tmp_path / f"file{i}.txt"
        f.write_text(f"content {i}")
        files.append((f, f"a{i}b2c3d4e5f6789a"))
    callback_values = list[int]()

    def callback(completed: int, total: int, filename: str) -> None:
        callback_values.append(completed)

    await gcs_remote.upload_batch(files, concurrency=1, callback=callback)

    assert sorted(callback_values) == [1, 2, 3]


async def test_upload_batch_empty() -> None:
    """upload_batch with empty list returns empty results."""
    r = gcs.GCSRemote("gs://bucket/prefix")
    assert await r.upload_batch([]) == []


async def test_upload_batch_reports_each_failure_and_never_raises(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path, gcs_client: gcloud_storage.Storage
) -> None:
    """One failed upload is a TransferResult(success=False); the rest still upload."""
    good = tmp_path / "good.txt"
    good.write_bytes(b"good")
    good_hash = _helper_hash(b"good")
    missing_hash = "0123456789abcdef"

    results = await gcs_remote.upload_batch(
        [(good, good_hash), (tmp_path / "missing.txt", missing_hash)]
    )

    assert {r["hash"]: r["success"] for r in results} == {good_hash: True, missing_hash: False}
    errors = {r["hash"]: r["error"] for r in results if "error" in r}
    assert "missing.txt" in errors[missing_hash]
    assert await _get(gcs_client, gcs_remote, good_hash) == b"good"


async def test_upload_batch_shares_one_http_session(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """A batch of uploads opens one aiohttp session, not one per file."""
    files = list[tuple[pathlib.Path, str]]()
    for i in range(3):
        f = tmp_path / f"file{i}.txt"
        f.write_text(f"content {i}")
        files.append((f, f"a{i}b2c3d4e5f6789a"))
    session_init = mocker.spy(aiohttp.ClientSession, "__init__")

    results = await gcs_remote.upload_batch(files)

    assert all(r["success"] for r in results)
    assert session_init.call_count == 1


async def test_upload_batch_calls_metrics_end_on_error(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """upload_batch calls metrics.end even when uploads fail."""
    from pivot import metrics

    mocker.patch.object(
        gcloud_storage.Storage, "upload", new=mocker.AsyncMock(side_effect=OSError("upload boom"))
    )
    spy_end = mocker.spy(metrics, "end")
    f = tmp_path / "file.txt"
    f.write_text("content")

    results = await gcs_remote.upload_batch([(f, "a1b2c3d4e5f6789a")])

    assert [r["success"] for r in results] == [False]
    spy_end.assert_any_call("gcs.upload_batch", mocker.ANY)


# -----------------------------------------------------------------------------
# download_file
# -----------------------------------------------------------------------------


async def test_download_file(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path, gcs_client: gcloud_storage.Storage
) -> None:
    """download_file gets object from GCS."""
    cache_hash = _helper_hash(b"downloaded content")
    await _put(gcs_client, gcs_remote, cache_hash, b"downloaded content")
    dest_file = tmp_path / "dest.txt"

    await gcs_remote.download_file(cache_hash, dest_file)

    assert dest_file.read_bytes() == b"downloaded content"


async def test_upload_download_round_trip_large_file(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path
) -> None:
    """A file above 8 MiB pushed with upload_file comes back byte-identical."""
    payload = os.urandom(gcs.STREAM_CHUNK_SIZE * 2 + 17)
    cache_hash = _helper_hash(payload)
    src = tmp_path / "src.bin"
    src.write_bytes(payload)

    await gcs_remote.upload_file(src, cache_hash)
    dest = tmp_path / "dest.bin"
    await gcs_remote.download_file(cache_hash, dest)

    assert dest.read_bytes() == payload


async def test_download_file_hash_mismatch(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path, gcs_client: gcloud_storage.Storage
) -> None:
    """Bytes whose xxh64 is not the requested hash raise the hash-mismatch error, leaving nothing."""
    expected_hash = _helper_hash(b"expected content")
    await _put(gcs_client, gcs_remote, expected_hash, b"corrupt content")
    dest = tmp_path / "dest.txt"

    with pytest.raises(exceptions.RemoteError) as exc_info:
        await gcs_remote.download_file(expected_hash, dest)

    key = f"gs://{gcs_remote.bucket}/{keys.hash_to_key(gcs_remote.prefix, expected_hash)}"
    message = str(exc_info.value)
    assert message.startswith(f"Hash mismatch: expected {expected_hash}, got ")
    assert f"Delete {key} from the remote, then run `pivot push`" in message
    assert not dest.exists()
    assert os.listdir(tmp_path) == []


async def test_download_file_readonly(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path, gcs_client: gcloud_storage.Storage
) -> None:
    """download_file with readonly=True sets 0o444 permissions (for cache files)."""
    cache_hash = _helper_hash(b"cached content")
    await _put(gcs_client, gcs_remote, cache_hash, b"cached content")
    dest_file = tmp_path / "cached.txt"

    await gcs_remote.download_file(cache_hash, dest_file, readonly=True)

    assert dest_file.read_bytes() == b"cached content"
    mode = dest_file.stat().st_mode & 0o777
    assert mode == 0o444, f"Expected 0o444, got {oct(mode)}"


async def test_download_file_default_permissions(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path, gcs_client: gcloud_storage.Storage
) -> None:
    """download_file without readonly flag creates file with default permissions."""
    cache_hash = _helper_hash(b"writable content")
    await _put(gcs_client, gcs_remote, cache_hash, b"writable content")
    dest_file = tmp_path / "writable.txt"

    await gcs_remote.download_file(cache_hash, dest_file, readonly=False)

    assert dest_file.read_bytes() == b"writable content"
    mode = dest_file.stat().st_mode & 0o777
    assert mode & 0o200, f"Expected owner write permission, got {oct(mode)}"


async def test_gcs_download_file_auth_error(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """download_file raises RemoteConnectionError with the ADC refresh command on 401."""
    mocker.patch.object(
        gcloud_storage.Storage,
        "download_stream",
        new=mocker.AsyncMock(side_effect=_response_error(401, "Invalid Credentials")),
    )

    with pytest.raises(
        exceptions.RemoteConnectionError, match="gcloud auth application-default login"
    ):
        await gcs_remote.download_file("abc123def4567890", tmp_path / "downloaded.txt")


async def test_download_file_permission_denied(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """download_file names the object and Google's permission message on 403."""
    mocker.patch.object(
        gcloud_storage.Storage,
        "download_stream",
        new=mocker.AsyncMock(
            side_effect=_response_error(403, "u@x.dev does not have storage.objects.get access")
        ),
    )

    with pytest.raises(exceptions.RemoteConnectionError, match="storage.objects.get") as exc_info:
        await gcs_remote.download_file("abc123def4567890", tmp_path / "downloaded.txt")

    assert f"Permission denied to read gs://{gcs_remote.bucket}/" in str(exc_info.value)


async def test_download_file_cleans_up_on_error(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path
) -> None:
    """download_file of a missing object says so and leaves no temp file behind."""
    dest_file = tmp_path / "dest.txt"

    with pytest.raises(exceptions.RemoteError, match="Remote cache file not found: gs://"):
        await gcs_remote.download_file("abc123def4567890", dest_file)

    temp_files = [f for f in os.listdir(tmp_path) if f.startswith(".pivot_download_")]
    assert len(temp_files) == 0, f"Temp files not cleaned up: {temp_files}"


async def test_download_file_mid_stream_error(
    gcs_remote: gcs.GCSRemote,
    tmp_path: pathlib.Path,
    gcs_client: gcloud_storage.Storage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """download_file cleans up when stream.read() raises mid-transfer."""
    monkeypatch.setattr(gcs, "STREAM_CHUNK_SIZE", 256)
    cache_hash = "abc123def4567890"
    await _put(gcs_client, gcs_remote, cache_hash, b"x" * 1024)
    call_count = 0
    real_stream_download = gcs._stream_download_to_fd

    async def _inject_read_failure(stream: Any, fd: int) -> str:  # noqa: ANN401 - gcloud StreamResponse
        """Wrap stream.read to fail on second call, then delegate to real impl."""
        original_read = stream.read

        async def _failing_read(size: int) -> bytes:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise ConnectionError("network interrupted")
            return await original_read(size)

        stream.read = _failing_read
        return await real_stream_download(stream, fd)

    monkeypatch.setattr(gcs, "_stream_download_to_fd", _inject_read_failure)
    dest_file = tmp_path / "dest.txt"

    with pytest.raises(ConnectionError, match="network interrupted"):
        await gcs_remote.download_file(cache_hash, dest_file)

    temp_files = [f for f in os.listdir(tmp_path) if f.startswith(".pivot_download_")]
    assert len(temp_files) == 0, f"Temp files not cleaned up: {temp_files}"
    assert not dest_file.exists(), "Dest file should not exist after mid-stream error"


async def test_download_file_stream_read_timeout(
    gcs_remote: gcs.GCSRemote,
    tmp_path: pathlib.Path,
    gcs_client: gcloud_storage.Storage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """download_file cleans up when stream.read() times out."""
    monkeypatch.setattr(gcs, "STREAM_CHUNK_SIZE", 256)
    monkeypatch.setattr(gcs, "STREAM_READ_TIMEOUT", 0.1)
    cache_hash = "abc123def4567890"
    await _put(gcs_client, gcs_remote, cache_hash, b"x" * 1024)
    real_stream_download = gcs._stream_download_to_fd
    hang_event = asyncio.Event()

    async def _inject_hanging_read(stream: Any, fd: int) -> str:  # noqa: ANN401 - gcloud StreamResponse
        original_read = stream.read
        call_count = 0

        async def _hanging_read(size: int) -> bytes:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                await hang_event.wait()  # Hangs until the read timeout
            return await original_read(size)

        stream.read = _hanging_read
        return await real_stream_download(stream, fd)

    monkeypatch.setattr(gcs, "_stream_download_to_fd", _inject_hanging_read)
    dest_file = tmp_path / "dest.txt"

    with pytest.raises(asyncio.TimeoutError):
        await gcs_remote.download_file(cache_hash, dest_file)

    temp_files = [f for f in os.listdir(tmp_path) if f.startswith(".pivot_download_")]
    assert len(temp_files) == 0, f"Temp files not cleaned up: {temp_files}"
    assert not dest_file.exists(), "Dest file should not exist after timeout"


# -----------------------------------------------------------------------------
# download_batch
# -----------------------------------------------------------------------------


async def test_download_batch(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path, gcs_client: gcloud_storage.Storage
) -> None:
    """download_batch downloads multiple files in parallel."""
    items = [(_helper_hash(f"content {i}".encode()), tmp_path / f"dest{i}.txt") for i in range(3)]
    for i, (cache_hash, _) in enumerate(items):
        await _put(gcs_client, gcs_remote, cache_hash, f"content {i}".encode())

    results = await gcs_remote.download_batch(items, concurrency=10)

    assert len(results) == 3
    assert all(r["success"] for r in results)
    for i, (_, path) in enumerate(items):
        assert path.read_bytes() == f"content {i}".encode()


async def test_download_batch_empty() -> None:
    """download_batch with empty list returns empty results."""
    r = gcs.GCSRemote("gs://bucket/prefix")
    assert await r.download_batch([]) == []


async def test_download_batch_reports_each_failure_and_never_raises(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path, gcs_client: gcloud_storage.Storage
) -> None:
    """A missing or corrupt object is a TransferResult(success=False); the rest still download."""
    good_hash = _helper_hash(b"good")
    corrupt_hash = _helper_hash(b"expected")
    missing_hash = "0123456789abcdef"
    await _put(gcs_client, gcs_remote, good_hash, b"good")
    await _put(gcs_client, gcs_remote, corrupt_hash, b"corrupt")
    files_dir = tmp_path / "files"

    results = await gcs_remote.download_batch(
        [(h, cache.get_cache_path(files_dir, h)) for h in (good_hash, corrupt_hash, missing_hash)],
        readonly=True,
    )

    assert {r["hash"]: r["success"] for r in results} == {
        good_hash: True,
        corrupt_hash: False,
        missing_hash: False,
    }
    errors = {r["hash"]: r["error"] for r in results if "error" in r}
    assert "Hash mismatch" in errors[corrupt_hash]
    assert "Remote cache file not found" in errors[missing_hash]
    assert cache.get_cache_path(files_dir, good_hash).read_bytes() == b"good"
    assert not cache.get_cache_path(files_dir, corrupt_hash).exists()
    assert not cache.get_cache_path(files_dir, missing_hash).exists()


async def test_download_batch_with_callback(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path, gcs_client: gcloud_storage.Storage
) -> None:
    """download_batch calls callback for each completed download."""
    items = [(_helper_hash(f"content {i}".encode()), tmp_path / f"dest{i}.txt") for i in range(3)]
    for i, (cache_hash, _) in enumerate(items):
        await _put(gcs_client, gcs_remote, cache_hash, f"content {i}".encode())
    callback_values = list[int]()

    def callback(completed: int, total: int, filename: str) -> None:
        callback_values.append(completed)

    await gcs_remote.download_batch(items, concurrency=10, callback=callback)

    assert sorted(callback_values) == [1, 2, 3]


async def test_download_batch_readonly(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path, gcs_client: gcloud_storage.Storage
) -> None:
    """download_batch with readonly=True sets 0o444 permissions on all files."""
    items = [(_helper_hash(f"content {i}".encode()), tmp_path / f"cached{i}.txt") for i in range(3)]
    for i, (cache_hash, _) in enumerate(items):
        await _put(gcs_client, gcs_remote, cache_hash, f"content {i}".encode())

    results = await gcs_remote.download_batch(items, concurrency=10, readonly=True)

    assert all(r["success"] for r in results)
    for _, path in items:
        mode = path.stat().st_mode & 0o777
        assert mode == 0o444, f"Expected 0o444 for {path}, got {oct(mode)}"


async def test_download_batch_keeps_existing_destination(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path, gcs_client: gcloud_storage.Storage
) -> None:
    """download_batch never replaces a destination that already exists (#461)."""
    content = b"content"
    cache_hash = _helper_hash(content)
    await _put(gcs_client, gcs_remote, cache_hash, content)
    dest = tmp_path / "entry"
    dest.write_bytes(content)
    inode = dest.stat().st_ino

    results = await gcs_remote.download_batch([(cache_hash, dest)], readonly=True)

    assert [r["success"] for r in results] == [True]
    assert dest.stat().st_ino == inode, "the existing cache entry must not be replaced"
    left = [p.name for p in tmp_path.iterdir() if p.name != cache._PREFIX_LOCK]
    assert left == ["entry"], "no download temp file left"


@pytest.mark.parametrize("link_errno", [errno.EPERM, errno.ENOTSUP])
@pytest.mark.parametrize("exists", [False, True])
async def test_download_batch_without_hard_links(
    gcs_remote: gcs.GCSRemote,
    tmp_path: pathlib.Path,
    gcs_client: gcloud_storage.Storage,
    monkeypatch: pytest.MonkeyPatch,
    link_errno: int,
    exists: bool,
) -> None:
    """On a filesystem that refuses link(), download still promotes the verified file, and
    still keeps an existing entry."""
    content = b"content"
    cache_hash = _helper_hash(content)
    await _put(gcs_client, gcs_remote, cache_hash, content)
    dest = tmp_path / "entry"
    inode = None
    if exists:
        dest.write_bytes(content)
        inode = dest.stat().st_ino

    def _refuse_link(src: object, dst: object) -> None:
        raise OSError(link_errno, os.strerror(link_errno))

    monkeypatch.setattr(os, "link", _refuse_link)

    results = await gcs_remote.download_batch([(cache_hash, dest)], readonly=True)

    assert [r["success"] for r in results] == [True], results
    assert dest.read_bytes() == content
    if inode is not None:
        assert dest.stat().st_ino == inode, "the existing cache entry must not be replaced"
    left = [p.name for p in tmp_path.iterdir() if p.name != cache._PREFIX_LOCK]
    assert left == ["entry"], "no download temp file left"


async def test_download_batch_shares_one_http_session(
    gcs_remote: gcs.GCSRemote,
    tmp_path: pathlib.Path,
    gcs_client: gcloud_storage.Storage,
    mocker: MockerFixture,
) -> None:
    """A batch of downloads opens one aiohttp session, not one per file."""
    items = [(_helper_hash(f"content {i}".encode()), tmp_path / f"dest{i}.txt") for i in range(3)]
    for i, (cache_hash, _) in enumerate(items):
        await _put(gcs_client, gcs_remote, cache_hash, f"content {i}".encode())
    session_init = mocker.spy(aiohttp.ClientSession, "__init__")

    results = await gcs_remote.download_batch(items)

    assert all(r["success"] for r in results)
    assert session_init.call_count == 1


# -----------------------------------------------------------------------------
# Metrics / Finally Tests
# -----------------------------------------------------------------------------


async def test_bulk_exists_calls_metrics_end_on_error(
    gcs_remote: gcs.GCSRemote, mocker: MockerFixture
) -> None:
    """bulk_exists calls metrics.end even when an error occurs."""
    from pivot import metrics

    mocker.patch.object(
        gcloud_storage.Storage,
        "download_metadata",
        new=mocker.AsyncMock(side_effect=_response_error(403, "Forbidden")),
    )
    spy_end = mocker.spy(metrics, "end")

    with pytest.raises(exceptions.RemoteConnectionError):
        await gcs_remote.bulk_exists(["a1b2c3d4e5f6789a"])

    spy_end.assert_any_call("gcs.bulk_exists", mocker.ANY)


async def test_download_batch_calls_metrics_end_on_error(
    gcs_remote: gcs.GCSRemote, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """download_batch calls metrics.end even when downloads fail."""
    from pivot import metrics

    mocker.patch.object(
        gcloud_storage.Storage,
        "download_stream",
        new=mocker.AsyncMock(side_effect=OSError("download boom")),
    )
    spy_end = mocker.spy(metrics, "end")

    results = await gcs_remote.download_batch([("a1b2c3d4e5f6789a", tmp_path / "dest.txt")])

    assert [r["success"] for r in results] == [False]
    spy_end.assert_any_call("gcs.download_batch", mocker.ANY)
