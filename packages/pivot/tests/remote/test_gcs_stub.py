"""GCSRemote against a local stand-in for Google's OAuth, IAM and Cloud Storage endpoints.

The stand-in receives real requests, bearer tokens included. These tests check what reaches a
user when Google refuses a request: no error, result, log line or traceback may carry the token
or an HTTP client exception. They also check that an upload counts only when the object the
server reports storing matches the local file.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import datetime
import gc
import hashlib
import json
import logging
import os
import socket
import traceback
import tracemalloc
import types
from typing import TYPE_CHECKING

import aiohttp
import pytest
import xxhash
from aiohttp import web
from gcloud.aio.auth import token as gcloud_token
from gcloud.aio.storage import storage as gcloud_storage_impl

from pivot import config, exceptions, project
from pivot.remote import gcs, sync
from pivot.storage import cache, restore, state

if TYPE_CHECKING:
    import pathlib
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from pytest_mock import MockerFixture

    from conftest import GitRepo
    from pivot.types import TransferResult

    Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]

_SENTINEL = "ya29.pivot-test-sentinel-bearer-token"
_BUCKET = "pivot-stub"
_HASH = "abcdef1234567890"
_LOGIN = "gcloud auth application-default login"


def _md5(body: bytes) -> str:
    return base64.b64encode(hashlib.md5(body).digest()).decode()


def _resource(name: str, body: bytes) -> dict[str, str]:
    """The object resource Cloud Storage returns for ``body`` stored as ``name``."""
    return {"name": name, "size": str(len(body)), "md5Hash": _md5(body)}


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"error": {"code": status, "message": message}}, status=status)


class _Google:
    """Google's endpoints, each a replaceable handler in ``routes``, on one local server."""

    url: str
    authorizations: list[str]
    content_ranges: list[str]
    upload_types: list[str]
    counts: collections.Counter[str]
    routes: dict[str, Handler]

    def __init__(self) -> None:
        self.url = ""
        self.authorizations = []
        self.content_ranges = []
        self.upload_types = []
        self.counts = collections.Counter()
        self.routes = {
            "token": self._token,
            "impersonate": self._impersonate,
            "metadata": self._not_found,
            "media": self._not_found,
            "list": self._empty_listing,
            "simple": self._store_simple,
        }
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        app = web.Application(middlewares=[self._record], client_max_size=64 * 1024 * 1024)
        app.router.add_post("/token", self._route("token"))
        app.router.add_post("/impersonate", self._route("impersonate"))
        app.router.add_get("/storage/v1/b/{bucket}/o", self._route("list"))
        app.router.add_get("/storage/v1/b/{bucket}/o/{name:.+}", self._object)
        app.router.add_post("/upload/storage/v1/b/{bucket}/o", self._upload)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        host, port = self._runner.addresses[0][:2]
        self.url = f"http://{host}:{port}"

    async def stop(self) -> None:
        assert self._runner is not None
        await self._runner.cleanup()

    @web.middleware
    async def _record(self, request: web.Request, handler: Handler) -> web.StreamResponse:
        if "Authorization" in request.headers:
            self.authorizations.append(request.headers["Authorization"])
        if "Content-Range" in request.headers:
            self.content_ranges.append(request.headers["Content-Range"])
        return await handler(request)

    def _route(self, name: str) -> Handler:
        async def dispatch(request: web.Request) -> web.StreamResponse:
            self.counts[name] += 1
            return await self.routes[name](request)

        return dispatch

    async def _object(self, request: web.Request) -> web.StreamResponse:
        name = "media" if request.query.get("alt") == "media" else "metadata"
        return await self._route(name)(request)

    async def _upload(self, request: web.Request) -> web.StreamResponse:
        """Every upload POST, with its ``uploadType`` recorded (``media`` for a simple upload)."""
        self.upload_types.append(request.query.get("uploadType", ""))
        return await self._route("simple")(request)

    async def _token(self, request: web.Request) -> web.StreamResponse:
        return web.json_response({"access_token": _SENTINEL, "expires_in": 3600})

    async def _impersonate(self, request: web.Request) -> web.StreamResponse:
        return _error(403, "Permission 'iam.serviceAccounts.getAccessToken' denied")

    async def _not_found(self, request: web.Request) -> web.StreamResponse:
        return _error(404, "No such object")

    async def _empty_listing(self, request: web.Request) -> web.StreamResponse:
        return web.json_response({"kind": "storage#objects"})

    async def _store_simple(self, request: web.Request) -> web.StreamResponse:
        return web.json_response(_resource(request.query["name"], await request.read()))


@pytest.fixture
async def google(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> AsyncGenerator[_Google]:
    """The stand-in, with gcloud-aio-storage pointed at it as if it were Cloud Storage.

    Credentials are an authorized-user key whose token endpoint is the stand-in, so every
    storage request carries ``Authorization: Bearer <_SENTINEL>``. No test here waits on a
    retry backoff.
    """
    stub = _Google()
    await stub.start()
    credentials = tmp_path / "credentials.json"
    credentials.write_text(
        json.dumps(
            {
                "type": "authorized_user",
                "token_uri": f"{stub.url}/token",
                "client_id": "client",
                "client_secret": "secret",
                "refresh_token": "refresh",
            }
        )
    )
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials))
    monkeypatch.delenv("STORAGE_EMULATOR_HOST", raising=False)

    def api_root(api_root: str | None, api_is_dev: bool | None = None) -> tuple[bool, str]:
        return False, stub.url

    monkeypatch.setattr(gcloud_storage_impl, "init_api_root", api_root)
    monkeypatch.setattr(gcs, "_RETRY_BASE_DELAY", 0)
    monkeypatch.setattr(gcloud_token.BaseToken.acquire_access_token.retry, "enabled", False)
    mocker.patch.object(config, "get_remote_retries", return_value=0)
    yield stub
    await stub.stop()


def _impersonating(
    google: _Google, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switch to a key that impersonates a service account through the stand-in's IAM endpoint.

    gcloud-aio-auth gets a source token (``_SENTINEL``) and sends it as a bearer token to the
    impersonation endpoint, which refuses by default.
    """
    credentials = tmp_path / "impersonated.json"
    credentials.write_text(
        json.dumps(
            {
                "type": "impersonated_service_account",
                "token_uri": f"{google.url}/token",
                "service_account_impersonation_url": f"{google.url}/impersonate",
                "source_credentials": {
                    "type": "authorized_user",
                    "client_id": "client",
                    "client_secret": "secret",
                    "refresh_token": "refresh",
                },
            }
        )
    )
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials))


def _shown(exc: BaseException) -> list[BaseException]:
    """The exception and every cause or context a traceback of it prints."""
    chain = list[BaseException]()
    current: BaseException | None = exc
    while current is not None:
        chain.append(current)
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__suppress_context__:
            current = None
        else:
            current = current.__context__
    return chain


# -----------------------------------------------------------------------------
# No bearer token or HTTP client exception reaches the user
# -----------------------------------------------------------------------------


async def _exists(remote: gcs.GCSRemote, tmp_path: pathlib.Path) -> object:
    return await remote.exists(_HASH)


async def _bulk_exists_head(remote: gcs.GCSRemote, tmp_path: pathlib.Path) -> object:
    return await remote.bulk_exists([_HASH, "0123456789abcdef"])


async def _bulk_exists_list(remote: gcs.GCSRemote, tmp_path: pathlib.Path) -> object:
    return await remote.bulk_exists([f"a{i:015x}" for i in range(60)])


async def _list_hashes(remote: gcs.GCSRemote, tmp_path: pathlib.Path) -> object:
    return await remote.list_hashes()


def _file(tmp_path: pathlib.Path, size: int) -> pathlib.Path:
    path = tmp_path / f"local-{size}"
    path.write_bytes(b"x" * size)
    return path


async def _upload_small(remote: gcs.GCSRemote, tmp_path: pathlib.Path) -> object:
    return await remote.upload_file(_file(tmp_path, 100), _HASH)


async def _upload_large(remote: gcs.GCSRemote, tmp_path: pathlib.Path) -> object:
    return await remote.upload_file(_file(tmp_path, gcs.STREAM_CHUNK_SIZE + 1), _HASH)


async def _upload_batch(remote: gcs.GCSRemote, tmp_path: pathlib.Path) -> object:
    return await remote.upload_batch([(_file(tmp_path, 100), _HASH)])


async def _download_file(remote: gcs.GCSRemote, tmp_path: pathlib.Path) -> object:
    return await remote.download_file(_HASH, tmp_path / "dest")


async def _download_batch(remote: gcs.GCSRemote, tmp_path: pathlib.Path) -> object:
    return await remote.download_batch([(_HASH, tmp_path / "dest")])


async def _forbidden(request: web.Request) -> web.StreamResponse:
    return _error(403, "pivot@test does not have storage.objects.get access")


async def _unauthorized(request: web.Request) -> web.StreamResponse:
    return _error(401, "Invalid Credentials")


async def _server_error_page(request: web.Request) -> web.StreamResponse:
    return web.Response(status=500, text="<html>Internal Server Error</html>")


async def _cut_off(request: web.Request) -> web.StreamResponse:
    """Promise 1000 bytes, send 10, then drop the connection."""
    response = web.StreamResponse(headers={"Content-Length": "1000"})
    await response.prepare(request)
    await response.write(b"x" * 10)
    assert request.transport is not None
    request.transport.close()
    return response


_Operation = "Callable[[gcs.GCSRemote, pathlib.Path], Awaitable[object]]"


@pytest.mark.parametrize(
    ("impersonate", "route", "handler", "operation", "reported"),
    [
        pytest.param(True, None, None, _exists, _LOGIN, id="impersonation-refused-exists"),
        pytest.param(True, None, None, _list_hashes, _LOGIN, id="impersonation-refused-list"),
        pytest.param(
            True, None, None, _upload_batch, _LOGIN, id="impersonation-refused-upload-batch"
        ),
        pytest.param(
            True, None, None, _download_batch, _LOGIN, id="impersonation-refused-download-batch"
        ),
        pytest.param(False, "metadata", _unauthorized, _exists, _LOGIN, id="metadata-401"),
        pytest.param(
            False, "metadata", _forbidden, _exists, "Permission denied to read", id="metadata-403"
        ),
        pytest.param(
            False,
            "metadata",
            _forbidden,
            _bulk_exists_head,
            "GCS bulk_exists failed: Permission denied to read",
            id="metadata-403-bulk",
        ),
        pytest.param(
            False, "metadata", _server_error_page, _exists, "HTTP 500", id="metadata-500-page"
        ),
        pytest.param(
            False, "list", _forbidden, _list_hashes, "Permission denied to list", id="list-403"
        ),
        pytest.param(
            False,
            "list",
            _forbidden,
            _bulk_exists_list,
            "GCS bulk_exists LIST failed: Permission denied to list",
            id="list-403-bulk",
        ),
        pytest.param(
            False,
            "simple",
            _forbidden,
            _upload_small,
            "Permission denied to write",
            id="upload-403",
        ),
        pytest.param(
            False,
            "simple",
            _forbidden,
            _upload_batch,
            "Permission denied to write",
            id="upload-403-batch",
        ),
        pytest.param(
            False,
            "simple",
            _forbidden,
            _upload_large,
            "Permission denied to write",
            id="upload-large-403",
        ),
        pytest.param(
            False, "media", _forbidden, _download_file, "Permission denied to read", id="media-403"
        ),
        pytest.param(
            False,
            "media",
            _forbidden,
            _download_batch,
            "Permission denied to read",
            id="media-403-batch",
        ),
        pytest.param(
            False,
            "media",
            _cut_off,
            _download_file,
            f"GCS read error for gs://{_BUCKET}/p/files/ab/cdef1234567890: ",
            id="media-cut-off",
        ),
    ],
)
async def test_refused_request_reports_status_and_reason_but_never_the_token(
    google: _Google,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
    impersonate: bool,
    route: str | None,
    handler: Handler | None,
    operation: Callable[[gcs.GCSRemote, pathlib.Path], Awaitable[object]],
    reported: str,
) -> None:
    caplog.set_level(logging.DEBUG)
    if impersonate:
        _impersonating(google, tmp_path, monkeypatch)
    if route is not None:
        assert handler is not None
        google.routes[route] = handler
    remote = gcs.GCSRemote(f"gs://{_BUCKET}/p")

    texts = list[str]()
    try:
        outcome = await operation(remote, tmp_path)
    except exceptions.RemoteError as e:
        shown = _shown(e)
        assert not [x for x in shown if isinstance(x, aiohttp.ClientError)], shown
        texts += [str(e), repr(e), "".join(traceback.format_exception(e))]
    else:
        results: list[TransferResult] = outcome  # pyright: ignore[reportAssignmentType] - only batch calls return
        assert [r["success"] for r in results] == [False]
        texts += [r["error"] for r in results if "error" in r]
    captured = capfd.readouterr()
    texts += [captured.out, captured.err, caplog.text]

    assert f"Bearer {_SENTINEL}" in google.authorizations, "the stand-in must see the token"
    assert any(reported in text for text in texts), texts
    leaks = [text for text in texts if _SENTINEL in text or "RequestInfo" in text]
    assert leaks == []


async def test_refused_impersonation_names_the_status_and_googles_reason(
    google: _Google, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal itself is still explained: the impersonation endpoint's status and message.
    Refused credentials are not an unreachable remote: the next target tries again."""
    _impersonating(google, tmp_path, monkeypatch)

    with pytest.raises(exceptions.RemoteConnectionError) as exc_info:
        await gcs.GCSRemote(f"gs://{_BUCKET}/p").exists(_HASH)

    assert str(exc_info.value).endswith(
        "Getting a token failed: HTTP 403: Permission 'iam.serviceAccounts.getAccessToken' denied"
    )
    assert not isinstance(exc_info.value, exceptions.RemoteUnreachableError)


class _ShiftedDatetime:
    """Stands in for ``datetime.datetime`` in gcloud-aio-auth, running ``offset`` seconds ahead."""

    offset: float

    def __init__(self) -> None:
        self.offset = 0.0

    def __call__(self, *args: int) -> datetime.datetime:
        return datetime.datetime(*args)  # pyright: ignore[reportArgumentType] - forwards datetime's own arguments

    def now(self, tz: datetime.tzinfo | None = None) -> datetime.datetime:
        return datetime.datetime.now(tz) + datetime.timedelta(seconds=self.offset)


async def test_failed_token_refresh_partway_through_a_call_never_logs_the_token(
    google: _Google,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """A token refresh that falls due partway through a long call is awaited by the request that
    needs it, so its failure is reported like any other. gcloud-aio's unawaited background
    refresh would instead end in asyncio's 'Task exception was never retrieved' log line, whose
    repr of the failure carries the source token."""
    caplog.set_level(logging.DEBUG)
    _impersonating(google, tmp_path, monkeypatch)
    clock = _ShiftedDatetime()
    monkeypatch.setattr(
        gcloud_token,
        "datetime",
        types.SimpleNamespace(datetime=clock, UTC=datetime.UTC, timedelta=datetime.timedelta),
    )

    async def grant_once(request: web.Request) -> web.StreamResponse:
        if google.counts["impersonate"] > 1:
            return _error(403, "Permission 'iam.serviceAccounts.getAccessToken' denied")
        return web.json_response({"accessToken": "ya29.pivot-test-impersonated"})

    async def slow_metadata(request: web.Request) -> web.StreamResponse:
        clock.offset += 400  # the stand-in's 3600 s token: refresh due at 1800 s, forced at 3420 s
        return _error(404, "No such object")

    google.routes["impersonate"] = grant_once
    google.routes["metadata"] = slow_metadata

    with pytest.raises(exceptions.RemoteConnectionError) as exc_info:
        await gcs.GCSRemote(f"gs://{_BUCKET}/p").bulk_exists(
            [f"{i:016x}" for i in range(12)], concurrency=1
        )
    gc.collect()
    await asyncio.sleep(0)

    assert google.counts["impersonate"] >= 2, "a refresh must have been attempted"
    assert "Permission 'iam.serviceAccounts.getAccessToken' denied" in str(exc_info.value)
    captured = capfd.readouterr()
    texts = [str(exc_info.value), captured.out, captured.err, caplog.text]
    assert [text for text in texts if _SENTINEL in text or "RequestInfo" in text] == []


async def test_missing_key_file_is_a_credentials_error_naming_the_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "moved-key.json"
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(missing))
    monkeypatch.delenv("STORAGE_EMULATOR_HOST", raising=False)

    with pytest.raises(exceptions.RemoteConnectionError, match=_LOGIN) as exc_info:
        await gcs.GCSRemote(f"gs://{_BUCKET}/p").bulk_exists([_HASH])

    assert str(missing) in str(exc_info.value)


# -----------------------------------------------------------------------------
# Every upload is one streamed simple upload, retried whole
# -----------------------------------------------------------------------------


def _cache_holding(tmp_path: pathlib.Path, content: bytes) -> tuple[pathlib.Path, str]:
    """A cache directory holding ``content`` as its only object, and that object's hash."""
    content_hash = xxhash.xxh64(content).hexdigest()
    cache_dir = tmp_path / "cache"
    cached = cache.get_cache_path(cache_dir / "files", content_hash)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(content)
    return cache_dir, content_hash


async def test_large_file_goes_up_in_one_streamed_simple_upload(
    google: _Google, tmp_path: pathlib.Path
) -> None:
    """A file three times the 8 MiB download read size goes up in one POST, never through a
    resumable session, and is streamed from disk: the process never holds it in memory."""
    size = 3 * gcs.STREAM_CHUNK_SIZE
    local = tmp_path / "large"
    with local.open("wb") as f:
        for _ in range(size // (1024 * 1024)):
            f.write(os.urandom(1024 * 1024))

    async def store_streamed(request: web.Request) -> web.StreamResponse:
        received = hashlib.md5()
        length = 0
        async for chunk in request.content.iter_chunked(64 * 1024):
            received.update(chunk)
            length += len(chunk)
        resource = {
            "name": request.query["name"],
            "size": str(length),
            "md5Hash": base64.b64encode(received.digest()).decode(),
        }
        return web.json_response(resource)

    google.routes["simple"] = store_streamed
    remote = gcs.GCSRemote(f"gs://{_BUCKET}/p")

    tracemalloc.start()
    try:
        await remote.upload_file(local, _HASH)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert google.upload_types == ["media"]
    assert peak < gcs.STREAM_CHUNK_SIZE, f"{peak} bytes held while uploading {size}"


async def _unavailable(request: web.Request) -> web.StreamResponse:
    return _error(503, "Backend Error")


async def _rate_limited(request: web.Request) -> web.StreamResponse:
    return _error(429, "rateLimitExceeded")


async def _stalled(request: web.Request) -> web.StreamResponse:
    """Answer correctly, but only after the 0.2 s read timeout the test sets: the object counts
    as stored, so only the read timeout can make pivot send a second POST."""
    await asyncio.sleep(1)
    return web.json_response(_resource(request.query["name"], await request.read()))


@pytest.mark.parametrize(
    "failure", [_unavailable, _rate_limited, _stalled], ids=["503", "429", "read-timeout"]
)
async def test_transient_upload_failure_is_retried_as_a_fresh_upload_from_byte_0(
    google: _Google,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    failure: Handler,
) -> None:
    """The retry is a new POST that reopens the file and sends it whole; the stored object's
    MD5 then matches the local file, so the upload counts."""
    mocker.patch.object(config, "get_remote_retries", return_value=1)
    monkeypatch.setattr(gcs, "STREAM_READ_TIMEOUT", 0.2)
    received = list[bytes]()

    async def fail_once(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        received.append(body)
        if len(received) == 1:
            return await failure(request)
        return web.json_response(_resource(request.query["name"], body))

    google.routes["simple"] = fail_once
    content = os.urandom(gcs.STREAM_CHUNK_SIZE + 1)
    local = tmp_path / "local"
    local.write_bytes(content)

    await gcs.GCSRemote(f"gs://{_BUCKET}/p").upload_file(local, _HASH)

    assert received == [content, content], "each attempt sends the whole file"
    assert google.upload_types == ["media", "media"]
    assert google.content_ranges == []


@pytest.mark.parametrize(
    ("status", "attempts", "reported"),
    [
        pytest.param(400, 1, "GCS write error for {key}: HTTP 400: refused", id="400-at-once"),
        pytest.param(403, 1, "Permission denied to write {key}: refused", id="403-at-once"),
        pytest.param(
            503, 3, "GCS write error for {key}: HTTP 503: refused", id="503-after-retries"
        ),
    ],
)
async def test_failed_upload_reports_its_own_status(
    google: _Google,
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
    status: int,
    attempts: int,
    reported: str,
) -> None:
    """A 4xx other than 408 and 429 fails without a retry; a transient status is retried
    ``remote.retries`` times. Either way the user sees the status the server sent."""
    mocker.patch.object(config, "get_remote_retries", return_value=2)

    async def refuse(request: web.Request) -> web.StreamResponse:
        await request.read()
        return _error(status, "refused")

    google.routes["simple"] = refuse

    with pytest.raises(exceptions.RemoteConnectionError) as exc_info:
        await gcs.GCSRemote(f"gs://{_BUCKET}/p").upload_file(
            _file(tmp_path, gcs.STREAM_CHUNK_SIZE + 1), _HASH
        )

    assert google.counts["simple"] == attempts
    key = f"gs://{_BUCKET}/p/files/ab/cdef1234567890"
    assert str(exc_info.value) == reported.format(key=key)


async def test_upload_answered_without_the_stored_object_is_not_recorded_as_pushed(
    google: _Google, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """gcloud-aio returns the JSON of any reply below 400, so an empty 200 returns no object;
    pivot retries it as a fresh upload, then reports it failed rather than pushed."""
    mocker.patch.object(config, "get_remote_retries", return_value=1)

    async def empty_reply(request: web.Request) -> web.StreamResponse:
        await request.read()
        return web.Response()

    google.routes["simple"] = empty_reply
    cache_dir, content_hash = _cache_holding(tmp_path, os.urandom(gcs.STREAM_CHUNK_SIZE + 1))
    remote = gcs.GCSRemote(f"gs://{_BUCKET}/p")

    with state.StateDB(tmp_path / "state") as state_db:
        summary = await sync._push_async(cache_dir, tmp_path / "state", remote, state_db, "origin")
        recorded = state_db.remote_hash_exists("origin", content_hash)

    assert google.counts["simple"] == 2, "pivot retries with a fresh upload"
    assert (summary["transferred"], summary["failed"]) == (0, 1)
    assert summary["errors"] == [
        f"Upload to gs://{_BUCKET}/p/files/{content_hash[:2]}/{content_hash[2:]} did not "
        + "complete: the server returned no object"
    ]
    assert not recorded, "a hash that is not on the remote must not be recorded as pushed"


async def test_resumable_upload_retries_a_transient_503_the_way_cloud_storage_answers(
    google: _Google, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """One transient 503 on a file large enough for gcloud-aio's resumable upload is retried, as
    on a small file.

    gcloud-aio's resumable upload would retry the failed PUT on the same session with
    ``Content-Range: bytes */*`` and the whole body again, which Cloud Storage answers with
    ``400 Client sent query request with a non-empty body.`` (a status query must have an empty
    body). Pivot sends every file as one simple upload instead, so ``remote.retries`` covers the
    503 with a second POST of the whole file.
    """
    mocker.patch.object(config, "get_remote_retries", return_value=1)
    received = list[bytes]()

    async def one_503_then_as_cloud_storage(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        received.append(body)
        if len(received) == 1:
            return _error(503, "Backend Error")
        if request.headers.get("Content-Range") == "bytes */*" and body:
            return web.Response(status=400, text="Client sent query request with a non-empty body.")
        return web.json_response(_resource(request.query["name"], body))

    google.routes["simple"] = one_503_then_as_cloud_storage
    content = os.urandom(gcs.STREAM_CHUNK_SIZE + 1)
    cache_dir, content_hash = _cache_holding(tmp_path, content)
    remote = gcs.GCSRemote(f"gs://{_BUCKET}/p")

    with state.StateDB(tmp_path / "state") as state_db:
        summary = await sync._push_async(cache_dir, tmp_path / "state", remote, state_db, "origin")
        recorded = state_db.remote_hash_exists("origin", content_hash)

    assert "bytes */*" not in google.content_ranges, "the 503 reaches pivot, not a status query"
    assert google.upload_types == ["media", "media"], "two whole POSTs, no resumable session"
    assert _md5(received[-1]) == _md5(content)
    assert (summary["transferred"], summary["failed"]) == (1, 0), summary["errors"]
    assert recorded


def _short(name: str, body: bytes) -> dict[str, str]:
    return {**_resource(name, body), "size": str(len(body) - 1)}


def _wrong_md5(name: str, body: bytes) -> dict[str, str]:
    return {**_resource(name, body), "md5Hash": _md5(body + b"!")}


def _no_md5(name: str, body: bytes) -> dict[str, str]:
    return {"name": name, "size": str(len(body))}


def _wrong_name(name: str, body: bytes) -> dict[str, str]:
    return {**_resource(name, body), "name": "p/files/00/0000000000000000"}


@pytest.mark.parametrize(
    ("stored", "reason"),
    [
        pytest.param(_short, "the server stored 99 of 100 bytes", id="short-object"),
        pytest.param(
            _wrong_md5, "the stored object's MD5 does not match the local file", id="wrong-md5"
        ),
        pytest.param(_no_md5, "the stored object's MD5 does not match the local file", id="no-md5"),
        pytest.param(
            _wrong_name, "the server stored 'p/files/00/0000000000000000'", id="wrong-name"
        ),
    ],
)
async def test_upload_whose_stored_object_differs_is_a_failed_transfer(
    google: _Google,
    tmp_path: pathlib.Path,
    stored: Callable[[str, bytes], dict[str, str]],
    reason: str,
) -> None:
    async def misreport(request: web.Request) -> web.StreamResponse:
        return web.json_response(stored(request.query["name"], await request.read()))

    google.routes["simple"] = misreport

    results = await gcs.GCSRemote(f"gs://{_BUCKET}/p").upload_batch([(_file(tmp_path, 100), _HASH)])

    assert [r["success"] for r in results] == [False]
    error = results[0]["error"] if "error" in results[0] else ""
    assert error == f"Upload to gs://{_BUCKET}/p/files/ab/cdef1234567890 did not complete: {reason}"


async def test_upload_retried_after_a_short_object_succeeds_when_the_retry_is_stored(
    google: _Google, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    mocker.patch.object(config, "get_remote_retries", return_value=1)

    async def short_then_whole(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        whole = _resource(request.query["name"], body)
        if google.counts["simple"] == 1:
            return web.json_response(_short(request.query["name"], body))
        return web.json_response(whole)

    google.routes["simple"] = short_then_whole

    results = await gcs.GCSRemote(f"gs://{_BUCKET}/p").upload_batch([(_file(tmp_path, 100), _HASH)])

    assert results == [{"hash": _HASH, "success": True}]
    assert google.counts["simple"] == 2


# -----------------------------------------------------------------------------
# One object's failure is that object's; only a remote no connection reaches is unreachable
# -----------------------------------------------------------------------------


def _refused_url() -> str:
    """The URL of a local port nothing listens on, so a connection to it is refused at once."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


async def test_download_whose_body_breaks_off_is_downloaded_again_from_its_start(
    google: _Google, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """A body that breaks off is retried like a failed request: the retry writes the object whole
    over the partial download, and the download counts."""
    mocker.patch.object(config, "get_remote_retries", return_value=1)
    body = os.urandom(1000)
    content_hash = xxhash.xxh64(body).hexdigest()

    async def cut_off_once(request: web.Request) -> web.StreamResponse:
        if google.counts["media"] == 1:
            return await _cut_off(request)
        return web.Response(body=body)

    google.routes["media"] = cut_off_once

    results = await gcs.GCSRemote(f"gs://{_BUCKET}/p").download_batch(
        [(content_hash, tmp_path / "dest")], readonly=True
    )

    assert results == [{"hash": content_hash, "success": True}]
    assert google.counts["media"] == 2
    assert (tmp_path / "dest").read_bytes() == body


async def test_download_whose_body_keeps_breaking_off_fails_only_that_object(
    google: _Google, tmp_path: pathlib.Path
) -> None:
    """A body that breaks off on every attempt is that object's failed transfer: the remote
    answered, so the result does not say the remote could not be reached."""
    google.routes["media"] = _cut_off

    results = await gcs.GCSRemote(f"gs://{_BUCKET}/p").download_batch(
        [(_HASH, tmp_path / "dest")], readonly=True
    )

    assert [r["success"] for r in results] == [False]
    assert "connection_failed" not in results[0], results
    error = results[0]["error"] if "error" in results[0] else ""
    assert error.startswith(f"GCS read error for gs://{_BUCKET}/p/files/ab/cdef1234567890: "), error


async def test_download_from_a_host_that_refuses_connections_is_unreachable(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """A connection that cannot be made, after remote.retries retries, marks the transfer, so get
    stops using the remote: every object would fail the same way."""
    monkeypatch.setenv("STORAGE_EMULATOR_HOST", _refused_url())
    mocker.patch.object(config, "get_remote_retries", return_value=0)

    results = await gcs.GCSRemote(f"gs://{_BUCKET}/p").download_batch(
        [(_HASH, tmp_path / "dest")], readonly=True
    )

    assert [(r["success"], r.get("connection_failed")) for r in results] == [(False, True)]
    error = results[0]["error"] if "error" in results[0] else ""
    assert error.startswith(
        f"Cannot reach Google Cloud Storage to read gs://{_BUCKET}/p/files/ab/cdef1234567890: "
    ), error


async def test_token_endpoint_that_cannot_be_reached_is_an_unreachable_remote(
    google: _Google, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a credentials file in hand, a token request that cannot connect means Google cannot
    be reached: every target would fail the same way, so get stops using the remote."""
    credentials = tmp_path / "unreachable-token-endpoint.json"
    credentials.write_text(
        json.dumps(
            {
                "type": "authorized_user",
                "token_uri": f"{_refused_url()}/token",
                "client_id": "client",
                "client_secret": "secret",
                "refresh_token": "refresh",
            }
        )
    )
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials))

    with pytest.raises(exceptions.RemoteUnreachableError) as exc_info:
        await gcs.GCSRemote(f"gs://{_BUCKET}/p").download_batch([(_HASH, tmp_path / "dest")])

    assert str(exc_info.value).startswith("Cannot reach Google to get a Cloud Storage token: ")


async def test_no_credentials_is_a_credentials_error_though_the_metadata_server_is_unreachable(
    google: _Google, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a credentials file gcloud-aio asks the GCE metadata server, whose name does not
    resolve off GCE; that failure names the credentials and does not end get's remote step."""
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("CLOUDSDK_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(gcloud_token, "GCE_ENDPOINT_TOKEN", f"{_refused_url()}/token")

    with pytest.raises(exceptions.RemoteConnectionError, match=_LOGIN) as exc_info:
        await gcs.GCSRemote(f"gs://{_BUCKET}/p").download_batch([(_HASH, tmp_path / "dest")])

    assert not isinstance(exc_info.value, exceptions.RemoteUnreachableError)


@pytest.mark.parametrize("failure", [_forbidden, _cut_off], ids=["403", "body-cut-off"])
async def test_get_keeps_using_gcs_after_one_object_fails(
    google: _Google, git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch, failure: Handler
) -> None:
    """a.csv's object fails on a remote that answers (a 403, or a body that breaks off on every
    attempt): a.csv fails, and get still downloads b.csv and c.csv from the remote."""
    repo_path, commit = git_repo
    contents = {"a.csv": b"aaa\n", "b.csv": b"bbb\n", "c.csv": b"ccc\n"}
    hashes = {name: xxhash.xxh64(body).hexdigest() for name, body in contents.items()}
    for name, content_hash in hashes.items():
        (repo_path / f"{name}.pvt").write_text(f"path: {name}\nhash: '{content_hash}'\nsize: 4\n")
    sha = commit("track three files")
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True, exist_ok=True)

    async def media(request: web.Request) -> web.StreamResponse:
        name = request.match_info["name"]
        if name.endswith(hashes["a.csv"][2:]):
            return await failure(request)
        served = next(contents[n] for n, h in hashes.items() if name.endswith(h[2:]))
        return web.Response(body=served)

    google.routes["media"] = media
    remote = gcs.GCSRemote(f"gs://{_BUCKET}/p")

    messages, success = await asyncio.to_thread(
        restore.restore_targets_from_revision,
        targets=list(contents),
        rev=sha[:7],
        output=None,
        cache_dir=repo_path / ".pivot" / "cache",
        stage_targets={},
        checkout_modes=[cache.CheckoutMode.COPY],
        force=False,
        remote_factory=lambda: remote,
    )

    assert not success, messages
    assert "a.csv - failed to download from remote: " in messages[0], messages
    restored = [(repo_path / name).read_bytes() for name in ("b.csv", "c.csv")]
    assert restored == [b"bbb\n", b"ccc\n"], messages
    assert google.counts["media"] == 3, "each target asked the remote for its own object"
