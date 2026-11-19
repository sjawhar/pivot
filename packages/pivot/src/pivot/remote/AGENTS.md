# Pivot Remote

Remote cache storage on S3 and GCS (`push`, `pull`, `fetch`, `get`), and the GitHub and git
access behind `pivot import` and `pivot update`.

Tests: `uv run pytest packages/pivot/tests/remote -n auto`.

## Async I/O

- All network I/O here is async. A sync caller wraps the call in `asyncio.run()` at the CLI
  boundary (`cli/import_cmd.py`, `cli/update.py`).
- Run independent requests concurrently (`asyncio.gather()` or a task group) on one shared
  `aiohttp.ClientSession`; the functions in `import_artifact.py` take it as `session=`.
- HTTP goes through `aiohttp`, a direct dependency. `httpx` and `requests` are not dependencies,
  and `urllib.request` blocks the event loop.
- S3 (`storage.S3Remote`, aioboto3): one `aioboto3.Session()` per instance, one client per method
  call (`async with self._client()`), shared by a batch's concurrent tasks. Type the client as
  `types_aiobotocore_s3.S3Client`, imported under `TYPE_CHECKING` (stubs: `types-aioboto3[all]`,
  a dev dependency), not `Any`.
- GCS (`gcs.GCSRemote`, gcloud-aio-storage on aiohttp): one `aiohttp.ClientSession` per method
  call, and Pivot's own `aiohttp.ClientTimeout` on every storage request, because gcloud-aio's
  defaults are whole-request deadlines too short for large objects. Read the module docstring and
  `_upload` before you change uploads, retries or error text.

## Backends

- Callers use the `protocol.RemoteStorage` interface and get a backend from
  `protocol.open_remote(url)`, never from a backend class. `config.validate_remote_url` is the
  one URL grammar.
- What both backends must agree on lives once: the object key layout, hash validation, the
  digest and the atomic download in `keys.py`; the existence strategy, which object names are
  cache objects, and a failed transfer as `TransferResult(success=False)` in `batch.py`. A check
  on one object belongs in the backend's per-object coroutine.
- Raise `RemoteUnreachableError` only when no connection could be made after the retries (DNS,
  refused, connect timeout, TLS, proxy connect): `get` then stops using the remote for the rest
  of the command. An HTTP status (403 included), a reply that breaks off, a missing or corrupt
  object, and missing or refused credentials are a `RemoteConnectionError` for that object or
  call.
- No message, log line or exception chain carries an aiohttp exception's str or repr: it holds
  the request headers, and so the bearer token. Report the status and the server's message and
  raise `from None` (`gcs._describe`, `gcs._remote_error`).

## Auth and errors

- GitHub: `GH_TOKEN`, then `GITHUB_TOKEN`. S3: the standard AWS credential chain. GCS:
  Application Default Credentials; `STORAGE_EMULATOR_HOST` points it at an emulator.
- An error names the action, the location and the fix (the variable to set or the command to
  run), as `github.py` and `gcs._AUTH_ERROR_MESSAGE` do.

## Test fixtures

S3 tests use moto through `moto_s3_bucket`; GCS tests use fake-gcs-server through `gcs_bucket`,
which needs Docker (skipped locally without it, failed in CI). Both are in
`packages/pivot/tests/conftest.py`.
