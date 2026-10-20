# Pivot Remote - Development Guidelines

## Async-First I/O (Critical)

**All network I/O in the remote layer MUST be async.** This is not optional.

The remote layer uses `aiohttp` (direct dependency) for HTTP, `aioboto3` for S3, and
`gcloud-aio-storage` (itself on aiohttp) for GCS.
CLI commands expose sync wrappers that call `asyncio.run()` at the boundary.

### Pattern: Async Internals + Sync CLI Wrapper

```python
# Internal: fully async
async def _fetch_remote_file(url: str, token: str | None) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=_auth_headers(token)) as resp:
            resp.raise_for_status()
            return await resp.read()

# CLI boundary: sync wrapper
def fetch_remote_file(url: str, token: str | None) -> bytes:
    return asyncio.run(_fetch_remote_file(url, token))
```

### Why Async

1. **Parallelism**: Import resolution scans multiple lock files — `asyncio.gather()` reads them concurrently
2. **Consistency**: both cache backends are fully async; mixing sync HTTP would create two patterns
3. **Status checks**: `--check-imports` checks multiple imports concurrently
4. **Connection reuse**: `aiohttp.ClientSession` reuses TCP connections across requests

### HTTP Client: aiohttp

Use `aiohttp` for all HTTP calls (GitHub API, git forge APIs). Do NOT use:
- `httpx` — not in our dependencies
- `urllib.request` — synchronous, no connection reuse, no async support
- `requests` — not in our dependencies

`aiohttp` is a direct dependency (also available transitively via `aioboto3 → aiobotocore`).

### Cache Backends

`protocol.RemoteStorage` is the interface every caller uses: `url`, `exists`, `bulk_exists`,
`iter_hashes`, `list_hashes`, `upload_file`, `download_file`, `upload_batch`, `download_batch`.
`protocol.open_remote(url)` picks the backend from the URL scheme (`s3://` → `storage.S3Remote`,
`gs://` → `gcs.GCSRemote`); construct remotes through it, never a backend class directly.
`config.validate_remote_url` is the one URL grammar; config validation, `doctor`, and import all
call it.

`keys.py` owns what both backends must agree on: the object key layout
(`<prefix>files/<hash[:2]>/<hash[2:]>`), hash validation, the streaming xxh64 digest, the
hash-mismatch error, and the staged atomic download. A change to the hash or the layout goes there.

`batch.py` owns the orchestration both backends run: `bulk_exists` (per-hash requests below 50
hashes, a listing per 2-char prefix from 50 up), `cache_hashes` (which object names are cache
objects), and `transfer_batch` (a failed transfer is a `TransferResult(success=False)`, never an
exception). A backend passes its client context and per-object coroutines; checks on one object,
such as GCS's upload verification, belong in those coroutines, not in `batch.py`.

- **S3Remote** uses `aioboto3.Session()` — one session per instance, one client per method call;
  batch methods share that client across concurrent tasks.
- **GCSRemote** opens one `aiohttp.ClientSession` per method call, shared by a batch's concurrent
  transfers, and fetches the token before the first request, so missing credentials raise one
  `RemoteConnectionError` before any transfer (S3 batches instead report each item failed). It
  passes its own `aiohttp.ClientTimeout` (no total deadline, `remote.connect_timeout` to connect)
  to every storage request it makes, because gcloud-aio's defaults are whole-request deadlines of
  10-30s, too short for large objects. Two requests keep gcloud-aio's fixed 10s total, because
  its public API exposes no timeout for them: a resumable upload's session-start POST, and the
  token request. It retries transient failures `remote.retries` times. Uploads above 8 MiB use a
  resumable upload. `gcs._storage_class` makes its data PUT a single attempt, so a transient
  failure is retried with a fresh upload session: gcloud-aio's own retry resends the body with
  `Content-Range: bytes */*` on the same session, which Cloud Storage refuses with a 400. That
  overrides gcloud-aio's private `Storage._do_upload`; the `gcs` extra pins gcloud-aio-storage to
  9.6, and `test_gcs_stub.py` fails if `_do_upload` or any method on the path to it from
  `Storage.upload` changes.
- **A GCS upload counts only when the object resource the server returns has the key, the local
  size and the local file's MD5** (`md5Hash`). Any other reply (gcloud-aio returns the JSON of any
  reply below 400, a `308` included) is retried as a fresh upload, then fails, so `sync` never
  records a hash the remote lacks.
- **No GCS error or log line carries an HTTP client exception's str or repr, or chains one as its
  cause.** aiohttp's repr includes the request headers, and so the bearer token. Report the status
  and Google's message (`gcs._describe`, `gcs._remote_error`) and raise `from None`. The same rule
  is why `_client` turns off gcloud-aio's background token refresh: nothing awaits that task, so a
  failed refresh would reach asyncio's "Task exception was never retrieved" log with its repr.

### Auth Conventions

| Service | Token Source | Env Vars |
|---------|-------------|----------|
| GitHub API | Environment variable | `GH_TOKEN`, then `GITHUB_TOKEN` |
| S3 | Standard AWS credential chain | `AWS_*` env vars, `~/.aws/`, instance profile |
| GCS | Application Default Credentials | `GOOGLE_APPLICATION_CREDENTIALS`, `gcloud auth application-default login`, metadata server; `STORAGE_EMULATOR_HOST` for an emulator |

### Error Handling

Network errors are **expected** in the remote layer. Always provide actionable messages:
- 403 on GitHub → "Authentication required. Set GH_TOKEN environment variable."
- 429 on GitHub → "Rate limited. Set GH_TOKEN for higher limits."
- S3 AccessDenied → "Cannot access {url}. Check AWS credentials or ask source repo owner for access."
- GCS 403 → "Permission denied to {read|list|write} gs://{bucket}/{key}: {Google's message}" (Google's message names the missing permission)
- GCS 401 or no token → the `gcloud auth application-default login` instruction
- Connection errors → "Cannot reach {host}. Check network connectivity."
