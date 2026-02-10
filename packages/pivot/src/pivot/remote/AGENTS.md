# Pivot Remote - Development Guidelines

## Async-First I/O (Critical)

**All network I/O in the remote layer MUST be async.** This is not optional.

The remote layer uses `aiohttp` (direct dependency) for HTTP and `aioboto3` for S3.
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
2. **Consistency**: S3Remote is already fully async; mixing sync HTTP would create two patterns
3. **Status checks**: `--check-imports` checks multiple imports concurrently
4. **Connection reuse**: `aiohttp.ClientSession` reuses TCP connections across requests

### HTTP Client: aiohttp

Use `aiohttp` for all HTTP calls (GitHub API, git forge APIs). Do NOT use:
- `httpx` — not in our dependencies
- `urllib.request` — synchronous, no connection reuse, no async support
- `requests` — not in our dependencies

`aiohttp` is a direct dependency (also available transitively via `aioboto3 → aiobotocore`).

### S3 Access Pattern

S3Remote uses `aioboto3.Session()` — one session per instance, one client per method call.
See root `AGENTS.md` for details. When creating temporary S3Remote instances (e.g., for
importing from a source repo's bucket), follow the same pattern.

### Auth Conventions

| Service | Token Source | Env Vars |
|---------|-------------|----------|
| GitHub API | Environment variable | `GH_TOKEN`, then `GITHUB_TOKEN` |
| S3 | Standard AWS credential chain | `AWS_*` env vars, `~/.aws/`, instance profile |

### Error Handling

Network errors are **expected** in the remote layer. Always provide actionable messages:
- 403 on GitHub → "Authentication required. Set GH_TOKEN environment variable."
- 429 on GitHub → "Rate limited. Set GH_TOKEN for higher limits."
- S3 AccessDenied → "Cannot access {url}. Check AWS credentials or ask source repo owner for access."
- Connection errors → "Cannot reach {host}. Check network connectivity."
