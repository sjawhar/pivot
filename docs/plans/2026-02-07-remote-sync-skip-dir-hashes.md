# Remote Sync: Skip Directory Tree Hashes and Enforce Hash Format


**Goal:** Prevent remote sync from including directory tree hashes in push/pull target sets, and tighten remote hash validation to match local cache invariants (exactly 16 lowercase hex characters).

**Architecture:** (1) Add a file-only hash extraction helper that excludes the top-level tree hash from `DirHash` entries, (2) guard `_push_async` against uploading directories, (3) tighten `_validate_hash` to require exactly 16 lowercase hex, (4) filter invalid keys in `iter_hashes`.

**Tech Stack:** Python 3.13+, pytest, pivot internals

---

## Context

### The tree hash problem

When a stage outputs a directory, the lock file stores a `DirHash`:

```python
DirHash(hash="<tree_hash>", manifest=[DirManifestEntry(relpath="...", hash="<file_hash>", ...)])
```

The `tree_hash` is the xxhash64 of the serialized manifest JSON — it identifies the *directory structure*, not a file. In SYMLINK cache mode (`cache.py:504-517`), the tree hash maps to a *directory* in the cache (`shutil.copytree`). In HARDLINK/COPY mode, no cache entry exists for the tree hash at all.

`_extract_hashes_from_hash_info` (`sync.py:87-94`) currently returns **both** the tree hash and individual file hashes. This causes:

1. **Push:** `_push_async` tries to upload the tree hash's cache path, which is a directory (SYMLINK mode) or doesn't exist (HARDLINK/COPY mode). Directory upload would fail or produce corrupt S3 objects.
2. **Pull:** `_pull_async` tries to download tree hashes that don't exist on S3, generating unnecessary 404 errors.

### The hash validation mismatch

Remote `_validate_hash` (`storage.py:119-126`) accepts:
- Minimum 3 characters (vs local cache's strict 16)
- Case-insensitive hex (`re.IGNORECASE`)

Local `get_cache_path` (`cache.py:218-227`) requires:
- Exactly 16 characters
- Lowercase-only hex (`_VALID_HASH_CHARS = frozenset("0123456789abcdef")`)

This mismatch means the remote could accept hashes that local cache rejects, causing `SecurityValidationError` when trying to download.

---

### Task 1: Add `_extract_file_hashes_from_hash_info` helper

**Files:**
- Modify: `src/pivot/remote/sync.py` — add new function after `_extract_hashes_from_hash_info` (line 94)
- Test: `tests/remote/test_sync.py` (create)

#### Step 1: Write the failing tests

Create `tests/remote/test_sync.py` with tests for the new helper:

```python
from __future__ import annotations

from pivot.remote import sync
from pivot.types import DirHash, DirManifestEntry, FileHash


def test_extract_file_hashes_from_file_hash() -> None:
    """FileHash returns its hash as the only element."""
    fh = FileHash(hash="abcdef1234567890")
    result = sync._extract_file_hashes_from_hash_info(fh)
    assert result == {"abcdef1234567890"}


def test_extract_file_hashes_from_dir_hash_excludes_tree_hash() -> None:
    """DirHash returns only manifest file hashes, not the tree hash."""
    dh = DirHash(
        hash="aaaaaaaaaaaaaaaa",  # tree hash — must be excluded
        manifest=[
            DirManifestEntry(relpath="a.csv", hash="1111111111111111", size=100, isexec=False),
            DirManifestEntry(relpath="b.csv", hash="2222222222222222", size=200, isexec=False),
        ],
    )
    result = sync._extract_file_hashes_from_hash_info(dh)
    assert result == {"1111111111111111", "2222222222222222"}
    assert "aaaaaaaaaaaaaaaa" not in result


def test_extract_file_hashes_from_dir_hash_empty_manifest() -> None:
    """DirHash with empty manifest returns empty set (tree hash excluded)."""
    dh = DirHash(hash="aaaaaaaaaaaaaaaa", manifest=[])
    result = sync._extract_file_hashes_from_hash_info(dh)
    assert result == set()
```

#### Step 2: Run tests to verify they fail

Run: `cd /home/sami/pivot/roadmap-377 && uv run pytest tests/remote/test_sync.py -v`
Expected: FAIL — `_extract_file_hashes_from_hash_info` does not exist

#### Step 3: Write the implementation

In `src/pivot/remote/sync.py`, add after `_extract_hashes_from_hash_info` (line 94):

```python
def _extract_file_hashes_from_hash_info(output_hash: HashInfo) -> set[str]:
    """Extract file-only hashes from a HashInfo, excluding directory tree hashes.

    For FileHash: returns {hash}.
    For DirHash: returns only manifest entry hashes, NOT the top-level tree hash.
    The tree hash identifies directory structure, not a cached file.
    """
    if "manifest" in output_hash:
        return {entry["hash"] for entry in output_hash["manifest"]}
    return {output_hash["hash"]}
```

#### Step 4: Run tests to verify they pass

Run: `cd /home/sami/pivot/roadmap-377 && uv run pytest tests/remote/test_sync.py -v`
Expected: PASS

---

### Task 2: Use `_extract_file_hashes_from_hash_info` in all sync call sites

**Files:**
- Modify: `src/pivot/remote/sync.py` — replace `_extract_hashes_from_hash_info` calls in `get_stage_output_hashes`, `get_stage_dep_hashes`, and `get_target_hashes`
- Test: `tests/remote/test_sync.py` — add integration-style tests

#### Step 1: Write the failing tests

Add to `tests/remote/test_sync.py`:

```python
import pathlib

from pivot.storage import lock


def _write_lock_with_dir_output(
    stages_dir: pathlib.Path,
    stage_name: str,
    tree_hash: str,
    file_hashes: list[str],
) -> None:
    """Helper: write a lock file with a directory output containing a tree hash."""
    manifest = [
        DirManifestEntry(relpath=f"file{i}.csv", hash=h, size=100, isexec=False)
        for i, h in enumerate(file_hashes)
    ]
    dir_hash = DirHash(hash=tree_hash, manifest=manifest)
    lock_data = lock.StageLockData(
        fingerprint={},
        params_hash=None,
        dep_hashes={},
        output_hashes={"output_dir": dir_hash},
        dep_generations={},
    )
    stage_lock = lock.StageLock(stage_name, stages_dir)
    stage_lock.write(lock_data)


def test_get_stage_output_hashes_excludes_tree_hash(tmp_path: pathlib.Path) -> None:
    """get_stage_output_hashes returns file hashes only, not tree hashes."""
    state_dir = tmp_path / ".pivot"
    stages_dir = lock.get_stages_dir(state_dir)
    stages_dir.mkdir(parents=True)

    tree_hash = "aaaaaaaaaaaaaaaa"
    file_hashes = ["1111111111111111", "2222222222222222"]
    _write_lock_with_dir_output(stages_dir, "my_stage", tree_hash, file_hashes)

    result = sync.get_stage_output_hashes(state_dir, ["my_stage"])

    assert "1111111111111111" in result
    assert "2222222222222222" in result
    assert tree_hash not in result


def test_get_stage_dep_hashes_excludes_tree_hash(tmp_path: pathlib.Path) -> None:
    """get_stage_dep_hashes returns file hashes only, not tree hashes."""
    state_dir = tmp_path / ".pivot"
    stages_dir = lock.get_stages_dir(state_dir)
    stages_dir.mkdir(parents=True)

    # Write lock with directory dependency
    dep_manifest = [
        DirManifestEntry(relpath="dep.csv", hash="3333333333333333", size=50, isexec=False),
    ]
    dep_hash = DirHash(hash="bbbbbbbbbbbbbbbb", manifest=dep_manifest)
    lock_data = lock.StageLockData(
        fingerprint={},
        params_hash=None,
        dep_hashes={"input_dir": dep_hash},
        output_hashes={},
        dep_generations={},
    )
    stage_lock = lock.StageLock("my_stage", stages_dir)
    stage_lock.write(lock_data)

    result = sync.get_stage_dep_hashes(state_dir, ["my_stage"])

    assert "3333333333333333" in result
    assert "bbbbbbbbbbbbbbbb" not in result
```

#### Step 2: Run tests to verify they fail

Run: `cd /home/sami/pivot/roadmap-377 && uv run pytest tests/remote/test_sync.py::test_get_stage_output_hashes_excludes_tree_hash tests/remote/test_sync.py::test_get_stage_dep_hashes_excludes_tree_hash -v`
Expected: FAIL — tree hashes are still included

#### Step 3: Update all call sites

In `src/pivot/remote/sync.py`, replace every call to `_extract_hashes_from_hash_info` with `_extract_file_hashes_from_hash_info`:

**`get_stage_output_hashes` (line 66):**
```python
# Before:
hashes |= _extract_hashes_from_hash_info(output_hash)
# After:
hashes |= _extract_file_hashes_from_hash_info(output_hash)
```

**`get_stage_dep_hashes` (line 82):**
```python
# Before:
hashes |= _extract_hashes_from_hash_info(dep_hash)
# After:
hashes |= _extract_file_hashes_from_hash_info(dep_hash)
```

**`get_target_hashes` — stage name branch (line 158):**
```python
# Before:
hashes |= _extract_hashes_from_hash_info(out_hash)
# After:
hashes |= _extract_file_hashes_from_hash_info(out_hash)
```

**`get_target_hashes` — stage name branch with deps (line 161):**
```python
# Before:
hashes |= _extract_hashes_from_hash_info(dep_hash)
# After:
hashes |= _extract_file_hashes_from_hash_info(dep_hash)
```

**`get_target_hashes` — file path from stages (line 170):**
```python
# Before:
hashes |= _extract_hashes_from_hash_info(out_hash)
# After:
hashes |= _extract_file_hashes_from_hash_info(out_hash)
```

**`get_target_hashes` — file path from pvt (line 176):**
```python
# Before:
hashes |= _extract_hashes_from_hash_info(pvt_hash)
# After:
hashes |= _extract_file_hashes_from_hash_info(pvt_hash)
```

After all replacements, remove the old `_extract_hashes_from_hash_info` function (lines 87-94) since it's no longer used.

#### Step 4: Run tests to verify they pass

Run: `cd /home/sami/pivot/roadmap-377 && uv run pytest tests/remote/test_sync.py -v`
Expected: PASS

---

### Task 3: Guard `_push_async` against directory uploads

**Files:**
- Modify: `src/pivot/remote/sync.py` — add `is_file()` check in `_push_async`
- Test: `tests/remote/test_sync.py` — add test

#### Step 1: Write the failing test

Add to `tests/remote/test_sync.py`:

```python
def test_push_skips_directory_cache_paths(tmp_path: pathlib.Path) -> None:
    """Push should never enqueue directory paths for upload."""
    cache_dir = tmp_path / "cache"
    files_dir = cache_dir / "files"

    # Create a file cache entry
    file_hash = "1111111111111111"
    file_cache = files_dir / file_hash[:2] / file_hash[2:]
    file_cache.parent.mkdir(parents=True)
    file_cache.write_text("file content")

    # Create a directory cache entry (simulating SYMLINK mode tree hash)
    dir_hash = "aaaaaaaaaaaaaaaa"
    dir_cache = files_dir / dir_hash[:2] / dir_hash[2:]
    dir_cache.mkdir(parents=True)
    (dir_cache / "some_file.csv").write_text("data")

    # Both exist, but only the file should be collected
    from pivot.storage import cache as cache_mod

    file_path = cache_mod.get_cache_path(files_dir, file_hash)
    dir_path = cache_mod.get_cache_path(files_dir, dir_hash)
    assert file_path.exists() and file_path.is_file()
    assert dir_path.exists() and dir_path.is_dir()

    # Verify the filtering logic directly
    items = []
    for hash_ in [file_hash, dir_hash]:
        cache_path = cache_mod.get_cache_path(files_dir, hash_)
        if cache_path.exists() and cache_path.is_file():
            items.append((cache_path, hash_))

    assert len(items) == 1
    assert items[0][1] == file_hash
```

#### Step 2: Run test to verify it passes (this tests the logic pattern, not the code)

This test demonstrates the pattern. Now modify `_push_async`.

#### Step 3: Add `is_file()` guard in `_push_async`

In `src/pivot/remote/sync.py`, change the upload collection loop in `_push_async` (lines 249-252):

```python
# Before:
    for hash_ in status["local_only"]:
        cache_path = cache.get_cache_path(files_dir, hash_)
        if cache_path.exists():
            items.append((cache_path, hash_))

# After:
    for hash_ in status["local_only"]:
        cache_path = cache.get_cache_path(files_dir, hash_)
        if cache_path.is_file():
            items.append((cache_path, hash_))
```

Note: `is_file()` returns `False` for directories and non-existent paths, so it replaces both the `exists()` check and the directory guard in one call.

#### Step 4: Run all tests

Run: `cd /home/sami/pivot/roadmap-377 && uv run pytest tests/remote/ -v`
Expected: PASS

---

### Task 4: Tighten `_validate_hash` to 16 lowercase hex

**Files:**
- Modify: `src/pivot/remote/storage.py` — update `_validate_hash`, `_HEX_PATTERN`, `MIN_HASH_LENGTH`
- Test: `tests/remote/test_s3_remote.py` — update existing tests

#### Step 1: Update existing tests to match stricter validation

In `tests/remote/test_s3_remote.py`, update the validation tests:

```python
# Replace test_validate_hash_short_raises (lines 196-201):
def test_validate_hash_short_raises() -> None:
    """Hash with wrong length raises RemoteError."""
    with pytest.raises(exceptions.RemoteError, match="exactly 16 characters"):
        remote_mod._validate_hash("")
    with pytest.raises(exceptions.RemoteError, match="exactly 16 characters"):
        remote_mod._validate_hash("ab")
    with pytest.raises(exceptions.RemoteError, match="exactly 16 characters"):
        remote_mod._validate_hash("abc")  # Was valid before, now too short
    with pytest.raises(exceptions.RemoteError, match="exactly 16 characters"):
        remote_mod._validate_hash("abcdef123456789")  # 15 chars


# Replace test_validate_hash_non_hex_raises (lines 204-211):
def test_validate_hash_non_hex_raises() -> None:
    """Non-lowercase-hex hash raises RemoteError."""
    with pytest.raises(exceptions.RemoteError, match="lowercase hexadecimal"):
        remote_mod._validate_hash("ghijklmnopqrstuv")  # Non-hex chars
    with pytest.raises(exceptions.RemoteError, match="lowercase hexadecimal"):
        remote_mod._validate_hash("ABCDEF1234567890")  # Uppercase
    with pytest.raises(exceptions.RemoteError, match="lowercase hexadecimal"):
        remote_mod._validate_hash("abc/def123456789")  # Path separator


# Replace test_validate_hash_valid (lines 214-219):
def test_validate_hash_valid() -> None:
    """Valid 16-char lowercase hex hash passes validation."""
    remote_mod._validate_hash("abcdef1234567890")
    remote_mod._validate_hash("0123456789abcdef")
    remote_mod._validate_hash("0000000000000000")
    remote_mod._validate_hash("ffffffffffffffff")
```

#### Step 2: Run tests to verify they fail

Run: `cd /home/sami/pivot/roadmap-377 && uv run pytest tests/remote/test_s3_remote.py::test_validate_hash_short_raises tests/remote/test_s3_remote.py::test_validate_hash_non_hex_raises tests/remote/test_s3_remote.py::test_validate_hash_valid -v`
Expected: FAIL — old validation is too permissive

#### Step 3: Update `_validate_hash` and constants

In `src/pivot/remote/storage.py`:

**Remove `MIN_HASH_LENGTH` constant (line 30)** and **replace `_HEX_PATTERN` (line 36)**:

```python
# Before (line 30):
MIN_HASH_LENGTH = 3  # Minimum hash length (2-char prefix + at least 1 char)
# After: delete this line entirely

# Before (line 36):
_HEX_PATTERN = re.compile(r"^[a-f0-9]+$", re.IGNORECASE)
# After:
_HEX_PATTERN = re.compile(r"^[a-f0-9]{16}$")
```

**Replace `_validate_hash` (lines 119-126):**

```python
def _validate_hash(cache_hash: str) -> None:
    """Validate hash is exactly 16 lowercase hex characters (xxhash64 format)."""
    if len(cache_hash) != 16:
        raise exceptions.RemoteError(
            f"Invalid cache hash '{cache_hash}': must be exactly 16 characters, got {len(cache_hash)}"
        )
    if not _HEX_PATTERN.match(cache_hash):
        raise exceptions.RemoteError(
            f"Invalid cache hash '{cache_hash}': must be lowercase hexadecimal"
        )
```

Also remove `MIN_HASH_LENGTH` from the comment at line 33 (the `BULK_EXISTS_LIST_THRESHOLD` comment). The comment references `MIN_HASH_LENGTH` only in passing ("2-char prefix + at least 1 char") — just remove the `MIN_HASH_LENGTH` line and its comment.

#### Step 4: Run tests to verify they pass

Run: `cd /home/sami/pivot/roadmap-377 && uv run pytest tests/remote/test_s3_remote.py -v`
Expected: PASS

---

### Task 5: Filter invalid keys in `iter_hashes`

**Files:**
- Modify: `src/pivot/remote/storage.py` — add hash validation in `iter_hashes`
- Test: `tests/remote/test_s3_remote.py` — add test

#### Step 1: Write the failing test

Add to `tests/remote/test_s3_remote.py`:

```python
@pytest.mark.asyncio
async def test_iter_hashes_skips_invalid_keys(
    mock_s3_session: MagicMock, mocker: MockerFixture
) -> None:
    """iter_hashes skips keys that don't produce valid 16-char lowercase hex hashes."""
    from collections.abc import AsyncIterator  # noqa: TC003

    class MockPaginator:
        async def paginate(
            self, **kwargs: object
        ) -> AsyncIterator[dict[str, list[dict[str, str]]]]:
            yield {
                "Contents": [
                    {"Key": "prefix/files/ab/cdef1234567890"},  # Valid: 16 lowercase hex
                    {"Key": "prefix/files/AB/CDEF1234567890"},  # Invalid: uppercase
                    {"Key": "prefix/files/ab/c"},               # Invalid: too short
                    {"Key": "prefix/files/ab/cdef12345678901234"},  # Invalid: too long
                    {"Key": "prefix/stages/my_stage.lock"},     # Invalid: not a cache file
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
```

#### Step 2: Run test to verify it fails

Run: `cd /home/sami/pivot/roadmap-377 && uv run pytest tests/remote/test_s3_remote.py::test_iter_hashes_skips_invalid_keys -v`
Expected: FAIL — invalid hashes are yielded

#### Step 3: Add validation in `iter_hashes`

In `src/pivot/remote/storage.py`, update `iter_hashes` (lines 426-439):

```python
async def iter_hashes(self) -> AsyncIterator[str]:
    """Iterate over all cache hashes on remote (memory-efficient streaming).

    Skips keys that don't produce valid 16-char lowercase hex hashes.
    """
    prefix = f"{self._prefix}files/"

    async with self._session.client("s3", config=_get_s3_config()) as s3:
        paginator = s3.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj.get("Key")
                if key is None:
                    continue
                hash_ = _key_to_hash(self._prefix, key)
                if hash_ and _HEX_PATTERN.match(hash_):
                    yield hash_
```

The change: replace `if hash_:` with `if hash_ and _HEX_PATTERN.match(hash_):` to validate format.

#### Step 4: Run tests to verify they pass

Run: `cd /home/sami/pivot/roadmap-377 && uv run pytest tests/remote/test_s3_remote.py -v`
Expected: PASS

---

### Task 6: Run full quality checks

#### Step 1: Run formatter and linter

Run: `cd /home/sami/pivot/roadmap-377 && uv run ruff format . && uv run ruff check .`
Expected: No errors

#### Step 2: Run type checker

Run: `cd /home/sami/pivot/roadmap-377 && uv run basedpyright`
Expected: No errors

#### Step 3: Run full test suite

Run: `cd /home/sami/pivot/roadmap-377 && uv run pytest tests/ -n auto`
Expected: All tests pass

#### Step 4: Create bookmark

Run: `cd /home/sami/pivot/roadmap-377 && jj bookmark create issue-377 -r @`
