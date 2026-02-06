# Cache Full Stage Fingerprint Manifests — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Cache the full `dict[str, str]` manifest from `get_stage_fingerprint()` in StateDB, validated by stat-checking source files. On hit, skip the expensive closure walk entirely.

**Architecture:** Add a `_SourceCollector` context manager to `fingerprint.py` that records which source files are visited during fingerprinting. Cache the manifest + source file stats in StateDB under `sm:` prefix. On subsequent runs, stat-check the source files — if all match, return the cached manifest. Register a unified atexit handler that flushes both AST hash and manifest caches.

**Tech Stack:** Python 3.13, LMDB (via `pivot.storage.state`), xxhash, json

---

### Task 1: Add `sm:` prefix and StateDB methods

**Files:**
- Modify: `src/pivot/storage/state.py:29` (add prefix constant)
- Modify: `src/pivot/storage/state.py:302-322` (after `clear_ast_hashes`, add new methods)
- Test: `tests/storage/test_state.py`

**Step 1: Write the failing tests**

Add to `tests/storage/test_state.py`:

```python
def test_stage_manifest_roundtrip(tmp_path: pathlib.Path) -> None:
    """Save and retrieve a stage manifest."""
    db_path = tmp_path / "state.db"
    key = "sm:my_stage\x003.13\x001"
    manifest = {"self:train": "aabb", "func:helper": "ccdd"}
    sources = {"src/train.py": [1000, 200, 555], "src/helper.py": [2000, 300, 666]}
    value = json.dumps({"m": manifest, "s": sources}, separators=(",", ":"))

    with state.StateDB(db_path) as db:
        db.put_raw(key.encode(), value.encode())
        result = db.get_raw(key.encode())

    assert result is not None
    assert json.loads(result.decode()) == {"m": manifest, "s": sources}


def test_stage_manifest_not_found(tmp_path: pathlib.Path) -> None:
    """Returns None for unknown key."""
    db_path = tmp_path / "state.db"
    with state.StateDB(db_path) as db:
        result = db.get_raw(b"sm:nonexistent\x003.13\x001")
    assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/storage/test_state.py::test_stage_manifest_roundtrip tests/storage/test_state.py::test_stage_manifest_not_found -v`
Expected: FAIL — `StateDB` has no `get_raw` / `put_raw` methods.

**Step 3: Implement StateDB methods**

In `src/pivot/storage/state.py`, add the `_SM_PREFIX` constant after the existing prefix constants (~line 29):

```python
_SM_PREFIX = b"sm:"  # Stage manifest cache entries
```

Add `get_raw` and `put_raw` methods to `StateDB` (after the AST hash section, ~line 322):

```python
    # -------------------------------------------------------------------------
    # Raw key-value access for stage manifest cache
    # -------------------------------------------------------------------------

    def get_raw(self, key: bytes) -> bytes | None:
        """Get raw value by key. Returns None if not found."""
        self._check_closed()
        with self._env.begin() as txn:
            return txn.get(key)

    def put_raw(self, key: bytes, value: bytes) -> None:
        """Put raw key-value pair."""
        self._check_closed()
        self._check_write_allowed()
        if len(key) > _MAX_KEY_SIZE:
            raise PathTooLongError(
                f"Key too long for state cache ({len(key)} bytes, max {_MAX_KEY_SIZE})"
            )
        try:
            with self._env.begin(write=True) as txn:
                txn.put(key, value)
        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e

    def put_raw_many(self, entries: list[tuple[bytes, bytes]]) -> None:
        """Batch put raw key-value pairs atomically."""
        self._check_closed()
        self._check_write_allowed()
        if not entries:
            return
        try:
            with self._env.begin(write=True) as txn:
                for key, value in entries:
                    if len(key) > _MAX_KEY_SIZE:
                        continue  # Skip oversized keys
                    txn.put(key, value)
        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/storage/test_state.py::test_stage_manifest_roundtrip tests/storage/test_state.py::test_stage_manifest_not_found -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(storage): add raw key-value access to StateDB for manifest cache"
```

---

### Task 2: Add `_SourceCollector` and `_collecting_sources` context manager

**Files:**
- Modify: `src/pivot/fingerprint.py:88-90` (add after `_pending_ast_writes`)
- Modify: `src/pivot/fingerprint.py:641` (instrument `hash_function_ast`)
- Test: `tests/fingerprint/test_fingerprint.py`

**Step 1: Write the failing test**

Add to `tests/fingerprint/test_fingerprint.py`:

```python
def test_source_collector_records_source_files(tmp_path, monkeypatch):
    """_collecting_sources context manager records source files visited during fingerprinting."""
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)

    test_module = tmp_path / "collected_stage.py"
    test_module.write_text("""
def my_stage():
    return 42
""")

    spec = importlib.util.spec_from_file_location("collected_stage", test_module)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["collected_stage"] = module
    try:
        spec.loader.exec_module(module)

        with fingerprint._collecting_sources() as collector:
            fingerprint.get_stage_fingerprint(module.my_stage)

        # Should have recorded at least the stage's source file
        assert len(collector.source_files) >= 1
        assert any("collected_stage.py" in path for path in collector.source_files)
        # Each entry should have (mtime_ns, size, inode)
        for rel_path, stats in collector.source_files.items():
            mtime_ns, size, ino = stats
            assert mtime_ns > 0
            assert size > 0
            assert ino > 0
    finally:
        sys.modules.pop("collected_stage", None)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/fingerprint/test_fingerprint.py::test_source_collector_records_source_files -v`
Expected: FAIL — `fingerprint._collecting_sources` doesn't exist yet.

**Step 3: Implement `_SourceCollector` and context manager**

In `src/pivot/fingerprint.py`, after `_pending_ast_writes` (line ~89), add:

```python
class _SourceCollector:
    """Collects source files visited during a single stage's fingerprinting."""

    __slots__ = ("source_files",)

    def __init__(self) -> None:
        self.source_files: dict[str, tuple[int, int, int]] = {}  # rel_path → (mtime_ns, size, ino)


_active_collector: _SourceCollector | None = None


@contextlib.contextmanager
def _collecting_sources() -> contextlib.AbstractContextManager[_SourceCollector]:
    """Scope a _SourceCollector for the duration of a fingerprint walk."""
    global _active_collector
    collector = _SourceCollector()
    _active_collector = collector
    try:
        yield collector
    finally:
        _active_collector = None
```

Note: the return type annotation uses `contextlib.AbstractContextManager` because `@contextmanager` returns that type. Actually, `Iterator[_SourceCollector]` is the correct annotation for the generator function. Use:

```python
from collections.abc import Iterator

@contextlib.contextmanager
def _collecting_sources() -> Iterator[_SourceCollector]:
```

Then in `hash_function_ast` (line ~641), add recording at the very top of the function, before the memory cache check:

```python
def hash_function_ast(func: Callable[..., Any]) -> str:
    # Record source file for manifest cache (before any cache check)
    if _active_collector is not None:
        _record_source_file(func)
    # ... rest unchanged
```

And implement `_record_source_file`:

```python
def _record_source_file(func: Callable[..., Any]) -> None:
    """Record a function's source file in the active collector."""
    assert _active_collector is not None
    info = _get_func_source_info(func)
    if info is not None:
        rel_path, mtime_ns, size, ino = info
        if rel_path not in _active_collector.source_files:
            _active_collector.source_files[rel_path] = (mtime_ns, size, ino)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/fingerprint/test_fingerprint.py::test_source_collector_records_source_files -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(fingerprint): add _SourceCollector to track source files during fingerprinting"
```

---

### Task 3: Implement `get_stage_fingerprint_cached`

**Files:**
- Modify: `src/pivot/fingerprint.py` (add `get_stage_fingerprint_cached`, `_try_manifest_cache_hit`, `flush_manifest_cache`)
- Test: `tests/fingerprint/test_fingerprint.py`

**Step 1: Write the failing tests**

Add to `tests/fingerprint/test_fingerprint.py`:

```python
def test_manifest_cache_hit(tmp_path, monkeypatch):
    """Compute → flush → compute again returns cached manifest (walk not called)."""
    from pivot.storage import state as state_mod

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir / "state.db"
    with state_mod.StateDB(db_path):
        pass

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_db_path", lambda: db_path)

    # Reset fingerprint state
    fingerprint._pending_ast_writes.clear()
    fingerprint._pending_manifest_writes.clear()
    fingerprint._hash_function_ast_cache.clear()
    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    test_module = tmp_path / "cached_stage.py"
    test_module.write_text("""
def cached_stage():
    return 42
""")

    spec = importlib.util.spec_from_file_location("cached_stage", test_module)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["cached_stage"] = module
    try:
        spec.loader.exec_module(module)

        # First call — computes and queues
        manifest1 = fingerprint.get_stage_fingerprint_cached("cached_stage", module.cached_stage)
        assert "self:cached_stage" in manifest1
        assert len(fingerprint._pending_manifest_writes) == 1

        # Flush
        fingerprint.flush_manifest_cache()
        assert len(fingerprint._pending_manifest_writes) == 0

        # Clear in-memory caches to force persistent lookup
        fingerprint._hash_function_ast_cache.clear()
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False

        # Second call — should hit manifest cache
        manifest2 = fingerprint.get_stage_fingerprint_cached("cached_stage", module.cached_stage)
        assert manifest2 == manifest1
        # No new pending writes (cache hit)
        assert len(fingerprint._pending_manifest_writes) == 0
    finally:
        sys.modules.pop("cached_stage", None)
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


def test_manifest_cache_miss_on_source_change(tmp_path, monkeypatch):
    """Touch source file between runs → recomputes manifest."""
    import time
    from pivot.storage import state as state_mod

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir / "state.db"
    with state_mod.StateDB(db_path):
        pass

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_db_path", lambda: db_path)

    fingerprint._pending_ast_writes.clear()
    fingerprint._pending_manifest_writes.clear()
    fingerprint._hash_function_ast_cache.clear()
    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    test_module = tmp_path / "changing_stage.py"
    test_module.write_text("""
def changing_stage():
    return 42
""")

    spec = importlib.util.spec_from_file_location("changing_stage", test_module)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["changing_stage"] = module
    try:
        spec.loader.exec_module(module)

        manifest1 = fingerprint.get_stage_fingerprint_cached("changing_stage", module.changing_stage)
        fingerprint.flush_manifest_cache()

        # Modify file
        time.sleep(0.01)
        test_module.write_text("""
def changing_stage():
    return 43
""")

        # Clear caches
        fingerprint._hash_function_ast_cache.clear()
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False

        # Reload module
        del sys.modules["changing_stage"]
        spec2 = importlib.util.spec_from_file_location("changing_stage", test_module)
        assert spec2 is not None and spec2.loader is not None
        module2 = importlib.util.module_from_spec(spec2)
        sys.modules["changing_stage"] = module2
        spec2.loader.exec_module(module2)

        manifest2 = fingerprint.get_stage_fingerprint_cached("changing_stage", module2.changing_stage)
        assert manifest2 != manifest1  # Different code = different manifest
        assert len(fingerprint._pending_manifest_writes) == 1  # Queued for flush
    finally:
        sys.modules.pop("changing_stage", None)
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


def test_manifest_cache_miss_on_file_deleted(tmp_path, monkeypatch):
    """Delete source file → recomputes manifest."""
    from pivot.storage import state as state_mod

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir / "state.db"
    with state_mod.StateDB(db_path):
        pass

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_db_path", lambda: db_path)

    fingerprint._pending_ast_writes.clear()
    fingerprint._pending_manifest_writes.clear()
    fingerprint._hash_function_ast_cache.clear()
    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    # Two-file stage: main imports helper
    helper_file = tmp_path / "helper_mod.py"
    helper_file.write_text("""
def helper():
    return 99
""")
    main_file = tmp_path / "main_mod.py"
    main_file.write_text("""
import helper_mod

def main_stage():
    return helper_mod.helper()
""")

    spec_h = importlib.util.spec_from_file_location("helper_mod", helper_file)
    assert spec_h is not None and spec_h.loader is not None
    mod_h = importlib.util.module_from_spec(spec_h)
    sys.modules["helper_mod"] = mod_h
    spec_h.loader.exec_module(mod_h)

    spec_m = importlib.util.spec_from_file_location("main_mod", main_file)
    assert spec_m is not None and spec_m.loader is not None
    mod_m = importlib.util.module_from_spec(spec_m)
    sys.modules["main_mod"] = mod_m
    try:
        spec_m.loader.exec_module(mod_m)

        manifest1 = fingerprint.get_stage_fingerprint_cached("main_stage", mod_m.main_stage)
        fingerprint.flush_manifest_cache()

        # Delete helper file — stat will fail for cached source
        helper_file.unlink()

        # Clear caches
        fingerprint._hash_function_ast_cache.clear()
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False

        # Recompute — should miss because helper_mod.py stat fails
        manifest2 = fingerprint.get_stage_fingerprint_cached("main_stage", mod_m.main_stage)
        assert len(fingerprint._pending_manifest_writes) == 1  # Cache miss → re-queued
    finally:
        sys.modules.pop("helper_mod", None)
        sys.modules.pop("main_mod", None)
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


def test_manifest_cache_non_file_backed_function(tmp_path, monkeypatch):
    """Stage referencing builtins is cacheable; builtins not tracked as source files."""
    from pivot.storage import state as state_mod

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir / "state.db"
    with state_mod.StateDB(db_path):
        pass

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_db_path", lambda: db_path)

    fingerprint._pending_ast_writes.clear()
    fingerprint._pending_manifest_writes.clear()
    fingerprint._hash_function_ast_cache.clear()
    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    test_module = tmp_path / "builtin_stage.py"
    test_module.write_text("""
def builtin_stage(items):
    return len(items)
""")

    spec = importlib.util.spec_from_file_location("builtin_stage", test_module)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["builtin_stage"] = module
    try:
        spec.loader.exec_module(module)

        with fingerprint._collecting_sources() as collector:
            fingerprint.get_stage_fingerprint(module.builtin_stage)

        # Only the stage's own source file should be tracked, not builtins
        assert len(collector.source_files) == 1
        assert any("builtin_stage.py" in path for path in collector.source_files)
    finally:
        sys.modules.pop("builtin_stage", None)
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/fingerprint/test_fingerprint.py::test_manifest_cache_hit tests/fingerprint/test_fingerprint.py::test_manifest_cache_miss_on_source_change tests/fingerprint/test_fingerprint.py::test_manifest_cache_miss_on_file_deleted tests/fingerprint/test_fingerprint.py::test_manifest_cache_non_file_backed_function -v`
Expected: FAIL — `get_stage_fingerprint_cached`, `_pending_manifest_writes`, `flush_manifest_cache` don't exist.

**Step 3: Implement the manifest cache**

In `src/pivot/fingerprint.py`, add after `_pending_ast_writes` (line ~89):

```python
# Pending manifest cache writes, flushed at process exit.
# Format: list of (key_bytes, value_bytes) tuples.
_pending_manifest_writes: list[tuple[bytes, bytes]] = []
```

Add the cache key builder (near the top, after imports):

```python
def _make_manifest_cache_key(stage_name: str) -> bytes:
    """Build StateDB key for manifest cache entry."""
    return f"sm:{stage_name}\x00{_PYTHON_VERSION}\x00{_CACHE_SCHEMA_VERSION}".encode()
```

Add `_try_manifest_cache_hit`:

```python
def _try_manifest_cache_hit(stage_name: str) -> dict[str, str] | None:
    """Try to load a cached manifest; returns None on miss."""
    db = _get_state_db()
    if db is None:
        return None

    key = _make_manifest_cache_key(stage_name)
    try:
        raw = db.get_raw(key)
    except Exception:
        return None
    if raw is None:
        return None

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    manifest: dict[str, str] = data.get("m", {})
    sources: dict[str, list[int]] = data.get("s", {})

    # Stat-check every source file
    from pivot import project
    project_root = project.get_project_root()
    for rel_path, (cached_mtime, cached_size, cached_ino) in sources.items():
        try:
            st = (project_root / rel_path).stat()
        except OSError:
            return None  # File deleted or inaccessible
        if st.st_mtime_ns != cached_mtime or st.st_size != cached_size or st.st_ino != cached_ino:
            return None  # File changed

    return manifest
```

Add `get_stage_fingerprint_cached`:

```python
def get_stage_fingerprint_cached(
    stage_name: str, func: Callable[..., Any]
) -> dict[str, str]:
    """Like get_stage_fingerprint, but with manifest-level caching.

    On hit, skips the entire closure walk. On miss, computes normally and
    queues the result for flush at process exit.
    """
    _t = metrics.start()

    # Try cache hit
    cached = _try_manifest_cache_hit(stage_name)
    if cached is not None:
        metrics.count("fingerprint.manifest_cache.hit")
        metrics.end("fingerprint.get_stage_fingerprint_cached", _t)
        return cached

    metrics.count("fingerprint.manifest_cache.miss")

    # Compute with source tracking
    with _collecting_sources() as collector:
        manifest = get_stage_fingerprint(func)

    # Queue for flush
    key = _make_manifest_cache_key(stage_name)
    value = json.dumps(
        {
            "m": manifest,
            "s": {
                rel_path: list(stats)
                for rel_path, stats in collector.source_files.items()
            },
        },
        separators=(",", ":"),
    ).encode()
    _pending_manifest_writes.append((key, value))

    metrics.end("fingerprint.get_stage_fingerprint_cached", _t)
    return manifest
```

Add `flush_manifest_cache`:

```python
def flush_manifest_cache() -> None:
    """Flush pending manifest writes to StateDB."""
    global _pending_manifest_writes
    if not _pending_manifest_writes:
        return

    pending = _pending_manifest_writes
    _pending_manifest_writes = []

    try:
        from pivot.config import io
        from pivot.storage import state

        with state.StateDB(io.get_state_db_path(), readonly=False) as db:
            db.put_raw_many(pending)
        metrics.count("fingerprint.manifest_cache.flush")
    except Exception:
        _pending_manifest_writes.extend(pending)
        _logger.debug("Failed to flush manifest cache (%d entries)", len(pending), exc_info=True)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/fingerprint/test_fingerprint.py::test_manifest_cache_hit tests/fingerprint/test_fingerprint.py::test_manifest_cache_miss_on_source_change tests/fingerprint/test_fingerprint.py::test_manifest_cache_miss_on_file_deleted tests/fingerprint/test_fingerprint.py::test_manifest_cache_non_file_backed_function -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(fingerprint): add manifest-level cache for get_stage_fingerprint"
```

---

### Task 4: Wire `_compute_fingerprint` to use `get_stage_fingerprint_cached`

**Files:**
- Modify: `src/pivot/registry.py:908-918`

**Step 1: Write no new test — this is a one-line wiring change covered by existing tests**

The existing test suite for registry fingerprinting and status should continue passing. The change is a direct substitution.

**Step 2: Make the change**

In `src/pivot/registry.py:911`, change:

```python
result = fingerprint.get_stage_fingerprint(info["func"])
```

to:

```python
result = fingerprint.get_stage_fingerprint_cached(stage_name, info["func"])
```

**Step 3: Run existing tests**

Run: `uv run pytest tests/config/test_registry.py tests/test_status.py tests/fingerprint/test_fingerprint.py -x`
Expected: PASS

**Step 4: Commit**

```bash
jj describe -m "perf(registry): use manifest-cached fingerprinting in _compute_fingerprint"
```

---

### Task 5: Register unified atexit handler

**Files:**
- Modify: `src/pivot/fingerprint.py:92-100` (replace existing atexit)

**Step 1: No new test needed — atexit ordering is verified by Task 3 tests and existing AST cache tests**

**Step 2: Replace the existing atexit registration**

Replace the existing `_close_state_db` atexit (line ~92-100):

```python
def _close_state_db() -> None:
    """Close the readonly StateDB on process exit."""
    global _state_db
    if _state_db is not None:
        _state_db.close()
        _state_db = None


atexit.register(_close_state_db)
```

with:

```python
def _close_state_db() -> None:
    """Close the readonly StateDB on process exit."""
    global _state_db
    if _state_db is not None:
        _state_db.close()
        _state_db = None


@atexit.register
def _flush_pending_caches() -> None:
    """Flush all pending cache writes at process exit.

    Registered AFTER _close_state_db so LIFO ordering runs this first.
    Flush opens its own writable StateDB, so readonly close is irrelevant.
    """
    flush_ast_hash_cache()
    flush_manifest_cache()


atexit.register(_close_state_db)
```

Note: `atexit` is LIFO — `_close_state_db` is registered first (runs last), `_flush_pending_caches` is registered second (runs first via `@atexit.register`). Wait — `@atexit.register` on `_flush_pending_caches` registers it at decoration time, then `atexit.register(_close_state_db)` registers _close_state_db after. LIFO means _close_state_db runs first, _flush runs second. That's backwards.

Correct order: register _close_state_db FIRST (runs last in LIFO), then _flush (runs first in LIFO):

```python
atexit.register(_close_state_db)  # runs last (registered first)


@atexit.register
def _flush_pending_caches() -> None:
    """Flush all pending cache writes at process exit."""
    flush_ast_hash_cache()
    flush_manifest_cache()
```

This is already the current order (`atexit.register(_close_state_db)` at line 100, then we add `_flush_pending_caches` below it). LIFO: `_flush_pending_caches` runs first (most recently registered), `_close_state_db` runs second (earliest registered). Flush opens its own writable db, close closes the readonly handle — ordering doesn't matter for correctness, but flush-before-close is cleaner.

**Step 3: Run all fingerprint and storage tests**

Run: `uv run pytest tests/fingerprint/ tests/storage/ -x`
Expected: PASS

**Step 4: Commit**

```bash
jj describe -m "perf(fingerprint): unified atexit flush for AST hash and manifest caches"
```

---

### Task 6: Run full quality checks

**Step 1: Run linting and type checking**

Run: `uv run ruff format . && uv run ruff check . && uv run basedpyright`
Expected: PASS — fix any issues found.

**Step 2: Run full test suite**

Run: `uv run pytest tests/ -n auto`
Expected: PASS

**Step 3: Squash and push**

```bash
jj git push
```

---

## Key Design Decisions

1. **`get_raw`/`put_raw` instead of typed methods** — The manifest cache value is a JSON blob with its own schema. Adding typed methods (like `get_stage_manifest()`) would create unnecessary coupling between StateDB and fingerprint.py's internal format. Raw access keeps the boundary clean.

2. **Source tracking via context manager, not always-on** — Only stages that call `get_stage_fingerprint_cached` need source tracking. Other callers of `hash_function_ast` (loader fingerprinting, ad-hoc) don't need it. The context manager scopes tracking precisely.

3. **No separate stat cache** — ~30 unique source files × ~0.06ms each = ~2ms. Not worth the complexity of a separate stat cache. Add one only if profiling shows it matters.

4. **Loader fingerprints excluded from manifest cache** — Loaders are fast to compute (dataclass config hash + method hash) and their cache keys would need to encode the full loader configuration. Not worth the complexity for a negligible speedup.
