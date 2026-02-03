# Code Quality Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove dead code and consolidate duplicated utilities identified in PR review (#327)

**Architecture:** Remove unused code paths, extract shared CLI utilities to `_run_common.py`, add `path_utils.py` for trailing slash normalization

**Tech Stack:** Python 3.13+, pytest

---

## Task 1: Remove `hash_artifact()` and `HashResolution` Dead Code

**Files:**
- Modify: `src/pivot/explain.py`
- Delete tests: `tests/test_explain.py` (partial - remove `test_hash_artifact_*` functions)

**Step 1: Read the current explain.py to identify exact lines**

Run: `head -250 src/pivot/explain.py`

**Step 2: Remove `HashResolution` TypedDict (around lines 35-43)**

Remove:
```python
class HashResolution(TypedDict):
    """Result of resolving a hash to its source."""

    path: str
    content_hash: str
    source: Literal["file", "cache", "not_found"]
    cached_path: str | None
```

**Step 3: Remove `hash_artifact()` function (around lines 169-211)**

Remove the entire `hash_artifact()` function.

**Step 4: Remove tests for deleted code**

In `tests/test_explain.py`, remove all `test_hash_artifact_*` test functions (approximately lines 18-161).

**Step 5: Run tests to verify nothing breaks**

Run: `uv run pytest tests/test_explain.py tests/core/test_explain.py -v`
Expected: All remaining tests pass

**Step 6: Commit**

```bash
jj describe -m "refactor(explain): remove unused hash_artifact() and HashResolution"
```

---

## Task 2: Remove `apply_dep_path_overrides()` Dead Code

**Files:**
- Modify: `src/pivot/stage_def.py`
- Delete tests: `tests/test_dep_injection.py` (partial)

**Step 1: Read stage_def.py to identify the function and helper**

Run: `grep -n "apply_dep_path_overrides\|_validate_path_overrides_common" src/pivot/stage_def.py`

**Step 2: Verify `_validate_path_overrides_common()` is only used by `apply_dep_path_overrides()`**

Run: `grep -n "_validate_path_overrides_common" src/pivot/stage_def.py`

If only called from `apply_dep_path_overrides()`, remove both.

**Step 3: Remove `_validate_path_overrides_common()` (around lines 108-152)**

Remove the helper function.

**Step 4: Remove `apply_dep_path_overrides()` (around lines 741-773)**

Remove the function.

**Step 5: Remove the related tests in `tests/test_dep_injection.py`**

Remove:
- `test_dep_spec_path_override()`
- `test_dep_spec_path_override_partial()`
- `test_dep_spec_path_override_unknown_key_raises()`

**Step 6: Run tests to verify nothing breaks**

Run: `uv run pytest tests/test_dep_injection.py -v`
Expected: Remaining tests pass

**Step 7: Commit**

```bash
jj describe -m "refactor(stage_def): remove unused apply_dep_path_overrides()"
```

---

## Task 3: Remove `EngineState.SHUTDOWN` Dead Code

**Files:**
- Modify: `src/pivot/engine/types.py`
- Delete tests: `tests/engine/test_types.py` (partial - remove SHUTDOWN assertion)

**Step 1: Read types.py to find the enum**

Run: `grep -n "SHUTDOWN" src/pivot/engine/types.py`

**Step 2: Remove `SHUTDOWN = "shutdown"` from EngineState enum**

**Step 3: Update docstring if it references SHUTDOWN**

**Step 4: Remove test assertion for SHUTDOWN in `tests/engine/test_types.py`**

**Step 5: Run tests**

Run: `uv run pytest tests/engine/test_types.py -v`
Expected: PASS

**Step 6: Commit**

```bash
jj describe -m "refactor(engine): remove unused EngineState.SHUTDOWN"
```

---

## Task 4: Remove `ignore_filter` Parameter from `hash_directory()`

**Files:**
- Modify: `src/pivot/storage/cache.py`
- Delete tests: `tests/storage/test_cache.py` (partial - remove `test_hash_directory_*ignore_filter*` tests)

**Step 1: Read cache.py to understand the parameter usage**

Run: `grep -n "ignore_filter" src/pivot/storage/cache.py`

**Step 2: Remove `ignore_filter` parameter from `_scandir_recursive()` function**

Remove the parameter and the conditional block (lines 174-181 approximately).

**Step 3: Remove `ignore_filter` parameter from `hash_directory()` function**

Remove the parameter and stop passing it to `_scandir_recursive()`.

**Step 4: Remove ignore_filter tests in `tests/storage/test_cache.py`**

Remove all `test_hash_directory_*ignore_filter*` test functions.

**Step 5: Run tests**

Run: `uv run pytest tests/storage/test_cache.py -v`
Expected: Remaining tests pass

**Step 6: Commit**

```bash
jj describe -m "refactor(cache): remove unused ignore_filter parameter"
```

---

## Task 5: Extract Shared CLI Utilities to `_run_common.py`

**Files:**
- Create/Modify: `src/pivot/cli/_run_common.py`
- Modify: `src/pivot/cli/run.py`
- Modify: `src/pivot/cli/repro.py`

**Step 1: Read current `_run_common.py` to understand existing utilities**

Run: `cat src/pivot/cli/_run_common.py`

**Step 2: Read the duplicated code in `run.py`**

Read `_JsonlSink`, `_configure_result_collector`, `_configure_output_sink`, and `_convert_results` from `run.py`.

**Step 3: Add `JsonlSink` class to `_run_common.py`**

Copy from `run.py` (has better docs), rename to `JsonlSink` (public).

**Step 4: Add `configure_result_collector()` to `_run_common.py`**

```python
def configure_result_collector(eng: engine.Engine) -> sinks.ResultCollectorSink:
    """Add ResultCollectorSink to collect execution results."""
    result_sink = sinks.ResultCollectorSink()
    eng.add_sink(result_sink)
    return result_sink
```

**Step 5: Add `configure_output_sink()` to `_run_common.py`**

Copy from `run.py`.

**Step 6: Add `convert_results()` to `_run_common.py`**

```python
def convert_results(
    stage_results: dict[str, StageCompleted],
) -> dict[str, executor_core.ExecutionSummary]:
    """Convert StageCompleted events to ExecutionSummary."""
    return {
        name: executor_core.ExecutionSummary(
            status=event["status"],
            reason=event["reason"],
        )
        for name, event in stage_results.items()
    }
```

**Step 7: Update `run.py` to import from `_run_common`**

Replace local definitions with imports.

**Step 8: Update `repro.py` to import from `_run_common`**

Replace local definitions with imports.

**Step 9: Run CLI tests**

Run: `uv run pytest tests/cli/ -v`
Expected: PASS

**Step 10: Commit**

```bash
jj describe -m "refactor(cli): extract shared utilities to _run_common.py"
```

---

## Task 6: Create `path_utils.py` for Trailing Slash Normalization

**Files:**
- Create: `src/pivot/path_utils.py`
- Modify: `src/pivot/executor/worker.py`
- Modify: `src/pivot/pipeline/pipeline.py`
- Modify: `src/pivot/storage/lock.py`
- Modify: `src/pivot/registry.py`

**Step 1: Create `src/pivot/path_utils.py` with shared utilities**

```python
"""Path manipulation utilities."""

from __future__ import annotations


def preserve_trailing_slash(original: str, normalized: str) -> str:
    """Restore trailing slash if original had it.

    pathlib.Path operations strip trailing slashes, but DirectoryOut paths
    must preserve them. Use this after any path normalization.
    """
    if original.endswith("/") and not normalized.endswith("/"):
        return normalized + "/"
    return normalized


def ensure_trailing_slash(path: str) -> str:
    """Add trailing slash if not present.

    Use for paths that must be directories (e.g., DirectoryOut).
    """
    if path and not path.endswith("/"):
        return path + "/"
    return path
```

**Step 2: Update `worker.py` to use `preserve_trailing_slash()`**

Replace inline logic in `_normalize_out_path()`.

**Step 3: Update `pipeline.py` to use `preserve_trailing_slash()`**

Replace inline logic in `_resolve_path()`.

**Step 4: Update `lock.py` to use `preserve_trailing_slash()`**

Replace inline logic in `_convert_to_storage_format()` and `_convert_from_storage_format()`.

**Step 5: Update `registry.py` to use `preserve_trailing_slash()`**

Replace inline logic in `_normalize_paths()`.

**Step 6: Create tests for path_utils**

Create `tests/test_path_utils.py`:

```python
"""Tests for path_utils module."""

from pivot import path_utils


def test_preserve_trailing_slash_with_slash() -> None:
    assert path_utils.preserve_trailing_slash("foo/", "foo") == "foo/"


def test_preserve_trailing_slash_without_slash() -> None:
    assert path_utils.preserve_trailing_slash("foo", "foo") == "foo"


def test_preserve_trailing_slash_already_has_slash() -> None:
    assert path_utils.preserve_trailing_slash("foo/", "foo/") == "foo/"


def test_ensure_trailing_slash_adds() -> None:
    assert path_utils.ensure_trailing_slash("foo") == "foo/"


def test_ensure_trailing_slash_idempotent() -> None:
    assert path_utils.ensure_trailing_slash("foo/") == "foo/"


def test_ensure_trailing_slash_empty() -> None:
    assert path_utils.ensure_trailing_slash("") == ""
```

**Step 7: Run tests**

Run: `uv run pytest tests/test_path_utils.py -v`
Expected: PASS

Run: `uv run pytest tests/ -n auto`
Expected: All tests pass

**Step 8: Commit**

```bash
jj describe -m "refactor: extract trailing slash utilities to path_utils.py"
```

---

## Task 7: Final Verification

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -n auto`
Expected: All tests pass

**Step 2: Run type checker**

Run: `uv run basedpyright`
Expected: No errors

**Step 3: Run linter**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: Clean

**Step 4: Squash commits if needed**

```bash
jj squash
```

---

## Out of Scope (Documented for Future)

The following issues from #327 are **not addressed** in this plan:

1. **Atomic writes for existing loaders (JSON/YAML/CSV)** - Requires more investigation into whether pandas/PyYAML handle atomicity internally
2. **Author name mismatch in marketplace.json** - Minor, can be fixed separately
3. **TypedDict `.get()` usage in #300** - Low priority
4. **Test coverage for determinism functions (#268)** - Additive, not cleanup
5. **`_find_tracked_ancestor()` / `_is_tracked_path()` near-duplication** - Would need deeper analysis

---

## Summary

| Task | Description | Risk | Lines Removed |
|------|-------------|------|---------------|
| 1 | Remove `hash_artifact()` / `HashResolution` | Low | ~180 |
| 2 | Remove `apply_dep_path_overrides()` | Low | ~80 |
| 3 | Remove `EngineState.SHUTDOWN` | Low | ~5 |
| 4 | Remove `ignore_filter` param | Low | ~50 |
| 5 | Extract CLI utilities | Medium | ~150 (net reduction) |
| 6 | Extract trailing slash utils | Medium | ~30 (net reduction) |
| 7 | Final verification | N/A | N/A |

**Total estimated reduction:** ~500 lines of code
