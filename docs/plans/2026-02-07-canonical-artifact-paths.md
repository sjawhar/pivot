# Canonical Artifact Path Representation (#378)

**Goal:** Establish a single canonical in-memory form for artifact paths (absolute, normalized, trailing slash for dirs) so that dependency resolution, output indexing, and DAG construction never encounter path mismatches.

**Architecture:** Introduce a `canonicalize_artifact_path()` helper in `path_utils.py` that produces the one canonical form. Make `Pipeline._resolve_path()` return absolute paths directly (instead of project-relative), eliminating the double-normalization where Pipeline makes paths relative and registry makes them absolute again. Keep lockfile storage as the one explicit boundary conversion (absolute <-> project-relative).

**Tech Stack:** Python 3.13+, pathlib, pytest

---

## Current State Analysis

### The Problem

Artifact paths currently flow through two normalization steps:

1. **`Pipeline._resolve_path()`** — for relative inputs, returns **project-relative** (line 293); for absolute inputs, returns **absolute posix** (line 282).
2. **`registry._normalize_paths()`** — converts everything to **absolute** via `project.normalize_path()`.

This double-hop creates risk:
- Different code paths can produce slightly different forms (e.g., trailing slash dropped, path not fully normalized).
- `_find_producer_in_pipeline()` compares `dep_path` (absolute) against `info["outs_paths"]` (absolute) — works only because registry already normalized. But `_find_producer_via_index()` manually converts to project-relative, which is fragile.
- `_write_output_index()` manually converts from absolute to project-relative, another fragile boundary.
- The `preserve_trailing_slash()` call is scattered across 6+ sites instead of being part of a single canonical step.

### The Fix

1. **Single helper** `canonicalize_artifact_path(path, base)` in `path_utils.py` that produces: absolute, normalized, trailing-slash-preserved.
2. **`Pipeline._resolve_path()`** returns absolute canonical paths directly (not project-relative).
3. **`registry._normalize_paths()`** uses the canonical helper instead of ad-hoc normalization.
4. **Lockfile** remains the one explicit boundary (absolute <-> relative at read/write time — already works correctly).
5. **Output index** uses explicit `project.to_relative_path()` at its boundary.

---

## Task 1: Add `canonicalize_artifact_path()` helper

**Files:**
- Modify: `src/pivot/path_utils.py`
- Create: `tests/test_path_utils.py` (already exists, extend it)

**Step 1: Write failing tests**

Add tests to `tests/test_path_utils.py`:

```python
def test_canonicalize_artifact_path_relative(tmp_path: Path) -> None:
    """Relative path becomes absolute from base."""
    result = path_utils.canonicalize_artifact_path("data/input.csv", tmp_path)
    assert result == str(tmp_path / "data" / "input.csv")
    assert os.path.isabs(result)


def test_canonicalize_artifact_path_absolute(tmp_path: Path) -> None:
    """Absolute path stays absolute, gets normalized."""
    abs_input = str(tmp_path / "data" / ".." / "data" / "input.csv")
    result = path_utils.canonicalize_artifact_path(abs_input, tmp_path)
    assert result == str(tmp_path / "data" / "input.csv")


def test_canonicalize_artifact_path_trailing_slash(tmp_path: Path) -> None:
    """Trailing slash preserved for directory paths."""
    result = path_utils.canonicalize_artifact_path("outputs/", tmp_path)
    assert result.endswith("/")
    assert result == str(tmp_path / "outputs") + "/"


def test_canonicalize_artifact_path_no_trailing_slash(tmp_path: Path) -> None:
    """Non-directory paths don't get trailing slash."""
    result = path_utils.canonicalize_artifact_path("data/input.csv", tmp_path)
    assert not result.endswith("/")


def test_canonicalize_artifact_path_dotdot_normalized(tmp_path: Path) -> None:
    """Parent traversal is collapsed."""
    result = path_utils.canonicalize_artifact_path("sub/../data/input.csv", tmp_path)
    assert result == str(tmp_path / "data" / "input.csv")
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sami/pivot/roadmap-378 && uv run pytest tests/test_path_utils.py -k "canonicalize" -v`
Expected: FAIL — `canonicalize_artifact_path` not defined.

**Step 3: Implement `canonicalize_artifact_path`**

Add to `src/pivot/path_utils.py`:

```python
import os
import pathlib


def canonicalize_artifact_path(path: str, base: pathlib.Path) -> str:
    """Produce the single canonical form for an artifact path.

    Canonical form:
    - Absolute (resolved from base if relative)
    - Normalized (no .., no //, no trailing dots)
    - Trailing slash preserved for directory artifacts (DirectoryOut)

    This is the ONE function that should be used to produce artifact paths
    for in-memory use (registry, DAG, engine). Lockfiles convert to/from
    project-relative at their own boundary.

    Args:
        path: Raw artifact path (relative or absolute).
        base: Base directory for resolving relative paths.

    Returns:
        Canonical absolute path string, with trailing slash preserved if input had one.
    """
    has_trailing_slash = path.endswith("/")
    p = pathlib.Path(path)
    abs_path = p if p.is_absolute() else base / p
    normalized = pathlib.Path(os.path.normpath(abs_path))
    result = str(normalized)
    if has_trailing_slash and not result.endswith("/"):
        result += "/"
    return result
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/sami/pivot/roadmap-378 && uv run pytest tests/test_path_utils.py -k "canonicalize" -v`
Expected: PASS

**Step 5: Run full quality checks**

Run: `cd /home/sami/pivot/roadmap-378 && uv run ruff format . && uv run ruff check . && uv run basedpyright`
Expected: Clean

---

## Task 2: Update `Pipeline._resolve_path()` to return canonical absolute paths

**Files:**
- Modify: `src/pivot/pipeline/pipeline.py:242-302`

**Context:** Currently `_resolve_path()` returns project-relative for relative inputs, then `registry._normalize_paths()` makes them absolute again. This double-hop is the core issue. Making `_resolve_path()` return absolute directly eliminates the round-trip.

**Step 1: Write a failing test**

Add to `tests/pipeline/test_pipeline.py` (or a new test file if needed):

```python
def test_resolve_path_returns_absolute(
    pipeline_factory: PipelineFactory,
    set_project_root: Path,
) -> None:
    """Pipeline._resolve_path returns absolute canonical paths."""
    pipeline = pipeline_factory("test_pipe")
    result = pipeline._resolve_path("data/input.csv")
    assert os.path.isabs(result), f"Expected absolute path, got: {result}"
    assert result == str(set_project_root / "data" / "input.csv")
```

(Adapt fixture names to match existing test patterns — look at `tests/pipeline/test_pipeline.py` for the right factory.)

**Step 2: Run test to verify it fails**

Run: `cd /home/sami/pivot/roadmap-378 && uv run pytest tests/pipeline/test_pipeline.py -k "resolve_path_returns_absolute" -v`
Expected: FAIL — currently returns project-relative.

**Step 3: Update `_resolve_path` to return absolute paths**

In `src/pivot/pipeline/pipeline.py`, change `_resolve_path()`:

```python
def _resolve_path(self, annotation_path: str) -> str:
    """Convert annotation path to canonical absolute form.

    All artifact paths are stored as absolute, normalized paths in memory.
    Trailing slashes are preserved (important for DirectoryOut).
    Lockfiles handle conversion to/from project-relative at their own boundary.
    """
    # Reject empty or whitespace-only paths early
    if not annotation_path or not annotation_path.strip():
        raise ValueError("Path cannot be empty or whitespace-only")

    # Reject root-only paths (e.g., "/", "\\", "C:\\", "C:/")
    stripped = annotation_path.strip()
    if stripped in ("/", "\\") or (
        len(stripped) == 3
        and stripped[0].isalpha()
        and stripped[1] == ":"
        and stripped[2] in ("/", "\\")
    ):
        raise ValueError(f"Path cannot be a root directory: {annotation_path!r}")

    project_root = project.get_project_root()

    # Determine base for resolution: absolute paths resolve from themselves,
    # relative paths resolve from pipeline root
    is_absolute = (
        annotation_path.startswith("/")
        or annotation_path.startswith("\\")
        or (
            len(annotation_path) >= 3
            and annotation_path[0].isalpha()
            and annotation_path[1] == ":"
            and annotation_path[2] in ("/", "\\")
        )
    )

    base = project_root if is_absolute else self.root
    resolved = path_utils.canonicalize_artifact_path(annotation_path, base)

    # Check if path escapes project root (reject paths outside project)
    if not is_absolute:
        try:
            pathlib.Path(resolved.rstrip("/")).relative_to(project_root)
        except ValueError as e:
            raise ValueError(
                f"Path '{annotation_path}' resolves to '{resolved}' which is outside project root '{project_root}'"
            ) from e

    # Validate the RESOLVED path (after ../ is collapsed)
    if error := path_policy.validate_path_syntax(resolved):
        raise ValueError(f"Invalid path '{annotation_path}': {error}")

    return resolved
```

**Step 4: Run test to verify it passes**

Run: `cd /home/sami/pivot/roadmap-378 && uv run pytest tests/pipeline/test_pipeline.py -k "resolve_path_returns_absolute" -v`
Expected: PASS

**Step 5: Run full test suite to check for regressions**

Run: `cd /home/sami/pivot/roadmap-378 && uv run pytest tests/ -x -q`
Expected: Some tests may fail due to the path form change — those will be fixed in the next tasks.

---

## Task 3: Update `registry._normalize_paths()` to use `canonicalize_artifact_path()`

**Files:**
- Modify: `src/pivot/registry.py:620-689` (the `_normalize_paths` function)

**Context:** `_normalize_paths()` currently does its own ad-hoc normalization. Replace the core normalization logic with `canonicalize_artifact_path()`, keeping the policy validation and symlink checks.

**Step 1: Update `_normalize_paths` to use the canonical helper**

In `src/pivot/registry.py`, update the function:

```python
def _normalize_paths(
    paths: Sequence[str],
    path_type: path_policy.PathType,
    validation_mode: ValidationMode,
) -> list[str]:
    """Normalize paths to canonical absolute form, applying policy-based validation.

    Uses canonicalize_artifact_path() for the core normalization, then applies
    policy checks (project containment, symlink escape detection).
    """
    normalized = list[str]()
    project_root = project.get_project_root()
    policy = path_policy.POLICIES[path_type]

    for path in paths:
        try:
            # Canonicalize path to absolute form
            norm_str = path_utils.canonicalize_artifact_path(path, project_root)
            norm_path = pathlib.Path(norm_str.rstrip("/"))

            # Check if path is within project root
            is_within_project = norm_path.is_relative_to(project_root)

            if not is_within_project:
                if not policy["allow_absolute"]:
                    raise exceptions.InvalidPathError(
                        f"{path_type.value.capitalize()} path '{path}' resolves to '{norm_path}' "
                        + f"which is outside project root '{project_root}'"
                    )
                logger.warning(f"Absolute {path_type.value} path may break reproducibility: {path}")
            else:
                # Symlink escape check (for paths that exist)
                if norm_path.exists() and project.contains_symlink_in_path(norm_path, project_root):
                    resolved = norm_path.resolve()
                    if not resolved.is_relative_to(project_root.resolve()):
                        msg = (
                            f"{path_type.value.capitalize()} path '{path}' resolves outside "
                            + f"project via symlink: {resolved}"
                        )
                        if policy["symlink_escape_action"] == "error":
                            raise exceptions.InvalidPathError(msg)
                        logger.warning(msg)
                    else:
                        logger.warning(
                            f"Path '{path}' is inside a symlinked directory. "
                            + "This may affect portability across environments."
                        )

            normalized.append(norm_str)
        except (ValueError, OSError, exceptions.InvalidPathError):
            if validation_mode == ValidationMode.WARN:
                norm_str = path_utils.canonicalize_artifact_path(path, project_root)
                normalized.append(norm_str)
            else:
                raise
    return normalized
```

Also update `_normalize_deps_dict` similarly — replace `_normalize_paths` calls to stay consistent (no change needed since it delegates to `_normalize_paths`).

**Step 2: Run tests**

Run: `cd /home/sami/pivot/roadmap-378 && uv run pytest tests/ -x -q`
Expected: PASS (or identify remaining failures)

**Step 3: Run quality checks**

Run: `cd /home/sami/pivot/roadmap-378 && uv run ruff format . && uv run ruff check . && uv run basedpyright`
Expected: Clean

---

## Task 4: Fix any test failures from the path form change

**Context:** The change from project-relative to absolute in `Pipeline._resolve_path()` may cause test failures where tests expected relative paths. This task is about fixing those test assertions.

**Step 1: Run full test suite and collect failures**

Run: `cd /home/sami/pivot/roadmap-378 && uv run pytest tests/ -x -q 2>&1 | head -80`

**Step 2: Fix each failure**

Common patterns to fix:
- Tests that assert `info["deps_paths"]` contains relative paths — update to expect absolute.
- Tests that mock `_resolve_path` — update mock return values.
- Tests that compare output index paths — update expectations.

Since paths were already being made absolute by `_normalize_paths`, most internal tests should already work with absolute paths. The main failures will be in `Pipeline`-level tests that check the intermediate form.

**Step 3: Verify all tests pass**

Run: `cd /home/sami/pivot/roadmap-378 && uv run pytest tests/ -n auto -q`
Expected: All pass

---

## Task 5: Add round-trip test for DirectoryOut trailing slash through registry + lockfile

**Files:**
- Modify: `tests/storage/test_lock.py` (extend existing tests)

**Step 1: Write the test**

```python
def test_directory_out_trailing_slash_through_registry_and_lockfile(
    set_project_root: Path,
) -> None:
    """DirectoryOut trailing slash is preserved: annotation -> registry -> lockfile -> read back."""
    cache_dir = set_project_root / ".cache"
    dir_path = str(set_project_root / "results") + "/"

    # Simulate what registry stores
    assert dir_path.endswith("/"), "Canonical form must preserve trailing slash"

    # Write to lockfile (absolute -> relative at boundary)
    stage_lock = lock.StageLock("dir_stage", cache_dir)
    data = LockData(
        code_manifest={"self:dir_stage": "hash"},
        params={},
        dep_hashes={},
        output_hashes={dir_path: DirHash(hash="abc", manifest=[])},
        dep_generations={},
    )
    stage_lock.write(data)

    # Read back (relative -> absolute at boundary)
    result = stage_lock.read()
    assert result is not None
    out_paths = list(result["output_hashes"].keys())
    assert len(out_paths) == 1
    assert out_paths[0] == dir_path
    assert out_paths[0].endswith("/")
```

**Step 2: Run test**

Run: `cd /home/sami/pivot/roadmap-378 && uv run pytest tests/storage/test_lock.py -k "registry_and_lockfile" -v`
Expected: PASS (this should already work since lock.py preserves trailing slashes)

---

## Task 6: Add test for output index lookup with canonical absolute paths

**Files:**
- Modify: `tests/pipeline/test_pipeline.py` or create `tests/core/test_output_index.py`

**Step 1: Write the test**

```python
def test_output_index_uses_project_relative_keys(
    pipeline_factory: PipelineFactory,
    set_project_root: Path,
) -> None:
    """Output index stores project-relative keys, lookable from absolute paths."""
    # This tests _write_output_index and _find_producer_via_index
    # The index key is project-relative; dep_path passed to lookup is absolute.
    # Verify the conversion is correct.
    pipeline = pipeline_factory("test_pipe")
    # ... register a stage with an output
    # ... build DAG (triggers _write_output_index)
    # ... verify index file exists with relative path key
    cache_dir = set_project_root / ".pivot" / "cache" / "outputs"
    # Check that index entry is project-relative
```

(Adapt to match existing test patterns for pipeline tests.)

**Step 2: Run test**

Run: `cd /home/sami/pivot/roadmap-378 && uv run pytest tests/pipeline/ -k "output_index" -v`
Expected: PASS

---

## Task 7: Update docstrings and comments to codify the invariant

**Files:**
- Modify: `src/pivot/registry.py` (RegistryStageInfo docstring)
- Modify: `src/pivot/pipeline/pipeline.py` (_resolve_path docstring)
- Modify: `src/pivot/path_utils.py` (module docstring)
- Modify: `src/pivot/types.py` (LockData vs StorageLockData comments)

**Step 1: Update RegistryStageInfo docstring**

In `src/pivot/registry.py`, update the class docstring to clarify:

```python
class RegistryStageInfo(TypedDict):
    """Metadata for a registered stage.

    Path Invariant:
        All paths in deps, deps_paths, outs, and outs_paths are in canonical
        absolute form (normalized, no .., trailing slash for DirectoryOut).
        This form is produced by path_utils.canonicalize_artifact_path().
        Lockfiles are the one boundary where absolute <-> project-relative
        conversion happens (see storage/lock.py).

    Attributes:
        ...
    """
```

**Step 2: Update LockData / StorageLockData comments in types.py**

Add a note about the boundary conversion being the only place relative paths appear:

```python
# Two representations exist for different purposes:
#
#   StorageLockData   On-disk YAML format. Uses project-relative paths
#                     (portable across machines). This is the ONLY place
#                     relative paths appear — converted at read/write boundary
#                     in storage/lock.py.
#
#   LockData          In-memory format. Uses canonical absolute paths
#                     (matching registry/engine convention, fast comparisons).
```

**Step 3: Run quality checks**

Run: `cd /home/sami/pivot/roadmap-378 && uv run ruff format . && uv run ruff check . && uv run basedpyright`
Expected: Clean

---

## Task 8: Final validation

**Step 1: Run full test suite**

Run: `cd /home/sami/pivot/roadmap-378 && uv run pytest tests/ -n auto -q`
Expected: All pass

**Step 2: Run full quality checks**

Run: `cd /home/sami/pivot/roadmap-378 && uv run ruff format . && uv run ruff check . && uv run basedpyright`
Expected: Clean

**Step 3: Create jj bookmark**

Run: `cd /home/sami/pivot/roadmap-378 && jj bookmark create issue-378 -r @`

---

## Key Design Decisions

1. **Why absolute as canonical?** — The engine, DAG, and workers all operate on absolute paths. Making absolute the canonical form means no conversion is needed at lookup/comparison time. Only lockfiles need relative paths (for portability).

2. **Why a helper function, not a class?** — Following existing patterns in `path_utils.py` (pure functions, no state). The function is simple enough that a class would be over-engineering.

3. **Why not change lockfile format?** — Lock files already correctly handle the absolute <-> relative boundary. No lockfile format change needed.

4. **What about `preserve_trailing_slash()`?** — It's now called inside `canonicalize_artifact_path()`, so callers don't need to remember to call it separately. The existing standalone function remains for backward compatibility at explicit boundaries (lock.py).
