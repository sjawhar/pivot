# PR #447 Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all 28 code review items from PR #447 (compositional API rewrite) — bugs, dead code, stale docs, missing validation. One item (I1) needs no fix (already centralized).

**Architecture:** Grouped by file/subsystem to minimize context switches. Each group is independent. TDD where behavior changes; direct fixes for dead code removal and doc updates.

**Tech Stack:** Python 3.13+, pytest, basedpyright, ruff

---

## Task 1: Delete scratch files (C1)

**Files:**
- Delete: `ANALYSIS.md`
- Delete: `PRESENTATION_LAYER_PATTERNS.md`

**Step 1: Delete files**
```bash
rm ANALYSIS.md PRESENTATION_LAYER_PATTERNS.md
```

**Step 2: Commit**
```bash
git add -A && git commit -m "chore: delete scratch files from PR"
```

---

## Task 2: Fix artifact lock key collision (C5)

**Files:**
- Modify: `packages/pivot/src/pivot/storage/artifact_lock.py:59-60`
- Test: `packages/pivot/tests/storage/test_artifact_lock.py`

**Step 1: Write the failing test**

In `test_artifact_lock.py`, add:

```python
def test_lock_key_none_matches_identity_key():
    """_lock_key with key=None must match types.identity_key for consistency."""
    from pivot.storage import artifact_lock
    from pivot import types

    ref_no_key = types.ArtifactRef(
        identity=types.ArtifactIdentity("stage", None),
        format=None,
        python_type=object,
        tag=types.ArtifactTag.DATA,
    )
    ref_with_key = types.ArtifactRef(
        identity=types.ArtifactIdentity("stage", "out"),
        format=None,
        python_type=object,
        tag=types.ArtifactTag.DATA,
    )
    # Lock key must match identity_key
    assert artifact_lock._lock_key(ref_no_key) == types.identity_key(ref_no_key.identity)
    assert artifact_lock._lock_key(ref_with_key) == types.identity_key(ref_with_key.identity)
    # key=None must NOT produce "stage:None"
    assert "None" not in artifact_lock._lock_key(ref_no_key)
```

**Step 2: Run test — expect FAIL**
```bash
pytest packages/pivot/tests/storage/test_artifact_lock.py::test_lock_key_none_matches_identity_key -xvs
```

**Step 3: Fix `_lock_key`**

In `artifact_lock.py:59-60`, change:
```python
def _lock_key(ref: types.ArtifactRef) -> str:
    return f"{ref.identity.producer}:{ref.identity.key}"
```
to:
```python
def _lock_key(ref: types.ArtifactRef) -> str:
    return types.identity_key(ref.identity)
```

**Step 4: Run test — expect PASS**

**Step 5: Commit**
```bash
git add -A && git commit -m "fix: artifact lock key collision for key=None (C5)"
```

---

## Task 3: Fix CacheStore StateDB path (C6)

**Files:**
- Modify: `packages/pivot/src/pivot/storage/store.py:268`
- Test: `packages/pivot/tests/storage/test_store.py`

**Step 1: Write the failing test**

```python
def test_cache_store_state_db_path_inside_pivot_dir(tmp_path: pathlib.Path):
    """CacheStore StateDB must live inside .pivot/, not at project root."""
    from pivot.storage import store as store_mod

    project_root = tmp_path
    pivot_dir = project_root / ".pivot"
    pivot_dir.mkdir()
    cache_dir = pivot_dir / "cache"
    cache_dir.mkdir()

    spec = store_mod.StoreSpec(
        kind="cache",
        cache_dir=str(cache_dir),
        project_root=str(project_root),
        pipeline_name="test",
        input_bindings={},
    )
    s = store_mod.store_from_spec(spec)
    assert isinstance(s, store_mod.CacheStore)
    # StateDB path must be inside .pivot/, not at project root
    assert s._state_db_path is not None
    assert ".pivot" in str(s._state_db_path)
    assert str(s._state_db_path).startswith(str(pivot_dir))
```

**Step 2: Run test — expect FAIL**

**Step 3: Fix path**

In `store.py:268`, change:
```python
state_db_path = project_root / ".pivot"
```
to:
```python
state_db_path = project_root / ".pivot" / "cache-state"
```

This way `StateDB.__init__` computes `db_path.parent / "state.lmdb"` = `.pivot/state.lmdb` (inside `.pivot/`).

**Step 4: Run test — expect PASS**

**Step 5: Commit**
```bash
git add -A && git commit -m "fix: CacheStore StateDB path leaking to project root (C6)"
```

---

## Task 4: Fix dangling symlink checks (I12)

**Files:**
- Modify: `packages/pivot/src/pivot/storage/store.py:259-261`
- Modify: `packages/pivot/src/pivot/engine/engine.py:1603`
- Test: `packages/pivot/tests/storage/test_store.py`

**Step 1: Write the failing test**

```python
def test_workspace_store_exists_dangling_symlink(tmp_path: pathlib.Path):
    """exists() must return False for dangling symlinks."""
    from pivot.storage import store as store_mod
    from pivot import types

    ws = store_mod.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="test",
        input_bindings={},
    )
    ref = types.ArtifactRef(
        identity=types.ArtifactIdentity("stage", None),
        format=None,
        python_type=object,
        tag=types.ArtifactTag.DATA,
    )
    # Create output dir and a dangling symlink
    out_path = ws._resolve_output_path(ref)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.symlink_to(tmp_path / "nonexistent_target")
    assert out_path.is_symlink()
    assert not out_path.exists()  # target gone
    assert not ws.exists(ref), "exists() should return False for dangling symlink"
```

**Step 2: Run test — expect FAIL**

**Step 3: Fix `exists()` in both locations**

In `store.py:259-261`, change:
```python
def exists(self, ref: types.ArtifactRef) -> bool:
    path = self._resolve_path(ref)
    return path.exists() or path.is_symlink()
```
to:
```python
def exists(self, ref: types.ArtifactRef) -> bool:
    path = self._resolve_path(ref)
    return path.exists()
```

In `engine.py:1603`, change:
```python
if not (out_path.exists() or out_path.is_symlink()):
```
to:
```python
if not out_path.exists():
```

Note: `Path.exists()` follows symlinks — returns True if the symlink target exists, False if dangling. This is the correct behavior.

**Step 4: Run test — expect PASS**

**Step 5: Commit**
```bash
git add -A && git commit -m "fix: dangling symlinks incorrectly treated as existing (I12)"
```

---

## Task 5: Guard `_ensure_symlink` for directories (M5)

**Files:**
- Modify: `packages/pivot/src/pivot/storage/presentation.py:77-79`

**Step 1: Fix**

In `presentation.py:77-79`, change:
```python
    if display_path.is_symlink() or display_path.exists():
        display_path.unlink()
```
to:
```python
    if display_path.is_symlink() or display_path.exists():
        if display_path.is_dir() and not display_path.is_symlink():
            import shutil
            shutil.rmtree(display_path)
        else:
            display_path.unlink()
```

**Step 2: Commit**
```bash
git add -A && git commit -m "fix: _ensure_symlink handles directory outputs (M5)"
```

---

## Task 6: compose.py fixes (I2, M1, C7, I9, I10)

**Files:**
- Modify: `packages/pivot/src/pivot/compose.py:44, 96-98, 256-258, 315, 521-534`
- Test: `packages/pivot/tests/test_compose.py`

### Step 1: Import SINGLE_OUTPUT_KEY from stage_def (I2)

In `compose.py:44`, change:
```python
SINGLE_OUTPUT_KEY = "_single"
```
to:
```python
SINGLE_OUTPUT_KEY = stage_def.SINGLE_OUTPUT_KEY
```

This works because line 17 already imports `stage_def`.

### Step 2: Fix error message (M1)

In `compose.py:96-98`, change:
```python
                f"Use .field or ['key']. Available: {available}"
```
to:
```python
                f"Use handle.key or handle['key']. Available: {available}"
```

### Step 3: Add pipeline handle validation (C7)

In `compose.py:256-258`, after `for param_name, value in bound.arguments.items():`, add validation. Change:
```python
        for param_name, value in bound.arguments.items():
            if isinstance(value, ArtifactHandle):
                input_handles[param_name] = value
```
to:
```python
        for param_name, value in bound.arguments.items():
            if isinstance(value, ArtifactHandle):
                if value._pipeline is not self:
                    self._validation_errors.append(
                        f"{stage_name}: parameter '{param_name}' is a handle from a "
                        f"different Pipeline instance. All handles must come from the "
                        f"same pipeline."
                    )
                input_handles[param_name] = value
```

Also add the same check for list handles. After:
```python
            elif isinstance(value, (list, tuple)) and all(
                isinstance(v, ArtifactHandle) for v in value
            ):
```
add before `list_input_handles[param_name] = list(value)`:
```python
                for v in value:
                    if isinstance(v, ArtifactHandle) and v._pipeline is not self:
                        self._validation_errors.append(
                            f"{stage_name}: parameter '{param_name}' contains a handle "
                            f"from a different Pipeline instance."
                        )
                        break
```

### Step 4: Add validation in build() (I9)

In `compose.py:315`, change:
```python
    def build(self) -> pipeline_mod.Pipeline:
        legacy = pipeline_mod.Pipeline(self._name, root=self._root)
```
to:
```python
    def build(self) -> pipeline_mod.Pipeline:
        self._validate()
        legacy = pipeline_mod.Pipeline(self._name, root=self._root)
```

### Step 5: Add Required/NotRequired unwrapping (I10)

In `compose.py:521-534`, the `_parse_output_type` function iterates TypedDict fields. `get_type_hints(hint, include_extras=True)` already strips `Required[]`/`NotRequired[]` wrappers in Python 3.13+. Verify this with a test:

```python
def test_typeddict_required_fields_unwrapped():
    """Required[]/NotRequired[] wrappers must not break output parsing."""
    from typing import Required, NotRequired
    from pivot.compose import _analyze_return_type

    class Outputs(TypedDict):
        required_out: Required[Annotated[DataFrame, CSV()]]
        optional_out: NotRequired[Annotated[dict, YAML()]]

    def fn() -> Outputs: ...

    specs = _analyze_return_type(fn)
    keys = {s.key for s in specs}
    assert "required_out" in keys
    assert "optional_out" in keys
```

Note: `_analyze_return_type` expects the raw function, NOT `fn.__wrapped__`.

If `get_type_hints` already handles this (likely in 3.13+), we just need the test to prove it. If it fails, add unwrapping at the start of `_parse_output_type`:

```python
from typing import Required, NotRequired

def _parse_output_type(hint: Any, key: str) -> list[_OutputSpec]:
    # Unwrap Required[T] / NotRequired[T]
    origin = get_origin(hint)
    if origin is Required or origin is NotRequired:
        hint = get_args(hint)[0]
    # ... rest of function
```

### Step 6: Write tests for C7 and I9

```python
def test_cross_pipeline_handle_rejected(tmp_path: pathlib.Path):
    """Handles from different Pipeline instances must be rejected."""
    from pivot.compose import Pipeline, stage

    @stage
    def source() -> Annotated[DataFrame, CSV()]: ...

    @stage
    def consumer(data: Any) -> Annotated[DataFrame, CSV()]: ...

    p1 = Pipeline("p1", root=tmp_path)
    p2 = Pipeline("p2", root=tmp_path)

    with p1:
        h = source()

    with pytest.raises(ValueError, match="different Pipeline"):
        with p2:
            consumer(h)
            p2.build()


def test_build_without_context_manager_validates(tmp_path: pathlib.Path):
    """build() must run validation even without context manager."""
    from pivot.compose import Pipeline

    p = Pipeline("test", root=tmp_path)
    # No stages registered, no errors — should succeed
    p.build()
```

### Step 7: Run tests, commit

```bash
pytest packages/pivot/tests/test_compose.py -xvs
git add -A && git commit -m "fix: compose.py - pipeline validation, SINGLE_OUTPUT_KEY dedup, error msg, Required unwrap (I2, M1, C7, I9, I10)"
```

---

## Task 7: Extract shared `_format_extension` (I3)

**Files:**
- Modify: `packages/pivot/src/pivot/compose.py:463-483`
- Modify: `packages/pivot/src/pivot/storage/store.py:164-181`
- Create: shared utility location

**Step 1:** Move `_format_extension` from `compose.py` to `loaders.py` (or a shared location). Since both `compose.py` and `store.py` import `loaders`, put it there:

In `packages/pivot/src/pivot/loaders.py`, add at bottom:
```python
def format_extension(fmt: object) -> str:
    """Map a loader/writer instance to a file extension."""
    match fmt:
        case DataFrameJSONL():
            return "jsonl"
        case CSV():
            return "csv"
        case YAML():
            return "yaml"
        case JSON():
            return "json"
        case Text():
            return "txt"
        case Pickle():
            return "pkl"
        case MatplotlibFigure():
            return "png"
        case _:
            return "dat"
```

**Step 2:** In `compose.py`, replace the function with:
```python
def _format_extension(fmt: object) -> str:
    loaders = _get_loaders()
    return loaders.format_extension(fmt)
```

**Step 3:** In `store.py`, replace the method:
```python
def _format_extension(self, fmt: object) -> str:
    return loaders.format_extension(fmt)
```

**Step 4: Run tests, commit**
```bash
pytest packages/pivot/tests/test_compose.py packages/pivot/tests/storage/test_store.py -xvs
git add -A && git commit -m "refactor: extract _format_extension to loaders module (I3)"
```

---

## Task 8: Delete worker dead code (I4)

**Files:**
- Modify: `packages/pivot/src/pivot/executor/worker.py`

**Step 1:** Delete dead code sections. **Work bottom-up** (highest line numbers first) so earlier line references stay valid:

1. **Lines 942-1010**: Delete `_run_stage_function_with_injection`.
2. **Lines 817-895**: Delete `_prepare_outputs_for_execution`, `_save_outputs_to_cache`, `_hash_outputs_only`.
3. **Lines 611-634**: In `_deps_list_for_input_hash`, remove the `isinstance(deps_info, dict)` check — always use the dict path. Delete the dead `return [...]` list branch (lines 630-633).
4. **Lines 590-596**: Delete `_uses_artifact_refs()`. Remove its call at line 216.
5. **Lines 578-587**: Delete `_legacy_lock_requests()`.
6. **Lines 455-492**: Remove the old injection execution path (the `else` branch of `if use_store`).
7. **Lines 257-260**: Remove the `isinstance(deps_info, list)` branch. Inline the `else` body.
8. **Line 216**: After deleting `_uses_artifact_refs`, remove the `use_store` variable entirely. Verify all `if use_store:` branches now only have the `True` path remaining, then remove the conditions.

**Step 2: Run full test suite for worker**
```bash
pytest packages/pivot/tests/execution/ -xvs
```

**Step 3: Commit**
```bash
git add -A && git commit -m "chore: delete worker dead dual-path execution code (I4)"
```

---

## Task 9: Delete explain.py re-exports (I5)

**Files:**
- Modify: `packages/pivot/src/pivot/explain.py:31-34`

**Step 1:** Delete these lines:
```python
# Re-exports for backward compatibility (tests reference these)
diff_code_manifests = skip.diff_code_manifests
diff_params = skip.diff_params
diff_dep_hashes = skip.diff_dep_hashes
```

No callers found (verified: `from pivot.explain import` has zero matches in tests).

**Step 2: Run tests, commit**
```bash
pytest packages/pivot/tests/ -x --timeout=30
git add -A && git commit -m "chore: delete unused explain.py re-exports (I5)"
```

---

## Task 10: Delete ValidationMode dead code (M2)

**Files:**
- Modify: `packages/pivot/src/pivot/registry.py:54-59, 66, 69`

**Step 1:** Delete `ValidationMode` enum (lines 54-58). Change `StageRegistry.__init__`:

From:
```python
def __init__(self, validation_mode: ValidationMode = ValidationMode.ERROR) -> None:
    self._stages = dict[str, RegistryStageInfo]()
    self._cached_dag: DiGraph[str] | None = None
    self.validation_mode: ValidationMode = validation_mode
```
To:
```python
def __init__(self) -> None:
    self._stages = dict[str, RegistryStageInfo]()
    self._cached_dag: DiGraph[str] | None = None
```

Also remove the `import enum` if no longer needed.

**Step 2: Search for any callers passing `validation_mode`**
```bash
grep -r "validation_mode\|ValidationMode" packages/pivot/
```
Update any callers found.

**Step 3: Run tests, commit**
```bash
pytest packages/pivot/tests/ -x --timeout=60
git add -A && git commit -m "chore: delete unused ValidationMode enum (M2)"
```

---

## Task 11: Fix @no_fingerprint decorator stacking (I11)

**Files:**
- Modify: `packages/pivot/src/pivot/compose.py:315-320` (build method)
- Modify: `packages/pivot/src/pivot/registry.py:170-171`
- Test: `packages/pivot/tests/test_compose.py`

**Architecture note:** The bug has two layers:
1. `_compute_fingerprint` only checks the innermost function (after `inspect.unwrap`)
2. compose.py's `build()` passes `node.original_func` (the inner function) to RegistryStageInfo — the `@no_fingerprint` attribute on the outer wrapper is never seen

Both must be fixed. The compose.py fix propagates the attribute; the registry fix adds defense-in-depth.

**Step 1: Write the failing test**

```python
def test_no_fingerprint_outside_stage_decorator(tmp_path: pathlib.Path):
    """@no_fingerprint() applied outside @stage must be detected through compose → registry."""
    from pivot import no_fingerprint
    from pivot.compose import Pipeline, stage
    from pivot.registry import _compute_fingerprint

    @no_fingerprint()
    @stage
    def my_stage() -> Annotated[str, Text()]:
        return "hello"

    # Build pipeline to trigger compose.py → RegistryStageInfo creation
    p = Pipeline("test", root=tmp_path)
    with p:
        my_stage()
    built = p.build()

    info = built._registry._stages["my_stage"]
    fp = _compute_fingerprint("my_stage", info)
    # File-hash fingerprint uses "file:" prefix keys, not AST analysis
    assert any(k.startswith("file:") for k in fp), (
        "Expected file-hash fingerprint when @no_fingerprint is applied outside @stage"
    )
```

**Step 2: Run test — expect FAIL**

**Step 3: Fix compose.py build() — propagate attribute**

In `compose.py`, inside `build()`, after `func = node.original_func` (line 320), add:

```python
            func = node.original_func
            # Propagate @no_fingerprint from outer wrapper (e.g. @no_fingerprint() @stage)
            if getattr(node.func, "__pivot_no_fingerprint__", False):
                func.__pivot_no_fingerprint__ = True  # type: ignore[attr-defined]
```

Here `node.func` is the `@stage` wrapper (which `@no_fingerprint` set the attribute on), and `func` is the original function that goes into RegistryStageInfo.

**Step 4: Fix registry.py — defense-in-depth**

In `registry.py:170-171`, also walk the wrapper chain:
```python
        # Check for __pivot_no_fingerprint__ at each wrapper layer
        no_fp = getattr(info["func"], "__pivot_no_fingerprint__", False)
        if not no_fp:
            unwrapped = inspect.unwrap(info["func"])
            no_fp = getattr(unwrapped, "__pivot_no_fingerprint__", False)
        if no_fp:
```

**Step 5: Run test — expect PASS**

**Step 6: Commit**
```bash
git add -A && git commit -m "fix: detect @no_fingerprint at any decorator layer (I11)"
```

---

## Task 12: Collapse graph.py old types (I6)

**Files:**
- Modify: `packages/pivot/src/pivot/engine/graph.py:92-110`

**Step 1:** Simplify `_output_identities` and `_dep_identity`:

Replace:
```python
def _output_identities(
    out: pivot_types.ArtifactRef | outputs.BaseOut,
) -> list[pivot_types.ArtifactIdentity]:
    if isinstance(out, pivot_types.ArtifactRef):
        return [out.identity]
    path = out.path
    if isinstance(path, (list, tuple)):
        return [_path_identity(str(item)) for item in path]
    return [_path_identity(str(path))]


def _dep_identity(dep: pivot_types.ArtifactRef | str) -> pivot_types.ArtifactIdentity:
    if isinstance(dep, pivot_types.ArtifactRef):
        return dep.identity
    return _path_identity(dep)
```

With:
```python
def _output_identities(
    out: pivot_types.ArtifactRef,
) -> list[pivot_types.ArtifactIdentity]:
    return [out.identity]


def _dep_identity(dep: pivot_types.ArtifactRef) -> pivot_types.ArtifactIdentity:
    return dep.identity
```

Remove the `outputs` import if no longer needed.

**Step 2: Run tests**
```bash
pytest packages/pivot/tests/engine/test_graph.py -xvs
```

**Step 3: Commit**
```bash
git add -A && git commit -m "refactor: collapse graph.py to accept only ArtifactRef (I6)"
```

---

## Task 13: types.py and stage_def.py doc fixes (M3, M4, I8)

**Files:**
- Modify: `packages/pivot/src/pivot/types.py:245-258, 453-456`
- Modify: `packages/pivot/src/pivot/stage_def.py:24-41`

### Step 1: Update lock file comments (M3)

In `types.py:245-258`, replace the stale comment block with:
```python
# Two representations exist for different purposes:
#
#   StorageLockData   On-disk YAML format. Uses project-relative identity keys
#                     and list-based deps/outs (stable YAML output).
#
#   LockData          In-memory format. Uses ArtifactIdentity keys
#                     and dict-based deps/outs (O(1) lookups by identity).
#
# Conversion happens at read/write time in storage/lock.py.
#
```

### Step 2: Rename OutputChange.path to identity (M4)

In `types.py:453-456`, change:
```python
class OutputChange(TypedDict):
    """Change info for an output file."""

    path: ArtifactIdentity
```
to:
```python
class OutputChange(TypedDict):
    """Change info for an output artifact."""

    identity: ArtifactIdentity
```

(This type is not referenced anywhere in the codebase, so no callers to update.)

### Step 3: Fix stale stage_def.py docstring (I8)

In `stage_def.py:24-41`, replace:
```python
class StageParams(pydantic.BaseModel):
    """Base class for stage parameters (Pydantic model).

    Use as a simple base class for parameter-only stages:

        class TrainParams(StageParams):
            learning_rate: float = 0.01
            batch_size: int = 32

        def train(
            config: TrainParams,
            data: Annotated[DataFrame, Dep("input.csv", CSV())],
        ) -> TrainOutputs:
            ...

    For testing, just pass the data directly:

        result = train(TrainParams(learning_rate=0.5), test_df)
    """
```
with:
```python
class StageParams(pydantic.BaseModel):
    """Base class for stage parameters (Pydantic model).

    Use as a simple base class for stage configuration:

        class TrainParams(StageParams):
            learning_rate: float = 0.01
            batch_size: int = 32

        @pivot.stage
        def train(config: TrainParams, data: DataFrame) -> DataFrame:
            ...

    For testing, call the function directly:

        result = train(TrainParams(learning_rate=0.5), test_df)
    """
```

### Step 4: Commit
```bash
git add -A && git commit -m "docs: fix stale comments in types.py, stage_def.py, rename OutputChange.path (M3, M4, I8)"
```

---

## Task 14: Fix merkle input_merkle_ids iteration (M6)

**Files:**
- Modify: `packages/pivot/src/pivot/engine/engine.py:1771-1775`
- Test: `packages/pivot/tests/engine/test_merkle.py`

**Step 1: Fix the iteration (no unit test feasible)**

The `_update_merkle_ids_for_stage` is a private method deeply coupled to engine state. A unit test would require mocking the entire engine. The fix is a one-line change with clear intent (iterate deps, not outs). Existing integration tests cover merkle correctness.

**Step 2: Apply the fix**

In `engine.py:1771-1775`, change:
```python
        input_merkle_ids = {
            types.identity_key(ref.identity): self._merkle_ids[types.identity_key(ref.identity)]
            for ref in stage_info["outs"]
            if types.identity_key(ref.identity) in self._merkle_ids
        }
```
to:
```python
        input_merkle_ids = {
            types.identity_key(ref.identity): self._merkle_ids[types.identity_key(ref.identity)]
            for ref in stage_info["deps"].values()
            if types.identity_key(ref.identity) in self._merkle_ids
        }
```

**Step 3: Run tests, commit**
```bash
pytest packages/pivot/tests/engine/ -xvs
git add -A && git commit -m "fix: merkle input_merkle_ids should iterate deps, not outs (M6)"
```

---

## Task 15: CLI/API cleanup — pivot.yaml removal (C8)

**Files:**
- Modify: `packages/pivot/src/pivot/discovery.py:26-51`
- Modify: `packages/pivot/src/pivot/pipeline/yaml.py` — delete model classes (keep only rejection function)
- Modify: `packages/pivot/src/pivot/cli/completion.py:208-248`
- Modify: `packages/pivot/src/pivot/cli/helpers.py:29-33`
- Modify: `packages/pivot/src/pivot/cli/init.py:80`
- Delete: `packages/pivot/src/pivot/cli/schema.py`

### Step 1: Simplify discovery.py

In `discovery.py:26-51`, replace `find_config_in_dir` to only look for `pipeline.py`:

```python
def find_config_in_dir(directory: pathlib.Path) -> pathlib.Path | None:
    """Find the pipeline config file in a directory.

    Returns the path to pipeline.py if found, None otherwise.
    """
    pipeline_path = directory / PIPELINE_PY_NAME
    if pipeline_path.is_file():
        return pipeline_path
    return None
```

Delete `PIVOT_YAML_NAMES` constant. Also update its two external callers:
- `cli/targets.py:140`: Remove PIVOT_YAML_NAMES from `_PIPELINE_FILENAMES` frozenset
- `cli/completion.py:140`: Remove PIVOT_YAML_NAMES from the pipeline filename iteration

### Step 2: Simplify yaml.py

In `pipeline/yaml.py`, delete the Pydantic model classes (`StageConfig`, `NamedOutputOptions`, `PipelineConfig`) and `load_pipeline_file` (only used internally, no external callers). Keep only `load_pipeline_from_yaml` (the rejection function) and `PipelineConfigError` (if referenced by callers).

### Step 3: Update completion.py fast path

In `cli/completion.py:208-248`, the `_get_stages_fast()` function reads pivot.yaml. Since pivot.yaml is no longer supported, this fast path is dead. Replace with returning `None` (to always use the fallback):

```python
def _get_stages_fast() -> list[str] | None:
    """Fast path disabled — pivot.yaml no longer supported."""
    root = _find_project_root_fast()
    if root is None:
        return None
    return _get_stages_from_cache(root)
```

### Step 4: Update helpers.py error message

In `cli/helpers.py:29-33`, change:
```python
            "This command requires a pipeline to be defined in one of:\n"
            "  - pivot.yaml (or pivot.yml)\n"
            "  - pipeline.py"
```
to:
```python
            "This command requires a pipeline defined in pipeline.py"
```

### Step 5: Update init.py suggestion

In `cli/init.py:80`, change:
```python
click.echo("  1. Create pivot.yaml to define your pipeline stages")
```
to:
```python
click.echo("  1. Create pipeline.py to define your pipeline stages")
```

### Step 6: Delete schema.py

`cli/schema.py` outputs JSON schema for `PipelineConfig` — the now-rejected YAML format. Delete the file and remove its registration from `cli/__init__.py:70`.

### Step 7: Run tests, commit
```bash
pytest packages/pivot/tests/ -x --timeout=120
git add -A && git commit -m "chore: remove pivot.yaml references, update discovery to pipeline.py only (C8)"
```

---

## Task 16: Remove dvc_compat module (M7)

**Files:**
- Delete: `packages/pivot/src/pivot/dvc_compat.py`
- Delete: `packages/pivot/src/pivot/cli/export.py`
- Delete: `packages/pivot/src/pivot/cli/import_dvc.py`
- Modify: `packages/pivot/src/pivot/cli/__init__.py` (remove registrations)

**Step 1:** Delete all three files and their CLI registrations:
- Delete `packages/pivot/src/pivot/dvc_compat.py`
- Delete `packages/pivot/src/pivot/cli/export.py`
- Delete `packages/pivot/src/pivot/cli/import_dvc.py`
- Remove registrations from `cli/__init__.py`: line 34 (`export`) and lines 40-44 (`import-dvc`)

**Step 2: Run tests, commit**
```bash
pytest packages/pivot/tests/ -x --timeout=60
git add -A && git commit -m "chore: remove dvc_compat, export, import-dvc commands (M7)"
```

---

## Task 17: Remove CLI target filesystem fallback (I7) and dead BaseOut branches

**Files:**
- Modify: `packages/pivot/src/pivot/cli/targets.py:198-206, 275-283`

### Step 1: Remove path fallback in resolve_targets_to_stages

In `targets.py:198-206`, change:
```python
            # Treat as artifact path - use absolute path to match graph node format
            norm_path = project.normalize_path(target)
            identity = engine_graph.parse_artifact_identity(str(norm_path))
            producer = engine_graph.get_producer(bipartite_graph, identity)
            if producer:
                result.add(producer)
            else:
                unresolved.append(target)
```
to:
```python
            unresolved.append(target)
```

### Step 2: Remove dead BaseOut branches

In `resolve_output_paths` (around line 275), remove the `isinstance(out, output_type)` branch:
```python
                elif isinstance(out, output_type):
                    # Registry always stores single-file outputs (multi-file are expanded)
                    expanded = outputs.require_expanded(
                        cast("outputs.Metric | outputs.Plot[Any]", out)
                    )
                    rel_path = project.to_relative_path(
                        project.normalize_path(expanded.path), proj_root
                    )
                    resolved.add(rel_path)
```

Similarly remove any `BaseOut` branches in `resolve_plot_infos`.

### Step 3: Run tests, commit
```bash
pytest packages/pivot/tests/cli/ -xvs
git add -A && git commit -m "refactor: remove CLI target path fallback and dead BaseOut branches (I7)"
```

---

## Task 18: Clean up public API exports

**Files:**
- Modify: `packages/pivot/src/pivot/__init__.py`

**Step 1:** Remove old declarative API exports. Change `__init__.py` to only export the compositional API:

Remove from `TYPE_CHECKING` block:
```python
    from pivot.outputs import DirectoryOut as DirectoryOut
    from pivot.outputs import IncrementalOut as IncrementalOut
    from pivot.outputs import Metric as Metric
    from pivot.outputs import Out as Out
    from pivot.outputs import Plot as Plot
```

Remove from `_LAZY_IMPORTS`:
```python
    "DirectoryOut": ("pivot.outputs", "DirectoryOut"),
    "IncrementalOut": ("pivot.outputs", "IncrementalOut"),
    "Metric": ("pivot.outputs", "Metric"),
    "Out": ("pivot.outputs", "Out"),
    "Plot": ("pivot.outputs", "Plot"),
```

Keep: `Pipeline`, `stage`, `metric`, `plot`, `StageParams`, `no_fingerprint`, `loaders`, `stage_def`, `merkle`.

**Step 2: Run tests, commit**
```bash
pytest packages/pivot/tests/ -x --timeout=60
git add -A && git commit -m "chore: remove old declarative API from public exports"
```

---

## Task 19: Document watch mode as known-broken (C2)

**Files:**
- Modify: `packages/pivot/src/pivot/engine/graph.py:270-276` (add docstring note)

**Step 1:** Update `get_watch_paths` docstring:
```python
def get_watch_paths(g: nx.DiGraph[str]) -> list[str]:
    """Return watch paths (STUB — watch mode broken for data artifacts).

    Watch mode for data artifact changes is not functional in this release.
    Code/config file watching still works via inotify on source files.
    TODO: Resolve ArtifactIdentity to filesystem paths via Store.
    """
    _ = g
    return []
```

**Step 2: Commit**
```bash
git add -A && git commit -m "docs: document watch mode as known limitation (C2)"
```

---

## Task 20: Fix output cache backup and restoration (C4)

**Files:**
- Modify: `packages/pivot/src/pivot/executor/worker.py` (backup outputs to CAS after commit)
- Modify: `packages/pivot/src/pivot/engine/engine.py:1598-1605, 1710-1715` (restore from CAS)
- Modify: `packages/pivot/src/pivot/storage/store.py` (add restore method to WorkspaceStore)
- Test: `packages/pivot/tests/storage/test_store.py`, `packages/pivot/tests/execution/`

**Architecture note:** The old code backed up outputs to CAS (`.pivot/cache/files/`) during
execution and restored from CAS during skip detection. The refactor removed both. Two things
must be fixed:

1. **Backup**: After WorkspaceStore.commit() (which just hashes), also save to CAS
2. **Restore**: During skip detection, if workspace files are missing, restore from CAS

The worker already has `cache_dir` available (passed to `execute_stage`). The engine has
`files_cache_dir = cache_dir / "files"` in the skip path.

### Step 1: Add CAS-backed methods to WorkspaceStore

Add a `save_to_cache` and `restore_from_cache` capability. The simplest approach:
add an optional `cache_dir` to WorkspaceStore so it can back up during commit
and restore during skip.

```python
class WorkspaceStore:
    _cache_dir: pathlib.Path | None  # NEW: optional CAS cache directory

    def __init__(
        self,
        project_root: pathlib.Path,
        pipeline_name: str,
        input_bindings: dict[str, str],
        cache_dir: pathlib.Path | None = None,  # NEW
    ) -> None:
        ...
        self._cache_dir = cache_dir
```

Add a `backup_to_cache` method:
```python
    def backup_to_cache(self, ref: types.ArtifactRef, path: pathlib.Path) -> None:
        """Save output to CAS cache after writing to workspace."""
        if self._cache_dir is None:
            return
        cache.save_to_cache(
            path,
            self._cache_dir,
            state_db=None,
            checkout_mode=config.CheckoutMode.COPY,
        )
```

Add a `restore_from_cache` method:
```python
    def restore_from_cache(
        self, ref: types.ArtifactRef, expected_hash: str
    ) -> bool:
        """Restore output from CAS cache to workspace path.

        Returns True if restoration succeeded, False if not found in cache.
        """
        if self._cache_dir is None:
            return False
        cache_path = cache.get_cache_path(self._cache_dir, expected_hash)
        if not cache_path.exists():
            return False
        output_path = self._resolve_output_path(ref)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Use hardlink or copy (not symlink — workspace files should be independent)
        cache.checkout_file(cache_path, output_path, config.CheckoutMode.COPY)
        return True
```

### Step 2: Wire cache_dir into WorkspaceStore creation

In `store_from_spec` (store.py:264-279), pass `cache_dir`:
```python
    if kind == "workspace":
        cache_dir_str = spec.get("cache_dir")
        return WorkspaceStore(
            project_root=pathlib.Path(spec["project_root"]),
            pipeline_name=spec["pipeline_name"],
            input_bindings=spec["input_bindings"],
            cache_dir=pathlib.Path(cache_dir_str) if cache_dir_str else None,
        )
```

### Step 3: Backup outputs after commit in worker

In worker.py `_commit_outputs_with_store`, after `store.commit()`:
```python
def _commit_outputs_with_store(
    outputs_by_ref: dict[types.ArtifactRef, pathlib.Path],
    store: store_mod.Store,
) -> dict[str, HashInfo]:
    output_hashes = {}
    for ref, path in outputs_by_ref.items():
        if not path.exists():
            raise exceptions.OutputMissingError(...)
        store.commit(ref, path)
        # Backup to CAS if store supports it
        if isinstance(store, store_mod.WorkspaceStore):
            store.backup_to_cache(ref, path)
        output_hashes[types.identity_key(ref.identity)] = store.hash_artifact(ref)
    return output_hashes
```

### Step 4: Restore from CAS in engine skip detection

In engine.py, replace `_restore_outputs` (line 1710-1711):
```python
        def _restore_outputs() -> bool:
            if not _outputs_exist():
                # Try to restore from CAS cache
                if not isinstance(store, store_mod.WorkspaceStore):
                    return False
                for out_ref in stage_info["outs"]:
                    identity = out_ref.identity
                    hash_info = lock_data["output_hashes"].get(identity)
                    if hash_info is None:
                        return False
                    if not store.restore_from_cache(out_ref, hash_info["hash"]):
                        return False
            return True
```

Also apply the same pattern to the worker's `_outputs_exist_with_store` (line 316):
```python
def _outputs_exist_with_store(
    store: store_mod.Store,
    outs: list[types.ArtifactRef],
    lock_data: LockData | None = None,
) -> bool:
    for out in outs:
        if not store.exists(out):
            # Try CAS restoration
            if lock_data is None or not isinstance(store, store_mod.WorkspaceStore):
                return False
            identity = out.identity
            hash_info = lock_data["output_hashes"].get(identity)
            if hash_info is None or not store.restore_from_cache(out, hash_info["hash"]):
                return False
    return True
```

### Step 5: Write tests

```python
def test_workspace_store_backup_and_restore(tmp_path: pathlib.Path):
    """Outputs backed up during commit can be restored after deletion."""
    from pivot.storage import store as store_mod, cache
    from pivot import types, loaders

    cache_dir = tmp_path / ".pivot" / "cache" / "files"
    cache_dir.mkdir(parents=True)

    ws = store_mod.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="test",
        input_bindings={},
        cache_dir=cache_dir,
    )
    ref = types.ArtifactRef(
        identity=types.ArtifactIdentity("stage", None),
        format=loaders.Text(),
        python_type=str,
        tag=types.ArtifactTag.DATA,
    )

    # Write output and commit (should backup to CAS)
    out_path = ws.prepare_output(ref)
    out_path.write_text("hello world")
    hash_val = ws.commit(ref, out_path)
    ws.backup_to_cache(ref, out_path)

    # Delete workspace file
    out_path.unlink()
    assert not out_path.exists()
    assert not ws.exists(ref)

    # Restore from CAS
    assert ws.restore_from_cache(ref, hash_val)
    assert out_path.exists()
    assert out_path.read_text() == "hello world"
```

### Step 6: Run tests, commit
```bash
pytest packages/pivot/tests/storage/test_store.py packages/pivot/tests/execution/ -xvs
git add -A && git commit -m "fix: restore output cache backup and CAS restoration (C4)"
```

---

## Task 21: Update fingerprint test README (C9)

**Files:**
- Modify: `packages/pivot/tests/fingerprint/README.md`

**Step 1:** Replace the "Test Files" section at the top (lines 6-14) with:

```markdown
## Test Files

> **Status:** Test files were removed during the compositional API rewrite.
> Tests need to be rebuilt. The change detection matrix below documents the
> expected behaviors that tests should cover.

See `tests/test_compose.py` for loader fingerprint collision tests.
```

**Step 2: Commit**
```bash
git add -A && git commit -m "docs: update fingerprint test README to reflect deleted test files (C9)"
```

---

## Task 22: Fix explain.py identity-as-path bug (C3)

**Files:**
- Modify: `packages/pivot/src/pivot/explain.py:168-195`
- Modify: `packages/pivot/src/pivot/engine/engine.py:1795-1804`
- Modify: `packages/pivot/src/pivot/engine/agent_rpc.py:298-310`
- Test: existing explain/status tests

**The bug has two layers:**
1. The `allow_missing=True` branch (used by `status.py`) wraps identity keys in `pathlib.Path(dep).exists()` — always False, falls through to lock data fallback (works accidentally)
2. The `allow_missing=False` branch (used by engine, agent_rpc) calls `worker.hash_dependencies(deps)` on identity keys — fails because they're not filesystem paths. Explain returns None (caught by exception handler)

**Step 1: Fix callers — pass `allow_missing=True`**

With identity keys, deps can never be hashed by filesystem path. All callers must use `allow_missing=True` so explain uses lock data.

In `engine.py:1795-1804`, add `allow_missing=True`:
```python
            return explain_mod.get_stage_explanation(
                stage_name=stage_name,
                fingerprint=worker_info["fingerprint"],
                deps=[types.identity_key(ref.identity) for ref in stage_info["deps"].values()],
                outs_paths=[types.identity_key(ref.identity) for ref in stage_info["outs"]],
                params_instance=worker_info["params"],
                overrides=overrides,
                state_dir=worker_info["state_dir"],
                force=force,
                allow_missing=True,
            )
```

In `agent_rpc.py:298-310`, add `allow_missing=True`:
```python
                    return explain_mod.get_stage_explanation(
                        stage_name=stage,
                        fingerprint=fingerprint,
                        deps=[
                            types.identity_key(dep.identity) for dep in reg_info["deps"].values()
                        ],
                        outs_paths=[types.identity_key(out.identity) for out in reg_info["outs"]],
                        params_instance=reg_info["params"],
                        overrides=overrides,
                        state_dir=registry_mod.get_stage_state_dir(
                            reg_info, config_io.get_state_dir()
                        ),
                        allow_missing=True,
                    )
```

`status.py:134` already passes `allow_missing` from its caller — no change needed.

**Step 2: Fix the `allow_missing=True` branch in explain.py**

In `explain.py:168-195`, replace the `allow_missing` branch. Change:
```python
    if allow_missing:
        deps_to_hash = list[str]()
        fallback_hashes = dict[ArtifactIdentity, HashInfo]()
        missing_deps = list[str]()

        for dep in deps:
            dep_path = pathlib.Path(dep)
            if dep_path.exists():
                deps_to_hash.append(dep)
            else:
                hash_info = None
                if tracked_files is not None and tracked_trie is not None:
                    hash_info = _find_tracked_hash(dep_path, tracked_files, tracked_trie)
                dep_id = identity_from_key(dep)
                if hash_info is None:
                    hash_info = lock_data["dep_hashes"].get(dep_id)
                if hash_info:
                    fallback_hashes[dep_id] = hash_info
                else:
                    missing_deps.append(dep)

        str_hashes, more_missing, unreadable_deps, _ = worker.hash_dependencies(deps_to_hash)
        dep_hashes = _to_identity_keyed(str_hashes)
        dep_hashes.update(fallback_hashes)
        missing_deps.extend(more_missing)
    else:
        str_hashes, missing_deps, unreadable_deps, _ = worker.hash_dependencies(deps)
        dep_hashes = _to_identity_keyed(str_hashes)
```

to:
```python
    if allow_missing:
        # Deps are identity keys (not filesystem paths). Look up hashes from lock data.
        fallback_hashes = dict[ArtifactIdentity, HashInfo]()
        missing_deps = list[str]()

        for dep in deps:
            dep_id = identity_from_key(dep)
            hash_info = lock_data["dep_hashes"].get(dep_id)
            if hash_info:
                fallback_hashes[dep_id] = hash_info
            else:
                missing_deps.append(dep)

        dep_hashes = fallback_hashes
        unreadable_deps = list[str]()
    else:
        str_hashes, missing_deps, unreadable_deps, _ = worker.hash_dependencies(deps)
        dep_hashes = _to_identity_keyed(str_hashes)
```

The `else` branch is now dead code (all callers pass `allow_missing=True`), but keep it for safety until the `deps` parameter is fully migrated to identity keys everywhere.

**Step 3: Run tests, commit**
```bash
pytest packages/pivot/tests/cli/test_console.py packages/pivot/tests/test_explain.py packages/pivot/tests/ -x --timeout=60
git add -A && git commit -m "fix: explain.py treats identity keys as filesystem paths (C3)"
```

---

## Task 23: Run full quality checks (always last)

**Step 1: Format and lint**
```bash
uv run ruff format . && uv run ruff check . --fix
```

**Step 2: Type check**
```bash
uv run basedpyright
```

**Step 3: Run full test suite**
```bash
uv run pytest packages/pivot/tests packages/pivot-tui/tests -x --timeout=120
```

**Step 4: Fix any failures**

**Step 5: Final commit**
```bash
git add -A && git commit -m "chore: fix lint/type errors from review fixes"
```

---

## Summary: Review Item → Task Mapping

| Review Item | Task | Description |
|-------------|------|-------------|
| C1 | 1 | Delete scratch files |
| C2 | 19 | Document watch mode as broken |
| C3 | 22 | Fix explain.py identity-as-path bug |
| C4 | 20 | Fix output cache backup and restoration |
| C5 | 2 | Fix artifact lock key collision |
| C6 | 3 | Fix CacheStore StateDB path |
| C7 | 6 | Pipeline handle cross-contamination |
| C8 | 15 | Remove pivot.yaml references |
| C9 | 21 | Update fingerprint test README |
| I1 | — | No fix needed: already centralized at StageLock.is_changed_with_lock_data() |
| I2 | 6 | Consolidate SINGLE_OUTPUT_KEY |
| I3 | 7 | Extract shared _format_extension |
| I4 | 8 | Delete worker dead code |
| I5 | 9 | Delete explain.py re-exports |
| I6 | 12 | Collapse graph old types |
| I7 | 17 | Remove CLI path fallback |
| I8 | 13 | Fix stale stage_def docstring |
| I9 | 6 | Validate in build() |
| I10 | 6 | Required/NotRequired unwrap |
| I11 | 11 | Fix no_fingerprint stacking |
| I12 | 4 | Fix dangling symlink checks |
| Public API | 18 | Clean up __init__.py exports |
| Dead BaseOut | 17 | Remove dead branches in targets.py |
| M1 | 6 | Fix error message |
| M2 | 10 | Delete ValidationMode |
| M3 | 13 | Update lock file comments |
| M4 | 13 | Rename OutputChange.path |
| M5 | 5 | Guard _ensure_symlink |
| M6 | 14 | Fix merkle deps iteration |
| M7 | 16 | Remove dvc_compat |

### No Longer Deferred

- **C3** is a 5-minute fix (Task 21) — just remove the dead `.exists()` branch in explain.py.
- **I1** needs no fix — skip logic is already centralized at `StageLock.is_changed_with_lock_data()`, which both worker (directly) and engine (via `skip.check_stage()`) use.
