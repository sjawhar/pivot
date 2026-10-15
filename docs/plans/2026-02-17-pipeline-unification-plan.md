# Pipeline Unification & Codebase Cleanup — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate `pipeline.Pipeline`, make `compose.Pipeline` the sole pipeline
class with a `PipelineLike` Protocol for the engine, and delete all vestigial
declarative-API code.

**Architecture:** Define a `PipelineLike` Protocol matching what the engine needs.
Make `compose.Pipeline` own a `StageRegistry` and implement this protocol directly.
Delete `pipeline.Pipeline`, the `build()` bridge, external dep resolution, and dead
modules (`matrix.py`, `dvc_import.py`). Delete aggressively as each phase completes.

**Tech Stack:** Python 3.13, basedpyright, ruff, pytest. No new dependencies.

**Design Doc:** `docs/plans/2026-02-17-pipeline-unification-cleanup.md`

---

## Phase 1: Dead Code Removal

Clean out modules with zero production imports before touching live code.

### Task 1.1: Delete `matrix.py`

**Files:**
- Delete: `packages/pivot/src/pivot/matrix.py` (143 lines)
- Delete: `packages/pivot/tests/test_matrix.py` (371 lines)

**Step 1: Verify no production imports**

Run: `grep -r "from pivot import matrix\|from pivot.matrix\|import matrix" packages/pivot/src/ --include="*.py"`
Expected: No matches (only test imports exist)

**Step 2: Delete files**

Delete `packages/pivot/src/pivot/matrix.py` and `packages/pivot/tests/test_matrix.py`.

**Step 3: Run tests to confirm nothing breaks**

Run: `uv run pytest packages/pivot/tests packages/pivot-tui/tests -x -q`
Expected: All tests pass (no production code depends on matrix.py)

**Step 4: Run quality checks**

Run: `uv run ruff check packages/pivot/src/pivot/ && uv run basedpyright packages/pivot/src/pivot/`
Expected: No new errors

**Step 5: Commit**

`jj describe -m "chore: delete vestigial matrix.py (zero production imports)"`

### Task 1.2: Delete `dvc_import.py`

**Files:**
- Delete: `packages/pivot/src/pivot/dvc_import.py` (880 lines)
- Delete: `packages/pivot/tests/test_dvc_import.py` (1150 lines)

**Step 1: Verify no production imports**

Run: `grep -r "from pivot import dvc_import\|from pivot.dvc_import\|import dvc_import" packages/pivot/src/ --include="*.py"`
Expected: No matches

**Step 2: Delete files**

Delete `packages/pivot/src/pivot/dvc_import.py` and `packages/pivot/tests/test_dvc_import.py`.

**Step 3: Run tests**

Run: `uv run pytest packages/pivot/tests packages/pivot-tui/tests -x -q`
Expected: All tests pass

**Step 4: Run quality checks**

Run: `uv run ruff check packages/pivot/src/pivot/ && uv run basedpyright packages/pivot/src/pivot/`
Expected: No new errors

**Step 5: Commit**

`jj describe -m "chore: delete vestigial dvc_import.py (zero production imports)"`

### Task 1.3: Delete `pipeline/yaml.py` stub, relocate `PipelineConfigError`

**Files:**
- Delete: `packages/pivot/src/pivot/pipeline/yaml.py` (18 lines)
- Modify: `packages/pivot/src/pivot/exceptions.py` — add `PipelineConfigError`
- Modify: `packages/pivot/src/pivot/pipeline/pipeline.py` — update import

**Step 1: Add `PipelineConfigError` to `exceptions.py`**

Add to `packages/pivot/src/pivot/exceptions.py`:
```python
class PipelineConfigError(PivotError):
    """Raised when pipeline configuration is invalid."""
```

**Step 2: Update import in `pipeline/pipeline.py`**

Change `from pivot.pipeline.yaml import PipelineConfigError` to
`from pivot.exceptions import PipelineConfigError`.

**Step 3: Delete `pipeline/yaml.py`**

**Step 4: Verify no other imports of `pipeline.yaml`**

Run: `grep -r "from pivot.pipeline.yaml\|from pivot.pipeline import yaml" packages/pivot/ --include="*.py"`
Expected: No matches (the only importer was `pipeline/pipeline.py`, already updated)

**Step 5: Run tests and quality checks**

Run: `uv run pytest packages/pivot/tests -x -q && uv run ruff check packages/pivot/src/pivot/ && uv run basedpyright packages/pivot/src/pivot/`
Expected: All pass

**Step 6: Commit**

`jj describe -m "refactor: relocate PipelineConfigError to exceptions.py, delete yaml.py stub"`

### Task 1.4: Delete external dep resolution

**Files:**
- Modify: `packages/pivot/src/pivot/pipeline/pipeline.py` — remove functions and method
- Modify: `packages/pivot/src/pivot/discovery.py` — remove traversal helpers
- Modify: `packages/pivot/src/pivot/engine/engine.py` — remove resolve call

**Step 1: Delete module-level helper functions from `pipeline/pipeline.py`**

Delete these functions (they are only called by `resolve_external_dependencies`):
- `_find_producer_via_traversal()` (lines 83-102)
- `_find_producer_via_index()` (lines 105-135)
- `_find_producer_via_scan()` (lines 138-154)
- `_find_producer_in_pipeline()` (lines 58-70)
- `_load_pipeline()` (lines 73-80)
- `_find_pipeline_dir_for_stage()` — find and delete (helper for `_write_output_index`)

**Step 2: Delete methods from `Pipeline` class**

Delete from `Pipeline`:
- `resolve_external_dependencies()` method (lines 325-407)
- `_write_output_index()` method (lines 409-445+)
- Remove `_external_deps_resolved` field from `__init__` and `_reset_resolution_cache`

**Step 3: Simplify `build_dag()`**

Change from:
```python
def build_dag(self) -> DiGraph[str]:
    self.resolve_external_dependencies()
    dag = self._registry.build_dag()
    self._write_output_index()
    return dag
```
To:
```python
def build_dag(self) -> DiGraph[str]:
    return self._registry.build_dag()
```

**Step 4: Simplify `_reset_resolution_cache` and `invalidate_dag_cache`**

Remove `_external_deps_resolved` references. `_reset_resolution_cache` can be
inlined or simplified to just `self._registry.invalidate_dag_cache()`.

**Step 5: Delete traversal helpers from `discovery.py`**

Delete from `packages/pivot/src/pivot/discovery.py`:
- `find_parent_pipeline_paths()` (lines 196-225)
- `find_pipeline_paths_for_dependency()` (lines 228-265)

Verify no other callers:
Run: `grep -r "find_parent_pipeline_paths\|find_pipeline_paths_for_dependency" packages/pivot/ --include="*.py"`

**Step 6: Remove engine's resolve call**

In `packages/pivot/src/pivot/engine/engine.py`, line 2091, delete:
```python
self._require_pipeline().resolve_external_dependencies()
```

Also remove `_external_deps_resolved` hack in `discovery.py:188`:
```python
combined._external_deps_resolved = True  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
```

**Step 7: Run tests**

Run: `uv run pytest packages/pivot/tests -x -q`
Expected: Tests pass. Some tests that explicitly tested external dep resolution may
fail — those tests should be deleted since the feature is removed.

**Step 8: Run quality checks**

Run: `uv run ruff check packages/pivot/src/pivot/ && uv run basedpyright packages/pivot/src/pivot/`
Expected: No new errors (some unused imports may need cleanup)

**Step 9: Commit**

`jj describe -m "refactor: delete external dep resolution (vestigial in compose API)"`

---

## Phase 2: PipelineLike Protocol + Engine Migration

### Task 2.1: Define `PipelineLike` Protocol

**Files:**
- Modify: `packages/pivot/src/pivot/registry.py` — add Protocol

**Step 1: Add Protocol definition**

Add to `packages/pivot/src/pivot/registry.py` (where `RegistryStageInfo` already lives):

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class PipelineLike(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def root(self) -> pathlib.Path: ...

    @property
    def state_dir(self) -> pathlib.Path: ...

    @property
    def input_bindings(self) -> dict[str, str]: ...

    def list_stages(self) -> list[str]: ...

    def get_stage(self, name: str) -> RegistryStageInfo: ...

    def ensure_fingerprint(self, name: str) -> dict[str, str]: ...

    def build_dag(self) -> DiGraph[str]: ...

    def invalidate_dag_cache(self) -> None: ...

    def snapshot(self) -> dict[str, RegistryStageInfo]: ...

    def restore(self, snapshot: dict[str, RegistryStageInfo]) -> None: ...

    def include(self, other: PipelineLike) -> None: ...
```

**Step 2: Verify `pipeline.Pipeline` satisfies the Protocol**

`pipeline.Pipeline` already has all these methods (some with slightly different names).
`get_stage()` already exists as an alias for `get()`. Ensure the signatures match.

**Step 3: Run quality checks**

Run: `uv run basedpyright packages/pivot/src/pivot/registry.py`
Expected: Clean

**Step 4: Commit**

`jj describe -m "feat: define PipelineLike Protocol for engine contract"`

### Task 2.2: Replace engine's private `._registry` access with public methods

**Files:**
- Modify: `packages/pivot/src/pivot/engine/engine.py`
- Modify: `packages/pivot/src/pivot/executor/core.py`

**Step 1: Fix `prepare_worker_info` in `executor/core.py`**

Change `executor/core.py:309-350`. The function currently takes `stage_registry: registry.StageRegistry`
and calls `stage_registry.ensure_fingerprint(stage_info["name"])`.

Change the parameter from `stage_registry` to `pipeline: registry.PipelineLike` and call
`pipeline.ensure_fingerprint(stage_info["name"])` instead.

**Step 2: Fix caller in `engine.py` (line 1435-1437)**

Change from:
```python
worker_info = executor_core.prepare_worker_info(
    stage_info,
    pipeline._registry,  # pyright: ignore[reportPrivateUsage]
    ...
)
```
To:
```python
worker_info = executor_core.prepare_worker_info(
    stage_info,
    pipeline,
    ...
)
```

**Step 3: Fix `_reload_registry` in `engine.py` (line 2264)**

Change from `old_registry = old_pipeline._registry` to using `old_pipeline` directly.
The registry is only used for `ensure_fingerprint()` comparison in `_emit_reload_event`.

**Step 4: Fix `_emit_reload_event` in `engine.py` (lines 2307-2368)**

Change parameter from `old_registry: registry.StageRegistry | None` to
`old_pipeline: PipelineLike | None`. Call `old_pipeline.ensure_fingerprint()`
and `new_pipeline.ensure_fingerprint()` instead of `old_registry.ensure_fingerprint()`
and `new_registry.ensure_fingerprint()`.

**Step 5: Fix coordinator skip check in `engine.py` (line 1525)**

Change from `stage_registry = pipeline._registry` to
`pipeline.ensure_fingerprint(stage_name)` directly.

**Step 6: Verify no remaining `._registry` access**

Run: `grep -r "_registry" packages/pivot/src/pivot/engine/ --include="*.py"`
Expected: No `pipeline._registry` references remain

**Step 7: Run tests and quality checks**

Run: `uv run pytest packages/pivot/tests -x -q && uv run basedpyright packages/pivot/src/pivot/`
Expected: All pass

**Step 8: Commit**

`jj describe -m "refactor: replace private _registry access with PipelineLike methods"`

### Task 2.3: Switch discovery to Protocol validation

**Files:**
- Modify: `packages/pivot/src/pivot/discovery.py`

**Step 1: Change `_load_pipeline_from_module`**

Replace `isinstance(pipeline_obj, Pipeline)` check with Protocol validation:

```python
from pivot.registry import PipelineLike

def _validate_pipeline_object(obj: object, path: pathlib.Path) -> PipelineLike:
    """Validate that an object satisfies PipelineLike."""
    if not isinstance(obj, PipelineLike):
        # Give specific feedback about what's missing
        required = ["name", "list_stages", "get_stage", "build_dag",
                     "ensure_fingerprint", "snapshot", "restore"]
        missing = [attr for attr in required if not hasattr(obj, attr)]
        if missing:
            raise DiscoveryError(
                f"{path} defines 'pipeline' but it's missing required methods: {missing}"
            )
        raise DiscoveryError(
            f"{path} defines 'pipeline' but it doesn't satisfy the Pipeline interface "
            f"(got {type(obj).__name__})"
        )
    return obj
```

**Step 2: Update return types throughout discovery.py**

Change `Pipeline` type hints to `PipelineLike` in:
- `discover_pipeline()` return type
- `_load_pipeline_from_module()` return type
- `_discover_all_pipelines()` return type and internal usage
- `load_pipeline_from_path()` return type

**Step 3: Update all importers of Pipeline from discovery**

Search for `from pivot.pipeline.pipeline import Pipeline` across CLI, engine, etc.
Replace with `from pivot.registry import PipelineLike` where the type is used as
a type hint (not as a constructor).

**Step 4: Run tests and quality checks**

Run: `uv run pytest packages/pivot/tests -x -q && uv run basedpyright packages/pivot/src/pivot/`
Expected: All pass. Note: `_discover_all_pipelines` still creates a `pipeline.Pipeline`
for the combined pipeline — this changes in Phase 3 Task 3.4 when we delete
`pipeline.Pipeline` and switch discovery to use `compose.Pipeline`.

**Step 5: Commit**

`jj describe -m "refactor: switch discovery and consumers to PipelineLike Protocol"`

---

## Phase 3: Compose Pipeline Unification

### Task 3.1: Add `StageRegistry` to `compose.Pipeline`

**Files:**
- Modify: `packages/pivot/src/pivot/compose.py`
- Test: `packages/pivot/tests/test_compose.py`

**Key design decision — Lifecycle:**
compose.Pipeline starts in "materialized" state (empty registry, Protocol methods work).
Entering the context manager transitions to "construction" state. Exiting re-materializes.
This means `Pipeline("all")` + `include()` works without `with`, and `@stage` only
works inside `with`.

**Step 1: Write failing test**

Add a test that verifies stages are accessible after `__exit__`. Use an existing
`@stage`-decorated function from the test fixtures (check `test_compose.py` for
existing patterns — likely `identity` or similar). The test should:
- Create a Pipeline, register one stage via `@stage` inside `with`
- After `with` exits, call `p.list_stages()` and assert the stage name
- Call `p.get_stage(name)` and assert it returns `RegistryStageInfo` with the right `func`

Run: `uv run pytest packages/pivot/tests/test_compose.py::test_pipeline_has_registry_after_exit -v`
Expected: FAIL (list_stages doesn't exist yet)

**Step 2: Add StageRegistry and implement lifecycle**

In `compose.Pipeline.__init__`, add:
```python
self._registry = registry.StageRegistry()
self._materialized = True  # Start materialized (empty is fine)
```

In `__enter__`, transition to construction:
```python
self._materialized = False
```

In `__exit__`, after `_validate()`, materialize all stages:
```python
if exc_type is None:
    self._validate()
    self._materialize_stages()
    self._materialized = True
```

Add `_materialize_stages()` method: move the body of `build()` (upstream closure
processing + stage emission loop) but write to `self._registry` instead of creating
a new `pipeline.Pipeline`. Handle `_merge_foreign_input_bindings` inline.

Add public methods with materialization guards:
```python
def list_stages(self) -> list[str]:
    if not self._materialized:
        raise RuntimeError("Pipeline not yet materialized (still inside 'with' block)")
    return self._registry.list_stages()

def get_stage(self, name: str) -> registry.RegistryStageInfo:
    if not self._materialized:
        raise RuntimeError("Pipeline not yet materialized (still inside 'with' block)")
    return self._registry.get(name)
```

**Step 3: Run test**

Expected: PASS

**Step 4: Write test for non-context-manager usage**

Verify `Pipeline("all")` + `include()` works without entering context manager.
This is the `_discover_all_pipelines` use case.

**Step 5: Commit**

`jj describe -m "feat: compose.Pipeline materializes StageRegistry on __exit__"`

### Task 3.2: Implement remaining PipelineLike methods on compose.Pipeline

**Files:**
- Modify: `packages/pivot/src/pivot/compose.py`
- Test: `packages/pivot/tests/test_compose.py`

**Step 1: Add remaining protocol methods**

Add properties (`root`, `state_dir`, `input_bindings`) and methods (`ensure_fingerprint`,
`build_dag`, `invalidate_dag_cache`, `snapshot`, `restore`, `include`).

Methods that read from the registry (`ensure_fingerprint`, `build_dag`, `snapshot`,
`list_stages`, `get_stage`) need the materialization guard. Methods that modify
the registry (`include`, `restore`, `invalidate_dag_cache`) work in both states.

For `include()`: deep-copy stages from the other pipeline, add to registry. Merge
input bindings — if the same input name has different paths, raise a
`PipelineConfigError` (match existing behavior from `pipeline.Pipeline.include`).
Store merged bindings in `self._inputs` as `_InputNode` entries, or add a separate
`_extra_input_bindings: dict[str, str]` to avoid creating `_InputNode` objects
outside the normal `input()` flow.

**Step 2: Write test for Protocol satisfaction**

```python
def test_pipeline_satisfies_protocol():
    p = Pipeline("test", root=tmp_path)
    with p:
        result = identity(p.input("data", t=str))
    assert isinstance(p, PipelineLike)
```

**Step 3: Run tests**

Run: `uv run pytest packages/pivot/tests/test_compose.py -x -q`
Expected: All pass

**Step 4: Commit**

`jj describe -m "feat: compose.Pipeline implements full PipelineLike Protocol"`

### Task 3.3: Fix `__pivot_no_fingerprint__` function mutation

**Files:**
- Modify: `packages/pivot/src/pivot/compose.py` (`_emit_stage_info`)
- Modify: `packages/pivot/src/pivot/registry.py` (`_compute_fingerprint`)

**Step 1: Add `no_fingerprint` field to `RegistryStageInfo`**

In `registry.py`, add to the TypedDict:
```python
no_fingerprint: bool
```

**Step 2: Set `no_fingerprint` in `_emit_stage_info` instead of mutating func**

In `compose.py`, replace:
```python
if getattr(node.func, "__pivot_no_fingerprint__", False):
    func.__pivot_no_fingerprint__ = True
```
With setting `no_fingerprint=True` on the `RegistryStageInfo` dict.

**Step 3: Update `_compute_fingerprint` to check `info["no_fingerprint"]`**

In `registry.py`, change fingerprint check from:
```python
no_fp = getattr(info["func"], "__pivot_no_fingerprint__", False)
if not no_fp:
    unwrapped = inspect.unwrap(info["func"])
    no_fp = getattr(unwrapped, "__pivot_no_fingerprint__", False)
```
To:
```python
no_fp = info["no_fingerprint"]
```

The `@no_fingerprint()` decorator (in `decorators.py`) still sets the attribute on
the function object — that's correct and unchanged. The fix is:
1. `@no_fingerprint()` sets `func.__pivot_no_fingerprint__` at decoration time (no change)
2. `_emit_stage_info()` reads the attribute and sets `no_fingerprint=True` on `RegistryStageInfo`
3. `_compute_fingerprint()` reads ONLY from `RegistryStageInfo`, not from the function

The bug was that `build()`/`_emit_stage_info()` also mutated `original_func` (leaking
across pipelines). Remove that mutation — just read the attribute, don't propagate it.

**Step 4: Run tests**

Run: `uv run pytest packages/pivot/tests -x -q`
Expected: All pass

**Step 5: Commit**

`jj describe -m "fix: store no_fingerprint flag on RegistryStageInfo instead of mutating function"`

### Task 3.4: Delete `build()` and `pipeline.Pipeline`

**Files:**
- Modify: `packages/pivot/src/pivot/compose.py` — delete `build()` and `_merge_foreign_input_bindings`
- Delete: `packages/pivot/src/pivot/pipeline/pipeline.py`
- Delete: `packages/pivot/src/pivot/pipeline/__init__.py`
- Modify: `packages/pivot/src/pivot/discovery.py` — update `_discover_all_pipelines`
- Modify: All files importing `from pivot.pipeline.pipeline import Pipeline`

**Step 1: Delete `build()` method from `compose.Pipeline`**

Remove the `build()` method and `_merge_foreign_input_bindings()` static method.
The `_built` flag is replaced by the `_materialized` flag from Task 3.1.

**User migration:** Any existing pipeline.py files that end with
`pipeline = p.build()` must change to `pipeline = p` (the Pipeline is now usable
directly after the `with` block exits). Check example pipelines and test fixtures.

**Step 2: Update discovery to use compose.Pipeline**

In `_discover_all_pipelines`, change:
```python
combined = Pipeline("all", root=root)
```
To:
```python
from pivot.compose import Pipeline
combined = Pipeline("all", root=root)
```
And use `combined.include()` instead of the legacy include.

Remove the `_external_deps_resolved` hack (line 188).

**Step 3: Update all imports**

Find and replace all `from pivot.pipeline.pipeline import Pipeline` with
`from pivot.compose import Pipeline` (or use `PipelineLike` where only the
type is needed).

Key files:
- `packages/pivot/src/pivot/discovery.py`
- `packages/pivot/src/pivot/engine/engine.py`
- `packages/pivot/src/pivot/executor/core.py`
- `packages/pivot/src/pivot/cli/targets.py`
- `packages/pivot/src/pivot/cli/decorators.py`
- `packages/pivot/src/pivot/cli/completion.py`
- `packages/pivot/src/pivot/cli/_run_common.py`
- `packages/pivot/src/pivot/cli/helpers.py`

**Step 4: Delete `pipeline/pipeline.py` and `pipeline/__init__.py`**

Verify no remaining imports:
Run: `grep -r "from pivot.pipeline" packages/pivot/src/ --include="*.py"`
Expected: No matches

Delete the `packages/pivot/src/pivot/pipeline/` directory entirely.

**Step 5: Run full test suite**

Run: `uv run pytest packages/pivot/tests packages/pivot-tui/tests -x -q`
Expected: Test failures from tests importing `pipeline.Pipeline`. Before deleting,
grep for affected tests:
`grep -r "from pivot.pipeline" packages/pivot/tests/ --include="*.py" -l`
Fix each to use `compose.Pipeline` or `PipelineLike` as appropriate.

**Step 6: Run quality checks**

Run: `uv run ruff format . && uv run ruff check . && uv run basedpyright`
Expected: Clean

**Step 7: Commit**

`jj describe -m "refactor: delete pipeline.Pipeline, compose.Pipeline is sole implementation"`

### Task 3.5: Add unresolved dependency error

**Files:**
- Modify: `packages/pivot/src/pivot/engine/graph.py` or `packages/pivot/src/pivot/registry.py`
- Test: `packages/pivot/tests/test_compose.py`

**Step 1: Write failing test**

Create a test where a stage has a dependency with an identity that doesn't match any
registered output and isn't declared as an input. This requires either:
- Manually constructing a `RegistryStageInfo` with a bogus dep identity and adding
  it to the registry, OR
- Finding the existing graph-building code to understand how unresolved deps are
  currently handled (they may already error — check `engine/graph.py:build_graph`)

The test should verify the error message mentions the dep identity and suggests
`pipeline.input()`. Match against the specific error text.

**Step 2: Add error message to DAG build**

In the graph building code (`engine/graph.py:build_graph` or `registry.build_dag`),
when a dependency's identity has no matching producer, raise a clear error:

```python
raise PivotError(
    f"Stage '{consumer}' depends on '{dep_key}' which has no producer. "
    f"Use pipeline.input(\"{dep_key}\") to declare external data sources."
)
```

**Step 3: Run tests**

Expected: PASS

**Step 4: Commit**

`jj describe -m "feat: clear error message for unresolved dependencies"`

---

## Phase 4: Final Verification

### Task 4.1: End-to-end verification

**Step 1: Run full test suite**

Run: `uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto`
Expected: All pass

**Step 2: Run all quality checks**

Run: `uv run ruff format . && uv run ruff check . && uv run basedpyright`
Expected: Clean

**Step 3: Verify no references to deleted code remain**

```bash
grep -r "from pivot.pipeline" packages/ --include="*.py"
grep -r "from pivot import matrix" packages/ --include="*.py"
grep -r "from pivot import dvc_import" packages/ --include="*.py"
grep -r "resolve_external_dependencies" packages/ --include="*.py"
grep -r "_external_deps_resolved" packages/ --include="*.py"
grep -r "pipeline._registry" packages/ --include="*.py"
```
Expected: No matches for any of these

**Step 4: Commit any final fixes**

`jj describe -m "chore: final cleanup after pipeline unification"`
