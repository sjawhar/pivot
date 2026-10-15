# Single-Field TypedDict Output Bug Fix

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix compose.py so single-field TypedDict returns preserve their field key instead of collapsing to `key=None`, which causes the worker to pass the raw dict to writers instead of extracting the value.

**Architecture:** The root cause is in `compose.py:build()` which unconditionally sets `output_key=None` when a stage has exactly one output. The fix changes the condition: only set `key=None` for bare returns (`SINGLE_OUTPUT_KEY`), not for TypedDict fields. The worker's `_resolve_output_values` already handles keyed outputs via dict unpacking, so no worker changes needed.

**Tech Stack:** Python, pytest, pivot internals (compose, worker, types)

---

## Root Cause Analysis

When a stage returns a single-field TypedDict like `{"plot": fig}`:

1. `_analyze_return_type` → `_parse_output_type` sees a TypedDict, iterates fields, creates `_OutputSpec(key="plot", ...)`
2. `compose.py:build()` line 316: `output_key = None if is_single_output else output_spec.key` → collapses `"plot"` to `None`
3. `ArtifactIdentity(producer="stage", key=None)` → worker treats this as a bare return
4. `_resolve_output_values` line 1036: `if len(outs) == 1 and outs[0].identity.key is None` → returns entire `{"plot": fig}` dict as the value
5. Writer receives the dict instead of `fig` → `AttributeError: 'dict' object has no attribute 'savefig'`

**Why bare returns still work after the fix:** For bare `Annotated` returns (not TypedDict), `_analyze_return_type` calls `_parse_output_type(return_hint, SINGLE_OUTPUT_KEY)`. The spec key is `"_single"`, which the fix still maps to `None`.

## Files Changed

- **Modify:** `packages/pivot/src/pivot/compose.py:287-289` (dep reference key) and `:314-316` (output registration key)
- **Test:** `packages/pivot/tests/test_compose.py` (registration tests)
- **Test:** `packages/pivot/tests/execution/test_worker_store.py` (execution integration test)

---

### Task 1: Write failing compose registration test for single-field TypedDict

**Files:**
- Modify: `packages/pivot/tests/test_compose.py`

**Step 1: Add helper TypedDict and stage function at module level**

After the existing `_HelperMultiOutput` / `_helper_multi_output` definitions (around line 185):

```python
class _HelperSingleFieldTypedDict(TypedDict):
    result: pd.DataFrame


@stage
def _helper_single_field_typeddict() -> _HelperSingleFieldTypedDict:
    return {"result": pd.DataFrame()}
```

**Step 2: Add test for single-field TypedDict key preservation**

After `test_pipeline_build_bridge` (around line 431):

```python
def test_pipeline_build_single_field_typeddict_preserves_key(tmp_path: pathlib.Path) -> None:
    """Single-field TypedDict outputs must preserve their field name as the identity key."""
    with Pipeline("test_single_td", root=tmp_path) as pipeline:
        _helper_single_field_typeddict()

    legacy = pipeline.build()
    stage_info = legacy.get("_helper_single_field_typeddict")

    assert len(stage_info["outs"]) == 1
    out = stage_info["outs"][0]
    assert out.identity.key == "result", (
        f"Single-field TypedDict key should be 'result', not {out.identity.key!r}"
    )
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest packages/pivot/tests/test_compose.py::test_pipeline_build_single_field_typeddict_preserves_key -xvs`
Expected: FAIL — `assert out.identity.key == "result"` fails because key is `None`

---

### Task 2: Write failing test for single-field TypedDict dep reference

**Files:**
- Modify: `packages/pivot/tests/test_compose.py`

**Step 1: Add a consumer stage and test**

Add another helper near the one from Task 1:

```python
@stage
def _helper_consume_single_td(data: pd.DataFrame) -> dict:
    return {"rows": len(data)}
```

Add test:

```python
def test_pipeline_build_single_field_typeddict_dep_key(tmp_path: pathlib.Path) -> None:
    """Dep references to single-field TypedDict outputs must use the field key."""
    with Pipeline("test_single_td_dep", root=tmp_path) as pipeline:
        data = _helper_single_field_typeddict()
        _helper_consume_single_td(data)

    legacy = pipeline.build()
    consumer = legacy.get("_helper_consume_single_td")

    dep = consumer["deps"]["data"]
    assert dep.identity.key == "result", (
        f"Dep to single-field TypedDict should use key 'result', not {dep.identity.key!r}"
    )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pivot/tests/test_compose.py::test_pipeline_build_single_field_typeddict_dep_key -xvs`
Expected: FAIL — dep key is `None` instead of `"result"`

---

### Task 3: Fix compose.py — stop collapsing single-field TypedDict keys

**Files:**
- Modify: `packages/pivot/src/pivot/compose.py:287-289,314-316`

**Step 1: Fix output registration (lines 314-316)**

Replace:

```python
            is_single_output = len(node.output_specs) == 1
            for output_spec in node.output_specs:
                output_key = None if is_single_output else output_spec.key
```

With:

```python
            for output_spec in node.output_specs:
                output_key = None if output_spec.key == SINGLE_OUTPUT_KEY else output_spec.key
```

**Step 2: Fix dep reference (lines 287-289)**

Replace:

```python
                if len(source.output_specs) == 1:
                    output_spec = source.output_specs[0]
                    output_key = None
```

With:

```python
                if len(source.output_specs) == 1:
                    output_spec = source.output_specs[0]
                    output_key = None if output_spec.key == SINGLE_OUTPUT_KEY else output_spec.key
```

**Step 3: Run the two failing tests to verify they pass**

Run: `uv run pytest packages/pivot/tests/test_compose.py::test_pipeline_build_single_field_typeddict_preserves_key packages/pivot/tests/test_compose.py::test_pipeline_build_single_field_typeddict_dep_key -xvs`
Expected: PASS

---

### Task 4: Write integration test for single-field TypedDict execution

**Files:**
- Modify: `packages/pivot/tests/execution/test_worker_store.py`

**Step 1: Add helper stage at module level**

```python
from typing import TypedDict

class _SingleFieldOutput(TypedDict):
    upper: str


def _stage_single_field_typeddict(data: str) -> _SingleFieldOutput:
    return {"upper": data.upper()}
```

**Step 2: Add integration test**

```python
def test_execute_stage_single_field_typeddict(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    """Single-field TypedDict return extracts the field value, not the whole dict."""
    (tmp_path / "input.txt").write_text("hello")

    deps = {
        "data": _make_artifact_ref(
            "input",
            None,
            tag=types.ArtifactTag.DATA,
            loader=loaders.Text(),
            python_type=str,
        )
    }
    outs = [
        _make_artifact_ref(
            "stage",
            "upper",
            tag=types.ArtifactTag.DATA,
            loader=loaders.Text(),
            python_type=str,
        )
    ]
    store_spec = cast(
        "store_mod.StoreSpec",
        cast(
            "object",
            {
                "kind": "workspace",
                "cache_dir": str(worker_env),
                "project_root": str(tmp_path),
                "pipeline_name": "pipe",
                "input_bindings": {"input": "input.txt"},
            },
        ),
    )

    stage_info = _make_worker_stage_info(
        _stage_single_field_typeddict,
        tmp_path,
        deps=deps,
        outs=outs,
        store_spec=store_spec,
    )

    result = worker.execute_stage("stage", stage_info, worker_env, output_queue)

    output_path = tmp_path / "data" / "pipe" / "stage" / "upper.txt"
    assert output_path.exists(), "Keyed output should be in subdirectory"
    assert output_path.read_text() == "HELLO"
    assert result["status"] == types.StageStatus.RAN
```

**Step 3: Run test to verify it passes**

Run: `uv run pytest packages/pivot/tests/execution/test_worker_store.py::test_execute_stage_single_field_typeddict -xvs`
Expected: PASS — the output key `"upper"` causes worker to unpack `result["upper"]` correctly

---

### Task 5: Run full test suite and quality checks

**Step 1: Run all tests**

Run: `uv run pytest packages/pivot/tests packages/pivot-tui/tests -x --timeout=60`

**Step 2: Run quality checks**

Run: `uv run ruff format . && uv run ruff check . && uv run basedpyright`

**Step 3: Fix any failures**

All existing tests for bare returns (e.g., `test_pipeline_build_bridge`) should still pass because `SINGLE_OUTPUT_KEY` maps to `key=None`.

---

### Task 6: Commit

```
fix(compose): preserve key for single-field TypedDict outputs

Single-field TypedDict returns had their field key collapsed to None,
causing the worker to pass the entire dict to writers instead of
extracting the value. Use SINGLE_OUTPUT_KEY to distinguish bare returns
(key=None) from TypedDict fields (key preserved).
```
