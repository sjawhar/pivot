# PlaceholderDep Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `PlaceholderDep` — a dependency marker that has no default path and must be overridden at registration time.

**Architecture:** `PlaceholderDep` is a dataclass like `Dep` but without a `path` field. During `get_dep_specs_from_signature()`, we pass overrides so placeholders resolve immediately to real paths. Validation happens early in `register()` before extraction proceeds.

**Tech Stack:** Python dataclasses, typing (Generic, Annotated), pytest

---

### Task 1: Add PlaceholderDep class to outputs.py

**Files:**
- Modify: `src/pivot/outputs.py:53` (after Dep class)
- Test: `tests/test_dep_injection.py`

**Step 1: Write the failing test**

Add at end of `tests/test_dep_injection.py`:

```python
# ==============================================================================
# Test: PlaceholderDep (dependencies that must be overridden)
# ==============================================================================


def test_placeholder_dep_has_no_path() -> None:
    """PlaceholderDep should have loader but no path attribute."""
    placeholder = outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())

    assert hasattr(placeholder, "loader")
    assert isinstance(placeholder.loader, loaders.CSV)
    assert not hasattr(placeholder, "path")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dep_injection.py::test_placeholder_dep_has_no_path -v`
Expected: FAIL with `AttributeError: module 'pivot.outputs' has no attribute 'PlaceholderDep'`

**Step 3: Write minimal implementation**

In `src/pivot/outputs.py`, add after the `Dep` class (after line 53):

```python
@dataclasses.dataclass(frozen=True)
class PlaceholderDep(Generic[T]):  # noqa: UP046 - basedpyright doesn't support PEP 695 syntax yet
    """Dependency marker with no default path — must be overridden at registration.

    Use when a stage needs a dependency that has no sensible default.
    Registration fails if dep_path_overrides doesn't include this dependency.

        def compare(
            baseline: Annotated[DataFrame, PlaceholderDep(CSV())],
            experiment: Annotated[DataFrame, PlaceholderDep(CSV())],
        ) -> CompareOutputs:
            ...

        REGISTRY.register(
            compare,
            dep_path_overrides={
                "baseline": "model_a/results.csv",
                "experiment": "model_b/results.csv",
            },
        )
    """

    loader: loaders_module.Loader[T]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dep_injection.py::test_placeholder_dep_has_no_path -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(outputs): add PlaceholderDep class"
```

---

### Task 2: Add helper to identify placeholder deps in a function signature

**Files:**
- Modify: `src/pivot/stage_def.py`
- Test: `tests/test_dep_injection.py`

**Step 1: Write the failing test**

Add to `tests/test_dep_injection.py`:

```python
def test_get_placeholder_dep_names_identifies_placeholders() -> None:
    """Should identify which parameters use PlaceholderDep."""

    def compare(
        baseline: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
        experiment: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
        config: Annotated[dict[str, int], outputs.Dep("config.json", loaders.JSON[dict[str, int]]())],
    ) -> _MultiDepOutputs:
        return {"combined": {"count": len(baseline) + len(experiment)}}

    placeholder_names = stage_def.get_placeholder_dep_names(compare)

    assert placeholder_names == {"baseline", "experiment"}


def test_get_placeholder_dep_names_returns_empty_for_no_placeholders() -> None:
    """Should return empty set when no PlaceholderDep annotations."""

    def process(
        data: Annotated[pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV[pandas.DataFrame]())],
    ) -> _ProcessOutputs:
        return {"result": {"count": len(data)}}

    placeholder_names = stage_def.get_placeholder_dep_names(process)

    assert placeholder_names == set()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dep_injection.py::test_get_placeholder_dep_names_identifies_placeholders tests/test_dep_injection.py::test_get_placeholder_dep_names_returns_empty_for_no_placeholders -v`
Expected: FAIL with `AttributeError: module 'pivot.stage_def' has no attribute 'get_placeholder_dep_names'`

**Step 3: Write minimal implementation**

In `src/pivot/stage_def.py`, add after `get_dep_specs_from_signature` function (around line 434):

```python
def get_placeholder_dep_names(func: Callable[..., Any]) -> set[str]:
    """Get parameter names that use PlaceholderDep annotations.

    Scans function parameters for Annotated hints containing PlaceholderDep.
    Used to validate that all placeholders have overrides before registration.

    Returns:
        Set of parameter names that have PlaceholderDep annotations.
    """
    import inspect as inspect_module

    hints = _get_type_hints_safe(func, func.__name__, include_extras=True)
    if hints is None:
        return set()

    sig = inspect_module.signature(func)
    placeholder_names = set[str]()

    for param_name in sig.parameters:
        if param_name not in hints:
            continue

        param_type = _unwrap_type_alias(hints[param_name])

        if get_origin(param_type) is not Annotated:
            continue

        args = get_args(param_type)
        if len(args) < 2:
            continue

        for metadata in args[1:]:
            if isinstance(metadata, outputs.PlaceholderDep):
                placeholder_names.add(param_name)
                break

    return placeholder_names
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dep_injection.py::test_get_placeholder_dep_names_identifies_placeholders tests/test_dep_injection.py::test_get_placeholder_dep_names_returns_empty_for_no_placeholders -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(stage_def): add get_placeholder_dep_names helper"
```

---

### Task 3: Update get_dep_specs_from_signature to accept overrides and handle PlaceholderDep

**Files:**
- Modify: `src/pivot/stage_def.py:364-433` (get_dep_specs_from_signature)
- Test: `tests/test_dep_injection.py`

**Step 1: Write the failing tests**

Add to `tests/test_dep_injection.py`:

```python
def test_get_dep_specs_with_placeholder_and_overrides() -> None:
    """Should resolve PlaceholderDep using provided overrides."""

    def compare(
        baseline: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
        experiment: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
    ) -> _MultiDepOutputs:
        return {"combined": {"count": 0}}

    overrides = {
        "baseline": "model_a/results.csv",
        "experiment": "model_b/results.csv",
    }
    specs = stage_def.get_dep_specs_from_signature(compare, overrides)

    assert specs["baseline"].path == "model_a/results.csv"
    assert specs["experiment"].path == "model_b/results.csv"
    assert isinstance(specs["baseline"].loader, loaders.CSV)


def test_get_dep_specs_mixed_placeholder_and_regular() -> None:
    """Should handle mix of PlaceholderDep and regular Dep."""

    def compare(
        baseline: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
        config: Annotated[dict[str, int], outputs.Dep("config.json", loaders.JSON[dict[str, int]]())],
    ) -> _MultiDepOutputs:
        return {"combined": {"count": 0}}

    overrides = {"baseline": "model_a/results.csv"}
    specs = stage_def.get_dep_specs_from_signature(compare, overrides)

    assert specs["baseline"].path == "model_a/results.csv"
    assert specs["config"].path == "config.json"


def test_get_dep_specs_placeholder_without_override_raises() -> None:
    """Should raise when PlaceholderDep has no override."""

    def compare(
        baseline: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
    ) -> _MultiDepOutputs:
        return {"combined": {"count": 0}}

    with pytest.raises(ValueError, match="PlaceholderDep .* requires override"):
        stage_def.get_dep_specs_from_signature(compare, {})


def test_get_dep_specs_placeholder_none_overrides_raises() -> None:
    """Should raise when PlaceholderDep exists but overrides is None."""

    def compare(
        baseline: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
    ) -> _MultiDepOutputs:
        return {"combined": {"count": 0}}

    with pytest.raises(ValueError, match="PlaceholderDep .* requires override"):
        stage_def.get_dep_specs_from_signature(compare, None)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dep_injection.py::test_get_dep_specs_with_placeholder_and_overrides tests/test_dep_injection.py::test_get_dep_specs_mixed_placeholder_and_regular tests/test_dep_injection.py::test_get_dep_specs_placeholder_without_override_raises tests/test_dep_injection.py::test_get_dep_specs_placeholder_none_overrides_raises -v`
Expected: FAIL (signature mismatch or incorrect behavior)

**Step 3: Write implementation**

Update `get_dep_specs_from_signature` in `src/pivot/stage_def.py`:

```python
def get_dep_specs_from_signature(
    func: Callable[..., Any],
    dep_path_overrides: Mapping[str, outputs.PathType] | None = None,
) -> dict[str, FuncDepSpec]:
    """Extract dependency specs from a function's parameter annotations.

    Looks for Annotated type hints containing Dep, PlaceholderDep, or IncrementalOut markers:

        def process(
            data: Annotated[DataFrame, Dep("input.csv", CSV())],
            config: Annotated[dict, Dep("config.json", JSON())],
        ) -> OutputType:
            ...

        specs = get_dep_specs_from_signature(process)
        # specs["data"].path == "input.csv"
        # specs["config"].path == "config.json"

    PlaceholderDep requires a path override:

        def compare(
            baseline: Annotated[DataFrame, PlaceholderDep(CSV())],
        ) -> OutputType:
            ...

        specs = get_dep_specs_from_signature(compare, {"baseline": "data.csv"})

    IncrementalOut as input creates a FuncDepSpec with creates_dep_edge=False:

        MyCache = Annotated[dict | None, IncrementalOut("cache.json", JSON())]

        def my_stage(existing: MyCache) -> MyCache:
            ...

        specs = get_dep_specs_from_signature(my_stage)
        # specs["existing"].creates_dep_edge == False

    Args:
        func: The function to inspect.
        dep_path_overrides: Path overrides for PlaceholderDep and Dep parameters.

    Returns:
        Dict mapping parameter names to FuncDepSpec objects.
        Empty dict if no Dep/PlaceholderDep/IncrementalOut annotations found.

    Raises:
        ValueError: If a PlaceholderDep parameter has no override.
    """
    import inspect as inspect_module

    hints = _get_type_hints_safe(func, func.__name__, include_extras=True)
    if hints is None:
        return {}

    sig = inspect_module.signature(func)
    specs = dict[str, FuncDepSpec]()
    overrides = dep_path_overrides or {}

    for param_name in sig.parameters:
        if param_name not in hints:
            continue

        param_type = _unwrap_type_alias(hints[param_name])

        # Check if it's an Annotated type
        if get_origin(param_type) is not Annotated:
            continue

        # Get the annotation args (first is the actual type, rest are metadata)
        args = get_args(param_type)
        if len(args) < 2:
            continue

        # Look for Dep, PlaceholderDep, or IncrementalOut in the metadata
        for metadata in args[1:]:
            if isinstance(metadata, outputs.PlaceholderDep):
                # PlaceholderDep requires override
                if param_name not in overrides:
                    raise ValueError(
                        f"PlaceholderDep '{param_name}' requires override in dep_path_overrides"
                    )
                placeholder = cast("outputs.PlaceholderDep[Any]", metadata)
                specs[param_name] = FuncDepSpec(
                    path=overrides[param_name],
                    loader=placeholder.loader,
                )
                break
            elif isinstance(metadata, outputs.Dep):
                # Cast to Dep[Any] - isinstance narrows to Dep[Unknown]
                dep = cast("outputs.Dep[Any]", metadata)
                # Use override if provided, otherwise annotation path
                path = overrides.get(param_name, dep.path)
                specs[param_name] = FuncDepSpec(path=path, loader=dep.loader)
                break
            elif isinstance(metadata, outputs.IncrementalOut):
                # IncrementalOut as input: loads file if exists, returns None if not
                # Does NOT create DAG edge (self-referential, avoids circular dependency)
                inc = cast("outputs.IncrementalOut[Any]", metadata)
                specs[param_name] = FuncDepSpec(
                    path=inc.path,
                    loader=inc.loader,
                    creates_dep_edge=False,
                )
                break

    return specs
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dep_injection.py::test_get_dep_specs_with_placeholder_and_overrides tests/test_dep_injection.py::test_get_dep_specs_mixed_placeholder_and_regular tests/test_dep_injection.py::test_get_dep_specs_placeholder_without_override_raises tests/test_dep_injection.py::test_get_dep_specs_placeholder_none_overrides_raises -v`
Expected: PASS

**Step 5: Run existing tests to ensure no regression**

Run: `uv run pytest tests/test_dep_injection.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
jj describe -m "feat(stage_def): update get_dep_specs_from_signature to handle PlaceholderDep"
```

---

### Task 4: Update registry.register() to validate PlaceholderDep overrides

**Files:**
- Modify: `src/pivot/registry.py:266-287`
- Test: `tests/test_registry.py`

**Step 1: Write the failing tests**

Add to appropriate test file (or create `tests/test_placeholder_dep.py` if more suitable):

```python
# Add to tests/test_registry.py or create tests/test_placeholder_dep.py

def test_register_placeholder_dep_without_override_raises() -> None:
    """Registration should fail when PlaceholderDep has no override."""
    from pivot import REGISTRY, exceptions, loaders, outputs
    from pivot.stage_def import StageParams

    class _CompareOutputs(TypedDict):
        result: Annotated[dict[str, int], outputs.Out("result.json", loaders.JSON[dict[str, int]]())]

    def compare(
        baseline: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
        experiment: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
    ) -> _CompareOutputs:
        return {"result": {"diff": 0}}

    with pytest.raises(exceptions.ValidationError, match="Placeholder dependencies missing overrides"):
        REGISTRY.register(compare, name="compare_test")


def test_register_placeholder_dep_partial_override_raises() -> None:
    """Registration should fail when only some PlaceholderDeps have overrides."""
    from pivot import REGISTRY, exceptions, loaders, outputs

    class _CompareOutputs(TypedDict):
        result: Annotated[dict[str, int], outputs.Out("result.json", loaders.JSON[dict[str, int]]())]

    def compare(
        baseline: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
        experiment: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
    ) -> _CompareOutputs:
        return {"result": {"diff": 0}}

    with pytest.raises(exceptions.ValidationError, match="baseline|experiment"):
        REGISTRY.register(
            compare,
            name="compare_partial",
            dep_path_overrides={"baseline": "model_a/results.csv"},
            # Missing: experiment
        )


def test_register_placeholder_dep_with_all_overrides_succeeds() -> None:
    """Registration should succeed when all PlaceholderDeps have overrides."""
    from pivot import REGISTRY, loaders, outputs

    class _CompareOutputs(TypedDict):
        result: Annotated[dict[str, int], outputs.Out("result.json", loaders.JSON[dict[str, int]]())]

    def compare_success(
        baseline: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
        experiment: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
    ) -> _CompareOutputs:
        return {"result": {"diff": 0}}

    # Should not raise
    REGISTRY.register(
        compare_success,
        name="compare_success_test",
        dep_path_overrides={
            "baseline": "model_a/results.csv",
            "experiment": "model_b/results.csv",
        },
    )

    info = REGISTRY.get("compare_success_test")
    assert info["deps"]["baseline"] is not None
    assert info["deps"]["experiment"] is not None


def test_register_placeholder_dep_error_message_lists_all_missing() -> None:
    """Error message should list all missing placeholder overrides."""
    from pivot import REGISTRY, exceptions, loaders, outputs

    class _CompareOutputs(TypedDict):
        result: Annotated[dict[str, int], outputs.Out("result.json", loaders.JSON[dict[str, int]]())]

    def compare_many(
        a: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
        b: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
        c: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
    ) -> _CompareOutputs:
        return {"result": {"count": 0}}

    with pytest.raises(exceptions.ValidationError) as exc_info:
        REGISTRY.register(compare_many, name="compare_many_test")

    # All three should be mentioned
    assert "a" in str(exc_info.value)
    assert "b" in str(exc_info.value)
    assert "c" in str(exc_info.value)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_registry.py::test_register_placeholder_dep_without_override_raises tests/test_registry.py::test_register_placeholder_dep_partial_override_raises tests/test_registry.py::test_register_placeholder_dep_with_all_overrides_succeeds tests/test_registry.py::test_register_placeholder_dep_error_message_lists_all_missing -v`
Expected: Tests fail (ValidationError not raised, or wrong message)

**Step 3: Write implementation**

Modify `src/pivot/registry.py` in the `register()` method, around line 266. Replace the current dep_specs extraction and validation logic:

```python
            # Identify placeholder deps BEFORE extraction
            placeholder_names = stage_def.get_placeholder_dep_names(func)

            # Validate all placeholders have overrides
            if placeholder_names:
                provided_overrides = set(dep_path_overrides.keys()) if dep_path_overrides else set()
                missing = placeholder_names - provided_overrides
                if missing:
                    raise exceptions.ValidationError(
                        f"Stage '{stage_name}' has invalid dependencies:\n"
                        f"  - Placeholder dependencies missing overrides: {', '.join(sorted(missing))}"
                    )

            # Extract deps from function annotations (PlaceholderDep resolved via overrides)
            dep_specs = stage_def.get_dep_specs_from_signature(func, dep_path_overrides)

            # Validate dep_path_overrides match annotation dep names (for regular Deps)
            if dep_path_overrides:
                unknown = set(dep_path_overrides.keys()) - set(dep_specs.keys())
                if unknown:
                    raise exceptions.ValidationError(
                        f"Stage '{stage_name}': dep_path_overrides contains unknown deps: {unknown}. "
                        + f"Available: {list(dep_specs.keys())}"
                    )
                # Disallow overrides for IncrementalOut inputs - path must match output annotation
                incremental_overrides = [
                    name for name in dep_path_overrides if not dep_specs[name].creates_dep_edge
                ]
                if incremental_overrides:
                    raise exceptions.ValidationError(
                        f"Stage '{stage_name}': cannot override IncrementalOut input paths: "
                        + f"{incremental_overrides}. IncrementalOut paths must match between "
                        + "input and output annotations."
                    )
```

Note: The `apply_dep_path_overrides` call is no longer needed for regular Deps since `get_dep_specs_from_signature` now handles overrides internally.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_registry.py::test_register_placeholder_dep_without_override_raises tests/test_registry.py::test_register_placeholder_dep_partial_override_raises tests/test_registry.py::test_register_placeholder_dep_with_all_overrides_succeeds tests/test_registry.py::test_register_placeholder_dep_error_message_lists_all_missing -v`
Expected: PASS

**Step 5: Run full test suite to check for regressions**

Run: `uv run pytest tests/ -x --tb=short`
Expected: All tests PASS

**Step 6: Commit**

```bash
jj describe -m "feat(registry): validate PlaceholderDep overrides at registration"
```

---

### Task 5: Remove apply_dep_path_overrides call from registry (now handled in extraction)

**Files:**
- Modify: `src/pivot/registry.py:287`
- Test: existing tests should still pass

Since `get_dep_specs_from_signature` now applies overrides internally for both regular Deps and PlaceholderDeps, we need to remove the separate `apply_dep_path_overrides` call.

**Step 1: Verify current behavior**

Run: `uv run pytest tests/test_dep_injection.py tests/test_registry.py -v`
Expected: All PASS

**Step 2: Remove redundant apply_dep_path_overrides call**

In `src/pivot/registry.py`, delete these lines (around line 286-287):

```python
                # Apply overrides  <- DELETE
                dep_specs = stage_def.apply_dep_path_overrides(dep_specs, dep_path_overrides)  <- DELETE
```

The validation of unknown keys and IncrementalOut overrides should remain.

**Step 3: Run tests to verify no regression**

Run: `uv run pytest tests/test_dep_injection.py tests/test_registry.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
jj describe -m "refactor(registry): remove redundant apply_dep_path_overrides call"
```

---

### Task 6: Add multifile PlaceholderDep tests (list and tuple paths)

**Files:**
- Test: `tests/test_dep_injection.py`

**Step 1: Write the tests**

Add to `tests/test_dep_injection.py`:

```python
def test_placeholder_dep_list_path_override() -> None:
    """PlaceholderDep should work with list path overrides."""

    def process_shards(
        shards: Annotated[list[pandas.DataFrame], outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
    ) -> _ProcessOutputs:
        return {"result": {"count": len(shards)}}

    overrides = {"shards": ["shard1.csv", "shard2.csv", "shard3.csv"]}
    specs = stage_def.get_dep_specs_from_signature(process_shards, overrides)

    assert specs["shards"].path == ["shard1.csv", "shard2.csv", "shard3.csv"]


def test_placeholder_dep_tuple_path_override() -> None:
    """PlaceholderDep should work with tuple path overrides."""

    def compare_pair(
        pair: Annotated[tuple[pandas.DataFrame, pandas.DataFrame], outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
    ) -> _ProcessOutputs:
        return {"result": {"count": 2}}

    overrides = {"pair": ("left.csv", "right.csv")}
    specs = stage_def.get_dep_specs_from_signature(compare_pair, overrides)

    assert specs["pair"].path == ("left.csv", "right.csv")
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_dep_injection.py::test_placeholder_dep_list_path_override tests/test_dep_injection.py::test_placeholder_dep_tuple_path_override -v`
Expected: PASS (should work with existing implementation)

**Step 3: Commit**

```bash
jj describe -m "test(dep_injection): add multifile PlaceholderDep tests"
```

---

### Task 7: Add integration test with full pipeline execution

**Files:**
- Test: `tests/test_placeholder_dep.py` (new file)

**Step 1: Write the integration test**

Create `tests/test_placeholder_dep.py`:

```python
# pyright: reportUnusedFunction=false
"""Integration tests for PlaceholderDep functionality."""

from __future__ import annotations

import pathlib
from typing import Annotated, TypedDict

import pandas
import pytest

from pivot import REGISTRY, loaders, outputs


class _CompareOutputs(TypedDict):
    diff: Annotated[dict[str, float], outputs.Out("diff.json", loaders.JSON[dict[str, float]]())]


def _compare_datasets(
    baseline: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
    experiment: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
) -> _CompareOutputs:
    """Compare two datasets and compute difference in means."""
    baseline_mean = baseline["value"].mean()
    experiment_mean = experiment["value"].mean()
    return {"diff": {"delta": float(experiment_mean - baseline_mean)}}


@pytest.fixture
def comparison_data(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Create baseline and experiment CSV files."""
    baseline = tmp_path / "model_a" / "results.csv"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("value\n10\n20\n30\n")

    experiment = tmp_path / "model_b" / "results.csv"
    experiment.parent.mkdir(parents=True)
    experiment.write_text("value\n15\n25\n35\n")

    return baseline, experiment


def test_placeholder_dep_e2e_execution(
    tmp_path: pathlib.Path,
    comparison_data: tuple[pathlib.Path, pathlib.Path],
    set_project_root: None,
) -> None:
    """PlaceholderDep stage should execute correctly with overridden paths."""
    baseline_path, experiment_path = comparison_data

    # Register with overrides
    REGISTRY.register(
        _compare_datasets,
        name="compare_ab",
        dep_path_overrides={
            "baseline": str(baseline_path.relative_to(tmp_path)),
            "experiment": str(experiment_path.relative_to(tmp_path)),
        },
    )

    # Execute via run
    from pivot import commands

    commands.run(["compare_ab"], root=tmp_path, force=True)

    # Verify output
    import json

    output = tmp_path / "diff.json"
    assert output.exists()
    result = json.loads(output.read_text())
    assert result["delta"] == 5.0  # (15+25+35)/3 - (10+20+30)/3 = 25 - 20 = 5


def test_placeholder_dep_reuse_function_different_overrides(
    tmp_path: pathlib.Path,
    comparison_data: tuple[pathlib.Path, pathlib.Path],
    set_project_root: None,
) -> None:
    """Same function can be registered multiple times with different overrides."""
    baseline_path, experiment_path = comparison_data

    # Create a third dataset
    third = tmp_path / "model_c" / "results.csv"
    third.parent.mkdir(parents=True)
    third.write_text("value\n100\n200\n300\n")

    # Register same function twice with different overrides
    REGISTRY.register(
        _compare_datasets,
        name="compare_ab_v2",
        dep_path_overrides={
            "baseline": str(baseline_path.relative_to(tmp_path)),
            "experiment": str(experiment_path.relative_to(tmp_path)),
        },
        out_path_overrides={"diff": "diff_ab.json"},
    )

    REGISTRY.register(
        _compare_datasets,
        name="compare_ac",
        dep_path_overrides={
            "baseline": str(baseline_path.relative_to(tmp_path)),
            "experiment": str(third.relative_to(tmp_path)),
        },
        out_path_overrides={"diff": "diff_ac.json"},
    )

    # Both should be registered
    assert REGISTRY.get("compare_ab_v2") is not None
    assert REGISTRY.get("compare_ac") is not None

    # Dependencies should be different
    ab_info = REGISTRY.get("compare_ab_v2")
    ac_info = REGISTRY.get("compare_ac")

    assert ab_info["deps"]["experiment"] != ac_info["deps"]["experiment"]
```

**Step 2: Run the test**

Run: `uv run pytest tests/test_placeholder_dep.py -v`
Expected: PASS

**Step 3: Commit**

```bash
jj describe -m "test: add PlaceholderDep integration tests"
```

---

### Task 8: Run full quality checks and fix any issues

**Files:**
- All modified files

**Step 1: Run type checker**

Run: `uv run basedpyright src/pivot/outputs.py src/pivot/stage_def.py src/pivot/registry.py`
Expected: No errors

**Step 2: Run linter**

Run: `uv run ruff check src/pivot/outputs.py src/pivot/stage_def.py src/pivot/registry.py`
Expected: No errors

**Step 3: Run formatter**

Run: `uv run ruff format src/pivot/outputs.py src/pivot/stage_def.py src/pivot/registry.py`

**Step 4: Run full test suite**

Run: `uv run pytest tests/ -n auto`
Expected: All tests PASS

**Step 5: Final commit**

```bash
jj describe -m "feat: PlaceholderDep for dependencies without defaults

PlaceholderDep is a dependency marker that has no default path and must be
overridden at registration time. This enables stages that compare data from
different sources without awkward workarounds like duplicate paths.

- Add PlaceholderDep class to outputs.py
- Add get_placeholder_dep_names() helper to stage_def.py
- Update get_dep_specs_from_signature() to handle PlaceholderDep with overrides
- Validate all PlaceholderDeps have overrides in registry.register()
- Add comprehensive tests"
```

---

## Summary

| Task | Description | Files Changed |
|------|-------------|---------------|
| 1 | Add PlaceholderDep class | `src/pivot/outputs.py` |
| 2 | Add get_placeholder_dep_names helper | `src/pivot/stage_def.py` |
| 3 | Update get_dep_specs_from_signature | `src/pivot/stage_def.py` |
| 4 | Validate placeholders in registry | `src/pivot/registry.py` |
| 5 | Remove redundant override call | `src/pivot/registry.py` |
| 6 | Add multifile tests | `tests/test_dep_injection.py` |
| 7 | Add integration tests | `tests/test_placeholder_dep.py` |
| 8 | Quality checks | All files |
