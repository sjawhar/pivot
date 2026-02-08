# Single StageDefinition Extraction Seam — Implementation Plan


**Goal:** Parse stage function annotations exactly once per registration and make type-hint resolution failures explicit.

**Architecture:** Introduce a `StageDefinition` dataclass in `stage_def.py` that bundles the results of all annotation parsing (dep specs, output specs, single output spec, placeholder dep names, params info) into one object. A single `extract_stage_definition()` function calls `get_type_hints()` once and produces the definition. Pipeline and Registry consume this pre-extracted definition instead of re-parsing annotations independently.

**Tech Stack:** Python 3.13+, dataclasses, existing `stage_def` / `registry` / `pipeline` modules.

---

## Problem Analysis

Currently, annotation parsing happens **redundantly** across three call sites:

| Call site | Functions called | `get_type_hints()` calls |
|-----------|-----------------|-------------------------|
| `Pipeline.register()` | `get_dep_specs_from_signature`, `get_output_specs_from_return`, `get_single_output_spec_from_return` | 3 |
| `Registry.register()` | `get_placeholder_dep_names`, `get_dep_specs_from_signature`, `get_output_specs_from_return`, `get_single_output_spec_from_return`, `find_params_in_signature` | 5+ |
| `yaml.py` `_expand_simple_stage` / `_expand_matrix` | `get_output_specs_from_return`, `find_params_in_signature` | 2 |

Additionally, `_get_type_hints_safe` silently returns `None` on failure, causing downstream functions to return empty dicts — a silent degradation the issue explicitly flags.

## Design Decisions

1. **`StageDefinition` is a dataclass** (not TypedDict) — it's an internal struct with methods disallowed on TypedDicts (potential future `validate()` etc.), and it's never serialized to JSON.

2. **`extract_stage_definition()` takes `dep_path_overrides`** — because `get_dep_specs_from_signature` needs overrides to resolve `PlaceholderDep`. This means the extraction is override-aware from the start.

3. **Hint resolution failures raise `StageDefinitionError`** by default — the current "return empty" behavior is replaced. A `strict=True` default parameter controls this; callers who genuinely need graceful degradation can pass `strict=False`.

4. **Existing public functions stay but delegate** — `get_dep_specs_from_signature()`, `get_output_specs_from_return()`, etc. remain as thin wrappers calling `extract_stage_definition()` internally. This avoids breaking any external callers or tests, and they can be deprecated later.

5. **Pipeline.register() passes `StageDefinition` to Registry** — Pipeline extracts once, resolves paths, then passes the definition. Registry receives it and skips re-extraction.

---

## Task 1: Add `StageDefinition` dataclass to `stage_def.py`

**Files:**
- Modify: `src/pivot/stage_def.py` (add dataclass near top, after `FuncDepSpec`)

**Step 1: Define the dataclass**

Add after the `FuncDepSpec` class (around line 434):

```python
@dataclasses.dataclass(frozen=True)
class StageDefinition:
    """Complete parsed definition of a stage function's annotations.

    Produced once by extract_stage_definition() and consumed by Pipeline/Registry.
    Avoids redundant get_type_hints() calls across registration layers.
    """

    dep_specs: dict[str, FuncDepSpec]
    out_specs: dict[str, outputs.BaseOut]
    single_out_spec: outputs.BaseOut | None
    placeholder_dep_names: frozenset[str]
    params_arg_name: str | None
    params_type: type[StageParams] | None
    hints_resolved: bool
```

Fields:
- `dep_specs`: from `get_dep_specs_from_signature` — includes PlaceholderDep resolution
- `out_specs`: from `get_output_specs_from_return` (TypedDict outputs)
- `single_out_spec`: from `get_single_output_spec_from_return` (single-return outputs)
- `placeholder_dep_names`: from `get_placeholder_dep_names`
- `params_arg_name`: param name holding StageParams (or None)
- `params_type`: StageParams subclass type (or None)
- `hints_resolved`: True if type hints were successfully resolved

**Step 2: Run type checker**

Run: `cd /home/sami/pivot/roadmap-380 && uv run basedpyright src/pivot/stage_def.py`
Expected: PASS (new class is just a data definition)

---

## Task 2: Implement `extract_stage_definition()` in `stage_def.py`

**Files:**
- Modify: `src/pivot/stage_def.py`

**Step 1: Write the failing test**

Create test in existing test structure. Add a new test file:

Create: `tests/core/test_stage_definition.py`

```python
# pyright: reportUnusedFunction=false
from __future__ import annotations

import pathlib
from typing import Annotated, TypedDict

import pytest

from pivot import exceptions, loaders, outputs, stage_def


class _SimpleOutput(TypedDict):
    result: Annotated[pathlib.Path, outputs.Out("result.txt", loaders.PathOnly())]


def _stage_with_dep(
    data: Annotated[pathlib.Path, outputs.Dep("input.csv", loaders.PathOnly())],
) -> _SimpleOutput:
    return _SimpleOutput(result=pathlib.Path("result.txt"))


def _stage_no_annotations() -> None:
    pass


class _TestParams(stage_def.StageParams):
    lr: float = 0.01


def _stage_with_params(
    params: _TestParams,
    data: Annotated[pathlib.Path, outputs.Dep("input.csv", loaders.PathOnly())],
) -> _SimpleOutput:
    return _SimpleOutput(result=pathlib.Path("result.txt"))


def _stage_with_placeholder(
    data: Annotated[pathlib.Path, outputs.PlaceholderDep(loaders.PathOnly())],
) -> _SimpleOutput:
    return _SimpleOutput(result=pathlib.Path("result.txt"))


def _single_output_stage(
    data: Annotated[pathlib.Path, outputs.Dep("input.csv", loaders.PathOnly())],
) -> Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]:
    return pathlib.Path("output.txt")


class TestExtractStageDefinition:
    def test_basic_extraction(self) -> None:
        defn = stage_def.extract_stage_definition(_stage_with_dep, "test_stage")
        assert defn.hints_resolved is True
        assert "data" in defn.dep_specs
        assert defn.dep_specs["data"].path == "input.csv"
        assert "result" in defn.out_specs
        assert defn.out_specs["result"].path == "result.txt"
        assert defn.single_out_spec is None
        assert defn.params_arg_name is None
        assert defn.params_type is None

    def test_no_annotations(self) -> None:
        defn = stage_def.extract_stage_definition(_stage_no_annotations, "bare")
        assert defn.hints_resolved is True
        assert defn.dep_specs == {}
        assert defn.out_specs == {}
        assert defn.single_out_spec is None

    def test_params_extraction(self) -> None:
        defn = stage_def.extract_stage_definition(_stage_with_params, "with_params")
        assert defn.params_arg_name == "params"
        assert defn.params_type is _TestParams

    def test_placeholder_dep_names(self) -> None:
        defn = stage_def.extract_stage_definition(
            _stage_with_placeholder,
            "placeholder",
            dep_path_overrides={"data": "override.csv"},
        )
        assert "data" in defn.placeholder_dep_names
        assert defn.dep_specs["data"].path == "override.csv"

    def test_placeholder_without_override_raises(self) -> None:
        with pytest.raises(ValueError, match="PlaceholderDep"):
            stage_def.extract_stage_definition(_stage_with_placeholder, "placeholder")

    def test_single_output_extraction(self) -> None:
        defn = stage_def.extract_stage_definition(_single_output_stage, "single")
        assert defn.out_specs == {}
        assert defn.single_out_spec is not None
        assert defn.single_out_spec.path == "output.txt"

    def test_hint_resolution_failure_raises(self) -> None:
        """Type hint resolution failure should raise, not silently degrade."""
        # Create a function with unresolvable type hint
        ns: dict[str, object] = {}
        exec(  # noqa: S102
            "from typing import Annotated\n"
            "from pivot import outputs, loaders\n"
            "def bad_func(x: 'UnresolvableType') -> None: pass\n",
            ns,
        )
        bad_func = ns["bad_func"]
        with pytest.raises(exceptions.StageDefinitionError, match="resolve type hints"):
            stage_def.extract_stage_definition(bad_func, "bad_stage")  # type: ignore[arg-type]

    def test_hint_resolution_failure_lenient(self) -> None:
        """With strict=False, hint failure returns hints_resolved=False."""
        ns: dict[str, object] = {}
        exec(  # noqa: S102
            "def bad_func(x: 'UnresolvableType') -> None: pass\n",
            ns,
        )
        bad_func = ns["bad_func"]
        defn = stage_def.extract_stage_definition(bad_func, "bad_stage", strict=False)  # type: ignore[arg-type]
        assert defn.hints_resolved is False
        assert defn.dep_specs == {}
        assert defn.out_specs == {}
```

**Step 2: Run the test to confirm it fails**

Run: `cd /home/sami/pivot/roadmap-380 && uv run pytest tests/core/test_stage_definition.py -v`
Expected: FAIL — `extract_stage_definition` does not exist yet.

**Step 3: Implement `extract_stage_definition()`**

Add to `src/pivot/stage_def.py` after the `StageDefinition` dataclass:

```python
def extract_stage_definition(
    func: Callable[..., Any],
    stage_name: str,
    dep_path_overrides: Mapping[str, outputs.PathType] | None = None,
    *,
    strict: bool = True,
) -> StageDefinition:
    """Extract complete stage definition from function annotations in a single pass.

    Calls get_type_hints() once and derives all dep/output/params specs from the
    result. This is the single extraction point — Pipeline and Registry should
    call this instead of individual extraction functions.

    Args:
        func: Stage function to extract from.
        stage_name: Name for error messages.
        dep_path_overrides: Override paths for PlaceholderDep and Dep annotations.
        strict: If True (default), raise StageDefinitionError when type hints
            can't be resolved. If False, return a definition with hints_resolved=False
            and empty specs.

    Returns:
        StageDefinition with all parsed annotation data.

    Raises:
        StageDefinitionError: If strict=True and type hints can't be resolved.
        ValueError: If PlaceholderDep has no override in dep_path_overrides.
    """
    import inspect as inspect_module

    hints = _get_type_hints_safe(func, func.__name__, include_extras=True)
    if hints is None:
        if strict:
            raise exceptions.StageDefinitionError(
                f"Stage '{stage_name}': failed to resolve type hints for '{func.__name__}'. "
                + "Check that all type annotations are importable."
            )
        return StageDefinition(
            dep_specs={},
            out_specs={},
            single_out_spec=None,
            placeholder_dep_names=frozenset(),
            params_arg_name=None,
            params_type=None,
            hints_resolved=False,
        )

    sig = inspect_module.signature(func)

    # --- Extract deps, placeholders, and params from parameters ---
    overrides = dep_path_overrides or {}
    dep_specs = dict[str, FuncDepSpec]()
    placeholder_dep_names = set[str]()
    params_arg_name: str | None = None
    params_type: type[StageParams] | None = None

    # Also get non-extras hints for params detection
    hints_no_extras = _get_type_hints_safe(func, func.__name__)

    for param_name in sig.parameters:
        if param_name not in hints:
            continue

        param_type = _unwrap_type_alias(hints[param_name])

        # Check for StageParams (use non-extras hints to avoid Annotated wrapper)
        if hints_no_extras and param_name in hints_no_extras:
            raw_type = hints_no_extras[param_name]
            if isinstance(raw_type, type) and issubclass(raw_type, StageParams):
                params_arg_name = param_name
                params_type = raw_type

        # Check for Annotated deps
        if get_origin(param_type) is not Annotated:
            continue

        args = get_args(param_type)
        if len(args) < 2:
            continue

        for metadata in args[1:]:
            if isinstance(metadata, outputs.PlaceholderDep):
                placeholder_dep_names.add(param_name)
                if param_name not in overrides:
                    raise ValueError(
                        f"PlaceholderDep '{param_name}' requires override in dep_path_overrides"
                    )
                override_path = overrides[param_name]
                if isinstance(override_path, (list, tuple)):
                    if not override_path or any(not p for p in override_path):
                        raise ValueError(
                            f"PlaceholderDep '{param_name}' override contains empty path"
                        )
                elif not override_path:
                    raise ValueError(f"PlaceholderDep '{param_name}' override cannot be empty")
                placeholder = cast("outputs.PlaceholderDep[Any]", metadata)
                dep_specs[param_name] = FuncDepSpec(
                    path=override_path,
                    loader=placeholder.loader,
                )
                break
            elif isinstance(metadata, outputs.Dep):
                dep = cast("outputs.Dep[Any]", metadata)
                path = overrides.get(param_name, dep.path)
                dep_specs[param_name] = FuncDepSpec(path=path, loader=dep.loader)
                break
            elif isinstance(metadata, outputs.IncrementalOut):
                inc = cast("outputs.IncrementalOut[Any]", metadata)
                dep_specs[param_name] = FuncDepSpec(
                    path=inc.path,
                    loader=inc.loader,
                    creates_dep_edge=False,
                )
                break

    # --- Extract output specs from return type ---
    out_specs = dict[str, outputs.BaseOut]()
    single_out_spec: outputs.BaseOut | None = None

    return_type = hints.get("return")
    if return_type is not None and return_type is not type(None):
        return_type = _unwrap_type_alias(return_type)

        if is_typeddict(return_type):
            out_specs = _extract_typeddict_outputs(return_type, stage_name)
        elif get_origin(return_type) is Annotated:
            rt_args = get_args(return_type)
            if len(rt_args) >= 2:
                for metadata in rt_args[1:]:
                    if isinstance(metadata, (outputs.Out, outputs.IncrementalOut, outputs.DirectoryOut)):
                        single_out_spec = cast("outputs.BaseOut", metadata)
                        break

    return StageDefinition(
        dep_specs=dep_specs,
        out_specs=out_specs,
        single_out_spec=single_out_spec,
        placeholder_dep_names=frozenset(placeholder_dep_names),
        params_arg_name=params_arg_name,
        params_type=params_type,
        hints_resolved=True,
    )
```

**Step 4: Run the tests**

Run: `cd /home/sami/pivot/roadmap-380 && uv run pytest tests/core/test_stage_definition.py -v`
Expected: ALL PASS

**Step 5: Run existing tests to confirm no regressions**

Run: `cd /home/sami/pivot/roadmap-380 && uv run pytest tests/ -n auto --timeout=120 -q`
Expected: ALL PASS (no behavior changes yet)

---

## Task 3: Update `Pipeline.register()` to use `extract_stage_definition()`

**Files:**
- Modify: `src/pivot/pipeline/pipeline.py:329-391` (the `register` method)

**Step 1: Write a focused test**

Add to `tests/core/test_stage_definition.py`:

```python
class TestPipelineUsesExtraction:
    """Verify Pipeline.register() uses extract_stage_definition internally."""

    def test_pipeline_register_calls_extract_once(self, mocker: MockerFixture) -> None:
        """Pipeline.register should call extract_stage_definition, not individual functions."""
        spy = mocker.spy(stage_def, "extract_stage_definition")
        p = Pipeline("test", root=pathlib.Path(__file__).parent)
        p.register(_stage_with_dep)
        spy.assert_called_once()
```

Add `MockerFixture` import:
```python
from pytest_mock import MockerFixture
```

Also add the `TYPE_CHECKING` guard for that import.

**Step 2: Run to confirm it fails**

Run: `cd /home/sami/pivot/roadmap-380 && uv run pytest tests/core/test_stage_definition.py::TestPipelineUsesExtraction -v`
Expected: FAIL — `extract_stage_definition` is not called by Pipeline yet.

**Step 3: Refactor `Pipeline.register()`**

Replace the body of `Pipeline.register()` in `src/pivot/pipeline/pipeline.py`. The key change: call `stage_def.extract_stage_definition()` once, then use the result for path resolution and pass it to `Registry.register()`.

```python
def register(
    self,
    func: StageFunc,
    *,
    name: str | None = None,
    params: registry.ParamsArg = None,
    mutex: list[str] | None = None,
    variant: str | None = None,
    dep_path_overrides: Mapping[str, outputs.PathType] | None = None,
    out_path_overrides: Mapping[str, registry.OutOverrideInput] | None = None,
) -> None:
    """Register a stage with this pipeline.

    Paths in annotations and overrides are resolved relative to pipeline root.
    """

    stage_name = name or func.__name__

    # 1. Extract stage definition (single pass over annotations)
    definition = stage_def.extract_stage_definition(func, stage_name, dep_path_overrides)

    # 2. Resolve annotation paths relative to pipeline root
    # Skip IncrementalOut - registry disallows path overrides for them
    resolved_deps: dict[str, outputs.PathType] = {
        dep_name: self._resolve_path_type(spec.path)
        for dep_name, spec in definition.dep_specs.items()
        if spec.creates_dep_edge
    }

    out_specs = definition.out_specs
    if not out_specs and definition.single_out_spec is not None:
        out_specs = {stage_def.SINGLE_OUTPUT_KEY: definition.single_out_spec}

    resolved_outs: dict[str, registry.OutOverride] = {
        out_name: registry.OutOverride(path=self._resolve_path_type(spec.path))
        for out_name, spec in out_specs.items()
        if not isinstance(spec, outputs.IncrementalOut)
    }

    # 3. Apply explicit overrides (also pipeline-relative)
    if dep_path_overrides:
        for dep_name, path in dep_path_overrides.items():
            resolved_deps[dep_name] = self._resolve_path_type(path)
    if out_path_overrides:
        for out_name, override in out_path_overrides.items():
            resolved_outs[out_name] = self._resolve_out_override(override)

    # 4. Pass definition + resolved overrides to registry
    self._registry.register(
        func=func,
        name=name,
        params=params,
        mutex=mutex,
        variant=variant,
        dep_path_overrides=resolved_deps,
        out_path_overrides=resolved_outs,
        state_dir=self.state_dir,
        definition=definition,
    )

    self._reset_resolution_cache()
```

**Step 4: Run the test**

Run: `cd /home/sami/pivot/roadmap-380 && uv run pytest tests/core/test_stage_definition.py::TestPipelineUsesExtraction -v`
Expected: FAIL until Task 4 (Registry must accept `definition` parameter). Continue to Task 4.

---

## Task 4: Update `Registry.register()` to accept and use `StageDefinition`

**Files:**
- Modify: `src/pivot/registry.py:256-491` (the `register` method)

**Step 1: Add `definition` parameter to `Registry.register()`**

Add optional `definition: stage_def.StageDefinition | None = None` parameter. When provided, skip all annotation parsing and use the pre-extracted data. When `None` (direct registry usage, tests), extract internally.

The key changes in `Registry.register()`:

```python
def register(
    self,
    func: Callable[..., Any],
    name: str | None = None,
    params: ParamsArg = None,
    mutex: Sequence[str] | None = None,
    variant: str | None = None,
    dep_path_overrides: Mapping[str, outputs.PathType] | None = None,
    out_path_overrides: Mapping[str, OutOverrideInput] | None = None,
    state_dir: pathlib.Path | None = None,
    definition: stage_def.StageDefinition | None = None,
) -> None:
```

Near the top of the method body, add:

```python
# Extract or reuse stage definition
if definition is None:
    definition = stage_def.extract_stage_definition(func, stage_name, dep_path_overrides)
```

Then replace all individual calls:
- `stage_def.get_placeholder_dep_names(func)` → `definition.placeholder_dep_names`
- `stage_def.get_dep_specs_from_signature(func, dep_path_overrides)` → `definition.dep_specs`
- `stage_def.get_output_specs_from_return(func, stage_name)` → `definition.out_specs`
- `stage_def.get_single_output_spec_from_return(func)` → `definition.single_out_spec`
- `stage_def.find_params_in_signature(func)` → `(definition.params_arg_name, definition.params_type)`

**Step 2: Run all tests**

Run: `cd /home/sami/pivot/roadmap-380 && uv run pytest tests/ -n auto --timeout=120 -q`
Expected: ALL PASS

The spy test from Task 3 should now pass too:
Run: `cd /home/sami/pivot/roadmap-380 && uv run pytest tests/core/test_stage_definition.py -v`
Expected: ALL PASS

---

## Task 5: Update `yaml.py` to use `extract_stage_definition()` for metrics/plots validation

**Files:**
- Modify: `src/pivot/pipeline/yaml.py:295-336` (`_expand_simple_stage`) and `440-465` (`_expand_matrix`)

**Step 1: Update `_expand_simple_stage`**

Replace:
```python
return_out_specs = stage_def.get_output_specs_from_return(func, name)
```

With:
```python
definition = stage_def.extract_stage_definition(func, name)
return_out_specs = definition.out_specs
```

**Step 2: Update `_expand_matrix`**

Similarly, inside the matrix combo loop replace:
```python
return_out_specs = stage_def.get_output_specs_from_return(func, full_name)
```

With:
```python
# Extract once before loop (for type validation of metrics/plots)
definition = stage_def.extract_stage_definition(func, full_name)
return_out_specs = definition.out_specs
```

Note: The `definition` extraction for the matrix loop should happen **once before the loop** (the function is the same for all combos), not per-combo. So extract before the `for combo in combinations:` loop and use `definition.out_specs` inside.

**Step 3: Update `_resolve_params` in yaml.py**

Replace:
```python
params_arg_name, params_type = stage_def.find_params_in_signature(func)
```

With accepting the definition as a parameter (or keeping as-is since `_resolve_params` is called from multiple sites and `find_params_in_signature` is cached). This is a judgment call — since `_resolve_params` in yaml.py is called after extraction, we could thread the definition through. But `find_params_in_signature` already uses a `WeakKeyDictionary` cache, so the redundancy is minimal. **Leave this call as-is** — it's already cached and changing it would require threading the definition through `_expand_variants` too, adding complexity for negligible gain.

**Step 4: Run tests**

Run: `cd /home/sami/pivot/roadmap-380 && uv run pytest tests/ -n auto --timeout=120 -q`
Expected: ALL PASS

---

## Task 6: Make existing public functions delegate to `extract_stage_definition()`

**Files:**
- Modify: `src/pivot/stage_def.py`

**Context:** The existing public functions (`get_dep_specs_from_signature`, `get_output_specs_from_return`, `get_single_output_spec_from_return`, `get_placeholder_dep_names`) are still called from tests and potentially external code. Make them delegate to `extract_stage_definition()` so there's truly a single parsing path.

**Step 1: Rewrite delegating functions**

Replace `get_dep_specs_from_signature`:
```python
def get_dep_specs_from_signature(
    func: Callable[..., Any],
    dep_path_overrides: Mapping[str, outputs.PathType] | None = None,
) -> dict[str, FuncDepSpec]:
    """Extract dependency specs from function annotations.

    Delegates to extract_stage_definition() for single-pass extraction.
    """
    defn = extract_stage_definition(func, func.__name__, dep_path_overrides, strict=False)
    return defn.dep_specs
```

Replace `get_output_specs_from_return`:
```python
def get_output_specs_from_return(
    func: Callable[..., Any],
    stage_name: str,
) -> dict[str, outputs.BaseOut]:
    """Extract output specs from return type annotation.

    Delegates to extract_stage_definition() for single-pass extraction.
    """
    defn = extract_stage_definition(func, stage_name, strict=False)
    return defn.out_specs
```

Replace `get_single_output_spec_from_return`:
```python
def get_single_output_spec_from_return(func: Callable[..., Any]) -> outputs.BaseOut | None:
    """Extract single output spec from return annotation.

    Delegates to extract_stage_definition() for single-pass extraction.
    """
    defn = extract_stage_definition(func, func.__name__, strict=False)
    return defn.single_out_spec
```

Replace `get_placeholder_dep_names`:
```python
def get_placeholder_dep_names(func: Callable[..., Any]) -> set[str]:
    """Get parameter names with PlaceholderDep annotations.

    Delegates to extract_stage_definition() for single-pass extraction.
    """
    # Can't call extract_stage_definition with strict placeholders — we don't have overrides.
    # Keep the original implementation for this one since it doesn't need overrides.
```

**Important:** `get_placeholder_dep_names` is special — it's called *before* overrides are known (to validate that overrides exist). It cannot delegate to `extract_stage_definition()` because that function raises on unresolved PlaceholderDep. **Keep the original implementation** for this function.

Similarly, `get_output_specs_from_return` has a subtle difference — when called with `strict=False`, hint failures won't raise a `StageDefinitionError` which matches the current behavior of returning `{}`.

**Step 2: Run all tests**

Run: `cd /home/sami/pivot/roadmap-380 && uv run pytest tests/ -n auto --timeout=120 -q`
Expected: ALL PASS

---

## Task 7: Add test for explicit hint-failure behavior in Registry

**Files:**
- Create: `tests/core/test_stage_definition.py` (add more tests)

**Step 1: Test that Registry with strict extraction raises on bad hints**

```python
class TestHintResolutionExplicitFailure:
    """Verify hint resolution failures are explicit, not silent."""

    def test_registry_raises_on_unresolvable_hints(self) -> None:
        """Registry should raise StageDefinitionError for bad type hints."""
        ns: dict[str, object] = {}
        exec(  # noqa: S102
            "from typing import Annotated\n"
            "from pivot import outputs, loaders\n"
            "def bad_stage(x: 'CompletelyFakeType') -> None: pass\n",
            ns,
        )
        bad_func = ns["bad_stage"]
        reg = registry.StageRegistry()
        with pytest.raises(exceptions.StageDefinitionError, match="resolve type hints"):
            reg.register(bad_func, name="bad_stage")  # type: ignore[arg-type]
```

Add `registry` to the imports at the top of the test file.

**Step 2: Run the test**

Run: `cd /home/sami/pivot/roadmap-380 && uv run pytest tests/core/test_stage_definition.py::TestHintResolutionExplicitFailure -v`
Expected: PASS

---

## Task 8: Quality checks and final validation

**Step 1: Run formatter and linter**

Run: `cd /home/sami/pivot/roadmap-380 && uv run ruff format . && uv run ruff check .`
Expected: PASS

**Step 2: Run type checker**

Run: `cd /home/sami/pivot/roadmap-380 && uv run basedpyright`
Expected: PASS (fix any type issues found)

**Step 3: Run full test suite**

Run: `cd /home/sami/pivot/roadmap-380 && uv run pytest tests/ -n auto --timeout=120`
Expected: ALL PASS

---

## Summary of Changes

| File | Change |
|------|--------|
| `src/pivot/stage_def.py` | Add `StageDefinition` dataclass + `extract_stage_definition()` function. Delegate existing public functions. |
| `src/pivot/pipeline/pipeline.py` | `register()` calls `extract_stage_definition()` once, passes result to Registry. |
| `src/pivot/registry.py` | `register()` accepts optional `definition` param, skips re-extraction when provided. |
| `src/pivot/pipeline/yaml.py` | `_expand_simple_stage` and `_expand_matrix` use `extract_stage_definition()` for output spec validation. |
| `tests/core/test_stage_definition.py` | New test file for extraction, integration, and hint-failure behavior. |

## Risk Areas

1. **PlaceholderDep validation ordering** — `get_placeholder_dep_names` must still work without overrides (it's called before overrides are validated in Registry). The plan keeps this function's original implementation.

2. **`strict=False` in delegating functions** — existing callers expect empty results on hint failure, not exceptions. The delegating wrappers use `strict=False` to preserve backward compatibility.

3. **`_get_type_hints_safe` calls `get_type_hints` twice** — once with `include_extras=True` (for Annotated metadata) and once without (for params type detection). The new extraction function still needs both calls since params detection requires unwrapped types. This is still a reduction from 5+ calls to 2.
