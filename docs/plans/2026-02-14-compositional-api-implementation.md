# Compositional Pipeline API — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Pivot's verbose `Dep()`/`Out()`/`Pipeline.register()` API with a
compositional `@stage` + `with Pipeline()` model where you pass artifact handles
between function calls to define the DAG.

**Architecture:** New API is a frontend that produces the same `RegistryStageInfo` the
existing engine consumes. The engine, executor, worker, fingerprinting, caching — all
unchanged initially. The new frontend generates deterministic file paths from artifact
identity (pipeline name + stage name + output key + format), so the engine never sees
handles — only the paths it already understands.

**Tech Stack:** Python 3.13, Pydantic, basedpyright, ruff, pytest. No new dependencies.

**Design Doc:** `docs/plans/2026-02-14-compositional-pipeline-api.md`

**Validation Strategy:** Each phase converts a real eval-pipeline sub-pipeline to the
new API and runs `pivot repro` end-to-end. Order: `difficulty/` (no deps) →
`base/` (variants) → `horizon/` (cross-pipeline, 80+ stages).

---

## Phase 1: Core Primitives

Build `@stage`, `ArtifactHandle`, and `Pipeline` context manager. By the end of this
phase, you can define the difficulty/ pipeline with the new API and it produces a
valid `Pipeline` (old internal type) that the existing engine can execute.

### Task 1.1: `ArtifactHandle` and `StageNode`

The internal data model for the compositional DAG. These are created during pipeline
definition (inside `with Pipeline()`) and converted to `RegistryStageInfo` at exit.

**Files:**
- Create: `packages/pivot/src/pivot/compose.py`
- Test: `packages/pivot/tests/test_compose.py`

**Implementation:**

```python
# packages/pivot/src/pivot/compose.py
"""Compositional pipeline API: @stage, ArtifactHandle, Pipeline context."""
from __future__ import annotations

import contextvars
import dataclasses
import functools
import inspect
import pathlib
from typing import TYPE_CHECKING, Any, Annotated, get_args, get_origin, get_type_hints

from pivot import outputs, stage_def

if TYPE_CHECKING:
    from collections.abc import Callable
    from pivot import loaders as loaders_mod

# IMPORTANT: Do NOT import pivot.loaders at module level — it imports pandas.
# All loaders access must go through _get_loaders() to keep compose.py importable
# without heavy dependencies. This enables fast @stage decoration at import time.
def _get_loaders():  # type: ignore[return-value]
    from pivot import loaders
    return loaders

# Sentinel tags for output categorization
class _MetricTag:
    """Tag for metric outputs."""
class _PlotTag:
    """Tag for plot outputs."""

# Module-level instances used in Annotated[] annotations
metric = _MetricTag()
plot = _PlotTag()

# Active pipeline context variable
_active_pipeline: contextvars.ContextVar[Pipeline | None] = contextvars.ContextVar(
    "_active_pipeline", default=None
)

# --- Format inference (lazy — no pandas import at module level) ---

_DEFAULT_FORMATS: dict[str, str] = {}  # type_qualname -> loader_class_name
_DEFAULT_FORMATS_RESOLVED: dict[type, str] | None = None  # type -> loader_class_name

def _register_default_format(type_qualname: str, loader_class_name: str) -> None:
    """Register a default format by fully-qualified type name (lazy resolution)."""
    _DEFAULT_FORMATS[type_qualname] = loader_class_name

# Register defaults by string name — resolved lazily on first use.
# This avoids importing pandas/matplotlib at compose.py import time.
# Factory values are STRINGS — resolved to actual loader classes via _get_loaders()
# when _resolve_default_formats() is first called.
_register_default_format("pandas.core.frame.DataFrame", "DataFrameJSONL")
_register_default_format("builtins.dict", "YAML")
_register_default_format("builtins.list", "YAML")
_register_default_format("builtins.str", "Text")
_register_default_format("pathlib.Path", "PathOnly")

def _resolve_default_formats() -> dict[type, str]:
    """Lazily resolve string type names to actual Python types."""
    global _DEFAULT_FORMATS_RESOLVED
    if _DEFAULT_FORMATS_RESOLVED is not None:
        return _DEFAULT_FORMATS_RESOLVED
    import importlib
    resolved: dict[type, str] = {}
    for qualname, loader_name in _DEFAULT_FORMATS.items():
        mod_name, cls_name = qualname.rsplit(".", 1)
        try:
            mod = importlib.import_module(mod_name)
            resolved[getattr(mod, cls_name)] = loader_name
        except (ImportError, AttributeError):
            pass
    _DEFAULT_FORMATS_RESOLVED = resolved
    return resolved

def _infer_format(python_type: type) -> Any:
    """Infer serialization format from Python type.

    Returns a Writer instance. All loaders access goes through _get_loaders().
    """
    loaders = _get_loaders()
    defaults = _resolve_default_formats()
    if python_type in defaults:
        return getattr(loaders, defaults[python_type])()
    for registered_type, loader_name in defaults.items():
        if isinstance(python_type, type) and issubclass(python_type, registered_type):
            return getattr(loaders, loader_name)()
    raise ValueError(
        f"Cannot infer serialization format for type {python_type.__name__}. "
        "Use Annotated[T, format] on the return type to specify explicitly."
    )

def _format_extension(fmt: Any) -> str:
    """Get file extension for a format."""
    loaders = _get_loaders()
    match fmt:
        case loaders.DataFrameJSONL():
            return "jsonl"
        case loaders.CSV():
            return "csv"
        case loaders.YAML():
            return "yaml"
        case loaders.JSON():
            return "json"
        case loaders.Text():
            return "txt"
        case loaders.Pickle():
            return "pkl"
        case loaders.MatplotlibFigure():
            return "png"
        case _:
            return "dat"


# --- Data model ---

@dataclasses.dataclass
class _OutputSpec:
    """Specification for a single output of a stage."""
    key: str  # output name (SINGLE_OUTPUT_KEY for single-output, field name for TypedDict)
    python_type: type
    format: loaders_mod.Writer[Any]
    tag: _MetricTag | _PlotTag | None = None


@dataclasses.dataclass
class _StageNode:
    """A stage call recorded during pipeline definition."""
    func: Callable[..., Any]
    original_func: Callable[..., Any]  # unwrapped function (for fingerprinting)
    name: str  # stage name (function name, possibly with disambiguation suffix)
    params: stage_def.StageParams | None
    input_handles: dict[str, ArtifactHandle]  # param_name -> handle
    output_specs: list[_OutputSpec]
    call_index: int  # for disambiguation when same function called multiple times


@dataclasses.dataclass
class _InputNode:
    """An external input declared via p.input()."""
    name: str
    python_type: type | None
    path: str  # relative path to the input file (e.g. "sql/difficulty/fetch_baselines.sql.jinja")
    format: loaders_mod.Reader[Any] | loaders_mod.Loader[Any, Any]  # reader for loading from disk


class ArtifactHandle:
    """Represents the output of a stage (or an input) in a pipeline definition.

    Passed between @stage calls to wire the DAG. Not actual data — just a reference.
    """

    _pipeline: Pipeline
    _source: _StageNode | _InputNode
    _output_key: str | None  # None for single-output stages or inputs
    _python_type: type

    def __init__(
        self,
        pipeline: Pipeline,
        source: _StageNode | _InputNode,
        output_key: str | None,
        python_type: type,
    ) -> None:
        self._pipeline = pipeline
        self._source = source
        self._output_key = output_key
        self._python_type = python_type

    def __getattr__(self, name: str) -> ArtifactHandle:
        """Access sub-handle for multi-output stages: tw.filtered_runs"""
        if name.startswith("_"):
            raise AttributeError(name)
        if not isinstance(self._source, _StageNode):
            raise AttributeError(f"Input '{self._source.name}' has no sub-outputs")
        # Find the matching output spec
        for spec in self._source.output_specs:
            if spec.key == name:
                return ArtifactHandle(
                    pipeline=self._pipeline,
                    source=self._source,
                    output_key=name,
                    python_type=spec.python_type,
                )
        available = [s.key for s in self._source.output_specs]
        raise AttributeError(
            f"Stage '{self._source.name}' has no output '{name}'. "
            f"Available: {available}"
        )
```

**Step 1:** Write the file above.

**Step 2:** Write basic tests:

```python
# packages/pivot/tests/test_compose.py
"""Tests for the compositional pipeline API."""
from __future__ import annotations
from pivot.compose import ArtifactHandle, _StageNode, _InputNode, _OutputSpec, _infer_format
from pivot import loaders
import pandas as pd


def test_infer_format_dataframe():
    fmt = _infer_format(pd.DataFrame)
    assert isinstance(fmt, loaders.DataFrameJSONL)


def test_infer_format_dict():
    fmt = _infer_format(dict)
    assert isinstance(fmt, loaders.YAML)
```

**Step 3:** Run tests: `uv run pytest packages/pivot/tests/test_compose.py -v`

**Step 4:** Commit: `feat(compose): add ArtifactHandle and StageNode data model`

---

### Task 1.2: `@stage` decorator

The decorator wraps a function so it returns an `ArtifactHandle` when called inside a
`with Pipeline()` block, and executes normally otherwise.

**Files:**
- Modify: `packages/pivot/src/pivot/compose.py`
- Test: `packages/pivot/tests/test_compose.py`

**Implementation** (add to `compose.py`):

```python
# --- Return type analysis ---

def _analyze_return_type(func: Callable[..., Any]) -> list[_OutputSpec]:
    """Extract output specs from a function's return type annotation.

    Rules:
    - pd.DataFrame -> single output, JSONL format
    - dict -> single output, YAML format
    - Annotated[pd.DataFrame, CSV()] -> single output, CSV format
    - Annotated[dict, metric] -> single output, YAML format, metric tag
    - TypedDict with annotated fields -> multi-output
    """
    hints = get_type_hints(func, include_extras=True)
    return_hint = hints.get("return")
    if return_hint is None:
        raise ValueError(f"Stage function {func.__name__} must have a return type annotation")

    return _parse_output_type(return_hint, stage_def.SINGLE_OUTPUT_KEY)


def _parse_output_type(hint: Any, key: str) -> list[_OutputSpec]:
    """Parse a type hint into OutputSpec(s)."""
    from typing_extensions import is_typeddict

    # Check for TypedDict (multi-output)
    origin = get_origin(hint)
    if isinstance(hint, type) and is_typeddict(hint):
        specs = []
        field_hints = get_type_hints(hint, include_extras=True)
        for field_name, field_hint in field_hints.items():
            specs.extend(_parse_output_type(field_hint, field_name))
        return specs

    # Check for Annotated
    if origin is Annotated:
        args = get_args(hint)
        base_type = args[0]
        tag = None
        fmt = None
        loaders = _get_loaders()
        for arg in args[1:]:
            if isinstance(arg, _MetricTag):
                tag = arg
            elif isinstance(arg, _PlotTag):
                tag = arg
            elif isinstance(arg, loaders.Writer):
                fmt = arg

        if fmt is None:
            fmt = _infer_format(base_type)
        return [_OutputSpec(key=key, python_type=base_type, format=fmt, tag=tag)]

    # Plain type
    python_type = hint
    fmt = _infer_format(python_type)
    return [_OutputSpec(key=key, python_type=python_type, format=fmt)]


# --- @stage decorator ---

def stage(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a function as a pipeline stage.

    In pipeline context (inside `with Pipeline()`): records a DAG node, returns ArtifactHandle.
    Outside pipeline context: calls the function normally (for tests/notebooks).

    IMPORTANT: Return type analysis is DEFERRED to first pipeline-context use,
    not performed at decoration time. This allows stage modules to use in-body
    imports for heavy dependencies (pandas, matplotlib) while still having
    type annotations that reference those types via string annotations
    (from __future__ import annotations).
    """
    # Cache for lazily-computed output specs
    _cached_specs: list[_OutputSpec] | None = None

    def _get_output_specs() -> list[_OutputSpec]:
        nonlocal _cached_specs
        if _cached_specs is None:
            _cached_specs = _analyze_return_type(func)
        return _cached_specs

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        pipeline = _active_pipeline.get()
        if pipeline is None:
            # Direct execution mode
            return func(*args, **kwargs)
        # Pipeline definition mode — analyze return type on first call
        return pipeline._record_stage(func, wrapper, _get_output_specs(), args, kwargs)

    wrapper._is_stage = True  # type: ignore[attr-defined]
    wrapper._original_func = func  # type: ignore[attr-defined]
    return wrapper
```

**Step 1:** Add the code above to `compose.py`.

**Step 2:** Write tests:

```python
def test_stage_direct_execution():
    """@stage functions work normally outside pipeline context."""
    @stage
    def add_one(x: int) -> int:
        return x + 1

    assert add_one(5) == 6


def test_stage_preserves_metadata():
    @stage
    def my_func(x: pd.DataFrame) -> pd.DataFrame:
        return x

    assert my_func.__name__ == "my_func"
    assert my_func._is_stage is True


def test_analyze_return_type_single():
    """Test return type analysis directly (not via wrapper attribute)."""
    def my_func() -> pd.DataFrame: ...
    specs = _analyze_return_type(my_func)
    assert len(specs) == 1
    assert specs[0].python_type is pd.DataFrame
    assert isinstance(specs[0].format, loaders.DataFrameJSONL)


def test_analyze_return_type_annotated_override():
    def my_func() -> Annotated[pd.DataFrame, CSV()]:  ...
    specs = _analyze_return_type(my_func)
    assert isinstance(specs[0].format, loaders.CSV)


def test_analyze_return_type_metric_tag():
    def my_func() -> Annotated[dict, metric]: ...
    specs = _analyze_return_type(my_func)
    assert isinstance(specs[0].tag, _MetricTag)
```

**Step 3:** Run tests: `uv run pytest packages/pivot/tests/test_compose.py -v`

**Step 4:** Commit: `feat(compose): add @stage decorator with return type analysis`

---

### Task 1.3: `Pipeline` context manager

The context manager that collects `@stage` calls and converts them to the internal
representation the engine understands.

**Files:**
- Modify: `packages/pivot/src/pivot/compose.py`
- Test: `packages/pivot/tests/test_compose.py`

**Implementation** (add to `compose.py`):

```python
# --- Path generation ---

def _artifact_dir_prefix(tag: _MetricTag | _PlotTag | None) -> str:
    """Get directory prefix based on output tag."""
    if isinstance(tag, _MetricTag):
        return "metrics"
    if isinstance(tag, _PlotTag):
        return "plots"
    return "data"


def _generate_artifact_path(
    pipeline_name: str,
    stage_name: str,
    output_spec: _OutputSpec,
    is_single_output: bool,
) -> str:
    """Generate a deterministic path for an artifact."""
    prefix = _artifact_dir_prefix(output_spec.tag)
    ext = _format_extension(output_spec.format)
    if is_single_output:
        return f"{prefix}/{pipeline_name}/{stage_name}.{ext}"
    return f"{prefix}/{pipeline_name}/{stage_name}/{output_spec.key}.{ext}"


# --- Pipeline ---

class Pipeline:
    """Compositional pipeline definition context.

    Usage:
        with Pipeline("my_pipeline") as p:
            data = p.input("raw_data")
            result = my_stage(data, params=MyParams(...))
    """

    _name: str
    _root: pathlib.Path
    _stages: list[_StageNode]
    _inputs: dict[str, _InputNode]
    _call_counts: dict[str, int]  # function_name -> call count (for disambiguation)
    _validation_errors: list[str]
    _token: contextvars.Token[Pipeline | None] | None

    def __init__(self, name: str, *, root: pathlib.Path | None = None) -> None:
        self._name = name
        self._stages = []
        self._inputs = {}
        self._call_counts = {}
        self._validation_errors = []
        self._token = None

        if root is not None:
            self._root = root.resolve()
        else:
            frame = inspect.currentframe()
            try:
                if frame is None or frame.f_back is None:
                    raise RuntimeError("Cannot determine caller frame")
                caller_file = frame.f_back.f_globals.get("__file__")
                if caller_file is None:
                    raise RuntimeError("Provide explicit root= for Pipeline")
                self._root = pathlib.Path(caller_file).resolve().parent
            finally:
                del frame

    @property
    def name(self) -> str:
        return self._name

    def __enter__(self) -> Pipeline:
        self._token = _active_pipeline.set(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._token is not None:
            _active_pipeline.reset(self._token)
            self._token = None
        if exc_type is None:
            self._validate()

    def input(
        self,
        name: str,
        path: str,
        format: loaders_mod.Reader[Any] | loaders_mod.Loader[Any, Any] | None = None,
        python_type: type | None = None,
    ) -> ArtifactHandle:
        """Declare an external input (not produced by any stage).

        Args:
            name: Human-readable name for this input.
            path: Relative path to the input file (resolved from pipeline root).
            format: Reader/Loader for deserializing. Inferred from file extension
                    if not provided (e.g. .yaml → YAML(), .csv → CSV(), .jsonl → DataFrameJSONL()).
            python_type: Expected Python type (for validation). Inferred from format if not provided.

        Note: In a future phase, path will become optional — Pivot will resolve
        inputs by name via .pvt sidecar files. For now, path is required to
        bridge to the existing engine which needs file paths.
        """
        if format is None:
            format = _infer_format_from_extension(path)
        node = _InputNode(name=name, python_type=python_type, path=path, format=format)
        self._inputs[name] = node
        return ArtifactHandle(
            pipeline=self,
            source=node,
            output_key=None,
            python_type=python_type or Any,
        )

    def _record_stage(
        self,
        original_func: Callable[..., Any],
        wrapper: Callable[..., Any],
        output_specs: list[_OutputSpec],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ArtifactHandle:
        """Record a stage call during pipeline definition. Returns ArtifactHandle."""
        func_name = original_func.__name__

        # Disambiguate repeated calls to the same function
        count = self._call_counts.get(func_name, 0)
        self._call_counts[func_name] = count + 1
        stage_name = func_name if count == 0 else f"{func_name}@{count}"

        # Bind args to parameter names
        sig = inspect.signature(original_func)
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError as e:
            self._validation_errors.append(f"{stage_name}: {e}")
            # Return a dummy handle so pipeline definition can continue
            # (validation errors are collected and reported at __exit__)
            dummy_input = _InputNode(
                name="__error__",
                python_type=None,
                path="__error__",
                format=_get_loaders().PathOnly(),
            )
            return ArtifactHandle(self, dummy_input, None, Any)

        # Separate handles (deps), params, and plain values
        input_handles: dict[str, ArtifactHandle] = {}
        params: stage_def.StageParams | None = None
        for param_name, value in bound.arguments.items():
            if isinstance(value, ArtifactHandle):
                input_handles[param_name] = value
                # Type check: handle type vs parameter annotation
                param_annotation = sig.parameters[param_name].annotation
                # (validation deferred to _validate())
            elif isinstance(value, stage_def.StageParams):
                params = value

        node = _StageNode(
            func=wrapper,
            original_func=original_func,
            name=stage_name,
            params=params,
            input_handles=input_handles,
            output_specs=output_specs,
            call_index=count,
        )
        self._stages.append(node)

        # Return handle(s)
        if len(output_specs) == 1:
            return ArtifactHandle(
                pipeline=self,
                source=node,
                output_key=None,
                python_type=output_specs[0].python_type,
            )
        # Multi-output: return a handle that supports attribute access
        return ArtifactHandle(
            pipeline=self,
            source=node,
            output_key=None,
            python_type=dict,  # placeholder — access via .<field_name>
        )

    def _validate(self) -> None:
        """Validate the pipeline after the with-block exits."""
        if self._validation_errors:
            msg = f"Pipeline \"{self._name}\" has {len(self._validation_errors)} validation error(s):\n\n"
            for err in self._validation_errors:
                msg += f"  {err}\n"
            raise ValueError(msg)
```

**Step 1:** Add the code above.

**Step 2:** Write tests:

```python
def test_pipeline_context_basic():
    """Pipeline context records stages and wires handles."""
    @stage
    def produce(params: StageParams) -> pd.DataFrame:
        return pd.DataFrame()

    @stage
    def consume(data: pd.DataFrame) -> dict:
        return {}

    with Pipeline("test", root=Path("/tmp")) as p:
        data = produce(params=StageParams())
        result = consume(data)

    assert len(p._stages) == 2
    assert p._stages[0].name == "produce"
    assert p._stages[1].name == "consume"
    # consume's input should reference produce's output
    assert "data" in p._stages[1].input_handles


def test_pipeline_disambiguation():
    """Repeated calls to same function get @N suffix."""
    @stage
    def my_stage(params: StageParams) -> dict:
        return {}

    with Pipeline("test", root=Path("/tmp")) as p:
        my_stage(params=StageParams())
        my_stage(params=StageParams())
        my_stage(params=StageParams())

    assert p._stages[0].name == "my_stage"
    assert p._stages[1].name == "my_stage@1"
    assert p._stages[2].name == "my_stage@2"


def test_pipeline_input():
    @stage
    def consume(data: dict) -> dict:
        return data

    with Pipeline("test", root=Path("/tmp")) as p:
        raw = p.input("raw_data", path="data/raw/input.yaml", python_type=dict)
        consume(raw)

    assert "raw_data" in p._inputs
    assert "data" in p._stages[0].input_handles
```

**Step 3:** Run tests: `uv run pytest packages/pivot/tests/test_compose.py -v`

**Step 4:** Commit: `feat(compose): add Pipeline context manager`

---

### Task 1.4: Bridge — convert to `RegistryStageInfo`

The critical bridge: convert the compositional DAG into `RegistryStageInfo` entries
that the existing engine can execute unchanged. Construct `RegistryStageInfo` directly
— do NOT synthesize fake `Annotated[T, Dep(...)]` annotations.

**Files:**
- Modify: `packages/pivot/src/pivot/compose.py`
- Test: `packages/pivot/tests/test_compose.py`

**Field-by-field mapping from compose internals → `RegistryStageInfo`:**

| RegistryStageInfo field | Source |
|---|---|
| `func` | `node.original_func` — **MUST be the unwrapped module-level function**, not the `@stage` wrapper. The wrapper is a closure and will fail pickling in worker processes. **CRITICAL**: All `@stage` wrappers share identical `__code__` bytecode. If the wrapper is passed, `fingerprint.py`'s `__wrapped__` branch hashes that shared bytecode, giving **every stage the same fingerprint** — silent, catastrophic. Add a defensive assertion: `assert not hasattr(func, "__wrapped__"), "must use original_func, not @stage wrapper"` |
| `name` | `node.name` (e.g. `"difficulty_fetch_baselines"` or `"wrangle_bootstrap@1"`) |
| `deps` | `{param_name: resolved_path}` — for each `node.input_handles`, resolve the handle to its generated file path (from path_map) |
| `deps_paths` | Flat list of all dep paths (from `deps.values()`, flattened for multi-file deps) |
| `outs` | List of `outputs.ExpandedOut` — one per output spec. Construct from `_OutputSpec.format` (which is a `Writer`) and the generated path. |
| `outs_paths` | Flat list of all output paths |
| `params` | `node.params` (the `StageParams` instance, or None) |
| `mutex` | `[]` (or passed through if we add mutex support to `@stage`) |
| `variant` | `None` (variants are handled by caller via loops + disambiguation) |
| `signature` | `inspect.signature(node.original_func)` |
| `fingerprint` | `None` — computed lazily by the engine on first access, same as today |
| `dep_specs` | `{param_name: FuncDepSpec(path=resolved_path, loader=inferred_reader, creates_dep_edge=True)}` — construct a `FuncDepSpec` for each input handle. The `loader` (Reader) is inferred: if the producing stage's output format is a `Loader` (bidirectional), use it as the reader. If it's a write-only `Writer`, we need the corresponding `Reader`. For default formats this is straightforward (DataFrameJSONL is a Loader, YAML is a Loader, etc.) |
| `out_specs` | `{output_key: Out(path=generated_path, loader=spec.format)}` — construct `outputs.Out` or `outputs.Metric` from `_OutputSpec`. Use `outputs.Metric` when `spec.tag` is `_MetricTag`. |
| `params_arg_name` | Name of the parameter with type `StageParams` in the original function signature, or None |
| `state_dir` | `self._root / ".pivot"` (same as old Pipeline) |

**Key implementation detail — `ExpandedOut` construction:**

The engine needs `list[outputs.ExpandedOut]` in `outs`. `ExpandedOut` is a **Protocol**
(not a concrete class) — you must construct the concrete dataclasses that satisfy it:

```python
# Regular output:
outputs.Out(path=generated_path, loader=spec.format, cache=True)

# Metric-tagged output:
outputs.Metric(path=generated_path, loader=spec.format)

# Plot-tagged output:
outputs.Plot(path=generated_path, loader=spec.format)
```

The registry internally casts these to `list[outputs.ExpandedOut]` after path
normalization (see `registry.py` `_expand_outs`). All information needed is
available from `_OutputSpec` + path generation.

**Key implementation detail — input handle path resolution:**

Build a `path_map: dict[tuple[_StageNode | _InputNode, str | None], str]` that maps
`(source, output_key)` → absolute file path. For stage outputs, the path is generated
by `_generate_artifact_path`. For inputs, the path comes from `_InputNode.path` (see
Task 1.3 fix below). When resolving a handle's path, look up
`(handle._source, handle._output_key)` in path_map.

**Step 1:** Implement `Pipeline.build()` with direct `RegistryStageInfo` construction
using the field mapping above.

**Step 2:** Write integration test: create a two-stage pipeline with compose API,
call `.build()`, verify the returned Pipeline has correct `RegistryStageInfo` entries
with all required fields populated.

**Step 3:** Run tests.

**Step 4:** Commit: `feat(compose): bridge compositional Pipeline to RegistryStageInfo`

---

### Task 1.2b: Extension-based format inference for inputs

> **Note:** This must be implemented BEFORE Task 1.3 because `Pipeline.input()`
> calls `_infer_format_from_extension()` when no explicit format is given.

Previously Task 1.5 — moved here to fix dependency ordering.

`p.input()` infers format from file extension when no explicit format is given.

**Files:**
- Modify: `packages/pivot/src/pivot/compose.py`
- Test: `packages/pivot/tests/test_compose.py`

**Implementation** (add to `compose.py`):

```python
def _infer_format_from_extension(path: str) -> Any:
    """Infer loader from file extension for p.input() declarations."""
    loaders = _get_loaders()
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    match ext:
        case "yaml" | "yml":
            return loaders.YAML()
        case "csv":
            return loaders.CSV()
        case "jsonl":
            return loaders.DataFrameJSONL()
        case "json":
            return loaders.JSON()
        case "txt" | "sql" | "jinja":
            return loaders.Text()
        case "pkl" | "pickle":
            return loaders.Pickle()
        case _:
            return loaders.PathOnly()
```

Note: matplotlib.Figure format (PNG) should use lazy import to avoid importing
matplotlib at DAG-build time. Register it conditionally or at first use.

**Step 1:** Add `_infer_format_from_extension`.

**Step 2:** Test: `_infer_format_from_extension("data/foo.yaml")` returns YAML(), etc.

**Step 3:** Commit: `feat(compose): extension-based format inference for inputs`

---

## Phase 2: Validate with difficulty/ Pipeline + Gut Old API

Convert the eval-pipeline's difficulty/ sub-pipeline to the new API and run it
end-to-end. Once validated, immediately rip out the old annotation-based registration
from Pivot's internals. No "Phase 8 cleanup" — the old code dies as soon as it's
proven unnecessary.

### Task 2.1: Convert difficulty/ stage functions

Remove `Dep()`/`Out()`/`Annotated` from the five difficulty stage functions. Add
`@stage` decorator. Keep the internal logic unchanged.

**Files to modify** (in `~/eval-pipeline/pivot/`):
- `eval_pipeline/difficulty/fetch_baselines.py`
- `eval_pipeline/difficulty/patch_baselines.py`
- `eval_pipeline/difficulty/compile_manifests.py`
- `eval_pipeline/difficulty/compute_task_difficulty.py`
- `eval_pipeline/difficulty/compile_human_run_data.py`

**Example** — `fetch_baselines.py` before/after:

```python
# BEFORE
def difficulty_fetch_baselines(
    params: FetchBaselinesParams,
    query_template: Annotated[str, Dep("sql/difficulty/fetch_baselines.sql.jinja", Text())],
) -> Annotated[pd.DataFrame, Out("data/difficulty/raw/baselines/mp4_server.jsonl", DataFrameJSONL())]:
    return _fetch_baselines(query_template_text=query_template, modified_before=params.modified_before)

# AFTER
@stage
def difficulty_fetch_baselines(
    params: FetchBaselinesParams,
    query_template: str,
) -> pd.DataFrame:
    return _fetch_baselines(query_template_text=query_template, modified_before=params.modified_before)
```

**For each file:**
1. Add `from pivot.compose import stage` import
2. Add `@stage` decorator to the stage function
3. Remove `Annotated[..., Dep(...)]` from parameters — use plain types
4. Remove `Annotated[..., Out(...)]` from return — use plain types (or
   `Annotated[T, metric]` for metrics, `Annotated[T, CSV()]` for format overrides)
5. Remove `from pivot.outputs import Dep, Out, Metric` imports
6. Keep TypedDict output classes but simplify their fields

**Key conversions:**

| Current | New |
|---------|-----|
| `Annotated[pd.DataFrame, Dep("...", DataFrameJSONL())]` | `pd.DataFrame` |
| `Annotated[str, Dep("...", Text())]` | `str` |
| `Annotated[dict, Dep("...", YAML())]` | `dict` |
| `Annotated[pd.DataFrame, Out("...", DataFrameJSONL())]` | `pd.DataFrame` |
| `Annotated[dict, Metric("...", YAML())]` | `Annotated[dict, metric]` |
| `Annotated[pd.DataFrame, Out("...", CSV())]` | `Annotated[pd.DataFrame, CSV()]` |

**Step 1:** Convert all five files.

**Step 2:** Verify the functions still work when called directly (e.g., in tests)
by running any existing tests for these modules.

**Step 3:** Commit: `refactor(difficulty): convert stage functions to @stage API`

---

### Task 2.2: Convert difficulty/ pipeline.py

Replace the `Pipeline.register()` calls with compositional `with Pipeline()` block.

**File:** `eval_pipeline/difficulty/pipeline.py`

```python
# BEFORE
pipeline = Pipeline("eval_pipeline_difficulty", root=project.get_project_root())
pipeline.register(fetch_baselines.difficulty_fetch_baselines)
pipeline.register(patch_baselines.difficulty_patch_baselines)
pipeline.register(compile_manifests.difficulty_compile_manifests)
pipeline.register(compute_task_difficulty.difficulty_compute_task_difficulty)
pipeline.register(compile_human_run_data.difficulty_compile_human_run_data)

# AFTER
from pivot.compose import Pipeline as CompositionalPipeline

with CompositionalPipeline("eval_pipeline_difficulty", root=project.get_project_root()) as p:
    # Inputs — path required now, will become optional with .pvt sidecar system later
    query_template = p.input("fetch_baselines_query", path="sql/difficulty/fetch_baselines.sql.jinja")
    estimates = p.input("estimates", path="data/difficulty/raw/estimates.csv")
    starting_scores = p.input("starting_scores", path="data/difficulty/raw/aird_formatted_task_data.csv")
    portbench_baselines = p.input("portbench_baselines", path="data/difficulty/external/portbench_baseline_scores.yaml")

    # Stages — ALWAYS use keyword args to avoid positional binding errors
    # (stage functions may have params as the first parameter)
    raw_baselines = fetch_baselines.difficulty_fetch_baselines(
        query_template=query_template,
        params=FetchBaselinesParams(),
    )
    patched = patch_baselines.difficulty_patch_baselines(
        raw_baselines=raw_baselines,
        params=PatchBaselinesParams(),
    )
    manifests = compile_manifests.difficulty_compile_manifests(
        params=CompileManifestsParams(),
    )
    compute_task_difficulty.difficulty_compute_task_difficulty(
        baselines=patched.baselines,  # multi-output: .baselines field
        manifests=manifests,
        estimates=estimates,
        starting_scores=starting_scores,
        portbench_baselines=portbench_baselines,
        params=ComputeTaskDifficultyParams(),
    )
    compile_human_run_data.difficulty_compile_human_run_data(
        baselines=patched.baselines,
        manifests=manifests,
        estimates=estimates,
    )

pipeline = p.build()  # Bridge to legacy Pipeline for engine
```

**Step 1:** Write the new pipeline.py.

**Step 2:** Run `pivot repro` in the eval-pipeline and verify it builds the DAG.

**Step 3:** Verify stage execution produces the same outputs as before (compare
data files).

**Step 4:** Commit: `refactor(difficulty): convert pipeline to compositional API`

---

### Task 2.3: End-to-end validation

Run the full difficulty/ pipeline and verify outputs match.

**Steps:**
1. `cd ~/eval-pipeline/pivot`
2. `pivot repro difficulty_fetch_baselines difficulty_patch_baselines difficulty_compile_manifests difficulty_compute_task_difficulty difficulty_compile_human_run_data`
3. Diff the output files against the previously-cached versions
4. Fix any discrepancies

### Task 2.4: Rip out old annotation parsing (keep runtime types)

Now that the compositional API is validated, gut the old **annotation-parsing entry
path**. The bridge in Task 1.4 constructs `RegistryStageInfo` directly — it never
calls `extract_stage_definition`. But the engine/worker still USE `Out`, `Metric`,
`ExpandedOut`, etc. at runtime for executing stages — those classes stay.

**What to DELETE** (annotation-parsing entrypoints only):
- `packages/pivot/src/pivot/stage_def.py`: Delete `extract_stage_definition()` and
  all the `Annotated[T, Dep(...)]` / `Annotated[T, Out(...)]` parsing helpers.
  Keep `StageParams`, `SINGLE_OUTPUT_KEY`, `FuncDepSpec`, and any types the
  engine/worker reference.
- `packages/pivot/src/pivot/pipeline/pipeline.py`: Delete `Pipeline.register()` and
  its override-resolution helpers (`_resolve_path`, `_resolve_out_override`, etc.).
  Keep the `Pipeline` class (name, root, state_dir, build_dag, include, etc.).
- `packages/pivot/src/pivot/registry.py`: Delete `_apply_out_overrides`,
  `_validate_overrides`, and all the override machinery. The registry now only
  receives fully-formed `RegistryStageInfo` from `compose.py`.

**What to KEEP** (engine runtime dependencies):
- `packages/pivot/src/pivot/outputs.py`: Keep `Dep`, `Out`, `Metric`, `DirectoryOut`,
  `IncrementalOut`, `ExpandedOut`, `BaseOut`, `PathType` — the engine, worker, and
  CLI code reference these at runtime for type checks, serialization, and cache
  management. These classes are no longer used in user-facing annotations, but
  remain as internal types.

  > NOTE: `Dep`/`Out`/`Metric` staying in `outputs.py` also prevents ImportError in
  > unconverted eval-pipeline modules (base/, horizon/) during the Phase 2→4 gap.
  > Those modules still have `from pivot.outputs import Dep, Out` imports — they
  > won't be used at registration time, but they must not crash on import.

**Step 1:** Delete old annotation-parsing code per the lists above.

**Step 2:** Run Pivot's own test suite. Fix/delete tests that tested the old
`register()` + annotation-parsing flow. Tests for the compose API (from Phase 1)
cover the new path.

**Step 3:** Run eval-pipeline difficulty/ again to confirm nothing broke.

**Step 4:** Verify that `import eval_pipeline.base.pipeline` and
`import eval_pipeline.horizon.pipeline` don't crash (they'll fail to build a
working pipeline, but they should import without ImportError).

**Step 5:** Commit: `refactor!: remove annotation-parsing registration path`

---

## Phase 3: Variants and base/ Pipeline

The base/ pipeline has variant loops (current/legacy) and cross-module stage reuse.
This validates that the compositional model handles real-world variant patterns.

### Task 3.1: Convert base/ stage functions to @stage

Same pattern as Phase 2 — strip Dep/Out annotations, add @stage decorator. The base/
pipeline has ~10 stage functions across multiple modules.

### Task 3.2: Convert base/ pipeline.py

The key challenge: the current pipeline.py has a `for` loop over variants that
registers the same function multiple times with different `dep_path_overrides` and
`out_path_overrides`. In the new model, this becomes a regular loop that passes
different handles — the exact pattern from the design doc.

### Task 3.3: End-to-end validation

Run `pivot repro` for the base/ pipeline and verify outputs match.

### Task 3.4: Delete any remaining old-API scaffolding

If Phase 2's gutting left any dead code paths that only base/ exercised (e.g.,
variant-specific override logic in registry.py), delete them now.

---

## Phase 4: Cross-pipeline Dependencies and horizon/

### Task 4.1: Cross-pipeline handle imports

Validate that importing handles from base/pipeline.py into horizon/pipeline.py
works correctly — the engine resolves cross-pipeline edges.

### Task 4.2: Convert horizon/ stage functions

~40 stage functions across wrangle/, plot/, and top-level modules.

### Task 4.3: Replace horizon/ report builder

The 832-line report.py. This is the biggest payoff — should shrink to ~50 lines of
compositional calls. The entire `ModelReportConfig.build_model_report` static method
and its 240 lines of override wiring become a simple function with `with Pipeline()`.

### Task 4.4: Convert model report pipelines

`model_reports/time_horizon_1_0/` and `model_reports/time_horizon_1_1/`.

### Task 4.5: End-to-end validation

Full pipeline run. Compare all outputs, metrics, and plots.

### Task 4.6: Final old-API sweep

Grep the entire Pivot codebase for any remaining references to `Dep`, `Out`,
`Metric`, `dep_path_overrides`, `out_path_overrides`, `Pipeline.register`. Delete
all dead code. Update AGENTS.md, README, and all docs to reflect the new API.
Delete or rewrite all existing tests that tested old-API code paths.

---

## Phase 5: Storage Layer

### Task 5.1: Content-addressed object store

Add `.pivot/cache/objects/` directory. Store artifacts by content hash.

### Task 5.2: Symlink tree generation

After each run, generate the human-readable symlink tree
(`data/`, `metrics/`, `plots/`).

### Task 5.3: Auto-categorization

Implement the DAG-based categorization: inputs → `data/raw/`,
intermediate → `data/interim/`, leaves → `data/processed/`,
metrics → `metrics/`, plots → `plots/`.

---

## Phase 6: Input System

### Task 6.1: `.pvt` sidecar discovery

Implement scanning `data/raw/**/*.pvt` and `data/external/**/*.pvt` for input
registration.

### Task 6.2: `pivot input add` command

CLI command to register a local file as an input (creates `.pvt` sidecar).

### Task 6.3: `pivot import` command

CLI command to import data from external sources (S3, other repos) into
`data/external/`.

---

## Phase 7: Cached Manifest

### Task 7.1: Manifest generation

After DAG build, serialize to `.pivot/cache/manifest.json`.

### Task 7.2: Manifest reading

Tab-complete and TUI read from manifest instead of importing Python.

### Task 7.3: Invalidation

Track `pipeline.py` + import graph mtimes. Invalidate when changed.
