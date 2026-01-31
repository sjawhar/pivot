# Pipeline Class Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace global `REGISTRY` with a `Pipeline` class that supports multiple isolated pipelines with separate state tracking but shared cache.

**Architecture:** Each `Pipeline` has its own internal registry, state directory (for lock files and state.db), and home directory (for resolving relative paths). Pipelines share a project-wide cache. When Pipeline A includes Pipeline B, B's stages use B's state directory, enabling seamless composition.

**Tech Stack:** Python 3.13+, pytest, pydantic

---

## Background

### Current State
- Global `REGISTRY` singleton in `src/pivot/registry.py`
- All stages share one `state_dir` (`.pivot/`)
- `state_dir` determined at engine runtime in `_orchestrate_execution()`
- `RegistryStageInfo` has no concept of "owner" or "home directory"

### Target State
- `Pipeline` class with internal registry
- Each stage carries its `state_dir` (set at registration time)
- `project_root` remains runtime-determined for shared cache
- `pivot.yaml` creates implicit Pipeline (name from YAML or parent directory)
- Global `REGISTRY` removed (breaking change)

### Key Design Decisions
1. `state_dir` is per-stage (stored in `RegistryStageInfo`), set at registration
2. `project_root` is project-wide (determined at runtime), used for shared cache
3. Pipeline home directory inferred from caller's `__file__`, overridable with `root=`
4. Stage names are isolated per pipeline (two pipelines can both have "train")
5. When Pipeline A includes Pipeline B, B's stages still use B's state_dir

---

## Task 1: Add `state_dir` Field to RegistryStageInfo

**Files:**
- Modify: `src/pivot/registry.py:57-93` (RegistryStageInfo TypedDict)
- Test: `tests/config/test_registry.py`

**Step 1: Write the failing test**

Add to `tests/config/test_registry.py`:

```python
def test_registry_stage_info_has_state_dir() -> None:
    """RegistryStageInfo should include state_dir field."""
    reg = StageRegistry()

    def my_stage() -> None:
        pass

    state_dir = pathlib.Path("/tmp/test_pipeline/.pivot")
    reg.register(my_stage, name="my_stage", state_dir=state_dir)

    info = reg.get("my_stage")
    assert info["state_dir"] == state_dir
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/config/test_registry.py::test_registry_stage_info_has_state_dir -v`
Expected: FAIL with TypeError (unexpected keyword argument 'state_dir')

**Step 3: Add state_dir to RegistryStageInfo**

In `src/pivot/registry.py`, add to `RegistryStageInfo` TypedDict (around line 93):

```python
class RegistryStageInfo(TypedDict):
    # ... existing fields ...
    params_arg_name: str | None
    state_dir: pathlib.Path | None  # Pipeline's state directory, None uses default
```

**Step 4: Update StageRegistry.register() to accept state_dir**

In `src/pivot/registry.py`, update the `register` method signature and body to accept and store `state_dir`:

```python
def register(
    self,
    func: Callable[..., Any],
    *,
    name: str | None = None,
    params: ParamsArg = None,
    mutex: list[str] | None = None,
    variant: str | None = None,
    dep_path_overrides: Mapping[str, outputs.PathType] | None = None,
    out_path_overrides: Mapping[str, OutOverrideInput] | None = None,
    state_dir: pathlib.Path | None = None,  # Add this parameter
) -> None:
```

And in the RegistryStageInfo construction:

```python
stage_info = RegistryStageInfo(
    # ... existing fields ...
    params_arg_name=params_arg_name,
    state_dir=state_dir,
)
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/config/test_registry.py::test_registry_stage_info_has_state_dir -v`
Expected: PASS

**Step 6: Run full test suite to check for regressions**

Run: `uv run pytest tests/config/test_registry.py -v`
Expected: All tests pass (existing tests pass None for state_dir)

**Step 7: Commit**

```bash
jj describe -m "feat(registry): add state_dir field to RegistryStageInfo

Stages can now carry their own state directory, enabling per-pipeline
state isolation. When state_dir is None, the default behavior applies.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Update Worker to Use Stage's state_dir

**Files:**
- Modify: `src/pivot/executor/core.py:287-318` (prepare_worker_info)
- Modify: `src/pivot/engine/engine.py:1338` (where prepare_worker_info is called)
- Test: `tests/execution/test_executor_worker.py`

**Step 1: Write the failing test**

Add to `tests/execution/test_executor_worker.py`:

```python
def test_prepare_worker_info_uses_stage_state_dir(
    set_project_root: pathlib.Path,
) -> None:
    """prepare_worker_info should use stage's state_dir when set."""
    from pivot.executor import core as executor_core
    from pivot import registry, outputs, loaders

    class _Output(TypedDict):
        result: Annotated[pathlib.Path, outputs.Out("result.txt", loaders.PathOnly())]

    def _stage_with_state_dir() -> _Output:
        return {"result": pathlib.Path("result.txt")}

    custom_state_dir = set_project_root / "custom_pipeline" / ".pivot"
    registry.REGISTRY.register(
        _stage_with_state_dir,
        name="stage_with_custom_state",
        state_dir=custom_state_dir,
    )

    stage_info = registry.REGISTRY.get("stage_with_custom_state")
    worker_info = executor_core.prepare_worker_info(
        stage_info=stage_info,
        overrides={},
        checkout_modes=[],
        run_id="test-run",
        force=False,
        no_commit=False,
        no_cache=False,
        project_root=set_project_root,
        default_state_dir=set_project_root / ".pivot",  # Fallback
    )

    assert worker_info["state_dir"] == custom_state_dir
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/execution/test_executor_worker.py::test_prepare_worker_info_uses_stage_state_dir -v`
Expected: FAIL (prepare_worker_info doesn't have default_state_dir param or doesn't check stage_info)

**Step 3: Update prepare_worker_info signature**

In `src/pivot/executor/core.py`, update `prepare_worker_info`:

```python
def prepare_worker_info(
    stage_info: registry.RegistryStageInfo,
    overrides: parameters.ParamsOverrides,
    checkout_modes: list[cache.CheckoutMode],
    run_id: str,
    force: bool,
    no_commit: bool,
    no_cache: bool,
    project_root: pathlib.Path,
    default_state_dir: pathlib.Path,  # Renamed from state_dir
) -> worker.WorkerStageInfo:
    """Prepare worker info for stage execution.

    Uses stage's state_dir if set, otherwise falls back to default_state_dir.
    """
    # Use stage's state_dir if set, otherwise use default
    state_dir = stage_info["state_dir"] or default_state_dir

    return worker.WorkerStageInfo(
        # ... existing fields ...
        state_dir=state_dir,
        # ...
    )
```

**Step 4: Update engine.py call sites**

In `src/pivot/engine/engine.py`, update calls to `prepare_worker_info` to pass `default_state_dir`:

```python
worker_info = executor_core.prepare_worker_info(
    stage_info=stage_info,
    overrides=overrides,
    checkout_modes=checkout_modes,
    run_id=run_id,
    force=force,
    no_commit=no_commit,
    no_cache=no_cache,
    project_root=project_root,
    default_state_dir=state_dir,  # Renamed parameter
)
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/execution/test_executor_worker.py::test_prepare_worker_info_uses_stage_state_dir -v`
Expected: PASS

**Step 6: Run broader test suite**

Run: `uv run pytest tests/execution/ tests/engine/ -v`
Expected: All tests pass

**Step 7: Commit**

```bash
jj describe -m "feat(executor): use stage's state_dir when preparing worker info

prepare_worker_info now checks stage_info['state_dir'] and uses it if set,
falling back to default_state_dir otherwise. This enables per-pipeline
state isolation.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Create Pipeline Class (Core Structure)

**Files:**
- Create: `src/pivot/pipeline/pipeline.py`
- Test: `tests/pipeline/test_pipeline.py`

**Step 1: Create test file and write failing test**

Create `tests/pipeline/test_pipeline.py`:

```python
from __future__ import annotations

import inspect
import pathlib
from typing import Annotated, TypedDict

import pytest

from pivot import loaders, outputs


class _SimpleOutput(TypedDict):
    result: Annotated[pathlib.Path, outputs.Out("result.txt", loaders.PathOnly())]


def _simple_stage() -> _SimpleOutput:
    pathlib.Path("result.txt").write_text("done")
    return {"result": pathlib.Path("result.txt")}


def test_pipeline_creation_with_name() -> None:
    """Pipeline should be creatable with a name."""
    from pivot.pipeline.pipeline import Pipeline

    p = Pipeline("my_pipeline")

    assert p.name == "my_pipeline"


def test_pipeline_infers_root_from_caller() -> None:
    """Pipeline should infer root directory from caller's __file__."""
    from pivot.pipeline.pipeline import Pipeline

    p = Pipeline("test")

    # Should be the directory containing this test file
    expected = pathlib.Path(__file__).parent
    assert p.root == expected


def test_pipeline_accepts_explicit_root() -> None:
    """Pipeline should accept explicit root override."""
    from pivot.pipeline.pipeline import Pipeline

    custom_root = pathlib.Path("/custom/path")
    p = Pipeline("test", root=custom_root)

    assert p.root == custom_root


def test_pipeline_state_dir_derived_from_root() -> None:
    """Pipeline state_dir should be root/.pivot."""
    from pivot.pipeline.pipeline import Pipeline

    custom_root = pathlib.Path("/custom/path")
    p = Pipeline("test", root=custom_root)

    assert p.state_dir == custom_root / ".pivot"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pipeline/test_pipeline.py -v`
Expected: FAIL with ModuleNotFoundError (pivot.pipeline.pipeline doesn't exist)

**Step 3: Create Pipeline class**

Create `src/pivot/pipeline/pipeline.py`:

```python
from __future__ import annotations

import inspect
import pathlib


class Pipeline:
    """A pipeline with its own stage registry and state directory.

    Each pipeline maintains isolated state (lock files, state.db) while
    sharing the project-wide cache.

    Args:
        name: Pipeline identifier for logging and display.
        root: Home directory for this pipeline. Defaults to the directory
            containing the file where Pipeline() is called.
    """

    def __init__(
        self,
        name: str,
        *,
        root: pathlib.Path | None = None,
    ) -> None:
        self._name = name

        if root is not None:
            self._root = root
        else:
            # Infer from caller's __file__
            frame = inspect.currentframe()
            if frame is None or frame.f_back is None:
                raise RuntimeError("Cannot determine caller frame")
            caller_file = frame.f_back.f_globals.get("__file__")
            if caller_file is None:
                raise RuntimeError("Cannot determine caller's __file__")
            self._root = pathlib.Path(caller_file).parent

    @property
    def name(self) -> str:
        """Pipeline name."""
        return self._name

    @property
    def root(self) -> pathlib.Path:
        """Pipeline root directory."""
        return self._root

    @property
    def state_dir(self) -> pathlib.Path:
        """State directory for this pipeline's lock files and state.db."""
        return self._root / ".pivot"
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pipeline/test_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(pipeline): add Pipeline class with name and root

Pipeline class provides isolated state directories for multi-pipeline
support. Root is inferred from caller's __file__ or can be explicitly set.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Add Pipeline.register() Method

**Files:**
- Modify: `src/pivot/pipeline/pipeline.py`
- Test: `tests/pipeline/test_pipeline.py`

**Step 1: Write failing test**

Add to `tests/pipeline/test_pipeline.py`:

```python
def test_pipeline_register_stage(tmp_path: pathlib.Path) -> None:
    """Pipeline.register should register a stage with the pipeline's state_dir."""
    from pivot.pipeline.pipeline import Pipeline

    p = Pipeline("test", root=tmp_path)
    p.register(_simple_stage, name="my_stage")

    assert "my_stage" in p.list_stages()
    info = p.get("my_stage")
    assert info["state_dir"] == tmp_path / ".pivot"


def test_pipeline_stages_isolated(tmp_path: pathlib.Path) -> None:
    """Two pipelines can have stages with the same name."""
    from pivot.pipeline.pipeline import Pipeline

    p1 = Pipeline("pipeline1", root=tmp_path / "p1")
    p2 = Pipeline("pipeline2", root=tmp_path / "p2")

    p1.register(_simple_stage, name="train")
    p2.register(_simple_stage, name="train")

    assert "train" in p1.list_stages()
    assert "train" in p2.list_stages()

    # Each has its own state_dir
    assert p1.get("train")["state_dir"] == tmp_path / "p1" / ".pivot"
    assert p2.get("train")["state_dir"] == tmp_path / "p2" / ".pivot"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pipeline/test_pipeline.py::test_pipeline_register_stage -v`
Expected: FAIL (Pipeline has no register method)

**Step 3: Add register, list_stages, and get methods**

Update `src/pivot/pipeline/pipeline.py`:

```python
from __future__ import annotations

import inspect
import pathlib
from typing import TYPE_CHECKING, Any

from pivot import registry

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pivot import outputs, stage_def


class Pipeline:
    """A pipeline with its own stage registry and state directory."""

    def __init__(
        self,
        name: str,
        *,
        root: pathlib.Path | None = None,
    ) -> None:
        self._name = name

        if root is not None:
            self._root = root
        else:
            frame = inspect.currentframe()
            if frame is None or frame.f_back is None:
                raise RuntimeError("Cannot determine caller frame")
            caller_file = frame.f_back.f_globals.get("__file__")
            if caller_file is None:
                raise RuntimeError("Cannot determine caller's __file__")
            self._root = pathlib.Path(caller_file).parent

        self._registry = registry.StageRegistry()

    @property
    def name(self) -> str:
        return self._name

    @property
    def root(self) -> pathlib.Path:
        return self._root

    @property
    def state_dir(self) -> pathlib.Path:
        return self._root / ".pivot"

    def register(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        params: registry.ParamsArg = None,
        mutex: list[str] | None = None,
        variant: str | None = None,
        dep_path_overrides: Mapping[str, outputs.PathType] | None = None,
        out_path_overrides: Mapping[str, registry.OutOverrideInput] | None = None,
    ) -> None:
        """Register a stage with this pipeline.

        The stage will use this pipeline's state_dir for lock files and state.db.
        """
        self._registry.register(
            func=func,
            name=name,
            params=params,
            mutex=mutex,
            variant=variant,
            dep_path_overrides=dep_path_overrides,
            out_path_overrides=out_path_overrides,
            state_dir=self.state_dir,
        )

    def list_stages(self) -> list[str]:
        """List all registered stage names."""
        return self._registry.list_stages()

    def get(self, name: str) -> registry.RegistryStageInfo:
        """Get stage info by name."""
        return self._registry.get(name)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(pipeline): add register, list_stages, and get methods

Pipeline now wraps StageRegistry and automatically sets state_dir on
registered stages. Each pipeline maintains isolated state.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Add Pipeline Discovery

**Files:**
- Modify: `src/pivot/discovery.py`
- Test: `tests/core/test_discovery.py`

**Step 1: Write failing test**

Add to `tests/core/test_discovery.py`:

```python
def test_discover_pipeline_from_pipeline_py(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """discover_pipeline should find Pipeline instance in pipeline.py."""
    from pivot import discovery
    from pivot.pipeline.pipeline import Pipeline

    mocker.patch.object(project, "_project_root_cache", tmp_path)

    # Create pipeline.py that defines a Pipeline
    pipeline_code = '''
from pivot.pipeline.pipeline import Pipeline

pipeline = Pipeline("test_pipeline")

def _stage():
    pass

pipeline.register(_stage, name="my_stage")
'''
    (tmp_path / "pipeline.py").write_text(pipeline_code)

    result = discovery.discover_pipeline(tmp_path)

    assert result is not None
    assert result.name == "test_pipeline"
    assert "my_stage" in result.list_stages()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_discovery.py::test_discover_pipeline_from_pipeline_py -v`
Expected: FAIL (discover_pipeline doesn't exist)

**Step 3: Add discover_pipeline function**

Update `src/pivot/discovery.py`:

```python
from __future__ import annotations

import logging
import runpy
from typing import TYPE_CHECKING

from pivot import fingerprint, metrics, project, registry
from pivot.pipeline import yaml as pipeline_config

if TYPE_CHECKING:
    from pathlib import Path

    from pivot.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)

PIVOT_YAML_NAMES = ("pivot.yaml", "pivot.yml")
PIPELINE_PY_NAME = "pipeline.py"


def discover_pipeline(project_root: Path | None = None) -> Pipeline | None:
    """Discover and return Pipeline from pivot.yaml or pipeline.py.

    Looks in project root for:
    1. pivot.yaml (or pivot.yml) - creates implicit Pipeline
    2. pipeline.py - looks for `pipeline` variable (Pipeline instance)

    Args:
        project_root: Override project root (default: auto-detect)

    Returns:
        Pipeline instance, or None if nothing found

    Raises:
        DiscoveryError: If discovery fails, or if both config types exist
    """
    from pivot.pipeline.pipeline import Pipeline

    with metrics.timed("discovery.total"):
        root = project_root or project.get_project_root()

        # Check which files exist upfront
        yaml_path = None
        for yaml_name in PIVOT_YAML_NAMES:
            candidate = root / yaml_name
            if candidate.exists():
                yaml_path = candidate
                break

        pipeline_path = root / PIPELINE_PY_NAME
        pipeline_exists = pipeline_path.exists()

        # Error if both exist
        if yaml_path and pipeline_exists:
            raise DiscoveryError(
                f"Found both {yaml_path.name} and {PIPELINE_PY_NAME} in {root}. "
                "Remove one to resolve ambiguity."
            )

        # Load from yaml if found
        if yaml_path:
            logger.info(f"Discovered {yaml_path}")
            try:
                return pipeline_config.load_pipeline_from_yaml(yaml_path)
            except pipeline_config.PipelineConfigError as e:
                raise DiscoveryError(f"Failed to load {yaml_path}: {e}") from e

        # Try pipeline.py
        if pipeline_exists:
            logger.info(f"Discovered {pipeline_path}")
            try:
                return _load_pipeline_from_module(pipeline_path)
            except SystemExit as e:
                raise DiscoveryError(
                    f"Pipeline {pipeline_path} called sys.exit({e.code})"
                ) from e
            except Exception as e:
                raise DiscoveryError(f"Failed to load {pipeline_path}: {e}") from e
            finally:
                fingerprint.flush_ast_hash_cache()

        return None


def _load_pipeline_from_module(path: Path) -> Pipeline:
    """Load Pipeline instance from a pipeline.py file."""
    from pivot.pipeline.pipeline import Pipeline

    module_dict = runpy.run_path(str(path), run_name="_pivot_pipeline")

    # Look for 'pipeline' variable
    pipeline = module_dict.get("pipeline")
    if pipeline is None:
        raise DiscoveryError(
            f"{path} does not define a 'pipeline' variable. "
            "Expected: pipeline = Pipeline('name')"
        )

    if not isinstance(pipeline, Pipeline):
        raise DiscoveryError(
            f"{path} defines 'pipeline' but it's not a Pipeline instance "
            f"(got {type(pipeline).__name__})"
        )

    return pipeline


# Keep legacy function for backwards compatibility during transition
def discover_and_register(project_root: Path | None = None) -> str | None:
    """Legacy: Discover and register pipeline into global REGISTRY.

    DEPRECATED: Use discover_pipeline() instead.
    """
    # ... existing implementation ...
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/test_discovery.py::test_discover_pipeline_from_pipeline_py -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(discovery): add discover_pipeline returning Pipeline instance

New discover_pipeline() function returns Pipeline directly instead of
registering into global REGISTRY. Legacy discover_and_register() preserved
for backwards compatibility.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Update pivot.yaml to Create Pipeline

**Files:**
- Modify: `src/pivot/pipeline/yaml.py`
- Test: `tests/core/test_pipeline_config.py`

**Step 1: Write failing test**

Add to `tests/core/test_pipeline_config.py`:

```python
def test_load_pipeline_from_yaml_creates_pipeline(tmp_path: pathlib.Path) -> None:
    """load_pipeline_from_yaml should return a Pipeline instance."""
    from pivot.pipeline import yaml as pipeline_yaml
    from pivot.pipeline.pipeline import Pipeline

    yaml_content = """
stages:
  process:
    python: tests.helpers._simple_stage
"""
    yaml_path = tmp_path / "pivot.yaml"
    yaml_path.write_text(yaml_content)

    result = pipeline_yaml.load_pipeline_from_yaml(yaml_path)

    assert isinstance(result, Pipeline)
    assert result.root == tmp_path


def test_load_pipeline_from_yaml_uses_pipeline_name(tmp_path: pathlib.Path) -> None:
    """load_pipeline_from_yaml should use 'pipeline' field for name if present."""
    from pivot.pipeline import yaml as pipeline_yaml

    yaml_content = """
pipeline: my_custom_name
stages:
  process:
    python: tests.helpers._simple_stage
"""
    yaml_path = tmp_path / "pivot.yaml"
    yaml_path.write_text(yaml_content)

    result = pipeline_yaml.load_pipeline_from_yaml(yaml_path)

    assert result.name == "my_custom_name"


def test_load_pipeline_from_yaml_defaults_to_directory_name(tmp_path: pathlib.Path) -> None:
    """load_pipeline_from_yaml should default name to parent directory."""
    from pivot.pipeline import yaml as pipeline_yaml

    yaml_content = """
stages:
  process:
    python: tests.helpers._simple_stage
"""
    subdir = tmp_path / "my_project"
    subdir.mkdir()
    yaml_path = subdir / "pivot.yaml"
    yaml_path.write_text(yaml_content)

    result = pipeline_yaml.load_pipeline_from_yaml(yaml_path)

    assert result.name == "my_project"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_pipeline_config.py::test_load_pipeline_from_yaml_creates_pipeline -v`
Expected: FAIL (load_pipeline_from_yaml doesn't exist)

**Step 3: Add load_pipeline_from_yaml and update PipelineConfig**

Update `src/pivot/pipeline/yaml.py`:

```python
class PipelineConfig(pydantic.BaseModel):
    """Top-level pivot.yaml configuration."""

    model_config = pydantic.ConfigDict(extra="forbid")

    pipeline: str | None = None  # Add optional pipeline name
    stages: dict[str, StageConfig]
    vars: list[str] = []


def load_pipeline_from_yaml(pipeline_file: Path) -> Pipeline:
    """Load pivot.yaml and return a Pipeline instance.

    The pipeline name comes from:
    1. 'pipeline' field in YAML if present
    2. Otherwise, parent directory name
    """
    from pivot.pipeline.pipeline import Pipeline

    config = load_pipeline_file(pipeline_file)
    pipeline_dir = pipeline_file.parent

    # Determine pipeline name
    name = config.pipeline or pipeline_dir.name

    # Create Pipeline with explicit root
    p = Pipeline(name, root=pipeline_dir)

    # Register all stages
    for stage_name, stage_config in config.stages.items():
        expanded = _expand_stage(stage_name, stage_config, pipeline_dir, config.vars)
        for stage in expanded:
            p.register(
                func=stage.func,
                name=stage.name,
                params=stage.params,
                mutex=stage.mutex,
                variant=stage.variant,
                dep_path_overrides=stage.dep_path_overrides,
                out_path_overrides=stage.out_path_overrides,
            )

    return p
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_pipeline_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(yaml): add load_pipeline_from_yaml returning Pipeline

pivot.yaml now creates a Pipeline instance with stages registered.
Pipeline name comes from 'pipeline' field or defaults to directory name.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Update Engine to Accept Pipeline

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Test: `tests/engine/test_engine.py`

**Step 1: Write failing test**

Add to `tests/engine/test_engine.py`:

```python
def test_engine_accepts_pipeline(tmp_path: pathlib.Path) -> None:
    """Engine should accept a Pipeline instance."""
    from pivot.engine.engine import Engine
    from pivot.pipeline.pipeline import Pipeline

    p = Pipeline("test", root=tmp_path)

    def _stage() -> None:
        pass

    p.register(_stage, name="my_stage")

    with Engine(pipeline=p) as engine:
        assert "my_stage" in engine._graph.nodes()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_engine.py::test_engine_accepts_pipeline -v`
Expected: FAIL (Engine doesn't accept pipeline parameter)

**Step 3: Update Engine.__init__ to accept Pipeline**

In `src/pivot/engine/engine.py`, update the `Engine` class:

```python
class Engine:
    """Pipeline execution engine."""

    def __init__(
        self,
        *,
        pipeline: Pipeline | None = None,
        # ... existing parameters ...
    ) -> None:
        self._pipeline = pipeline
        # ... rest of __init__ ...

    def _build_graph(self) -> DiGraph:
        """Build execution graph from registered stages."""
        if self._pipeline is not None:
            all_stages = {
                name: self._pipeline.get(name)
                for name in self._pipeline.list_stages()
            }
        else:
            # Legacy: use global REGISTRY
            all_stages = {
                name: registry.REGISTRY.get(name)
                for name in registry.REGISTRY.list_stages()
            }
        return engine_graph.build_graph(all_stages)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/engine/test_engine.py::test_engine_accepts_pipeline -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(engine): accept Pipeline instance

Engine can now receive a Pipeline directly instead of reading from
global REGISTRY. Legacy behavior (reading REGISTRY) preserved when
pipeline=None.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Update CLI to Use discover_pipeline

**Files:**
- Modify: `src/pivot/cli/run.py`
- Test: `tests/cli/test_run.py`

**Step 1: Write failing test**

Add to `tests/cli/test_run.py`:

```python
def test_cli_run_discovers_pipeline(
    tmp_path: pathlib.Path,
    runner: CliRunner,
    mocker: MockerFixture,
) -> None:
    """pivot run should discover and use Pipeline from pipeline.py."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)

    # Create pipeline.py
    pipeline_code = '''
from pivot.pipeline.pipeline import Pipeline

pipeline = Pipeline("test")

def _stage():
    pass

pipeline.register(_stage, name="my_stage")
'''
    (tmp_path / "pipeline.py").write_text(pipeline_code)
    (tmp_path / ".pivot").mkdir()

    result = runner.invoke(cli, ["run", "--dry-run"])

    assert result.exit_code == 0
    assert "my_stage" in result.output
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_run.py::test_cli_run_discovers_pipeline -v`
Expected: FAIL (CLI still uses global REGISTRY)

**Step 3: Update CLI run command**

In `src/pivot/cli/run.py`, update to use `discover_pipeline`:

```python
@cli.command()
def run(...):
    """Run pipeline stages."""
    from pivot import discovery

    # Discover pipeline
    pipeline = discovery.discover_pipeline()
    if pipeline is None:
        raise click.ClickException("No pipeline found (pivot.yaml or pipeline.py)")

    # Create engine with pipeline
    with Engine(pipeline=pipeline, ...) as engine:
        # ... rest of run logic ...
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_run.py::test_cli_run_discovers_pipeline -v`
Expected: PASS

**Step 5: Run full CLI test suite**

Run: `uv run pytest tests/cli/test_run.py -v`
Expected: All tests pass

**Step 6: Commit**

```bash
jj describe -m "feat(cli): use discover_pipeline in run command

CLI now discovers Pipeline instance and passes it to Engine, instead
of relying on global REGISTRY side effects.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Remove Global REGISTRY Usage (Breaking Change)

**Files:**
- Modify: `src/pivot/registry.py` (remove REGISTRY singleton)
- Modify: Multiple files that import REGISTRY
- Test: Ensure all tests pass without global REGISTRY

**Step 1: Identify all REGISTRY imports**

Run: `uv run rg "from pivot.registry import REGISTRY|from pivot import.*registry.*REGISTRY" --files-with-matches`

**Step 2: Update each file to use Pipeline instead**

This is a larger refactoring task. For each file:
1. Remove REGISTRY import
2. Accept Pipeline as parameter or use discover_pipeline()

**Step 3: Mark REGISTRY as deprecated (optional intermediate step)**

```python
# In src/pivot/registry.py
import warnings

class _DeprecatedRegistry(StageRegistry):
    def register(self, *args, **kwargs):
        warnings.warn(
            "REGISTRY is deprecated. Use Pipeline().register() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().register(*args, **kwargs)

REGISTRY = _DeprecatedRegistry()
```

**Step 4: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass (may show deprecation warnings)

**Step 5: Commit**

```bash
jj describe -m "refactor!: deprecate global REGISTRY

BREAKING CHANGE: Global REGISTRY is deprecated. Use Pipeline class instead.
All internal usage updated to use Pipeline. REGISTRY kept for backwards
compatibility with deprecation warning.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Integration Test - Multi-Pipeline Execution

**Files:**
- Test: `tests/integration/test_multi_pipeline.py`

**Step 1: Write integration test**

Create `tests/integration/test_multi_pipeline.py`:

```python
from __future__ import annotations

import pathlib
from typing import Annotated, TypedDict

import pytest

from pivot import loaders, outputs
from pivot.engine.engine import Engine
from pivot.pipeline.pipeline import Pipeline


class _DataOutput(TypedDict):
    data: Annotated[pathlib.Path, outputs.Out("data.csv", loaders.PathOnly())]


def _produce_data() -> _DataOutput:
    pathlib.Path("data.csv").write_text("id,value\n1,10\n")
    return {"data": pathlib.Path("data.csv")}


class _ResultOutput(TypedDict):
    result: Annotated[pathlib.Path, outputs.Out("result.txt", loaders.PathOnly())]


def _consume_data(
    data: Annotated[pathlib.Path, outputs.Dep("data.csv", loaders.PathOnly())],
) -> _ResultOutput:
    pathlib.Path("result.txt").write_text("processed")
    return {"result": pathlib.Path("result.txt")}


def test_two_pipelines_isolated_state(tmp_path: pathlib.Path) -> None:
    """Two pipelines should have isolated state directories."""
    # Create two pipeline directories
    p1_dir = tmp_path / "pipeline1"
    p2_dir = tmp_path / "pipeline2"
    p1_dir.mkdir()
    p2_dir.mkdir()

    # Create pipelines
    p1 = Pipeline("producer", root=p1_dir)
    p2 = Pipeline("consumer", root=p2_dir)

    p1.register(_produce_data, name="produce")
    p2.register(_consume_data, name="consume")

    # Verify isolated state_dirs
    assert p1.get("produce")["state_dir"] == p1_dir / ".pivot"
    assert p2.get("consume")["state_dir"] == p2_dir / ".pivot"

    # Run pipeline 1
    with Engine(pipeline=p1) as engine:
        engine.run_once()

    # Verify state files created in p1's state_dir
    assert (p1_dir / ".pivot" / "stages" / "produce.lock").exists()
    assert not (p2_dir / ".pivot" / "stages" / "produce.lock").exists()


def test_same_stage_name_different_pipelines(tmp_path: pathlib.Path) -> None:
    """Two pipelines can have stages with the same name."""
    p1 = Pipeline("pipeline1", root=tmp_path / "p1")
    p2 = Pipeline("pipeline2", root=tmp_path / "p2")

    def _train_v1() -> None:
        pass

    def _train_v2() -> None:
        pass

    # Both register "train" - should not conflict
    p1.register(_train_v1, name="train")
    p2.register(_train_v2, name="train")

    assert p1.get("train")["func"] == _train_v1
    assert p2.get("train")["func"] == _train_v2
```

**Step 2: Run integration tests**

Run: `uv run pytest tests/integration/test_multi_pipeline.py -v`
Expected: PASS

**Step 3: Commit**

```bash
jj describe -m "test: add integration tests for multi-pipeline isolation

Tests verify that two pipelines maintain isolated state directories
and can have stages with the same name without conflict.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 11: Final Verification and Quality Checks

**Step 1: Run type checker**

Run: `uv run basedpyright .`
Expected: No errors or warnings

**Step 2: Run linter**

Run: `uv run ruff check .`
Expected: No errors

**Step 3: Run formatter**

Run: `uv run ruff format .`
Expected: Files formatted

**Step 4: Run full test suite with coverage**

Run: `uv run pytest tests/ -v --cov=pivot --cov-report=term-missing`
Expected: All tests pass, coverage >= 90%

**Step 5: Final commit**

```bash
jj describe -m "feat(pipeline): complete Pipeline class implementation

Adds Pipeline class for multi-pipeline support with:
- Isolated state directories per pipeline
- Shared project-wide cache
- Per-stage state_dir in RegistryStageInfo
- Updated discovery to return Pipeline instances
- Updated Engine to accept Pipeline
- Updated CLI to use discover_pipeline

BREAKING CHANGE: Global REGISTRY is deprecated in favor of Pipeline class.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add state_dir to RegistryStageInfo | registry.py |
| 2 | Update worker to use stage's state_dir | core.py, engine.py |
| 3 | Create Pipeline class core | pipeline/pipeline.py |
| 4 | Add Pipeline.register() | pipeline/pipeline.py |
| 5 | Add Pipeline discovery | discovery.py |
| 6 | Update pivot.yaml to create Pipeline | pipeline/yaml.py |
| 7 | Update Engine to accept Pipeline | engine/engine.py |
| 8 | Update CLI to use discover_pipeline | cli/run.py |
| 9 | Deprecate global REGISTRY | registry.py, multiple |
| 10 | Integration tests | test_multi_pipeline.py |
| 11 | Final verification | - |

## Deferred Work (Post-MVP)

- Pipeline composition (Pipeline A includes Pipeline B)
- Cross-pipeline dependency validation
- Shared cache directory discovery (walk up to find `.pivot/`)
- `PIVOT_PROJECT_ROOT` environment variable override
