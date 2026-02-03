# Lazy Pipeline Dependency Resolution

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable pipelines to automatically discover and include stages from parent pipelines when dependencies are unresolved locally.

**Architecture:** Add `resolve_from_parents()` method to Pipeline that traverses up the directory tree, loads parent pipelines, and includes stages that produce needed artifacts. Uses iterative work-queue algorithm with per-call caching (parent pipelines cached within single resolve call, discarded after).

**Tech Stack:** Python 3.13+, networkx, pytest

---

## Task 1: Add Parent Pipeline Path Discovery to discovery.py

**Files:**
- Modify: `src/pivot/discovery.py`
- Test: `tests/test_discovery.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_discovery.py
import pathlib

from pivot import discovery


def test_find_parent_pipeline_paths_finds_pipeline_py(tmp_path: pathlib.Path) -> None:
    """Should find pipeline.py in parent directories, closest first."""
    (tmp_path / "pipeline.py").touch()
    mid = tmp_path / "mid"
    mid.mkdir()
    (mid / "pipeline.py").touch()
    child = mid / "child"
    child.mkdir()

    result = list(discovery.find_parent_pipeline_paths(child, stop_at=tmp_path))

    assert result == [mid / "pipeline.py", tmp_path / "pipeline.py"]


def test_find_parent_pipeline_paths_finds_pivot_yaml(tmp_path: pathlib.Path) -> None:
    """Should find pivot.yaml in parent directories."""
    (tmp_path / "pivot.yaml").touch()
    child = tmp_path / "child"
    child.mkdir()

    result = list(discovery.find_parent_pipeline_paths(child, stop_at=tmp_path))

    assert result == [tmp_path / "pivot.yaml"]


def test_find_parent_pipeline_paths_errors_on_both(tmp_path: pathlib.Path) -> None:
    """Should error if directory has both pipeline.py and pivot.yaml."""
    (tmp_path / "pipeline.py").touch()
    (tmp_path / "pivot.yaml").touch()
    child = tmp_path / "child"
    child.mkdir()

    with pytest.raises(discovery.DiscoveryError, match="Found both"):
        list(discovery.find_parent_pipeline_paths(child, stop_at=tmp_path))


def test_find_parent_pipeline_paths_skips_own_directory(tmp_path: pathlib.Path) -> None:
    """Should not include start directory's own pipeline file."""
    (tmp_path / "pipeline.py").touch()

    result = list(discovery.find_parent_pipeline_paths(tmp_path, stop_at=tmp_path.parent))

    assert result == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_discovery.py::test_find_parent_pipeline_paths_finds_pipeline_py -v`
Expected: FAIL with "AttributeError"

**Step 3: Write minimal implementation**

```python
# Add to src/pivot/discovery.py after existing imports
from collections.abc import Iterator


def find_parent_pipeline_paths(
    start_dir: Path,
    stop_at: Path,
) -> Iterator[Path]:
    """Find pipeline config files in parent directories.

    Traverses up from start_dir (exclusive) to stop_at (inclusive),
    yielding each pivot.yaml or pipeline.py found. Closest parents first.
    Errors if any directory has both.

    Args:
        start_dir: Directory to start from (its config is NOT included).
        stop_at: Stop traversal at this directory (inclusive).

    Yields:
        Paths to pivot.yaml or pipeline.py files.

    Raises:
        DiscoveryError: If a directory has both pivot.yaml and pipeline.py.
    """
    import pathlib

    current = pathlib.Path(start_dir).resolve().parent
    stop_at_resolved = pathlib.Path(stop_at).resolve()

    while True:
        # Check for config files (same logic as discover_pipeline)
        yaml_path = None
        for yaml_name in PIVOT_YAML_NAMES:
            candidate = current / yaml_name
            if candidate.exists():
                yaml_path = candidate
                break

        pipeline_path = current / PIPELINE_PY_NAME
        pipeline_exists = pipeline_path.exists()

        if yaml_path and pipeline_exists:
            raise DiscoveryError(
                f"Found both {yaml_path.name} and {PIPELINE_PY_NAME} in {current}. "
                "Remove one to resolve ambiguity."
            )

        if yaml_path:
            yield yaml_path
        elif pipeline_exists:
            yield pipeline_path

        if current == stop_at_resolved or current.parent == current:
            break
        current = current.parent
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_discovery.py::test_find_parent_pipeline_paths_finds_pipeline_py tests/test_discovery.py::test_find_parent_pipeline_paths_finds_pivot_yaml tests/test_discovery.py::test_find_parent_pipeline_paths_errors_on_both tests/test_discovery.py::test_find_parent_pipeline_paths_skips_own_directory -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(discovery): add find_parent_pipeline_paths"
```

---

## Task 2: Make load_pipeline_from_module Public

**Files:**
- Modify: `src/pivot/discovery.py`
- Test: `tests/test_discovery.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_discovery.py
from pytest_mock import MockerFixture

from pivot import project


def test_load_pipeline_from_path_loads_pipeline_py(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Should load Pipeline from pipeline.py file."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    pipeline_code = '''
from pivot.pipeline import Pipeline
pipeline = Pipeline("test")
'''
    (tmp_path / "pipeline.py").write_text(pipeline_code)

    result = discovery.load_pipeline_from_path(tmp_path / "pipeline.py")

    assert result is not None
    assert result.name == "test"


def test_load_pipeline_from_path_loads_pivot_yaml(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Should load Pipeline from pivot.yaml file."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    yaml_content = '''
stages:
  - name: example
    cmd: echo hello
'''
    (tmp_path / "pivot.yaml").write_text(yaml_content)

    result = discovery.load_pipeline_from_path(tmp_path / "pivot.yaml")

    assert result is not None


def test_load_pipeline_from_path_returns_none_for_no_pipeline(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Should return None if file doesn't define a pipeline."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "pipeline.py").write_text("x = 1\n")

    result = discovery.load_pipeline_from_path(tmp_path / "pipeline.py")

    assert result is None


def test_load_pipeline_from_path_logs_errors(
    tmp_path: pathlib.Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    """Should log errors when loading fails."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "pipeline.py").write_text("raise RuntimeError('fail')")

    import logging
    with caplog.at_level(logging.DEBUG):
        result = discovery.load_pipeline_from_path(tmp_path / "pipeline.py")

    assert result is None
    assert "Failed to load" in caplog.text
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_discovery.py::test_load_pipeline_from_path_loads_pipeline_py -v`
Expected: FAIL with "AttributeError"

**Step 3: Write implementation**

```python
# Add to src/pivot/discovery.py

def load_pipeline_from_path(path: Path) -> Pipeline | None:
    """Load a Pipeline from a pivot.yaml or pipeline.py file.

    Args:
        path: Path to pivot.yaml or pipeline.py file.

    Returns:
        Pipeline instance, or None if file doesn't define one.
        Returns None (with debug log) on load errors.
    """
    import pathlib

    path = pathlib.Path(path)

    # Determine file type and load accordingly
    if path.name in PIVOT_YAML_NAMES:
        try:
            return pipeline_config.load_pipeline_from_yaml(path)
        except Exception as e:
            logger.debug(f"Failed to load pipeline from {path}: {e}")
            return None
    elif path.name == PIPELINE_PY_NAME:
        try:
            return _load_pipeline_from_module(path)
        except DiscoveryError:
            # DiscoveryError means file exists but no valid pipeline
            return None
        except Exception as e:
            logger.debug(f"Failed to load pipeline from {path}: {e}")
            return None
    else:
        logger.debug(f"Unknown pipeline file type: {path}")
        return None
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_discovery.py::test_load_pipeline_from_path_loads_pipeline_py tests/test_discovery.py::test_load_pipeline_from_path_loads_pivot_yaml tests/test_discovery.py::test_load_pipeline_from_path_returns_none_for_no_pipeline tests/test_discovery.py::test_load_pipeline_from_path_logs_errors -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(discovery): add load_pipeline_from_path for lazy resolution"
```

---

## Task 3: Add resolve_from_parents Method to Pipeline

**Files:**
- Modify: `src/pivot/pipeline/pipeline.py`
- Test: `tests/pipeline/test_pipeline.py`

**Step 1: Write the failing test for basic resolution**

```python
# Add to tests/pipeline/test_pipeline.py
from typing import Annotated, TypedDict
from pathlib import Path

from pivot import loaders
from pivot.outputs import Out, Dep


def test_pipeline_resolve_from_parents_includes_producer(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Should include producer stage from parent pipeline."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    # Parent pipeline at project root
    parent_code = '''
from typing import Annotated, TypedDict
from pathlib import Path
from pivot.pipeline import Pipeline
from pivot import loaders
from pivot.outputs import Out

pipeline = Pipeline("parent")

class ProducerOutput(TypedDict):
    data: Annotated[Path, Out("shared/data.txt", loaders.PathOnly())]

def producer() -> ProducerOutput:
    Path("shared").mkdir(exist_ok=True)
    Path("shared/data.txt").write_text("data")
    return ProducerOutput(data=Path("shared/data.txt"))

pipeline.register(producer)
'''
    (tmp_path / "pipeline.py").write_text(parent_code)

    # Child pipeline in subdirectory
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    child = Pipeline("child", root=child_dir)

    class ConsumerOutput(TypedDict):
        result: Annotated[Path, Out("result.txt", loaders.PathOnly())]

    def consumer(
        data: Annotated[Path, Dep("shared/data.txt", loaders.PathOnly())]
    ) -> ConsumerOutput:
        return ConsumerOutput(result=Path("result.txt"))

    child.register(consumer)

    # Resolve should find producer in parent
    child.resolve_from_parents()

    assert "producer" in child.list_stages()
    assert "consumer" in child.list_stages()


def test_pipeline_resolve_from_parents_includes_transitive_deps(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Should include transitive dependencies from parent pipeline."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    # Parent pipeline with chain: stage_a -> stage_b
    parent_code = '''
from typing import Annotated, TypedDict
from pathlib import Path
from pivot.pipeline import Pipeline
from pivot import loaders
from pivot.outputs import Out, Dep

pipeline = Pipeline("parent")

class StageAOutput(TypedDict):
    a: Annotated[Path, Out("a.txt", loaders.PathOnly())]

def stage_a() -> StageAOutput:
    return StageAOutput(a=Path("a.txt"))

class StageBOutput(TypedDict):
    b: Annotated[Path, Out("b.txt", loaders.PathOnly())]

def stage_b(a: Annotated[Path, Dep("a.txt", loaders.PathOnly())]) -> StageBOutput:
    return StageBOutput(b=Path("b.txt"))

pipeline.register(stage_a)
pipeline.register(stage_b)
'''
    (tmp_path / "pipeline.py").write_text(parent_code)

    # Child depends on b.txt (which depends on a.txt)
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    child = Pipeline("child", root=child_dir)

    class ConsumerOutput(TypedDict):
        c: Annotated[Path, Out("c.txt", loaders.PathOnly())]

    def consumer(
        b: Annotated[Path, Dep("b.txt", loaders.PathOnly())]
    ) -> ConsumerOutput:
        return ConsumerOutput(c=Path("c.txt"))

    child.register(consumer)

    child.resolve_from_parents()

    assert "stage_a" in child.list_stages()
    assert "stage_b" in child.list_stages()
    assert "consumer" in child.list_stages()


def test_pipeline_resolve_from_parents_skips_existing_files(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Should treat existing files as external inputs."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    # Create external input file
    (tmp_path / "external.txt").write_text("external data")

    # No parent pipeline
    child = Pipeline("child", root=tmp_path)

    class ConsumerOutput(TypedDict):
        result: Annotated[Path, Out("result.txt", loaders.PathOnly())]

    def consumer(
        data: Annotated[Path, Dep("external.txt", loaders.PathOnly())]
    ) -> ConsumerOutput:
        return ConsumerOutput(result=Path("result.txt"))

    child.register(consumer)

    # Should not raise, external.txt exists
    child.resolve_from_parents()

    assert child.list_stages() == ["consumer"]


def test_pipeline_resolve_from_parents_is_idempotent(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Calling resolve_from_parents multiple times should be safe."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    parent_code = '''
from typing import Annotated, TypedDict
from pathlib import Path
from pivot.pipeline import Pipeline
from pivot import loaders
from pivot.outputs import Out

pipeline = Pipeline("parent")

class Output(TypedDict):
    data: Annotated[Path, Out("data.txt", loaders.PathOnly())]

def producer() -> Output:
    return Output(data=Path("data.txt"))

pipeline.register(producer)
'''
    (tmp_path / "pipeline.py").write_text(parent_code)

    child_dir = tmp_path / "child"
    child_dir.mkdir()
    child = Pipeline("child", root=child_dir)

    class ConsumerOutput(TypedDict):
        result: Annotated[Path, Out("result.txt", loaders.PathOnly())]

    def consumer(
        data: Annotated[Path, Dep("data.txt", loaders.PathOnly())]
    ) -> ConsumerOutput:
        return ConsumerOutput(result=Path("result.txt"))

    child.register(consumer)

    child.resolve_from_parents()
    count_after_first = len(child.list_stages())

    child.resolve_from_parents()
    count_after_second = len(child.list_stages())

    assert count_after_first == count_after_second == 2
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pipeline/test_pipeline.py::test_pipeline_resolve_from_parents_includes_producer -v`
Expected: FAIL with "AttributeError: 'Pipeline' object has no attribute 'resolve_from_parents'"

**Step 3: Write implementation**

```python
# Add to src/pivot/pipeline/pipeline.py

# Add import at top (after existing imports)
from pivot import discovery

# Add method to Pipeline class
def resolve_from_parents(self) -> None:
    """Resolve unresolved dependencies by searching parent pipelines.

    For each dependency that has no local producer:
    1. Traverse up directory tree looking for pivot.yaml or pipeline.py
    2. Load each parent pipeline and search for a stage producing the artifact
    3. Include that stage and add its dependencies to the work queue

    Dependencies that exist on disk are treated as external inputs.
    Uses per-call caching (parents loaded once per resolve, discarded after).
    """
    project_root = project.get_project_root()

    # Build set of locally produced outputs
    local_outputs = set[str]()
    for stage_name in self.list_stages():
        local_outputs.update(self.get(stage_name)["outs_paths"])

    # Build work queue of unresolved dependencies
    work = set[str]()
    for stage_name in self.list_stages():
        for dep_path in self.get(stage_name)["deps_paths"]:
            if dep_path not in local_outputs:
                work.add(dep_path)

    if not work:
        return

    # Find parent pipeline files once
    parent_files = list(discovery.find_parent_pipeline_paths(self.root, project_root))
    if not parent_files:
        return

    # Per-call cache: avoid reloading same parent for each unresolved dep
    loaded_parents: dict[pathlib.Path, Pipeline | None] = {}

    # Process work queue iteratively
    while work:
        dep_path = work.pop()

        # Skip if already resolved (by a stage we just added)
        if dep_path in local_outputs:
            continue

        # Skip if file exists on disk (external input)
        if pathlib.Path(dep_path).exists():
            continue

        # Search parent pipelines for producer
        for parent_file in parent_files:
            # Load parent (cached within this call)
            if parent_file not in loaded_parents:
                loaded_parents[parent_file] = discovery.load_pipeline_from_path(parent_file)
            parent = loaded_parents[parent_file]
            if parent is None:
                continue

            # Find stage that produces this artifact
            producer_name = None
            for stage_name in parent.list_stages():
                if dep_path in parent.get(stage_name)["outs_paths"]:
                    producer_name = stage_name
                    break

            if producer_name is None:
                continue

            # Skip if already included (idempotency)
            if producer_name in self._registry.list_stages():
                break

            # Include the producer stage
            stage_info = copy.deepcopy(parent.get(producer_name))
            self._registry.add_existing(stage_info)
            local_outputs.update(stage_info["outs_paths"])

            # Add producer's dependencies to work queue
            for producer_dep in stage_info["deps_paths"]:
                if producer_dep not in local_outputs:
                    work.add(producer_dep)

            logger.debug(
                f"Included stage '{producer_name}' from parent pipeline '{parent.name}'"
            )
            break
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_pipeline.py::test_pipeline_resolve_from_parents_includes_producer tests/pipeline/test_pipeline.py::test_pipeline_resolve_from_parents_includes_transitive_deps tests/pipeline/test_pipeline.py::test_pipeline_resolve_from_parents_skips_existing_files tests/pipeline/test_pipeline.py::test_pipeline_resolve_from_parents_is_idempotent -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(pipeline): add resolve_from_parents for lazy dependency resolution"
```

---

## Task 4: Add Integration Test

**Files:**
- Create: `tests/integration/test_lazy_resolution.py`

**Step 1: Write integration test**

```python
# tests/integration/test_lazy_resolution.py
from __future__ import annotations

import pathlib

import pytest
from pytest_mock import MockerFixture

from pivot import project
from pivot.pipeline.pipeline import Pipeline


@pytest.fixture
def lazy_project(tmp_path: pathlib.Path, mocker: MockerFixture) -> pathlib.Path:
    """Create project with parent/child pipeline structure."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    # Parent pipeline at root
    parent_code = '''
from typing import Annotated, TypedDict
from pathlib import Path
from pivot.pipeline import Pipeline
from pivot import loaders
from pivot.outputs import Out

pipeline = Pipeline("parent")

class ProducerOutput(TypedDict):
    data: Annotated[Path, Out("data/output.txt", loaders.PathOnly())]

def producer() -> ProducerOutput:
    Path("data").mkdir(exist_ok=True)
    Path("data/output.txt").write_text("produced")
    return ProducerOutput(data=Path("data/output.txt"))

pipeline.register(producer)
'''
    (tmp_path / "pipeline.py").write_text(parent_code)

    # Child pipeline
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    child_code = '''
from typing import Annotated, TypedDict
from pathlib import Path
from pivot.pipeline import Pipeline
from pivot import loaders
from pivot.outputs import Out, Dep

pipeline = Pipeline("child")

class ConsumerOutput(TypedDict):
    result: Annotated[Path, Out("result.txt", loaders.PathOnly())]

def consumer(
    data: Annotated[Path, Dep("data/output.txt", loaders.PathOnly())]
) -> ConsumerOutput:
    content = data.read_text()
    Path("result.txt").write_text(f"consumed: {content}")
    return ConsumerOutput(result=Path("result.txt"))

pipeline.register(consumer)
'''
    (child_dir / "pipeline.py").write_text(child_code)

    return tmp_path


def test_lazy_resolution_builds_complete_dag(lazy_project: pathlib.Path) -> None:
    """Child pipeline should build complete DAG including parent stages."""
    from pivot import discovery

    child_dir = lazy_project / "child"
    child = discovery.load_pipeline_from_path(child_dir / "pipeline.py")
    assert child is not None

    child.resolve_from_parents()
    dag = child.build_dag(validate=True)

    assert "producer" in dag.nodes
    assert "consumer" in dag.nodes
    assert dag.has_edge("consumer", "producer")


def test_lazy_resolution_preserves_parent_state_dir(lazy_project: pathlib.Path) -> None:
    """Included parent stages should retain their original state_dir."""
    from pivot import discovery

    child_dir = lazy_project / "child"
    child = discovery.load_pipeline_from_path(child_dir / "pipeline.py")
    assert child is not None

    child.resolve_from_parents()

    producer_info = child.get("producer")
    # Producer's state_dir should be parent's .pivot, not child's
    assert producer_info["state_dir"] == lazy_project / ".pivot"
```

**Step 2: Run integration tests**

Run: `uv run pytest tests/integration/test_lazy_resolution.py -v`
Expected: PASS

**Step 3: Commit**

```bash
jj describe -m "test: add integration tests for lazy pipeline resolution"
```

---

## Task 5: Run Quality Checks

**Step 1: Run all tests**

Run: `uv run pytest tests/ -n auto`
Expected: All tests pass

**Step 2: Run type checker**

Run: `uv run basedpyright`
Expected: No errors

**Step 3: Run linter and formatter**

Run: `uv run ruff check . && uv run ruff format .`
Expected: No errors

**Step 4: Final commit**

```bash
jj describe -m "feat(pipeline): lazy dependency resolution from parent pipelines

When a child pipeline has unresolved dependencies, resolve_from_parents()
traverses up the directory tree to find parent pivot.yaml or pipeline.py
files, loads them, and includes stages that produce the needed artifacts.

Key behaviors:
- Searches for both pivot.yaml and pipeline.py (errors if both exist)
- Closest parent is searched first
- Transitive dependencies are resolved iteratively
- Files on disk are treated as external inputs
- No caching - each resolve sees fresh parent state (watch mode safe)
- Parent stages retain their original state_dir
- build_dag() is unchanged (non-mutating) - call resolve_from_parents() explicitly"
```

---

## Summary

This simplified implementation:

1. **~100 lines of new code** (vs ~400 in original plan)
2. **No new modules** - extends existing `discovery.py` and `pipeline.py`
3. **Per-call caching** - parents cached within single `resolve_from_parents()` call, discarded after (safe, no staleness issues, no test isolation concerns)
4. **Non-mutating `build_dag()`** - explicit `resolve_from_parents()` call
5. **Supports both pivot.yaml and pipeline.py** in parent directories
6. **Iterative algorithm** - simple work queue, no recursion

**Future optimization:** If watch mode performance becomes an issue (150ms per event vs ~0.5ms with mtime cache), can add mtime-based caching with proper mitigations (deep copy, bounded size, error handling).
