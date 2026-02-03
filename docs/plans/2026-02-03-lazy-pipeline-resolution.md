# Lazy Pipeline Dependency Resolution

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable pipelines to automatically discover and include stages from parent pipelines when dependencies are unresolved locally.

**Architecture:** When `build_dag()` finds an unresolved dependency, traverse up the directory tree looking for `pipeline.py` files, load each parent pipeline, find the stage that produces the needed artifact, and include it (plus transitive dependencies). This is lazy loading - the project DAG is unchanged, we just don't load all of it upfront.

**Tech Stack:** Python 3.13+, networkx, pytest

---

## Task 1: Add Parent Pipeline Discovery Function

**Files:**
- Create: `src/pivot/pipeline/discovery.py` (new module for pipeline discovery utilities)
- Test: `tests/pipeline/test_discovery.py`

**Step 1: Write the failing test for directory traversal**

```python
# tests/pipeline/test_discovery.py
from __future__ import annotations

import pathlib

from pivot.pipeline import discovery


def test_find_parent_pipeline_files_finds_pipeline_in_parent(tmp_path: pathlib.Path) -> None:
    """Should find pipeline.py in parent directories."""
    # Create directory structure:
    # tmp_path/
    #   pipeline.py
    #   sub/
    #     child/
    #       pipeline.py  <- start here
    (tmp_path / "pipeline.py").touch()
    child_dir = tmp_path / "sub" / "child"
    child_dir.mkdir(parents=True)
    (child_dir / "pipeline.py").touch()

    result = list(discovery.find_parent_pipeline_files(child_dir, stop_at=tmp_path))

    # Should find parent's pipeline.py (not child's own)
    assert result == [tmp_path / "pipeline.py"]


def test_find_parent_pipeline_files_stops_at_boundary(tmp_path: pathlib.Path) -> None:
    """Should stop traversal at specified boundary."""
    # Create pipeline.py above the stop boundary
    above_boundary = tmp_path / "above"
    above_boundary.mkdir()
    (above_boundary / "pipeline.py").touch()

    project_root = above_boundary / "project"
    project_root.mkdir()
    (project_root / "pipeline.py").touch()

    child = project_root / "sub"
    child.mkdir()
    (child / "pipeline.py").touch()

    result = list(discovery.find_parent_pipeline_files(child, stop_at=project_root))

    # Should find project root's pipeline.py but not above
    assert result == [project_root / "pipeline.py"]


def test_find_parent_pipeline_files_returns_multiple_in_order(tmp_path: pathlib.Path) -> None:
    """Should return all parent pipeline.py files, closest first."""
    (tmp_path / "pipeline.py").touch()
    mid = tmp_path / "mid"
    mid.mkdir()
    (mid / "pipeline.py").touch()
    child = mid / "child"
    child.mkdir()
    (child / "pipeline.py").touch()

    result = list(discovery.find_parent_pipeline_files(child, stop_at=tmp_path))

    # Closest parent first
    assert result == [mid / "pipeline.py", tmp_path / "pipeline.py"]


def test_find_parent_pipeline_files_skips_own_directory(tmp_path: pathlib.Path) -> None:
    """Should not include pipeline.py from the start directory itself."""
    (tmp_path / "pipeline.py").touch()

    result = list(discovery.find_parent_pipeline_files(tmp_path, stop_at=tmp_path.parent))

    assert result == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pipeline/test_discovery.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
# src/pivot/pipeline/discovery.py
from __future__ import annotations

import pathlib
from collections.abc import Iterator

PIPELINE_PY_NAME = "pipeline.py"


def find_parent_pipeline_files(
    start_dir: pathlib.Path,
    stop_at: pathlib.Path,
) -> Iterator[pathlib.Path]:
    """Find pipeline.py files in parent directories.

    Traverses up from start_dir (exclusive) to stop_at (inclusive),
    yielding each pipeline.py found. Closest parents are yielded first.

    Args:
        start_dir: Directory to start from (its pipeline.py is NOT included).
        stop_at: Stop traversal at this directory (inclusive).

    Yields:
        Paths to pipeline.py files found in parent directories.
    """
    current = start_dir.resolve()
    stop_at = stop_at.resolve()

    # Move to parent (don't include start_dir's own pipeline.py)
    current = current.parent

    while True:
        pipeline_file = current / PIPELINE_PY_NAME
        if pipeline_file.exists():
            yield pipeline_file

        # Stop if we've reached the boundary
        if current == stop_at:
            break

        # Stop if we've reached filesystem root
        if current.parent == current:
            break

        current = current.parent
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pipeline/test_discovery.py -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(pipeline): add find_parent_pipeline_files for lazy resolution"
```

---

## Task 2: Add Parent Pipeline Loading Function

**Files:**
- Modify: `src/pivot/pipeline/discovery.py`
- Test: `tests/pipeline/test_discovery.py`

**Step 1: Write the failing test for loading parent pipeline**

```python
# Add to tests/pipeline/test_discovery.py
from pivot.pipeline.pipeline import Pipeline


def test_load_parent_pipeline_loads_valid_pipeline(
    tmp_path: pathlib.Path, mocker
) -> None:
    """Should load and return Pipeline from pipeline.py file."""
    from pivot import project

    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    # Create a pipeline.py with a simple pipeline
    pipeline_code = '''
from pivot.pipeline import Pipeline

pipeline = Pipeline("parent")
'''
    (tmp_path / "pipeline.py").write_text(pipeline_code)

    result = discovery.load_parent_pipeline(tmp_path / "pipeline.py")

    assert result is not None
    assert result.name == "parent"


def test_load_parent_pipeline_returns_none_for_no_pipeline_var(
    tmp_path: pathlib.Path, mocker
) -> None:
    """Should return None if pipeline.py doesn't define 'pipeline' variable."""
    from pivot import project

    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    # Create a pipeline.py without pipeline variable
    (tmp_path / "pipeline.py").write_text("x = 1\n")

    result = discovery.load_parent_pipeline(tmp_path / "pipeline.py")

    assert result is None


def test_load_parent_pipeline_caches_loaded_pipelines(
    tmp_path: pathlib.Path, mocker
) -> None:
    """Should cache loaded pipelines to avoid re-parsing."""
    from pivot import project

    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    pipeline_code = '''
from pivot.pipeline import Pipeline

pipeline = Pipeline("cached")
'''
    (tmp_path / "pipeline.py").write_text(pipeline_code)

    # Load twice
    result1 = discovery.load_parent_pipeline(tmp_path / "pipeline.py")
    result2 = discovery.load_parent_pipeline(tmp_path / "pipeline.py")

    # Should return same object (cached)
    assert result1 is result2
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pipeline/test_discovery.py::test_load_parent_pipeline_loads_valid_pipeline -v`
Expected: FAIL with "AttributeError" (function doesn't exist)

**Step 3: Write minimal implementation**

```python
# Add to src/pivot/pipeline/discovery.py
import runpy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pivot.pipeline.pipeline import Pipeline

# Cache for loaded parent pipelines (path -> Pipeline)
_parent_pipeline_cache: dict[pathlib.Path, Pipeline | None] = {}


def load_parent_pipeline(pipeline_path: pathlib.Path) -> Pipeline | None:
    """Load a Pipeline from a pipeline.py file.

    Results are cached to avoid re-parsing the same file multiple times.

    Args:
        pipeline_path: Path to pipeline.py file.

    Returns:
        Pipeline instance if file defines 'pipeline' variable, None otherwise.
    """
    from pivot.pipeline.pipeline import Pipeline

    resolved = pipeline_path.resolve()
    if resolved in _parent_pipeline_cache:
        return _parent_pipeline_cache[resolved]

    try:
        module_dict = runpy.run_path(str(resolved), run_name="_pivot_parent_pipeline")
    except Exception:
        _parent_pipeline_cache[resolved] = None
        return None

    pipeline = module_dict.get("pipeline")
    if pipeline is not None and isinstance(pipeline, Pipeline):
        _parent_pipeline_cache[resolved] = pipeline
        return pipeline

    _parent_pipeline_cache[resolved] = None
    return None


def clear_parent_pipeline_cache() -> None:
    """Clear the parent pipeline cache (for testing)."""
    _parent_pipeline_cache.clear()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pipeline/test_discovery.py::test_load_parent_pipeline_loads_valid_pipeline tests/pipeline/test_discovery.py::test_load_parent_pipeline_returns_none_for_no_pipeline_var tests/pipeline/test_discovery.py::test_load_parent_pipeline_caches_loaded_pipelines -v`
Expected: PASS

**Step 5: Add fixture to clear cache between tests**

```python
# Add to tests/pipeline/test_discovery.py at top
import pytest

@pytest.fixture(autouse=True)
def clear_discovery_cache() -> None:
    """Clear parent pipeline cache before each test."""
    discovery.clear_parent_pipeline_cache()
```

**Step 6: Commit**

```bash
jj describe -m "feat(pipeline): add load_parent_pipeline with caching"
```

---

## Task 3: Add Producer Search in Parent Pipeline

**Files:**
- Modify: `src/pivot/pipeline/discovery.py`
- Test: `tests/pipeline/test_discovery.py`

**Step 1: Write the failing test**

```python
# Add to tests/pipeline/test_discovery.py
from typing import Annotated
from pathlib import Path

from pivot import loaders
from pivot.outputs import Out, Dep


def test_find_producer_in_pipeline_finds_matching_stage(
    tmp_path: pathlib.Path, mocker
) -> None:
    """Should find stage that produces the requested path."""
    from pivot import project

    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    # Create pipeline with a stage that produces output.txt
    pipeline = Pipeline("test", root=tmp_path)

    class Output(TypedDict):
        data: Annotated[Path, Out("output.txt", loaders.PathOnly())]

    def producer() -> Output:
        return Output(data=Path("output.txt"))

    pipeline.register(producer)

    # Search for the producer of output.txt (absolute path)
    abs_path = str(tmp_path / "output.txt")
    result = discovery.find_producer_in_pipeline(pipeline, abs_path)

    assert result == "producer"


def test_find_producer_in_pipeline_returns_none_if_not_found(
    tmp_path: pathlib.Path, mocker
) -> None:
    """Should return None if no stage produces the path."""
    from pivot import project

    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    pipeline = Pipeline("test", root=tmp_path)

    result = discovery.find_producer_in_pipeline(pipeline, "/nonexistent/path.txt")

    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pipeline/test_discovery.py::test_find_producer_in_pipeline_finds_matching_stage -v`
Expected: FAIL with "AttributeError"

**Step 3: Write minimal implementation**

```python
# Add to src/pivot/pipeline/discovery.py
def find_producer_in_pipeline(pipeline: Pipeline, dep_path: str) -> str | None:
    """Find stage in pipeline that produces the given path.

    Args:
        pipeline: Pipeline to search.
        dep_path: Absolute path to look for as an output.

    Returns:
        Stage name if found, None otherwise.
    """
    for stage_name in pipeline.list_stages():
        stage_info = pipeline.get(stage_name)
        if dep_path in stage_info["outs_paths"]:
            return stage_name
    return None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pipeline/test_discovery.py::test_find_producer_in_pipeline_finds_matching_stage tests/pipeline/test_discovery.py::test_find_producer_in_pipeline_returns_none_if_not_found -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(pipeline): add find_producer_in_pipeline"
```

---

## Task 4: Add resolve_from_parents Method to Pipeline

**Files:**
- Modify: `src/pivot/pipeline/pipeline.py`
- Test: `tests/pipeline/test_pipeline.py`

**Step 1: Write the failing test**

```python
# Add to tests/pipeline/test_pipeline.py
from typing import Annotated, TypedDict
from pathlib import Path

from pivot import loaders
from pivot.outputs import Out, Dep


def test_pipeline_resolve_from_parents_includes_producer(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Should include producer stage from parent pipeline when dependency unresolved."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    # Parent pipeline at project root
    parent = Pipeline("parent", root=tmp_path)

    class ProducerOutput(TypedDict):
        data: Annotated[Path, Out("shared/data.txt", loaders.PathOnly())]

    def producer() -> ProducerOutput:
        Path("shared/data.txt").write_text("data")
        return ProducerOutput(data=Path("shared/data.txt"))

    parent.register(producer)

    # Save parent pipeline to file
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

    # Should include both stage_a and stage_b
    assert "stage_a" in child.list_stages()
    assert "stage_b" in child.list_stages()
    assert "consumer" in child.list_stages()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pipeline/test_pipeline.py::test_pipeline_resolve_from_parents_includes_producer -v`
Expected: FAIL with "AttributeError: 'Pipeline' object has no attribute 'resolve_from_parents'"

**Step 3: Write minimal implementation**

```python
# Add to src/pivot/pipeline/pipeline.py

# Add import at top
from pivot.pipeline import discovery as pipeline_discovery

# Add method to Pipeline class
def resolve_from_parents(self) -> None:
    """Resolve unresolved dependencies by searching parent pipelines.

    For each dependency that has no local producer:
    1. Traverse up directory tree looking for pipeline.py files
    2. Load each parent pipeline and search for a stage producing the dependency
    3. Include that stage (and recursively resolve its dependencies)

    Dependencies that exist on disk or have no producer anywhere are left as-is
    (treated as external inputs).

    Raises:
        PipelineConfigError: If multiple parent pipelines produce the same artifact.
    """
    project_root = project.get_project_root()

    # Build local outputs map
    local_outputs = set[str]()
    for stage_name in self.list_stages():
        stage_info = self.get(stage_name)
        local_outputs.update(stage_info["outs_paths"])

    # Find unresolved dependencies
    unresolved = set[str]()
    for stage_name in self.list_stages():
        stage_info = self.get(stage_name)
        for dep_path in stage_info["deps_paths"]:
            if dep_path not in local_outputs:
                unresolved.add(dep_path)

    # Resolve from parents
    self._resolve_deps_from_parents(unresolved, local_outputs, project_root)


def _resolve_deps_from_parents(
    self,
    unresolved: set[str],
    local_outputs: set[str],
    project_root: pathlib.Path,
) -> None:
    """Recursively resolve dependencies from parent pipelines."""
    if not unresolved:
        return

    # Find parent pipeline files
    parent_files = list(pipeline_discovery.find_parent_pipeline_files(self.root, project_root))
    if not parent_files:
        return  # No parents to search

    newly_included = list[str]()

    for dep_path in list(unresolved):
        # Skip if file exists on disk (external input)
        if pathlib.Path(dep_path).exists():
            unresolved.discard(dep_path)
            continue

        # Search parent pipelines for producer
        for parent_file in parent_files:
            parent = pipeline_discovery.load_parent_pipeline(parent_file)
            if parent is None:
                continue

            producer_name = pipeline_discovery.find_producer_in_pipeline(parent, dep_path)
            if producer_name is None:
                continue

            # Check for duplicate producer (error condition)
            if producer_name in self._registry.list_stages():
                # Already included, dependency resolved
                unresolved.discard(dep_path)
                break

            # Include the producer stage
            import copy
            stage_info = copy.deepcopy(parent.get(producer_name))
            self._registry.add_existing(stage_info)
            newly_included.append(producer_name)
            local_outputs.update(stage_info["outs_paths"])
            unresolved.discard(dep_path)

            # Add producer's dependencies to unresolved
            for producer_dep in stage_info["deps_paths"]:
                if producer_dep not in local_outputs:
                    unresolved.add(producer_dep)

            logger.debug(f"Included stage '{producer_name}' from parent pipeline '{parent.name}'")
            break

    # Recurse if we added stages with new dependencies
    if newly_included and unresolved:
        self._resolve_deps_from_parents(unresolved, local_outputs, project_root)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pipeline/test_pipeline.py::test_pipeline_resolve_from_parents_includes_producer tests/pipeline/test_pipeline.py::test_pipeline_resolve_from_parents_includes_transitive_deps -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(pipeline): add resolve_from_parents for lazy dependency resolution"
```

---

## Task 5: Integrate resolve_from_parents into build_dag

**Files:**
- Modify: `src/pivot/pipeline/pipeline.py`
- Test: `tests/pipeline/test_pipeline.py`

**Step 1: Write the failing test**

```python
# Add to tests/pipeline/test_pipeline.py
def test_pipeline_build_dag_auto_resolves_from_parents(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """build_dag should automatically resolve dependencies from parents."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    # Parent pipeline
    parent_code = '''
from typing import Annotated, TypedDict
from pathlib import Path
from pivot.pipeline import Pipeline
from pivot import loaders
from pivot.outputs import Out

pipeline = Pipeline("parent")

class ProducerOutput(TypedDict):
    data: Annotated[Path, Out("data.txt", loaders.PathOnly())]

def producer() -> ProducerOutput:
    return ProducerOutput(data=Path("data.txt"))

pipeline.register(producer)
'''
    (tmp_path / "pipeline.py").write_text(parent_code)

    # Child pipeline
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

    # build_dag should auto-resolve
    dag = child.build_dag(validate=True)

    assert "producer" in dag.nodes
    assert "consumer" in dag.nodes
    assert dag.has_edge("consumer", "producer")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pipeline/test_pipeline.py::test_pipeline_build_dag_auto_resolves_from_parents -v`
Expected: FAIL with "DependencyNotFoundError" (producer not included)

**Step 3: Modify build_dag to call resolve_from_parents**

```python
# Modify build_dag in src/pivot/pipeline/pipeline.py
def build_dag(self, validate: bool = True) -> DiGraph[str]:
    """Build DAG from registered stages.

    Automatically resolves unresolved dependencies by searching parent
    pipelines (lazy dependency resolution).

    Args:
        validate: If True, validate that all dependencies exist

    Returns:
        NetworkX DiGraph with stages as nodes and dependencies as edges

    Raises:
        CyclicGraphError: If graph contains cycles
        DependencyNotFoundError: If dependency doesn't exist (when validate=True)
    """
    # Auto-resolve from parents before building DAG
    self.resolve_from_parents()
    return self._registry.build_dag(validate=validate)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pipeline/test_pipeline.py::test_pipeline_build_dag_auto_resolves_from_parents -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(pipeline): auto-resolve from parents in build_dag"
```

---

## Task 6: Add Error for Multiple Producers

**Files:**
- Modify: `src/pivot/pipeline/pipeline.py`
- Modify: `src/pivot/pipeline/yaml.py` (add new exception)
- Test: `tests/pipeline/test_pipeline.py`

**Step 1: Write the failing test**

```python
# Add to tests/pipeline/test_pipeline.py
def test_pipeline_resolve_from_parents_errors_on_multiple_producers(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Should error if multiple parent pipelines produce the same artifact."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    # First parent at project root
    parent1_code = '''
from typing import Annotated, TypedDict
from pathlib import Path
from pivot.pipeline import Pipeline
from pivot import loaders
from pivot.outputs import Out

pipeline = Pipeline("parent1")

class Output(TypedDict):
    data: Annotated[Path, Out("shared.txt", loaders.PathOnly())]

def producer1() -> Output:
    return Output(data=Path("shared.txt"))

pipeline.register(producer1)
'''
    (tmp_path / "pipeline.py").write_text(parent1_code)

    # Second parent at mid level
    mid_dir = tmp_path / "mid"
    mid_dir.mkdir()
    parent2_code = '''
from typing import Annotated, TypedDict
from pathlib import Path
from pivot.pipeline import Pipeline
from pivot import loaders
from pivot.outputs import Out

pipeline = Pipeline("parent2")

class Output(TypedDict):
    data: Annotated[Path, Out("shared.txt", loaders.PathOnly())]

def producer2() -> Output:
    return Output(data=Path("shared.txt"))

pipeline.register(producer2)
'''
    (mid_dir / "pipeline.py").write_text(parent2_code)

    # Child pipeline
    child_dir = mid_dir / "child"
    child_dir.mkdir()
    child = Pipeline("child", root=child_dir)

    class ConsumerOutput(TypedDict):
        result: Annotated[Path, Out("result.txt", loaders.PathOnly())]

    def consumer(
        data: Annotated[Path, Dep("shared.txt", loaders.PathOnly())]
    ) -> ConsumerOutput:
        return ConsumerOutput(result=Path("result.txt"))

    child.register(consumer)

    from pivot.pipeline.yaml import PipelineConfigError

    with pytest.raises(PipelineConfigError, match="Multiple parent pipelines produce"):
        child.resolve_from_parents()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pipeline/test_pipeline.py::test_pipeline_resolve_from_parents_errors_on_multiple_producers -v`
Expected: FAIL (no error raised, or wrong error)

**Step 3: Update implementation to detect multiple producers**

```python
# Update _resolve_deps_from_parents in src/pivot/pipeline/pipeline.py
def _resolve_deps_from_parents(
    self,
    unresolved: set[str],
    local_outputs: set[str],
    project_root: pathlib.Path,
) -> None:
    """Recursively resolve dependencies from parent pipelines."""
    if not unresolved:
        return

    parent_files = list(pipeline_discovery.find_parent_pipeline_files(self.root, project_root))
    if not parent_files:
        return

    newly_included = list[str]()

    for dep_path in list(unresolved):
        if pathlib.Path(dep_path).exists():
            unresolved.discard(dep_path)
            continue

        # Find all producers across all parents
        producers_found: list[tuple[str, Pipeline]] = []
        for parent_file in parent_files:
            parent = pipeline_discovery.load_parent_pipeline(parent_file)
            if parent is None:
                continue

            producer_name = pipeline_discovery.find_producer_in_pipeline(parent, dep_path)
            if producer_name is not None:
                producers_found.append((producer_name, parent))

        # Error if multiple producers
        if len(producers_found) > 1:
            producer_names = [f"'{name}' in '{p.name}'" for name, p in producers_found]
            raise PipelineConfigError(
                f"Multiple parent pipelines produce '{dep_path}': {', '.join(producer_names)}. "
                f"This indicates a malformed project DAG."
            )

        if not producers_found:
            continue  # No producer found, leave as external input

        producer_name, parent = producers_found[0]

        if producer_name in self._registry.list_stages():
            unresolved.discard(dep_path)
            continue

        import copy
        stage_info = copy.deepcopy(parent.get(producer_name))
        self._registry.add_existing(stage_info)
        newly_included.append(producer_name)
        local_outputs.update(stage_info["outs_paths"])
        unresolved.discard(dep_path)

        for producer_dep in stage_info["deps_paths"]:
            if producer_dep not in local_outputs:
                unresolved.add(producer_dep)

        logger.debug(f"Included stage '{producer_name}' from parent pipeline '{parent.name}'")

    if newly_included and unresolved:
        self._resolve_deps_from_parents(unresolved, local_outputs, project_root)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pipeline/test_pipeline.py::test_pipeline_resolve_from_parents_errors_on_multiple_producers -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(pipeline): error on multiple producers in lazy resolution"
```

---

## Task 7: Add Integration Test with Real Execution

**Files:**
- Test: `tests/integration/test_lazy_resolution.py`

**Step 1: Write integration test**

```python
# tests/integration/test_lazy_resolution.py
from __future__ import annotations

import pathlib

import pytest

from pivot.cli import cli
from click.testing import CliRunner


@pytest.fixture
def lazy_resolution_project(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a project with parent/child pipeline structure."""
    # Initialize git repo
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
    Path("data/output.txt").write_text("produced data")
    return ProducerOutput(data=Path("data/output.txt"))

pipeline.register(producer)
'''
    (tmp_path / "pipeline.py").write_text(parent_code)

    # Child pipeline in subdirectory
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


def test_lazy_resolution_repro_from_child(
    lazy_resolution_project: pathlib.Path,
) -> None:
    """Running repro from child directory should auto-include parent stages."""
    runner = CliRunner()
    child_dir = lazy_resolution_project / "child"

    with runner.isolated_filesystem(temp_dir=lazy_resolution_project):
        # Change to child directory
        import os
        os.chdir(child_dir)

        result = runner.invoke(cli, ["repro"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "producer" in result.output or "consumer" in result.output

    # Verify outputs exist
    assert (lazy_resolution_project / "data" / "output.txt").exists()
    assert (child_dir / "result.txt").exists()
    assert (child_dir / "result.txt").read_text() == "consumed: produced data"
```

**Step 2: Run integration test**

Run: `uv run pytest tests/integration/test_lazy_resolution.py -v`
Expected: PASS

**Step 3: Commit**

```bash
jj describe -m "test: add integration test for lazy pipeline resolution"
```

---

## Task 8: Run Quality Checks and Final Verification

**Step 1: Run all tests**

Run: `uv run pytest tests/ -n auto`
Expected: All tests pass

**Step 2: Run type checker**

Run: `uv run basedpyright`
Expected: No errors

**Step 3: Run linter**

Run: `uv run ruff check .`
Expected: No errors

**Step 4: Run formatter**

Run: `uv run ruff format .`
Expected: Files formatted

**Step 5: Final commit**

```bash
jj describe -m "feat(pipeline): lazy dependency resolution from parent pipelines

When building a pipeline's DAG, unresolved dependencies are automatically
resolved by traversing up the directory tree, loading parent pipeline.py
files, and including stages that produce the needed artifacts.

Key behaviors:
- Closest parent pipeline is searched first
- Transitive dependencies are recursively resolved
- Multiple producers for the same artifact is an error
- Files that exist on disk are treated as external inputs
- Parent stages retain their original state_dir"
```

---

## Summary

This implementation adds lazy pipeline dependency resolution through:

1. **`find_parent_pipeline_files()`** - Traverses up directory tree to find parent pipeline.py files
2. **`load_parent_pipeline()`** - Loads and caches parent Pipeline instances
3. **`find_producer_in_pipeline()`** - Searches a pipeline for a stage producing a path
4. **`Pipeline.resolve_from_parents()`** - Main entry point that resolves unresolved deps
5. **Integration with `build_dag()`** - Auto-resolution happens before DAG construction

The design preserves existing behavior while enabling child pipelines to depend on parent pipeline outputs without explicit `include()` calls.
