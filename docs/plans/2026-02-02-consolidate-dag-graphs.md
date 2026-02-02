# Consolidate DAG Graph Implementations

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate `dag/build.py` (stage-only DAG) and `engine/graph.py` (bipartite artifact-stage graph) into a single bipartite graph with trie-based directory resolution.

**Architecture:** The bipartite graph in `engine/graph.py` becomes the single source of truth. Move validation logic (cycle detection, missing dependency detection with trie-based directory resolution, tracked file support) from `dag/build.py` into `engine/graph.py`. Delete `dag/build.py` entirely.

**Tech Stack:** NetworkX, pygtrie

**Downstream enabler:** This consolidation is a prerequisite for GitHub issue #323 (cross-pipeline dependency resolution), which needs artifact-level queries like "which stage produces this file?"

---

## Current State

Two graph implementations exist with overlapping functionality:

| File | Nodes | Edge Direction | Unique Features |
|------|-------|----------------|-----------------|
| `dag/build.py` | Stages only | consumer → producer | Trie-based dir deps, tracked files, cycle detection |
| `engine/graph.py` | Stages + Artifacts | data flow | Artifact queries, `update_stage()`, `get_watch_paths()` |

Current redundant flow in `engine.py:457-460`:
```python
self._graph = engine_graph.build_graph(all_stages)  # bipartite
dag.build_dag(all_stages, validate=True)            # REDUNDANT - just for validation
```

## Target State

Single bipartite graph with all validation integrated. `dag/build.py` deleted.

```python
self._graph = engine_graph.build_graph(all_stages, validate=True, tracked_files=tracked_files)
stage_dag = engine_graph.get_stage_dag(self._graph)
execution_order = engine_graph.get_execution_order(stage_dag, ...)
```

---

## Task 1: Move trie helpers from `dag/build.py` to `engine/graph.py`

Move existing code. Do not rewrite.

**Files:**
- Modify: `src/pivot/engine/graph.py` (add moved functions)
- Modify: `src/pivot/dag/build.py` (delete moved functions)

**Step 1: Add pygtrie import to `engine/graph.py`**

At top of `src/pivot/engine/graph.py`, add:

```python
import pygtrie
```

**Step 2: Move `_build_outputs_map` from `dag/build.py:86-102`**

Cut from `dag/build.py` and paste into `engine/graph.py` after the `parse_node` function:

```python
def _build_outputs_map(stages: dict[str, RegistryStageInfo]) -> dict[str, str]:
    """Build mapping from output path to stage name.

    Returns:
        Dict of output_path -> stage_name
    """
    return {
        out_path: stage_name
        for stage_name, stage_info in stages.items()
        for out_path in stage_info["outs_paths"]
    }
```

**Step 3: Move `_build_outputs_trie` from `dag/build.py:105-112`**

Cut and paste into `engine/graph.py`:

```python
def _build_outputs_trie(stages: dict[str, RegistryStageInfo]) -> pygtrie.Trie[tuple[str, str]]:
    """Build trie of output paths for directory dependency resolution."""
    trie: pygtrie.Trie[tuple[str, str]] = pygtrie.Trie()
    for stage_name, stage_info in stages.items():
        for out_path in stage_info["outs_paths"]:
            out_key = pathlib.Path(out_path).parts
            trie[out_key] = (stage_name, out_path)
    return trie
```

**Step 4: Move `_find_producers_for_path` from `dag/build.py:115-135`**

Cut and paste into `engine/graph.py`:

```python
def _find_producers_for_path(
    dep_path: str, outputs_trie: pygtrie.Trie[tuple[str, str]]
) -> list[str]:
    """Find stages with outputs overlapping the dependency path (parent or child)."""
    dep_key = pathlib.Path(dep_path).parts
    producers = list[str]()

    # Case 1: Dependency is parent of outputs (dir depends on files inside)
    if outputs_trie.has_subtrie(dep_key):
        for stage_name, _ in outputs_trie.values(prefix=dep_key):
            if stage_name not in producers:
                producers.append(stage_name)

    # Case 2: Dependency is child of output (file depends on parent dir)
    prefix_item = outputs_trie.shortest_prefix(dep_key)
    if prefix_item is not None and prefix_item.value is not None:
        stage_name, _ = prefix_item.value
        if stage_name not in producers:
            producers.append(stage_name)

    return producers
```

**Step 5: Move `build_tracked_trie` from `dag/build.py:138-147`**

Cut and paste into `engine/graph.py`. Add to `__all__`:

```python
def build_tracked_trie(tracked_files: dict[str, Any]) -> pygtrie.Trie[str]:
    """Build trie of tracked file paths for dependency checking.

    Keys are path tuples (from Path.parts), values are the absolute path string.
    """
    trie: pygtrie.Trie[str] = pygtrie.Trie()
    for abs_path in tracked_files:
        path_key = pathlib.Path(abs_path).parts
        trie[path_key] = abs_path
    return trie
```

Add `Any` import: change `from typing import TYPE_CHECKING` to `from typing import TYPE_CHECKING, Any`

**Step 6: Move `_is_tracked_path` from `dag/build.py:150-164`**

Cut and paste into `engine/graph.py`:

```python
def _is_tracked_path(dep: str, tracked_trie: pygtrie.Trie[str]) -> bool:
    """Check if dependency is a tracked file (exact match or inside tracked directory)."""
    dep_key = pathlib.Path(dep).parts

    # Exact match
    if dep_key in tracked_trie:
        return True

    # Dependency is inside a tracked directory
    prefix_item = tracked_trie.shortest_prefix(dep_key)
    if prefix_item is not None and prefix_item.value is not None:
        return True

    # Dependency is a directory containing tracked files
    return tracked_trie.has_subtrie(dep_key)
```

**Step 7: Update `__all__` in `engine/graph.py`**

Add `"build_tracked_trie"` to `__all__`.

**Step 8: Run tests to verify nothing broke**

```bash
uv run pytest tests/engine/test_graph.py tests/core/test_dag.py -v
```

Expected: Some `test_dag.py` tests may fail because we removed functions from `dag/build.py`. That's expected - we'll fix imports in later tasks.

**Step 9: Commit**

```bash
jj describe -m "refactor(graph): move trie helpers from dag/build.py to engine/graph.py"
```

---

## Task 2: Add validation to `build_graph()`

Integrate cycle detection and dependency validation into the bipartite graph builder.

**Files:**
- Modify: `src/pivot/engine/graph.py`

**Step 1: Move `_check_acyclic` from `dag/build.py:167-177`**

Cut and paste into `engine/graph.py`. Adapt for bipartite graph (extract stage names from cycle):

```python
def _check_acyclic(g: nx.DiGraph[str]) -> None:
    """Check graph for cycles, raise if found."""
    from pivot import exceptions

    try:
        cycle = nx.find_cycle(g, orientation="original")
    except nx.NetworkXNoCycle:
        return

    # Extract stage names from cycle for error message
    stages_in_cycle = []
    for from_node, to_node, _ in cycle:
        node_type, name = parse_node(from_node)
        if node_type == NodeType.STAGE and name not in stages_in_cycle:
            stages_in_cycle.append(name)

    if not stages_in_cycle:
        # Fallback if cycle is artifact-only (shouldn't happen in valid graph)
        stages_in_cycle = [str(edge[0]) for edge in cycle]

    raise exceptions.CyclicGraphError(
        f"Circular dependency detected: {' -> '.join(stages_in_cycle)}"
    )
```

**Step 2: Update `build_graph` signature and implementation**

Replace the existing `build_graph` function:

```python
def build_graph(
    stages: dict[str, RegistryStageInfo],
    validate: bool = False,
    tracked_files: dict[str, Any] | None = None,
) -> nx.DiGraph[str]:
    """Build bipartite artifact-stage graph from stage definitions.

    Args:
        stages: Dict mapping stage name to RegistryStageInfo.
        validate: If True, validate that all dependencies exist.
        tracked_files: Dict of tracked file paths -> PvtData (from .pvt files).
            If provided, tracked files are recognized as valid dependency sources.

    Returns:
        Directed graph where:
        - Nodes are either artifacts (files) or stages (functions)
        - Edges go: artifact -> stage (consumed by) and stage -> artifact (produces)

    Raises:
        CyclicGraphError: If graph contains cycles (always checked)
        DependencyNotFoundError: If dependency doesn't exist (when validate=True)
    """
    from pivot import exceptions

    g: nx.DiGraph[str] = nx.DiGraph()

    # Build lookup structures for validation
    outputs_map = _build_outputs_map(stages)
    outputs_trie = _build_outputs_trie(stages) if validate else None
    tracked_trie = build_tracked_trie(tracked_files) if tracked_files else None

    for stage_name, info in stages.items():
        stage = stage_node(stage_name)
        g.add_node(stage, type=NodeType.STAGE)

        # Deps: artifact -> stage
        for dep_path in info["deps_paths"]:
            artifact = artifact_node(pathlib.Path(dep_path))
            g.add_node(artifact, type=NodeType.ARTIFACT)
            g.add_edge(artifact, stage)

            # Validation: check dependency source exists
            if validate:
                producer = outputs_map.get(dep_path)
                if not producer:
                    # Check for directory dependency via trie
                    producers = _find_producers_for_path(dep_path, outputs_trie) if outputs_trie else []
                    if not producers:
                        # Check if exists on disk
                        if pathlib.Path(dep_path).exists():
                            continue
                        # Check if tracked file
                        if tracked_trie and _is_tracked_path(dep_path, tracked_trie):
                            continue
                        # Dependency not found
                        raise exceptions.DependencyNotFoundError(
                            stage=stage_name,
                            dep=dep_path,
                            available_outputs=list(outputs_map.keys()),
                        )

        # Outs: stage -> artifact
        for out in info["outs"]:
            artifact = artifact_node(pathlib.Path(str(out.path)))
            g.add_node(artifact, type=NodeType.ARTIFACT)
            g.add_edge(stage, artifact)

    # Always check for cycles - a cyclic graph is never valid
    _check_acyclic(g)

    return g
```

**Step 3: Run graph tests**

```bash
uv run pytest tests/engine/test_graph.py -v
```

Expected: PASS

**Step 4: Commit**

```bash
jj describe -m "feat(graph): add validation to build_graph (cycles always, deps when validate=True)"
```

---

## Task 3: Move `get_execution_order` to `engine/graph.py`

**Files:**
- Modify: `src/pivot/engine/graph.py`
- Modify: `src/pivot/dag/build.py` (delete moved functions)

**Step 1: Move `get_execution_order` from `dag/build.py:180-214`**

Cut and paste into `engine/graph.py`:

```python
def get_execution_order(
    graph: nx.DiGraph[str],
    stages: list[str] | None = None,
    single_stage: bool = False,
) -> list[str]:
    """Get execution order using DFS postorder traversal.

    Args:
        graph: Stage-only DAG (from get_stage_dag)
        stages: Optional target stages to execute (default: all stages)
        single_stage: If True, run only the specified stages without dependencies.
            Stages are executed in the order provided, not DAG order.

    Returns:
        List of stage names in execution order (dependencies first, unless single_stage)
    """
    if stages:
        if single_stage:
            return stages
        subgraph = _get_subgraph(graph, stages)
        return list(nx.dfs_postorder_nodes(subgraph))

    return list(nx.dfs_postorder_nodes(graph))
```

**Step 2: Move `_get_subgraph` from `dag/build.py:217-222`**

Cut and paste into `engine/graph.py`:

```python
def _get_subgraph(graph: nx.DiGraph[str], source_stages: list[str]) -> nx.DiGraph[str]:
    """Get subgraph containing sources and all their dependencies."""
    nodes = set[str]()
    for stage in source_stages:
        nodes.update(nx.dfs_postorder_nodes(graph, stage))
    return graph.subgraph(nodes)
```

**Step 3: Update `__all__` in `engine/graph.py`**

Add `"get_execution_order"` to `__all__`.

**Step 4: Commit**

```bash
jj describe -m "refactor(graph): move get_execution_order from dag/build.py to engine/graph.py"
```

---

## Task 4: Update all call sites to use `engine_graph`

**Files to modify:**
- `src/pivot/engine/engine.py`
- `src/pivot/registry.py`
- `src/pivot/status.py`
- `src/pivot/cli/repro.py`
- `src/pivot/dag/__init__.py`

**Step 1: Update `engine/engine.py`**

Find line ~460 and replace:

```python
# BEFORE:
dag.build_dag(all_stages, validate=True)

# AFTER: Delete this line entirely - validation now happens in build_graph
```

Find where `build_graph` is called (~line 457) and update:

```python
# BEFORE:
self._graph = engine_graph.build_graph(all_stages)

# AFTER:
from pivot.storage import track
tracked_files = track.discover_pvt_files(project_root)
self._graph = engine_graph.build_graph(all_stages, validate=True, tracked_files=tracked_files)
```

Find `dag.get_execution_order` usage (~line 473) and replace:

```python
# BEFORE:
execution_order = dag.get_execution_order(stage_dag, stages, single_stage=single_stage)

# AFTER:
execution_order = engine_graph.get_execution_order(stage_dag, stages, single_stage=single_stage)
```

Remove `from pivot import dag` import if no longer used.

**Step 2: Update `registry.py`**

Find `build_dag` method (~line 521) and update:

```python
def build_dag(self, validate: bool = True) -> DiGraph[str]:
    """Build DAG from registered stages.

    Returns:
        NetworkX DiGraph with stages as nodes and dependencies as edges
    """
    if validate and self._cached_dag is not None:
        return self._cached_dag

    from pivot.engine import graph as engine_graph
    from pivot.storage import track

    tracked_files = None
    if validate:
        tracked_files = track.discover_pvt_files(project.get_project_root())

    # Build bipartite graph with validation, extract stage DAG
    bipartite = engine_graph.build_graph(
        self._stages,
        validate=validate,
        tracked_files=tracked_files,
    )
    graph = engine_graph.get_stage_dag(bipartite)

    if validate:
        self.validate_outputs()
        self._cached_dag = graph

    return graph
```

Remove `from pivot import dag` import.

**Step 3: Update `status.py`**

Find imports and usages of `dag.get_execution_order` and `dag.build_tracked_trie`:

```python
# BEFORE:
from pivot import dag
# ... later ...
execution_order = dag.get_execution_order(stage_graph, stages, single_stage=single_stage)
tracked_trie = dag.build_tracked_trie(tracked_files)

# AFTER:
from pivot.engine import graph as engine_graph
# ... later ...
execution_order = engine_graph.get_execution_order(stage_graph, stages, single_stage=single_stage)
tracked_trie = engine_graph.build_tracked_trie(tracked_files)
```

**Step 4: Update `cli/repro.py`**

Find line ~303:

```python
# BEFORE:
execution_order = dag.get_execution_order(graph, stages_list, single_stage=False)

# AFTER:
from pivot.engine import graph as engine_graph
execution_order = engine_graph.get_execution_order(graph, stages_list, single_stage=False)
```

**Step 5: Update `dag/__init__.py`**

Replace entire file with rendering-only exports:

```python
from pivot.dag.render import (
    render_ascii,
    render_dot,
    render_mermaid,
)

__all__ = [
    "render_ascii",
    "render_dot",
    "render_mermaid",
]
```

**Step 6: Run tests to find any missed call sites**

```bash
uv run pytest tests/ -v 2>&1 | head -100
```

Fix any import errors that surface.

**Step 7: Commit**

```bash
jj describe -m "refactor: update all call sites to use engine_graph instead of dag"
```

---

## Task 5: Delete `dag/build.py`

**Files:**
- Delete: `src/pivot/dag/build.py`

**Step 1: Verify no remaining imports**

```bash
uv run ruff check . 2>&1 | grep -i "dag.build\|dag\.build_dag\|dag\.get_execution"
```

Expected: No matches (or only test files we'll fix next)

**Step 2: Delete the file**

```bash
rm src/pivot/dag/build.py
```

**Step 3: Commit**

```bash
jj describe -m "chore(dag): delete dag/build.py - consolidated into engine/graph.py"
```

---

## Task 6: Update and move tests

**Files:**
- Modify: `tests/engine/test_graph.py` (add validation tests)
- Modify: `tests/core/test_dag.py` (update imports, remove obsolete tests)

**Step 1: Add validation tests to `test_graph.py`**

Add these tests (adapted from `test_dag.py`):

```python
# --- Validation tests ---

def test_build_graph_raises_on_cycle(tmp_path: Path) -> None:
    """build_graph raises CyclicGraphError when graph has cycles."""
    from pivot import exceptions

    file_a = tmp_path / "a.csv"
    file_b = tmp_path / "b.csv"

    stages = {
        "stage_a": _create_stage("stage_a", [str(file_b)], [str(file_a)]),
        "stage_b": _create_stage("stage_b", [str(file_a)], [str(file_b)]),
    }

    with pytest.raises(exceptions.CyclicGraphError, match="Circular dependency"):
        graph.build_graph(stages)  # Cycles always checked


def test_build_graph_raises_on_missing_dependency(tmp_path: Path) -> None:
    """build_graph raises DependencyNotFoundError when validate=True."""
    from pivot import exceptions

    output_file = tmp_path / "output.csv"
    missing_dep = tmp_path / "missing.csv"

    stages = {
        "stage_a": _create_stage("stage_a", [str(missing_dep)], [str(output_file)]),
    }

    with pytest.raises(exceptions.DependencyNotFoundError):
        graph.build_graph(stages, validate=True)


def test_build_graph_allows_missing_when_validate_false(tmp_path: Path) -> None:
    """build_graph allows missing deps when validate=False."""
    output_file = tmp_path / "output.csv"
    missing_dep = tmp_path / "missing.csv"

    stages = {
        "stage_a": _create_stage("stage_a", [str(missing_dep)], [str(output_file)]),
    }

    # Should not raise
    g = graph.build_graph(stages, validate=False)
    assert "stage:stage_a" in g


def test_build_graph_accepts_tracked_file(tmp_path: Path) -> None:
    """build_graph accepts tracked files as valid dependency sources."""
    output_file = tmp_path / "output.csv"
    tracked_input = tmp_path / "tracked.csv"

    tracked_files = {str(tracked_input): {"hash": "abc123"}}

    stages = {
        "stage_a": _create_stage("stage_a", [str(tracked_input)], [str(output_file)]),
    }

    # Should not raise - tracked file is valid
    g = graph.build_graph(stages, validate=True, tracked_files=tracked_files)
    assert "stage:stage_a" in g


def test_build_graph_directory_dependency(tmp_path: Path) -> None:
    """build_graph resolves directory dependencies via trie."""
    input_file = tmp_path / "input.csv"
    output_dir = tmp_path / "outputs"
    file_a = output_dir / "a.csv"
    input_file.touch()

    stages = {
        "producer": _create_stage("producer", [str(input_file)], [str(file_a)]),
        "consumer": _create_stage("consumer", [str(output_dir)], [str(tmp_path / "final.csv")]),
    }

    # Should not raise - output_dir contains file_a from producer
    g = graph.build_graph(stages, validate=True)
    assert "stage:producer" in g
    assert "stage:consumer" in g
```

**Step 2: Ensure `_create_stage` helper exists**

Check if `tests/engine/test_graph.py` has a `_create_stage` helper. If not, add:

```python
def _create_stage(name: str, deps: list[str], outs: list[str]) -> RegistryStageInfo:
    """Create a minimal RegistryStageInfo for testing."""
    from pivot.loaders import CSV
    from pivot.outputs import Out

    return {
        "func": lambda: None,
        "deps": [],
        "deps_paths": deps,
        "outs": [Out(p, CSV()) for p in outs],
        "outs_paths": outs,
        "params": None,
        "fingerprint": {"hash": "test", "source": "test"},
        "source_file": "test.py",
        "module_name": "test",
    }
```

**Step 3: Update `tests/core/test_dag.py`**

Remove tests for deleted functions. Keep tests that exercise behavior through `Registry.build_dag()` or rendering. Update imports:

```python
# BEFORE:
from pivot import dag
dag.build_dag(...)
dag.get_execution_order(...)

# AFTER:
from pivot.engine import graph as engine_graph
# Use registry.build_dag() for integration tests
# Use engine_graph.build_graph() + get_stage_dag() for unit tests
```

**Step 4: Run full test suite**

```bash
uv run pytest tests/ -v
```

Fix any failures.

**Step 5: Commit**

```bash
jj describe -m "test: update tests for graph consolidation"
```

---

## Task 7: Final verification

**Step 1: Run quality checks**

```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright
```

Expected: PASS

**Step 2: Run full test suite with coverage**

```bash
uv run pytest tests/ -v --cov=src/pivot --cov-report=term-missing
```

Expected: PASS, 90%+ coverage

**Step 3: Verify no dead code**

```bash
# Should find NO matches in src/
grep -r "dag\.build_dag\|from pivot\.dag\.build import\|from pivot import dag" src/
```

**Step 4: Verify `dag/build.py` is gone**

```bash
ls src/pivot/dag/
# Should show only: __init__.py  render.py
```

**Step 5: Final commit**

```bash
jj describe -m "feat(graph): consolidate dag/build.py into engine/graph.py

- Move trie-based directory/tracked file resolution to engine/graph.py
- Add validation (cycle detection always, dep validation when validate=True)
- Move get_execution_order to engine/graph.py
- Update all call sites to use engine_graph directly
- Delete dag/build.py entirely
- dag/ module now only contains rendering utilities

Enables: GitHub issue #323 (cross-pipeline dependency resolution)
"
```

---

## Summary

| Before | After |
|--------|-------|
| `dag/build.py` (245 lines) | Deleted |
| `dag.build_dag()` | `engine_graph.build_graph(validate=True)` + `get_stage_dag()` |
| `dag.get_execution_order()` | `engine_graph.get_execution_order()` |
| `dag.build_tracked_trie()` | `engine_graph.build_tracked_trie()` |
| `dag.get_downstream_stages()` | `engine_graph.get_downstream_stages()` (already existed) |
| `dag/__init__.py` | Rendering exports only |

**LOC change:** ~-200 lines (moved + deleted, no duplication)

**Validation behavior:**
- Cycle detection: **Always runs** (cyclic graph is never valid)
- Dependency validation: Runs when `validate=True`
