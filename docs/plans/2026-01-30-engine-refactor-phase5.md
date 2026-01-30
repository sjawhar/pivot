# Engine Refactor Phase 5: Query API Unification

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify all query operations (status, explain, dry-run, verify) to use Engine's bipartite graph, eliminating duplicate graph construction and ensuring consistent behavior between execution and queries.

**Architecture:** Engine exposes its graph via public property (already implemented in Phase 4). The `status` module builds a stage-only DAG internally via `registry.REGISTRY.build_dag()` - we add an optional `graph` parameter to accept Engine's bipartite graph instead. Note: `explain.py` provides low-level functions that take individual stage data (not a graph) - the graph is used by `status.py` which calls explain functions. A new `hash_artifact()` function encapsulates the disk → .pvt fallback logic. CLI commands that need queries can use `engine_graph.build_graph()` directly or pass through Engine.

**Tech Stack:** Python 3.13+, networkx, TypedDict, Protocol

---

## Prerequisites

Before starting, verify Phase 4 is complete:
```bash
uv run pytest tests/engine/ -v
uv run pytest tests/ -n auto
```

Expected: All tests pass. Engine owns orchestration, WatchEngine deleted.

---

## Task 1: Verify Engine.graph Property Exists

**Files:**
- Read: `src/pivot/engine/engine.py`
- Test: `tests/engine/test_engine.py`

Phase 4 should have added the `graph` property. Verify it's present and returns the bipartite graph.

**Step 1: Write test for graph property**

Add to `tests/engine/test_engine.py`:

```python
def test_engine_graph_property_returns_bipartite_graph() -> None:
    """Engine.graph returns the bipartite artifact-stage graph."""
    eng = engine.Engine()

    # Before any execution, graph may be None
    # After building, should be a networkx DiGraph
    # This test documents the expected behavior

    # Build graph by triggering execution (or calling internal method)
    from pivot.engine import graph as engine_graph

    all_stages = {name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()}
    expected_graph = engine_graph.build_graph(all_stages)

    # Engine should expose this via property
    assert hasattr(eng, "graph")

    # After run_once, graph should be populated
    # (requires registered stages - use fixture if needed)
```

**Step 2: Run test to verify it passes**

```bash
uv run pytest tests/engine/test_engine.py::test_engine_graph_property_returns_bipartite_graph -v
```

Expected: PASS (Phase 4 added this)

**Step 3: Commit**

```bash
jj describe -m "test(engine): verify Engine.graph property exists

Phase 5 prerequisite: confirms Phase 4 added graph property."
```

---

## Task 2: Add hash_artifact() Function to explain.py

**Files:**
- Modify: `src/pivot/explain.py` (add new function)
- Modify: `tests/test_explain.py` (add tests)

Add the hash fallback function to the existing explain module. This encapsulates the disk → .pvt resolution logic that's currently scattered across worker.py and status.py.

**Step 1: Write failing test for hash_artifact**

Create or add to `tests/test_explain.py`:

```python
"""Tests for explain module hash functions."""

from __future__ import annotations

import pathlib

import pytest

from pivot import explain


def test_hash_artifact_existing_file(tmp_path: pathlib.Path) -> None:
    """hash_artifact returns hash for existing file."""
    test_file = tmp_path / "data.csv"
    test_file.write_text("a,b,c\n1,2,3\n")

    result = explain.hash_artifact(test_file)

    assert result is not None
    assert "hash" in result
    assert result["hash"] != ""


def test_hash_artifact_missing_file_no_fallback(tmp_path: pathlib.Path) -> None:
    """hash_artifact returns None for missing file without allow_missing."""
    missing = tmp_path / "nonexistent.csv"

    result = explain.hash_artifact(missing)

    assert result is None


def test_hash_artifact_missing_file_with_pvt_fallback(tmp_path: pathlib.Path) -> None:
    """hash_artifact falls back to .pvt hash when allow_missing=True."""
    import pygtrie

    missing = tmp_path / "data.csv"
    pvt_file = tmp_path / "data.csv.pvt"

    # Create .pvt file with known hash
    pvt_file.write_text("abc123def456")

    # Build tracked trie
    tracked_trie: pygtrie.StringTrie[str] = pygtrie.StringTrie()
    tracked_trie[str(missing)] = "abc123def456"

    result = explain.hash_artifact(
        missing,
        allow_missing=True,
        tracked_trie=tracked_trie,
    )

    assert result is not None
    assert result["hash"] == "abc123def456"


def test_hash_artifact_missing_no_tracked_returns_none(tmp_path: pathlib.Path) -> None:
    """hash_artifact returns None when file missing and not in tracked trie."""
    import pygtrie

    missing = tmp_path / "data.csv"
    tracked_trie: pygtrie.StringTrie[str] = pygtrie.StringTrie()

    result = explain.hash_artifact(
        missing,
        allow_missing=True,
        tracked_trie=tracked_trie,
    )

    assert result is None


def test_hash_artifact_existing_directory(tmp_path: pathlib.Path) -> None:
    """hash_artifact returns hash for existing directory."""
    test_dir = tmp_path / "data_dir"
    test_dir.mkdir()
    (test_dir / "file1.txt").write_text("content1")
    (test_dir / "file2.txt").write_text("content2")

    result = explain.hash_artifact(test_dir)

    assert result is not None
    assert "hash" in result
    assert result["hash"] != ""
    assert result["source"] == "disk"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_explain.py::test_hash_artifact_existing_file -v
```

Expected: FAIL (function doesn't exist yet)

**Step 3: Implement hash_artifact()**

Add to `src/pivot/explain.py`:

```python
"""Explain module: query functions for stage skip detection and explanations."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import pygtrie


class HashInfo(TypedDict):
    """Hash information for an artifact."""

    hash: str
    source: str  # "disk", "pvt", "lockfile"


def hash_artifact(
    path: pathlib.Path,
    allow_missing: bool = False,
    tracked_trie: pygtrie.StringTrie[str] | None = None,
) -> HashInfo | None:
    """Get artifact hash with fallback to .pvt if allow_missing.

    Resolution order:
    1. Actual file on disk (primary)
    2. .pvt file hash (fallback when file missing + allow_missing)

    Args:
        path: Path to the artifact.
        allow_missing: If True, fall back to .pvt hash when file is missing.
        tracked_trie: Trie mapping paths to tracked hashes (from .pvt files).

    Returns:
        HashInfo with hash and source, or None if not found.
    """
    from pivot.storage import cache

    # Primary: file on disk
    if path.exists():
        if path.is_dir():
            file_hash = cache.hash_directory(path)
        else:
            file_hash = cache.hash_file(path)
        return HashInfo(hash=file_hash, source="disk")

    # Fallback: .pvt tracked hash
    if allow_missing and tracked_trie is not None:
        path_str = str(path)
        if path_str in tracked_trie:
            return HashInfo(hash=tracked_trie[path_str], source="pvt")

    return None
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_explain.py -v
```

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/explain.py && uv run ruff check src/pivot/explain.py && uv run basedpyright src/pivot/explain.py
```

**Step 6: Commit**

```bash
jj describe -m "feat(explain): add hash_artifact() with fallback logic

Encapsulates disk → .pvt resolution for query operations.
Used by status, explain, verify to determine artifact hashes."
```

---

## Task 3: Add get_stage_dag() to Engine Graph Module

**Files:**
- Modify: `src/pivot/engine/graph.py`
- Test: `tests/engine/test_graph.py`

Phase 4 should have added `get_stage_dag()`. Verify it exists and works correctly.

**Step 1: Write test for get_stage_dag**

Add to `tests/engine/test_graph.py`:

```python
def test_get_stage_dag_extracts_stage_only_graph() -> None:
    """get_stage_dag() returns stage-only DAG from bipartite graph."""
    from pivot.engine import graph as engine_graph

    # Build bipartite graph
    stages = {
        "preprocess": {
            "deps_paths": [pathlib.Path("input.csv")],
            "outs": [_MockOut(pathlib.Path("cleaned.csv"))],
        },
        "train": {
            "deps_paths": [pathlib.Path("cleaned.csv")],
            "outs": [_MockOut(pathlib.Path("model.pkl"))],
        },
    }
    bipartite = engine_graph.build_graph(stages)

    # Extract stage-only DAG
    stage_dag = engine_graph.get_stage_dag(bipartite)

    # Should have stage nodes (not artifact:... prefixed)
    assert "preprocess" in stage_dag
    assert "train" in stage_dag

    # Should NOT have artifact nodes
    for node in stage_dag.nodes():
        assert not node.startswith("artifact:")
        assert not node.startswith("stage:")

    # Should have edge: preprocess -> train (train depends on preprocess)
    assert stage_dag.has_edge("preprocess", "train")
```

**Step 2: Run test**

```bash
uv run pytest tests/engine/test_graph.py::test_get_stage_dag_extracts_stage_only_graph -v
```

Expected: PASS (Phase 4 added this)

**Step 3: Commit**

```bash
jj describe -m "test(engine): verify get_stage_dag() works correctly

Phase 5 prerequisite: confirms stage DAG extraction from bipartite graph."
```

---

## Task 4: Refactor status.py to Use Engine Graph

**Files:**
- Modify: `src/pivot/status.py`
- Modify: `tests/test_status.py`

Update status module to accept an optional graph parameter instead of always building its own.

**Step 1: Write test for status with engine graph**

Add to `tests/test_status.py`:

```python
def test_get_pipeline_status_uses_provided_graph(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_pipeline_status uses provided graph instead of building one."""
    import networkx as nx

    from pivot import status as status_mod
    from pivot.engine import graph as engine_graph

    # Register a test stage
    registry.REGISTRY.register(
        name="test_stage",
        func=lambda params: {},
        deps=[],
        outs=[],
        deps_paths=[],
        mutex=[],
    )

    # Build graph externally (simulating Engine)
    all_stages = {name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()}
    external_graph = engine_graph.build_graph(all_stages)

    # Call status with provided graph
    # (This tests that status doesn't rebuild the graph internally)
    result = status_mod.get_pipeline_status(
        stages=["test_stage"],
        graph=external_graph,  # New parameter
    )

    assert "test_stage" in result
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_status.py::test_get_pipeline_status_uses_provided_graph -v
```

Expected: FAIL (graph parameter doesn't exist)

**Step 3: Update get_pipeline_status() signature**

The current signature is:
```python
def get_pipeline_status(
    stages: list[str] | None,
    single_stage: bool,
    validate: bool = True,
    allow_missing: bool = False,
) -> tuple[list[PipelineStatusInfo], DiGraph[str]]:
```

Add optional `graph` parameter. Modify `src/pivot/status.py`:

```python
def get_pipeline_status(
    stages: list[str] | None,
    single_stage: bool,
    validate: bool = True,
    allow_missing: bool = False,
    graph: nx.DiGraph[str] | None = None,  # New parameter: bipartite graph from Engine
) -> tuple[list[PipelineStatusInfo], DiGraph[str]]:
    """Get status for all stages, tracking upstream staleness.

    Args:
        stages: Stage names to check, or None for all stages.
        single_stage: If True, check only specified stages without dependencies.
        validate: If True, validate dependency files exist during DAG building.
            Set to False with --allow-missing to skip validation.
        allow_missing: If True, use .pvt hashes for missing dependency files.
        graph: Optional bipartite graph from Engine. If provided, extracts
            stage DAG from it instead of building via registry.build_dag().

    Returns:
        Tuple of (status list, stage DAG used for computation).
    """
    with metrics.timed("status.get_pipeline_status"):
        tracked_files, tracked_trie = _discover_tracked_files(allow_missing)

        if graph is not None:
            # New path: extract stage DAG from Engine's bipartite graph
            from pivot.engine import graph as engine_graph
            stage_graph = engine_graph.get_stage_dag(graph)
        else:
            # Legacy path: build DAG from registry
            stage_graph = registry.REGISTRY.build_dag(validate=validate)

        execution_order = dag.get_execution_order(stage_graph, stages, single_stage=single_stage)
        # ... rest of implementation uses stage_graph ...
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_status.py -v
```

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/status.py && uv run ruff check src/pivot/status.py && uv run basedpyright src/pivot/status.py
```

**Step 6: Commit**

```bash
jj describe -m "feat(status): accept optional graph parameter

get_pipeline_status() can now use Engine's bipartite graph.
Falls back to registry.build_dag() for backwards compatibility."
```

---

## Task 5: Understand explain.py Architecture (No Changes Needed)

**Files:**
- Read: `src/pivot/explain.py`

**Note:** The `explain.py` module provides low-level functions that operate on individual stage data (fingerprint, deps, outs_paths, params, etc.). It does NOT build or use a graph directly.

The graph is used by `status.py`, which:
1. Builds/receives the graph
2. Extracts execution order from the graph
3. Calls `explain.get_stage_explanation()` with stage-specific data

Therefore, **no changes are needed to explain.py** for graph unification. The graph parameter in `status.py` (Task 4) is sufficient.

**Step 1: Verify explain.py signature**

Review `src/pivot/explain.py` to confirm `get_stage_explanation()` takes individual stage parameters:

```python
def get_stage_explanation(
    stage_name: str,
    fingerprint: dict[str, str],
    deps: list[str],
    outs_paths: list[str],
    params_instance: pydantic.BaseModel | None,
    overrides: parameters.ParamsOverrides | None,
    state_dir: Path,
    force: bool = False,
    allow_missing: bool = False,
    tracked_files: dict[str, PvtData] | None = None,
    tracked_trie: pygtrie.Trie[str] | None = None,
) -> StageExplanation:
```

This function receives pre-computed data from `status.py` - no graph needed.

**Step 2: Document in code**

Add a comment to `status.py` clarifying the relationship:

```python
# status.py uses the graph for:
# 1. Getting execution order via dag.get_execution_order()
# 2. Computing upstream staleness relationships
#
# explain.py receives stage-specific data from status.py:
# - No graph needed at explain level
# - Status orchestrates, explain computes per-stage details
```

**Step 3: Commit documentation**

```bash
jj describe -m "docs(status): document relationship with explain module

Clarifies that explain.py operates on stage data, not graphs.
Status orchestrates graph queries; explain computes per-stage details."
```

---

## Task 6: Update cli/verify.py to Use Engine Graph

**Files:**
- Modify: `src/pivot/cli/verify.py` (not `src/pivot/verify.py` - module doesn't exist)
- Test: Manual integration test

Note: Verification logic lives in `src/pivot/cli/verify.py`, not a separate module. The verify command calls `status.get_pipeline_status()` internally.

**Step 1: Review current verify implementation**

```bash
# Check how verify uses status module
```

The verify command at `src/pivot/cli/verify.py` uses:
- `status_mod.get_pipeline_status()` for stage status
- Direct lock file reads for output hash comparison

**Step 2: Verify status.py changes propagate**

Since verify calls `status.get_pipeline_status()`, the graph parameter added in Task 4 is automatically available. No additional changes needed to verify command itself.

**Step 3: Write integration test**

Add to `tests/cli/test_verify.py` (if exists) or verify manually:

```bash
# Create test pipeline
cd /tmp && mkdir test-verify && cd test-verify
cat > pivot.yaml << 'EOF'
stages:
  process:
    cmd: echo "output" > output.txt
    outs:
      - output.txt
EOF

# Run and verify
pivot run
pivot verify
```

**Step 4: Commit**

```bash
jj describe -m "test(cli/verify): verify graph parameter propagates from status

No changes needed - verify uses status.get_pipeline_status() internally."
```

---

## Task 7: Update CLI Status Command to Use Engine Graph

**Files:**
- Modify: `src/pivot/cli/status.py`
- Test: Integration test via CLI

The CLI status command should optionally use Engine's graph when available.

**Step 1: Review current CLI status implementation**

```bash
# Check how status command builds/uses DAG
```

**Step 2: Add graph parameter to CLI status helpers**

For now, CLI commands will continue building the graph internally (no active Engine). The graph parameter is available for future use when we have a persistent Engine instance.

Modify `src/pivot/cli/status.py`:

```python
def _output_status(
    stages: list[str] | None,
    single_stage: bool,
    force: bool,
    allow_missing: bool,
    graph: nx.DiGraph[str] | None = None,
) -> None:
    """Output pipeline status."""
    from pivot import status as status_mod

    result = status_mod.get_pipeline_status(
        stages=stages,
        single_stage=single_stage,
        force=force,
        allow_missing=allow_missing,
        graph=graph,
    )
    # ... output formatting ...
```

**Step 3: Run quality checks**

```bash
uv run ruff format src/pivot/cli/status.py && uv run ruff check src/pivot/cli/status.py && uv run basedpyright src/pivot/cli/status.py
```

**Step 4: Commit**

```bash
jj describe -m "feat(cli/status): add graph parameter to status helpers

Enables future use of Engine graph for status queries."
```

---

## Task 8: Update CLI Run --explain to Use Engine Graph

**Files:**
- Modify: `src/pivot/cli/run.py`
- Test: `tests/cli/test_run.py`

The `--explain` flag should use the same graph that would be used for execution.

**Step 1: Review current --explain implementation**

The `_output_explain()` function in `cli/run.py` calls `status_mod.get_pipeline_explanations()`.

**Step 2: Update _output_explain to build graph once**

Modify `src/pivot/cli/run.py`:

```python
def _output_explain(
    stages_list: list[str] | None,
    single_stage: bool,
    force: bool = False,
    allow_missing: bool = False,
) -> None:
    """Output detailed stage explanations using status logic."""
    from pivot import status as status_mod
    from pivot.cli import status as status_cli
    from pivot.engine import graph as engine_graph

    # Build graph once for all explanations
    all_stages = {name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()}
    graph = engine_graph.build_graph(all_stages)

    explanations = status_mod.get_pipeline_explanations(
        stages_list,
        single_stage,
        force,
        allow_missing=allow_missing,
        graph=graph,  # Pass bipartite graph
    )
    status_cli.output_explain_text(explanations)
```

**Step 3: Update get_pipeline_explanations signature**

Modify `src/pivot/status.py`:

```python
def get_pipeline_explanations(
    stages: list[str] | None = None,
    single_stage: bool = False,
    force: bool = False,
    allow_missing: bool = False,
    graph: nx.DiGraph[str] | None = None,
) -> list[StageExplanation]:
    """Get explanations for all stages.

    Args:
        stages: Stage names to explain (None = all stages).
        single_stage: If True, only explain specified stages.
        force: If True, report all stages as needing to run.
        allow_missing: If True, allow missing tracked files.
        graph: Optional bipartite graph from Engine.

    Returns:
        List of StageExplanation for each stage.
    """
```

**Step 4: Run tests**

```bash
uv run pytest tests/cli/test_run.py -v -k "explain"
```

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/cli/run.py src/pivot/status.py && uv run ruff check src/pivot/cli/run.py src/pivot/status.py && uv run basedpyright src/pivot/cli/run.py src/pivot/status.py
```

**Step 6: Commit**

```bash
jj describe -m "feat(cli/run): --explain builds graph once for all stages

Uses engine_graph.build_graph() to create consistent bipartite graph."
```

---

## Task 9: Update CLI Run --dry-run to Use Engine Graph

**Files:**
- Modify: `src/pivot/cli/run.py`
- Test: `tests/cli/test_run.py`

The `--dry-run` flag shows what would run without executing. It should use the same graph.

**Step 1: Review current --dry-run implementation**

Check how dry-run builds its execution plan.

**Step 2: Update dry-run to use bipartite graph**

The dry-run implementation should use `engine_graph.get_stage_dag()` to get execution order:

```python
def _output_dry_run(
    stages_list: list[str] | None,
    single_stage: bool,
    force: bool,
) -> None:
    """Output what would run without executing."""
    from pivot import dag as dag_mod
    from pivot.engine import graph as engine_graph

    # Build bipartite graph
    all_stages = {name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()}
    bipartite = engine_graph.build_graph(all_stages)

    # Extract stage DAG for execution order
    stage_dag = engine_graph.get_stage_dag(bipartite)

    execution_order = dag_mod.get_execution_order(
        stage_dag,
        stages_list,
        single_stage=single_stage,
    )

    # ... output what would run ...
```

**Step 3: Run tests**

```bash
uv run pytest tests/cli/test_run.py -v -k "dry"
```

**Step 4: Run quality checks**

```bash
uv run ruff format src/pivot/cli/run.py && uv run ruff check src/pivot/cli/run.py && uv run basedpyright src/pivot/cli/run.py
```

**Step 5: Commit**

```bash
jj describe -m "feat(cli/run): --dry-run uses bipartite graph for execution order

Consistent with Engine's graph construction."
```

---

## Task 10: Add get_artifact_consumers() Helper to Engine Graph

**Files:**
- Modify: `src/pivot/engine/graph.py`
- Test: `tests/engine/test_graph.py`

Add helper to find all stages affected by an artifact change (for status queries).

**Step 1: Write test for get_artifact_consumers**

Add to `tests/engine/test_graph.py`:

```python
def test_get_artifact_consumers_returns_direct_and_downstream(
    tmp_path: pathlib.Path,
) -> None:
    """get_artifact_consumers returns stages that depend on artifact."""
    from pivot.engine import graph as engine_graph

    # Build graph: input.csv -> preprocess -> cleaned.csv -> train -> model.pkl
    stages = {
        "preprocess": {
            "deps_paths": [tmp_path / "input.csv"],
            "outs": [_MockOut(tmp_path / "cleaned.csv")],
        },
        "train": {
            "deps_paths": [tmp_path / "cleaned.csv"],
            "outs": [_MockOut(tmp_path / "model.pkl")],
        },
    }
    g = engine_graph.build_graph(stages)

    # Input change should affect both preprocess AND train
    consumers = engine_graph.get_artifact_consumers(
        g, tmp_path / "input.csv", include_downstream=True
    )

    assert "preprocess" in consumers
    assert "train" in consumers  # Downstream of preprocess

    # Without downstream, only direct consumers
    direct = engine_graph.get_artifact_consumers(
        g, tmp_path / "input.csv", include_downstream=False
    )

    assert "preprocess" in direct
    assert "train" not in direct
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_graph.py::test_get_artifact_consumers_returns_direct_and_downstream -v
```

Expected: FAIL (function doesn't exist)

**Step 3: Implement get_artifact_consumers**

Add to `src/pivot/engine/graph.py`:

```python
def get_artifact_consumers(
    g: nx.DiGraph[str],
    path: pathlib.Path,
    include_downstream: bool = True,
) -> list[str]:
    """Get all stages affected by a change to this artifact.

    Args:
        g: The bipartite graph.
        path: Path to the artifact.
        include_downstream: If True, include transitive dependents.

    Returns:
        List of stage names that would be affected.
    """
    # Get direct consumers
    direct = get_consumers(g, path)
    if not direct:
        return []

    if not include_downstream:
        return direct

    # Add downstream stages
    all_affected = set(direct)
    for stage in direct:
        downstream = get_downstream_stages(g, stage)
        all_affected.update(downstream)

    return list(all_affected)
```

**Step 4: Run tests**

```bash
uv run pytest tests/engine/test_graph.py -v
```

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine/graph.py && uv run ruff check src/pivot/engine/graph.py && uv run basedpyright src/pivot/engine/graph.py
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine/graph): add get_artifact_consumers() helper

Returns direct and optionally downstream stages affected by artifact change."
```

---

## Task 11: Update Status to Use get_artifact_consumers

**Files:**
- Modify: `src/pivot/status.py`
- Test: `tests/test_status.py`

Status queries about "what would run if X changed" should use the graph helper.

**Step 1: Add what_if_changed() function to status**

```python
def what_if_changed(
    paths: list[pathlib.Path],
    graph: nx.DiGraph[str] | None = None,
) -> list[str]:
    """Determine which stages would run if these paths changed.

    Args:
        paths: Paths that hypothetically changed.
        graph: Optional bipartite graph from Engine.

    Returns:
        List of stage names that would be affected.
    """
    from pivot.engine import graph as engine_graph

    if graph is None:
        all_stages = {name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()}
        graph = engine_graph.build_graph(all_stages)

    affected = set[str]()
    for path in paths:
        consumers = engine_graph.get_artifact_consumers(graph, path, include_downstream=True)
        affected.update(consumers)

    return list(affected)
```

**Step 2: Run tests**

```bash
uv run pytest tests/test_status.py -v
```

**Step 3: Run quality checks**

```bash
uv run ruff format src/pivot/status.py && uv run ruff check src/pivot/status.py && uv run basedpyright src/pivot/status.py
```

**Step 4: Commit**

```bash
jj describe -m "feat(status): add what_if_changed() for impact analysis

Uses bipartite graph to determine affected stages."
```

---

## Task 12: Document Query API in Module Docstrings

**Files:**
- Modify: `src/pivot/status.py`
- Modify: `src/pivot/explain.py` (hash_artifact docstring only)

Add clear documentation about the graph parameter and module architecture.

**Step 1: Update status.py module docstring**

Add to `src/pivot/status.py`:

```python
"""Pipeline status queries using Engine's bipartite graph.

This module provides the primary query API for determining pipeline status.
It orchestrates calls to explain.py for individual stage explanations.

Graph Parameter
---------------
Query functions accept an optional `graph` parameter:

- If provided, uses the bipartite artifact-stage graph from Engine
- If None, builds a stage-only DAG from the registry (legacy path)

The Engine's bipartite graph is preferred because:
1. It's already built for execution
2. It includes artifact nodes for path-based queries
3. It ensures consistency between queries and execution

Module Relationships
--------------------
- status.py: Orchestrates queries, handles graph, computes execution order
- explain.py: Provides per-stage explanations (receives data from status.py)
- cli/verify.py: Verification commands (uses status.py internally)

Example::

    from pivot.engine import graph as engine_graph
    from pivot import status

    # Build bipartite graph
    all_stages = {name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()}
    graph = engine_graph.build_graph(all_stages)

    # Use graph for status query
    statuses, dag = status.get_pipeline_status(["train"], single_stage=False, graph=graph)
"""
```

**Step 2: Run quality checks**

```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright .
```

**Step 3: Commit**

```bash
jj describe -m "docs: document graph parameter in query modules

Explains when to use Engine graph vs registry DAG."
```

---

## Task 13: Final Verification and Cleanup

**Step 1: Run full test suite**

```bash
uv run pytest tests/ -n auto --cov=pivot --cov-report=term-missing
```

Expected: PASS with coverage ≥90%

**Step 2: Run all quality checks**

```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright .
```

**Step 3: Manual integration test**

```bash
cd /tmp && mkdir test-phase5 && cd test-phase5

# Create simple pipeline
cat > pivot.yaml << 'EOF'
stages:
  process:
    cmd: python -c "print('processed')"
    deps:
      - input.txt
    outs:
      - output.txt
EOF

echo "test data" > input.txt

# Test status uses graph correctly
pivot status

# Test explain uses graph correctly
pivot run --explain

# Test dry-run uses graph correctly
pivot run --dry-run

# Test actual run
pivot run
```

**Step 4: Final commit**

```bash
jj describe -m "feat(engine): complete Phase 5 - Query API Unification

- hash_artifact() encapsulates disk → .pvt fallback logic
- status.py accepts optional graph parameter (explain.py receives data from status)
- CLI commands use bipartite graph via engine_graph.build_graph()
- get_artifact_consumers() for impact analysis
- what_if_changed() for hypothetical queries
- Documented module relationships (status orchestrates, explain computes)

Query operations now share the same graph structure as execution,
ensuring consistent behavior."
```

---

## Summary

After completing Phase 5 (13 tasks):

1. **hash_artifact()** - New function in explain.py encapsulating disk → .pvt fallback

2. **Graph parameter in status.py** - `get_pipeline_status()` and `get_pipeline_explanations()` accept optional bipartite graph. Note: explain.py operates on stage data passed from status.py, not graphs directly.

3. **get_artifact_consumers()** - New helper in engine/graph.py for impact analysis

4. **what_if_changed()** - Query function in status.py for hypothetical changes

5. **CLI integration** - `--explain` and `--dry-run` use `engine_graph.build_graph()` for consistent graph construction

6. **Documentation** - Clear guidance on module relationships (status orchestrates, explain computes per-stage)

**Files changed:**

| File | Action |
|------|--------|
| `src/pivot/explain.py` | Add hash_artifact() function |
| `src/pivot/status.py` | Add graph param to get_pipeline_status/explanations, add what_if_changed() |
| `src/pivot/engine/graph.py` | Add get_artifact_consumers() |
| `src/pivot/cli/run.py` | Update --explain, --dry-run to use bipartite graph |
| `src/pivot/cli/status.py` | Add graph param to helpers |
| `tests/test_explain.py` | Tests for hash_artifact |
| `tests/test_status.py` | Tests for graph param and what_if_changed |
| `tests/engine/test_graph.py` | Tests for get_artifact_consumers |

**Note:** `src/pivot/verify.py` does not exist - verification is in `cli/verify.py` which uses `status.get_pipeline_status()` internally, so it automatically benefits from the graph parameter.

**Architectural benefits:**

- **Consistency**: Queries use same graph structure as execution
- **Efficiency**: Graph built once, reused for multiple queries
- **Testability**: Can inject mock graphs for testing
- **Future-proof**: Ready for persistent Engine instance
