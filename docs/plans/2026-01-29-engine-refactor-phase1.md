# Engine Refactor Phase 1: Foundation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create the foundational types, event model, and bipartite graph without changing any existing behavior.

**Architecture:** Add new `pivot/engine/` module with typed events, stage execution states, and a bipartite artifact-stage graph built on NetworkX. All additions are pure—nothing uses them yet.

**Tech Stack:** Python 3.13+, NetworkX, TypedDict, IntEnum

---

## Task 1: Create Engine Module Structure

**Files:**
- Create: `src/pivot/engine/__init__.py`
- Create: `src/pivot/engine/types.py`
- Create: `tests/engine/__init__.py`
- Create: `tests/engine/test_types.py`

**Step 1: Create the engine module directory**

```bash
mkdir -p src/pivot/engine tests/engine
```

**Step 2: Write failing test for StageExecutionState**

Create `tests/engine/__init__.py`:
```python
"""Tests for the engine module."""
```

Create `tests/engine/test_types.py`:
```python
"""Tests for engine type definitions."""

from __future__ import annotations

from pivot.engine import types


def test_stage_execution_state_ordering() -> None:
    """Stage states have logical ordering for comparison."""
    assert types.StageExecutionState.PENDING < types.StageExecutionState.BLOCKED
    assert types.StageExecutionState.BLOCKED < types.StageExecutionState.READY
    assert types.StageExecutionState.READY < types.StageExecutionState.PREPARING
    assert types.StageExecutionState.PREPARING < types.StageExecutionState.RUNNING
    assert types.StageExecutionState.RUNNING < types.StageExecutionState.COMPLETED


def test_stage_execution_state_comparison() -> None:
    """Can use >= comparisons for state checks."""
    state = types.StageExecutionState.RUNNING
    assert state >= types.StageExecutionState.PREPARING  # Execution has begun
    assert state < types.StageExecutionState.COMPLETED  # Not done yet


def test_node_type_enum() -> None:
    """NodeType distinguishes artifacts from stages."""
    assert types.NodeType.ARTIFACT.value == "artifact"
    assert types.NodeType.STAGE.value == "stage"


def test_engine_state_enum() -> None:
    """EngineState has idle, active, shutdown states."""
    assert types.EngineState.IDLE.value == "idle"
    assert types.EngineState.ACTIVE.value == "active"
    assert types.EngineState.SHUTDOWN.value == "shutdown"
```

**Step 3: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_types.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pivot.engine'`

**Step 4: Write minimal implementation**

Create `src/pivot/engine/__init__.py`:
```python
"""Engine module for event-driven pipeline execution."""

from __future__ import annotations

from pivot.engine import types

__all__ = ["types"]
```

Create `src/pivot/engine/types.py`:
```python
"""Core type definitions for the engine module."""

from __future__ import annotations

from enum import Enum, IntEnum

__all__ = [
    "StageExecutionState",
    "NodeType",
    "EngineState",
]


class StageExecutionState(IntEnum):
    """Execution state of a single stage.

    States progress forward (with exception of re-triggering after completion).
    IntEnum enables ordered comparisons: state >= PREPARING means execution began.
    """

    PENDING = 0  # Not yet considered
    BLOCKED = 1  # Waiting for upstream stages
    READY = 2  # Can run, waiting for worker
    PREPARING = 3  # Pivot clearing outputs
    RUNNING = 4  # Stage function executing
    COMPLETED = 5  # Terminal (ran/skipped/failed)


class NodeType(Enum):
    """Node type in the bipartite artifact-stage graph."""

    ARTIFACT = "artifact"
    STAGE = "stage"


class EngineState(Enum):
    """Top-level engine state."""

    IDLE = "idle"  # Not started
    ACTIVE = "active"  # Processing events and executing
    SHUTDOWN = "shutdown"  # Draining, no new stages started
```

**Step 5: Run test to verify it passes**

```bash
uv run pytest tests/engine/test_types.py -v
```

Expected: PASS (all 4 tests)

**Step 6: Run quality checks**

```bash
uv run ruff format src/pivot/engine tests/engine && uv run ruff check src/pivot/engine tests/engine && uv run basedpyright src/pivot/engine tests/engine
```

Expected: No errors

**Step 7: Commit**

```bash
jj describe -m "feat(engine): add core type definitions

StageExecutionState (IntEnum), NodeType, EngineState enums."
```

---

## Task 2: Add Input Event Types

**Files:**
- Modify: `src/pivot/engine/types.py`
- Modify: `tests/engine/test_types.py`

**Step 1: Write failing test for input events**

Add to `tests/engine/test_types.py`:
```python
def test_data_artifact_changed_event() -> None:
    """DataArtifactChanged event has required fields."""
    event: types.DataArtifactChanged = {
        "type": "data_artifact_changed",
        "paths": ["/path/to/data.csv", "/path/to/other.csv"],
    }
    assert event["type"] == "data_artifact_changed"
    assert len(event["paths"]) == 2


def test_code_or_config_changed_event() -> None:
    """CodeOrConfigChanged event has required fields."""
    event: types.CodeOrConfigChanged = {
        "type": "code_or_config_changed",
        "paths": ["/path/to/stages.py"],
    }
    assert event["type"] == "code_or_config_changed"


def test_run_requested_event() -> None:
    """RunRequested event has required fields."""
    event: types.RunRequested = {
        "type": "run_requested",
        "stages": ["train", "evaluate"],
        "force": False,
        "reason": "cli",
    }
    assert event["type"] == "run_requested"
    assert event["stages"] == ["train", "evaluate"]
    assert event["force"] is False

    # stages can be None (all stages)
    event_all: types.RunRequested = {
        "type": "run_requested",
        "stages": None,
        "force": True,
        "reason": "agent:run-123",
    }
    assert event_all["stages"] is None


def test_cancel_requested_event() -> None:
    """CancelRequested event has required fields."""
    event: types.CancelRequested = {"type": "cancel_requested"}
    assert event["type"] == "cancel_requested"


def test_input_event_union() -> None:
    """InputEvent is a union of all input event types."""
    # This test verifies the type alias exists and accepts all event types
    events: list[types.InputEvent] = [
        {"type": "data_artifact_changed", "paths": []},
        {"type": "code_or_config_changed", "paths": []},
        {"type": "run_requested", "stages": None, "force": False, "reason": "test"},
        {"type": "cancel_requested"},
    ]
    assert len(events) == 4
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_types.py::test_data_artifact_changed_event -v
```

Expected: FAIL with `AttributeError: module 'pivot.engine.types' has no attribute 'DataArtifactChanged'`

**Step 3: Write minimal implementation**

Add to `src/pivot/engine/types.py` (after the enums, before `__all__`):
```python
from typing import Literal, TypedDict
```

Update `__all__` to include the new types:
```python
__all__ = [
    "StageExecutionState",
    "NodeType",
    "EngineState",
    # Input events
    "DataArtifactChanged",
    "CodeOrConfigChanged",
    "RunRequested",
    "CancelRequested",
    "InputEvent",
]
```

Add after the enums:
```python
# =============================================================================
# Input Events (triggers)
# =============================================================================


class DataArtifactChanged(TypedDict):
    """Dep/Out files changed on disk."""

    type: Literal["data_artifact_changed"]
    paths: list[str]


class CodeOrConfigChanged(TypedDict):
    """Python files or pivot.yaml/pipeline.py changed."""

    type: Literal["code_or_config_changed"]
    paths: list[str]


class RunRequested(TypedDict):
    """Explicit run request from CLI, RPC, or agent."""

    type: Literal["run_requested"]
    stages: list[str] | None  # None = all stages
    force: bool
    reason: str  # "cli", "agent:{run_id}", "watch:initial"


class CancelRequested(TypedDict):
    """Stop scheduling new stages, let running ones complete."""

    type: Literal["cancel_requested"]


InputEvent = DataArtifactChanged | CodeOrConfigChanged | RunRequested | CancelRequested
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_types.py -v
```

Expected: PASS (all 9 tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add input event types

DataArtifactChanged, CodeOrConfigChanged, RunRequested, CancelRequested."
```

---

## Task 3: Add Output Event Types

**Files:**
- Modify: `src/pivot/engine/types.py`
- Modify: `tests/engine/test_types.py`

**Step 1: Write failing test for output events**

Add to `tests/engine/test_types.py`:
```python
from pivot.types import StageStatus


def test_engine_state_changed_event() -> None:
    """EngineStateChanged event has required fields."""
    event: types.EngineStateChanged = {
        "type": "engine_state_changed",
        "state": types.EngineState.ACTIVE,
    }
    assert event["type"] == "engine_state_changed"
    assert event["state"] == types.EngineState.ACTIVE


def test_pipeline_reloaded_event() -> None:
    """PipelineReloaded event has required fields."""
    event: types.PipelineReloaded = {
        "type": "pipeline_reloaded",
        "stages_added": ["new_stage"],
        "stages_removed": ["old_stage"],
        "stages_modified": ["changed_stage"],
        "error": None,
    }
    assert event["type"] == "pipeline_reloaded"
    assert event["stages_added"] == ["new_stage"]
    assert event["error"] is None

    # With error
    event_err: types.PipelineReloaded = {
        "type": "pipeline_reloaded",
        "stages_added": [],
        "stages_removed": [],
        "stages_modified": [],
        "error": "SyntaxError in stages.py",
    }
    assert event_err["error"] == "SyntaxError in stages.py"


def test_stage_started_event() -> None:
    """StageStarted event has required fields."""
    event: types.StageStarted = {
        "type": "stage_started",
        "stage": "train",
        "index": 3,
        "total": 5,
    }
    assert event["type"] == "stage_started"
    assert event["stage"] == "train"
    assert event["index"] == 3
    assert event["total"] == 5


def test_stage_completed_event() -> None:
    """StageCompleted event has required fields."""
    event: types.StageCompleted = {
        "type": "stage_completed",
        "stage": "train",
        "status": StageStatus.RAN,
        "reason": "inputs changed",
        "duration_ms": 1234.5,
    }
    assert event["type"] == "stage_completed"
    assert event["status"] == StageStatus.RAN

    # Skipped stage
    event_skip: types.StageCompleted = {
        "type": "stage_completed",
        "stage": "evaluate",
        "status": StageStatus.SKIPPED,
        "reason": "unchanged",
        "duration_ms": 0.0,
    }
    assert event_skip["status"] == StageStatus.SKIPPED


def test_log_line_event() -> None:
    """LogLine event has required fields."""
    event: types.LogLine = {
        "type": "log_line",
        "stage": "train",
        "line": "Epoch 1/10 loss=0.5",
        "is_stderr": False,
    }
    assert event["type"] == "log_line"
    assert event["is_stderr"] is False

    event_err: types.LogLine = {
        "type": "log_line",
        "stage": "train",
        "line": "Warning: deprecated API",
        "is_stderr": True,
    }
    assert event_err["is_stderr"] is True


def test_output_event_union() -> None:
    """OutputEvent is a union of all output event types."""
    events: list[types.OutputEvent] = [
        {"type": "engine_state_changed", "state": types.EngineState.IDLE},
        {
            "type": "pipeline_reloaded",
            "stages_added": [],
            "stages_removed": [],
            "stages_modified": [],
            "error": None,
        },
        {"type": "stage_started", "stage": "x", "index": 1, "total": 1},
        {
            "type": "stage_completed",
            "stage": "x",
            "status": StageStatus.RAN,
            "reason": "",
            "duration_ms": 0,
        },
        {"type": "log_line", "stage": "x", "line": "", "is_stderr": False},
    ]
    assert len(events) == 5
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_types.py::test_engine_state_changed_event -v
```

Expected: FAIL with `AttributeError: module 'pivot.engine.types' has no attribute 'EngineStateChanged'`

**Step 3: Write minimal implementation**

Update imports at top of `src/pivot/engine/types.py`:
```python
from pivot.types import StageStatus
```

Update `__all__`:
```python
__all__ = [
    "StageExecutionState",
    "NodeType",
    "EngineState",
    # Input events
    "DataArtifactChanged",
    "CodeOrConfigChanged",
    "RunRequested",
    "CancelRequested",
    "InputEvent",
    # Output events
    "EngineStateChanged",
    "PipelineReloaded",
    "StageStarted",
    "StageCompleted",
    "LogLine",
    "OutputEvent",
]
```

Add after input events section:
```python
# =============================================================================
# Output Events (notifications)
# =============================================================================


class EngineStateChanged(TypedDict):
    """Engine transitioned to a new state."""

    type: Literal["engine_state_changed"]
    state: EngineState


class PipelineReloaded(TypedDict):
    """Registry was reloaded, DAG structure may have changed."""

    type: Literal["pipeline_reloaded"]
    stages_added: list[str]
    stages_removed: list[str]
    stages_modified: list[str]
    error: str | None


class StageStarted(TypedDict):
    """A stage began executing."""

    type: Literal["stage_started"]
    stage: str
    index: int
    total: int


class StageCompleted(TypedDict):
    """A stage finished (ran, skipped, or failed)."""

    type: Literal["stage_completed"]
    stage: str
    status: StageStatus
    reason: str
    duration_ms: float


class LogLine(TypedDict):
    """A line of output from a running stage."""

    type: Literal["log_line"]
    stage: str
    line: str
    is_stderr: bool


OutputEvent = EngineStateChanged | PipelineReloaded | StageStarted | StageCompleted | LogLine
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_types.py -v
```

Expected: PASS (all 15 tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add output event types

EngineStateChanged, PipelineReloaded, StageStarted, StageCompleted, LogLine."
```

---

## Task 4: Create Bipartite Graph Module

**Files:**
- Create: `src/pivot/engine/graph.py`
- Create: `tests/engine/test_graph.py`

**Step 1: Write failing test for node helpers**

Create `tests/engine/test_graph.py`:
```python
"""Tests for the bipartite artifact-stage graph."""

from __future__ import annotations

from pathlib import Path

from pivot.engine import graph, types


# --- Node naming tests ---


def test_artifact_node_creates_prefixed_string() -> None:
    """artifact_node creates 'artifact:' prefixed string."""
    node = graph.artifact_node(Path("/data/input.csv"))
    assert node == "artifact:/data/input.csv"


def test_stage_node_creates_prefixed_string() -> None:
    """stage_node creates 'stage:' prefixed string."""
    node = graph.stage_node("train")
    assert node == "stage:train"


def test_parse_node_extracts_type_and_value() -> None:
    """parse_node extracts NodeType and value from prefixed string."""
    node_type, value = graph.parse_node("artifact:/data/input.csv")
    assert node_type == types.NodeType.ARTIFACT
    assert value == "/data/input.csv"

    node_type, value = graph.parse_node("stage:train")
    assert node_type == types.NodeType.STAGE
    assert value == "train"


def test_parse_node_handles_colons_in_path() -> None:
    """parse_node handles paths with colons (Windows, URLs)."""
    node_type, value = graph.parse_node("artifact:C:/data/input.csv")
    assert node_type == types.NodeType.ARTIFACT
    assert value == "C:/data/input.csv"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_graph.py::test_artifact_node_creates_prefixed_string -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pivot.engine.graph'`

**Step 3: Write minimal implementation**

Create `src/pivot/engine/graph.py`:
```python
"""Bipartite artifact-stage graph built on NetworkX."""

from __future__ import annotations

from pathlib import Path

from pivot.engine.types import NodeType

__all__ = [
    "artifact_node",
    "stage_node",
    "parse_node",
]


def artifact_node(path: Path) -> str:
    """Create artifact node ID from path."""
    return f"artifact:{path}"


def stage_node(name: str) -> str:
    """Create stage node ID from name."""
    return f"stage:{name}"


def parse_node(node: str) -> tuple[NodeType, str]:
    """Extract NodeType and value from node ID.

    Handles colons in paths by only splitting on the first colon.
    """
    prefix, value = node.split(":", 1)
    return NodeType(prefix), value
```

Update `src/pivot/engine/__init__.py`:
```python
"""Engine module for event-driven pipeline execution."""

from __future__ import annotations

from pivot.engine import graph, types

__all__ = ["graph", "types"]
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_graph.py -v
```

Expected: PASS (all 4 tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add graph node helpers

artifact_node, stage_node, parse_node for bipartite graph."
```

---

## Task 5: Add Graph Building Function

**Files:**
- Modify: `src/pivot/engine/graph.py`
- Modify: `tests/engine/test_graph.py`

**Step 1: Write failing test for build_graph**

Add to `tests/engine/test_graph.py`:
```python
import networkx as nx
import pytest

from pivot import loaders, outputs
from pivot.registry import REGISTRY, RegistryStageInfo


def _create_stage(name: str, deps: list[str], outs: list[str]) -> RegistryStageInfo:
    """Create a stage dict for testing."""
    return RegistryStageInfo(
        func=lambda: None,
        name=name,
        deps={f"_{i}": d for i, d in enumerate(deps)},
        deps_paths=deps,
        outs=[outputs.Out(path=out, loader=loaders.PathOnly()) for out in outs],
        outs_paths=outs,
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        dep_specs={},
        out_specs={},
        params_arg_name=None,
    )


# --- Graph building tests ---


@pytest.fixture
def clean_registry() -> None:
    """Ensure registry is clean before and after test."""
    REGISTRY.clear()
    yield
    REGISTRY.clear()


@pytest.mark.usefixtures("clean_registry")
def test_build_graph_simple_chain(tmp_path: Path) -> None:
    """Build bipartite graph for simple chain: input -> A -> intermediate -> B -> output."""
    input_file = tmp_path / "input.csv"
    intermediate = tmp_path / "intermediate.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    # Register stages directly (bypass REGISTRY for isolated test)
    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(intermediate)]),
        "stage_b": _create_stage("stage_b", [str(intermediate)], [str(output_file)]),
    }

    g = graph.build_graph(stages)

    # Check we have both stage and artifact nodes
    stage_nodes = [n for n in g.nodes() if g.nodes[n]["type"] == types.NodeType.STAGE]
    artifact_nodes = [n for n in g.nodes() if g.nodes[n]["type"] == types.NodeType.ARTIFACT]

    assert len(stage_nodes) == 2
    assert len(artifact_nodes) == 3  # input, intermediate, output

    # Check edges: artifact -> stage (consumed by) and stage -> artifact (produces)
    assert g.has_edge(graph.artifact_node(input_file), graph.stage_node("stage_a"))
    assert g.has_edge(graph.stage_node("stage_a"), graph.artifact_node(intermediate))
    assert g.has_edge(graph.artifact_node(intermediate), graph.stage_node("stage_b"))
    assert g.has_edge(graph.stage_node("stage_b"), graph.artifact_node(output_file))


@pytest.mark.usefixtures("clean_registry")
def test_build_graph_diamond(tmp_path: Path) -> None:
    """Build bipartite graph for diamond pattern.

    input -> preprocess -> clean
          -> features -> feats
    clean + feats -> train -> model
    """
    input_file = tmp_path / "input.csv"
    clean = tmp_path / "clean.csv"
    feats = tmp_path / "feats.csv"
    model = tmp_path / "model.pkl"
    input_file.touch()

    stages = {
        "preprocess": _create_stage("preprocess", [str(input_file)], [str(clean)]),
        "features": _create_stage("features", [str(input_file)], [str(feats)]),
        "train": _create_stage("train", [str(clean), str(feats)], [str(model)]),
    }

    g = graph.build_graph(stages)

    # Both preprocess and features consume input
    assert g.has_edge(graph.artifact_node(input_file), graph.stage_node("preprocess"))
    assert g.has_edge(graph.artifact_node(input_file), graph.stage_node("features"))

    # Train consumes both clean and feats
    assert g.has_edge(graph.artifact_node(clean), graph.stage_node("train"))
    assert g.has_edge(graph.artifact_node(feats), graph.stage_node("train"))


@pytest.mark.usefixtures("clean_registry")
def test_build_graph_empty() -> None:
    """Build graph with no stages returns empty graph."""
    g = graph.build_graph({})
    assert len(g.nodes()) == 0
    assert len(g.edges()) == 0
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_graph.py::test_build_graph_simple_chain -v
```

Expected: FAIL with `AttributeError: module 'pivot.engine.graph' has no attribute 'build_graph'`

**Step 3: Write minimal implementation**

Add imports to `src/pivot/engine/graph.py`:
```python
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from pivot.registry import RegistryStageInfo
```

Update `__all__`:
```python
__all__ = [
    "artifact_node",
    "stage_node",
    "parse_node",
    "build_graph",
]
```

Add the function:
```python
def build_graph(stages: dict[str, RegistryStageInfo]) -> nx.DiGraph:
    """Build bipartite artifact-stage graph from stage definitions.

    Args:
        stages: Dict mapping stage name to RegistryStageInfo.

    Returns:
        Directed graph where:
        - Nodes are either artifacts (files) or stages (functions)
        - Edges go: artifact -> stage (consumed by) and stage -> artifact (produces)
    """
    g: nx.DiGraph[str] = nx.DiGraph()

    for stage_name, info in stages.items():
        stage = stage_node(stage_name)
        g.add_node(stage, type=NodeType.STAGE)

        # Deps: artifact -> stage
        for dep_path in info["deps_paths"]:
            artifact = artifact_node(Path(dep_path))
            g.add_node(artifact, type=NodeType.ARTIFACT)
            g.add_edge(artifact, stage)

        # Outs: stage -> artifact
        for out in info["outs"]:
            artifact = artifact_node(Path(str(out.path)))
            g.add_node(artifact, type=NodeType.ARTIFACT)
            g.add_edge(stage, artifact)

    return g
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_graph.py -v
```

Expected: PASS (all 7 tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add build_graph for bipartite graph

Creates artifact and stage nodes with directed edges for data flow."
```

---

## Task 6: Add Graph Query Functions

**Files:**
- Modify: `src/pivot/engine/graph.py`
- Modify: `tests/engine/test_graph.py`

**Step 1: Write failing tests for query functions**

Add to `tests/engine/test_graph.py`:
```python
@pytest.mark.usefixtures("clean_registry")
def test_get_consumers_returns_dependent_stages(tmp_path: Path) -> None:
    """get_consumers returns stages that depend on an artifact."""
    input_file = tmp_path / "input.csv"
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(out_a)]),
        "stage_b": _create_stage("stage_b", [str(input_file)], [str(out_b)]),
    }

    g = graph.build_graph(stages)
    consumers = graph.get_consumers(g, input_file)

    assert set(consumers) == {"stage_a", "stage_b"}


@pytest.mark.usefixtures("clean_registry")
def test_get_consumers_returns_empty_for_unknown_path(tmp_path: Path) -> None:
    """get_consumers returns empty list for unknown path."""
    g = graph.build_graph({})
    consumers = graph.get_consumers(g, tmp_path / "unknown.csv")
    assert consumers == []


@pytest.mark.usefixtures("clean_registry")
def test_get_producer_returns_producing_stage(tmp_path: Path) -> None:
    """get_producer returns the stage that produces an artifact."""
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(output_file)]),
    }

    g = graph.build_graph(stages)
    producer = graph.get_producer(g, output_file)

    assert producer == "stage_a"


@pytest.mark.usefixtures("clean_registry")
def test_get_producer_returns_none_for_input_artifact(tmp_path: Path) -> None:
    """get_producer returns None for artifacts that are inputs (not produced by any stage)."""
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(output_file)]),
    }

    g = graph.build_graph(stages)
    producer = graph.get_producer(g, input_file)

    assert producer is None


@pytest.mark.usefixtures("clean_registry")
def test_get_watch_paths_returns_all_artifacts(tmp_path: Path) -> None:
    """get_watch_paths returns all artifact paths."""
    input_file = tmp_path / "input.csv"
    intermediate = tmp_path / "intermediate.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(intermediate)]),
        "stage_b": _create_stage("stage_b", [str(intermediate)], [str(output_file)]),
    }

    g = graph.build_graph(stages)
    paths = graph.get_watch_paths(g)

    assert set(paths) == {input_file, intermediate, output_file}


@pytest.mark.usefixtures("clean_registry")
def test_get_downstream_stages(tmp_path: Path) -> None:
    """get_downstream_stages returns all transitively downstream stages."""
    input_file = tmp_path / "input.csv"
    intermediate = tmp_path / "intermediate.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(intermediate)]),
        "stage_b": _create_stage("stage_b", [str(intermediate)], [str(output_file)]),
    }

    g = graph.build_graph(stages)
    downstream = graph.get_downstream_stages(g, "stage_a")

    assert set(downstream) == {"stage_b"}


@pytest.mark.usefixtures("clean_registry")
def test_get_downstream_stages_empty_for_leaf(tmp_path: Path) -> None:
    """get_downstream_stages returns empty for leaf stage."""
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(output_file)]),
    }

    g = graph.build_graph(stages)
    downstream = graph.get_downstream_stages(g, "stage_a")

    assert downstream == []
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_graph.py::test_get_consumers_returns_dependent_stages -v
```

Expected: FAIL with `AttributeError: module 'pivot.engine.graph' has no attribute 'get_consumers'`

**Step 3: Write minimal implementation**

Update `__all__` in `src/pivot/engine/graph.py`:
```python
__all__ = [
    "artifact_node",
    "stage_node",
    "parse_node",
    "build_graph",
    "get_consumers",
    "get_producer",
    "get_watch_paths",
    "get_downstream_stages",
]
```

Add the functions:
```python
def get_consumers(g: nx.DiGraph, path: Path) -> list[str]:
    """Get stages that depend on this artifact.

    Args:
        g: The bipartite graph.
        path: Path to the artifact.

    Returns:
        List of stage names that consume this artifact.
    """
    node = artifact_node(path)
    if node not in g:
        return []
    return [
        parse_node(n)[1]
        for n in g.successors(node)
        if g.nodes[n]["type"] == NodeType.STAGE
    ]


def get_producer(g: nx.DiGraph, path: Path) -> str | None:
    """Get the stage that produces this artifact.

    Args:
        g: The bipartite graph.
        path: Path to the artifact.

    Returns:
        Stage name that produces this artifact, or None if it's an input.
    """
    node = artifact_node(path)
    if node not in g:
        return None
    for pred in g.predecessors(node):
        if g.nodes[pred]["type"] == NodeType.STAGE:
            return parse_node(pred)[1]
    return None


def get_watch_paths(g: nx.DiGraph) -> list[Path]:
    """Get all artifact paths (for filesystem watcher).

    Args:
        g: The bipartite graph.

    Returns:
        List of all artifact paths in the graph.
    """
    return [
        Path(parse_node(n)[1])
        for n in g.nodes()
        if g.nodes[n]["type"] == NodeType.ARTIFACT
    ]


def get_downstream_stages(g: nx.DiGraph, stage_name: str) -> list[str]:
    """Get all stages transitively downstream of this one.

    Args:
        g: The bipartite graph.
        stage_name: Name of the stage.

    Returns:
        List of stage names that transitively depend on this stage's outputs.
    """
    node = stage_node(stage_name)
    if node not in g:
        return []

    downstream = []
    for descendant in nx.descendants(g, node):
        if g.nodes[descendant]["type"] == NodeType.STAGE:
            downstream.append(parse_node(descendant)[1])
    return downstream
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_graph.py -v
```

Expected: PASS (all 14 tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add graph query functions

get_consumers, get_producer, get_watch_paths, get_downstream_stages."
```

---

## Task 7: Add Incremental Graph Update Function

**Files:**
- Modify: `src/pivot/engine/graph.py`
- Modify: `tests/engine/test_graph.py`

**Step 1: Write failing tests for update_stage**

Add to `tests/engine/test_graph.py`:
```python
@pytest.mark.usefixtures("clean_registry")
def test_update_stage_adds_new_dep(tmp_path: Path) -> None:
    """update_stage adds new dependency edges."""
    input_a = tmp_path / "a.csv"
    input_b = tmp_path / "b.csv"
    output_file = tmp_path / "output.csv"
    input_a.touch()
    input_b.touch()

    # Initial: stage_a depends on input_a
    stages = {
        "stage_a": _create_stage("stage_a", [str(input_a)], [str(output_file)]),
    }
    g = graph.build_graph(stages)

    assert graph.get_consumers(g, input_a) == ["stage_a"]
    assert graph.get_consumers(g, input_b) == []

    # Update: stage_a now also depends on input_b
    new_info = _create_stage("stage_a", [str(input_a), str(input_b)], [str(output_file)])
    graph.update_stage(g, "stage_a", new_info)

    assert set(graph.get_consumers(g, input_a)) == {"stage_a"}
    assert set(graph.get_consumers(g, input_b)) == {"stage_a"}


@pytest.mark.usefixtures("clean_registry")
def test_update_stage_removes_old_dep(tmp_path: Path) -> None:
    """update_stage removes old dependency edges and orphaned artifacts."""
    input_a = tmp_path / "a.csv"
    input_b = tmp_path / "b.csv"
    output_file = tmp_path / "output.csv"
    input_a.touch()
    input_b.touch()

    # Initial: stage_a depends on both inputs
    stages = {
        "stage_a": _create_stage("stage_a", [str(input_a), str(input_b)], [str(output_file)]),
    }
    g = graph.build_graph(stages)

    # Update: stage_a now only depends on input_a
    new_info = _create_stage("stage_a", [str(input_a)], [str(output_file)])
    graph.update_stage(g, "stage_a", new_info)

    assert graph.get_consumers(g, input_a) == ["stage_a"]
    assert graph.get_consumers(g, input_b) == []

    # input_b should be removed from graph (orphaned)
    assert graph.artifact_node(input_b) not in g


@pytest.mark.usefixtures("clean_registry")
def test_update_stage_preserves_shared_artifacts(tmp_path: Path) -> None:
    """update_stage doesn't remove artifacts used by other stages."""
    shared_input = tmp_path / "shared.csv"
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    shared_input.touch()

    # Both stages depend on shared_input
    stages = {
        "stage_a": _create_stage("stage_a", [str(shared_input)], [str(out_a)]),
        "stage_b": _create_stage("stage_b", [str(shared_input)], [str(out_b)]),
    }
    g = graph.build_graph(stages)

    # Update stage_a to have no deps - shared_input should remain (used by stage_b)
    new_info = _create_stage("stage_a", [], [str(out_a)])
    graph.update_stage(g, "stage_a", new_info)

    # shared_input still in graph
    assert graph.artifact_node(shared_input) in g
    assert graph.get_consumers(g, shared_input) == ["stage_b"]
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_graph.py::test_update_stage_adds_new_dep -v
```

Expected: FAIL with `AttributeError: module 'pivot.engine.graph' has no attribute 'update_stage'`

**Step 3: Write minimal implementation**

Update `__all__`:
```python
__all__ = [
    "artifact_node",
    "stage_node",
    "parse_node",
    "build_graph",
    "get_consumers",
    "get_producer",
    "get_watch_paths",
    "get_downstream_stages",
    "update_stage",
]
```

Add the function:
```python
def update_stage(g: nx.DiGraph, stage_name: str, new_info: RegistryStageInfo) -> None:
    """Incrementally update graph when a stage's definition changes.

    Efficiently diffs current and new deps/outs, adding and removing edges
    as needed. Removes orphaned artifact nodes (no longer connected to any stage).

    Args:
        g: The bipartite graph to modify in place.
        stage_name: Name of the stage to update.
        new_info: New stage definition from registry.
    """
    stage = stage_node(stage_name)

    # Get current deps and outs from graph
    current_deps = {
        Path(parse_node(n)[1])
        for n in g.predecessors(stage)
        if g.nodes[n]["type"] == NodeType.ARTIFACT
    }
    current_outs = {
        Path(parse_node(n)[1])
        for n in g.successors(stage)
        if g.nodes[n]["type"] == NodeType.ARTIFACT
    }

    # Get new deps and outs from info
    new_deps = {Path(p) for p in new_info["deps_paths"]}
    new_outs = {Path(str(out.path)) for out in new_info["outs"]}

    # Remove old deps
    for removed_dep in current_deps - new_deps:
        artifact = artifact_node(removed_dep)
        g.remove_edge(artifact, stage)
        if g.degree(artifact) == 0:
            g.remove_node(artifact)

    # Add new deps
    for added_dep in new_deps - current_deps:
        artifact = artifact_node(added_dep)
        if artifact not in g:
            g.add_node(artifact, type=NodeType.ARTIFACT)
        g.add_edge(artifact, stage)

    # Remove old outs
    for removed_out in current_outs - new_outs:
        artifact = artifact_node(removed_out)
        g.remove_edge(stage, artifact)
        if g.degree(artifact) == 0:
            g.remove_node(artifact)

    # Add new outs
    for added_out in new_outs - current_outs:
        artifact = artifact_node(added_out)
        if artifact not in g:
            g.add_node(artifact, type=NodeType.ARTIFACT)
        g.add_edge(stage, artifact)
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_graph.py -v
```

Expected: PASS (all 17 tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add update_stage for incremental graph updates

Efficiently diffs deps/outs, removes orphaned artifact nodes."
```

---

## Task 8: Final Integration Test and Cleanup

**Files:**
- Modify: `tests/engine/test_graph.py`

**Step 1: Write integration test comparing to current DAG**

Add to `tests/engine/test_graph.py`:
```python
from pivot import dag as legacy_dag


@pytest.mark.usefixtures("clean_registry")
def test_graph_consistent_with_legacy_dag(tmp_path: Path) -> None:
    """Bipartite graph encodes same relationships as legacy stage-only DAG."""
    # Build a pipeline with multiple stages
    input_file = tmp_path / "input.csv"
    clean = tmp_path / "clean.csv"
    feats = tmp_path / "feats.csv"
    model = tmp_path / "model.pkl"
    input_file.touch()

    stages = {
        "preprocess": _create_stage("preprocess", [str(input_file)], [str(clean)]),
        "features": _create_stage("features", [str(input_file)], [str(feats)]),
        "train": _create_stage("train", [str(clean), str(feats)], [str(model)]),
    }

    # Build both graphs
    bipartite = graph.build_graph(stages)
    legacy = legacy_dag.build_dag(stages)

    # Verify same stage relationships
    # Legacy DAG has edge (consumer -> producer), meaning train -> preprocess, train -> features
    for stage_name in stages:
        legacy_deps = set(legacy.successors(stage_name))  # Stages this depends on
        bipartite_deps = set()
        for dep_path in stages[stage_name]["deps_paths"]:
            producer = graph.get_producer(bipartite, Path(dep_path))
            if producer:
                bipartite_deps.add(producer)

        assert legacy_deps == bipartite_deps, f"Mismatch for {stage_name}"
```

**Step 2: Run test to verify it passes**

```bash
uv run pytest tests/engine/test_graph.py::test_graph_consistent_with_legacy_dag -v
```

Expected: PASS

**Step 3: Run full test suite to ensure no regressions**

```bash
uv run pytest tests/engine/ -v
```

Expected: PASS (all 18 tests)

**Step 4: Run all quality checks**

```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright .
```

Expected: No errors

**Step 5: Run full test suite**

```bash
uv run pytest tests/ -n auto
```

Expected: All existing tests still pass

**Step 6: Final commit**

```bash
jj describe -m "test(engine): add integration test comparing to legacy DAG

Verifies bipartite graph encodes same stage relationships."
```

---

## Summary

After completing Phase 1, you will have:

1. **`src/pivot/engine/types.py`** - Core types:
   - `StageExecutionState` (IntEnum)
   - `NodeType`, `EngineState` (Enum)
   - Input events: `DataArtifactChanged`, `CodeOrConfigChanged`, `RunRequested`, `CancelRequested`
   - Output events: `EngineStateChanged`, `PipelineReloaded`, `StageStarted`, `StageCompleted`, `LogLine`

2. **`src/pivot/engine/graph.py`** - Bipartite graph:
   - Node helpers: `artifact_node`, `stage_node`, `parse_node`
   - Building: `build_graph`
   - Queries: `get_consumers`, `get_producer`, `get_watch_paths`, `get_downstream_stages`
   - Updates: `update_stage`

3. **`tests/engine/`** - Comprehensive tests for all new code

**No existing behavior has changed.** The new module is purely additive and unused by the rest of the codebase. Phase 2 will wire up the Engine to use these foundations.
