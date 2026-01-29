# Engine Refactor: Event-Driven Artifact-First Architecture

## Overview

This design refactors Pivot's execution model from a stage-centric batch system to an event-driven, artifact-first architecture. The goal is to eliminate divergent code paths between watch/non-watch and TUI/non-TUI modes by routing all execution through a single Engine.

## Problem Statement

The current architecture has behavior branching too early:

1. **CLI layer** (`cli/run.py`): Decides TUI vs plain vs JSON, wires different execution paths
2. **Watch engine** (`watch/engine.py`): Own planning via `_get_affected_stages()`, own reload logic
3. **Executor** (`executor/core.py`): Own planning (DAG order, skip logic) and execution

This causes bugs where watch mode and non-watch mode detect different changes or use different caching strategies.

## Design Principles

1. **Single source of truth**: All execution flows through the Engine
2. **Event-driven**: Changes propagate as events, not batch "plans"
3. **Artifact-first**: The graph represents artifacts and stages; file changes drive execution
4. **Concurrent by default**: No global "executing" state that blocks new events

---

## Event Model

### Input Events (triggers)

```python
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
```

### Output Events (notifications)

```python
class EngineStateChanged(TypedDict):
    type: Literal["engine_state_changed"]
    state: EngineState  # idle | active | shutdown

class PipelineReloaded(TypedDict):
    """Registry was reloaded, DAG structure may have changed."""
    type: Literal["pipeline_reloaded"]
    stages_added: list[str]
    stages_removed: list[str]
    stages_modified: list[str]
    error: str | None

class StageStarted(TypedDict):
    type: Literal["stage_started"]
    stage: str
    index: int
    total: int

class StageCompleted(TypedDict):
    type: Literal["stage_completed"]
    stage: str
    status: StageStatus  # ran | skipped | failed
    reason: str
    duration_ms: float

class LogLine(TypedDict):
    type: Literal["log_line"]
    stage: str
    line: str
    is_stderr: bool
```

---

## Stage Execution States

Stages have individual states (no global "executing" mode):

```python
class StageExecutionState(IntEnum):
    PENDING = 0      # Not yet considered
    BLOCKED = 1      # Waiting for upstream
    READY = 2        # Can run, waiting for worker
    PREPARING = 3    # Pivot clearing outputs
    RUNNING = 4      # Stage function executing
    COMPLETED = 5    # Terminal
```

The IntEnum allows ordered comparisons (e.g., `state >= PREPARING` means execution has begun).

### Output Filtering by State

- **PREPARING**: Silence filesystem events for this stage's outputs (it's Pivot's cleanup)
- **RUNNING**: Defer filesystem events for outputs (collect, don't act yet)
- **COMPLETED**: Process deferred events, compare output hashes, trigger downstream

---

## Engine Architecture

The Engine is the single coordinator. All code paths go through it.

```
Engine (always running when active)
  │
  ├─ Event Queue ◀── sources push events
  │      │
  │      ▼
  ├─ Event Processor (updates stage states)
  │      │
  │      ▼
  ├─ Ready Queue (stages ready to execute)
  │      │
  │      ▼
  └─ Worker Pool (pulls from ready queue, executes)
         │
         ▼
      Completions ──▶ back to Event Processor
```

### Engine States (minimal)

- `idle` - not started
- `active` - processing events and executing
- `shutdown` - draining, no new stages started

### Key Methods

```python
class Engine:
    def run_once(self, stages: list[str] | None = None, force: bool = False) -> dict[str, ExecutionSummary]:
        """Execute stages and return. For 'pivot run' without --watch."""

    def run_loop(self) -> None:
        """Process events until shutdown. For 'pivot run --watch'."""

    def submit(self, event: InputEvent) -> None:
        """Submit an event for processing. Thread-safe."""

    def add_source(self, source: EventSource) -> None:
        """Register an event source."""

    def add_sink(self, sink: EventSink) -> None:
        """Register an event sink."""
```

The key insight: `run_once()` is just `run_loop()` that exits after one cycle. Same engine, same code path.

---

## Event Sources

Sources produce input events. They're simple and stateless.

### Interface

```python
class EventSource(Protocol):
    def start(self, submit: Callable[[InputEvent], None]) -> None:
        """Begin producing events."""

    def stop(self) -> None:
        """Stop producing events."""
```

### Implementations

**FilesystemSource**: Watches paths, reports ALL changes. Engine does filtering.

```python
class FilesystemSource(EventSource):
    def set_watch_paths(self, paths: list[Path]) -> None:
        """Update watched paths. Called by engine when DAG changes."""
```

**OneShotSource**: Emits a single `RunRequested`, then stops. For `pivot run` without `--watch`.

**AgentSource**: Listens on Unix socket, emits `RunRequested` and `CancelRequested`. For RPC/agent control.

### Dynamic Updates

The engine updates sources when the graph changes:

```python
def _handle_pipeline_reload(self) -> None:
    # ... rebuild graph ...
    self._filesystem_source.set_watch_paths(get_watch_paths(self._graph))
```

---

## Event Sinks

Sinks consume output events. They observe but don't affect execution.

### Interface

```python
class EventSink(Protocol):
    def handle(self, event: OutputEvent) -> None:
        """Process an event. Must be non-blocking."""

    def close(self) -> None:
        """Clean up resources."""
```

### Implementations

**ConsoleSink**: Rich-formatted output to terminal.

**TuiSink**: Forwards events to Textual app via queue.

**JsonlSink**: Newline-delimited JSON to stdout.

### Benefit

The engine doesn't know about display modes. It emits events; sinks decide presentation.

---

## Bipartite Artifact-Stage Graph

The graph has two node types: artifacts (files) and stages (functions). This replaces both the current stage-only DAG and any separate path-to-stage index.

### Structure

```
[data.csv] ──→ [preprocess] ──→ [cleaned.csv] ──→ [train] ──→ [model.pkl]
 (artifact)      (stage)         (artifact)       (stage)      (artifact)
```

### Node Naming

```python
def artifact_node(path: Path) -> str:
    return f"artifact:{path}"

def stage_node(name: str) -> str:
    return f"stage:{name}"
```

### Building

```python
def build_graph() -> nx.DiGraph:
    graph = nx.DiGraph()

    for stage_name in registry.REGISTRY.list_stages():
        info = registry.REGISTRY.get(stage_name)
        stage = stage_node(stage_name)
        graph.add_node(stage, type=NodeType.STAGE)

        # Deps: artifact → stage
        for dep_path in info["deps_paths"]:
            artifact = artifact_node(resolve_path(dep_path))
            graph.add_node(artifact, type=NodeType.ARTIFACT)
            graph.add_edge(artifact, stage)

        # Outs: stage → artifact
        for out in info["outs"]:
            artifact = artifact_node(resolve_path(out.path))
            graph.add_node(artifact, type=NodeType.ARTIFACT)
            graph.add_edge(stage, artifact)

    return graph
```

### Queries (Native NetworkX)

```python
def get_consumers(graph: nx.DiGraph, path: Path) -> list[str]:
    """Stages that depend on this artifact."""
    node = artifact_node(path)
    if node not in graph:
        return []
    return [parse_node(n)[1] for n in graph.successors(node)
            if graph.nodes[n]["type"] == NodeType.STAGE]

def get_producer(graph: nx.DiGraph, path: Path) -> str | None:
    """Stage that produces this artifact."""
    node = artifact_node(path)
    if node not in graph:
        return None
    for pred in graph.predecessors(node):
        if graph.nodes[pred]["type"] == NodeType.STAGE:
            return parse_node(pred)[1]
    return None

def get_watch_paths(graph: nx.DiGraph) -> list[Path]:
    """All artifact paths (for filesystem watcher)."""
    return [Path(parse_node(n)[1]) for n in graph.nodes()
            if graph.nodes[n]["type"] == NodeType.ARTIFACT]
```

### Incremental Updates

NetworkX supports efficient incremental updates (all O(1) or O(degree)):

```python
def update_stage(graph: nx.DiGraph, stage_name: str, new_info: RegistryStageInfo) -> None:
    stage = stage_node(stage_name)

    current_deps = {parse_node(n)[1] for n in graph.predecessors(stage)}
    new_deps = {resolve_path(p) for p in new_info["deps_paths"]}

    for removed_dep in current_deps - new_deps:
        artifact = artifact_node(removed_dep)
        graph.remove_edge(artifact, stage)
        if graph.degree(artifact) == 0:
            graph.remove_node(artifact)

    for added_dep in new_deps - current_deps:
        artifact = artifact_node(added_dep)
        graph.add_node(artifact, type=NodeType.ARTIFACT)
        graph.add_edge(artifact, stage)
```

---

## Migration Strategy

### Phase 1: Foundation (No Behavior Change)

**Step 1.1: Event Types**
- Create `pivot/engine/events.py` with all TypedDicts
- Create `pivot/engine/types.py` with StageExecutionState, NodeType

**Step 1.2: Bipartite Graph**
- Create `pivot/engine/graph.py`
- Test it builds same relationships as current DAG

**Step 1.3: Sink Interface**
- Create sinks wrapping existing display code
- `ConsoleSink`, `TuiSink`, `JsonlSink` as adapters

### Phase 2: Engine Core

**Step 2.1: Engine Skeleton**
- Create `pivot/engine/engine.py`
- `run_once()` delegates to existing `executor.run()`

**Step 2.2: Route CLI Through Engine**
- Change `pivot run` to use Engine
- Same behavior, new entry point

**Step 2.3: Engine Manages Stage States**
- Move stage state tracking from executor into Engine
- `StageLifecycle` moves into Engine

### Phase 3: Watch Mode Unification

**Step 3.1: Filesystem Source**
- Create `FilesystemSource` wrapping current watchfiles logic

**Step 3.2: Engine.run_loop()**
- Implement continuous event loop
- Watch mode uses Engine + FilesystemSource

**Step 3.3: Delete WatchEngine**
- Remove `pivot/watch/engine.py`
- Replaced by unified Engine

### Phase 4: Cleanup

**Step 4.1: Simplify executor/core.py**
- Executor becomes: "given a stage, run it, return result"
- No DAG traversal, no state tracking

**Step 4.2: Delete Redundant Code**
- `_execute_greedy()` logic now in Engine
- `StageLifecycle` replaced by Engine events

**Step 4.3: Update Tests**
- Tests go through Engine
- Watch tests use same Engine with FilesystemSource

### Validation at Each Step

| Step | Validation |
|------|------------|
| 1.1-1.3 | Unit tests for new modules. Existing tests pass. |
| 2.1-2.2 | `pivot run` works identically. Integration tests pass. |
| 2.3 | Stage state transitions emit correct events. TUI works. |
| 3.1-3.2 | `pivot run --watch` works identically. |
| 3.3 | Watch mode tests pass with new implementation. |
| 4.x | All tests pass. Code coverage maintained. |

---

## Open Questions

1. **Dry-run / explain mode**: Implemented as synchronous query on Engine state, not event stream. Need to define exact API.

2. **Run history**: Currently written by executor. Should move to Engine or remain separate?

3. **Metrics aggregation**: Currently aggregated from worker results. No change needed, but verify it still works through Engine.

4. **Agent RPC protocol**: Current protocol works, but could be simplified now that Engine handles state. Consider for future iteration.
