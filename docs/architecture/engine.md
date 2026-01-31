# Engine Architecture

The Engine is Pivot's central coordinator for all execution paths. It provides a unified event-driven architecture that eliminates divergent code paths between batch and watch modes.

## Overview

```
                          ┌───────────────────────┐
                          │        Engine         │
                          │                       │
    ┌─────────────────────┤  Event Queue          │
    │                     │        │              │
    ▼                     │        ▼              │───────────────┐
┌──────────────┐          │  Event Processor      │               │
│ Event Sources│──submit──▶                       │               ▼
│              │          │        │              │        ┌──────────────┐
│ Filesystem   │          │        ▼              │        │ Event Sinks  │
│ OneShot      │          │  Stage Orchestration  │──emit──▶              │
│              │          │        │              │        │ Console      │
└──────────────┘          │        ▼              │        │ TUI/JSONL    │
                          │  Worker Pool          │        └──────────────┘
                          └───────────────────────┘
```

## Key Components

### Engine States

The Engine has three states:

| State | Description |
|-------|-------------|
| `IDLE` | Not executing stages |
| `ACTIVE` | Processing events and executing stages |
| `SHUTDOWN` | Draining, no new stages started |

### Stage Execution States

Each stage has its own execution state, enabling parallel execution tracking:

```python
class StageExecutionState(IntEnum):
    PENDING = 0      # Not yet considered (waiting for upstream)
    BLOCKED = 1      # Upstream failed, cannot run
    READY = 2        # Can run, waiting for worker slot
    PREPARING = 3    # Engine clearing outputs
    RUNNING = 4      # Stage function executing
    COMPLETED = 5    # Terminal (ran/skipped/failed)
```

The IntEnum ordering enables comparisons like `state >= PREPARING` for output filtering.

### Event Sources

Sources produce input events via `engine.submit()`:

| Source | Events | Use Case |
|--------|--------|----------|
| `FilesystemSource` | `DataArtifactChanged`, `CodeOrConfigChanged` | Watch mode |
| `OneShotSource` | `RunRequested` | Batch mode |

For RPC control (agent integration), use the Engine's direct methods: `try_start_run()`, `get_execution_status()`, and `request_cancel()`.

### Event Sinks

Sinks consume output events via `sink.handle()`:

| Sink | Output | Use Case |
|------|--------|----------|
| `ConsoleSink` | Rich terminal | Plain CLI mode |
| `TuiSink` | Textual app | TUI mode |
| `JsonlSink` | Newline JSON | Tooling integration |
| `WatchSink` | Engine state | Watch mode state handling |

## Bipartite Graph

The Engine maintains a bipartite graph with artifact and stage nodes:

```
[input.csv] ──▶ [preprocess] ──▶ [cleaned.csv] ──▶ [train] ──▶ [model.pkl]
 (artifact)       (stage)         (artifact)        (stage)      (artifact)
```

### Node Types

```python
class NodeType(Enum):
    ARTIFACT = "artifact"  # Files
    STAGE = "stage"        # Functions
```

### Graph Queries

| Query | Description |
|-------|-------------|
| `get_consumers(graph, path)` | Stages that depend on this artifact |
| `get_producer(graph, path)` | Stage that produces this artifact |
| `get_upstream_stages(graph, stage)` | Dependencies of a stage |
| `get_downstream_stages(graph, stage)` | Stages that depend on this one |
| `get_stage_dag(graph)` | Extract stage-only DAG |
| `get_watch_paths(graph)` | All artifact paths for watching |

## Execution Modes

### Batch Mode (`run_once`)

```python
engine = Engine()
engine.add_sink(ConsoleSink())
results = engine.run_once(stages=["train"])
```

1. Builds bipartite graph
2. Computes execution order
3. Orchestrates parallel execution
4. Returns results dict

### Watch Mode (`run_loop`)

```python
engine = Engine()
engine.add_sink(TuiSink())
engine.add_source(FilesystemSource(watch_paths))
engine.run_loop()  # Blocks until shutdown
```

1. Starts all sources
2. Processes events from queue
3. Executes affected stages
4. Continues until `engine.shutdown()`

## Event Types

### Input Events

| Event | Trigger | Action |
|-------|---------|--------|
| `DataArtifactChanged` | File modified | Run affected stages |
| `CodeOrConfigChanged` | Python/config modified | Reload registry, run all |
| `RunRequested` | CLI/RPC command | Run specified stages |
| `CancelRequested` | User interrupt | Stop starting new stages |

### Output Events

| Event | When | Data |
|-------|------|------|
| `EngineStateChanged` | State transition | New state |
| `StageStarted` | Stage begins | Stage name, index |
| `StageCompleted` | Stage finishes | Status, reason, duration |
| `LogLine` | Stage output | Line, is_stderr |
| `PipelineReloaded` | Registry reload | Added/removed/modified stages |
| `StageStateChanged` | State transition | Stage, old/new state |

## Thread Safety

The Engine is designed for multi-threaded use:

- `submit()` is thread-safe (uses queue)
- `try_start_run()` uses lock for atomic check-and-start
- `get_execution_status()` reads thread-safe state
- `request_cancel()` sets threading.Event

## Agent RPC Integration

The Engine provides methods for external control:

```python
# Atomically start a run
result = engine.try_start_run(run_id, stages, force)

# Query current status
status = engine.get_execution_status(run_id)

# Request cancellation
engine.request_cancel()
```

## Code Locations

| Component | File |
|-----------|------|
| Engine class | `src/pivot/engine/engine.py` |
| Bipartite graph | `src/pivot/engine/graph.py` |
| Event types | `src/pivot/engine/types.py` |
| Event sources | `src/pivot/engine/sources.py` |
| Event sinks | `src/pivot/engine/sinks.py` |

## See Also

- [Execution Model](execution.md) - Parallel execution details
- [Watch Mode](watch.md) - Watch mode specifics
- [Code Tour](code-tour.md) - Code navigation guide
