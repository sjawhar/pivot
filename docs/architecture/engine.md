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

The Engine has two states:

| State | Description |
|-------|-------------|
| `IDLE` | Not executing stages |
| `ACTIVE` | Processing events and executing stages |

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

Sources push input events via memory channels (`MemoryObjectSendStream[InputEvent]`):

| Source | Events | Use Case |
|--------|--------|----------|
| `FilesystemSource` | `DataArtifactChanged`, `CodeOrConfigChanged` | Watch mode |
| `OneShotSource` | `RunRequested` | Batch mode |
| `AgentRpcSource` | `RunRequested`, `CancelRequested` | Agent RPC control |

For RPC control (agent integration), use `AgentRpcSource` which converts JSON-RPC commands into input events.

### Event Sinks

Sinks consume output events via `sink.handle()`:

| Sink | Output | Use Case |
|------|--------|----------|
| `ConsoleSink` | Rich terminal | Plain CLI mode |
| `TuiSink` | Textual app | TUI mode |
| `JsonlSink` | Newline JSON | Tooling integration |
| `WatchSink` | Engine state | Watch mode state handling |
| `ResultCollectorSink` | Dict collection | Programmatic result access |

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

Both batch and watch modes use the same `run()` method with the `exit_on_completion` parameter:

### Batch Mode (`exit_on_completion=True`)

```python
async with Engine() as engine:
    collector = ResultCollectorSink()
    engine.add_sink(collector)
    engine.add_sink(ConsoleSink())
    engine.add_source(OneShotSource(stages=["train"], force=True, reason="cli"))

    await engine.run(exit_on_completion=True)

    results = await collector.get_results()
```

1. Builds bipartite graph
2. Computes execution order
3. Orchestrates parallel execution
4. Exits when all requested stages complete

### Watch Mode (`exit_on_completion=False`)

```python
async with Engine() as engine:
    engine.add_sink(TuiSink())
    engine.add_source(FilesystemSource(watch_paths))

    await engine.run(exit_on_completion=False)  # Blocks until shutdown
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
| `PipelineReloaded` | Registry reload | Stages list, added/removed/modified |
| `StageStateChanged` | State transition | Stage, old/new state |

## Async Safety

The Engine uses structured concurrency with anyio:

- All state access occurs within the event loop task in `run()`
- Sources run in separate tasks but only send events to channels—they don't access engine state
- Memory channels provide implicit serialization, so no explicit locks are needed
- Cancellation uses `anyio.Event`, not `threading.Event`

## Agent RPC Integration

Agent RPC control uses event sources and handlers, not direct Engine methods:

```python
from pivot.engine.agent_rpc import AgentRpcSource, AgentRpcHandler

# Create handler for status/stages queries
handler = AgentRpcHandler(engine=engine)

# Add RPC source to Engine (converts JSON-RPC to events)
engine.add_source(AgentRpcSource(socket_path=socket_path, handler=handler))
```

## Serve Mode

For headless daemon operation (`pivot run --serve --watch`), the Engine supports RPC sources:

```python
async with Engine(pipeline=pipeline) as engine:
    engine.add_source(FilesystemSource(watch_paths=paths))
    engine.add_source(AgentRpcSource(socket_path=socket_path, handler=handler))
    engine.add_sink(AgentEventSink())

    await engine.run(exit_on_completion=False)
```

### Serve Mode Components

| Component | Purpose |
|-----------|---------|
| `AgentRpcSource` | JSON-RPC 2.0 over Unix socket |
| `AgentEventSink` | Broadcast events to subscribed clients |

### Agent RPC Protocol

The `AgentRpcSource` implements JSON-RPC 2.0 over Unix socket:

**Commands** (become input events):
- `run` - Start a run with optional stages/force
- `cancel` - Request cancellation

**Queries** (handled by `AgentRpcHandler`):
- `status` - Get engine state (idle/active)
- `stages` - List registered stages

```json
{"jsonrpc": "2.0", "method": "run", "params": {"stages": ["train"]}, "id": 1}
{"jsonrpc": "2.0", "result": "accepted", "id": 1}
```

### Event Broadcasting

`AgentEventSink` provides pub-sub event delivery to connected agents:

```python
# Subscribe a client
recv = event_sink.subscribe("client_id")

# Receive events
async for event in recv:
    process(event)

# Unsubscribe when done
event_sink.unsubscribe("client_id")
```

**Backpressure Handling:** If a client's buffer is full, events are dropped silently with a debug log. Clients should process events quickly or increase buffer size.

## Code Locations

| Component | File |
|-----------|------|
| Engine class | `src/pivot/engine/engine.py` |
| Bipartite graph | `src/pivot/engine/graph.py` |
| Event types | `src/pivot/engine/types.py` |
| Event sources | `src/pivot/engine/sources.py` |
| Event sinks | `src/pivot/engine/sinks.py` |
| Agent RPC | `src/pivot/engine/agent_rpc.py` |

## See Also

- [Execution Model](execution.md) - Parallel execution details
- [Watch Mode](watch.md) - Watch mode specifics
- [Code Tour](code-tour.md) - Code navigation guide
