# Engine Architecture

The Engine is Pivot's central coordinator for all execution paths. It provides a unified event-driven architecture that eliminates divergent code paths between batch and watch modes.

## Overview

```
                          ┌──────────────────────────┐
                          │          Engine          │
                          │  (async coordinator)     │
    ┌─────────────────────┤  Input channel           │
    │                     │        │                 │
    ▼                     │        ▼                 │───────────────┐
┌──────────────┐          │  Event processor         │               │
│ Event Sources│──submit──▶  + Scheduler             │               ▼
│              │          │  + WatchCoordinator      │        ┌──────────────┐
│ Filesystem   │          │        │                 │        │ Event Sinks  │
│ OneShot      │          │        ▼                 │──emit──▶ (supervised) │
│ Agent RPC    │          │    WorkerPool            │        └──────────────┘
└──────────────┘          └──────────────────────────┘
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
    WAITING_ON_LOCK = 4  # Worker waiting for artifact locks
    RUNNING = 5      # Stage function executing
    COMPLETED = 6    # Terminal (ran/cached/blocked/cancelled/failed)
```

The IntEnum ordering enables comparisons like `state >= PREPARING` for output filtering.

### Event Sources

Sources push input events via memory channels (`MemoryObjectSendStream[InputEvent]`):

| Source | Events | Use Case |
|--------|--------|----------|
| `FilesystemSource` | `DataArtifactChanged`, `CodeOrConfigChanged` | Watch mode |
| `OneShotSource` | `RunRequested` | Batch mode |
| `AgentRpcSource` | `RunRequested`, `CancelRequested` | Agent RPC control |

For RPC control (agent integration), use `AgentRpcSource` which converts JSON-RPC commands into input events and delegates query methods to an `AgentRpcHandler`.

### Event Sinks

Sinks consume output events via `sink.handle()`:

| Sink | Output | Use Case |
|------|--------|----------|
| `StaticConsoleSink` | Rich terminal (buffered) | Pipe/CI — buffers completions, prints sorted report on close |
| `LiveConsoleSink` | Rich terminal (live) | TTY — live progress bar with running/completed counts |
| `ResultCollectorSink` | Dict collection | Programmatic result access |
| `JsonlSink` | JSONL records | `--jsonl` output for machine consumption |
| `BroadcastEventSink` | Pub-sub | Agent RPC subscribers |
| `EventBuffer` | Ring buffer | Agent RPC polling (`events_since`) |

The CLI chooses between `StaticConsoleSink` and `LiveConsoleSink` automatically based on whether stdout is a TTY. The TUI is **not** a sink — it is a separate package (`pivot-tui`) that connects as a JSON-RPC client. See [TUI Architecture](tui.md).

Sinks are supervised by the Engine: each sink receives events via its own bounded
queue. Slow or failing sinks are temporarily disabled and re-enabled after
exponential backoff, emitting `SinkStateChanged` events.

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
import rich.console

async with Engine(pipeline=pipeline) as engine:
    collector = ResultCollectorSink()
    engine.add_sink(collector)

    console = rich.console.Console()
    engine.add_sink(LiveConsoleSink(console=console))

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
async with Engine(pipeline=pipeline) as engine:
    engine.add_source(FilesystemSource(watch_paths=paths))

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
| `SinkStateChanged` | Sink disabled/enabled | Backoff state |
| `EngineDiagnostic` | Non-fatal anomaly | Message, detail |

## Async Safety

The Engine uses structured concurrency with anyio:

- All state access occurs within the event loop task in `run()`
- Sources run in separate tasks but only send events to channels—they don't access engine state
- Memory channels provide implicit serialization, so no explicit locks are needed
- Cancellation uses `anyio.Event`, not `threading.Event`

## Agent RPC Integration

Agent RPC control uses event sources and handlers, not direct Engine methods:

```python
from pivot.engine.agent_rpc import AgentRpcSource, AgentRpcHandler, EventBuffer, BroadcastEventSink

# Create handler for status/stages/metadata queries
handler = AgentRpcHandler(engine=engine, event_buffer=event_buffer)

# Add RPC source to Engine (converts JSON-RPC to events; queries handled by handler)
engine.add_source(AgentRpcSource(socket_path=socket_path, handler=handler))
# Add sinks for polling and pub-sub
engine.add_sink(event_buffer)
engine.add_sink(BroadcastEventSink())
```

## Serve Mode

For headless daemon operation (`pivot repro --watch --serve`), the Engine supports RPC sources:

```python
async with Engine(pipeline=pipeline) as engine:
    event_buffer = EventBuffer()
    handler = AgentRpcHandler(engine=engine, event_buffer=event_buffer)

    engine.add_source(FilesystemSource(watch_paths=paths))
    engine.add_source(AgentRpcSource(socket_path=socket_path, handler=handler))
    engine.add_sink(event_buffer)
    engine.add_sink(BroadcastEventSink())

    await engine.run(exit_on_completion=False)
```

### Serve Mode Components

| Component | Purpose |
|-----------|---------|
| `AgentRpcSource` | JSON-RPC 2.0 over Unix socket |
| `BroadcastEventSink` | Broadcast events to subscribed clients |
| `EventBuffer` | Ring buffer for event polling |

### Agent RPC Protocol

The `AgentRpcSource` implements JSON-RPC 2.0 over Unix socket:

**Commands** (become input events):
- `run` - Start a run with optional stages/force
- `cancel` - Request cancellation
- `set_on_error` - Update error mode (`fail`/`keep_going`) for future runs

**Queries** (handled by `AgentRpcHandler`):
- `status` - Get engine state (idle/active)
- `stages` - List registered stages
- `stage_info` - Get deps/outs for a stage
- `explain` - Compute a `StageExplanation`
- `events_since` - Poll buffered output events (requires `EventBuffer` sink)
- `commit` - Persist current workspace state (`pivot commit`)
- `diff_output` - Diff cached outputs for TUI panels

```json
{"jsonrpc": "2.0", "method": "run", "params": {"stages": ["train"]}, "id": 1}
{"jsonrpc": "2.0", "result": "accepted", "id": 1}
{"jsonrpc": "2.0", "method": "events_since", "params": {"version": 0}, "id": 2}
```

### Event Broadcasting

`BroadcastEventSink` provides pub-sub event delivery to connected agents:

```python
# Subscribe a client
recv = await event_sink.subscribe("client_id")

# Receive events
async for event in recv:
    process(event)

# Unsubscribe when done
await event_sink.unsubscribe("client_id")
```

**Backpressure Handling:** If a client's buffer is full, events are dropped silently with a debug log. Clients should process events quickly or increase buffer size.

Alternatively, `EventBuffer` provides polling-based access via `events_since(version)`:

```python
# Poll for new events
result = event_buffer.events_since(last_version)
for versioned_event in result["events"]:
    process(versioned_event["event"])
last_version = result["version"]
```

## TUI Integration

The TUI (`pivot-tui` package) is **not** an engine sink. It runs in a separate process thread and communicates exclusively via JSON-RPC over the same Unix socket used by `AgentRpcSource`. The CLI's `run_tui_with_engine()` helper coordinates a three-thread model:

| Thread | Role |
|--------|------|
| Main | Textual TUI — signal handlers require main thread |
| Engine | `anyio.run()` with Engine + RPC socket server |
| Poller | Polls `events_since()`, posts `TuiUpdate` messages to app |

The poller thread converts engine output events into typed TUI messages (`TuiStatusMessage`, `TuiLogMessage`, etc.) and feeds them to the Textual app via `post_message()`. UI commands (`run`, `cancel`, `commit`) go from the TUI's own RPC client directly to the engine.

For details, see [TUI Architecture](tui.md).

## Code Locations

| Component | File |
|-----------|------|
| Engine class | `packages/pivot/src/pivot/engine/engine.py` |
| Scheduler | `packages/pivot/src/pivot/engine/scheduler.py` |
| Watch coordinator | `packages/pivot/src/pivot/engine/watch.py` |
| Worker pool | `packages/pivot/src/pivot/engine/worker_pool.py` |
| Bipartite graph | `packages/pivot/src/pivot/engine/graph.py` |
| Event types | `packages/pivot/src/pivot/engine/types.py` |
| Event sources | `packages/pivot/src/pivot/engine/sources.py` |
| Event sinks | `packages/pivot/src/pivot/engine/sinks.py` |
| Agent RPC | `packages/pivot/src/pivot/engine/agent_rpc.py` |
| TUI launch coordinator | `packages/pivot/src/pivot/cli/_run_common.py` |
| TUI app | `packages/pivot-tui/src/pivot_tui/run.py` |

## See Also

- [Execution Model](execution.md) - Parallel execution details
- [Watch Mode](watch.md) - Watch mode specifics
- [Code Tour](code-tour.md) - Code navigation guide
