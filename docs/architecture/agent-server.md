# Agent Server

The Agent Server provides a JSON-RPC interface for external tools to interact with Pivot's watch mode.

## Overview

When running in watch mode with the TUI, Pivot starts an Agent Server that exposes pipeline operations over a Unix socket. This enables IDE integrations, external automation tools, and custom interfaces to control pipeline execution.

## Protocol

**Transport:** Unix domain socket. `pivot repro --watch --serve`, `pivot repro --watch --tui` and the one-shot `pivot repro --tui` and `pivot run --tui` all listen on `agent.sock` in the state directory (`core.state_dir`, `.pivot` by default). A socket address holds at most 108 bytes on Linux and 104 on macOS; Pivot serves a project at any depth, and a client of a deeper project connects by the state directory's path relative to the project root, `.pivot/agent.sock` in the default layout, as in the example below. The `pivot` command's serve mode stops on SIGTERM as it does on Ctrl-C, including a stop sent more than once or to its whole process group (GNU `timeout`, `systemctl stop`): it removes the socket and its worker pool's temporary directory and exits 0. A SIGKILL leaves the socket, which the next start removes. Serve takes a stop signal only if it is at its default when `pivot` starts. A signal that whatever launched `pivot` ignored (`env --ignore-signal`, `trap ''`) or gave a handler (a C program that embeds Python, an `LD_PRELOAD` library, Python startup code) belongs to that launcher, and serve leaves it alone. Serve run in-process (`pivot.cli.main()` or the Click group) changes no signal state: the caller's own handlers decide what SIGTERM does, and the caller owns shutdown.

**Protocol:** JSON-RPC 2.0

**Security:** Socket permissions `0o600` (owner-only read/write)

Example request:

```json
{
  "jsonrpc": "2.0",
  "method": "run",
  "params": {"stages": ["train"], "force": false},
  "id": 1
}
```

Example response:

```json
{
  "jsonrpc": "2.0",
  "result": {"run_id": "abc123def456", "status": "started", "stages_queued": ["train"]},
  "id": 1
}
```

**Artifact identities:** every artifact identity in a response is sent as an object `{"producer": "<stage>", "key": "<output key or null>"}`. That covers `stage_info`, `explain`, and the events returned by `events_since`, including each `stage_completed.output_summary[].identity`.

## Available Methods

| Method | Type | Params | Description |
|--------|------|--------|-------------|
| `run` | Command | `stages?`, `force?` | Queue execution request |
| `cancel` | Command | - | Request cancellation of current execution |
| `set_on_error` | Command | `mode` | Set error handling (`"fail"` or `"keep_going"`) |
| `status` | Query | - | Get current execution state |
| `stages` | Query | - | List the registered stages and the mapped stages' templates |
| `stage_info` | Query | `stage` | Get detailed info for a specific stage |
| `explain` | Query | `stage` | Get explanation of why a stage will run |
| `events_since` | Query | `version` | Poll for events since a version cursor |
| `commit` | Query | - | Commit pending output changes |
| `diff_output` | Query | `path`, `old_hash?`, `new_hash?`, `max_rows?` | Get diff of an output file |

### run

Queue stages for execution.

**Parameters:**

- `stages` (optional): List of stage names to run. If omitted, runs all stages.
- `force` (optional): If true, ignore cache and force re-execution.

**Returns:**

- `run_id`: Unique identifier for this execution (12-character UUID prefix)
- `status`: Always `"started"` on success
- `stages_queued`: List of stage names that will execute

**Example:**

```json
// Request
{"jsonrpc": "2.0", "method": "run", "params": {"stages": ["preprocess", "train"]}, "id": 1}

// Response
{"jsonrpc": "2.0", "result": {"run_id": "abc123def456", "status": "started", "stages_queued": ["preprocess", "train"]}, "id": 1}
```

### status

Get the current state of the pipeline or a specific run.

**Parameters:**

- `run_id` (optional): Specific run to query. If omitted, returns current state.

**Returns:**

- `state`: Current state (`idle`, `watching`, `running`, `completed`, `failed`)
- `run_id`: Current or specified run ID (when applicable)
- `stages_completed`: List of completed stage names
- `stages_pending`: List of pending stage names
- `ran`, `skipped`, `failed`: Counts of stage outcomes

**Example:**

```json
// Request
{"jsonrpc": "2.0", "method": "status", "id": 2}

// Response
{
  "jsonrpc": "2.0",
  "result": {
    "state": "running",
    "run_id": "abc123def456",
    "stages_completed": ["preprocess"],
    "stages_pending": ["train", "evaluate"],
    "ran": 1,
    "skipped": 0,
    "failed": 0
  },
  "id": 2
}
```

### stages

List the registered stages, and each template of a mapped stage (`pipeline.map`) with the instances it expanded into.

**Parameters:** None

**Returns:**

- `stages`: The stage names: every stage the pipeline registered, and each instance a run has expanded so far (`<template>@<key>`). Never a template.
- `templates`: Each template's name mapped to its instances in key order (`Pipeline.instances_of`). An empty list means the template has not expanded yet, or expanded over an empty directory; the `stages_expanded` event tells the two apart. A run expands a template once the stages upstream of it finish, and each run starts with every template unexpanded.

**Example:**

```json
// Request
{"jsonrpc": "2.0", "method": "stages", "id": 3}

// Response
{
  "jsonrpc": "2.0",
  "result": {
    "stages": ["demo/plan", "demo/cut@a", "demo/cut@b"],
    "templates": {"demo/cut": ["demo/cut@a", "demo/cut@b"], "demo/report": []}
  },
  "id": 3
}
```

The TUI asks once it connects, and again after each reload, so that a template's row reads "not yet expanded" before the `stages_expanded` event arrives.

### Mapped stage names in `run`, `stage_info` and `explain`

A template (`demo/cut`) and an instance (`demo/cut@a`) are names these methods take, as the CLI does:

- `run` takes both: a run reaches an instance through its template and adds that instance alone, with its chain.
- `stage_info` of a template gives the stage as declared, its mapped dependency reading the producer's whole directory. An instance the template has expanded into gives its own info; any other instance of it is refused: `'demo/cut@a' is not yet expanded: 'demo/cut' adds its instances when a run finds 'demo/plan' current`, or, when the run's targets named other instances of it, `'demo/cut@b' is not yet expanded: 'demo/cut' added only the instances its run's targets name`.
- `explain` of a template explains each instance it expanded into: `will_run` is true when one would run, and `reason` reads `Expanded into 3 stages; would run: demo/cut@b (Input dependencies changed)` or `Expanded into 3 stages; none would run`. A template that has not expanded, or an instance not expanded, gets `will_run: true` and a reason starting `Not yet expanded:`, as `pivot status` shows it.
- Once a run has read the producer's files to expand a template (`Engine.expansion_keys`: until the next run starts), an instance whose key no file has names no stage: `stage_info` and `explain` refuse it as `pivot status` refuses its name, `Unknown stage(s): demo/cut@zzz`.

### cancel

Request cancellation of current execution.

**Parameters:** None

**Returns:**

- `cancelled`: `true` if a running execution was cancelled, `false` if nothing was running

When called during an active execution, sets a `threading.Event` that the executor checks between stages. Cancellation is **stage-level granularity**: the currently running stage completes normally (ensuring outputs are in a consistent state), but no new stages are started. Pending stages are marked as skipped with reason "cancelled".

## State Machine

The Agent Server tracks pipeline state through these transitions:

```
IDLE → WATCHING → RUNNING → COMPLETED/FAILED → WATCHING
                     ↑                              │
                     └──────────────────────────────┘
```

| State | Description |
|-------|-------------|
| `IDLE` | Server started, not yet watching |
| `WATCHING` | Monitoring for file changes |
| `RUNNING` | Executing stages |
| `COMPLETED` | Last run succeeded |
| `FAILED` | Last run had failures |

After `COMPLETED` or `FAILED`, the state returns to `WATCHING` to await further changes or explicit run requests.

## Thread Safety

The Engine uses locks to protect atomic state transitions. When a `run` request arrives during execution, it returns an error (`-32001 EXECUTION_IN_PROGRESS`) rather than queuing.

## Error Codes

| Code | Name | Description |
|------|------|-------------|
| `-32700` | Parse error | Invalid JSON |
| `-32600` | Invalid request | Request structure invalid |
| `-32601` | Method not found | Unknown RPC method |
| `-32602` | Invalid params | Method parameters invalid |
| `-32603` | Internal error | Server-side exception; a query that raises a Pivot error (for example a stage that fails to fingerprint) gets this code with the error's message |
| `-32001` | Execution in progress | A run is already executing |
| `-32002` | Stage not found | Requested stage doesn't exist (includes suggestions for typos) |
| `-32003` | Pipeline invalid | `run` or `commit` refused because the watched pipeline is invalid (it fails to load, a stage fails to fingerprint, or `params.yaml` fails its checks, at startup or after an edit; or a run found a stage over its resource capacity, or an input missing that left it no stage to run); the message carries the error. Refused until an edit reloads a good pipeline, or a data change finds the missing input, so a commit never records outputs the current code did not produce |
| `-32004` | Engine starting | `run` or `commit` refused because a watched engine is fingerprinting stages: its startup snapshot ("Engine is starting") or a code reload ("Engine is reloading"). The reply is immediate; retry once the engine is ready |

## Client Example

The socket is `<state dir>/agent.sock`, where `core.state_dir` defaults to `.pivot`. Using Python
with raw sockets:

```python
import socket
import json

def call_agent(method: str, params: dict | None = None) -> dict:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect("<state dir>/agent.sock")

    request = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
        "id": 1
    }

    sock.sendall(json.dumps(request).encode() + b"\n")
    response = sock.recv(4096).decode()
    sock.close()

    return json.loads(response)

# Trigger a run
result = call_agent("run", {"stages": ["train"], "force": True})
print(f"Run started: {result['result']['stages_queued']}")

# Check status
status = call_agent("status")
print(f"State: {status['result']['state']}")
```

## Limitations

- **Cancellation granularity:** Cancellation takes effect between stages, not mid-stage. The currently running stage will complete before cancellation is observed.

- **Single client:** The server handles one request at a time. Concurrent requests are serialized.

- **Local only:** Unix sockets are local-machine only. For remote access, use SSH tunneling or a proxy.

## See Also

- [Engine Architecture](engine.md) - Execution engine architecture
- [Watch Mode](watch.md) - Watch mode and file monitoring
- [TUI Architecture](tui.md) - Terminal UI communication
