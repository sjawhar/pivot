# Agent Server

The Agent Server provides a JSON-RPC interface for external tools to interact with Pivot's watch mode.

## Overview

When running in watch mode with the TUI, Pivot starts an Agent Server that exposes pipeline operations over a Unix socket. This enables IDE integrations, external automation tools, and custom interfaces to control pipeline execution.

## Protocol

**Transport:** Unix domain socket (path provided by the TUI when starting the server)

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

## Available Methods

| Method | Params | Description |
|--------|--------|-------------|
| `run` | `stages?`, `force?` | Queue execution request |
| `status` | `run_id?` | Get current execution state |
| `stages` | - | List all registered stages |
| `cancel` | - | Request cancellation of current execution |

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

List all registered stages and their metadata.

**Parameters:** None

**Returns:**

- `stages`: List of stage objects with `name`, `deps`, and `outs`

**Example:**

```json
// Request
{"jsonrpc": "2.0", "method": "stages", "id": 3}

// Response
{
  "jsonrpc": "2.0",
  "result": {
    "stages": [
      {"name": "preprocess", "deps": ["data/raw.csv"], "outs": ["data/clean.csv"]},
      {"name": "train", "deps": ["data/clean.csv"], "outs": ["models/model.pkl"]}
    ]
  },
  "id": 3
}
```

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

The watch engine uses locks to protect atomic state transitions. When a `run` request arrives during execution, it returns an error (`-32001 EXECUTION_IN_PROGRESS`) rather than queuing.

## Error Codes

| Code | Name | Description |
|------|------|-------------|
| `-32700` | Parse error | Invalid JSON |
| `-32600` | Invalid request | Request structure invalid |
| `-32601` | Method not found | Unknown RPC method |
| `-32602` | Invalid params | Method parameters invalid |
| `-32603` | Internal error | Server-side exception |
| `-32001` | Execution in progress | A run is already executing |
| `-32002` | Stage not found | Requested stage doesn't exist (includes suggestions for typos) |

## Client Example

Using Python with raw sockets:

```python
import socket
import json

def call_agent(method: str, params: dict | None = None) -> dict:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(".pivot/agent.sock")

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

- [Watch Execution Engine](watch.md) - Watch mode architecture
- [TUI Architecture](tui.md) - Terminal UI communication
