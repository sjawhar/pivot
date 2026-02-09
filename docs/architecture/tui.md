# TUI Architecture

The Terminal User Interface provides a real-time view of pipeline execution.

## Overview

Pivot's TUI is built with [Textual](https://textual.textualize.io/), an async Python TUI framework. It displays stage status, logs, and input/output changes during execution.

The TUI supports two modes:

- **Run mode:** Executes pipeline once with progress display
- **Watch mode:** Continuously monitors files and re-runs affected stages, with execution history tracking

## Communication Architecture

The TUI is a **pure RPC client** — it communicates with the engine exclusively via JSON-RPC 2.0 over a Unix socket. Zero imports of pivot runtime modules (`engine`, `storage`, `executor`, `config`, etc.) — only `pivot.types` is allowed.

```
┌─────────────┐                  ┌──────────────────┐                  ┌──────────────────┐
│   Engine    │  JSON-RPC/Unix   │  Event Poller    │   post_message   │  TUI (Textual)   │
│   Thread    │ ◀──────────────▶ │  Thread          │ ────────────────▶│  Main Thread     │
│  (anyio)    │   socket         │  (anyio)         │                  │                  │
│             │                  │  polls events    │                  │  UI commands     │
│             │  JSON-RPC/Unix   │                  │                  │  ──────────────▶ │
│             │ ◀────────────────┼──────────────────┼──────────────────│  (own client)    │
└─────────────┘   socket         └──────────────────┘                  └──────────────────┘
```

### Three-Thread Model

The `run_tui_with_engine()` helper in `_run_common.py` coordinates three threads:

1. **Main thread**: Runs the Textual TUI (`app.run()`). Required for signal handlers. Owns its own `RpcPivotClient` connected in `on_mount()` for UI commands (commit, run, cancel).

2. **Engine thread**: Runs `anyio.run(engine_fn)` with the pipeline engine and RPC socket server. Sets `socket_ready` event when the socket is listening.

3. **Poller thread**: Runs `anyio.run(poller_main)` with its own `RpcPivotClient` (separate connection). Polls `events_since()` and converts engine events to TUI messages via `app.post_message()`.

Each thread has its own anyio event loop — socket connections are never shared across threads.

### Message Flow

1. **Engine emits events** (`StageStarted`, `StageCompleted`, `LogLine`, etc.) into an `EventBuffer` exposed via the RPC server's `events_since` method.

2. **EventPoller** (in its own thread) polls `events_since(version)` periodically (100ms), converts raw events to typed TUI messages (`TuiStatusMessage`, `TuiLogMessage`, etc.), and posts them to the TUI via `app.post_message()`.

3. **TUI commands** (commit, run, cancel, set_on_error) go directly from the TUI's main thread to the engine via its own RPC client connection.

### Shutdown

- `EventPoller.stop()` uses a `threading.Event` + task group cancellation to immediately interrupt blocked RPC calls
- TUI quit handlers call `poller.stop()` then `client.disconnect()`
- `poller_thread.join(timeout=2.0)` ensures clean shutdown before process exit
- Both engine and poller threads are daemon threads — killed on process exit as fallback

## Message Types

All message types are defined in `src/pivot/types.py`:

| Message | Source | Purpose |
|---------|--------|---------|
| `TuiStatusMessage` | Coordinator | Stage lifecycle (started, completed, failed, skipped) with timing |
| `TuiLogMessage` | Worker | stdout/stderr lines from stage execution |
| `TuiWatchMessage` | Watch engine | Watch status (waiting, detecting, restarting, error) |
| `TuiReloadMessage` | Watch engine | Stage list changed after code reload (add/remove stages) |

Key fields:

- **TuiStatusMessage**: `type`, `stage`, `index`, `total`, `status`, `reason`, `elapsed` (seconds, or `None` if still running), `run_id`
- **TuiLogMessage**: `type`, `stage`, `line`, `is_stderr`, `timestamp`
- **TuiWatchMessage**: `type`, `status` (WatchStatus enum), `message`
- **TuiReloadMessage**: `type`, `stages` (list of current stage names after reload)

## Execution History

The TUI maintains a bounded history (50 entries per stage) of past executions. Each `ExecutionHistoryEntry` captures:

- **Timestamp:** When execution started
- **Duration:** How long it took (or None if still running)
- **Status:** Completed, failed, or skipped
- **Logs:** stdout/stderr output
- **Inputs:** Stage explanation (code/params/dependency changes)
- **Outputs:** Output file changes

This enables "time-travel" viewing of past executions. Select a stage and scroll through its history to see logs and inputs/outputs from previous runs.

## UI Components

```
┌─────────────────────────────────────────────────────────────────────┐
│  pivot repro --watch                                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Stages (3) ●1 ✓2                     │  train ● LIVE               │
│  ─────────────────────────────────────┼──────────────────────────────│
│  → ● train              0.5s          │  ┌─────┬───────┬────────┐   │
│    ✓ preprocess         0.2s          │  │ Logs│ Input │ Output │   │
│    ○ evaluate                         │  ├─────┴───────┴────────┘   │
│                                       │  │ [12:34:56] Epoch 1/10    │
│                                       │  │ [12:34:57] loss=0.523    │
│                                       │  │ [12:34:58] Epoch 2/10    │
│                                       │  │ [12:34:59] loss=0.412    │
│                                       │                              │
│  Watching for changes...              │                              │
└─────────────────────────────────────────────────────────────────────┘
```

| Component | Description |
|-----------|-------------|
| **Stage List** | Scrollable list with status indicators, selection (→), grouping for variants |
| **Tabbed Detail Panel** | Three tabs: Logs, Input (code/dep/param changes), Output (file changes) |
| **Status Header** | Stage counts by status (running/completed/failed) |
| **History Indicator** | Shows "● LIVE" or "Run X of Y" when viewing history |
| **Debug Panel** | Toggleable stats panel (queue throughput, memory, workers) |

### Stage Grouping

Stages with variants (e.g., `train@small`, `train@large`) are grouped under a collapsible header:

```
▼ train (2)  ●1 ✓1
  → ● train@small         0.5s
    ✓ train@large         1.2s
```

### Status Symbols

| Symbol | Meaning |
|--------|---------|
| `○` | Pending |
| `▶` | Running |
| `●` | Success (completed) |
| `$` | Cached |
| `⊘` | Blocked |
| `!` | Skipped |
| `✗` | Failed |

### Input/Output Diff Panels

The Input and Output tabs show changes with a split-view layout:

```
┌────────────────────────┬────────────────────────┐
│ [~] func:train         │ Hash: a1b2c3 → d4e5f6  │
│ [ ] func:preprocess    │                        │
│ [+] param:batch_size   │                        │
└────────────────────────┴────────────────────────┘
```

Change indicators: `[~]` modified, `[+]` added, `[-]` removed, `[ ]` unchanged

## Keyboard Shortcuts

### Stage Navigation
| Key | Action |
|-----|--------|
| `j`/`k` or ↑/↓ | Navigate stage list (skips collapsed/filtered) |
| `/` | Filter stages by name |
| `Enter` | Toggle collapse/expand for stage group |
| `-` | Collapse all groups |
| `=` | Expand all groups |

### Tab Navigation
| Key | Action |
|-----|--------|
| `Tab`, `h`/`l`, ←/→ | Cycle through tabs (Logs → Input → Output) |
| `L` | Jump to Logs tab |
| `I` | Jump to Input tab |
| `O` | Jump to Output tab |

### Detail Panel
| Key | Action |
|-----|--------|
| `Ctrl+J`/`Ctrl+K` | Scroll detail content |
| `n`/`N` | Jump to next/previous changed item |
| `Enter` | Expand item details to full width |
| `Escape` | Collapse expanded details |

### History (Watch Mode)
| Key | Action |
|-----|--------|
| `[`/`]` | Navigate to older/newer execution |
| `H` | Open history list modal |
| `G` | Jump to live view |

### Actions
| Key | Action |
|-----|--------|
| `c` | Commit pending changes (watch mode) |
| `g` | Toggle keep-going mode (watch mode) |
| `~` | Toggle debug panel |
| `?` | Show help screen |
| `Escape` | Clear filter, collapse details, or cancel |
| `q` | Quit (with confirmation if stages running or uncommitted changes) |

## Error Display

When a stage fails, the TUI shows:

1. **Stage status:** Red `✗` indicator
2. **Error in logs:** Full traceback and error message (stderr in red)
3. **Downstream impact:** Blocked stages show `⊘` indicator

```
│  ● preprocess         0.5s          │  [12:34:56] Traceback:        │
│  ✗ train              1.2s          │  [12:34:57]   File "model.py" │
│  ⊘ evaluate                         │  [12:34:58]     raise ValueError │
```

The watch engine continues monitoring. Fix the error, save the file, and the pipeline automatically re-runs.

## Performance Considerations

- **History limit:** 50 entries per stage prevents memory growth
- **Log buffering:** Large outputs are buffered and truncated in the UI
- **Reactive updates:** Textual only re-renders changed components

## See Also

- [Watch Execution Engine](watch.md) - Watch mode architecture
- [Agent Server](agent-server.md) - JSON-RPC interface
- [Execution Model](execution.md) - Stage execution details
