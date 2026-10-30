# TUI Architecture

The Terminal User Interface provides a real-time view of pipeline execution.

## Overview

Pivot's TUI is built with [Textual](https://textual.textualize.io/), an async Python TUI framework. It displays stage status, logs, and input/output changes during execution.

The TUI supports two modes:

- **Run mode:** Executes pipeline once with progress display
- **Watch mode:** Continuously monitors files and re-runs affected stages, with execution history tracking

## Communication Architecture

The TUI is a **pure RPC client** — it communicates with the engine exclusively via JSON-RPC 2.0 over a Unix socket. It does not import runtime modules (`engine`, `storage`, `executor`, `config`, etc.); only `pivot.types` is shared for message schemas.

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

1. **Main thread**: Runs the Textual TUI (`app.run()`). Required for signal handlers. Owns its own `RpcPivotClient` connected in `on_mount()` for UI commands (`run`, `cancel`, `commit`, `set_on_error`, `diff_output`). Before it starts the engine thread it enters `defer_console_logging()`, which attaches a filter to each root log handler that writes to stdout or stderr and wraps each handler's whole filter-and-emit cycle under one lock. Until `app.run()` starts, records print as usual and a warning is remembered only after its terminal write completes. Starting Textual takes those same locks before redirecting stderr. While the TUI runs, the handler prints nothing: the filter holds each distinct warning or error logged on any thread (same logger, level, message and exception), skipping those already printed, and prints them once `app.run()` returns. The engine's tracked-file and `Skipping:` warnings are among them. `Pipeline invalid` records carry the error they report; `pivot repro --watch --tui` supplies `Engine.pipeline_error` at release, which prints only a record the engine still reports. Records from `pivot_tui` describe the TUI itself and are dropped.

2. **Engine thread**: Runs `anyio.run(engine_fn)` with the pipeline engine and RPC socket server. Sets `socket_ready` after the Unix socket is listening.

3. **Poller thread**: Runs `anyio.run(poller_main)` with its own `RpcPivotClient` (separate connection). Polls `events_since()` and converts engine output events to TUI messages via `app.post_message()`.

Each thread has its own anyio event loop — socket connections are never shared across threads.

### Message Flow

1. **Engine emits events** (`stage_started`, `stage_completed`, `log_line`, `engine_state_changed`, `pipeline_reloaded`, `stages_expanded`) into an `EventBuffer` sink. The buffer is exposed via the RPC server's `events_since` method.

2. **EventPoller** (in its own thread) polls `events_since(version)` every 100ms, converts raw events to typed TUI messages (`TuiStatusMessage`, `TuiLogMessage`, `TuiWatchMessage`, `TuiReloadMessage`, `TuiExpandedMessage`), and posts them to the TUI via `app.post_message()`.

3. **TUI commands** (`run`, `cancel`, `commit`, `set_on_error`, `diff_output`) go directly from the TUI's main thread to the engine via its own RPC client connection.

### Shutdown

- `EventPoller.stop()` uses a `threading.Event` + task group cancellation to interrupt blocked RPC calls
- TUI quit handlers call `poller.stop()` then `client.disconnect()`
- `run_tui_with_engine()` owns a second `threading.Event`, `poller_stop`. The engine thread sets it in its `finally` when the engine finishes, and the main thread sets it once `app.run()` returns. The poller's connect-retry loop waits on it between attempts and gives up as soon as it is set, so a one-shot engine that finishes and removes its socket before the poller connects does not leave the poller retrying. A connected poller is stopped by the same event.
- Because `poller_stop` is always set by the time `app.run()` returns, `poller_thread.join()` waits with no timeout.
- When the engine thread ends it posts `TuiShutdown(error=...)`. Run mode exits on it. Watch mode ignores a normal shutdown and keeps running until the user quits, but on an engine error it exits with return code 1, and `run_tui_with_engine()` re-raises the engine's exception so the command fails.
- Both engine and poller threads are daemon threads — killed on process exit as fallback

## Message Types

`TuiStatusMessage`, `TuiLogMessage`, `TuiWatchMessage` and `TuiReloadMessage` are defined in `src/pivot/types.py`; `TuiExpandedMessage` and the `TuiMessage` union of all five are in `pivot_tui/types.py`:

| Message | Source | Purpose |
|---------|--------|---------|
| `TuiStatusMessage` | Coordinator | Stage lifecycle (started, completed, failed, skipped) with timing |
| `TuiLogMessage` | Worker | stdout/stderr lines from stage execution |
| `TuiWatchMessage` | Watch engine | Watch status (waiting, detecting, restarting, error) |
| `TuiReloadMessage` | Watch engine | Stage list changed after code reload, or why the edited pipeline is invalid |
| `TuiExpandedMessage` | Coordinator | A mapped stage's template (`pipeline.map`) added its instances to the run |

Key fields:

- **TuiStatusMessage**: `type`, `stage`, `index`, `total`, `status`, `reason`, `elapsed` (seconds, or `None` if still running), `run_id`
- **TuiLogMessage**: `type`, `stage`, `line`, `is_stderr`, `timestamp`
- **TuiWatchMessage**: `type`, `status` (WatchStatus enum), `message`
- **TuiReloadMessage**: `type`, `stages` (current stage names), `stages_added`, `stages_removed`, `stages_modified`, `error`. When `error` is set (a syntax error, a missing `pipeline` variable, a stage that fails to fingerprint, a `params.yaml` that fails its checks, or a run that found a stage over its resource capacity or an input missing), the app keeps its stage list, titles itself `Pipeline invalid - waiting for fix`, and shows the error on a line under the header until a message with `error` unset clears it: a good reload, or the data change that finds the missing input.
- **TuiExpandedMessage**: `type`, `template`, `stages` (the instance names `<template>@<key>`, in key order). The poller converts the engine's `stages_expanded` event into it.

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

### Mapped Stages

A mapped stage (`pipeline.map`) is a template until a run expands it into one instance per file of the directory it is mapped over. The template's row shows where it is, in place of a run time:

```
▼ demo/cut (3)  ●3
    demo/cut              expanded (3) ●
    @a                    0:01 ●
    @b                    0:01 ●
    @c                    0:02 ●
```

- **`not yet expanded`**: the template has not expanded in this session, or the latest run blocked it (its producer failed): that run added no instance, so the rows of the last expansion go. The app learns which stages are templates from the `stages` RPC answer's `templates` (see [Agent Server](agent-server.md)), asked once its client connects and again after each reload.
- **`expanded (N)`**: on a `TuiExpandedMessage` the app marks the template completed, puts the instance rows right after it, numbers every row by its place in the list, and rebuilds the list. The instance names are `<template>@<key>`, so they group under the template as variants do; the group header counts the instances, not the template. A later expansion (the next watch run) replaces the last one: an instance it keeps keeps its state and history, an instance it lacks loses its row. An answer to the `stages` query that shows a template expanded before its event arrives adds the instance rows too.
- **`expansion failed`**: a refused expansion (an instance name that cannot name a stage, an output path another stage holds) reaches the app as a failed `TuiStatusMessage` for the template. The row shows the failed symbol, the rows of the last expansion go, and the reason goes to the template's Logs tab as a stderr line, since a template runs nothing of its own.

A reload's stage list holds no template, so the app keeps each template row, before the first of its instances the list holds, and the next `stages` answer sets how far the new pipeline expanded it, adds a row for a template the edit added and drops the row of one it removed.

Whenever a status arrives for a stage the list does not hold (in run mode as in watch mode: a new stage, an instance whose expansion was missed, a stage only its log lines announced), the app rebuilds the rows.

### Status Symbols

| Symbol | Meaning |
|--------|---------|
| `○` | Pending |
| `▶` | Running |
| `●` | Success (completed) |
| `↺` | Cached |
| `◇` | Blocked |
| `!` | Cancelled |
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
