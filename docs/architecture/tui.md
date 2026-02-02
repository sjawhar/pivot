# TUI Architecture

The Terminal User Interface provides a real-time view of pipeline execution.

## Overview

Pivot's TUI is built with [Textual](https://textual.textualize.io/), an async Python TUI framework. It displays stage status, logs, and input/output changes during execution.

The TUI supports two modes:

- **Run mode:** Executes pipeline once with progress display
- **Watch mode:** Continuously monitors files and re-runs affected stages, with execution history tracking

## Communication Architecture

```
┌─────────────────┐    mp.Queue     ┌─────────────────┐   queue.Queue   ┌──────────────────┐
│  Worker Process │ ─────────────▶  │   Coordinator   │ ──────────────▶ │  TUI Event Loop  │
│  (stdout/stderr)│  (cross-process)│   (main thread) │   (TuiQueue)    │   (Textual)      │
└─────────────────┘                 └─────────────────┘                 └──────────────────┘
```

### Message Flow

Two communication channels exist:

1. **Output Queue** (`multiprocessing.Queue`): Stage stdout/stderr lines flow from worker processes to the coordinator. Workers capture print statements and write them to this queue for real-time display.

2. **TuiQueue** (`queue.Queue`): Status updates and watch events flow from the coordinator to the TUI within the same process. This is a stdlib thread queue (not multiprocessing) since both ends are in the main process.

The coordinator acts as a bridge: it receives output lines from workers via `multiprocessing.Queue`, then forwards them to the TUI via the thread-safe `TuiQueue`.

Textual's `@work` decorator and `post_message()` handle thread-to-event-loop safety.

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
