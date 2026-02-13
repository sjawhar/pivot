# Terminal UI (TUI)

Pivot includes an interactive terminal interface built with [Textual](https://textual.textualize.io/) for monitoring pipeline execution. The TUI shows stage progress, logs, and input/output diffs in a split-panel layout.

## Launching

Add `--tui` to any `pivot repro` (full pipeline with dependencies) or `pivot run` (individual stages, no dependency resolution) command:

```bash
# One-shot execution with TUI
pivot repro --tui

# Watch mode with TUI
pivot repro --watch --tui

# Run specific stages
pivot repro train evaluate --tui
```

The TUI requires the `pivot-tui` package. Install it with:

```bash
uv add "pivot[tui]"
```

### TUI Log File

For debugging, write all TUI messages to a JSONL file:

```bash
pivot repro --tui --tui-log messages.jsonl
```

### Minimum Terminal Size

The TUI requires at least **80 columns by 24 rows**. If your terminal is smaller, a warning notification appears.

## Layout

The TUI uses a two-panel layout:

```text
┌─────────────────────────────────────────────────────────┐
│ Header (mode + progress)                                │
├──────────────┬──────────────────────────────────────────┤
│              │  Logs │ Input │ Output                   │
│  Stage List  │──────────────────────────────────────────│
│              │                                          │
│  ○ preprocess│  Stage output, input diffs, or logs      │
│  ▶ train     │  displayed here depending on active tab  │
│  ○ evaluate  │                                          │
│              │                                          │
├──────────────┴──────────────────────────────────────────┤
│ Footer (context-sensitive keybinding hints)             │
└─────────────────────────────────────────────────────────┘
```

**Stage List** (left) — shows all stages with status symbols. Stages are grouped by matrix variants and sorted by DAG level. Use `j`/`k` to navigate, `/` to filter by name, `Enter` to collapse groups.

**Detail Panel** (right) — three tabs:

- **Logs** — stdout/stderr captured from the selected stage. Supports search with `Ctrl+f`.
- **Input** — diff view showing what changed in the stage's dependencies (code, params, dep files).
- **Output** — diff view showing what changed in the stage's output files.

**Footer** — displays context-sensitive keybinding hints that change based on the active panel and tab.

## Status Symbols

The stage list uses compact symbols to show each stage's current state:

| Symbol | Style | Meaning |
|--------|-------|---------|
| `○` | dim | **Pending** — not yet started |
| `⏳` | yellow | **Waiting on lock** — waiting to acquire execution lock |
| `▶` | blue bold | **Running** — currently executing |
| `●` | green bold | **Success** — ran and completed |
| `↺` | yellow | **Cached** — skipped, outputs already up to date |
| `◇` | red | **Blocked** — skipped because an upstream stage failed |
| `!` | yellow dim | **Cancelled** — execution was cancelled |
| `✗` | red bold | **Failed** — execution error |
| `?` | dim | **Unknown** — status not yet determined |

In run mode, the header title updates with progress (e.g., `pivot run (3/5)`). In watch mode, the title reflects the current state:

| Title Prefix | Meaning |
|-------------|---------|
| `[●] Watching for changes...` | Idle, waiting for file changes |
| `[↻] Reloading code...` | Detected changes, reloading pipeline code |
| `[▶] ...` | Executing stages |
| `[!] ...` | An error occurred |

## Keyboard Shortcuts

### Stage Navigation

| Key | Action |
|-----|--------|
| `j` / `k` | Move down / up in stage list |
| `↑` / `↓` | Move down / up in stage list |
| `/` | Filter stages by name |
| `Enter` | Toggle collapse for current group |
| `-` | Collapse all groups |
| `=` | Expand all groups |

### Tab Navigation

| Key | Action |
|-----|--------|
| `Tab` | Cycle to next tab |
| `h` / `l` | Previous / next tab |
| `←` / `→` | Previous / next tab |
| `L` | Jump to Logs tab |
| `I` | Jump to Input tab |
| `O` | Jump to Output tab |

### Detail Content

| Key | Action |
|-----|--------|
| `Ctrl+j` / `Ctrl+k` | Scroll detail content down / up |
| `n` / `N` | Jump to next / previous changed item |
| `Ctrl+f` | Search within logs (Logs tab only) |
| `Escape` | Collapse expanded detail, clear filter, or cancel commit |

### History (Watch Mode)

| Key | Action |
|-----|--------|
| `[` / `]` | Navigate to older / newer execution |
| `G` | Return to live view |
| `H` | Show history list modal |

### Actions

| Key | Action |
|-----|--------|
| `c` | Commit changes (watch mode) |
| `r` | Force re-run selected stage (watch mode) |
| `R` | Force re-run all stages (watch mode) |
| `g` | Toggle keep-going mode (watch mode) |
| `~` | Toggle debug panel |
| `?` | Show help screen with all keybindings |
| `q` | Quit |

## Run Mode vs Watch Mode

**Run mode** (`pivot repro --tui`) executes the pipeline once. The TUI shows progress and exits automatically when all stages complete. If you press `q` while stages are running, a confirmation dialog asks whether to kill workers.

**Watch mode** (`pivot repro --watch --tui`) monitors the filesystem for changes and re-runs affected stages. The TUI stays open until you press `q`. Additional features in watch mode:

- **History** — browse previous executions with `[`/`]` or `H`
- **Commit** — press `c` to write lock files for completed stages (or use `--no-commit` to defer)
- **Force re-run** — press `r` to re-run the selected stage or `R` to re-run all stages
- **Live reload** — when pipeline code changes, stages are automatically re-discovered and the stage list updates

## Debug Panel

Press `~` to toggle the debug panel, which shows message throughput, active worker count, memory usage, and uptime. Useful for diagnosing performance issues or confirming engine responsiveness.

## Plain-Text Mode

If you prefer non-interactive output (or are running in a non-TTY environment), omit `--tui`:

```bash
# Plain console output
pivot repro

# Stream results as JSONL (for scripting)
pivot repro --jsonl

# Show stage stdout/stderr inline
pivot repro --show-output
```

The `--tui` and `--jsonl` flags are mutually exclusive.

## Related

- [Watch Mode](watch-mode.md) — continuous re-execution on file changes
- [CI Integration](ci-integration.md) — using `--jsonl` output in automated pipelines
- [Quick Start](../getting-started/quickstart.md) — running your first pipeline
