# CLI Output Redesign

**Goal:** Replace the noisy, unformatted `ConsoleSink` with clean, aligned, colored output that auto-detects TTY vs pipe mode.

**Problem:** With large pipelines (175 stages), CLI output is a wall of unstyled text. Every stage emits multiple lines ("waiting for artifact lock", "skipped"), nothing aligns, there's no color, no progress indicator, and no way to tell what's running vs finished at a glance.

**Scope:** `ConsoleSink` replacement only. No changes to the event system, engine, TUI, or JSON output.

---

## Design

### Two sinks, one interface

The current `ConsoleSink` is replaced by two classes. The call site picks based on `console.is_terminal`:

- **`LiveConsoleSink`** — TTY: Rich `Live` pinned section + scrollback
- **`StaticConsoleSink`** — Pipes/CI: one line per stage completion

Both implement the existing `EventSink` protocol (`handle()` / `close()`). Shared formatting logic (alignment, colors, symbols) lives in helper functions within `sinks.py`.

### Event handling

| Event | Static sink | Live sink |
|-------|-------------|-----------|
| `stage_started` | ignore | add to running set, re-render |
| `stage_completed` | print completion line | move to completed, `console.print()` above pinned, re-render |
| `stage_state_changed` (`WAITING_ON_LOCK`) | **ignore** | **ignore** |
| `log_line` (if `--show-output`) | print dim line | `console.print()` dim line above pinned |
| `engine_diagnostic` | print warning | `console.print()` warning above pinned |

The "waiting for artifact lock" message is dropped by both sinks.

### Completion line format

```
  [ 3/175] ✓ base_download_data               done      3.4s
  [2–154/175] ○ 152 stages skipped
  [155/175] ✓ final_aggregate                  done      2.1s
  [156/175] ✗ export_results                   FAILED
             KeyError: 'score_normalized'...
```

**Columns:**
1. **Progress counter** `[N/TOTAL]` — right-aligned, fixed width
2. **Status symbol** — `✓` (green), `○` (dim), `✗` (bold red)
3. **Stage name** — left-padded to `min(max_name_seen, 50)`, truncated with `…` if longer
4. **Status word** — `done`, `skipped`, `FAILED`, colored to match
5. **Duration** — only for `done` stages

Error detail lines are indented below the failed stage in `dim red`.

### Skip collapsing

Skipped stages are buffered. The buffer flushes when a non-skip event arrives or the run ends:

- **≤20 buffered skips:** print each individually
- **>20 buffered skips:** collapse to `[N–M/TOTAL] ○ K stages skipped`

DAG execution naturally clusters skips (upstream skips → dependents skip), so collapsing works well in practice. Interleaved skip clusters are handled independently.

### Visual hierarchy (completion lines vs log lines)

When `--show-output` is on, completion lines and log lines are interleaved in scrollback. They're visually distinct:

- **Completion lines:** colored symbol, bold stage name, progress counter
- **Log lines:** `dim` text, `[stage_name]` prefix, no counter

### Live sink: pinned section

A Rich `Live` block pinned to the bottom of the terminal. Contains only:

```
  ▶ base_transform_data          running…  3.2s
  ▶ difficulty_compute_scores    running…  1.1s
  ━━━━━━━━━━━━━━━━━━━━━━━━  155/175 (88%)  3 ran · 152 skipped · 1 failed
```

- **Running stages** with elapsed timers, green `▶` prefix
- **Progress bar** (Rich `Progress`)
- **Summary counts** — updated on every completion

Height is `(currently running stages) + 2`. Bounded by `--jobs` (default 4), not total stages. Shrinks to just summary + progress when nothing's running between batches.

### Live sink: 1Hz timer tick

Running stage elapsed timers need to update even when no events arrive (long-running stages). An `anyio` background task sends a tick every second, triggering a `Live.update()` re-render. Cancelled on `close()`.

### Live sink: run end

When execution finishes, the running section empties, the progress bar fills, and `Live` exits. Clean scrollback remains — identical to what static mode would have produced, plus the final summary line.

### Static sink

Pure line-by-line output. No `Live`, no cursor manipulation. Works in CI, pipes, log files. Same completion line format and skip collapsing as the live sink's scrollback. Ends with:

```
  Done: 3 ran, 152 skipped, 1 failed (14.2s total)
```

### Flags

- **`--quiet`**: no output from either sink (unchanged from today)
- **`--json`**: `JsonlSink` handles it, `configure_output_sink()` returns early (unchanged)
- **`--show-output`**: log lines appear in scrollback (both sinks)

---

## Implementation

### Files changed

**`packages/pivot/src/pivot/engine/sinks.py`** — replace `ConsoleSink` with:
- `_format_stage_line()` — shared alignment, color, symbols
- `_format_error_detail()` — shared indented error lines
- `_format_summary()` — shared end summary
- `StaticConsoleSink` — buffered skip collapsing, line-by-line
- `LiveConsoleSink` — Rich `Live`, running set, 1Hz tick, `console.print()` for scrollback
- `ResultCollectorSink` — unchanged

**`packages/pivot/src/pivot/cli/_run_common.py`** — `configure_output_sink()` picks `LiveConsoleSink` or `StaticConsoleSink` based on `console.is_terminal`.

**`packages/pivot/src/pivot/cli/repro.py`** — same change for `--serve` mode sink setup.

### Files unchanged

- `engine/types.py` — all needed event types already exist
- `engine/engine.py` — event emission unchanged
- `pivot-tui/` — TUI is a separate code path

### Testing

Unit tests for formatting helpers (alignment, collapsing thresholds, color). Integration tests for both sinks: feed `OutputEvent` TypedDicts, assert output via `StringIO`-backed `Console`. Follow the existing `ResultCollectorSink` test pattern.
