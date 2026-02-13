# CLI Output Redesign — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace noisy `ConsoleSink` with two auto-detected sinks: `StaticConsoleSink` (pipes/CI) and `LiveConsoleSink` (TTY with Rich Live).

**Architecture:** Both sinks implement `EventSink` protocol. Shared formatting helpers handle alignment, color, symbols, and skip collapsing. The call site (`configure_output_sink`) picks sink based on `console.is_terminal`. The "waiting for artifact lock" message is dropped entirely.

**Tech Stack:** Rich Console, Rich Live, Rich Progress, anyio (for 1Hz tick in Live sink)

**Design:** See `docs/plans/2026-02-13-cli-output-redesign.md`

---

## Task 1: Shared Formatting Helpers

**Files:**
- Modify: `packages/pivot/src/pivot/engine/sinks.py`
- Test: `packages/pivot/tests/engine/test_sinks.py`

These are pure functions used by both sinks. Build and test them in isolation first.

**Step 1: Write the failing tests**

Add to `packages/pivot/tests/engine/test_sinks.py` — new section after the existing ConsoleSink tests. All imports at module level per project test rules.

```python
# =============================================================================
# Formatting Helper Tests
# =============================================================================


def test_format_stage_line_ran() -> None:
    """Completed (RAN) stage produces aligned line with green checkmark and duration."""
    from pivot.engine.sinks import _format_stage_line

    line = _format_stage_line(
        index=5, total=175, stage="train_model", status=StageStatus.RAN, duration_ms=3400.0, name_width=30,
    )
    # Should contain: progress counter, checkmark, stage name, "done", duration
    assert "[  6/175]" in line, "Progress counter should be 1-indexed and right-aligned"
    assert "✓" in line, "RAN status should show checkmark"
    assert "train_model" in line, "Stage name should appear"
    assert "done" in line, "Status word for RAN should be 'done'"
    assert "3.4s" in line, "Duration should be formatted as seconds"


def test_format_stage_line_skipped() -> None:
    """Skipped stage shows circle symbol, no duration."""
    from pivot.engine.sinks import _format_stage_line

    line = _format_stage_line(
        index=0, total=10, stage="fetch_data", status=StageStatus.SKIPPED, duration_ms=5.0, name_width=30,
    )
    assert "○" in line, "SKIPPED status should show circle"
    assert "skipped" in line, "Status word for SKIPPED should be 'skipped'"
    assert "0.0s" not in line, "Duration should NOT appear for skipped stages"


def test_format_stage_line_failed() -> None:
    """Failed stage shows X symbol."""
    from pivot.engine.sinks import _format_stage_line

    line = _format_stage_line(
        index=2, total=5, stage="export", status=StageStatus.FAILED, duration_ms=100.0, name_width=30,
    )
    assert "✗" in line, "FAILED status should show X"
    assert "FAILED" in line, "Status word for FAILED should be 'FAILED'"


def test_format_stage_line_truncates_long_name() -> None:
    """Stage names exceeding name_width are truncated with ellipsis."""
    from pivot.engine.sinks import _format_stage_line

    line = _format_stage_line(
        index=0, total=10, stage="a" * 60, status=StageStatus.RAN, duration_ms=1000.0, name_width=50,
    )
    assert "…" in line, "Long names should be truncated with ellipsis"
    assert "a" * 60 not in line, "Full long name should not appear"


def test_format_skip_group_line() -> None:
    """Collapsed skip group shows range and count."""
    from pivot.engine.sinks import _format_skip_group_line

    line = _format_skip_group_line(start_index=2, end_index=154, total=175, count=152)
    assert "152" in line, "Skip count should appear"
    assert "skipped" in line, "Should say 'skipped'"
    assert "○" in line, "Should use skip symbol"


def test_format_error_detail() -> None:
    """Error detail lines are indented and each line preserved."""
    from pivot.engine.sinks import _format_error_detail

    lines = _format_error_detail("Traceback:\n  File test.py\nValueError: bad", name_width=30)
    assert len(lines) == 3, "Should produce one output line per input line"
    assert all("  " in l for l in lines), "Each line should be indented"
    assert "Traceback" in lines[0]
    assert "ValueError" in lines[2]


def test_format_summary() -> None:
    """Summary line contains counts and total duration."""
    from pivot.engine.sinks import _format_summary

    line = _format_summary(ran=3, skipped=152, failed=1, total_ms=14200.0)
    assert "3 ran" in line
    assert "152 skipped" in line
    assert "1 failed" in line
    assert "14.2s" in line
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/pivot/tests/engine/test_sinks.py -k "format_stage_line or format_skip_group or format_error_detail or format_summary" -v`
Expected: FAIL — `_format_stage_line` doesn't exist yet

**Step 3: Implement the formatting helpers**

Add to `packages/pivot/src/pivot/engine/sinks.py`, above the class definitions. These are module-level functions prefixed with `_` since they're internal to the module.

Key decisions:
- Functions return plain strings (no Rich markup) — callers wrap in `console.print()` with markup enabled.
  Actually, return strings WITH Rich markup tags. The console handles rendering.
- `_format_stage_line` returns a single Rich-markup string.
- `_format_error_detail` returns a list of Rich-markup strings (one per error line).
- `_format_summary` returns a single Rich-markup string.
- `_format_skip_group_line` returns a single Rich-markup string.
- `name_width` parameter controls stage name column width. Callers track `max_name_seen`.

```python
import rich.markup

# These constants are used by both sinks
_STATUS_SYMBOL: dict[StageStatus, str] = {
    StageStatus.RAN: "[green]✓[/green]",
    StageStatus.SKIPPED: "[dim]○[/dim]",
    StageStatus.FAILED: "[bold red]✗[/bold red]",
}

_STATUS_WORD: dict[StageStatus, str] = {
    StageStatus.RAN: "[green]done[/green]",
    StageStatus.SKIPPED: "[dim]skipped[/dim]",
    StageStatus.FAILED: "[bold red]FAILED[/bold red]",
}

_MAX_NAME_WIDTH = 50


def _format_stage_line(
    *,
    index: int,
    total: int,
    stage: str,
    status: StageStatus,
    duration_ms: float,
    name_width: int,
) -> str:
    """Format a single stage completion line with alignment and color."""
    counter_width = len(str(total))
    counter = f"[{index + 1:>{counter_width}}/{total}]"

    display_name = stage
    capped_width = min(name_width, _MAX_NAME_WIDTH)
    if len(display_name) > capped_width:
        display_name = display_name[: capped_width - 1] + "…"

    symbol = _STATUS_SYMBOL[status]
    word = _STATUS_WORD[status]
    duration = f"  {duration_ms / 1000:.1f}s" if status == StageStatus.RAN else ""

    return f"  {counter} {symbol} [bold]{display_name:<{capped_width}}[/bold]  {word}{duration}"


def _format_skip_group_line(
    *, start_index: int, end_index: int, total: int, count: int,
) -> str:
    """Format a collapsed skip group line."""
    counter_width = len(str(total))
    counter = f"[{start_index + 1}–{end_index + 1}/{total}]"
    # Pad counter to roughly match individual line counters
    padded = f"{counter:>{counter_width * 2 + 4}}"
    return f"  {padded} [dim]○[/dim] [dim]{count} stages skipped[/dim]"


def _format_error_detail(reason: str, *, name_width: int) -> list[str]:
    """Format multi-line error detail, indented under the stage line."""
    # Indent to align with stage name column (counter + symbol + padding)
    indent = " " * 12  # Approximate alignment
    lines = []
    for line in reason.rstrip().split("\n"):
        escaped = rich.markup.escape(line)
        lines.append(f"{indent}[dim red]{escaped}[/dim red]")
    return lines


def _format_summary(*, ran: int, skipped: int, failed: int, total_ms: float) -> str:
    """Format the end-of-run summary line."""
    parts = []
    if ran:
        parts.append(f"[green]{ran} ran[/green]")
    if skipped:
        parts.append(f"[dim]{skipped} skipped[/dim]")
    if failed:
        parts.append(f"[bold red]{failed} failed[/bold red]")
    duration = f"{total_ms / 1000:.1f}s"
    return f"  Done: {', '.join(parts)} ({duration} total)"
```

**Step 4: Run the formatting tests**

Run: `uv run pytest packages/pivot/tests/engine/test_sinks.py -k "format_stage_line or format_skip_group or format_error_detail or format_summary" -v`
Expected: PASS

**Step 5: Commit**

```bash
jj desc -m "feat(cli): add shared formatting helpers for console sinks"
```

---

## Task 2: StaticConsoleSink

**Files:**
- Modify: `packages/pivot/src/pivot/engine/sinks.py`
- Test: `packages/pivot/tests/engine/test_sinks.py`

The simpler sink: line-by-line output with skip collapsing. No cursor manipulation. Works in CI/pipes.

**Step 1: Write the failing tests**

Add a new section to `test_sinks.py`:

```python
# =============================================================================
# StaticConsoleSink Tests
# =============================================================================


async def test_static_sink_prints_ran_stage() -> None:
    """StaticConsoleSink prints a formatted line for a completed RAN stage."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = StaticConsoleSink(console=console)

    await sink.handle(StageStarted(type="stage_started", stage="train", index=0, total=5))
    await sink.handle(StageCompleted(
        type="stage_completed", stage="train", status=StageStatus.RAN,
        reason="", duration_ms=1500.0, index=0, total=5, input_hash=None,
    ))
    await sink.close()

    result = output.getvalue()
    assert "1/5" in result, "Progress counter should appear"
    assert "train" in result, "Stage name should appear"
    assert "done" in result, "Status word should appear"
    assert "1.5s" in result, "Duration should appear"


async def test_static_sink_prints_skipped_stage() -> None:
    """StaticConsoleSink prints skipped stages individually when count is low."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = StaticConsoleSink(console=console)

    await sink.handle(StageCompleted(
        type="stage_completed", stage="fetch", status=StageStatus.SKIPPED,
        reason="up-to-date", duration_ms=5.0, index=0, total=5, input_hash=None,
    ))
    await sink.handle(StageCompleted(
        type="stage_completed", stage="train", status=StageStatus.RAN,
        reason="", duration_ms=1000.0, index=1, total=5, input_hash=None,
    ))
    await sink.close()

    result = output.getvalue()
    assert "fetch" in result, "Skipped stage should appear individually"
    assert "skipped" in result


async def test_static_sink_collapses_many_skips() -> None:
    """StaticConsoleSink collapses >20 consecutive skips into a summary line."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = StaticConsoleSink(console=console)

    # Send 25 skipped stages followed by one RAN stage to flush buffer
    for i in range(25):
        await sink.handle(StageCompleted(
            type="stage_completed", stage=f"stage_{i}", status=StageStatus.SKIPPED,
            reason="up-to-date", duration_ms=5.0, index=i, total=30, input_hash=None,
        ))
    await sink.handle(StageCompleted(
        type="stage_completed", stage="final", status=StageStatus.RAN,
        reason="", duration_ms=1000.0, index=25, total=30, input_hash=None,
    ))
    await sink.close()

    result = output.getvalue()
    # Should NOT list all 25 individually
    assert "stage_12" not in result, "Individual skip names should not appear when collapsed"
    assert "25" in result, "Skip count should appear"
    assert "skipped" in result, "Should say 'skipped'"


async def test_static_sink_does_not_collapse_few_skips() -> None:
    """StaticConsoleSink lists skips individually when <=20."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = StaticConsoleSink(console=console)

    for i in range(5):
        await sink.handle(StageCompleted(
            type="stage_completed", stage=f"stage_{i}", status=StageStatus.SKIPPED,
            reason="up-to-date", duration_ms=5.0, index=i, total=10, input_hash=None,
        ))
    # Flush with a ran stage
    await sink.handle(StageCompleted(
        type="stage_completed", stage="final", status=StageStatus.RAN,
        reason="", duration_ms=1000.0, index=5, total=10, input_hash=None,
    ))
    await sink.close()

    result = output.getvalue()
    for i in range(5):
        assert f"stage_{i}" in result, f"stage_{i} should appear individually"


async def test_static_sink_ignores_waiting_on_lock() -> None:
    """StaticConsoleSink does NOT print 'waiting for artifact lock'."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = StaticConsoleSink(console=console)

    await sink.handle(StageStateChanged(
        type="stage_state_changed", stage="train",
        state=StageExecutionState.WAITING_ON_LOCK,
        previous_state=StageExecutionState.PREPARING,
    ))
    await sink.close()

    assert output.getvalue() == "", "Should not print anything for WAITING_ON_LOCK"


async def test_static_sink_prints_failed_with_reason() -> None:
    """StaticConsoleSink prints FAILED stage with indented error detail."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = StaticConsoleSink(console=console)

    await sink.handle(StageCompleted(
        type="stage_completed", stage="export", status=StageStatus.FAILED,
        reason="KeyError: 'score'", duration_ms=100.0, index=0, total=5, input_hash=None,
    ))
    await sink.close()

    result = output.getvalue()
    assert "FAILED" in result, "Should show FAILED status"
    assert "KeyError" in result, "Error reason should appear"


async def test_static_sink_ignores_stage_started() -> None:
    """StaticConsoleSink does not print anything for stage_started events."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = StaticConsoleSink(console=console)

    await sink.handle(StageStarted(type="stage_started", stage="train", index=0, total=5))

    assert output.getvalue() == "", "Should not print for stage_started"


async def test_static_sink_prints_summary_on_close() -> None:
    """StaticConsoleSink prints summary line when closed."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = StaticConsoleSink(console=console)

    await sink.handle(StageCompleted(
        type="stage_completed", stage="a", status=StageStatus.RAN,
        reason="", duration_ms=1000.0, index=0, total=3, input_hash=None,
    ))
    await sink.handle(StageCompleted(
        type="stage_completed", stage="b", status=StageStatus.SKIPPED,
        reason="up-to-date", duration_ms=5.0, index=1, total=3, input_hash=None,
    ))
    await sink.handle(StageCompleted(
        type="stage_completed", stage="c", status=StageStatus.FAILED,
        reason="err", duration_ms=50.0, index=2, total=3, input_hash=None,
    ))
    await sink.close()

    result = output.getvalue()
    assert "1 ran" in result, "Summary should count ran"
    assert "1 skipped" in result, "Summary should count skipped"
    assert "1 failed" in result, "Summary should count failed"


async def test_static_sink_show_output_prints_log_lines() -> None:
    """StaticConsoleSink prints log lines in dim when show_output=True."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = StaticConsoleSink(console=console, show_output=True)

    await sink.handle(LogLine(
        type="log_line", stage="train", line="Epoch 1/10", is_stderr=False,
    ))
    await sink.close()

    result = output.getvalue()
    assert "train" in result, "Stage name prefix should appear"
    assert "Epoch 1/10" in result, "Log content should appear"


async def test_static_sink_hides_log_lines_by_default() -> None:
    """StaticConsoleSink ignores log lines when show_output=False."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = StaticConsoleSink(console=console)

    await sink.handle(LogLine(
        type="log_line", stage="train", line="Epoch 1/10", is_stderr=False,
    ))

    assert output.getvalue() == "", "Should not print log lines by default"


async def test_static_sink_log_line_flushes_skip_buffer() -> None:
    """Log lines arriving mid-skip-buffer flush pending skips first."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = StaticConsoleSink(console=console, show_output=True)

    # 3 skips then a log line
    for i in range(3):
        await sink.handle(StageCompleted(
            type="stage_completed", stage=f"s{i}", status=StageStatus.SKIPPED,
            reason="", duration_ms=5.0, index=i, total=10, input_hash=None,
        ))
    await sink.handle(LogLine(
        type="log_line", stage="train", line="hello", is_stderr=False,
    ))
    await sink.close()

    result = output.getvalue()
    lines = result.strip().split("\n")
    # Skipped stages should appear before the log line
    skip_line_indices = [i for i, l in enumerate(lines) if "skipped" in l]
    log_line_indices = [i for i, l in enumerate(lines) if "hello" in l]
    assert skip_line_indices, "Skipped lines should appear"
    assert log_line_indices, "Log line should appear"
    assert max(skip_line_indices) < min(log_line_indices), "Skips should flush before log line"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/pivot/tests/engine/test_sinks.py -k "static_sink" -v`
Expected: FAIL — `StaticConsoleSink` doesn't exist yet

**Step 3: Implement StaticConsoleSink**

Add to `packages/pivot/src/pivot/engine/sinks.py`. Key implementation details:

- Tracks: `_completed_count`, `_ran_count`, `_skipped_count`, `_failed_count`, `_total_duration_ms`, `_max_name_width`, `_skip_buffer` (list of `StageCompleted` events).
- `_skip_buffer` holds consecutive skipped stages. Flushed when a non-skip completion, log line (if `show_output`), or `close()` is called.
- Flush logic: if `len(buffer) <= 20`, print each individually; if `> 20`, print one collapsed line.
- On `close()`: flush remaining skip buffer, then print summary line.
- `stage_started`: ignored (no output).
- `stage_state_changed`: ignored entirely (no "waiting for lock").

**Step 4: Run the tests**

Run: `uv run pytest packages/pivot/tests/engine/test_sinks.py -k "static_sink" -v`
Expected: PASS

**Step 5: Commit**

```bash
jj desc -m "feat(cli): add StaticConsoleSink with skip collapsing and aligned output"
```

---

## Task 3: LiveConsoleSink

**Files:**
- Modify: `packages/pivot/src/pivot/engine/sinks.py`
- Test: `packages/pivot/tests/engine/test_sinks.py`

The complex sink: Rich `Live` pinned section with running stages, progress bar, and summary counts. Scrollback via `console.print()`.

**Step 1: Write the failing tests**

Testing `Live` is tricky — you can't easily capture Live output in a `StringIO`. Focus on testing:
1. State management (running set, completion tracking)
2. Scrollback output (completions and log lines printed via `console.print()`)
3. The pinned renderable content

Strategy: use `no_color=True` console to capture what `console.print()` emits (scrollback). For the Live renderable, test the `_build_live_renderable()` method directly.

```python
# =============================================================================
# LiveConsoleSink Tests
# =============================================================================


async def test_live_sink_prints_completion_to_scrollback() -> None:
    """LiveConsoleSink prints completed stage lines via console.print() (scrollback)."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = LiveConsoleSink(console=console)

    await sink.handle(StageCompleted(
        type="stage_completed", stage="train", status=StageStatus.RAN,
        reason="", duration_ms=1500.0, index=0, total=5, input_hash=None,
    ))
    await sink.close()

    result = output.getvalue()
    assert "train" in result, "Stage name should appear in scrollback"
    assert "done" in result, "Status should appear in scrollback"


async def test_live_sink_tracks_running_stages() -> None:
    """LiveConsoleSink adds stages to running set on stage_started."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = LiveConsoleSink(console=console)

    await sink.handle(StageStarted(type="stage_started", stage="train", index=0, total=5))

    assert "train" in sink._running, "Stage should be in running set after stage_started"

    await sink.handle(StageCompleted(
        type="stage_completed", stage="train", status=StageStatus.RAN,
        reason="", duration_ms=1000.0, index=0, total=5, input_hash=None,
    ))

    assert "train" not in sink._running, "Stage should be removed from running set after completion"
    await sink.close()


async def test_live_sink_ignores_waiting_on_lock() -> None:
    """LiveConsoleSink does NOT print 'waiting for artifact lock'."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = LiveConsoleSink(console=console)

    await sink.handle(StageStateChanged(
        type="stage_state_changed", stage="train",
        state=StageExecutionState.WAITING_ON_LOCK,
        previous_state=StageExecutionState.PREPARING,
    ))
    await sink.close()

    assert output.getvalue() == "", "Should not print anything for WAITING_ON_LOCK"


async def test_live_sink_collapses_many_skips_in_scrollback() -> None:
    """LiveConsoleSink uses same skip collapsing in scrollback as StaticConsoleSink."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = LiveConsoleSink(console=console)

    for i in range(25):
        await sink.handle(StageCompleted(
            type="stage_completed", stage=f"stage_{i}", status=StageStatus.SKIPPED,
            reason="up-to-date", duration_ms=5.0, index=i, total=30, input_hash=None,
        ))
    # Flush with a ran stage
    await sink.handle(StageCompleted(
        type="stage_completed", stage="final", status=StageStatus.RAN,
        reason="", duration_ms=1000.0, index=25, total=30, input_hash=None,
    ))
    await sink.close()

    result = output.getvalue()
    assert "stage_12" not in result, "Individual skip names should not appear when collapsed"
    assert "25" in result, "Skip count should appear"


async def test_live_sink_summary_counts() -> None:
    """LiveConsoleSink tracks ran/skipped/failed counts for summary."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = LiveConsoleSink(console=console)

    await sink.handle(StageCompleted(
        type="stage_completed", stage="a", status=StageStatus.RAN,
        reason="", duration_ms=1000.0, index=0, total=3, input_hash=None,
    ))
    await sink.handle(StageCompleted(
        type="stage_completed", stage="b", status=StageStatus.SKIPPED,
        reason="", duration_ms=5.0, index=1, total=3, input_hash=None,
    ))
    await sink.handle(StageCompleted(
        type="stage_completed", stage="c", status=StageStatus.FAILED,
        reason="err", duration_ms=50.0, index=2, total=3, input_hash=None,
    ))
    await sink.close()

    result = output.getvalue()
    assert "1 ran" in result
    assert "1 skipped" in result
    assert "1 failed" in result


async def test_live_sink_show_output_prints_log_lines() -> None:
    """LiveConsoleSink prints log lines to scrollback when show_output=True."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = LiveConsoleSink(console=console, show_output=True)

    await sink.handle(LogLine(
        type="log_line", stage="train", line="Epoch 1", is_stderr=False,
    ))
    await sink.close()

    result = output.getvalue()
    assert "train" in result
    assert "Epoch 1" in result
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/pivot/tests/engine/test_sinks.py -k "live_sink" -v`
Expected: FAIL — `LiveConsoleSink` doesn't exist

**Step 3: Implement LiveConsoleSink**

Add to `packages/pivot/src/pivot/engine/sinks.py`. Key implementation details:

**State:**
- `_running: dict[str, float]` — stage name → start timestamp (for elapsed timers)
- `_completed_count`, `_ran_count`, `_skipped_count`, `_failed_count`, `_total_duration_ms`
- `_total: int` — total stage count (set from first event's `total` field)
- `_max_name_width: int` — max stage name length seen so far
- `_skip_buffer: list[StageCompleted]` — same collapsing logic as StaticConsoleSink
- `_live: rich.live.Live | None` — the Rich Live context, created lazily on first event
- `_tick_scope: anyio.CancelScope | None` — scope for the 1Hz tick task

**Event handling:**
- `stage_started`: add to `_running` dict with `time.monotonic()` timestamp. Update Live.
- `stage_completed`: remove from `_running`. Buffer skips or flush+print. Update Live.
- `stage_state_changed`: ignored entirely.
- `log_line` (if `show_output`): flush skip buffer, `console.print()` dim line. Update Live.
- `engine_diagnostic`: flush skip buffer, `console.print()` warning. Update Live.

**Live renderable (`_build_live_renderable`):**
Return a `rich.console.Group` containing:
1. Running stage lines: `▶ stage_name    running…  3.2s` (green, with elapsed time from `_running` dict)
2. Progress bar: `rich.progress_bar.ProgressBar` or a formatted string
3. Summary line: `3 ran · 152 skipped · 1 failed`

**1Hz tick:**
- Started in `handle()` on the first event (not in `__init__`, to avoid needing an event loop at construction).
- Uses `anyio.CancelScope` + `anyio.sleep(1)` loop.
- Each tick calls `_live.update(self._build_live_renderable())`.
- **Important:** The sink's `handle()` is called from the engine's async dispatch loop, so we can't start a background task from `handle()` directly. Instead, the tick task must be started by the engine or passed in. **Alternative:** Since `Live` itself can set `refresh_per_second=1`, we might not need a manual tick at all — just call `_live.update()` on every event, and Rich's Live will re-render at 1Hz automatically, picking up the updated elapsed times from `_running` dict by calling `_build_live_renderable` as the Live's `get_renderable`.

**Simpler approach for ticking:** Pass `_build_live_renderable` as the Live's renderable (a callable). Rich Live calls `get_renderable()` on every refresh. Set `refresh_per_second=1`. Then:
- On each event, update internal state. Live re-renders at 1Hz automatically.
- Call `_live.update(self._build_live_renderable())` on events for immediate feedback too.

This eliminates the need for a separate anyio tick task entirely.

**`close()`:**
- Flush remaining skip buffer via `console.print()`.
- Print summary line via `console.print()`.
- Stop Live context (`_live.stop()` / `__exit__`).

**Note on testing:** Since `Live` uses terminal control codes, tests use `force_terminal=False` consoles where `Live` degrades gracefully (no live updates). The scrollback output (`console.print()` calls) is still captured and testable. The live renderable building logic can be tested separately by calling `_build_live_renderable()` directly.

**Step 4: Run the tests**

Run: `uv run pytest packages/pivot/tests/engine/test_sinks.py -k "live_sink" -v`
Expected: PASS

**Step 5: Commit**

```bash
jj desc -m "feat(cli): add LiveConsoleSink with Rich Live pinned section"
```

---

## Task 4: Wire Up Call Sites

**Files:**
- Modify: `packages/pivot/src/pivot/cli/_run_common.py:262-283`
- Modify: `packages/pivot/src/pivot/cli/repro.py:519-522`
- Modify: `packages/pivot/src/pivot/engine/sinks.py` (`__all__`)
- Test: `packages/pivot/tests/cli/test_cli_run_common.py:523-539`

**Step 1: Write the failing tests**

Update the existing test in `test_cli_run_common.py`:

```python
def test_configure_output_sink_console_mode_picks_static_for_non_terminal(
    mocker: MockerFixture,
) -> None:
    """configure_output_sink adds StaticConsoleSink when console is not a terminal."""
    mock_engine = mocker.MagicMock(spec=engine.Engine)
    # Patch Console to report non-terminal
    mock_console = mocker.patch("pivot.cli._run_common.rich.console.Console")
    mock_console.return_value.is_terminal = False

    _run_common.configure_output_sink(
        mock_engine, quiet=False, as_json=False, use_console=True, jsonl_callback=None,
    )

    mock_engine.add_sink.assert_called_once()
    added_sink = mock_engine.add_sink.call_args[0][0]
    assert isinstance(added_sink, sinks.StaticConsoleSink)


def test_configure_output_sink_console_mode_picks_live_for_terminal(
    mocker: MockerFixture,
) -> None:
    """configure_output_sink adds LiveConsoleSink when console is a terminal."""
    mock_engine = mocker.MagicMock(spec=engine.Engine)
    mock_console = mocker.patch("pivot.cli._run_common.rich.console.Console")
    mock_console.return_value.is_terminal = True

    _run_common.configure_output_sink(
        mock_engine, quiet=False, as_json=False, use_console=True, jsonl_callback=None,
    )

    mock_engine.add_sink.assert_called_once()
    added_sink = mock_engine.add_sink.call_args[0][0]
    assert isinstance(added_sink, sinks.LiveConsoleSink)
```

**Step 2: Run to verify failure**

Run: `uv run pytest packages/pivot/tests/cli/test_cli_run_common.py -k "picks_static or picks_live" -v`
Expected: FAIL

**Step 3: Update configure_output_sink**

In `_run_common.py`, change `configure_output_sink()`:

```python
def configure_output_sink(
    eng: engine.Engine,
    *,
    quiet: bool,
    as_json: bool,
    use_console: bool,
    jsonl_callback: Callable[[dict[str, object]], None] | None,
    show_output: bool = False,
) -> None:
    """Configure output sinks based on display mode."""
    import rich.console

    if as_json and jsonl_callback:
        eng.add_sink(JsonlSink(callback=jsonl_callback))
        return

    if quiet:
        return

    if use_console:
        console = rich.console.Console()
        if console.is_terminal:
            eng.add_sink(sinks.LiveConsoleSink(console=console, show_output=show_output))
        else:
            eng.add_sink(sinks.StaticConsoleSink(console=console, show_output=show_output))
```

In `repro.py`, update the `--serve` mode sink setup similarly:

```python
if not quiet:
    serve_console = rich.console.Console()
    if serve_console.is_terminal:
        eng.add_sink(sinks.LiveConsoleSink(console=serve_console, show_output=show_output))
    else:
        eng.add_sink(sinks.StaticConsoleSink(console=serve_console, show_output=show_output))
```

Update `sinks.py` `__all__` to export the new classes:

```python
__all__ = [
    "LiveConsoleSink",
    "ResultCollectorSink",
    "StaticConsoleSink",
]
```

**Step 4: Run all affected tests**

Run: `uv run pytest packages/pivot/tests/cli/test_cli_run_common.py packages/pivot/tests/engine/test_sinks.py -v`
Expected: PASS (new tests pass, but existing ConsoleSink tests will fail — addressed in Task 5)

**Step 5: Commit**

```bash
jj desc -m "feat(cli): auto-detect TTY for Live vs Static console sink"
```

---

## Task 5: Remove Old ConsoleSink and Update Tests

**Files:**
- Modify: `packages/pivot/src/pivot/engine/sinks.py` (remove `ConsoleSink`)
- Modify: `packages/pivot/tests/engine/test_sinks.py` (update/remove old tests)
- Modify: `packages/pivot/tests/engine/test_lock_state.py` (update tests)
- Modify: `packages/pivot/tests/cli/test_cli_run_common.py` (update isinstance checks)

**Step 1: Remove ConsoleSink class from sinks.py**

Delete the `ConsoleSink` class entirely. It's replaced by `StaticConsoleSink` and `LiveConsoleSink`. Remove from `__all__`.

**Step 2: Update test_sinks.py**

The old `ConsoleSink` tests (`test_console_sink_*`) need to be either:
- **Deleted** if covered by the new `StaticConsoleSink`/`LiveConsoleSink` tests (most are).
- **Migrated** if they test behavior still relevant (Rich markup escaping, etc.).

Tests to migrate to `StaticConsoleSink` equivalents:
- `test_console_sink_escapes_rich_markup_in_log_lines` → test that `StaticConsoleSink` escapes markup in log lines
- `test_console_sink_handles_special_characters` → test special chars in `StaticConsoleSink`

Tests to delete (covered by new tests):
- `test_console_sink_handles_stage_started` (Static ignores; Live tracks running set)
- `test_console_sink_handles_stage_completed_ran` → `test_static_sink_prints_ran_stage`
- `test_console_sink_handles_stage_completed_skipped` → `test_static_sink_prints_skipped_stage`
- `test_console_sink_handles_stage_completed_failed` → `test_static_sink_prints_failed_with_reason`
- `test_console_sink_handles_multiline_reason` → covered by `_format_error_detail` test
- `test_console_sink_ignores_other_events` → implicit (sinks only handle known events)
- `test_console_sink_formats_duration_correctly` → `test_format_stage_line_ran`
- `test_console_sink_running_message_format` → deleted (no more "Running..." line)
- `test_console_sink_handles_log_line_when_show_output_enabled` → `test_static_sink_show_output_prints_log_lines`
- `test_console_sink_stderr_line_contains_content` → add stderr test for new sinks
- `test_console_sink_ignores_log_line_when_show_output_disabled` → `test_static_sink_hides_log_lines_by_default`
- `test_console_sink_handles_empty_log_line` → add for new sinks if needed
- `test_console_sink_handles_multiline_log_output` → add for new sinks if needed

**Step 3: Update test_lock_state.py**

- `test_console_sink_displays_waiting_on_lock` → **Invert the assertion**: both new sinks should NOT print waiting-on-lock messages. Replace with tests for both sinks asserting empty output for WAITING_ON_LOCK events. Or simply rename the test to `test_static_sink_ignores_waiting_on_lock` and `test_live_sink_ignores_waiting_on_lock` (already in Task 2 and 3 tests).
- `test_console_sink_ignores_other_state_changes` → already covered by new tests.

**Step 4: Update test_cli_run_common.py**

- `test_configure_output_sink_console_mode` → Replace with the TTY/non-TTY tests from Task 4.

**Step 5: Run full test suite for sinks**

Run: `uv run pytest packages/pivot/tests/engine/test_sinks.py packages/pivot/tests/engine/test_lock_state.py packages/pivot/tests/cli/test_cli_run_common.py -v`
Expected: PASS

**Step 6: Run quality checks**

Run: `uv run ruff format packages/pivot/src/pivot/engine/sinks.py packages/pivot/tests/engine/test_sinks.py && uv run ruff check packages/pivot/src/pivot/engine/sinks.py packages/pivot/tests/engine/test_sinks.py && uv run basedpyright packages/pivot/src/pivot/engine/sinks.py`
Expected: Clean

**Step 7: Commit**

```bash
jj desc -m "refactor(cli): remove old ConsoleSink, migrate tests to new sinks"
```

---

## Task 6: Full Test Suite and Polish

**Files:**
- All modified files from Tasks 1-5

**Step 1: Run the full test suite**

Run: `uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto`
Expected: PASS (or only pre-existing failures unrelated to this change)

**Step 2: Run quality checks**

Run: `uv run ruff format . && uv run ruff check . && uv run basedpyright`
Expected: Clean

**Step 3: Manual verification**

If possible, run `pivot repro` on a real pipeline to visually verify:
1. Static mode (pipe): `pivot repro 2>&1 | cat` — should show aligned, collapsed output
2. Live mode (TTY): `pivot repro` — should show pinned running stages + progress bar
3. Quiet mode: `pivot repro --quiet` — should show nothing
4. Show output: `pivot repro --show-output` — should show dim log lines in scrollback

**Step 4: Final commit**

```bash
jj desc -m "feat(cli): redesign CLI output with auto-detected Live/Static console sinks"
```
