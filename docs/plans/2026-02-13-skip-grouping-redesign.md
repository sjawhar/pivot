# Skip Grouping Redesign — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make cached/blocked/cancelled stage display consistent — always collapse groups of 2+, dim singles, label with actual reason instead of "not run", and split groups by category.

**Architecture:** Changes are contained to `sinks.py` (formatting + collapsing logic) and `test_sinks.py`. No changes to engine, events, or TUI.

**Tech Stack:** Rich Console (existing)

---

### Task 1: Update `_format_skip_group_line` to accept category

**Files:**
- Modify: `packages/pivot/src/pivot/engine/sinks.py:66-69`
- Test: `packages/pivot/tests/engine/test_sinks.py:108-113`

**Step 1: Update the test**

In `test_format_skip_group_line_includes_range_and_count`, add `category=DisplayCategory.CACHED` to the call and change the assertion from `"3 stages not run"` to `"3 cached"`.

```python
def test_format_skip_group_line_includes_range_and_count() -> None:
    line = sinks._format_skip_group_line(
        start_index=1, end_index=3, total=9, count=3, category=DisplayCategory.CACHED,
    )
    rendered = _helper_render_markup(line)
    assert "1–3/9" in rendered, "Should include collapsed range"
    assert "3 cached" in rendered, "Should include count with category label"
    assert "○" in rendered, "Should include skip symbol"
```

**Step 2: Update the implementation**

Add `category: DisplayCategory` parameter to `_format_skip_group_line`. Use `category.value` (the plain string: "cached", "blocked", "cancelled") in the label instead of "stages not run".

```python
def _format_skip_group_line(
    *, start_index: int, end_index: int, total: int, count: int, category: DisplayCategory,
) -> str:
    total_digits = len(str(total))
    range_text = f"{start_index:>{total_digits}}–{end_index:>{total_digits}}/{total}"
    return f"[dim]  [{range_text}] ○ {count} {category.value}[/dim]"
```

**Step 3: Update the single call site in `_print_completions`**

Pass the group's category when calling `_format_skip_group_line`. (The grouping logic change comes in Task 3.)

**Step 4: Run test**

```bash
uv run pytest packages/pivot/tests/engine/test_sinks.py::test_format_skip_group_line_includes_range_and_count -v
```

Expected: PASS

**Step 5: Commit**

```bash
jj new && jj desc -m "refactor(cli): skip group label shows category name instead of 'not run'"
```

---

### Task 2: Dim single skip stage lines

**Files:**
- Modify: `packages/pivot/src/pivot/engine/sinks.py:44-63`
- Test: `packages/pivot/tests/engine/test_sinks.py:64-76`

**Step 1: Update `_format_stage_line`**

For skip categories (cached/blocked/cancelled), wrap the entire line in `[dim]` and drop `[bold]` from the stage name. The symbol and word markup will nest inside dim.

```python
def _format_stage_line(
    *,
    index: int,
    total: int,
    stage: str,
    category: DisplayCategory,
    duration_ms: float,
    name_width: int,
) -> str:
    total_digits = len(str(total))
    counter = f"[{index:>{total_digits}}/{total}]"
    display_width = max(1, min(name_width, _MAX_NAME_WIDTH))
    stage_name = f"{stage[: display_width - 1]}…" if len(stage) > display_width else stage
    padded_name = stage_name.ljust(display_width)
    symbol = _CATEGORY_SYMBOL.get(category, "[dim]?[/dim]")
    word = _CATEGORY_WORD.get(category, f"[dim]{category.value}[/dim]")
    if category == DisplayCategory.SUCCESS:
        duration_s = duration_ms / 1000
        return f"  {counter} {symbol} [bold]{padded_name}[/bold] {word} {duration_s:.1f}s"
    if category in _SKIP_CATEGORIES:
        return f"[dim]  {counter} {symbol} {padded_name} {word}[/dim]"
    return f"  {counter} {symbol} [bold]{padded_name}[/bold] {word}"
```

The key change: skip category lines get `[dim]` wrapping and no `[bold]` on the name.

**Step 2: Verify existing test still passes**

The test `test_format_stage_line_skipped_omits_duration` should still pass — it checks for `○` and `cached` in the rendered output, which will still be present (just dimmed).

```bash
uv run pytest packages/pivot/tests/engine/test_sinks.py::test_format_stage_line_skipped_omits_duration -v
```

Expected: PASS

**Step 3: Commit**

```bash
jj new && jj desc -m "refactor(cli): dim entire line for cached/blocked/cancelled stages"
```

---

### Task 3: Rewrite skip grouping logic in `_print_completions`

**Files:**
- Modify: `packages/pivot/src/pivot/engine/sinks.py:39-41,98-129`
- Test: `packages/pivot/tests/engine/test_sinks.py` (multiple tests)

This is the core change. The new algorithm:
1. Group consecutive skips by **same category** (not all skip categories together)
2. **Always collapse** groups of 2+ (no threshold)
3. Singles show individually (already dimmed from Task 2)

**Step 1: Remove `_SKIP_COLLAPSE_THRESHOLD`**

Delete line 41: `_SKIP_COLLAPSE_THRESHOLD = 20`

**Step 2: Rewrite the skip grouping in `_print_completions`**

Replace the current inner loop (lines 98-129):

```python
i = 0
while i < len(sorted_events):
    event = sorted_events[i]
    category = _categorize(event)
    if category in _SKIP_CATEGORIES:
        # Collect consecutive skips of the SAME category
        same_cat_group = [event]
        j = i + 1
        while j < len(sorted_events) and _categorize(sorted_events[j]) == category:
            same_cat_group.append(sorted_events[j])
            j += 1

        if len(same_cat_group) >= 2:
            line = _format_skip_group_line(
                start_index=same_cat_group[0]["index"],
                end_index=same_cat_group[-1]["index"],
                total=total,
                count=len(same_cat_group),
                category=category,
            )
            console.print(line)
        else:
            # Single skip — show individually (already dimmed)
            line = _format_stage_line(
                index=event["index"],
                total=total,
                stage=event["stage"],
                category=category,
                duration_ms=event["duration_ms"],
                name_width=max_name_width,
            )
            console.print(line)
        i = j
    else:
        line = _format_stage_line(
            index=event["index"],
            total=total,
            stage=event["stage"],
            category=category,
            duration_ms=event["duration_ms"],
            name_width=max_name_width,
        )
        console.print(line)
        if category == DisplayCategory.FAILED:
            stage_log = logs.get(event["stage"], [])
            if stage_log:
                for log_line in stage_log:
                    escaped = rich.markup.escape(log_line)
                    console.print(f"          [dim]{escaped}[/dim]")
            if event["reason"]:
                for detail in _format_error_detail(event["reason"], total=total):
                    console.print(detail)
        i += 1
```

**Step 3: Update tests**

**Rewrite `test_static_sink_prints_skipped_stages_individually_when_low_count`** — currently creates 2 cached stages (which will now collapse). Change to test a **single** cached stage between run stages:

```python
async def test_static_sink_shows_single_cached_stage_with_name() -> None:
    """A single cached stage between runs shows its name (not collapsed)."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.StaticConsoleSink(console=console)

    events = [
        StageCompleted(
            type="stage_completed", seq=0, stage="ran_first",
            status=StageStatus.RAN, reason="", duration_ms=100.0,
            index=1, total=3, run_id="test-run", input_hash=None,
        ),
        StageCompleted(
            type="stage_completed", seq=1, stage="cached_middle",
            status=StageStatus.CACHED, reason="up-to-date", duration_ms=10.0,
            index=2, total=3, run_id="test-run", input_hash=None,
        ),
        StageCompleted(
            type="stage_completed", seq=2, stage="ran_last",
            status=StageStatus.RAN, reason="", duration_ms=200.0,
            index=3, total=3, run_id="test-run", input_hash=None,
        ),
    ]
    for event in events:
        await sink.handle(event)
    await sink.close()

    result = output.getvalue()
    assert "cached_middle" in result, "Single cached stage should show its name"
    assert "cached" in result, "Should show cached status"
```

**Update `test_static_sink_collapses_skips_over_threshold`** — rename and reduce to 3 stages (any group of 2+ collapses now):

```python
async def test_static_sink_collapses_consecutive_cached_stages() -> None:
    """Two or more consecutive cached stages collapse into a single line."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.StaticConsoleSink(console=console)

    for idx in range(3):
        event = StageCompleted(
            type="stage_completed", seq=idx, stage=f"skip_{idx}",
            status=StageStatus.CACHED, reason="up-to-date", duration_ms=10.0,
            index=idx + 1, total=3, run_id="test-run", input_hash=None,
        )
        await sink.handle(event)
    await sink.close()

    result = output.getvalue()
    assert "3 cached" in result, "Should collapse with count and category"
    assert "skip_0" not in result, "Individual names should not appear"
    assert "skip_1" not in result, "Individual names should not appear"
    assert "skip_2" not in result, "Individual names should not appear"
```

**Delete `test_static_sink_does_not_collapse_skips_at_threshold`** — threshold no longer exists. All groups of 2+ collapse.

**Update `test_live_sink_collapses_many_skips_in_scrollback`** — update assertion:

```python
async def test_live_sink_collapses_many_skips_in_scrollback() -> None:
    ...
    assert "21 cached" in result, "Should collapse skipped stages with category label"
```

**Add new test: different categories form separate groups:**

```python
async def test_static_sink_splits_groups_by_category() -> None:
    """Consecutive skips of different categories form separate collapsed lines."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.StaticConsoleSink(console=console)

    # 3 cached then 2 blocked
    for idx in range(3):
        await sink.handle(StageCompleted(
            type="stage_completed", seq=idx, stage=f"cached_{idx}",
            status=StageStatus.CACHED, reason="up-to-date", duration_ms=10.0,
            index=idx + 1, total=5, run_id="test-run", input_hash=None,
        ))
    for idx in range(2):
        await sink.handle(StageCompleted(
            type="stage_completed", seq=3 + idx, stage=f"blocked_{idx}",
            status=StageStatus.BLOCKED, reason="upstream failed", duration_ms=0.0,
            index=4 + idx, total=5, run_id="test-run", input_hash=None,
        ))
    await sink.close()

    result = output.getvalue()
    assert "3 cached" in result, "Cached group should collapse separately"
    assert "2 blocked" in result, "Blocked group should collapse separately"
```

**Add new test: alternating categories produce no collapses** (Oracle edge case):

```python
async def test_static_sink_alternating_categories_no_collapse() -> None:
    """Alternating skip categories (cached, blocked, cached) stay individual — no group of 2+."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.StaticConsoleSink(console=console)

    statuses = [StageStatus.CACHED, StageStatus.BLOCKED, StageStatus.CACHED]
    for idx, status in enumerate(statuses):
        await sink.handle(StageCompleted(
            type="stage_completed", seq=idx, stage=f"stage_{idx}",
            status=status, reason="", duration_ms=10.0,
            index=idx + 1, total=3, run_id="test-run", input_hash=None,
        ))
    await sink.close()

    result = output.getvalue()
    assert "stage_0" in result, "Each alternating skip should show its name"
    assert "stage_1" in result, "Each alternating skip should show its name"
    assert "stage_2" in result, "Each alternating skip should show its name"
```

**Add new test: non-skip interrupting skips splits the groups** (Oracle edge case):

```python
async def test_static_sink_success_interrupts_cached_groups() -> None:
    """A success stage between cached runs produces two separate groups/singles."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.StaticConsoleSink(console=console)

    events = [
        StageCompleted(
            type="stage_completed", seq=0, stage="cached_a",
            status=StageStatus.CACHED, reason="", duration_ms=10.0,
            index=1, total=4, run_id="test-run", input_hash=None,
        ),
        StageCompleted(
            type="stage_completed", seq=1, stage="cached_b",
            status=StageStatus.CACHED, reason="", duration_ms=10.0,
            index=2, total=4, run_id="test-run", input_hash=None,
        ),
        StageCompleted(
            type="stage_completed", seq=2, stage="ran_middle",
            status=StageStatus.RAN, reason="", duration_ms=500.0,
            index=3, total=4, run_id="test-run", input_hash=None,
        ),
        StageCompleted(
            type="stage_completed", seq=3, stage="cached_c",
            status=StageStatus.CACHED, reason="", duration_ms=10.0,
            index=4, total=4, run_id="test-run", input_hash=None,
        ),
    ]
    for event in events:
        await sink.handle(event)
    await sink.close()

    result = output.getvalue()
    assert "2 cached" in result, "First pair should collapse"
    assert "ran_middle" in result, "Success stage should show normally"
    assert "cached_c" in result, "Trailing single cached should show its name"
    assert "cached_a" not in result, "Collapsed pair should not show individual names"
    assert "cached_b" not in result, "Collapsed pair should not show individual names"
```

**Step 4: Run all sink tests**

```bash
uv run pytest packages/pivot/tests/engine/test_sinks.py -v
```

Expected: PASS

**Step 5: Run full quality checks**

```bash
uv run ruff format packages/pivot/src/pivot/engine/sinks.py packages/pivot/tests/engine/test_sinks.py && uv run ruff check packages/pivot/src/pivot/engine/sinks.py packages/pivot/tests/engine/test_sinks.py && uv run basedpyright packages/pivot/src/pivot/engine/sinks.py
```

Expected: Clean

**Step 6: Run the full test suite**

```bash
uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto
```

Expected: PASS

**Step 7: Commit**

```bash
jj new && jj desc -m "refactor(cli): always collapse 2+ skip groups, split by category, remove threshold"
```
