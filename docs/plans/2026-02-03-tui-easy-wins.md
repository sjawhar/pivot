# TUI Easy Wins & Polish Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 6 small TUI improvements: terminal bell, status symbols, fixed-width stage list, active tab styling, group header styling, and context-aware footer.

**Architecture:** Each change is isolated. Status symbols are in `widgets/status.py`. Layout/styling in `styles/pivot.tcss`. Bell and footer logic in `run.py`. We'll use TDD where practical (symbol changes, footer logic), and manual verification for CSS-only changes.

**Tech Stack:** Python 3.13+, Textual TUI framework, TCSS styling

---

## Task 1: Terminal Bell on Completion

Add `self.bell()` when pipeline execution completes. This helps users running long pipelines in background terminals.

**Files:**
- Modify: `src/pivot/tui/run.py:649-665` (the `on_executor_complete` method)
- Test: `tests/tui/test_run.py` (add test for bell)

**Step 1: Write the failing test**

```python
# Add to tests/tui/test_run.py

async def test_bell_on_completion(mocker: MockerFixture) -> None:
    """Terminal bell should sound when pipeline completes."""
    bell_mock = mocker.patch.object(PivotApp, "bell")

    async with PivotApp(
        stages={},
        executor=lambda app, update_cb: {},
        run_mode=True,
    ).run_test() as pilot:
        # Simulate completion
        pilot.app.post_message(ExecutorComplete(results={}, error=None))
        await pilot.pause()

        bell_mock.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tui/test_run.py::test_bell_on_completion -v`
Expected: FAIL (bell not called)

**Step 3: Write minimal implementation**

In `src/pivot/tui/run.py`, modify `on_executor_complete`:

```python
def on_executor_complete(self, event: ExecutorComplete) -> None:
    self._results = event.results
    self._error = event.error

    # Sound terminal bell to notify user
    self.bell()

    if event.error:
        self.title = f"pivot run - FAILED: {event.error}"
    else:
        self.title = "pivot run - Complete"
    self._shutdown_event.set()
    self._close_log_file()
    self._shutdown_loky_pool()
    self.exit(self._results)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tui/test_run.py::test_bell_on_completion -v`
Expected: PASS

**Step 5: Run quality checks**

Run: `uv run ruff format src/pivot/tui/run.py && uv run ruff check src/pivot/tui/run.py && uv run basedpyright src/pivot/tui/run.py`
Expected: No errors

**Step 6: Commit**

```bash
jj describe -m "feat(tui): add terminal bell on pipeline completion

Helps users running long pipelines in background terminals.

Closes part of #328"
```

---

## Task 2: Update Status Symbols

Change cached symbol from `$` to `↺` and blocked symbol from `⊘` to `◇`.

**Files:**
- Modify: `src/pivot/tui/widgets/status.py:23-30` (the `_CATEGORY_SYMBOLS` dict)
- Test: `tests/tui/widgets/test_status.py` (verify symbol changes)

**Step 1: Write the failing tests**

```python
# Add to tests/tui/widgets/test_status.py (or create if doesn't exist)

from pivot.tui.widgets.status import get_status_symbol
from pivot.tui.types import StageStatus


def test_cached_symbol_is_recycle() -> None:
    """Cached stages should use ↺ (recycle) symbol."""
    symbol, _ = get_status_symbol(StageStatus.SKIPPED, reason="cache hit")
    assert symbol == "↺"


def test_blocked_symbol_is_diamond() -> None:
    """Blocked stages should use ◇ (hollow diamond) symbol."""
    symbol, _ = get_status_symbol(StageStatus.READY, reason="blocked by failed dependency")
    assert symbol == "◇"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tui/widgets/test_status.py -v -k "cached_symbol or blocked_symbol"`
Expected: FAIL (symbols are `$` and `⊘`)

**Step 3: Write minimal implementation**

In `src/pivot/tui/widgets/status.py`, find `_CATEGORY_SYMBOLS` dict and update:

```python
_CATEGORY_SYMBOLS: dict[DisplayCategory, tuple[str, str]] = {
    DisplayCategory.PENDING: ("○", "dim"),
    DisplayCategory.RUNNING: ("▶", "blue bold"),
    DisplayCategory.SUCCESS: ("●", "green bold"),
    DisplayCategory.CACHED: ("↺", "yellow"),      # Changed from "$"
    DisplayCategory.BLOCKED: ("◇", "red"),        # Changed from "⊘"
    DisplayCategory.CANCELLED: ("!", "yellow dim"),
    DisplayCategory.FAILED: ("✗", "red bold"),
    DisplayCategory.UNKNOWN: ("?", "dim"),
}
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tui/widgets/test_status.py -v -k "cached_symbol or blocked_symbol"`
Expected: PASS

**Step 5: Update any existing tests that assert old symbols**

Run: `uv run pytest tests/tui/ -v`
If any tests fail due to old symbol assertions, update them to match new symbols.

**Step 6: Run quality checks**

Run: `uv run ruff format src/pivot/tui/widgets/status.py && uv run ruff check src/pivot/tui/widgets/status.py && uv run basedpyright src/pivot/tui/widgets/status.py`
Expected: No errors

**Step 7: Commit**

```bash
jj describe -m "feat(tui): update cached and blocked status symbols

- Cached: $ → ↺ (recycle/reuse is clearer than dollar sign)
- Blocked: ⊘ → ◇ (hollow diamond = waiting, less aggressive)

Closes part of #328"
```

---

## Task 3: Fixed-Width Stage List

Change stage list from percentage-based to fixed-width (32 chars) for better space utilization on wide monitors.

**Files:**
- Modify: `src/pivot/tui/styles/pivot.tcss` (the `#stage-list` rule)

**Step 1: Locate current styling**

Current CSS:
```css
#stage-list {
    width: 40%;
    min-width: 35;
    max-width: 60;
    /* other rules... */
}
```

**Step 2: Update to fixed width**

Change to:
```css
#stage-list {
    width: 32;
    /* keep other rules unchanged */
}
```

Remove `min-width` and `max-width` since they're not needed with fixed width.

**Step 3: Manual verification**

Run: `uv run pivot repro --dry-run` (or any command that launches the TUI)
Verify: Stage list is fixed at 32 characters, detail panel gets remaining space

**Step 4: Run existing TUI tests**

Run: `uv run pytest tests/tui/ -v`
Expected: All pass (CSS changes shouldn't break tests)

**Step 5: Commit**

```bash
jj describe -m "feat(tui): use fixed-width stage list

32 character fixed width instead of 40% for better space
utilization on wide monitors.

Closes part of #328"
```

---

## Task 4: Active Tab Styling

Make the active tab more visually distinct with background color.

**Files:**
- Modify: `src/pivot/tui/styles/pivot.tcss` (add/modify tab styling)

**Step 1: Research current Textual tab styling**

Check the current TCSS for any existing tab rules. The TabbedContent widget uses:
- `Tab` widget for each tab
- `.-active` class on the selected tab
- `Tabs` container widget

**Step 2: Add active tab styling**

Add to `pivot.tcss`:

```css
/* Active tab styling */
Tab.-active {
    background: $primary-darken-2;
    text-style: bold;
}

Tab:hover {
    background: $primary-darken-3;
}
```

**Step 3: Manual verification**

Run TUI and verify:
- Active tab has visible background color
- Inactive tabs are clearly different
- Hover state provides feedback

**Step 4: Run existing TUI tests**

Run: `uv run pytest tests/tui/ -v`
Expected: All pass

**Step 5: Commit**

```bash
jj describe -m "feat(tui): add active tab background styling

Makes the currently selected tab clearly distinguishable
with a background color and bold text.

Closes part of #328"
```

---

## Task 5: Group Header Styling

Add left border to group headers to make them visually distinct from stage rows.

**Files:**
- Modify: `src/pivot/tui/styles/pivot.tcss` (add/modify `.stage-group-header` rule)

**Step 1: Locate current styling**

Find `.stage-group-header` in the TCSS.

**Step 2: Add left border styling**

Update the rule:

```css
.stage-group-header {
    /* keep existing rules */
    border-left: thick $primary;
    padding-left: 1;
}
```

**Step 3: Manual verification**

Run TUI with a pipeline that has grouped stages (stages with `@` variants).
Verify: Group headers have a visible left border that distinguishes them.

**Step 4: Run existing TUI tests**

Run: `uv run pytest tests/tui/ -v`
Expected: All pass

**Step 5: Commit**

```bash
jj describe -m "feat(tui): add left border to stage group headers

Visual distinction between group headers and stage rows.

Closes part of #328"
```

---

## Task 6: Context-Aware Footer Shortcuts

Replace Textual's default Footer with a custom widget that shows context-relevant shortcuts.

**Files:**
- Create: `src/pivot/tui/widgets/footer.py`
- Modify: `src/pivot/tui/run.py` (replace Footer with custom widget)
- Modify: `src/pivot/tui/styles/pivot.tcss` (add footer styling)
- Test: `tests/tui/widgets/test_footer.py`

### Step 6.1: Design the footer widget

The footer will show 5-6 shortcuts based on current context:
- **Stage list focused**: `j↓ k↑ / Filter Enter Toggle ? Help`
- **Logs tab active**: `Ctrl+j↓ Ctrl+k↑ L I O Tabs`
- **Input/Output tab active**: `Ctrl+j↓ Ctrl+k↑ n Next N Prev Enter Expand`

### Step 6.2: Write failing tests for footer content

```python
# tests/tui/widgets/test_footer.py

from pivot.tui.widgets.footer import PivotFooter, FooterContext


def test_footer_stage_list_context() -> None:
    """Footer shows stage list shortcuts when stage list is focused."""
    footer = PivotFooter()
    footer.set_context(FooterContext.STAGE_LIST)

    content = footer.get_shortcuts_text()
    assert "j↓" in content
    assert "k↑" in content
    assert "/" in content
    assert "?" in content


def test_footer_logs_context() -> None:
    """Footer shows logs shortcuts when logs tab is active."""
    footer = PivotFooter()
    footer.set_context(FooterContext.LOGS)

    content = footer.get_shortcuts_text()
    assert "Ctrl+j" in content
    assert "L" in content
    assert "I" in content
    assert "O" in content


def test_footer_diff_context() -> None:
    """Footer shows diff shortcuts when input/output tab is active."""
    footer = PivotFooter()
    footer.set_context(FooterContext.DIFF)

    content = footer.get_shortcuts_text()
    assert "n" in content
    assert "N" in content
    assert "Enter" in content
```

### Step 6.3: Run tests to verify they fail

Run: `uv run pytest tests/tui/widgets/test_footer.py -v`
Expected: FAIL (module doesn't exist)

### Step 6.4: Create the footer widget

```python
# src/pivot/tui/widgets/footer.py
"""Context-aware footer showing relevant keyboard shortcuts."""

from enum import Enum, auto

from textual.widgets import Static


class FooterContext(Enum):
    """Current UI context for footer shortcuts."""
    STAGE_LIST = auto()
    LOGS = auto()
    DIFF = auto()


_SHORTCUTS: dict[FooterContext, list[tuple[str, str]]] = {
    FooterContext.STAGE_LIST: [
        ("j/k", "↑↓"),
        ("/", "Filter"),
        ("Enter", "Toggle"),
        ("q", "Quit"),
        ("?", "Help"),
    ],
    FooterContext.LOGS: [
        ("Ctrl+j/k", "Scroll"),
        ("L", "Logs"),
        ("I", "Input"),
        ("O", "Output"),
        ("q", "Quit"),
        ("?", "Help"),
    ],
    FooterContext.DIFF: [
        ("Ctrl+j/k", "Scroll"),
        ("n/N", "Next/Prev"),
        ("Enter", "Expand"),
        ("L", "Logs"),
        ("q", "Quit"),
        ("?", "Help"),
    ],
}


class PivotFooter(Static):
    """Context-aware footer showing relevant keyboard shortcuts."""

    DEFAULT_CSS = """
    PivotFooter {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._context = FooterContext.STAGE_LIST

    def set_context(self, context: FooterContext) -> None:
        """Update the footer context and refresh display."""
        self._context = context
        self.update(self.get_shortcuts_text())

    def get_shortcuts_text(self) -> str:
        """Get formatted shortcuts string for current context."""
        shortcuts = _SHORTCUTS[self._context]
        parts = [f"[bold]{key}[/] {desc}" for key, desc in shortcuts]
        return "  ".join(parts)

    def on_mount(self) -> None:
        """Initialize footer content on mount."""
        self.update(self.get_shortcuts_text())
```

### Step 6.5: Run tests to verify they pass

Run: `uv run pytest tests/tui/widgets/test_footer.py -v`
Expected: PASS

### Step 6.6: Export from widgets module

Add to `src/pivot/tui/widgets/__init__.py`:

```python
from pivot.tui.widgets.footer import FooterContext, PivotFooter
```

### Step 6.7: Integrate footer into PivotApp

In `src/pivot/tui/run.py`:

1. Import the new footer:
```python
from pivot.tui.widgets import PivotFooter, FooterContext
```

2. Remove the old Footer import:
```python
# Remove: from textual.widgets import Footer
```

3. In `compose()`, replace `Footer()` with `PivotFooter()` and store reference:
```python
def compose(self) -> ComposeResult:
    # ... existing code ...
    self._footer = PivotFooter()
    yield self._footer
```

4. Add method to update footer context:
```python
def _update_footer_context(self) -> None:
    """Update footer based on current focus/tab."""
    # Check if stage list is focused
    stage_list = self.query_one("#stage-list", StageListPanel)
    if stage_list.has_focus:
        self._footer.set_context(FooterContext.STAGE_LIST)
        return

    # Check which tab is active
    tabs = self.query_one("#detail-tabs", TabbedContent)
    active_tab = tabs.active
    if active_tab == "logs":
        self._footer.set_context(FooterContext.LOGS)
    else:
        self._footer.set_context(FooterContext.DIFF)
```

5. Call `_update_footer_context()` on:
   - Focus changes (override `on_focus`)
   - Tab changes (in tab change handler)

### Step 6.8: Add footer styling to TCSS

Add to `src/pivot/tui/styles/pivot.tcss`:

```css
PivotFooter {
    dock: bottom;
    height: 1;
    background: $surface;
    padding: 0 1;
}

PivotFooter .key {
    color: $primary;
    text-style: bold;
}
```

### Step 6.9: Run full test suite

Run: `uv run pytest tests/tui/ -v`
Expected: All pass

### Step 6.10: Run quality checks

Run: `uv run ruff format . && uv run ruff check . && uv run basedpyright`
Expected: No errors

### Step 6.11: Manual verification

Run TUI and verify:
- Footer shows different shortcuts when stage list is focused
- Footer updates when switching tabs
- Shortcuts are readable and useful

### Step 6.12: Commit

```bash
jj describe -m "feat(tui): add context-aware footer shortcuts

Shows relevant keyboard shortcuts based on current context:
- Stage list: navigation, filter, toggle
- Logs tab: scroll, tab switching
- Input/Output tabs: diff navigation, expand

Closes #328"
```

---

## Final Verification

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -n auto`
Expected: All pass, coverage >= 90%

**Step 2: Run all quality checks**

Run: `uv run ruff format . && uv run ruff check . && uv run basedpyright`
Expected: No errors

**Step 3: Manual smoke test**

Run the TUI with a real pipeline and verify all 6 changes:
1. Bell sounds on completion
2. Cached stages show `↺`
3. Blocked stages show `◇`
4. Stage list is fixed width
5. Active tab has background
6. Footer shows context-aware shortcuts

**Step 4: Push**

```bash
jj git push
```

---

## Uncertainty Notes

1. **Footer focus detection**: The exact method for detecting stage list focus vs detail panel focus may need adjustment based on how Textual handles focus. May need to use `on_descendant_focus` or similar.

2. **Tab active detection**: The `tabs.active` property returns the tab ID. Need to verify the exact IDs used ("logs", "input", "output" or similar).

3. **Existing tests**: There may be existing tests that assert the old symbols (`$`, `⊘`). These will need updating in Step 2.5.

4. **TCSS specificity**: The tab styling rules may need higher specificity if Textual's default styles override them.
