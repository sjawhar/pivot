# TUI Log Search Implementation Plan (Simplified)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add vim-style search to the Logs tab with incremental highlighting, match navigation, and match counter.

**Architecture:** Add search state directly to `StageLogPanel` (no wrapper class). Search bar managed by `TabbedDetailPanel`. Performance-optimized: only re-render on query change, not on every new log.

**Tech Stack:** Textual widgets, Rich markup for highlighting, cached regex for matching.

---

## Simplification Summary (Post-Review)

Three parallel reviewers (DHH, Kieran, Simplicity) identified over-engineering in the deepened plan. Key simplifications:

| Original Enhancement | Simplified Approach | Rationale |
|---------------------|---------------------|-----------|
| New `LogEntry` TypedDict with seq_id | Use existing `LogEntry` NamedTuple from `types.py` | Avoid type collision, reuse existing type |
| Sequence IDs for stable tracking | Simple index with clamping on eviction | Solves a rare edge case with 3 lines of code |
| Cached match indices | Compute on demand | 150μs search doesn't need caching |
| `navigate_log_search()` wrapper | Access `_log_panel` directly | Unnecessary indirection |
| `_navigate_match(direction)` | Separate `next_match()`/`prev_match()` | Clarity over DRY |
| Two-phase Escape (clear then close) | Single Escape closes bar | Simpler UX |
| **Input debouncing** | **Keep 200ms debounce** | Prevents UI stutter during fast typing |

### What We're Keeping

1. **Debouncing** - 200ms delay prevents re-render stutter during fast typing
2. **E2E test** - Required per `critical-patterns.md`, but simplified
3. **Rich markup escaping** - Security best practice

---

## Design Decisions (Post-Review)

Based on feedback from DHH, Kieran, simplicity, and performance reviews:

| Original Plan | Revised Approach | Rationale |
|---------------|------------------|-----------|
| `SearchableLogPanel` wrapper | Search bar in `TabbedDetailPanel` | Eliminates unnecessary indirection, follows `FilterInput` pattern |
| Re-render all logs on every new log | Append with highlighting, re-render only on query change | 100x performance improvement |
| Store `_match_indices` list | Compute on-demand + handle deque eviction | Avoids stale indices bug |
| New `LogSearchInput` class | Reuse existing `FilterInput` pattern | DRY |
| Test internal state directly | Test through public interfaces | Tests behavior, not implementation |
| Overload `/` with context | Separate `ctrl+f` for log search | Simpler, single-responsibility |

### Performance Constraints

- Search 1000 lines: ~150μs (fast)
- Regex highlighting: ~10ms (fast)
- Full re-render (clear + 1000x write): **50-150ms** (slow - avoid in hot path)
- At 10 logs/sec with re-render = frozen UI

**Rule:** Only call `_rerender_logs()` when:
1. Search query changes (user typed)
2. User navigates with n/N
3. Search is cleared

**Never** call `_rerender_logs()` when a new log arrives.

---

## Task 1: Add Raw Log Storage and Search State

**Simplification:** Use existing `LogEntry` NamedTuple from `types.py`. No sequence IDs - use simple indices with clamping. No match caching - 150μs is fast enough.

**Files:**
- Modify: `src/pivot/tui/widgets/logs.py`
- Test: `tests/tui/widgets/test_logs.py` (new file)

**Step 1: Write the failing test**

```python
# tests/tui/widgets/test_logs.py
from __future__ import annotations

from pivot.tui.widgets.logs import StageLogPanel
from pivot.tui.types import LogEntry


def test_search_finds_matches() -> None:
    """apply_search should find case-insensitive matches."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("First line with ERROR", False, 1000.0))
    panel._raw_logs.append(LogEntry("Second line normal", False, 1001.0))
    panel._raw_logs.append(LogEntry("Third error here", False, 1002.0))

    panel.apply_search("error")

    assert panel.match_count == "1/2"


def test_search_empty_query_clears() -> None:
    """apply_search with empty query should clear search."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("Line with error", False, 1000.0))
    panel.apply_search("error")
    assert panel.match_count == "1/1"

    panel.apply_search("")

    assert panel.match_count == ""
    assert panel.is_search_active is False


def test_search_handles_special_regex_chars() -> None:
    """Search should handle regex special characters safely."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("Error: [timeout] in stage (test)", False, 1000.0))
    panel._raw_logs.append(LogEntry("Normal line", False, 1001.0))

    panel.apply_search("[timeout]")

    assert panel.match_count == "1/1"


def test_search_case_insensitive() -> None:
    """Search should be case-insensitive."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("ERROR: something", False, 1000.0))
    panel._raw_logs.append(LogEntry("error: another", False, 1001.0))
    panel._raw_logs.append(LogEntry("Error: mixed", False, 1002.0))

    panel.apply_search("error")
    assert panel.match_count == "1/3"

    panel.apply_search("ERROR")
    assert panel.match_count == "1/3"


def test_clear_resets_search() -> None:
    """Clearing panel should reset search state."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("error line", False, 1000.0))
    panel.apply_search("error")

    panel.clear()

    assert panel.match_count == ""
    assert panel.is_search_active is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tui/widgets/test_logs.py -v`
Expected: FAIL with AttributeError

**Step 3: Add storage and search state**

```python
# In src/pivot/tui/widgets/logs.py

# Add imports at top:
import collections
import re
from typing import override

from pivot.tui.types import LogEntry

# Add to StageLogPanel class - type annotations:
_raw_logs: collections.deque[LogEntry]
_search_query: str
_search_pattern: re.Pattern[str] | None  # Cached compiled pattern
_current_match_idx: int  # Index into match list (not raw_logs)

# In __init__, after self._pending_stage = None:
self._raw_logs = collections.deque[LogEntry](maxlen=1000)
self._search_query = ""
self._search_pattern = None
self._current_match_idx = 0

# Add public properties:
@property
def is_search_active(self) -> bool:
    """Whether search mode is active."""
    return bool(self._search_query)

@property
def match_count(self) -> str:
    """Return 'current/total' format, or empty if no search."""
    if not self._search_query:
        return ""
    matches = self._get_match_indices()
    if not matches:
        return "0/0"
    # Clamp index if matches changed (e.g., deque eviction)
    idx = min(self._current_match_idx, len(matches) - 1)
    return f"{idx + 1}/{len(matches)}"

def _get_match_indices(self) -> list[int]:
    """Get indices of matching lines in _raw_logs (computed fresh each time)."""
    if not self._search_query:
        return []
    query_lower = self._search_query.lower()
    return [i for i, entry in enumerate(self._raw_logs) if query_lower in entry.line.lower()]

# Override clear():
@override
def clear(self) -> "StageLogPanel":
    """Clear the log display, raw storage, and search state."""
    self._raw_logs.clear()
    self._search_query = ""
    self._search_pattern = None
    self._current_match_idx = 0
    return super().clear()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tui/widgets/test_logs.py -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(tui): add raw log storage and search state to StageLogPanel"
```

---

## Task 2: Implement Search Logic with Cached Regex

**Files:**
- Modify: `src/pivot/tui/widgets/logs.py`
- Test: `tests/tui/widgets/test_logs.py`

**Step 1: Write the failing test**

```python
# Add to tests/tui/widgets/test_logs.py

def test_highlight_matches_escapes_markup() -> None:
    """Highlighting should escape Rich markup in log text."""
    panel = StageLogPanel()
    panel.apply_search("test")

    result = panel._highlight_matches("[bold]test value[/bold]")

    # Should escape brackets AND highlight "test"
    assert "\\[bold\\]" in result or "bold" not in result.replace("[bold yellow]", "")
    assert "[bold yellow]test[/]" in result
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tui/widgets/test_logs.py::test_highlight_matches_escapes_markup -v`
Expected: FAIL with AttributeError

**Step 3: Implement apply_search with cached regex**

```python
# Add to StageLogPanel class in src/pivot/tui/widgets/logs.py

def apply_search(self, query: str) -> None:
    """Search logs for query and re-render with highlights.

    Args:
        query: Search query (case-insensitive). Empty string clears search.
    """
    if query == self._search_query:
        return  # No change

    self._search_query = query

    if query:
        # Cache compiled pattern for highlighting
        self._search_pattern = re.compile(re.escape(query), re.IGNORECASE)
        self._current_match_idx = 0  # Start at first match
    else:
        self._search_pattern = None
        self._current_match_idx = 0

    self._rerender_logs()

def _rerender_logs(self) -> None:  # pragma: no cover
    """Re-render all logs, applying search highlighting if active."""
    super().clear()  # Clear display, preserve _raw_logs

    matches = self._get_match_indices()
    # Clamp index if needed
    if matches:
        self._current_match_idx = min(self._current_match_idx, len(matches) - 1)
        current_line_idx = matches[self._current_match_idx]
    else:
        current_line_idx = -1

    for i, entry in enumerate(self._raw_logs):
        is_current = (i == current_line_idx)
        self._write_line_highlighted(entry, is_current)

    # Scroll to current match
    if current_line_idx >= 0:
        self.scroll_to(y=current_line_idx, animate=False)

    self.refresh()

def _write_line_highlighted(self, entry: LogEntry, is_current: bool = False) -> None:  # pragma: no cover
    """Write a log line with optional search highlighting."""
    time_str = time.strftime("[%H:%M:%S]", time.localtime(entry.timestamp))

    if self._search_query and self._search_pattern:
        highlighted = self._highlight_matches(entry.line)

        if entry.is_stderr:
            if is_current:
                self.write(f"[dim]{time_str}[/] [on yellow][red]{highlighted}[/][/]")
            else:
                self.write(f"[dim]{time_str}[/] [red]{highlighted}[/]")
        else:
            if is_current:
                self.write(f"[dim]{time_str}[/] [on dark_blue]{highlighted}[/]")
            else:
                self.write(f"[dim]{time_str}[/] {highlighted}")
    else:
        # No search - use original formatting
        escaped_line = rich.markup.escape(entry.line)
        if entry.is_stderr:
            self.write(f"[dim]{time_str}[/] [red]{escaped_line}[/]")
        else:
            self.write(f"[dim]{time_str}[/] {escaped_line}")

def _highlight_matches(self, line: str) -> str:
    """Highlight search matches in a line with Rich markup.

    Security: Uses rich.markup.escape() to prevent markup injection.
    """
    if not self._search_pattern:
        return rich.markup.escape(line)

    parts = list[str]()
    last_end = 0

    for match in self._search_pattern.finditer(line):
        # Escape text before match
        if match.start() > last_end:
            parts.append(rich.markup.escape(line[last_end:match.start()]))
        # Highlight the match
        parts.append(f"[bold yellow]{rich.markup.escape(match.group())}[/]")
        last_end = match.end()

    # Escape remaining text
    if last_end < len(line):
        parts.append(rich.markup.escape(line[last_end:]))

    return "".join(parts) if parts else rich.markup.escape(line)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tui/widgets/test_logs.py -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(tui): implement log search with cached regex and highlighting"
```

---

## Task 3: Implement Match Navigation

**Simplification:** Keep `next_match()` and `prev_match()` as separate methods. Clarity over DRY.

**Files:**
- Modify: `src/pivot/tui/widgets/logs.py`
- Test: `tests/tui/widgets/test_logs.py`

**Step 1: Write the failing test**

```python
# Add to tests/tui/widgets/test_logs.py

def test_navigation_cycles_through_matches() -> None:
    """next_match/prev_match should cycle through matches."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("match one", False, 1000.0))
    panel._raw_logs.append(LogEntry("no match", False, 1001.0))
    panel._raw_logs.append(LogEntry("match two", False, 1002.0))
    panel._raw_logs.append(LogEntry("match three", False, 1003.0))

    panel.apply_search("match")
    assert panel.match_count == "1/3"

    panel.next_match()
    assert panel.match_count == "2/3"

    panel.next_match()
    assert panel.match_count == "3/3"

    panel.next_match()  # Wrap to start
    assert panel.match_count == "1/3"

    panel.prev_match()  # Wrap to end
    assert panel.match_count == "3/3"


def test_navigation_noop_when_no_matches() -> None:
    """Navigation should do nothing when no matches."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("no match here", False, 1000.0))
    panel.apply_search("xyz")

    panel.next_match()  # Should not error
    panel.prev_match()  # Should not error

    assert panel.match_count == "0/0"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tui/widgets/test_logs.py::test_navigation_cycles_through_matches -v`
Expected: FAIL with AttributeError

**Step 3: Implement separate navigation methods**

```python
# Add to StageLogPanel class in src/pivot/tui/widgets/logs.py

def next_match(self) -> None:  # pragma: no cover
    """Move to the next search match (wraps around)."""
    matches = self._get_match_indices()
    if not matches:
        return
    self._current_match_idx = (self._current_match_idx + 1) % len(matches)
    self._rerender_logs()

def prev_match(self) -> None:  # pragma: no cover
    """Move to the previous search match (wraps around)."""
    matches = self._get_match_indices()
    if not matches:
        return
    self._current_match_idx = (self._current_match_idx - 1) % len(matches)
    self._rerender_logs()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tui/widgets/test_logs.py -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(tui): add match navigation to log search"
```

---

## Task 4: Handle Live Log Updates (Performance-Critical)

**Simplification:** No seq_ids needed. On deque eviction, `_current_match_idx` naturally clamps via `match_count` property. Just append new logs with highlighting.

**Files:**
- Modify: `src/pivot/tui/widgets/logs.py`
- Test: `tests/tui/widgets/test_logs.py`

**Step 1: Write the failing test**

```python
# Add to tests/tui/widgets/test_logs.py
import collections

def test_add_log_during_search_appends_without_rerender(mocker: MockerFixture) -> None:
    """Adding a log during active search should NOT trigger full re-render."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("error one", False, 1000.0))
    panel.apply_search("error")
    assert panel.match_count == "1/1"

    rerender_spy = mocker.patch.object(panel, '_rerender_logs', wraps=panel._rerender_logs)

    # Add new log - should NOT trigger re-render
    panel.add_log("error two", is_stderr=False, timestamp=1001.0)

    rerender_spy.assert_not_called()
    # match_count updates (computed on demand)
    assert panel.match_count == "1/2"


def test_add_log_handles_deque_eviction() -> None:
    """When deque evicts oldest line, match index clamps automatically."""
    panel = StageLogPanel()
    panel._raw_logs = collections.deque[LogEntry](maxlen=3)

    panel._raw_logs.append(LogEntry("error one", False, 1000.0))
    panel._raw_logs.append(LogEntry("normal", False, 1001.0))
    panel._raw_logs.append(LogEntry("error two", False, 1002.0))

    panel.apply_search("error")
    assert panel.match_count == "1/2"

    # Add a 4th line - evicts "error one"
    panel.add_log("error three", is_stderr=False, timestamp=1003.0)

    # Now: normal, error two, error three
    # Matches at indices 1, 2
    # _current_match_idx=0 clamps to valid range in match_count
    assert panel.match_count == "1/2"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tui/widgets/test_logs.py::test_add_log_during_search_appends_without_rerender -v`
Expected: FAIL

**Step 3: Implement optimized add_log**

```python
# In src/pivot/tui/widgets/logs.py

def _render_line(self, entry: LogEntry) -> None:  # pragma: no cover
    """Render a log line to the display (no storage)."""
    time_str = time.strftime("[%H:%M:%S]", time.localtime(entry.timestamp))
    escaped_line = rich.markup.escape(entry.line)
    if entry.is_stderr:
        self.write(f"[dim]{time_str}[/] [red]{escaped_line}[/]")
    else:
        self.write(f"[dim]{time_str}[/] {escaped_line}")

def add_log(self, line: str, is_stderr: bool, timestamp: float) -> None:  # pragma: no cover
    """Add a new log line.

    PERFORMANCE: Does NOT re-render during active search.
    Just appends with correct highlighting.
    """
    entry = LogEntry(line, is_stderr, timestamp)
    self._raw_logs.append(entry)

    # Write to display with appropriate formatting
    if self._search_query:
        # Check if this new line matches the search
        is_match = self._search_query.lower() in line.lower()
        self._write_line_highlighted(entry, is_current=False)
    else:
        self._render_line(entry)

    self.refresh()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tui/widgets/test_logs.py -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "perf(tui): optimize add_log to avoid re-render during search"
```

---

## Task 5: Add Search Input to TabbedDetailPanel with Debouncing

**Simplification:** No `navigate_log_search()` wrapper - access `_log_panel` directly. Single Escape closes bar (no two-phase).

**Files:**
- Modify: `src/pivot/tui/widgets/logs.py` (add message class)
- Modify: `src/pivot/tui/widgets/panels.py`
- Modify: `src/pivot/tui/styles/pivot.tcss`

**Step 1: Add the escape message class to logs.py**

```python
# Add at top of src/pivot/tui/widgets/logs.py, after imports:
from typing import ClassVar

import textual.binding
import textual.message

class LogSearchEscapePressed(textual.message.Message):
    """Posted when Escape is pressed in the log search input."""


class LogSearchInput(textual.widgets.Input):
    """Search input with Escape key handling."""

    BINDINGS: ClassVar[list[textual.binding.BindingType]] = [
        textual.binding.Binding("escape", "escape_pressed", "Cancel", show=False),
    ]

    def action_escape_pressed(self) -> None:
        self.post_message(LogSearchEscapePressed())
```

**Step 2: Add search bar to TabbedDetailPanel with debouncing**

```python
# In src/pivot/tui/widgets/panels.py

# Add imports:
from pivot.tui.widgets.logs import LogSearchEscapePressed, LogSearchInput, StageLogPanel

# Add to TabbedDetailPanel class - new attributes:
_search_input: LogSearchInput | None
_search_container: textual.containers.Horizontal | None
_search_debounce_timer: textual.timer.Timer | None

# In __init__, add:
self._search_input = None
self._search_container = None
self._search_debounce_timer = None

# Modify compose() - add search bar inside the Logs TabPane:
@override
def compose(self) -> textual.app.ComposeResult:  # pragma: no cover
    yield textual.widgets.Static(id="detail-header")
    with textual.widgets.TabbedContent(id="detail-tabs"):
        with textual.widgets.TabPane("Logs", id="tab-logs"):
            self._log_panel = StageLogPanel(id="stage-logs")
            yield self._log_panel
            # Search bar (hidden by default)
            self._search_input = LogSearchInput(
                placeholder="Search...",
                id="log-search-input",
            )
            self._search_container = textual.containers.Horizontal(
                textual.widgets.Static("/ ", id="search-label"),
                self._search_input,
                textual.widgets.Static("", id="search-count"),
                id="log-search-bar",
                classes="hidden",
            )
            yield self._search_container
        with textual.widgets.TabPane("Input", id="tab-input"):
            yield InputDiffPanel(id="input-panel")
        with textual.widgets.TabPane("Output", id="tab-output"):
            yield OutputDiffPanel(id="output-panel")

# Add methods:
def show_log_search(self) -> None:  # pragma: no cover
    """Show the log search bar and focus input."""
    if self._search_container and self._search_input:
        self._search_container.remove_class("hidden")
        self._search_input.focus()

def hide_log_search(self) -> None:  # pragma: no cover
    """Hide the log search bar and clear search."""
    if self._search_debounce_timer:
        self._search_debounce_timer.stop()
        self._search_debounce_timer = None
    if self._search_container and self._search_input and self._log_panel:
        self._search_container.add_class("hidden")
        self._search_input.value = ""
        self._log_panel.apply_search("")

def _update_search_count(self) -> None:  # pragma: no cover
    """Update the search match count display."""
    try:
        count_widget = self.query_one("#search-count", textual.widgets.Static)
        if self._log_panel:
            match_count = self._log_panel.match_count
            count_widget.update(f" {match_count}" if match_count else "")
    except textual.css.query.NoMatches:
        pass

def _apply_debounced_search(self) -> None:  # pragma: no cover
    """Apply the search after debounce delay."""
    if self._search_input and self._log_panel:
        self._log_panel.apply_search(self._search_input.value)
        self._update_search_count()

# Event handlers:
def on_input_changed(self, event: textual.widgets.Input.Changed) -> None:  # pragma: no cover
    """Handle search input changes with 200ms debounce."""
    if event.input.id == "log-search-input":
        if self._search_debounce_timer:
            self._search_debounce_timer.stop()
        self._search_debounce_timer = self.set_timer(0.2, self._apply_debounced_search)

def on_input_submitted(self, event: textual.widgets.Input.Submitted) -> None:  # pragma: no cover
    """Handle Enter in search: apply immediately and return focus to logs."""
    if event.input.id == "log-search-input" and self._log_panel:
        if self._search_debounce_timer:
            self._search_debounce_timer.stop()
            self._search_debounce_timer = None
        self._log_panel.apply_search(event.value)
        self._update_search_count()
        self._log_panel.focus()

def on_log_search_escape_pressed(self, event: LogSearchEscapePressed) -> None:  # pragma: no cover
    """Handle Escape: close search bar."""
    event.stop()
    self.hide_log_search()
    if self._log_panel:
        self._log_panel.focus()

# Modify set_stage to reset search:
def set_stage(self, stage: StageInfo | None) -> None:  # pragma: no cover
    """Update the displayed stage."""
    self._stage = stage
    self._history_index = None
    self._history_total = len(stage.history) if stage else 0

    self._update_header()
    self.hide_log_search()  # Reset search on stage change

    if self._log_panel:
        self._log_panel.set_stage(stage)
    # ... rest of existing implementation
```

**Step 3: Add CSS styles**

```css
/* Add to src/pivot/tui/styles/pivot.tcss */

/* Log search bar */
#log-search-bar {
    height: 1;
    background: $surface-lighten-1;
    padding: 0 1;
    dock: bottom;
}

#log-search-bar.hidden {
    display: none;
}

#log-search-input {
    width: 1fr;
    height: 1;
    border: none;
    background: $surface;
}

#search-label {
    width: auto;
    color: $text-muted;
}

#search-count {
    width: auto;
    color: $text-muted;
}
```

**Step 4: Run tests**

Run: `uv run pytest tests/tui/ -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(tui): add search bar to TabbedDetailPanel with debouncing"
```

---

## Task 6: Add Keybindings

**Simplification:** Access `_log_panel` directly - it's the same codebase, unnecessary wrapper removed.

**Files:**
- Modify: `src/pivot/tui/run.py`
- Modify: `src/pivot/tui/widgets/footer.py`

**Step 1: Add keybindings to run.py**

```python
# In _TUI_BINDINGS list, add:
textual.binding.Binding("ctrl+f", "log_search", "Search Logs", show=False),
```

**Step 2: Add action method**

```python
# Add to PivotApp class:

def action_log_search(self) -> None:  # pragma: no cover
    """Activate log search (Logs tab only)."""
    tabs = self._try_query_one("#detail-tabs", textual.widgets.TabbedContent)
    if tabs is None or tabs.active != "tab-logs":
        return

    detail_panel = self._try_query_one("#detail-panel", TabbedDetailPanel)
    if detail_panel:
        detail_panel.show_log_search()
```

**Step 3: Add n/N handlers**

```python
# Modify action_next_changed in PivotApp:

def action_next_changed(self) -> None:  # pragma: no cover
    """Navigate to next changed item or search match."""
    tabs = self._try_query_one("#detail-tabs", textual.widgets.TabbedContent)

    if tabs and tabs.active == "tab-logs":
        detail_panel = self._try_query_one("#detail-panel", TabbedDetailPanel)
        if detail_panel and detail_panel._log_panel and detail_panel._log_panel.is_search_active:
            detail_panel._log_panel.next_match()
            detail_panel._update_search_count()
            return

    # Default: navigate diff changes
    panel = self._get_active_diff_panel()
    if panel:
        panel.select_next_changed()

def action_prev_changed(self) -> None:  # pragma: no cover
    """Navigate to previous changed item or search match."""
    tabs = self._try_query_one("#detail-tabs", textual.widgets.TabbedContent)

    if tabs and tabs.active == "tab-logs":
        detail_panel = self._try_query_one("#detail-panel", TabbedDetailPanel)
        if detail_panel and detail_panel._log_panel and detail_panel._log_panel.is_search_active:
            detail_panel._log_panel.prev_match()
            detail_panel._update_search_count()
            return

    # Default: navigate diff changes
    panel = self._get_active_diff_panel()
    if panel:
        panel.select_prev_changed()
```

**Step 4: Update footer shortcuts**

```python
# In src/pivot/tui/widgets/footer.py, update FooterContext.LOGS:
FooterContext.LOGS: [
    ("Ctrl+f", "Search"),
    ("n/N", "Next/Prev"),
    ("Ctrl+j/k", "Scroll"),
    ("Esc", "Close"),
    ("q", "Quit"),
    ("?", "Help"),
],
```

**Step 5: Run tests**

Run: `uv run pytest tests/tui/ -v`
Expected: PASS

**Step 6: Commit**

```bash
jj describe -m "feat(tui): add keybindings for log search"
```

---

## Task 7: Add E2E Smoke Test Using Textual Pilot API

**Simplification:** One focused E2E test. Use explicit sleep for debounce. Verify through UI elements, not private state.

**Files:**
- Test: `tests/tui/test_log_search_e2e.py` (new file)

**Step 1: Write the E2E smoke test**

```python
# tests/tui/test_log_search_e2e.py
"""End-to-end smoke test for log search using Textual's pilot API."""
from __future__ import annotations

import asyncio

import pytest

from pivot.tui.run import PivotApp
from pivot.tui.types import LogEntry


@pytest.fixture
def app() -> PivotApp:
    """Create a PivotApp instance for testing."""
    return PivotApp()


@pytest.mark.asyncio
async def test_log_search_smoke(app: PivotApp) -> None:
    """E2E: Verify search bar opens, accepts input, and closes."""
    async with app.run_test() as pilot:
        # Switch to Logs tab
        await pilot.press("L")

        # Add logs via the log panel (need some content to search)
        detail_panel = app.query_one("#detail-panel")
        # Access through public add_log method on the log panel widget
        log_panel = detail_panel.query_one("#stage-logs")
        log_panel.add_log("First ERROR message", is_stderr=False, timestamp=1000.0)
        log_panel.add_log("Normal log line", is_stderr=False, timestamp=1001.0)
        log_panel.add_log("Second error here", is_stderr=False, timestamp=1002.0)

        # Open search with Ctrl+F
        await pilot.press("ctrl+f")

        # Verify search bar is visible
        search_bar = app.query_one("#log-search-bar")
        assert "hidden" not in search_bar.classes

        # Type search query
        await pilot.press("e", "r", "r", "o", "r")
        await asyncio.sleep(0.3)  # Wait for 200ms debounce + buffer

        # Verify match count is displayed (check through UI widget)
        count_widget = app.query_one("#search-count")
        assert "1/2" in str(count_widget.renderable)

        # Navigate with n
        await pilot.press("n")
        assert "2/2" in str(count_widget.renderable)

        # Close with Escape
        await pilot.press("escape")
        assert "hidden" in search_bar.classes
```

**Step 2: Run the E2E test**

Run: `uv run pytest tests/tui/test_log_search_e2e.py -v`
Expected: PASS

**Step 3: Commit**

```bash
jj describe -m "test(tui): add E2E smoke test for log search"
```

---

## Task 8: Run Full Quality Checks

**Step 1: Format and lint**

```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright
```

Expected: No errors

**Step 2: Run all tests**

```bash
uv run pytest tests/ -n auto
```

Expected: All tests pass, coverage ≥90%

**Step 3: Manual testing**

1. Start TUI: `uv run pivot repro --watch`
2. Navigate to Logs tab
3. Press `Ctrl+f` - search bar should appear
4. Type a search term - matches should highlight
5. Press `n`/`N` - should navigate between matches
6. Press `Escape` - should clear search
7. Press `Escape` again - should close search bar
8. Run a stage that produces logs - logs should appear with highlighting
9. Switch stages - search should reset

**Step 4: Final commit**

```bash
jj describe -m "feat(tui): add log search functionality (#329)

Add search to the Logs tab:
- Ctrl+f to enter search mode
- Incremental search with case-insensitive matching
- Matches highlighted in yellow, current match has blue background
- n/N to navigate between matches
- Match counter shows current/total
- Escape clears search, then closes search bar

Performance optimized:
- Only re-render on query change or navigation
- New logs append with highlighting (no full re-render)
- Properly handles deque eviction

Search state resets when switching stages or viewing history."
```

---

## Summary

### Files Modified

| File | Changes |
|------|---------|
| `src/pivot/tui/widgets/logs.py` | Add search state, `apply_search()`, `next_match()`, `prev_match()`, highlighting |
| `src/pivot/tui/widgets/panels.py` | Add search bar with 200ms debouncing |
| `src/pivot/tui/run.py` | Add `ctrl+f` binding, modify n/N for search context |
| `src/pivot/tui/widgets/footer.py` | Update shortcuts |
| `src/pivot/tui/styles/pivot.tcss` | Search bar CSS |
| `tests/tui/widgets/test_logs.py` | Unit tests |
| `tests/tui/test_log_search_e2e.py` | E2E smoke test using Textual pilot API |

### Simplifications Applied

| Removed | Why |
|---------|-----|
| New `LogEntry` TypedDict | Use existing `LogEntry` NamedTuple from `types.py` |
| Sequence IDs | Simple index with clamping handles deque eviction |
| Match caching | 150μs search doesn't need caching |
| `navigate_log_search()` wrapper | Direct `_log_panel` access is clearer |
| `_navigate_match(direction)` | Separate `next_match()`/`prev_match()` is clearer |
| Two-phase Escape | Single Escape closes bar (simpler UX) |

### What We Kept

1. **Input debouncing** - 200ms delay prevents UI stutter during fast typing
2. **E2E test** - Required per `critical-patterns.md`
3. **Rich markup escaping** - Security best practice

### Performance Characteristics

| Operation | Cost | When |
|-----------|------|------|
| New log during search | <1ms | Every log |
| Query change | 50-150ms | User typing (debounced 200ms) |
| n/N navigation | 50-150ms | User action |
| Search 1000 lines | ~150μs | Computed fresh each time |

### Estimated Code Size

~200 lines of implementation code (down from ~350 in the deepened plan)
