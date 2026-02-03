---
title: "TUI: Fix Log Coloring (Everything Shows as Red)"
type: feat
date: 2026-02-03
issue: "#330"
deepened: 2026-02-03
reviewed: 2026-02-03
---

# TUI: Fix Log Coloring Based on Log Levels

## Review Summary

**Reviewed by:** DHH Rails Reviewer, Kieran Python Reviewer, Code Simplicity Reviewer

### Key Changes After Review
1. **Removed Enum** - Was over-engineering; just map strings directly to styles
2. **Single data structure** - Merged 3 dicts into 1 (`_LEVEL_STYLES`)
3. **Inlined detection** - No need for separate `_detect_log_level()` function
4. **~40% LOC reduction** - From ~35 lines to ~20 lines

### Reviewer Consensus
> "The enum must die" - DHH
> "Carrying 3 data structures when 1 suffices" - Simplicity Reviewer
> "The enum would be justified if you needed to pass log levels around... For 'regex match -> pick a color string,' it's pure overhead" - Simplicity Reviewer

---

## Overview

Currently all Python logging output appears red in the TUI because Python's `logging` module writes to stderr by default. This makes logs hard to read and removes meaningful visual distinction between INFO messages and actual errors.

**Fix:** Detect log level prefixes and color appropriately instead of blindly coloring all stderr as red.

## Problem Statement

```
[14:32:15] [red]INFO: Loading data from input.csv[/]     <- red (wrong)
[14:32:16] [red]WARNING: Deprecated function used[/]     <- red (wrong)
[14:32:17] [red]ERROR: ValueError: invalid value[/]      <- red (correct)
```

**Root Cause:** `_write_line()` in `src/pivot/tui/widgets/logs.py:72-75` treats all `is_stderr=True` as red. Since Python's logging handler writes everything to stderr, all log levels appear red.

## Proposed Solution

Parse log lines for level prefixes and apply semantic colors:

| Level | Color | Rich Markup |
|-------|-------|-------------|
| DEBUG | dim | `[dim]` |
| INFO | default | (none) |
| WARNING/WARN | yellow | `[yellow]` |
| ERROR | red | `[red]` |
| CRITICAL/FATAL | red bold | `[red bold]` |
| No level detected + stderr | red | `[red]` (fallback) |

### Research Insights: Color Conventions

The industry-standard convention (from coloredlogs, colorlog, structlog) follows a **cool-to-warm severity gradient**:
- DEBUG = dim/grey (low importance, diagnostic)
- INFO = default/neutral (normal operation)
- WARNING = yellow (attention needed)
- ERROR = red (problem occurred)
- CRITICAL = bold red (immediate attention)

This aligns with existing TUI patterns in `src/pivot/tui/widgets/status.py` and `src/pivot/tui/console.py`.

## Technical Approach

### Implementation Location

All changes in `src/pivot/tui/widgets/logs.py` - keep it simple, no new files.

### Log Level Detection

Add a private function to detect log levels from common formats:

**Patterns to recognize:**
```python
# Pivot's own format (from _QueueLoggingHandler)
"INFO: message"
"WARNING: message"

# Common external formats
"[INFO] message"
"INFO - message"
"2024-01-01 10:00:00 INFO message"  # timestamp prefix
```

### Research Insights: Pattern Matching

**Compiled regex is the right choice** (from performance-oracle analysis):
- Pre-compiled regex executes in **1-5 microseconds** per line
- At 10,000 lines, that's 10-50ms total - negligible vs TUI rendering
- String methods would be more complex and not significantly faster
- The actual bottleneck is `self.write()` / TUI rendering, not detection

**Order alternations by frequency** for micro-optimization:
- Put INFO before DEBUG (INFO is most common)

### Edge Cases

**Multi-line logs:** Each line is detected independently. Continuation lines without a level prefix fall back to red if from stderr. This matches user expectations for tracebacks.

**ANSI sequences:** Strip ANSI escape codes before detection. Pivot's own logging doesn't emit them, and external library colors would conflict with semantic coloring.

**Partial matches:** Regex is anchored to line start and requires a delimiter, preventing false matches like "INFORMATION" or "WARNING_FILE.txt".

**Long lines:** Limit search to first 100 characters - log levels always appear at the start.

## Implementation

### Add Constants and Update `_write_line()`

**src/pivot/tui/widgets/logs.py**

Simplified implementation based on reviewer feedback - no enum, single dict:

```python
import re

# Matches: [optional ANSI][optional timestamp][LEVEL][delimiter]
# Examples: "INFO: msg", "[DEBUG] msg", "2024-01-01 10:00:00 WARNING msg"
_LOG_LEVEL_PATTERN: re.Pattern[str] = re.compile(
    r"^(?:\x1b\[[0-9;]*m)*"           # Skip leading ANSI escape sequences
    r"(?:\[?[\d\-:.\s,TZ]+\]?\s*)?"   # Optional timestamp (various formats)
    r"(?:\[?(INFO|WARNING|WARN|ERROR|DEBUG|CRITICAL|FATAL)\]?)"  # Level
    r"[\s:\-\]]",                      # Delimiter after level
    re.IGNORECASE,
)

# Single dict: level string -> Rich style (None = default color)
_LEVEL_STYLES: dict[str, str | None] = {
    "DEBUG": "dim",
    "INFO": None,
    "WARNING": "yellow",
    "WARN": "yellow",
    "ERROR": "red",
    "CRITICAL": "red bold",
    "FATAL": "red bold",
}


def _write_line(self, line: str, is_stderr: bool, timestamp: float) -> None:
    time_str = time.strftime("[%H:%M:%S]", time.localtime(timestamp))
    escaped_line = rich.markup.escape(line)

    # Detect level and get style in one step
    style: str | None = None
    if match := _LOG_LEVEL_PATTERN.match(line):
        style = _LEVEL_STYLES.get(match.group(1).upper())
    elif is_stderr:
        style = "red"  # Fallback for unrecognized stderr

    if style:
        self.write(f"[dim]{time_str}[/] [{style}]{escaped_line}[/]")
    else:
        self.write(f"[dim]{time_str}[/] {escaped_line}")
```

**Why this is better than the original plan:**
- No enum overhead - regex match gives us a string, we need a string (style)
- Single data structure instead of 3
- Detection inlined - no separate function for 3 lines of logic
- ~40% less code
- Equally testable via the dict and regex directly

### Add Unit Tests

**tests/tui/test_log_level_detection.py**

Test the regex pattern and style mapping directly:

```python
import pytest
from pivot.tui.widgets.logs import _LOG_LEVEL_PATTERN, _LEVEL_STYLES

@pytest.mark.parametrize("line,expected_level", [
    # Pivot format (colon)
    ("INFO: Loading data", "INFO"),
    ("WARNING: Deprecated", "WARNING"),
    ("ERROR: Failed", "ERROR"),
    ("DEBUG: Internal", "DEBUG"),
    ("CRITICAL: Fatal", "CRITICAL"),

    # Bracketed format
    ("[INFO] Processing", "INFO"),
    ("[WARNING] Check this", "WARNING"),

    # Dash format
    ("INFO - message", "INFO"),
    ("ERROR - failed", "ERROR"),

    # With timestamp
    ("2024-01-01 10:00:00 INFO Loading", "INFO"),
    ("[2024-01-01T10:00:00] ERROR Failed", "ERROR"),

    # Case insensitive
    ("info: lowercase", "INFO"),
    ("Info: mixed", "INFO"),

    # Aliases
    ("WARN: short form", "WARN"),
    ("FATAL: alias", "FATAL"),
])
def test_log_level_pattern_matches(line: str, expected_level: str) -> None:
    match = _LOG_LEVEL_PATTERN.match(line)
    assert match is not None
    assert match.group(1).upper() == expected_level.upper()

@pytest.mark.parametrize("line", [
    "Just a regular line",
    "This is INFORMATION",  # not at start
    "WARNING_FILE.txt processed",  # no delimiter
    "",
    "   ",
])
def test_log_level_pattern_no_match(line: str) -> None:
    assert _LOG_LEVEL_PATTERN.match(line) is None

def test_log_level_pattern_with_ansi() -> None:
    """ANSI sequences at start should not prevent matching."""
    line = "\033[31mERROR: message\033[0m"
    match = _LOG_LEVEL_PATTERN.match(line)
    assert match is not None
    assert match.group(1).upper() == "ERROR"

def test_level_styles_complete() -> None:
    """All matched levels should have a style defined."""
    levels = ["DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL"]
    for level in levels:
        assert level in _LEVEL_STYLES, f"Missing style for {level}"

def test_level_styles_values() -> None:
    """Verify expected style values."""
    assert _LEVEL_STYLES["DEBUG"] == "dim"
    assert _LEVEL_STYLES["INFO"] is None
    assert _LEVEL_STYLES["WARNING"] == "yellow"
    assert _LEVEL_STYLES["ERROR"] == "red"
    assert _LEVEL_STYLES["CRITICAL"] == "red bold"
```

## Acceptance Criteria

- [ ] INFO-level logs appear in default color (not red)
- [ ] DEBUG logs appear dim
- [ ] WARNING/WARN logs appear yellow
- [ ] ERROR logs appear red
- [ ] CRITICAL/FATAL logs appear red bold
- [ ] Non-logging stderr output still appears red (fallback)
- [ ] Works with common formats: `LEVEL:`, `[LEVEL]`, `LEVEL -`, timestamped
- [ ] Case-insensitive level matching
- [ ] Unit tests for detection logic pass
- [ ] `uv run ruff format . && uv run ruff check . && uv run basedpyright` passes

## Files to Modify

| File | Changes |
|------|---------|
| `src/pivot/tui/widgets/logs.py` | Add `_LOG_LEVEL_PATTERN` regex, `_LEVEL_STYLES` dict, update `_write_line()` |
| `tests/tui/test_log_level_detection.py` | New file with unit tests |

## Testing

```bash
# Run unit tests
uv run pytest tests/tui/test_log_level_detection.py -v

# Run all quality checks
uv run ruff format . && uv run ruff check . && uv run basedpyright

# Manual verification
uv run pivot repro  # Check TUI output colors
```

## Performance Considerations

From performance-oracle analysis:

| Operation | Relative Cost | Notes |
|-----------|---------------|-------|
| `_detect_log_level()` | Low | ~1-5μs per call |
| `rich.markup.escape()` | Low-Medium | String scanning |
| `time.strftime()` | Low | Called every line |
| `self.write()` / TUI rendering | **High** | Actual bottleneck |

**Conclusion:** The implementation is well-optimized. If performance becomes an issue, batch writes to the TUI would have far greater impact than optimizing detection.

## Alternative Considered: Even Simpler Regex

DHH suggested an even simpler approach using `re.search()`:

```python
_LOG_LEVEL_PATTERN = re.compile(r"(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)", re.IGNORECASE)
# Used with: _LOG_LEVEL_PATTERN.search(line[:100])
```

**Trade-offs:**
- Pro: Simpler regex, finds level anywhere in first 100 chars, more robust to format variations
- Con: Higher false-positive risk (e.g., "This is INFORMATION about...")

**Decision:** Keep the anchored regex with delimiter requirement since:
1. Handles ANSI codes correctly
2. Requires delimiters to prevent false matches
3. Matches the issue's proposed solution
4. False positives would be confusing (non-log lines getting colored)

If false negatives become a problem (legitimate logs not detected), we can switch to the simpler approach.

## References

- GitHub Issue: #330
- Current implementation: `src/pivot/tui/widgets/logs.py:69-75`
- Logging handler format: `src/pivot/executor/worker.py:80` (`%(levelname)s: %(message)s`)
- Python logging docs: https://docs.python.org/3/library/logging.html
- Rich markup docs: https://rich.readthedocs.io/en/stable/markup.html
- Textual RichLog: https://textual.textualize.io/widgets/rich_log/
- coloredlogs conventions: https://coloredlogs.readthedocs.io/
