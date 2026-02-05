# tests/tui/widgets/test_logs.py
from __future__ import annotations

import collections
from typing import TYPE_CHECKING

from pivot.tui.types import LogEntry
from pivot.tui.widgets.logs import StageLogPanel

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_search_finds_matches() -> None:
    """apply_search should find case-insensitive matches."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("First line with ERROR", False, 1000.0))
    panel._raw_logs.append(LogEntry("Second line normal", False, 1001.0))
    panel._raw_logs.append(LogEntry("Third error here", False, 1002.0))

    panel.apply_search("error")

    assert panel.match_count == "1/2", "Should find 2 matches and start at first"


def test_search_empty_query_clears() -> None:
    """apply_search with empty query should clear search."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("Line with error", False, 1000.0))
    panel.apply_search("error")
    assert panel.match_count == "1/1", "Should find match before clearing"

    panel.apply_search("")

    assert panel.match_count == "", "Should clear match count on empty query"
    assert panel.is_search_active is False, "Should deactivate search on empty query"


def test_search_handles_special_regex_chars() -> None:
    """Search should handle regex special characters safely."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("Error: [timeout] in stage (test)", False, 1000.0))
    panel._raw_logs.append(LogEntry("Normal line", False, 1001.0))

    panel.apply_search("[timeout]")

    assert panel.match_count == "1/1", "Should match brackets literally, not as regex"


def test_search_case_insensitive() -> None:
    """Search should be case-insensitive."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("ERROR: something", False, 1000.0))
    panel._raw_logs.append(LogEntry("error: another", False, 1001.0))
    panel._raw_logs.append(LogEntry("Error: mixed", False, 1002.0))

    panel.apply_search("error")
    assert panel.match_count == "1/3", "Lowercase query should match all case variations"

    panel.apply_search("ERROR")
    assert panel.match_count == "1/3", "Uppercase query should match all case variations"


def test_clear_resets_search() -> None:
    """Clearing panel should reset search state."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("error line", False, 1000.0))
    panel.apply_search("error")

    panel.clear()

    assert panel.match_count == "", "Should clear match count"
    assert panel.is_search_active is False, "Should deactivate search"


def test_highlight_matches_escapes_markup() -> None:
    """Highlighting should escape Rich markup in log text."""
    panel = StageLogPanel()
    panel.apply_search("test")

    result = panel._highlight_matches("[bold]test value[/bold]")

    assert "[bold yellow]test[/]" in result, "Should highlight matching text"
    assert "\\[bold]" in result, "Should escape Rich markup to prevent injection"


def test_navigation_cycles_through_matches() -> None:
    """next_match/prev_match should cycle through matches."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("error one", False, 1000.0))
    panel._raw_logs.append(LogEntry("normal line", False, 1001.0))
    panel._raw_logs.append(LogEntry("error two", False, 1002.0))
    panel._raw_logs.append(LogEntry("error three", False, 1003.0))

    panel.apply_search("error")
    assert panel.match_count == "1/3", "Should start at first match"

    panel.next_match()
    assert panel.match_count == "2/3", "Should advance to second match"

    panel.next_match()
    assert panel.match_count == "3/3", "Should advance to third match"

    panel.next_match()
    assert panel.match_count == "1/3", "Should wrap to first match"

    panel.prev_match()
    assert panel.match_count == "3/3", "Should wrap backwards to last match"


def test_navigation_noop_when_no_matches() -> None:
    """Navigation should do nothing when no matches."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("no match here", False, 1000.0))
    panel.apply_search("xyz")

    panel.next_match()  # Should not error
    panel.prev_match()  # Should not error

    assert panel.match_count == "0/0", "Should maintain 0/0 after navigation attempts"


def test_search_on_empty_panel() -> None:
    """Search on empty panel should not error."""
    panel = StageLogPanel()

    panel.apply_search("error")

    assert panel.match_count == "0/0", "Empty panel should show 0/0 matches"
    assert panel.is_search_active is True, "Search should be active even with no logs"


def test_search_unchanged_query_is_noop() -> None:
    """Applying same search query should not trigger re-render."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("error line", False, 1000.0))
    panel.apply_search("error")
    initial_count = panel.match_count

    # Apply same query again
    panel.apply_search("error")

    assert panel.match_count == initial_count, "Match count should be unchanged"
    assert panel.is_search_active is True, "Search should remain active"


def test_multiple_matches_on_same_line() -> None:
    """Highlighting should handle multiple matches on same line."""
    panel = StageLogPanel()
    panel.apply_search("error")

    result = panel._highlight_matches("error at start, error in middle, error at end")

    assert result.count("[bold yellow]error[/]") == 3, "Should highlight all 3 occurrences"
    assert " at start, " in result, "Should preserve text between matches"


def test_navigation_after_deque_eviction() -> None:
    """Navigation should handle match index clamping after deque eviction."""
    panel = StageLogPanel()
    # Fill deque to capacity (1000)
    for i in range(1000):
        panel._raw_logs.append(LogEntry(f"log {i}", False, float(i)))

    # Add 5 matching lines at end
    for i in range(5):
        panel._raw_logs.append(LogEntry(f"ERROR {i}", False, 1000.0 + i))

    panel.apply_search("ERROR")
    # Navigate to last match
    for _ in range(4):
        panel.next_match()
    assert panel.match_count == "5/5", "Should be at last match before eviction"

    # Add more logs to trigger eviction (should evict first 5 entries)
    for i in range(10):
        panel._raw_logs.append(LogEntry(f"new log {i}", False, 2000.0 + i))

    # Match count should update correctly
    matches = panel._get_match_indices()
    assert len(matches) == 5, "Should still have 5 ERROR matches after eviction"

    # Navigation should not crash
    panel.next_match()
    assert "/" in panel.match_count, "Should still show valid match count format after eviction"


def test_search_with_no_matches() -> None:
    """Search with no matches should show 0/0."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("normal log", False, 1000.0))
    panel._raw_logs.append(LogEntry("another log", False, 1001.0))

    panel.apply_search("nonexistent")

    assert panel.match_count == "0/0", "Should show 0/0 when no matches found"
    assert panel.is_search_active is True, "Search should still be active with no matches"


def test_highlight_preserves_non_matching_text() -> None:
    """Highlighting should escape and preserve text without matches."""
    panel = StageLogPanel()
    panel.apply_search("error")

    result = panel._highlight_matches("normal [markup] text")

    assert "[bold yellow]" not in result, "Should not highlight non-matching text"
    assert "\\[markup]" in result, "Should escape Rich markup characters"


def test_match_count_clamps_index_after_match_reduction() -> None:
    """match_count should clamp current index if matches decrease."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("error 1", False, 1000.0))
    panel._raw_logs.append(LogEntry("error 2", False, 1001.0))
    panel._raw_logs.append(LogEntry("error 3", False, 1002.0))

    panel.apply_search("error")
    # Navigate to third match
    panel.next_match()
    panel.next_match()
    assert panel.match_count == "3/3", "Should be at third match initially"

    # Manually set index beyond bounds (simulating match loss)
    panel._current_match_idx = 10

    # Getting match_count should clamp
    count = panel.match_count
    assert count == "3/3", "Should clamp out-of-bounds index to valid range"


def test_navigation_updates_match_position() -> None:
    """Navigation should change current match position through the list."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("error 1", False, 1000.0))
    panel._raw_logs.append(LogEntry("normal", False, 1001.0))
    panel._raw_logs.append(LogEntry("error 2", False, 1002.0))

    panel.apply_search("error")
    assert panel.match_count == "1/2", "Should start at first match"

    panel.next_match()
    assert panel.match_count == "2/2", "Should move to second match"

    panel.prev_match()
    assert panel.match_count == "1/2", "Should move back to first match"


def test_clear_empties_raw_logs() -> None:
    """clear() should remove raw logs, not just search state."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("test log", False, 1000.0))
    panel.apply_search("test")

    panel.clear()

    assert len(panel._raw_logs) == 0, "Should clear raw logs storage"
    assert panel.is_search_active is False, "Should clear search state"
    assert panel.match_count == "", "Should clear match count"


def test_highlight_match_at_line_end() -> None:
    """Highlighting should handle match at end of line correctly."""
    panel = StageLogPanel()
    panel.apply_search("error")

    result = panel._highlight_matches("This line ends with error")

    assert result.endswith("[bold yellow]error[/]"), "Should highlight match at end of line"
    assert "This line ends with " in result, "Should preserve text before match"


# =============================================================================
# Task 4: Live log update tests (performance-critical)
# =============================================================================


def test_add_log_during_search_appends_without_rerender(mocker: MockerFixture) -> None:
    """Adding a log during active search should NOT trigger full re-render."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("error one", False, 1000.0))
    panel.apply_search("error")
    assert panel.match_count == "1/1", "Should have one match before adding"

    rerender_spy = mocker.patch.object(panel, "_rerender_logs", wraps=panel._rerender_logs)

    # Add new log - should NOT trigger re-render
    panel.add_log("error two", is_stderr=False, timestamp=1001.0)

    rerender_spy.assert_not_called()
    # match_count updates (computed on demand)
    assert panel.match_count == "1/2", "Should update match count without re-render"


def test_add_log_handles_deque_eviction() -> None:
    """When deque evicts oldest line, match index clamps automatically."""
    panel = StageLogPanel()
    panel._raw_logs = collections.deque[LogEntry](maxlen=3)

    panel._raw_logs.append(LogEntry("error one", False, 1000.0))
    panel._raw_logs.append(LogEntry("normal", False, 1001.0))
    panel._raw_logs.append(LogEntry("error two", False, 1002.0))

    panel.apply_search("error")
    assert panel.match_count == "1/2", "Should have 2 matches before eviction"

    # Add a 4th line - evicts "error one"
    panel.add_log("error three", is_stderr=False, timestamp=1003.0)

    # Now: normal, error two, error three
    # Matches at indices 1, 2
    # _current_match_idx=0 clamps to valid range in match_count
    assert panel.match_count == "1/2", "Should still show 1/2 after eviction"


def test_search_handles_unicode() -> None:
    """Search should handle unicode characters correctly."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("Error: 日本語 test データ", False, 1000.0))
    panel._raw_logs.append(LogEntry("Normal: αβγδ symbols", False, 1001.0))
    panel._raw_logs.append(LogEntry("Plain text line", False, 1002.0))

    panel.apply_search("日本語")
    assert panel.match_count == "1/1", "Should find unicode match"

    result = panel._highlight_matches("Error: 日本語 test")
    assert "[bold yellow]日本語[/]" in result, "Should highlight unicode text"
    assert "Error: " in result, "Should preserve ASCII text before unicode"


def test_search_empty_string_clears_completely() -> None:
    """Search with empty string after having pattern should clear completely."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("error line", False, 1000.0))
    panel.apply_search("error")

    # Verify search is active
    assert panel.is_search_active is True, "Should be active before clearing"

    # Clear with empty string
    panel.apply_search("")

    # Verify complete reset via public interface
    assert panel.is_search_active is False, "Should be inactive after clearing"
    assert panel.match_count == "", "Match count should be empty"

    # Can re-apply search successfully
    panel.apply_search("line")
    assert panel.match_count == "1/1", "Should work after clearing"


def test_search_whitespace_only_query_clears() -> None:
    """Search with whitespace-only query should be treated as empty (clear search)."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("error line", False, 1000.0))
    panel.apply_search("error")
    assert panel.is_search_active is True, "Should be active before whitespace query"

    # Whitespace-only query should clear search
    panel.apply_search("   ")

    assert panel.is_search_active is False, "Should be inactive after whitespace query"
    assert panel.match_count == "", "Match count should be empty"


def test_navigation_clamps_out_of_bounds_index_before_navigating() -> None:
    """Navigation should clamp out-of-bounds index before computing next position."""
    panel = StageLogPanel()
    panel._raw_logs.append(LogEntry("error 1", False, 1000.0))
    panel._raw_logs.append(LogEntry("error 2", False, 1001.0))
    panel._raw_logs.append(LogEntry("error 3", False, 1002.0))

    panel.apply_search("error")
    # Manually set index beyond bounds (simulating deque eviction scenario)
    panel._current_match_idx = 10

    # next_match should clamp first, then navigate
    # Clamp: min(10, 2) = 2, then (2 + 1) % 3 = 0
    panel.next_match()
    assert panel.match_count == "1/3", "Should clamp then wrap to first match"

    # Same for prev_match
    panel._current_match_idx = 10
    # Clamp: min(10, 2) = 2, then (2 - 1) % 3 = 1
    panel.prev_match()
    assert panel.match_count == "2/3", "Should clamp then go to second match"
