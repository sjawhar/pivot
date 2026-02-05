# tests/tui/test_log_search_e2e.py
"""End-to-end smoke test for log search using Textual's pilot API."""

from __future__ import annotations

import asyncio

import pytest

from pivot.tui.run import PivotApp
from pivot.tui.widgets.logs import StageLogPanel
from pivot.tui.widgets.panels import TabbedDetailPanel


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=2)
async def test_log_search_smoke() -> None:
    """E2E: Verify search bar opens, accepts input, and closes."""
    # Use empty stage_names to avoid triggering pipeline lookups
    app = PivotApp(stage_names=[], watch_mode=True)

    async with app.run_test() as pilot:
        await pilot.pause()  # Let app initialize

        # Get the log panel directly (no stage selection needed)
        detail_panel = app.query_one("#detail-panel", TabbedDetailPanel)
        log_panel = detail_panel.query_one("#stage-logs", StageLogPanel)

        # Add logs directly to the panel
        log_panel.add_log("First ERROR message", is_stderr=False, timestamp=1000.0)
        log_panel.add_log("Normal log line", is_stderr=False, timestamp=1001.0)
        log_panel.add_log("Second error here", is_stderr=False, timestamp=1002.0)

        # Switch to Logs tab (should already be default, but ensure it)
        await pilot.press("L")
        await pilot.pause()

        # Open search with Ctrl+F
        await pilot.press("ctrl+f")
        await pilot.pause()

        # Verify search bar is visible
        search_bar = app.query_one("#log-search-bar")
        assert "hidden" not in search_bar.classes, "Search bar should be visible after Ctrl+F"

        # Type search query
        await pilot.press("e", "r", "r", "o", "r")
        await asyncio.sleep(0.3)  # Wait for 200ms debounce + buffer

        # Verify search is active with matches (check through log panel state)
        assert log_panel.is_search_active, "Search should be active after typing"
        assert log_panel.match_count == "1/2", "Should have 2 matches, starting at first"

        # Close with Escape (while focus is still on search input)
        await pilot.press("escape")
        await pilot.pause()
        assert "hidden" in search_bar.classes, "Search bar should be hidden after Escape"
        assert not log_panel.is_search_active, "Search should be cleared after Escape"


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=2)
async def test_log_search_navigation_keys() -> None:
    """E2E: Verify n/N navigation works after submitting search."""
    app = PivotApp(stage_names=[], watch_mode=True)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Get the log panel
        detail_panel = app.query_one("#detail-panel", TabbedDetailPanel)
        log_panel = detail_panel.query_one("#stage-logs", StageLogPanel)

        # Add logs with multiple matches
        log_panel.add_log("error one", is_stderr=False, timestamp=1000.0)
        log_panel.add_log("normal line", is_stderr=False, timestamp=1001.0)
        log_panel.add_log("error two", is_stderr=False, timestamp=1002.0)
        log_panel.add_log("error three", is_stderr=False, timestamp=1003.0)

        # Ensure we're on Logs tab
        await pilot.press("L")
        await pilot.pause()

        # Open search and type query
        await pilot.press("ctrl+f")
        await pilot.pause()
        await pilot.press("e", "r", "r", "o", "r")
        await asyncio.sleep(0.3)

        assert log_panel.match_count == "1/3", "Should start at first of 3 matches"

        # Submit with Enter to return focus to logs
        await pilot.press("enter")
        await pilot.pause()

        # Now n/N should work for navigation
        await pilot.press("n")
        assert log_panel.match_count == "2/3", "n should advance to second match"

        await pilot.press("n")
        assert log_panel.match_count == "3/3", "n should advance to third match"

        await pilot.press("n")
        assert log_panel.match_count == "1/3", "n should wrap to first match"

        await pilot.press("N")
        assert log_panel.match_count == "3/3", "N should go back to third match"
