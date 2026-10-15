# tests/tui/test_log_search_e2e.py
"""End-to-end smoke test for log search using Textual's pilot API."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
from textual.widgets import Input

from pivot_tui.run import PivotApp
from pivot_tui.widgets.logs import StageLogPanel
from pivot_tui.widgets.panels import TabbedDetailPanel

if TYPE_CHECKING:
    from textual.pilot import Pilot


async def _helper_wait_for_initial_render(
    pilot: Pilot[None], log_panel: StageLogPanel, timeout: float = 10.0
) -> None:
    """Wait until the app's deferred initial stage render has cleared and redrawn the panel.

    App.on_mount schedules the detail panel update after a refresh, and the log panel
    defers its own render once more; that render calls clear(), so logs added before it
    lands are wiped.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any("No stage selected" in line.text for line in log_panel.lines):
            return
        await pilot.pause(0.05)
    pytest.fail("Log panel never rendered its initial 'No stage selected' state")


async def _helper_wait_for_match_count(
    pilot: Pilot[None], log_panel: StageLogPanel, expected: str, timeout: float = 10.0
) -> None:
    """Let the app process events until the debounced search shows `expected` matches."""
    deadline = time.monotonic() + timeout
    while log_panel.match_count != expected and time.monotonic() < deadline:
        await pilot.pause(0.05)


@pytest.mark.asyncio
async def test_log_search_smoke() -> None:
    """E2E: Verify search bar opens, accepts input, and closes."""
    # Use empty stage_names to avoid triggering pipeline lookups
    app = PivotApp(stage_names=[], watch_mode=True)

    async with app.run_test() as pilot:
        await pilot.pause()  # Let app initialize

        # Get the log panel directly (no stage selection needed)
        detail_panel = app.query_one("#detail-panel", TabbedDetailPanel)
        log_panel = detail_panel.query_one("#stage-logs", StageLogPanel)
        await _helper_wait_for_initial_render(pilot, log_panel)

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
        await _helper_wait_for_match_count(pilot, log_panel, "1/2")

        # Verify search is active with matches (check through log panel state)
        assert log_panel.is_search_active, "Search should be active after typing"
        assert log_panel.match_count == "1/2", (
            f"Should have 2 matches, starting at first; got {log_panel.match_count!r}, "
            + f"input={app.query_one('#log-search-input', Input).value!r}, focused={app.focused!r}"
        )

        # Close with Escape (while focus is still on search input)
        await pilot.press("escape")
        await pilot.pause()
        assert "hidden" in search_bar.classes, "Search bar should be hidden after Escape"
        assert not log_panel.is_search_active, "Search should be cleared after Escape"


@pytest.mark.asyncio
async def test_log_search_navigation_keys() -> None:
    """E2E: Verify n/N navigation works after submitting search."""
    app = PivotApp(stage_names=[], watch_mode=True)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Get the log panel
        detail_panel = app.query_one("#detail-panel", TabbedDetailPanel)
        log_panel = detail_panel.query_one("#stage-logs", StageLogPanel)
        await _helper_wait_for_initial_render(pilot, log_panel)

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
        await _helper_wait_for_match_count(pilot, log_panel, "1/3")

        assert log_panel.match_count == "1/3", (
            f"Should start at first of 3 matches; got {log_panel.match_count!r}, "
            + f"input={app.query_one('#log-search-input', Input).value!r}, focused={app.focused!r}"
        )

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
