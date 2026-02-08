"""Integration tests for TUI with direct post_message."""

from __future__ import annotations

import pytest

from pivot_tui.run import PivotApp
from pivot_tui.sink import TuiSink


@pytest.mark.anyio
async def test_tui_sink_with_real_app() -> None:
    """Integration test: TuiSink posts to real PivotApp instance."""
    app = PivotApp(
        stage_names=["test_stage"],
        watch_mode=True,
    )

    sink = TuiSink(app=app, run_id="integration-test")

    # Post events before app.run() - they should queue in Textual
    await sink.handle(
        {
            "type": "stage_started",
            "stage": "test_stage",
            "index": 0,
            "total": 1,
        }
    )

    await sink.close()
    # If we get here without exception, post_message works correctly
