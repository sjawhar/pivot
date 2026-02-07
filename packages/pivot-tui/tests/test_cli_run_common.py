"""Tests for TUI-specific CLI run common functionality."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from pivot.cli import _run_common
from pivot.engine import engine
from pivot_tui.sink import TuiSink


def test_configure_output_sink_tui_mode(mocker: MockerFixture) -> None:
    """configure_output_sink adds TuiSink when tui=True."""
    mock_engine = mocker.MagicMock(spec=engine.Engine)
    mock_app = mocker.MagicMock()

    _run_common.configure_output_sink(
        mock_engine,
        quiet=False,
        as_json=False,
        tui=True,
        app=mock_app,
        run_id="test-run-123",
        use_console=False,
        jsonl_callback=None,
    )

    mock_engine.add_sink.assert_called_once()
    added_sink = mock_engine.add_sink.call_args[0][0]
    assert isinstance(added_sink, TuiSink)
