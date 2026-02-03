from __future__ import annotations

import pytest

from pivot.tui.types import StageInfo
from pivot.tui.widgets import status
from pivot.types import StageStatus

# =============================================================================
# Status Symbol Tests (Consolidated)
# =============================================================================


@pytest.mark.parametrize(
    ("status_val", "reason", "expected_symbol", "expected_style"),
    [
        pytest.param(StageStatus.READY, "", "\u25cb", "dim", id="pending"),
        pytest.param(StageStatus.IN_PROGRESS, "", "\u25b6", "blue bold", id="running"),
        pytest.param(StageStatus.RAN, "", "\u25cf", "green bold", id="success"),
        pytest.param(StageStatus.FAILED, "error", "\u2717", "red bold", id="failed"),
        pytest.param(StageStatus.SKIPPED, "cache hit", "\u21ba", "yellow", id="cached"),
        pytest.param(StageStatus.SKIPPED, "upstream failed", "\u25c7", "red", id="blocked"),
    ],
)
def test_get_status_symbol(
    status_val: StageStatus, reason: str, expected_symbol: str, expected_style: str
) -> None:
    """Status symbols and styles match expected values."""
    symbol, style = status.get_status_symbol(status_val, reason)
    assert symbol == expected_symbol
    assert style == expected_style


# =============================================================================
# get_status_icon Tests
# =============================================================================


@pytest.mark.parametrize(
    ("status_val", "reason", "expected_icon"),
    [
        pytest.param(StageStatus.RAN, "", "[green]\u2713[/]", id="success"),
        pytest.param(StageStatus.FAILED, "error", "[red]\u2717[/]", id="failed"),
        pytest.param(StageStatus.SKIPPED, "cache hit", "[yellow]\u21ba[/]", id="cached"),
        pytest.param(StageStatus.SKIPPED, "upstream failed", "[red]\u25c7[/]", id="blocked"),
        pytest.param(StageStatus.READY, "", "", id="pending_no_icon"),
        pytest.param(StageStatus.IN_PROGRESS, "", "", id="running_no_icon"),
    ],
)
def test_get_status_icon(status_val: StageStatus, reason: str, expected_icon: str) -> None:
    """get_status_icon returns correct Rich markup for all display categories."""
    assert status.get_status_icon(status_val, reason) == expected_icon


@pytest.mark.parametrize(
    ("status_val", "reason", "expected_plain"),
    [
        pytest.param(StageStatus.RAN, "", "\u2713", id="success"),
        pytest.param(StageStatus.FAILED, "error", "\u2717", id="failed"),
        pytest.param(StageStatus.SKIPPED, "cache hit", "\u21ba", id="cached"),
        pytest.param(StageStatus.SKIPPED, "upstream failed", "\u25c7", id="blocked"),
        pytest.param(StageStatus.READY, "", "", id="pending_no_icon"),
    ],
)
def test_get_status_icon_plain(status_val: StageStatus, reason: str, expected_plain: str) -> None:
    """get_status_icon_plain returns correct plain text for all display categories."""
    assert status.get_status_icon_plain(status_val, reason) == expected_plain


# =============================================================================
# get_status_table_cell Tests
# =============================================================================


@pytest.mark.parametrize(
    ("status_val", "reason", "should_contain"),
    [
        pytest.param(StageStatus.RAN, "", ["\u2713", "ran"], id="success"),
        pytest.param(StageStatus.FAILED, "error", ["\u2717", "fail"], id="failed"),
        pytest.param(StageStatus.SKIPPED, "cache hit", ["\u21ba", "cache"], id="cached"),
        pytest.param(StageStatus.SKIPPED, "upstream failed", ["\u25c7", "block"], id="blocked"),
        pytest.param(StageStatus.READY, "", ["PENDING"], id="pending"),
        pytest.param(StageStatus.IN_PROGRESS, "", ["RUNNING"], id="running"),
    ],
)
def test_get_status_table_cell(
    status_val: StageStatus, reason: str, should_contain: list[str]
) -> None:
    """get_status_table_cell returns properly formatted cells for all categories."""
    cell = status.get_status_table_cell(status_val, reason)
    for text in should_contain:
        assert text in cell or text.lower() in cell.lower(), f"Expected '{text}' in cell '{cell}'"


# =============================================================================
# format_elapsed Tests
# =============================================================================


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [
        pytest.param(None, "", id="none"),
        pytest.param(0.0, "(0:00)", id="zero"),
        pytest.param(5.0, "(0:05)", id="seconds_padded"),
        pytest.param(45.0, "(0:45)", id="seconds"),
        pytest.param(45.9, "(0:45)", id="fractional_truncated"),
        pytest.param(59.99, "(0:59)", id="just_under_minute"),
        pytest.param(60.0, "(1:00)", id="one_minute"),
        pytest.param(125.0, "(2:05)", id="minutes_seconds"),
        pytest.param(3661.0, "(61:01)", id="large_value"),
        pytest.param(7200.0, "(120:00)", id="two_hours"),
        pytest.param(-5.0, "(0:00)", id="negative_clamped"),
        pytest.param(float("inf"), "", id="infinity"),
        pytest.param(float("-inf"), "", id="negative_infinity"),
        pytest.param(float("nan"), "", id="nan"),
    ],
)
def test_format_elapsed(elapsed: float | None, expected: str) -> None:
    """format_elapsed handles various inputs correctly."""
    assert status.format_elapsed(elapsed) == expected


# =============================================================================
# get_status_label Tests
# =============================================================================


@pytest.mark.parametrize(
    ("status_val", "reason", "expected_label"),
    [
        pytest.param(StageStatus.READY, "", "PENDING", id="pending"),
        pytest.param(StageStatus.IN_PROGRESS, "", "RUNNING", id="running"),
        pytest.param(StageStatus.RAN, "", "SUCCESS", id="success"),
        pytest.param(StageStatus.COMPLETED, "", "SUCCESS", id="completed_as_success"),
        pytest.param(StageStatus.FAILED, "error", "FAILED", id="failed"),
        pytest.param(StageStatus.SKIPPED, "cache hit", "CACHED", id="cached"),
        pytest.param(StageStatus.SKIPPED, "upstream failed", "BLOCKED", id="blocked"),
        pytest.param(StageStatus.UNKNOWN, "", "UNKNOWN", id="unknown"),
    ],
)
def test_get_status_label(status_val: StageStatus, reason: str, expected_label: str) -> None:
    """get_status_label returns correct verbose labels for all categories."""
    label, style = status.get_status_label(status_val, reason)
    assert label == expected_label
    assert isinstance(style, str)
    assert len(style) > 0, "Style should not be empty"


# =============================================================================
# count_statuses Tests
# =============================================================================


def test_count_statuses_empty_list() -> None:
    """count_statuses handles empty stage list."""
    counts = status.count_statuses([])
    assert counts == {"running": 0, "completed": 0, "failed": 0}


def test_count_statuses_mixed_stages() -> None:
    """count_statuses correctly counts different stage statuses."""
    stages = [StageInfo(f"s{i}", i, 5) for i in range(1, 6)]
    stages[0].status = StageStatus.IN_PROGRESS
    stages[1].status = StageStatus.RAN
    stages[2].status = StageStatus.COMPLETED
    stages[3].status = StageStatus.FAILED
    stages[4].status = StageStatus.SKIPPED  # Not counted

    counts = status.count_statuses(stages)
    assert counts == {"running": 1, "completed": 2, "failed": 1}


def test_count_statuses_ignores_non_terminal_statuses() -> None:
    """count_statuses does not count READY, SKIPPED, or UNKNOWN."""
    stages = [StageInfo(f"s{i}", i, 3) for i in range(1, 4)]
    stages[0].status = StageStatus.READY
    stages[1].status = StageStatus.SKIPPED
    stages[2].status = StageStatus.UNKNOWN

    counts = status.count_statuses(stages)
    assert counts == {"running": 0, "completed": 0, "failed": 0}
