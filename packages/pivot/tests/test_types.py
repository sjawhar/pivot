"""Tests for pivot.types module."""

from pivot.types import CompletionType, StageStatus


def test_completion_type_includes_ran() -> None:
    """CompletionType should include RAN."""
    status: CompletionType = StageStatus.RAN
    assert status == StageStatus.RAN


def test_completion_type_includes_skipped() -> None:
    """CompletionType should include SKIPPED."""
    status: CompletionType = StageStatus.SKIPPED
    assert status == StageStatus.SKIPPED


def test_completion_type_includes_failed() -> None:
    """CompletionType should include FAILED."""
    status: CompletionType = StageStatus.FAILED
    assert status == StageStatus.FAILED
