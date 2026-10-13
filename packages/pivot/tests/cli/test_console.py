"""Tests for console output module."""

import io
from typing import override

import pytest

from pivot.cli import console
from pivot.types import ArtifactIdentity, ChangeType, StageExplanation


class _NoIsattyStream(io.StringIO):
    """Stream without isatty method for testing."""

    @override
    def __getattribute__(self, name: str) -> object:
        if name == "isatty":
            raise AttributeError("no isatty")
        return super().__getattribute__(name)


def test_console_detects_tty_support() -> None:
    """Console detects if stream is a TTY."""
    stream = io.StringIO()
    con = console.Console(stream=stream)
    # StringIO.isatty() returns False
    assert con.use_color is False


def test_console_respects_no_color_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Console disables color when NO_COLOR is set."""
    # Create a mock TTY stream
    stream = io.StringIO()
    stream.isatty = lambda: True  # type: ignore[method-assign]

    monkeypatch.setenv("NO_COLOR", "1")

    con = console.Console(stream=stream)
    assert con.use_color is False


def test_console_enables_color_on_tty_without_no_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Console enables color on TTY when NO_COLOR is not set."""
    stream = io.StringIO()
    stream.isatty = lambda: True  # type: ignore[method-assign]

    monkeypatch.delenv("NO_COLOR", raising=False)

    con = console.Console(stream=stream)
    assert con.use_color is True


def test_console_handles_stream_without_isatty() -> None:
    """Console handles stream without isatty method."""
    stream = _NoIsattyStream()

    con = console.Console(stream=stream)
    assert con.use_color is False


def test_console_force_color_on() -> None:
    """Console respects forced color=True."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=True)
    assert con.use_color is True


def test_console_force_color_off() -> None:
    """Console respects forced color=False."""
    stream = io.StringIO()
    stream.isatty = lambda: True  # type: ignore[method-assign]

    con = console.Console(stream=stream, color=False)
    assert con.use_color is False


def test_console_summary() -> None:
    """summary prints execution summary."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=False)

    con.summary(ran=5, cached=2, blocked=1, cancelled=0, failed=1, total_duration=10.5)

    output = stream.getvalue()
    assert "5" in output  # ran
    assert "2" in output  # cached
    assert "1" in output  # blocked and failed
    assert "10.50s" in output
    assert "ran" in output
    assert "cached" in output
    assert "blocked" in output
    assert "failed" in output


def test_console_color_output() -> None:
    """Console applies color codes when enabled."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=True)

    con.summary(ran=1, cached=0, blocked=0, cancelled=0, failed=0, total_duration=1.0)

    output = stream.getvalue()
    # Should contain ANSI color codes
    assert "\033[" in output
    assert "\033[0m" in output  # Reset code


def test_console_no_color_output() -> None:
    """Console omits color codes when disabled."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=False)

    con.summary(ran=1, cached=0, blocked=0, cancelled=0, failed=0, total_duration=1.0)

    output = stream.getvalue()
    # Should not contain ANSI color codes
    assert "\033[" not in output


# =============================================================================
# explain_stage tests
# =============================================================================


@pytest.mark.parametrize(
    ("will_run", "is_forced", "reason", "expected_in_output", "not_expected_in_output"),
    [
        pytest.param(
            True,
            False,
            "Code changed",
            ["my_stage", "WILL RUN", "Code changed"],
            ["(forced)"],
            id="will_run",
        ),
        pytest.param(
            True,
            True,
            "forced",
            ["my_stage", "WILL RUN (forced)", "forced"],
            [],
            id="forced",
        ),
        pytest.param(
            False,
            False,
            "",
            ["my_stage", "SKIP"],
            [],
            id="skip",
        ),
    ],
)
def test_console_explain_stage(
    will_run: bool,
    is_forced: bool,
    reason: str,
    expected_in_output: list[str],
    not_expected_in_output: list[str],
) -> None:
    """explain_stage displays correct status for different stage states."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=False)

    explanation = StageExplanation(
        stage_name="my_stage",
        will_run=will_run,
        is_forced=is_forced,
        reason=reason,
        code_changes=[],
        param_changes=[],
        dep_changes=[],
        upstream_stale=[],
    )

    con.explain_stage(explanation)

    output = stream.getvalue()
    for expected in expected_in_output:
        assert expected in output, f"Expected '{expected}' in output"
    for not_expected in not_expected_in_output:
        assert not_expected not in output, f"Did not expect '{not_expected}' in output"


def test_explain_stage_dep_changes() -> None:
    """Console.explain_stage should render dep changes without crashing."""
    output = io.StringIO()
    con = console.Console(stream=output, color=False)
    explanation: StageExplanation = {
        "stage_name": "train",
        "will_run": True,
        "is_forced": False,
        "reason": "Dependencies changed",
        "code_changes": [],
        "param_changes": [],
        "dep_changes": [
            {
                "identity": ArtifactIdentity("input_data", None),
                "old_hash": "aaa",
                "new_hash": "bbb",
                "change_type": ChangeType.MODIFIED,
            }
        ],
        "upstream_stale": [],
    }
    con.explain_stage(explanation)
    captured = output.getvalue()
    assert "input_data" in captured
