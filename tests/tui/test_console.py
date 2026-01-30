import io
from typing import override

import pytest

from pivot.tui import console
from pivot.types import StageDisplayStatus, StageExplanation, StageStatus


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


def test_console_stage_start_running() -> None:
    """stage_start prints running status."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=False)

    con.stage_start("my_stage", index=1, total=5, status=StageDisplayStatus.RUNNING)

    output = stream.getvalue()
    assert "[1/5]" in output
    assert "my_stage" in output
    assert "running" in output


def test_console_stage_start_fingerprinting() -> None:
    """stage_start prints fingerprinting status."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=False)

    con.stage_start("my_stage", index=1, total=5, status=StageDisplayStatus.FINGERPRINTING)

    output = stream.getvalue()
    assert "fingerprinting" in output


def test_console_stage_start_waiting() -> None:
    """stage_start prints waiting status."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=False)

    con.stage_start("my_stage", index=1, total=5, status=StageDisplayStatus.WAITING)

    output = stream.getvalue()
    assert "waiting" in output


def test_console_stage_result_ran() -> None:
    """stage_result prints ran status."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=False)

    con.stage_result("my_stage", index=1, total=5, status=StageStatus.RAN, reason="code changed")

    output = stream.getvalue()
    assert "[1/5]" in output
    assert "my_stage" in output
    assert "ran" in output
    assert "code changed" in output


def test_console_stage_result_cached() -> None:
    """stage_result prints cached status for unchanged stages."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=False)

    con.stage_result("my_stage", index=1, total=5, status=StageStatus.SKIPPED, reason="unchanged")

    output = stream.getvalue()
    assert "cached" in output


def test_console_stage_result_blocked() -> None:
    """stage_result prints blocked status for upstream failures."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=False)

    con.stage_result(
        "my_stage", index=1, total=5, status=StageStatus.SKIPPED, reason="upstream 'other' failed"
    )

    output = stream.getvalue()
    assert "blocked" in output


def test_console_stage_result_failed() -> None:
    """stage_result prints failed status."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=False)

    con.stage_result("my_stage", index=1, total=5, status=StageStatus.FAILED, reason="error")

    output = stream.getvalue()
    assert "FAILED" in output
    assert "error" in output


def test_console_stage_result_with_duration() -> None:
    """stage_result prints duration when provided."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=False)

    con.stage_result("my_stage", index=1, total=5, status=StageStatus.RAN, duration=1.234)

    output = stream.getvalue()
    assert "1.23s" in output


def test_console_stage_result_calculates_duration() -> None:
    """stage_result calculates duration from stage_start."""
    import time

    stream = io.StringIO()
    con = console.Console(stream=stream, color=False)

    con.stage_start("my_stage", index=1, total=5, status=StageDisplayStatus.RUNNING)
    time.sleep(0.01)  # Small delay
    con.stage_result("my_stage", index=1, total=5, status=StageStatus.RAN)

    output = stream.getvalue()
    # Duration should be calculated and shown
    assert "s]" in output  # Format is [X.XXs]


def test_console_summary() -> None:
    """summary prints execution summary."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=False)

    con.summary(ran=5, cached=2, blocked=1, failed=1, total_duration=10.5)

    output = stream.getvalue()
    assert "5" in output  # ran
    assert "2" in output  # cached
    assert "1" in output  # blocked and failed
    assert "10.50s" in output
    assert "ran" in output
    assert "cached" in output
    assert "blocked" in output
    assert "failed" in output


def test_console_error() -> None:
    """error prints error message."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=False)

    con.error("Something went wrong")

    output = stream.getvalue()
    assert "Error:" in output
    assert "Something went wrong" in output


def test_console_stage_output_stdout() -> None:
    """stage_output prints captured stdout."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=False)

    con.stage_output("my_stage", "Hello from stage", is_stderr=False)

    output = stream.getvalue()
    assert "[my_stage]" in output
    assert "Hello from stage" in output


def test_console_stage_output_stderr() -> None:
    """stage_output prints captured stderr distinctly."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=True)  # Color on to test stderr formatting

    con.stage_output("my_stage", "Error message", is_stderr=True)

    output = stream.getvalue()
    assert "[my_stage]" in output
    assert "Error message" in output
    # With color on, stderr should have red color codes
    assert "\033[31m" in output  # Red color code


def test_console_parallel_group_start() -> None:
    """parallel_group_start prints group header."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=False)

    con.parallel_group_start(1, ["stage_a", "stage_b", "stage_c"])

    output = stream.getvalue()
    assert "Parallel group 1" in output
    assert "3 stages" in output
    assert "stage_a" in output
    assert "stage_b" in output
    assert "stage_c" in output


def test_console_color_output() -> None:
    """Console applies color codes when enabled."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=True)

    con.stage_result("my_stage", index=1, total=5, status=StageStatus.RAN, reason="test")

    output = stream.getvalue()
    # Should contain ANSI color codes
    assert "\033[" in output
    assert "\033[0m" in output  # Reset code


def test_console_no_color_output() -> None:
    """Console omits color codes when disabled."""
    stream = io.StringIO()
    con = console.Console(stream=stream, color=False)

    con.stage_result("my_stage", index=1, total=5, status=StageStatus.RAN, reason="test")

    output = stream.getvalue()
    # Should not contain ANSI color codes
    assert "\033[" not in output


def test_get_console_returns_singleton() -> None:
    """get_console returns the same instance on repeated calls."""
    con1 = console.get_console()
    con2 = console.get_console()

    assert con1 is con2


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
