import os
import sys
import time
from collections.abc import Mapping, Sequence
from typing import TextIO

import click

from pivot.types import StageDisplayStatus, StageExplanation, StageStatus

# ANSI color codes
_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
}


def _supports_color(stream: TextIO) -> bool:
    """Check if terminal supports color output."""
    if not hasattr(stream, "isatty"):
        return False
    if not stream.isatty():
        return False
    # Check for NO_COLOR environment variable
    return not os.environ.get("NO_COLOR")


class Console:
    """Console output handler with colors and progress tracking."""

    stream: TextIO
    use_color: bool

    def __init__(self, stream: TextIO | None = None, color: bool | None = None) -> None:
        """Initialize console.

        Args:
            stream: Output stream (default: sys.stderr)
            color: Force color on/off (default: auto-detect)
        """
        self.stream = stream or sys.stderr
        self.use_color = color if color is not None else _supports_color(self.stream)
        self._current_stage: str | None = None
        self._stage_start: float | None = None

    def _color(self, text: str, *codes: str) -> str:
        """Apply color codes to text."""
        if not self.use_color:
            return text
        prefix = "".join(_COLORS.get(c, "") for c in codes)
        return f"{prefix}{text}{_COLORS['reset']}"

    def _echo(self, message: str) -> None:
        """Output message, preserving ANSI codes when colors are enabled."""
        click.echo(message, file=self.stream, color=self.use_color)

    def stage_start(
        self,
        name: str,
        index: int,
        total: int,
        status: StageDisplayStatus,
    ) -> None:
        """Print stage start message."""
        self._current_stage = name
        self._stage_start = time.perf_counter()

        progress = self._color(f"[{index}/{total}]", "dim")

        match status:
            case StageDisplayStatus.CHECKING:
                status_text = self._color("checking", "dim")
            case StageDisplayStatus.RUNNING:
                status_text = self._color("running", "blue", "bold")
            case StageDisplayStatus.WAITING:
                status_text = self._color("waiting", "dim")

        self._echo(f"{progress} {name}: {status_text}...")

    def stage_result(
        self,
        name: str,
        index: int,
        total: int,
        status: StageStatus,
        reason: str = "",
        duration: float | None = None,
    ) -> None:
        """Print stage result message."""
        progress = self._color(f"[{index}/{total}]", "dim")

        match status:
            case StageStatus.RAN:
                status_text = self._color("ran", "green", "bold")
            case StageStatus.SKIPPED:
                status_text = self._color("skipped", "yellow")
            case StageStatus.FAILED:
                status_text = self._color("FAILED", "red", "bold")
            case _:
                status_text = self._color(str(status), "dim")

        # Calculate duration if not provided
        if duration is None and self._stage_start is not None:
            duration = time.perf_counter() - self._stage_start

        parts = [f"{progress} {name}: {status_text}"]
        if reason:
            parts.append(self._color(f"({reason})", "dim"))
        if duration is not None:
            parts.append(self._color(f"[{duration:.2f}s]", "dim"))

        self._echo(" ".join(parts))
        self._current_stage = None
        self._stage_start = None

    def parallel_group_start(self, group_index: int, stage_names: list[str]) -> None:
        """Print parallel group start message."""
        stages_str = ", ".join(stage_names)
        header = self._color(f"=== Parallel group {group_index} ===", "cyan", "bold")
        self._echo(f"\n{header} ({len(stage_names)} stages: {stages_str})")

    def summary(
        self,
        ran: int,
        skipped: int,
        failed: int,
        total_duration: float,
    ) -> None:
        """Print execution summary."""
        self._echo("")  # blank line

        ran_text = self._color(str(ran), "green") if ran > 0 else str(ran)
        skipped_text = self._color(str(skipped), "yellow") if skipped > 0 else str(skipped)
        failed_text = self._color(str(failed), "red", "bold") if failed > 0 else str(failed)

        summary = f"Summary: {ran_text} ran, {skipped_text} skipped, {failed_text} failed"
        duration = self._color(f"[{total_duration:.2f}s total]", "dim")

        self._echo(f"{summary} {duration}")

    def error(self, message: str, suggestion: str | None = None) -> None:
        """Print error message with optional suggestion."""
        prefix = self._color("Error:", "red", "bold")
        self._echo(f"{prefix} {message}")
        if suggestion:
            hint = self._color("Tip:", "cyan")
            self._echo(f"  {hint} {suggestion}")

    def stage_output(self, name: str, line: str, is_stderr: bool = False) -> None:
        """Print captured stage output."""
        prefix = self._color(f"  [{name}]", "dim")
        line_colored = self._color(line, "red") if is_stderr else line
        self._echo(f"{prefix} {line_colored}")

    def _print_changes(
        self,
        header: str,
        changes: Sequence[Mapping[str, object]],
        key_field: str,
        old_field: str,
        new_field: str,
    ) -> None:
        """Print a list of changes with consistent formatting."""
        if not changes:
            return

        self._echo(f"\n  {self._color(header, 'cyan')}")

        for change in changes:
            key = change[key_field]
            change_type = change["change_type"]
            old_val = change[old_field]
            new_val = change[new_field]

            self._echo(f"    {key}")

            if change_type == "modified":
                self._echo(f"      Old: {self._color(str(old_val) if old_val else 'N/A', 'red')}")
                self._echo(f"      New: {self._color(str(new_val) if new_val else 'N/A', 'green')}")
            elif change_type == "added":
                self._echo(f"      {self._color('(added)', 'green')} {new_val}")
            else:
                self._echo(f"      {self._color('(removed)', 'red')} {old_val}")

    def explain_stage(self, explanation: StageExplanation) -> None:
        """Print detailed explanation of why a stage would run."""
        name = explanation["stage_name"]
        will_run = explanation["will_run"]
        is_forced = explanation["is_forced"]
        reason = explanation["reason"]

        self._echo(f"\nStage: {self._color(name, 'bold')}")

        if will_run:
            status_label = "WILL RUN (forced)" if is_forced else "WILL RUN"
            status_text = self._color(status_label, "green", "bold")
        else:
            status_text = self._color("SKIP", "yellow")
        self._echo(f"  Status: {status_text}")

        if reason:
            self._echo(f"  Reason: {reason}")

        self._print_changes(
            "Code Changes:", explanation["code_changes"], "key", "old_hash", "new_hash"
        )
        self._print_changes(
            "Param Changes:", explanation["param_changes"], "key", "old_value", "new_value"
        )
        self._print_changes(
            "Dependency Changes:", explanation["dep_changes"], "path", "old_hash", "new_hash"
        )

    def explain_summary(self, will_run: int, unchanged: int) -> None:
        """Print summary after explain output."""
        self._echo("")
        run_text = self._color(str(will_run), "green") if will_run > 0 else str(will_run)
        unchanged_text = self._color(str(unchanged), "yellow") if unchanged > 0 else str(unchanged)
        self._echo(f"Summary: {run_text} will run, {unchanged_text} unchanged")


# Global console instance for convenience
_console: Console | None = None


def get_console() -> Console:
    """Get or create global console instance."""
    global _console
    if _console is None:
        _console = Console()
    return _console
