"""Console output with colors and progress tracking.

Provides formatted output for pipeline execution with:
- Colored status indicators (green=ran, yellow=skipped, red=failed)
- Progress tracking (stage N of M)
- Timing information
"""

import sys
import time
from typing import TextIO

from pivot.types import StageDisplayStatus, StageStatus

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
    import os

    return not os.environ.get("NO_COLOR")


class Console:
    """Console output handler with colors and progress tracking."""

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

        print(f"{progress} {name}: {status_text}...", file=self.stream, flush=True)

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

        print(" ".join(parts), file=self.stream, flush=True)
        self._current_stage = None
        self._stage_start = None

    def parallel_group_start(self, group_index: int, stage_names: list[str]) -> None:
        """Print parallel group start message."""
        stages_str = ", ".join(stage_names)
        header = self._color(f"=== Parallel group {group_index} ===", "cyan", "bold")
        print(f"\n{header} ({len(stage_names)} stages: {stages_str})", file=self.stream, flush=True)

    def summary(
        self,
        ran: int,
        skipped: int,
        failed: int,
        total_duration: float,
    ) -> None:
        """Print execution summary."""
        print("", file=self.stream)  # blank line

        ran_text = self._color(str(ran), "green") if ran > 0 else str(ran)
        skipped_text = self._color(str(skipped), "yellow") if skipped > 0 else str(skipped)
        failed_text = self._color(str(failed), "red", "bold") if failed > 0 else str(failed)

        summary = f"Summary: {ran_text} ran, {skipped_text} skipped, {failed_text} failed"
        duration = self._color(f"[{total_duration:.2f}s total]", "dim")

        print(f"{summary} {duration}", file=self.stream, flush=True)

    def error(self, message: str) -> None:
        """Print error message."""
        prefix = self._color("Error:", "red", "bold")
        print(f"{prefix} {message}", file=self.stream, flush=True)

    def stage_output(self, name: str, line: str, is_stderr: bool = False) -> None:
        """Print captured stage output."""
        prefix = self._color(f"  [{name}]", "dim")
        line_colored = self._color(line, "red") if is_stderr else line
        print(f"{prefix} {line_colored}", file=self.stream, flush=True)


# Global console instance for convenience
_console: Console | None = None


def get_console() -> Console:
    """Get or create global console instance."""
    global _console
    if _console is None:
        _console = Console()
    return _console
