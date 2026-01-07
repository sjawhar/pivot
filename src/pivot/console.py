import os
import pathlib
import sys
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TextIO

from pivot.types import StageDisplayStatus, StageExplanation, StageStatus

if TYPE_CHECKING:
    from collections.abc import Set

    from watchfiles import Change

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

    def watch_start(self, paths: list[pathlib.Path]) -> None:
        """Print watch mode startup message."""
        header = self._color("Watch mode started", "cyan", "bold")
        paths_str = ", ".join(str(p) for p in paths[:3])
        if len(paths) > 3:
            paths_str += f" (+{len(paths) - 3} more)"
        print(f"\n{header}", file=self.stream, flush=True)
        print(f"Watching: {paths_str}", file=self.stream, flush=True)

    def watch_waiting(self) -> None:
        """Print waiting for changes message."""
        msg = self._color("Waiting for file changes... (Ctrl+C to exit)", "dim")
        print(f"\n{msg}\n", file=self.stream, flush=True)

    def watch_changes_detected(self, changes: "Set[tuple[Change, str]]") -> None:
        """Print detected changes summary."""
        files = [pathlib.Path(path).name for _, path in changes]
        files_str = ", ".join(files[:5])
        if len(files) > 5:
            files_str += f" (+{len(files) - 5} more)"
        header = self._color("Changes detected:", "yellow", "bold")
        print(f"\n{header} {files_str}", file=self.stream, flush=True)

    def watch_stopped(self) -> None:
        """Print watch mode stopped message."""
        msg = self._color("\nWatch mode stopped", "cyan")
        print(msg, file=self.stream, flush=True)

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

        print(f"\n  {self._color(header, 'cyan')}", file=self.stream, flush=True)

        for change in changes:
            key = change[key_field]
            change_type = change["change_type"]
            old_val = change[old_field]
            new_val = change[new_field]

            print(f"    {key}", file=self.stream, flush=True)

            if change_type == "modified":
                print(
                    f"      Old: {self._color(str(old_val) if old_val else 'N/A', 'red')}",
                    file=self.stream,
                    flush=True,
                )
                print(
                    f"      New: {self._color(str(new_val) if new_val else 'N/A', 'green')}",
                    file=self.stream,
                    flush=True,
                )
            elif change_type == "added":
                print(
                    f"      {self._color('(added)', 'green')} {new_val}",
                    file=self.stream,
                    flush=True,
                )
            else:
                print(
                    f"      {self._color('(removed)', 'red')} {old_val}",
                    file=self.stream,
                    flush=True,
                )

    def explain_stage(self, explanation: StageExplanation) -> None:
        """Print detailed explanation of why a stage would run."""
        name = explanation["stage_name"]
        will_run = explanation["will_run"]
        reason = explanation["reason"]

        print(f"\nStage: {self._color(name, 'bold')}", file=self.stream, flush=True)

        status_text = (
            self._color("WILL RUN", "green", "bold") if will_run else self._color("SKIP", "yellow")
        )
        print(f"  Status: {status_text}", file=self.stream, flush=True)

        if reason:
            print(f"  Reason: {reason}", file=self.stream, flush=True)

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
        print("", file=self.stream)
        run_text = self._color(str(will_run), "green") if will_run > 0 else str(will_run)
        unchanged_text = self._color(str(unchanged), "yellow") if unchanged > 0 else str(unchanged)
        print(
            f"Summary: {run_text} will run, {unchanged_text} unchanged",
            file=self.stream,
            flush=True,
        )


# Global console instance for convenience
_console: Console | None = None


def get_console() -> Console:
    """Get or create global console instance."""
    global _console
    if _console is None:
        _console = Console()
    return _console
