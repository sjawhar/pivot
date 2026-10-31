import os
import sys
from collections.abc import Mapping, Sequence
from typing import TextIO

import click

from pivot import display
from pivot.types import ArtifactIdentity, StageExplanation, identity_key

# ANSI color codes
_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
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
    """Plain-text CLI output for run summaries and stage explanations, with optional color."""

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

    def _color(self, text: str, *codes: str) -> str:
        """Apply color codes to text."""
        if not self.use_color:
            return text
        prefix = "".join(_COLORS.get(c, "") for c in codes)
        return f"{prefix}{text}{_COLORS['reset']}"

    def _echo(self, message: str) -> None:
        """Output message to the console stream."""
        click.echo(message, file=self.stream, color=self.use_color)

    def summary(
        self,
        ran: int,
        cached: int,
        blocked: int,
        cancelled: int,
        failed: int,
        total_duration: float,
    ) -> None:
        """Print execution summary."""
        self._echo("")  # blank line

        ran_text = self._color(str(ran), "green") if ran > 0 else str(ran)
        cached_text = self._color(str(cached), "yellow") if cached > 0 else str(cached)
        blocked_text = self._color(str(blocked), "red") if blocked > 0 else str(blocked)
        cancelled_text = self._color(str(cancelled), "yellow") if cancelled > 0 else str(cancelled)
        failed_text = self._color(str(failed), "red", "bold") if failed > 0 else str(failed)

        parts = [f"{ran_text} ran", f"{cached_text} cached", f"{blocked_text} blocked"]
        if cancelled > 0:
            parts.append(f"{cancelled_text} cancelled")
        parts.append(f"{failed_text} failed")
        summary = "Summary: " + ", ".join(parts)
        duration = self._color(f"[{total_duration:.2f}s total]", "dim")

        self._echo(f"{summary} {duration}")

    def failed_stages(self, failed_names: list[str]) -> None:
        """Print list of failed stage names after summary."""
        if not failed_names:
            return
        header = self._color("Failed stages:", "red", "bold")
        self._echo(header)
        for name in failed_names:
            self._echo(f"  {self._color(name, 'red')}")

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
            if isinstance(key, ArtifactIdentity):
                key = identity_key(key)
            change_type = change["change_type"]
            old_val = change[old_field]
            new_val = change[new_field]

            self._echo(f"    {display.escape_control(str(key))}")

            if change_type == "modified":
                old_text = display.escape_control(str(old_val) if old_val is not None else "N/A")
                new_text = display.escape_control(str(new_val) if new_val is not None else "N/A")
                self._echo(f"      Old: {self._color(old_text, 'red')}")
                self._echo(f"      New: {self._color(new_text, 'green')}")
            elif change_type == "added":
                added = display.escape_control(str(new_val))
                self._echo(f"      {self._color('(added)', 'green')} {added}")
            else:
                removed = display.escape_control(str(old_val))
                self._echo(f"      {self._color('(removed)', 'red')} {removed}")

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
            self._echo(f"  Reason: {display.escape_control(reason, keep_newlines=True)}")

        self._print_changes(
            "Code Changes:", explanation["code_changes"], "key", "old_hash", "new_hash"
        )
        self._print_changes(
            "Param Changes:", explanation["param_changes"], "key", "old_value", "new_value"
        )
        self._print_changes(
            "Dependency Changes:", explanation["dep_changes"], "identity", "old_hash", "new_hash"
        )

    def explain_not_expanded(self, name: str, reason: str) -> None:
        """Print a mapped stage (``Pipeline.map``) not yet expanded, and what it waits for."""
        self._echo(f"\nStage: {self._color(name, 'bold')}")
        self._echo(f"  Status: {self._color('NOT YET EXPANDED', 'cyan')}")
        self._echo(f"  Reason: {display.escape_control(reason, keep_newlines=True)}")

    def explain_summary(self, will_run: int, unchanged: int, not_expanded: int = 0) -> None:
        """Print summary after explain output."""
        self._echo("")
        run_text = self._color(str(will_run), "green") if will_run > 0 else str(will_run)
        unchanged_text = self._color(str(unchanged), "yellow") if unchanged > 0 else str(unchanged)
        summary = f"Summary: {run_text} will run, {unchanged_text} unchanged"
        if not_expanded:
            summary += f", {self._color(str(not_expanded), 'cyan')} not yet expanded"
        self._echo(summary)
