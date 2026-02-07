from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import rich.console
import rich.markup

from pivot.engine.types import StageCompleted
from pivot.types import StageStatus

if TYPE_CHECKING:
    from pivot.engine.types import OutputEvent

__all__ = [
    "ConsoleSink",
    "ResultCollectorSink",
]


class ConsoleSink:
    """Async sink that prints stage events to console."""

    _console: rich.console.Console
    _show_output: bool

    def __init__(self, *, console: rich.console.Console, show_output: bool = False) -> None:
        self._console = console
        self._show_output = show_output

    async def handle(self, event: OutputEvent) -> None:
        """Handle output event by printing to console."""
        match event["type"]:
            case "stage_started":
                self._console.print(f"Running {event['stage']}...")
            case "stage_completed":
                stage = event["stage"]
                duration = event["duration_ms"] / 1000
                match event["status"]:
                    case StageStatus.SKIPPED:
                        self._console.print(f"  {stage}: skipped")
                    case StageStatus.RAN:
                        self._console.print(f"  {stage}: done ({duration:.1f}s)")
                    case StageStatus.FAILED:
                        self._console.print(f"  {stage}: [red]FAILED[/red]")
                        if event["reason"]:
                            # Indent each line of the error for readability
                            # Escape to prevent Rich markup injection from error messages
                            for line in event["reason"].rstrip().split("\n"):
                                self._console.print(f"    [dim]{rich.markup.escape(line)}[/dim]")
            case "log_line" if self._show_output:
                stage = event["stage"]
                # Escape line content to prevent Rich markup injection from stage output
                line = rich.markup.escape(event["line"])
                if event["is_stderr"]:
                    self._console.print(f"[red]\\[{stage}][/red] [red]{line}[/red]")
                else:
                    self._console.print(f"\\[{stage}] {line}")
            case _:
                pass  # Ignore other events

    async def close(self) -> None:
        """No cleanup needed."""


class ResultCollectorSink:
    """Async sink that collects stage results for programmatic access."""

    _results: dict[str, StageCompleted]
    _lock: anyio.Lock

    def __init__(self) -> None:
        self._results = dict[str, StageCompleted]()
        self._lock = anyio.Lock()

    async def handle(self, event: OutputEvent) -> None:
        """Collect stage_completed events."""
        if event["type"] != "stage_completed":
            return

        async with self._lock:
            self._results[event["stage"]] = event

    async def get_results(self) -> dict[str, StageCompleted]:
        """Get collected results. Call after run() completes."""
        async with self._lock:
            return dict(self._results)

    async def close(self) -> None:
        """No cleanup needed."""
