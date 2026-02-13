from __future__ import annotations

import time
from typing import TYPE_CHECKING

import anyio
import rich.console
import rich.live
import rich.markup
import rich.text

from pivot.engine.types import StageCompleted
from pivot.types import DisplayCategory, categorize_stage_result

if TYPE_CHECKING:
    from pivot.engine.types import OutputEvent

__all__ = [
    "StaticConsoleSink",
    "LiveConsoleSink",
    "ResultCollectorSink",
]


_CATEGORY_SYMBOL: dict[DisplayCategory, str] = {
    DisplayCategory.SUCCESS: "[green]✓[/green]",
    DisplayCategory.CACHED: "[yellow]○[/yellow]",
    DisplayCategory.BLOCKED: "[red]○[/red]",
    DisplayCategory.CANCELLED: "[dim yellow]○[/dim yellow]",
    DisplayCategory.FAILED: "[bold red]✗[/bold red]",
}
_CATEGORY_WORD: dict[DisplayCategory, str] = {
    DisplayCategory.SUCCESS: "[green]done[/green]",
    DisplayCategory.CACHED: "[yellow]cached[/yellow]",
    DisplayCategory.BLOCKED: "[red]blocked[/red]",
    DisplayCategory.CANCELLED: "[dim yellow]cancelled[/dim yellow]",
    DisplayCategory.FAILED: "[bold red]FAILED[/bold red]",
}
_SKIP_CATEGORIES = {DisplayCategory.CACHED, DisplayCategory.BLOCKED, DisplayCategory.CANCELLED}
_MAX_NAME_WIDTH = 50


def _format_stage_line(
    *,
    index: int,
    total: int,
    stage: str,
    category: DisplayCategory,
    duration_ms: float,
    name_width: int,
) -> str:
    total_digits = len(str(total))
    counter = f"[{index:>{total_digits}}/{total}]"
    display_width = max(1, min(name_width, _MAX_NAME_WIDTH))
    stage_name = f"{stage[: display_width - 1]}…" if len(stage) > display_width else stage
    padded_name = stage_name.ljust(display_width)
    symbol = _CATEGORY_SYMBOL.get(category, "[dim]?[/dim]")
    word = _CATEGORY_WORD.get(category, f"[dim]{category.value}[/dim]")
    if category == DisplayCategory.SUCCESS:
        duration_s = duration_ms / 1000
        return f"  {counter} {symbol} [bold]{padded_name}[/bold] {word} {duration_s:.1f}s"
    if category in _SKIP_CATEGORIES:
        return f"[dim]  {counter} {symbol} {padded_name} {word}[/dim]"
    return f"  {counter} {symbol} [bold]{padded_name}[/bold] {word}"


def _format_skip_group_line(
    *, start_index: int, end_index: int, total: int, count: int, category: DisplayCategory
) -> str:
    total_digits = len(str(total))
    range_text = f"{start_index:>{total_digits}}–{end_index:>{total_digits}}/{total}"
    return f"[dim]  [{range_text}] ○ {count} {category.value}[/dim]"


def _format_error_detail(reason: str, *, total: int) -> list[str]:
    total_digits = len(str(total))
    counter_sample = f"[{1:>{total_digits}}/{total}]"
    indent = " " * (2 + len(counter_sample) + 3)
    return [f"{indent}[dim red]{rich.markup.escape(line)}[/dim red]" for line in reason.split("\n")]


def _categorize(event: StageCompleted) -> DisplayCategory:
    return categorize_stage_result(event["status"])


def _print_completions(
    console: rich.console.Console,
    events: list[StageCompleted],
    *,
    total: int,
    max_name_width: int,
    stage_logs: dict[str, list[str]] | None = None,
) -> None:
    """Print buffered completions sorted by index with skip collapsing."""
    if not events:
        return

    sorted_events = sorted(events, key=lambda e: e["index"])
    logs = stage_logs or {}

    i = 0
    while i < len(sorted_events):
        event = sorted_events[i]
        category = _categorize(event)
        if category in _SKIP_CATEGORIES:
            # Collect consecutive skips of the SAME category
            same_cat_group = [event]
            j = i + 1
            while j < len(sorted_events) and _categorize(sorted_events[j]) == category:
                same_cat_group.append(sorted_events[j])
                j += 1

            if len(same_cat_group) >= 2:
                line = _format_skip_group_line(
                    start_index=same_cat_group[0]["index"],
                    end_index=same_cat_group[-1]["index"],
                    total=total,
                    count=len(same_cat_group),
                    category=category,
                )
                console.print(line)
            else:
                # Single skip — show individually (already dimmed)
                line = _format_stage_line(
                    index=event["index"],
                    total=total,
                    stage=event["stage"],
                    category=category,
                    duration_ms=event["duration_ms"],
                    name_width=max_name_width,
                )
                console.print(line)
            i = j
        else:
            line = _format_stage_line(
                index=event["index"],
                total=total,
                stage=event["stage"],
                category=category,
                duration_ms=event["duration_ms"],
                name_width=max_name_width,
            )
            console.print(line)
            if category == DisplayCategory.FAILED:
                # Show captured stage output for failed stages
                stage_log = logs.get(event["stage"], [])
                if stage_log:
                    for log_line in stage_log:
                        escaped = rich.markup.escape(log_line)
                        console.print(f"          [dim]{escaped}[/dim]")
                if event["reason"]:
                    for detail in _format_error_detail(event["reason"], total=total):
                        console.print(detail)
            i += 1


_MAX_LOG_LINES_PER_STAGE = 50


class StaticConsoleSink:
    """Pipe/CI sink: buffers completions, prints sorted report on close."""

    _console: rich.console.Console
    _show_output: bool
    _max_name_width: int
    _completion_buffer: list[StageCompleted]
    _stage_logs: dict[str, list[str]]
    _total: int

    def __init__(self, *, console: rich.console.Console, show_output: bool = False) -> None:
        self._console = console
        self._show_output = show_output
        self._max_name_width = 0
        self._completion_buffer = list[StageCompleted]()
        self._stage_logs = dict[str, list[str]]()
        self._total = 0

    async def handle(self, event: OutputEvent) -> None:
        """Handle output event."""
        match event["type"]:
            case "stage_started":
                self._total = event["total"]
                self._max_name_width = max(self._max_name_width, len(event["stage"]))
            case "stage_completed":
                self._total = event["total"]
                self._max_name_width = max(self._max_name_width, len(event["stage"]))
                self._completion_buffer.append(event)
            case "log_line":
                # Always buffer for failed-stage output; stream if --show-output
                stage = event["stage"]
                buf = self._stage_logs.setdefault(stage, [])
                if len(buf) < _MAX_LOG_LINES_PER_STAGE:
                    buf.append(event["line"])
                if self._show_output:
                    line = rich.markup.escape(event["line"])
                    if event["is_stderr"]:
                        self._console.print(f"[dim red]\\[{stage}] {line}[/dim red]")
                    else:
                        self._console.print(f"[dim]\\[{stage}] {line}[/dim]")
            case "engine_diagnostic":
                msg = rich.markup.escape(event["message"])
                detail = rich.markup.escape(event["detail"]) if event["detail"] else ""
                self._console.print(f"[yellow bold]⚠ Engine diagnostic:[/yellow bold] {msg}")
                if detail:
                    self._console.print(f"  [dim]{detail}[/dim]")
            case _:
                pass

    async def close(self) -> None:
        """Print sorted completion report."""
        _print_completions(
            self._console,
            self._completion_buffer,
            total=self._total,
            max_name_width=self._max_name_width,
            stage_logs=self._stage_logs,
        )


class _LiveRenderable:
    _sink: LiveConsoleSink

    def __init__(self, sink: LiveConsoleSink) -> None:
        self._sink = sink

    def __rich_console__(
        self,
        console: rich.console.Console,
        options: rich.console.ConsoleOptions,
    ) -> rich.console.RenderResult:
        yield self._sink._build_live_group()  # pyright: ignore[reportPrivateUsage] - internal helper class


class LiveConsoleSink:
    """TTY sink with live pinned status area. Completions print on close."""

    _console: rich.console.Console
    _show_output: bool
    _running: dict[str, float]
    _completed_count: int
    _category_counts: dict[DisplayCategory, int]
    _max_name_width: int
    _completion_buffer: list[StageCompleted]
    _stage_logs: dict[str, list[str]]
    _total: int
    _live: rich.live.Live | None

    _RECENT_COMPLETIONS_LIMIT: int = 5

    def __init__(self, *, console: rich.console.Console, show_output: bool = False) -> None:
        self._console = console
        self._show_output = show_output
        self._running = dict[str, float]()
        self._completed_count = 0
        self._category_counts = dict[DisplayCategory, int]()
        self._max_name_width = 0
        self._completion_buffer = list[StageCompleted]()
        self._stage_logs = dict[str, list[str]]()
        self._total = 0
        self._live = None

    def _build_live_renderable(self) -> rich.console.RenderableType:
        return _LiveRenderable(self)

    def _build_live_group(self) -> rich.console.Group:
        renderables: list[rich.console.RenderableType] = []
        display_width = max(1, min(self._max_name_width, _MAX_NAME_WIDTH))

        # Recent completions (last N)
        recent = self._completion_buffer[-self._RECENT_COMPLETIONS_LIMIT :]
        for event in recent:
            category = _categorize(event)
            symbol = _CATEGORY_SYMBOL.get(category, "[dim]?[/dim]")
            word = _CATEGORY_WORD.get(category, f"[dim]{category.value}[/dim]")
            stage_name = event["stage"]
            if len(stage_name) > display_width:
                stage_name = stage_name[: display_width - 1] + "…"
            padded = stage_name.ljust(display_width)
            renderables.append(f"  {symbol} {padded}  {word}")

        # Currently running stages
        running_sorted = sorted(self._running.items(), key=lambda item: item[1])
        for stage, started_at in running_sorted:
            stage_name = f"{stage[: display_width - 1]}…" if len(stage) > display_width else stage
            padded_name = stage_name.ljust(display_width)
            elapsed_s = time.monotonic() - started_at
            renderables.append(f"  [green]▶[/green] {padded_name}  running…  {elapsed_s:.0f}s")

        # Progress bar
        total = self._total
        completed = self._completed_count
        fraction = min(1.0, completed / total) if total > 0 else 0.0
        bar_width = 40
        filled = int(bar_width * fraction)
        bar = "━" * filled + "─" * (bar_width - filled)
        renderables.append(f"  {bar} {completed}/{total} ({fraction * 100:.0f}%)")

        # Summary counts using display categories
        counts = self._category_counts
        summary = rich.text.Text("  ")
        ran = counts.get(DisplayCategory.SUCCESS, 0)
        cached = counts.get(DisplayCategory.CACHED, 0)
        blocked = counts.get(DisplayCategory.BLOCKED, 0)
        cancelled = counts.get(DisplayCategory.CANCELLED, 0)
        failed = counts.get(DisplayCategory.FAILED, 0)
        parts: list[tuple[str, str]] = []
        if ran:
            parts.append((f"{ran} ran", "green"))
        if cached:
            parts.append((f"{cached} cached", "yellow"))
        if blocked:
            parts.append((f"{blocked} blocked", "red"))
        if cancelled:
            parts.append((f"{cancelled} cancelled", "yellow"))
        if failed:
            parts.append((f"{failed} failed", "bold red"))
        for i, (text, style) in enumerate(parts):
            if i > 0:
                summary.append(" · ")
            summary.append(text, style=style)
        if parts:
            renderables.append(summary)
        return rich.console.Group(*renderables)

    def _ensure_live(self) -> None:
        if self._live is not None:
            return
        self._live = rich.live.Live(
            self._build_live_renderable(),
            console=self._console,
            refresh_per_second=1,
        )
        self._live.start()

    def _update_live(self) -> None:
        if self._live is None:
            return
        self._live.update(self._build_live_renderable())

    async def handle(self, event: OutputEvent) -> None:
        match event["type"]:
            case "stage_started":
                self._running[event["stage"]] = time.monotonic()
                self._total = event["total"]
                self._max_name_width = max(self._max_name_width, len(event["stage"]))
                self._ensure_live()
                self._update_live()
            case "stage_completed":
                _ = self._running.pop(event["stage"], None)
                self._total = event["total"]
                self._max_name_width = max(self._max_name_width, len(event["stage"]))
                self._completed_count += 1
                category = _categorize(event)
                self._category_counts[category] = self._category_counts.get(category, 0) + 1
                self._completion_buffer.append(event)
                self._update_live()
            case "log_line":
                stage = event["stage"]
                buf = self._stage_logs.setdefault(stage, [])
                if len(buf) < _MAX_LOG_LINES_PER_STAGE:
                    buf.append(event["line"])
                if self._show_output:
                    line = rich.markup.escape(event["line"])
                    if event["is_stderr"]:
                        self._console.print(f"[dim red]\\[{stage}] {line}[/dim red]")
                    else:
                        self._console.print(f"[dim]\\[{stage}] {line}[/dim]")
            case "engine_diagnostic":
                msg = rich.markup.escape(event["message"])
                detail = rich.markup.escape(event["detail"]) if event["detail"] else ""
                self._console.print(f"[yellow bold]⚠ Engine diagnostic:[/yellow bold] {msg}")
                if detail:
                    self._console.print(f"  [dim]{detail}[/dim]")
            case _:
                pass

    async def close(self) -> None:
        if self._live is not None:
            self._live.update(rich.text.Text(""))
            self._live.stop()
        _print_completions(
            self._console,
            self._completion_buffer,
            total=self._total,
            max_name_width=self._max_name_width,
            stage_logs=self._stage_logs,
        )


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
