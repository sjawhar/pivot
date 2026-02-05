# `--show-output` Flag Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `--show-output` flag to stream stage logs in non-TUI mode

**Architecture:** Extend `ConsoleSink` to handle `log_line` events when enabled, add flag to both `pivot run` and `pivot repro` with inline mutual exclusion validation.

**Tech Stack:** Click CLI, Rich console, async sinks

---

## Task 1: Extend ConsoleSink to Handle Log Lines

**Files:**
- Modify: `src/pivot/engine/sinks.py:73-105`
- Test: `tests/engine/test_sinks.py`

**Step 1: Write the failing tests**

Add to `tests/engine/test_sinks.py`:

```python
async def test_console_sink_handles_log_line_when_show_output_enabled() -> None:
    """ConsoleSink prints log lines when show_output=True."""
    from io import StringIO

    from rich.console import Console

    from pivot.engine.sinks import ConsoleSink
    from pivot.engine.types import LogLine

    output = StringIO()
    console = Console(file=output, force_terminal=True)
    sink = ConsoleSink(console=console, show_output=True)

    event = LogLine(
        type="log_line",
        stage="train",
        line="Processing batch 1...",
        is_stderr=False,
    )
    await sink.handle(event)

    result = output.getvalue()
    assert "[train]" in result
    assert "Processing batch 1..." in result


async def test_console_sink_stderr_line_contains_content() -> None:
    """ConsoleSink prints stderr lines with stage prefix."""
    from io import StringIO

    from rich.console import Console

    from pivot.engine.sinks import ConsoleSink
    from pivot.engine.types import LogLine

    output = StringIO()
    console = Console(file=output, force_terminal=True)
    sink = ConsoleSink(console=console, show_output=True)

    event = LogLine(
        type="log_line",
        stage="train",
        line="Warning: GPU not available",
        is_stderr=True,
    )
    await sink.handle(event)

    result = output.getvalue()
    assert "[train]" in result
    assert "Warning: GPU not available" in result
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/engine/test_sinks.py::test_console_sink_handles_log_line_when_show_output_enabled tests/engine/test_sinks.py::test_console_sink_stderr_line_contains_content -v`
Expected: FAIL - `show_output` parameter doesn't exist

**Step 3: Implement ConsoleSink log line handling**

Edit `src/pivot/engine/sinks.py`. Update the `ConsoleSink` class to add `show_output` parameter and handle `log_line` events:

```python
class ConsoleSink:
    """Async sink that prints stage events to console."""

    _console: rich.console.Console
    _show_output: bool

    def __init__(
        self, *, console: rich.console.Console, show_output: bool = False
    ) -> None:
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
                            for line in event["reason"].rstrip().split("\n"):
                                self._console.print(f"    [dim]{line}[/dim]")
            case "log_line" if self._show_output:
                stage = event["stage"]
                line = event["line"]
                if event["is_stderr"]:
                    self._console.print(f"[red]\\[{stage}][/red] [red]{line}[/red]")
                else:
                    self._console.print(f"\\[{stage}] {line}")
            case _:
                pass  # Ignore other events

    async def close(self) -> None:
        """No cleanup needed."""
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/engine/test_sinks.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
jj describe -m "feat(sinks): add show_output parameter to ConsoleSink for log streaming"
```

---

## Task 2: Add show_output Parameter to configure_output_sink

**Files:**
- Modify: `src/pivot/cli/_run_common.py:232-258`

**Step 1: Update configure_output_sink signature**

Edit `src/pivot/cli/_run_common.py`. Add `show_output` parameter:

```python
def configure_output_sink(
    eng: engine.Engine,
    *,
    quiet: bool,
    as_json: bool,
    tui: bool,
    app: MessagePoster | None,
    run_id: str | None,
    use_console: bool,
    jsonl_callback: Callable[[dict[str, object]], None] | None,
    show_output: bool = False,
) -> None:
    """Configure output sinks based on display mode."""
    import rich.console

    # JSON sink is always added when as_json=True, regardless of quiet mode
    if as_json and jsonl_callback:
        eng.add_sink(JsonlSink(callback=jsonl_callback))
        return

    if quiet:
        return

    if tui and app and run_id:
        eng.add_sink(sinks.TuiSink(app=app, run_id=run_id))
    elif use_console:
        eng.add_sink(sinks.ConsoleSink(console=rich.console.Console(), show_output=show_output))
```

**Step 2: Run existing tests to verify no regressions**

Run: `uv run pytest tests/cli/ -v -k "run_common or repro or run"`
Expected: All tests PASS

**Step 3: Commit**

```bash
jj describe -m "feat(cli): add show_output parameter to configure_output_sink"
```

---

## Task 3: Add --show-output Flag to pivot repro

**Files:**
- Modify: `src/pivot/cli/repro.py`
- Test: `tests/cli/test_repro.py`

**Step 1: Write the failing tests**

Add to `tests/cli/test_repro.py`:

```python
def test_repro_show_output_mutually_exclusive_with_tui(runner: CliRunner) -> None:
    """--show-output and --tui are mutually exclusive."""
    with runner.isolated_filesystem():
        pathlib.Path("pivot.yaml").write_text("stages: []")

        result = runner.invoke(cli, ["repro", "--show-output", "--tui"])

        assert result.exit_code != 0
        assert "--show-output and --tui are mutually exclusive" in result.output


def test_repro_show_output_mutually_exclusive_with_json(runner: CliRunner) -> None:
    """--show-output and --json are mutually exclusive."""
    with runner.isolated_filesystem():
        pathlib.Path("pivot.yaml").write_text("stages: []")

        result = runner.invoke(cli, ["repro", "--show-output", "--json"])

        assert result.exit_code != 0
        assert "--show-output and --json are mutually exclusive" in result.output


def test_repro_show_output_mutually_exclusive_with_quiet(runner: CliRunner) -> None:
    """--show-output and --quiet are mutually exclusive."""
    with runner.isolated_filesystem():
        pathlib.Path("pivot.yaml").write_text("stages: []")

        result = runner.invoke(cli, ["--quiet", "repro", "--show-output"])

        assert result.exit_code != 0
        assert "--show-output and --quiet are mutually exclusive" in result.output
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/cli/test_repro.py::test_repro_show_output_mutually_exclusive_with_tui tests/cli/test_repro.py::test_repro_show_output_mutually_exclusive_with_json tests/cli/test_repro.py::test_repro_show_output_mutually_exclusive_with_quiet -v`
Expected: FAIL - `--show-output` flag doesn't exist

**Step 3: Add --show-output flag to repro command**

Edit `src/pivot/cli/repro.py`:

1. Add the option after `--json` (around line 735):
```python
@click.option("--json", "as_json", is_flag=True, help="Output results as JSON")
@click.option(
    "--show-output",
    is_flag=True,
    help="Stream stage output (stdout/stderr) to terminal",
)
```

2. Add `show_output: bool` parameter to function signature (after `as_json`).

3. Add inline validation after the existing `--tui and --json` check (around line 817):
```python
    # Validate --tui and --json are mutually exclusive
    if tui_flag and as_json:
        raise click.ClickException("--tui and --json are mutually exclusive")

    # Validate --show-output combinations
    if show_output and tui_flag:
        raise click.ClickException("--show-output and --tui are mutually exclusive")
    if show_output and as_json:
        raise click.ClickException("--show-output and --json are mutually exclusive")
    if show_output and quiet:
        raise click.ClickException("--show-output and --quiet are mutually exclusive")
```

4. Pass `show_output` to `_run_pipeline` (add to call around line 854).

5. Update `_run_pipeline` signature to accept `show_output: bool`.

6. Pass `show_output` to both `_run_watch_mode` and `_run_oneshot_mode`.

7. Update `_run_watch_mode` signature and pass to `configure_output_sink` in the non-TUI branch.

8. Update `_run_oneshot_mode` signature and pass to `configure_output_sink` in the non-TUI branch.

9. Update `_run_serve_mode` to accept `show_output: bool` and pass to `ConsoleSink`:
```python
# In _run_serve_mode, around line 542
if not quiet:
    serve_console = rich.console.Console()
    eng.add_sink(sinks.ConsoleSink(console=serve_console, show_output=show_output))
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/cli/test_repro.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
jj describe -m "feat(cli): add --show-output flag to pivot repro"
```

---

## Task 4: Add --show-output Flag to pivot run

**Files:**
- Modify: `src/pivot/cli/run.py`
- Test: `tests/cli/test_run.py`

**Step 1: Write the failing tests**

Add to `tests/cli/test_run.py`:

```python
def test_run_show_output_mutually_exclusive_with_tui(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """--show-output and --tui are mutually exclusive."""
    (tmp_path / "pivot.yaml").write_text(
        """
stages:
  - name: test
    cmd: python -c "print('hello')"
"""
    )

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["run", "test", "--show-output", "--tui"])

        assert result.exit_code != 0
        assert "--show-output and --tui are mutually exclusive" in result.output


def test_run_show_output_mutually_exclusive_with_json(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """--show-output and --json are mutually exclusive."""
    (tmp_path / "pivot.yaml").write_text(
        """
stages:
  - name: test
    cmd: python -c "print('hello')"
"""
    )

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["run", "test", "--show-output", "--json"])

        assert result.exit_code != 0
        assert "--show-output and --json are mutually exclusive" in result.output


def test_run_show_output_mutually_exclusive_with_quiet(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """--show-output and --quiet are mutually exclusive."""
    (tmp_path / "pivot.yaml").write_text(
        """
stages:
  - name: test
    cmd: python -c "print('hello')"
"""
    )

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["--quiet", "run", "test", "--show-output"])

        assert result.exit_code != 0
        assert "--show-output and --quiet are mutually exclusive" in result.output
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/cli/test_run.py::test_run_show_output_mutually_exclusive_with_tui tests/cli/test_run.py::test_run_show_output_mutually_exclusive_with_json tests/cli/test_run.py::test_run_show_output_mutually_exclusive_with_quiet -v`
Expected: FAIL - `--show-output` flag doesn't exist

**Step 3: Add --show-output flag to run command**

Edit `src/pivot/cli/run.py`:

1. Add the option after `--json` (around line 324):
```python
@click.option("--json", "as_json", is_flag=True, help="Output results as JSON")
@click.option(
    "--show-output",
    is_flag=True,
    help="Stream stage output (stdout/stderr) to terminal",
)
```

2. Add `show_output: bool` parameter to function signature.

3. Add inline validation after `--tui and --json` check (around line 387):
```python
    # Validate --tui and --json are mutually exclusive
    if tui_flag and as_json:
        raise click.ClickException("--tui and --json are mutually exclusive")

    # Validate --show-output combinations
    if show_output and tui_flag:
        raise click.ClickException("--show-output and --tui are mutually exclusive")
    if show_output and as_json:
        raise click.ClickException("--show-output and --json are mutually exclusive")
    if show_output and quiet:
        raise click.ClickException("--show-output and --quiet are mutually exclusive")
```

4. Pass `show_output` to `_run_plain_mode`.

5. Update `_run_plain_mode` signature to accept `show_output: bool = False`.

6. Pass to `configure_output_sink`:
```python
            _run_common.configure_output_sink(
                eng,
                quiet=quiet,
                as_json=False,
                tui=False,
                app=None,
                run_id=None,
                use_console=console is not None,
                jsonl_callback=None,
                show_output=show_output,
            )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/cli/test_run.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
jj describe -m "feat(cli): add --show-output flag to pivot run"
```

---

## Task 5: Run Full Test Suite and Quality Checks

**Step 1: Run all tests**

Run: `uv run pytest tests/ -n auto`
Expected: All tests PASS

**Step 2: Run type checker**

Run: `uv run basedpyright`
Expected: No errors

**Step 3: Run linter and formatter**

Run: `uv run ruff format . && uv run ruff check .`
Expected: No issues

**Step 4: Final commit**

```bash
jj describe -m "feat(cli): add --show-output flag to stream stage logs

Add --show-output flag to both pivot run and pivot repro commands:
- Streams stage stdout/stderr to terminal with [stage] prefix
- Colors stderr lines red for visibility
- Mutually exclusive with --tui, --json, and --quiet"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Extend ConsoleSink with show_output | `sinks.py`, `test_sinks.py` |
| 2 | Add show_output to configure_output_sink | `_run_common.py` |
| 3 | Add --show-output to pivot repro | `repro.py`, `test_repro.py` |
| 4 | Add --show-output to pivot run | `run.py`, `test_run.py` |
| 5 | Run full test suite and quality checks | — |

**Changes from original plan based on review feedback:**
- Removed failure summary feature (unnecessary scope creep, terminal has scrollback)
- Inlined validation instead of separate function (follows existing pattern)
- Simplified ANSI test to just check content (not implementation details)
- Added `show_output` to `_run_serve_mode` (missing code path)
- Added `--quiet --show-output` mutual exclusion
- Reduced from 7 tasks to 5
