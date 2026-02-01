# PR #315 Regression Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 12 regressions identified during red-team review of PR #315 (engine overhaul)

**Architecture:** Three categories of fixes: (1) Critical missing function calls in CLI commands, (2) Parameter propagation through watch mode, (3) Minor inconsistencies

**Tech Stack:** Python 3.13+, Click CLI, anyio async

---

## Task 1: Add `ensure_stages_registered()` to `pivot data` Commands [CRITICAL]

**Files:**
- Modify: `src/pivot/cli/data.py:39-48` (data_diff function)
- Modify: `src/pivot/cli/data.py:161-168` (data_get function)
- Test: `tests/cli/test_data.py`

**Step 1: Write the failing test**

Add to `tests/cli/test_data.py`:

```python
def test_data_diff_without_prior_run_discovers_pipeline(
    tmp_path: pathlib.Path,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot data diff discovers pipeline without needing prior command."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    # Create minimal pivot.yaml with a stage
    (tmp_path / "pivot.yaml").write_text("""
stages:
  process:
    func: pipeline:process
    deps: [input.csv]
    outs: [output.csv]
""")
    (tmp_path / "pipeline.py").write_text("""
import pathlib
def process(): pass
""")
    (tmp_path / ".pivot").mkdir()
    (tmp_path / "input.csv").write_text("a,b\n1,2")
    (tmp_path / "output.csv").write_text("x,y\n3,4")

    result = runner.invoke(cli.cli, ["data", "diff", "output.csv"])

    # Should not raise NoPipelineError
    assert "No pipeline" not in result.output
    assert result.exit_code == 0 or "No data file changes" in result.output


def test_data_get_without_prior_run_discovers_pipeline(
    tmp_path: pathlib.Path,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot data get discovers pipeline without needing prior command."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    # Create minimal pivot.yaml
    (tmp_path / "pivot.yaml").write_text("""
stages:
  process:
    func: pipeline:process
    deps: [input.csv]
    outs: [output.csv]
""")
    (tmp_path / "pipeline.py").write_text("def process(): pass")
    (tmp_path / ".pivot").mkdir()

    # Create a fake revision (just need git to exist)
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=tmp_path, capture_output=True)

    result = runner.invoke(cli.cli, ["data", "get", "--rev", "HEAD", "output.csv"])

    # Should not raise NoPipelineError - may fail for other reasons but not discovery
    assert "No pipeline" not in result.output
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_data.py::test_data_diff_without_prior_run_discovers_pipeline tests/cli/test_data.py::test_data_get_without_prior_run_discovers_pipeline -v`
Expected: FAIL with `NoPipelineError` or similar

**Step 3: Write minimal implementation**

Edit `src/pivot/cli/data.py`:

```python
# At top of file, add import:
from pivot.cli.run import ensure_stages_registered

# In data_diff function, after decorators, add as first line of body:
@data.command("diff")
@click.argument("targets", nargs=-1, required=True)
# ... other decorators ...
@cli_decorators.with_error_handling
def data_diff(
    targets: tuple[str, ...],
    # ... params ...
) -> None:
    """Compare data files in workspace against git HEAD."""
    ensure_stages_registered()  # ADD THIS LINE
    from pivot.show import data as data_module
    # ... rest of function

# In data_get function, after decorators, add as first line of body:
@data.command("get")
@click.argument("targets", nargs=-1, required=True)
# ... other decorators ...
@cli_decorators.with_error_handling
def data_get(
    targets: tuple[str, ...],
    # ... params ...
) -> None:
    """Retrieve files or stage outputs from a specific git revision."""
    ensure_stages_registered()  # ADD THIS LINE
    cache_dir = config.get_cache_dir()
    # ... rest of function
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_data.py::test_data_diff_without_prior_run_discovers_pipeline tests/cli/test_data.py::test_data_get_without_prior_run_discovers_pipeline -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "fix(cli): add ensure_stages_registered to data diff/get commands

These commands were missing pipeline discovery, causing NoPipelineError
when run without a prior command that triggered discovery."
```

---

## Task 2: Add `ensure_stages_registered()` to `pivot params` Commands [CRITICAL]

**Files:**
- Modify: `src/pivot/cli/params.py:27-48` (params_show function)
- Modify: `src/pivot/cli/params.py:59-95` (params_diff function)
- Test: `tests/cli/test_params.py`

**Step 1: Write the failing test**

Add to `tests/cli/test_params.py` (create if doesn't exist):

```python
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from pivot import cli

if TYPE_CHECKING:
    import click.testing
    import pytest


def test_params_show_without_prior_run_discovers_pipeline(
    tmp_path: pathlib.Path,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot params show discovers pipeline without needing prior command."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    # Create minimal pivot.yaml
    (tmp_path / "pivot.yaml").write_text("""
stages:
  process:
    func: pipeline:process
""")
    (tmp_path / "pipeline.py").write_text("def process(): pass")
    (tmp_path / ".pivot").mkdir()

    result = runner.invoke(cli.cli, ["params", "show"])

    # Should not raise NoPipelineError
    assert "No pipeline" not in result.output
    assert result.exit_code == 0


def test_params_diff_without_prior_run_discovers_pipeline(
    tmp_path: pathlib.Path,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot params diff discovers pipeline without needing prior command."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    # Create minimal pivot.yaml
    (tmp_path / "pivot.yaml").write_text("""
stages:
  process:
    func: pipeline:process
""")
    (tmp_path / "pipeline.py").write_text("def process(): pass")
    (tmp_path / ".pivot").mkdir()

    # Initialize git repo
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=tmp_path, capture_output=True)

    result = runner.invoke(cli.cli, ["params", "diff"])

    # Should not raise NoPipelineError
    assert "No pipeline" not in result.output
    assert result.exit_code == 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_params.py::test_params_show_without_prior_run_discovers_pipeline tests/cli/test_params.py::test_params_diff_without_prior_run_discovers_pipeline -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Edit `src/pivot/cli/params.py`:

```python
# At top of file, add import:
from pivot.cli.run import ensure_stages_registered

# In params_show function, add as first line of body:
@params.command("show")
# ... decorators ...
@cli_decorators.with_error_handling
def params_show(
    stages: tuple[str, ...],
    output_format: OutputFormat | None,
    precision: int | None,
) -> None:
    """Display current parameter values."""
    ensure_stages_registered()  # ADD THIS LINE
    precision = precision if precision is not None else config.get_display_precision()
    # ... rest of function

# In params_diff function, add as first line of body:
@params.command("diff")
# ... decorators ...
@cli_decorators.with_error_handling
def params_diff(
    stages: tuple[str, ...],
    output_format: OutputFormat | None,
    precision: int | None,
) -> None:
    """Compare workspace parameters against git HEAD."""
    ensure_stages_registered()  # ADD THIS LINE
    precision = precision if precision is not None else config.get_display_precision()
    # ... rest of function
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_params.py::test_params_show_without_prior_run_discovers_pipeline tests/cli/test_params.py::test_params_diff_without_prior_run_discovers_pipeline -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "fix(cli): add ensure_stages_registered to params show/diff commands

These commands were missing pipeline discovery, causing NoPipelineError
when run without a prior command that triggered discovery."
```

---

## Task 3: Pass `--no-cache` Flag to Watch Mode [HIGH]

**Files:**
- Modify: `src/pivot/cli/run.py:277-295` (_run_watch_mode signature and calls)
- Modify: `src/pivot/cli/run.py:236-253` (call site in _run_pipeline)
- Test: `tests/cli/test_cli.py`

**Step 1: Write the failing test**

Add to `tests/cli/test_cli.py`:

```python
def test_run_watch_mode_respects_no_cache_flag(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """--watch --no-cache should pass no_cache to watch mode execution."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_process, name="process")

    # Mock the watch source to capture the parameters passed to OneShotSource
    captured_params = {}
    original_oneshot = engine_sources.OneShotSource

    def capture_oneshot(*args, **kwargs):
        captured_params.update(kwargs)
        return original_oneshot(*args, **kwargs)

    mocker.patch.object(engine_sources, "OneShotSource", capture_oneshot)

    # Run with --watch --no-cache --force (force triggers initial run)
    # Use short timeout to exit watch mode quickly
    import threading
    import time

    def stop_after_delay():
        time.sleep(0.5)
        # Send interrupt - in real test we'd need to signal the process

    result = runner.invoke(cli.cli, ["run", "--watch", "--no-cache", "--force", "--serve"], catch_exceptions=False)

    # Verify no_cache was passed to OneShotSource
    assert captured_params.get("no_cache") is True, f"no_cache not passed: {captured_params}"
```

Note: This test is conceptual. The actual test may need adjustment based on how watch mode can be tested (mocking the filesystem source, etc.)

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_cli.py::test_run_watch_mode_respects_no_cache_flag -v`
Expected: FAIL - no_cache not passed

**Step 3: Write minimal implementation**

Edit `src/pivot/cli/run.py`:

1. Update `_run_watch_mode` signature to accept `no_cache`:

```python
def _run_watch_mode(  # noqa: PLR0913 - many params needed for different modes
    stages_list: list[str] | None,
    execution_order: list[str],
    graph: nx.DiGraph[str],
    *,
    quiet: bool,
    tui: bool,
    as_json: bool,
    debounce: int,
    tui_log: pathlib.Path | None,
    on_error: OnError,
    serve: bool,
    force: bool,
    run_id: str | None,
    console: tui_console.Console | None,
    jsonl_callback: Callable[[dict[str, object]], None] | None,
    cancel_event: threading.Event | None,
    no_commit: bool,
    no_cache: bool,  # ADD THIS PARAMETER
) -> None:
```

2. Update the call in `_run_pipeline` to pass `no_cache`:

```python
    if watch:
        return _run_watch_mode(
            stages_list=stages_list,
            execution_order=execution_order,
            graph=graph,
            quiet=quiet,
            tui=tui,
            as_json=as_json,
            debounce=debounce,
            tui_log=tui_log,
            on_error=on_error,
            serve=serve,
            force=force,
            run_id=run_id,
            console=console,
            jsonl_callback=jsonl_callback,
            cancel_event=cancel_event,
            no_commit=no_commit,
            no_cache=no_cache,  # ADD THIS LINE
        )
```

3. Update `_configure_watch_sources` to accept and use `no_cache`:

```python
def _configure_watch_sources(
    eng: engine.Engine,
    watch_paths: list[pathlib.Path],
    debounce: int,
    *,
    force: bool,
    stages: list[str] | None,
    no_cache: bool = False,  # ADD THIS PARAMETER
    no_commit: bool = False,  # ADD THIS PARAMETER
    on_error: OnError = OnError.FAIL,  # ADD THIS PARAMETER
) -> None:
    """Configure sources for watch mode."""
    eng.add_source(engine_sources.FilesystemSource(watch_paths=watch_paths, debounce_ms=debounce))
    if force:
        eng.add_source(
            engine_sources.OneShotSource(
                stages=stages,
                force=True,
                reason="watch:initial:forced",
                no_cache=no_cache,  # ADD THIS
                no_commit=no_commit,  # ADD THIS
                on_error=on_error,  # ADD THIS
            )
        )
```

4. Update all call sites of `_configure_watch_sources` to pass the new params.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_cli.py::test_run_watch_mode_respects_no_cache_flag -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "fix(cli): pass --no-cache flag to watch mode initial execution

The --no-cache flag was accepted but ignored in watch mode."
```

---

## Task 4: Store Orchestration Params in Engine for Watch Re-runs [HIGH]

**Files:**
- Modify: `src/pivot/engine/engine.py:100-131` (Engine.__init__)
- Modify: `src/pivot/engine/engine.py:1127-1148` (_execute_affected_stages)
- Modify: `src/pivot/engine/engine.py:285-313` (_handle_run_requested)
- Test: `tests/engine/test_engine.py`

**Step 1: Write the failing test**

Add to `tests/engine/test_engine.py`:

```python
@pytest.mark.anyio
async def test_engine_watch_reruns_use_stored_orchestration_params(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Watch mode re-runs should use the same no_commit/no_cache as initial run."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".pivot").mkdir()
    (tmp_path / "input.txt").write_text("data")

    # Create test pipeline
    from pivot.pipeline.pipeline import Pipeline
    pipeline = Pipeline("test", root=tmp_path)

    # Track what params were used in orchestration
    orchestration_calls = []

    original_orchestrate = engine.Engine._orchestrate_execution

    async def tracking_orchestrate(self, **kwargs):
        orchestration_calls.append({
            "no_commit": kwargs.get("no_commit"),
            "no_cache": kwargs.get("no_cache"),
        })
        # Don't actually run
        return {}

    monkeypatch.setattr(engine.Engine, "_orchestrate_execution", tracking_orchestrate)

    async with engine.Engine(pipeline=pipeline) as eng:
        # Simulate initial run with no_commit=True, no_cache=True
        initial_event = RunRequested(
            type="run_requested",
            stages=None,
            force=False,
            reason="test",
            single_stage=False,
            parallel=True,
            max_workers=None,
            no_commit=True,
            no_cache=True,
            on_error=OnError.FAIL,
            cache_dir=None,
            allow_uncached_incremental=False,
            checkout_missing=False,
        )
        await eng._handle_run_requested(initial_event)

        # Simulate a watch re-run via _execute_affected_stages
        await eng._execute_affected_stages(["some_stage"])

    # Both calls should have the same params
    assert len(orchestration_calls) == 2
    assert orchestration_calls[1]["no_commit"] is True, "Watch re-run should use stored no_commit"
    assert orchestration_calls[1]["no_cache"] is True, "Watch re-run should use stored no_cache"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_engine.py::test_engine_watch_reruns_use_stored_orchestration_params -v`
Expected: FAIL - re-run uses False for both

**Step 3: Write minimal implementation**

Edit `src/pivot/engine/engine.py`:

1. Add instance variables to store orchestration params in `__init__`:

```python
def __init__(self, *, pipeline: Pipeline | None = None) -> None:
    """Initialize the async engine in IDLE state."""
    # ... existing code ...

    # Stored orchestration params for watch mode re-runs
    self._stored_no_commit: bool = False
    self._stored_no_cache: bool = False
    self._stored_on_error: OnError = OnError.FAIL
```

2. Store params in `_handle_run_requested`:

```python
async def _handle_run_requested(self, event: RunRequested) -> None:
    """Handle a RunRequested event by executing stages."""
    # Store orchestration params for watch mode re-runs
    self._stored_no_commit = event["no_commit"]
    self._stored_no_cache = event["no_cache"]
    self._stored_on_error = event["on_error"]

    # ... rest of existing code ...
```

3. Use stored params in `_execute_affected_stages`:

```python
async def _execute_affected_stages(self, stages: list[str]) -> None:
    """Execute the specified stages."""
    self._cancel_event = anyio.Event()  # Reset by creating new event

    self._state = EngineState.ACTIVE
    await self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.ACTIVE))

    try:
        await self._orchestrate_execution(
            stages=stages,
            force=False,
            single_stage=False,
            parallel=True,
            max_workers=None,
            no_commit=self._stored_no_commit,  # USE STORED VALUE
            no_cache=self._stored_no_cache,    # USE STORED VALUE
            on_error=self._stored_on_error,    # USE STORED VALUE (was hardcoded KEEP_GOING)
            cache_dir=None,
        )
    finally:
        self._state = EngineState.IDLE
        await self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE))
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/engine/test_engine.py::test_engine_watch_reruns_use_stored_orchestration_params -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "fix(engine): use stored orchestration params for watch mode re-runs

Watch mode re-runs were hardcoding no_commit=False, no_cache=False,
ignoring CLI flags. Now stores params from initial RunRequested event."
```

---

## Task 5: Fix `--quiet` Flag in `dry_run_cmd` [MEDIUM]

**Files:**
- Modify: `src/pivot/cli/run.py:1016-1088` (dry_run_cmd function)
- Test: `tests/cli/test_cli.py`

**Step 1: Write the failing test**

Add to `tests/cli/test_cli.py`:

```python
def test_dry_run_quiet_suppresses_output(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot run --dry-run --quiet should suppress output."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_process, name="process")

    result = runner.invoke(cli.cli, ["--quiet", "run", "--dry-run"])

    # With --quiet, should have no output
    assert result.exit_code == 0
    assert result.output.strip() == "", f"Expected empty output with --quiet, got: {result.output}"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_cli.py::test_dry_run_quiet_suppresses_output -v`
Expected: FAIL - output is not empty

**Step 3: Write minimal implementation**

Edit `src/pivot/cli/run.py`:

```python
@cli_decorators.pivot_command("dry-run")
@click.argument("stages", nargs=-1, shell_complete=completion.complete_stages)
@click.option(
    "--single-stage",
    "-s",
    is_flag=True,
    help="Run only the specified stages (in provided order), not their dependencies",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Show what would run if forced",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--allow-missing", is_flag=True, help="Allow missing dep files if tracked")
@click.pass_context  # ADD THIS DECORATOR
def dry_run_cmd(
    ctx: click.Context,  # ADD THIS PARAMETER
    stages: tuple[str, ...],
    single_stage: bool,
    force: bool,
    as_json: bool,
    allow_missing: bool,
) -> None:
    """Show what would run without executing."""
    from pivot import status as status_mod
    from pivot.engine import graph as engine_graph

    # Get quiet flag from CLI context
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]

    # If quiet and not JSON, suppress all output
    if quiet and not as_json:
        return

    stages_list = cli_helpers.stages_to_list(stages)
    _validate_stages(stages_list, single_stage)
    # ... rest of function unchanged ...
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_cli.py::test_dry_run_quiet_suppresses_output -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "fix(cli): respect --quiet flag in dry_run_cmd

The dry-run command was missing @click.pass_context and wasn't checking
the quiet flag from CLI context."
```

---

## Task 6: Add Stage Name Validation to Agent RPC `run` Command [MEDIUM]

**Files:**
- Modify: `src/pivot/engine/agent_rpc.py:276-303` (_handle_request method)
- Test: `tests/engine/test_agent_rpc.py`

**Step 1: Write the failing test**

Add to `tests/engine/test_agent_rpc.py`:

```python
@pytest.mark.anyio
async def test_rpc_run_invalid_stage_returns_error(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RPC run with invalid stage name returns descriptive error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".pivot").mkdir()

    # Create pipeline with known stages
    from pivot.pipeline.pipeline import Pipeline
    pipeline = Pipeline("test", root=tmp_path)

    def my_stage() -> None:
        pass

    pipeline.register(my_stage, name="valid_stage")

    async with engine.Engine(pipeline=pipeline) as eng:
        handler = AgentRpcHandler(engine=eng)
        source = AgentRpcSource(socket_path=tmp_path / "test.sock", handler=handler)

        send, recv = anyio.create_memory_object_stream[InputEvent](16)

        # Request with invalid stage name
        request = {
            "jsonrpc": "2.0",
            "method": "run",
            "params": {"stages": ["nonexistent_stage"]},
            "id": 1,
        }

        response = await source._handle_request(request, send)

        # Should return error about stage not found
        assert response is not None
        assert "error" in response
        assert response["error"]["code"] == -32001  # Stage not found error code
        assert "nonexistent_stage" in response["error"]["message"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_agent_rpc.py::test_rpc_run_invalid_stage_returns_error -v`
Expected: FAIL - currently accepts any stage name

**Step 3: Write minimal implementation**

Edit `src/pivot/engine/agent_rpc.py`:

1. Add custom error codes at module level:

```python
# Custom JSON-RPC error codes
_ERR_STAGE_NOT_FOUND = -32001
_ERR_EXECUTION_ERROR = -32002
```

2. Update `AgentRpcHandler` to support stage validation:

```python
class AgentRpcHandler:
    """Handles JSON-RPC queries that need engine/inspector access."""

    _engine: Engine

    def __init__(self, *, engine: Engine) -> None:
        self._engine = engine

    def validate_stages(self, stages: list[str] | None) -> str | None:
        """Validate stage names exist. Returns error message or None if valid."""
        if stages is None:
            return None

        pipeline = self._engine._pipeline  # pyright: ignore[reportPrivateUsage]
        if pipeline is None:
            return "No pipeline loaded"

        available = set(pipeline.list_stages())
        unknown = [s for s in stages if s not in available]
        if unknown:
            return f"Unknown stages: {', '.join(unknown)}"
        return None

    # ... rest of existing methods ...
```

3. Update `_handle_request` to validate stages:

```python
async def _handle_request(
    self,
    request: object,
    send: MemoryObjectSendStream[InputEvent],
) -> JsonRpcResponse | None:
    """Handle a single JSON-RPC request."""
    # ... existing validation code ...

    method, params, request_id = validated

    # Commands become input events
    if method == "run":
        # Validate stages param
        stages = _validate_stages_param(params.get("stages"))
        if isinstance(stages, str):
            return _json_rpc_error(-32602, f"Invalid params: {stages}", request_id)

        # Validate stage names exist (if handler available)
        if self._handler is not None and stages is not None:
            validation_error = self._handler.validate_stages(stages)
            if validation_error:
                return _json_rpc_error(_ERR_STAGE_NOT_FOUND, validation_error, request_id)

        # ... rest of existing code ...
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/engine/test_agent_rpc.py::test_rpc_run_invalid_stage_returns_error -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "fix(rpc): add stage name validation to agent RPC run command

Invalid stage names now return a descriptive JSON-RPC error (-32001)
instead of being passed through to cause cryptic failures later."
```

---

## Task 7: Add `--global/--local` Flags to `config get` Command [MEDIUM]

**Files:**
- Modify: `src/pivot/cli/config.py:76-90` (config_get function)
- Test: `tests/cli/test_config.py`

**Step 1: Write the failing test**

Add to `tests/cli/test_config.py`:

```python
def test_config_get_global_flag(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """config get --global reads from global config only."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".pivot").mkdir()

    # Set local value
    local_config = tmp_path / ".pivot" / "config.yaml"
    local_config.write_text("display:\n  precision: 3\n")

    # Set different global value
    global_dir = tmp_path / "global"
    global_dir.mkdir()
    global_config = global_dir / "config.yaml"
    global_config.write_text("display:\n  precision: 7\n")

    monkeypatch.setenv("PIVOT_CONFIG_DIR", str(global_dir))

    result = runner.invoke(cli.cli, ["config", "get", "--global", "display.precision"])

    assert result.exit_code == 0
    assert "7" in result.output  # Global value


def test_config_get_local_flag(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """config get --local reads from local config only."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".pivot").mkdir()

    # Set local value
    local_config = tmp_path / ".pivot" / "config.yaml"
    local_config.write_text("display:\n  precision: 3\n")

    result = runner.invoke(cli.cli, ["config", "get", "--local", "display.precision"])

    assert result.exit_code == 0
    assert "3" in result.output  # Local value
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_config.py::test_config_get_global_flag tests/cli/test_config.py::test_config_get_local_flag -v`
Expected: FAIL - flags don't exist

**Step 3: Write minimal implementation**

Edit `src/pivot/cli/config.py`:

```python
@config_cmd.command("get")
@click.argument("key", shell_complete=completion.complete_config_keys)
@click.option("--global", "use_global", is_flag=True, help="Get from global config only")
@click.option("--local", "use_local", is_flag=True, help="Get from local config only")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def config_get(key: str, use_global: bool, use_local: bool, output_json: bool) -> None:
    """Get a configuration value by dotted key."""
    if use_global and use_local:
        raise click.ClickException("Cannot use both --global and --local")

    if not config.is_valid_key(key):
        raise click.ClickException(f"Unknown config key: '{key}'")

    if use_global:
        data = config.load_config_file(config.get_global_config_path())
        value = config.get_nested_value(data, key)
        source = config.ConfigScope.GLOBAL
    elif use_local:
        data = config.load_config_file(config.get_local_config_path())
        value = config.get_nested_value(data, key)
        source = config.ConfigScope.LOCAL
    else:
        value, source = config.get_config_value(key)

    if output_json:
        click.echo(json.dumps({"key": key, "value": value, "source": str(source)}))
        return

    click.echo(f"{key} = {_format_value(value)} ({source})")
```

Note: May need to add `get_nested_value` helper to `config.py` if it doesn't exist.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_config.py::test_config_get_global_flag tests/cli/test_config.py::test_config_get_local_flag -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(cli): add --global/--local flags to config get command

Provides consistency with config set and config unset commands."
```

---

## Task 8: Run All Quality Checks

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -n auto`
Expected: All tests pass

**Step 2: Run type checking**

Run: `uv run basedpyright .`
Expected: No errors

**Step 3: Run linting**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: No errors

**Step 4: Commit any remaining fixes**

If any quality issues are found, fix them and update the commit.

---

## Summary of Changes

| Task | Severity | File | Change |
|------|----------|------|--------|
| 1 | CRITICAL | `src/pivot/cli/data.py` | Add `ensure_stages_registered()` |
| 2 | CRITICAL | `src/pivot/cli/params.py` | Add `ensure_stages_registered()` |
| 3 | HIGH | `src/pivot/cli/run.py` | Pass `no_cache` to watch mode |
| 4 | HIGH | `src/pivot/engine/engine.py` | Store orchestration params for re-runs |
| 5 | MEDIUM | `src/pivot/cli/run.py` | Add `@click.pass_context` to dry_run_cmd |
| 6 | MEDIUM | `src/pivot/engine/agent_rpc.py` | Add stage validation |
| 7 | MEDIUM | `src/pivot/cli/config.py` | Add `--global/--local` flags |

## Deferred Issues (Not in This Plan)

The following issues from the red-team review require larger architectural changes and should be addressed separately:

- **Task #45**: Restore WatchSink for watch+json output mode - requires new sink class
- **Task #46**: Fix `_classify_targets` NoPipelineError - needs analysis of metrics/plots commands
- **Task #47**: Make `set_watch_paths()` restart active watcher - requires FilesystemSource changes
- **Task #50**: Event type naming consistency - requires audit of all event types
