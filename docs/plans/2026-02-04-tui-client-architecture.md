# TUI Client Architecture Refactor

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make TUI a pure display client of the Engine, fixing immediate-exit race condition.

**Architecture:** Engine is the authority. TUI receives events via `TuiUpdate`, exits on `TuiShutdown` (run mode) or user quit (watch mode). TUI never starts execution threads.

**Tech Stack:** Python 3.13+, Textual, anyio

---

## Problem

Race condition: dummy `executor_func` returns `{}` immediately → `ExecutorComplete` posted → TUI exits before real engine runs.

## Solution

Remove `executor_func` pattern entirely. TUI waits for `TuiShutdown` from engine. Align `run.py` with `repro.py` threading pattern (background engine thread, main thread TUI).

---

## Changes

### `src/pivot/tui/run.py`

**Remove entirely:**
- `ExecutorComplete` message class
- `executor_func` parameter from `__init__`
- `_executor_func` instance variable
- `_executor_thread` instance variable
- `_run_executor()` method
- `on_executor_complete()` handler
- `_run_engine()` method (legacy pattern)
- `_engine_thread` instance variable
- `engine` parameter from `__init__` (no longer needed - engine always external)

**Simplify `__init__`:**
```python
def __init__(
    self,
    stage_names: list[str] | None = None,
    tui_log: pathlib.Path | None = None,
    *,
    cancel_event: threading.Event | None = None,
    watch_mode: bool = False,
    no_commit: bool = False,
    serve: bool = False,
) -> None:
```

**Simplify `on_mount`:**
```python
async def on_mount(self) -> None:
    self._start_time = time.monotonic()
    if self._log_file is not None:
        self._stats_log_timer = self.set_interval(1.0, self._write_stats_to_log)
    self.call_after_refresh(self._update_detail_panel)

    if self._watch_mode:
        prefix = self._get_keep_going_prefix()
        self.title = f"{prefix}[●] Watching for changes..."
    # Run mode: TUI waits for TuiShutdown from external engine
```

**Update `on_tui_shutdown` to ring bell in run mode:**
```python
def on_tui_shutdown(self, _event: TuiShutdown) -> None:
    self._write_to_log('{"type": "shutdown"}\n')
    if not self._watch_mode:
        self.bell()  # Notify user of completion
        self._shutdown_event.set()
        self._close_log_file()
        self._shutdown_loky_pool()
        self.exit(self._results)
```

### `src/pivot/cli/repro.py`

**Oneshot TUI mode (around line 591):**
- Remove `executor_wrapper`
- Remove `executor_func` from `PivotApp()` call
- Keep existing pattern: background engine thread posts `TuiShutdown` in `finally`

**Watch TUI mode (around line 360):**
- Remove `engine=None` parameter (no longer exists)
- Keep existing pattern: background engine thread, TUI in main thread

### `src/pivot/cli/run.py`

**Align with repro.py pattern** - currently uses `anyio.create_task_group()` which is different.

Change from:
```python
async def tui_oneshot_main() -> dict[str, ExecutionSummary]:
    app = tui_run.PivotApp(...)
    async with engine.Engine(...) as eng:
        ...
        async with anyio.create_task_group() as tg:
            tg.start_soon(run_engine_and_signal)
            await anyio.to_thread.run_sync(app.run)
            tg.cancel_scope.cancel()
        return engine_results
```

To match repro.py:
```python
def _run_with_tui(...) -> dict[str, ExecutionSummary]:
    app = tui_run.PivotApp(...)
    result_future: Future[dict[str, ExecutionSummary]] = Future()

    def engine_thread_target() -> None:
        async def engine_main() -> dict[str, ExecutionSummary]:
            async with engine.Engine(...) as eng:
                ...
                await eng.run(exit_on_completion=True)
                return results
        try:
            result_future.set_result(anyio.run(engine_main))
        except BaseException as e:
            result_future.set_exception(e)
        finally:
            with contextlib.suppress(Exception):
                app.post_message(tui_run.TuiShutdown())

    engine_thread = threading.Thread(target=engine_thread_target, daemon=True)
    engine_thread.start()

    with suppress_stderr_logging():
        app.run()  # Main thread

    engine_thread.join(timeout=5.0)
    return result_future.result() if result_future.done() else {}
```

### `tests/tui/test_run.py`

- Remove all `executor_func` from `PivotApp()` calls
- Remove `ExecutorComplete` tests
- Remove `test_run_tui_app_requires_executor_func_or_watch_mode`
- Remove `test_on_executor_complete_calls_bell`
- Update `simple_run_app` fixture
- Add `test_on_tui_shutdown_calls_bell_in_run_mode`
- Add `test_on_tui_shutdown_no_bell_in_watch_mode`
- Add `test_run_tui_app_defaults_to_run_mode`

### `tests/conftest.py`

- Remove `mock_watch_engine` fixture (no longer needed - engine always external)
- Update any tests that used it to use `watch_mode=True` directly

---

## Verification

```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright
uv run pytest tests/tui/ tests/engine/test_tui_sink.py -v
uv run pytest tests/ -n auto
```

Manual test:
```bash
cd ~/eval-pipeline/pivot
pivot repro --tui --force
```

---

## Commit

```
refactor(tui): make TUI a pure display client of the Engine

Remove executor_func pattern - TUI now waits for TuiShutdown from
external engine instead of managing its own execution threads.

Fixes race condition where dummy executor completed immediately,
causing TUI exit before real engine ran.

Changes:
- Remove ExecutorComplete, executor_func, _run_executor, _run_engine
- Remove engine parameter (engine always managed externally now)
- Align run.py threading pattern with repro.py
- Move bell notification to on_tui_shutdown

Architecture: Engine is authority. TUI, RPC, CLI are clients.
```
