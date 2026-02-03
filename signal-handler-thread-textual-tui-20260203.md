---
module: pivot CLI
date: 2026-02-03
problem_type: runtime_error
component: tooling
symptoms:
  - "ValueError: signal only works in main thread of the main interpreter"
  - "TUI crashes immediately when running pivot repro --tui"
root_cause: thread_violation
resolution_type: code_fix
severity: high
tags: [textual, signal-handler, threading, tui, anyio, python]
---

# Troubleshooting: Textual TUI Crashes with Signal Handler Error

## Problem
When running `pivot repro --tui`, the Textual TUI crashes immediately with `ValueError: signal only works in main thread of the main interpreter`. This occurs because Textual's LinuxDriver tries to register SIGTSTP/SIGCONT signal handlers, which Python only allows from the main thread.

## Environment
- Module: pivot CLI (`pivot/cli/repro.py`)
- Python Version: 3.13
- Textual Version: 7.4.0
- Affected Component: TUI mode for `pivot repro` and `pivot watch` commands
- Date: 2026-02-03

## Symptoms
- `ValueError: signal only works in main thread of the main interpreter` error
- TUI crashes immediately on startup
- Error traceback points to `textual/drivers/linux_driver.py:71` calling `signal.signal(signal.SIGTSTP, ...)`
- Only affects `--tui` mode; non-TUI mode works fine

## What Didn't Work

**Attempted Solution 1:** Use Textual's `app.run_async()` instead of `app.run()`
- **Why it failed:** `run_async()` still goes through `_build_driver()` → `LinuxDriver.__init__()` → `signal.signal()`. The signal registration happens regardless of whether you use the sync or async API.

**Verification test:**
```python
# Test 2: anyio.run() from worker thread
def worker():
    try:
        anyio.run(test_run_async)  # Uses app.run_async() internally
        print('Success')
    except ValueError as e:
        print(f'Error: {e}')  # Still fails with signal error

t = threading.Thread(target=worker)
t.start()
t.join()
# Result: ValueError: signal only works in main thread
```

## Solution

**Invert the threading relationship:** Run Textual's `app.run()` directly from the main thread, and run the async Engine in a background thread with its own event loop.

**Code changes in `pivot/cli/repro.py`:**

```python
# Before (broken) - TUI runs in worker thread:
async def tui_oneshot_main():
    app = PivotApp(...)
    async with engine.Engine(pipeline=pipeline) as eng:
        async with anyio.create_task_group() as tg:
            tg.start_soon(run_engine_and_signal)
            await anyio.to_thread.run_sync(app.run)  # PROBLEM: worker thread
            tg.cancel_scope.cancel()

return anyio.run(tui_oneshot_main)  # Main thread runs anyio loop

# After (fixed) - TUI runs in main thread:
from concurrent.futures import Future

app = PivotApp(...)
result_future: Future[dict[str, ExecutionSummary]] = Future()

def engine_thread_target() -> None:
    """Run async Engine in background thread with its own event loop."""
    async def engine_main():
        async with engine.Engine(pipeline=pipeline) as eng:
            # Configure sinks - TuiSink posts to app (thread-safe)
            # ... configure sources ...
            await eng.run(exit_on_completion=True)
            return await result_sink.get_results()

    try:
        result_future.set_result(anyio.run(engine_main))
    except BaseException as e:
        result_future.set_exception(e)
    finally:
        with contextlib.suppress(Exception):
            app.post_message(TuiShutdown())

engine_thread = threading.Thread(target=engine_thread_target, daemon=True)
engine_thread.start()

app.run()  # Main thread - signals work!

engine_thread.join(timeout=5.0)
return result_future.result()  # Re-raises if engine raised exception
```

## Why This Works

1. **Root cause**: Python's `signal.signal()` can only be called from the main thread of the main interpreter. Textual's `LinuxDriver.__init__` registers SIGTSTP (Ctrl+Z suspend) and SIGCONT (resume) handlers for proper terminal management.

2. **Why the original code failed**: `anyio.run()` was called from the main thread, making it the event loop host. Then `anyio.to_thread.run_sync(app.run)` offloaded Textual to a worker thread, where signal registration fails.

3. **Why the fix works**:
   - Textual runs directly in the main thread, so signal handlers register successfully
   - The async Engine runs in a background thread with its own `anyio.run()` event loop
   - Communication uses `app.post_message()` which is explicitly documented as thread-safe
   - Two separate event loops provide isolation (Engine blocking calls don't starve TUI refresh)

4. **Performance impact**: Negligible. Cross-thread `post_message()` latency is ~1-10 microseconds vs TUI refresh interval of ~16,666 microseconds (60 FPS).

## Prevention

- **When using Textual (or any library that registers signal handlers)**: Always run it from the main thread
- **Pattern for combining Textual with async code**: Let Textual own the main thread, run async work in background threads
- **Check library requirements**: Signal-handling libraries often document main-thread requirements
- **Test on Linux**: macOS may not exhibit the same signal restrictions, so test on Linux where LinuxDriver is used

## Related Issues

No related issues documented yet.
