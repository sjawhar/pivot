---
tags: [python, context-manager, resource-management]
category: gotcha
module: engine
symptoms: ["output not flushed", "resources not released", "hanging process on exit"]
---

# Engine is a Context Manager

## Problem

Instantiating `Engine` directly and calling methods on it without using the context manager protocol can leave resources unreleased:

```python
# Wrong - resources not cleaned up
engine = Engine(pipeline=pipeline)
engine.add_sink(LiveConsoleSink(console=console))
engine.add_source(OneShotSource(stages=["train"], force=True, reason="cli"))
await engine.run(exit_on_completion=True)
# If an exception occurs above, sinks are never closed
# Even without exception, no guarantee close() runs
```

Consequences:
1. **Output not flushed** - Console sinks may have buffered output that never reaches the terminal
2. **File handles leaked** - Sinks that write to files may leave them unclosed
3. **Process hangs on exit** - Background threads or queues may block process termination

## Solution

Always use `Engine` as an async context manager with `async with`:

```python
from pivot.engine.engine import Engine
from pivot.engine import sinks, sources

async with Engine(pipeline=pipeline) as engine:
    engine.add_sink(sinks.LiveConsoleSink(console=console))
    engine.add_source(sources.OneShotSource(stages=["train"], force=True, reason="cli"))
    await engine.run(exit_on_completion=True)
# __aexit__ called here, even on exception
```

The `__aexit__` method closes all channels and sinks, iterating through each registered sink:

```python
async def __aexit__(self, *_exc: object) -> None:
    # Close send channels first to signal end-of-stream
    # Close sinks after output_send is closed
    for sink in self._sinks:
        try:
            await sink.close()
        except Exception:
            _logger.exception("Error closing sink %s", sink)
    # Close receive channels last
```

Exceptions from individual sinks are logged but do not prevent other sinks from being closed.

### Watch Mode

For watch mode (long-running), the pattern is identical:

```python
async with Engine(pipeline=pipeline) as engine:
    engine.add_source(sources.FilesystemSource(watch_paths=paths))
    await engine.run(exit_on_completion=False)  # Blocks until shutdown signal
# Sinks closed on exit, even if interrupted
```

### Testing

In tests, use the context manager to ensure cleanup between test cases:

```python
async def test_stage_execution(test_pipeline: Pipeline) -> None:
    async with Engine(pipeline=test_pipeline) as engine:
        collector = sinks.ResultCollectorSink()
        engine.add_sink(collector)
        engine.add_source(sources.OneShotSource(stages=None, force=True, reason="test"))
        await engine.run(exit_on_completion=True)

        results = await collector.get_results()
        assert results["my_stage"]["status"] == "ran"
```

## Key Insight

Context managers guarantee cleanup regardless of how the block exits. Python's `async with` statement calls `__aexit__` on:
- Normal completion
- `return` from within the block
- Exceptions (caught or uncaught)
- `sys.exit()` calls

Without the context manager, you must manually close channels and sinks in a `finally` block, which is error-prone:

```python
# Manual cleanup is fragile
engine = Engine(pipeline=pipeline)
try:
    engine.add_sink(sink)
    await engine.run(exit_on_completion=True)
finally:
    # Must close channels and sinks in correct order — easy to get wrong
    ...
```

The context manager encapsulates this pattern, making correct usage the default.

