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
engine.add_sink(ConsoleSink(console))
engine.add_source(OneShotSource(stages=["train"], force=True, reason="cli"))
engine.run(exit_on_completion=True)
# If an exception occurs above, sinks are never closed
# Even without exception, no guarantee close() runs
```

Consequences:
1. **Output not flushed** - ConsoleSink may have buffered output that never reaches the terminal
2. **TUI termination signal not sent** - TuiSink sends `None` to signal completion; skipping this leaves the TUI hanging
3. **File handles leaked** - Sinks that write to files may leave them unclosed
4. **Process hangs on exit** - Background threads or queues may block process termination

## Solution

Always use `Engine` as a context manager with `with` statement:

```python
from pivot.engine.engine import Engine
from pivot.engine import sinks, sources

with Engine(pipeline=pipeline) as engine:
    engine.add_sink(sinks.ConsoleSink(console))
    engine.add_source(sources.OneShotSource(stages=["train"], force=True, reason="cli"))
    engine.run(exit_on_completion=True)
# __exit__ called here, even on exception
```

The `__exit__` method calls `engine.close()`, which iterates through all registered sinks and closes each one:

```python
def close(self) -> None:
    """Close all sinks and clean up resources."""
    for sink in self._sinks:
        try:
            sink.close()
        except Exception:
            _logger.exception("Sink %s failed to close", sink)
```

Exceptions from individual sinks are logged but do not prevent other sinks from being closed.

### Watch Mode

For watch mode (long-running), the pattern is identical:

```python
with Engine(pipeline=pipeline) as engine:
    engine.add_sink(sinks.TuiSink(tui_queue, run_id))
    engine.add_source(sources.FilesystemSource(watch_paths))
    engine.run(exit_on_completion=False)  # Blocks until shutdown signal
# Sinks closed on exit, even if interrupted
```

### Testing

In tests, use the context manager to ensure cleanup between test cases:

```python
def test_stage_execution(test_pipeline: Pipeline) -> None:
    with Engine(pipeline=test_pipeline) as engine:
        collector = sinks.ResultCollectorSink()
        engine.add_sink(collector)
        engine.add_source(sources.OneShotSource(stages=None, force=True, reason="test"))
        engine.run(exit_on_completion=True)

        assert collector.results["my_stage"]["status"] == "ran"
```

## Key Insight

Context managers guarantee cleanup regardless of how the block exits. Python's `with` statement calls `__exit__` on:
- Normal completion
- `return` from within the block
- Exceptions (caught or uncaught)
- `sys.exit()` calls

Without the context manager, you must manually call `close()` in a `finally` block, which is error-prone:

```python
# Manual cleanup is fragile
engine = Engine(pipeline=pipeline)
try:
    engine.add_sink(sink)
    engine.run(exit_on_completion=True)
finally:
    engine.close()  # Easy to forget
```

The context manager encapsulates this pattern, making correct usage the default.

