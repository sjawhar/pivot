---
tags: [python, anyio, async, race-condition, event-dispatch]
category: gotcha
module: engine
symptoms: ["flaky test", "events dropped", "blocked stage not reported", "0 blocked in summary when stages were blocked"]
---

# Engine Dispatcher Must Drain Before Task Group Cancellation

## Problem

The `test_repro_keep_going_flag_skips_downstream` test was failing intermittently (~20-40% failure rate). When a stage failed and downstream stages were blocked, the blocked stage completion events were sometimes not delivered to sinks.

**Observable symptoms:**
- Test expected `"second: skipped"` in output but it wasn't present
- Summary showed `"0 blocked"` when `second` should have been blocked
- The blocked stage was correctly marked as `BLOCKED` internally but the event was dropped

**Debug trace showed:**
```
EMIT OK: second skipped    # Event successfully sent to channel
DISPATCH: independent ran  # But only independent was dispatched
# Missing: DISPATCH: second skipped
```

The event was emitted to the memory channel but never received by the dispatcher.

## Root Cause

In `engine.py`, the shutdown sequence was:

```python
# Close output channel to signal end-of-stream to dispatcher.
if self._output_send:
    await self._output_send.aclose()

# Brief yield to let dispatcher process remaining buffered events.
await anyio.sleep(0)  # PROBLEM: Single yield isn't enough!

# Cancel remaining tasks (sources)
tg.cancel_scope.cancel()  # Cancels dispatcher mid-drain
```

A single `sleep(0)` yields control exactly once. If multiple events are buffered in the channel, the dispatcher may only process one before the yield returns and `cancel_scope.cancel()` cancels all tasks in the group - including the dispatcher that's still draining events.

**Timeline of the race:**
1. `first` stage fails, emits completion event
2. Post-loop runs, emits `second: skipped` event
3. `independent` stage completes, emits completion event
4. `_output_send.aclose()` closes the channel
5. `sleep(0)` yields - dispatcher processes ONE event
6. `cancel_scope.cancel()` - dispatcher cancelled with events still buffered

## Solution

Use an `anyio.Event` to signal when the dispatcher has finished draining:

```python
class Engine:
    _dispatch_complete: anyio.Event

    def __init__(self, ...):
        # Event signaling dispatcher has finished draining (recreated each run())
        self._dispatch_complete = anyio.Event()

    async def run(self, ...):
        # Create fresh dispatch completion event (Events can only be set once)
        self._dispatch_complete = anyio.Event()
        # ... run pipeline ...

    async def _dispatch_outputs(self) -> None:
        """Dispatch output events to all sinks."""
        try:
            async for event in self._output_recv:
                # dispatch to sinks...
        finally:
            self._dispatch_complete.set()  # Signal completion
```

In the shutdown sequence:

```python
# Close output channel to signal end-of-stream to dispatcher.
await self._output_send.aclose()

# Wait for dispatcher to finish draining all buffered events.
# Use a timeout to prevent infinite hang if dispatcher gets stuck.
with anyio.move_on_after(5.0):
    await self._dispatch_complete.wait()

# Cancel remaining tasks (sources and possibly stuck dispatcher)
tg.cancel_scope.cancel()
```

This ensures:
1. The dispatcher signals completion when it naturally exits (channel exhausted)
2. We wait for the signal rather than polling/guessing timing
3. A timeout prevents infinite hangs if the dispatcher gets stuck
4. After timeout, we cancel anyway so the engine doesn't block forever

## Key Insight

When shutting down async task groups that communicate via channels:

1. **Closing a channel send-end doesn't immediately drain receivers** - Events may still be buffered
2. **`sleep(0)` is ONE yield, not "until idle"** - Other tasks get one chance to run
3. **`cancel_scope.cancel()` is immediate** - No grace period for tasks to finish work
4. **Use explicit signaling, not timing assumptions** - Events are more reliable than sleep loops
5. **Always have a timeout** - Prevent infinite hangs if consumer gets stuck

### Pattern for Clean Shutdown

```python
dispatch_complete = anyio.Event()

async def consumer_task():
    try:
        async for item in recv_channel:
            # process item...
    finally:
        dispatch_complete.set()  # Signal done

async with anyio.create_task_group() as tg:
    tg.start_soon(producer_task)
    tg.start_soon(consumer_task)

    # ... do work ...

    # Signal producer to stop
    await send_channel.aclose()

    # Wait for consumer to drain (with safety timeout)
    with anyio.move_on_after(5.0):
        await dispatch_complete.wait()

    # Now safe to cancel remaining tasks
    tg.cancel_scope.cancel()
```

## Testing

After the fix: 50+ consecutive test passes (0% failure rate).
