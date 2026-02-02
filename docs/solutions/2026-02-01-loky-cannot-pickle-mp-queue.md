---
tags: [python, multiprocessing, loky, pickling]
category: gotcha
module: executor
symptoms: ["PicklingError", "can't pickle Queue", "worker process fails to start"]
---

# loky Cannot Pickle `mp.Queue()` - Use Manager Queues

## Problem

When using loky's process pool executor, passing a `multiprocessing.Queue()` to worker functions fails with a pickling error:

```python
import multiprocessing as mp
from loky import get_reusable_executor

queue = mp.Queue()  # Direct Queue instance

def worker(q):
    q.put("result")

executor = get_reusable_executor()
future = executor.submit(worker, queue)  # PicklingError!
```

The error occurs because `mp.Queue()` contains OS-level resources (file descriptors, semaphores, locks) that cannot be serialized and sent to another process. loky serializes function arguments using cloudpickle/pickle to transfer them to worker processes, and these OS resources have no meaningful serialized representation.

This differs from the standard `concurrent.futures.ProcessPoolExecutor` behavior because loky uses a "spawn" context and serializes everything explicitly, while the standard executor may use "fork" which shares memory.

## Solution

Use `multiprocessing.Manager().Queue()` instead, which creates a proxy object that can be pickled:

```python
import multiprocessing as mp
from loky import get_reusable_executor

# Create a Manager (spawns a server process)
manager = mp.Manager()
queue = manager.Queue()  # Proxy object - picklable

def worker(q):
    q.put("result")

executor = get_reusable_executor()
future = executor.submit(worker, queue)
future.result()  # Works!
```

For explicit spawn context (recommended for consistency):

```python
spawn_ctx = mp.get_context("spawn")
local_manager = spawn_ctx.Manager()
output_queue = local_manager.Queue()
```

The Manager creates a separate server process that owns the actual queue. The `manager.Queue()` returns a proxy object that communicates with this server via IPC. The proxy is just a lightweight handle that can be pickled and sent to workers.

## Key Insight

The fundamental difference:

| Type | What It Is | Picklable? |
|------|-----------|------------|
| `mp.Queue()` | Direct queue with OS resources (pipes, locks) | No |
| `manager.Queue()` | Proxy to queue in Manager server process | Yes |

When designing multiprocessing systems with loky:

1. **Any object passed to workers must be picklable** - this includes queues, events, and shared state
2. **Manager objects add latency** - IPC to the manager server, but this is usually negligible compared to the work being done
3. **Manager lifecycle matters** - keep the manager alive as long as workers need the queue

In Pivot, the engine creates a Manager-based queue for collecting output messages from worker processes:

```python
# From src/pivot/engine/engine.py
spawn_ctx = mp.get_context("spawn")
local_manager = spawn_ctx.Manager()
output_queue: mp.Queue[OutputMessage] = local_manager.Queue()
```

This allows workers to send log output and status messages back to the coordinator process for display in the TUI.
