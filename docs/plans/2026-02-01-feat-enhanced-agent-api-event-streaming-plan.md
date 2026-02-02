---
title: "feat(rpc): Enhanced RPC API with Event Streaming"
type: feat
date: 2026-02-01
github_issue: 261
depends_on: [307, 305]
deepened: 2026-02-01
---

# Enhanced RPC API with Event Streaming

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the RPC interface with event streaming and richer queries, enabling any client (agents, tools, scripts) to monitor and control pipeline execution.

**Architecture:** Add a ring buffer for event polling, plus query handlers that call existing `explain.py` and `status.py` functions directly. No new abstraction layers — just wiring.

**Tech Stack:** Python 3.13+, TypedDict, anyio, deque

**GitHub Issue:** #261

---

## Research Insights (Post-Deepening)

### Critical Fixes Required

1. **`get_stage_explanation()` Signature Mismatch** — Use `reg_info["deps_paths"]` and `reg_info["outs_paths"]` directly (already `list[str]`), not manual path conversion.

2. **Threading Lock in Async Context** — Use `threading.Lock` because the buffer is accessed from both sync (`events_since()`) and async (`handle()`) contexts. `threading.Lock` can be acquired from either; `anyio.Lock` cannot.

3. **Lazy Imports Violation** — CLAUDE.md forbids lazy imports. Move `from pivot import explain as explain_mod` to module level.

4. **Empty Collection Syntax** — Use `list[T]()` not `[]` per CLAUDE.md (e.g., `events=list[dict[str, object]]()`).

### Simplifications Recommended

1. **Skip Task 1 (rename)** — Renaming `AgentRpcHandler` → `RpcHandler` adds churn without value. Keep existing names.

2. **Remove `gap` field from EventBuffer** — No concrete use case. Clients can simply query current state.

3. **Consider removing `logs` handler** — Clients can filter `events_since` for `log_line` events. Dedicated endpoint duplicates functionality.

4. **Simplify `stage_info`** — Remove upstream/downstream (requires graph rebuild). Just return deps/outs.

### Security Hardening

1. **Add upper bounds** — `version` should be capped at `2**63-1`, `lines` at `10000`.

2. **Information disclosure is acceptable** — Local-only Unix socket with 0o600 permissions. Document that `explain` exposes parameter values.

### Performance Considerations

1. **Wrap `explain` in thread** — `get_stage_explanation()` does blocking file I/O. Use `anyio.to_thread.run_sync()`.

2. **Reuse engine graph** — In `stage_info`, use `self._engine._graph` instead of rebuilding.

### Agent-Native Gaps (Future Work)

After this plan, agents can observe but have limited action capability:

| Missing | Priority |
|---------|----------|
| Extend `run` params (--no-commit, --keep-going) | High |
| `checkout` method | High |
| `commit` + `list_pending` methods | Medium |
| `push` / `pull` methods | Medium |
| `verify` method | Medium |
| `history` + `show_run` methods | Low |

---

## Design Principles

1. **RPC is just another client** — Same as CLI and TUI, it calls existing query functions
2. **No duplication** — Queries use `explain.get_stage_explanation()` and `status.get_pipeline_status()` directly
3. **Minimal new code** — ~100 lines, not 500
4. **Unified event stream** — One ring buffer, filter for logs client-side

---

## ~~Task 1: Rename AgentRpcHandler → RpcHandler~~ (SKIPPED)

**Decision:** Keep existing names. Renaming adds git history noise without functional benefit. "Agent" accurately describes the primary use case.

---

## Task 2: Add Event Ring Buffer for Polling

**Files:**
- Modify: `src/pivot/engine/agent_rpc.py`
- Test: `tests/engine/test_agent_rpc.py`

> **Research Insight:** Use `threading.Lock` (not `anyio.Lock`) since buffer is accessed from both async event handlers and sync query methods.

**Step 1: Write the failing test**

Add to `tests/engine/test_agent_rpc.py`:

```python
async def test_event_buffer_captures_events() -> None:
    """EventBuffer should capture events with version numbers."""
    from pivot.engine.agent_rpc import EventBuffer

    buffer = EventBuffer(max_events=100)
    buffer.handle_sync({"type": "stage_started", "stage": "train", "index": 1, "total": 2})

    result = buffer.events_since(0)
    assert result["version"] == 1
    assert len(result["events"]) == 1


async def test_event_buffer_eviction() -> None:
    """EventBuffer should evict oldest events when full."""
    from pivot.engine.agent_rpc import EventBuffer

    buffer = EventBuffer(max_events=3)
    for i in range(5):
        buffer.handle_sync({"type": "stage_started", "stage": f"s{i}", "index": i, "total": 5})

    result = buffer.events_since(0)
    assert len(result["events"]) == 3  # Only last 3 events
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_agent_rpc.py::test_event_buffer_captures_events -v`
Expected: FAIL with `ImportError: cannot import name 'EventBuffer'`

**Step 3: Write minimal implementation**

Add to `src/pivot/engine/agent_rpc.py`:

```python
class VersionedEvent(TypedDict):
    """Event with version number."""
    version: int
    event: OutputEvent


class EventsResult(TypedDict):
    """Result of events_since query."""
    version: int
    events: list[VersionedEvent]


class EventBuffer:
    """Ring buffer for event polling via events_since.

    Thread-safe: uses threading.Lock since buffer is accessed from both
    sync (events_since) and async (handle) contexts.
    """

    _max_events: int
    _events: deque[tuple[int, OutputEvent]]
    _version: int
    _lock: threading.Lock

    def __init__(self, max_events: int = 1000) -> None:
        self._max_events = max_events
        self._events = deque[tuple[int, OutputEvent]](maxlen=max_events)
        self._version = 0
        self._lock = threading.Lock()

    def handle_sync(self, event: OutputEvent) -> None:
        """Store event with version number (sync, thread-safe)."""
        with self._lock:
            self._version += 1
            self._events.append((self._version, event))

    async def handle(self, event: OutputEvent) -> None:
        """Async wrapper for sink interface compatibility."""
        self.handle_sync(event)

    def events_since(self, since_version: int) -> EventsResult:
        """Return events after the given version."""
        with self._lock:
            result = [
                VersionedEvent(version=v, event=e)
                for v, e in self._events
                if v > since_version
            ]
            return EventsResult(version=self._version, events=result)
```

Update `__all__` to include `"EventBuffer"`, `"VersionedEvent"`, `"EventsResult"`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/engine/test_agent_rpc.py::test_event_buffer -v`
Expected: PASS (both tests)

**Step 5: Run type checker**

Run: `uv run basedpyright src/pivot/engine/agent_rpc.py`
Expected: No errors

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add EventBuffer for event polling"
```

---

## Task 3: Wire EventBuffer into Engine and CLI

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `src/pivot/cli/run.py`
- Modify: `src/pivot/engine/agent_rpc.py`

**Step 1: Add EventBuffer to Engine**

In `src/pivot/engine/engine.py`, add import and create buffer:

```python
# Add import
from pivot.engine.agent_rpc import EventBuffer

# In Engine class, add field
_event_buffer: EventBuffer

# In __init__ or __aenter__, create and register
self._event_buffer = EventBuffer(max_events=1000)
# Register as sink so it receives events
```

**Step 2: Update AgentRpcHandler constructor**

In `src/pivot/engine/agent_rpc.py`:

```python
class AgentRpcHandler:
    _engine: Engine
    _event_buffer: EventBuffer | None

    def __init__(
        self,
        *,
        engine: Engine,
        event_buffer: EventBuffer | None = None,
    ) -> None:
        self._engine = engine
        self._event_buffer = event_buffer
```

**Step 3: Pass buffer in CLI**

In `src/pivot/cli/run.py`, where `AgentRpcHandler` is created:

```python
handler = AgentRpcHandler(engine=eng, event_buffer=eng._event_buffer)
```

**Step 4: Rename AgentEventSink → EventSink**

In `src/pivot/engine/agent_rpc.py`:
- Rename class `AgentEventSink` → `EventSink`
- Update `__all__`
- Update imports in `engine.py` and `run.py`

**Step 5: Run type checker**

Run: `uv run basedpyright src/pivot/engine/`
Expected: No errors

**Step 6: Commit**

```bash
jj describe -m "feat(engine): wire EventBuffer into Engine and rename EventSink"
```

---

## Task 4: Add Query Handlers to AgentRpcHandler

**Files:**
- Modify: `src/pivot/engine/agent_rpc.py`
- Test: `tests/engine/test_agent_rpc.py`

> **Research Insight:** Move imports to module level (CLAUDE.md requires no lazy imports). Wrap `get_stage_explanation()` in `anyio.to_thread.run_sync()` since it does blocking file I/O.

**Step 1: Add module-level imports**

Add at top of `src/pivot/engine/agent_rpc.py`:

```python
from pivot import explain as explain_mod
from pivot import parameters
from pivot.config import io as config_io
from pivot.types import StageExplanation
```

Add module-level constant:

```python
_MAX_VERSION = 2**63 - 1
```

**Step 2: Write failing test for events_since**

Add to `tests/engine/test_agent_rpc.py`:

```python
async def test_handler_events_since_query() -> None:
    """Handler should return events from buffer."""
    from pivot.engine.agent_rpc import AgentRpcHandler, EventBuffer
    from unittest.mock import MagicMock

    buffer = EventBuffer(max_events=100)
    buffer.handle_sync({"type": "stage_started", "stage": "train", "index": 1, "total": 1})

    mock_engine = MagicMock()
    handler = AgentRpcHandler(engine=mock_engine, event_buffer=buffer)

    result = await handler.handle_query("events_since", {"version": 0})
    assert result["version"] == 1
    assert len(result["events"]) == 1
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_agent_rpc.py::test_handler_events_since_query -v`
Expected: FAIL with `ValueError: Unknown query method: events_since`

**Step 4: Add events_since handler**

In `AgentRpcHandler.handle_query()`, add case:

```python
case "events_since":
    if self._event_buffer is None:
        raise ValueError("Event buffer not configured")
    version = params["version"] if "version" in params else 0
    if not isinstance(version, int) or version < 0 or version > _MAX_VERSION:
        raise ValueError(f"version must be integer between 0 and {_MAX_VERSION}")
    return self._event_buffer.events_since(version)
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/engine/test_agent_rpc.py::test_handler_events_since_query -v`
Expected: PASS

**Step 6: Add explain handler**

In `AgentRpcHandler.handle_query()`, add case:

```python
case "explain":
    stage = params["stage"] if "stage" in params else None
    if not isinstance(stage, str):
        raise ValueError("stage must be string")

    pipeline = self._engine._pipeline  # pyright: ignore[reportPrivateUsage]
    if pipeline is None:
        raise ValueError("No pipeline loaded")

    reg_info = pipeline.get(stage)
    if reg_info is None:
        raise ValueError(f"Unknown stage: {stage}")

    def _get_explanation() -> StageExplanation:
        return explain_mod.get_stage_explanation(
            stage_name=stage,
            fingerprint=reg_info["fingerprint"],
            deps=reg_info["deps_paths"],
            outs_paths=reg_info["outs_paths"],
            params_instance=reg_info["params"],
            overrides=parameters.load_params_yaml(),
            state_dir=config_io.get_state_dir(),
        )

    return await anyio.to_thread.run_sync(_get_explanation)
```

**Step 7: Add stage_info handler**

In `AgentRpcHandler.handle_query()`, add case:

```python
case "stage_info":
    stage = params["stage"] if "stage" in params else None
    if not isinstance(stage, str):
        raise ValueError("stage must be string")

    pipeline = self._engine._pipeline  # pyright: ignore[reportPrivateUsage]
    if pipeline is None:
        raise ValueError("No pipeline loaded")

    reg_info = pipeline.get(stage)
    if reg_info is None:
        raise ValueError(f"Unknown stage: {stage}")

    return {
        "name": stage,
        "deps": reg_info["deps_paths"],
        "outs": reg_info["outs_paths"],
    }
```

**Step 8: Run type checker**

Run: `uv run basedpyright src/pivot/engine/agent_rpc.py`
Expected: No errors

**Step 9: Commit**

```bash
jj describe -m "feat(rpc): add events_since, explain, stage_info query handlers"
```

> **Note:** `logs` handler removed — clients filter `events_since` for `type == "log_line"` events.

---

## Task 5: E2E Integration Test

**Files:**
- Create: `tests/integration/test_rpc_queries.py`

> **Research Insight (CRITICAL):** Per `docs/solutions/integration-issues/missing-e2e-test-cli-serve-mode-20260201.md`, unit tests can pass but wiring can be broken. This E2E test is non-negotiable.

**Step 1: Write the E2E test file**

Create `tests/integration/test_rpc_queries.py`:

```python
# tests/integration/test_rpc_queries.py

import json
import socket
import subprocess
import time
from collections.abc import Generator
from pathlib import Path

import pytest


def send_rpc(sock_path: Path, method: str, params: dict[str, object] | None = None) -> dict[str, object]:
    """Send JSON-RPC request and return response."""
    request: dict[str, object] = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params:
        request["params"] = params

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(sock_path))
    sock.sendall(json.dumps(request).encode() + b"\n")
    response = sock.recv(4096).decode()
    sock.close()
    return json.loads(response)


@pytest.fixture
def serve_pipeline(tmp_path: Path) -> Generator[Path]:
    """Start a minimal pipeline in serve mode, yield socket path."""
    (tmp_path / "pivot.yaml").write_text("""
stages:
  - name: hello
    func: pipeline:hello
""")
    (tmp_path / "pipeline.py").write_text("""
def hello() -> None:
    print("Hello!")
""")

    sock_path = tmp_path / ".pivot" / "agent.sock"

    proc = subprocess.Popen(
        ["uv", "run", "pivot", "run", "--serve", "--force"],
        cwd=tmp_path,
    )

    try:
        # Wait for socket
        for _ in range(50):
            if sock_path.exists():
                break
            time.sleep(0.1)

        time.sleep(0.3)  # Brief wait for events to accumulate
        yield sock_path
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_events_since_query(serve_pipeline: Path) -> None:
    """events_since should return buffered events."""
    response = send_rpc(serve_pipeline, "events_since", {"version": 0})
    assert "result" in response, f"Expected result, got: {response}"
    result = response["result"]
    assert isinstance(result, dict)
    assert "version" in result
    assert "events" in result
    assert isinstance(result["events"], list)


def test_explain_query(serve_pipeline: Path) -> None:
    """explain should return staleness info."""
    response = send_rpc(serve_pipeline, "explain", {"stage": "hello"})
    assert "result" in response, f"Expected result, got: {response}"
    result = response["result"]
    assert isinstance(result, dict)
    assert "will_run" in result  # StageExplanation field
    assert "reason" in result


def test_stage_info_query(serve_pipeline: Path) -> None:
    """stage_info should return stage metadata."""
    response = send_rpc(serve_pipeline, "stage_info", {"stage": "hello"})
    assert "result" in response, f"Expected result, got: {response}"
    result = response["result"]
    assert isinstance(result, dict)
    assert result["name"] == "hello"
    assert "deps" in result
    assert "outs" in result
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/integration/test_rpc_queries.py -v`
Expected: All 3 tests PASS

**Step 3: Run type checker**

Run: `uv run basedpyright tests/integration/test_rpc_queries.py`
Expected: No errors

**Step 4: Commit**

```bash
jj describe -m "test(integration): add E2E tests for RPC query handlers"
```

---

## Summary

| Task | Description | LOC |
|------|-------------|-----|
| ~~1~~ | ~~Rename AgentRpcHandler~~ (skipped) | 0 |
| 2 | Add EventBuffer | ~35 |
| 3 | Wire EventBuffer into Engine | ~10 |
| 4 | Add query handlers (events_since, explain, stage_info) | ~40 |
| 5 | E2E tests | ~80 |

**Also during implementation:**
- Rename `AgentEventSink` → `EventSink`
- Add `VersionedEvent` and `EventsResult` TypedDicts
- Add `_MAX_VERSION` constant for input validation

**Total new code: ~75 lines** (plus ~80 lines tests)

**What we removed from original plan:**
- ❌ Rename refactor (Task 1) — cosmetic churn
- ❌ PipelineInspector (~100 lines)
- ❌ agent_types.py (~40 lines)
- ❌ LogStore (~80 lines)
- ❌ Separate EventStore (~80 lines)
- ❌ `logs` handler — clients filter events_since instead
- ❌ `gap` field in EventBuffer — YAGNI
- ❌ upstream/downstream in stage_info — requires graph rebuild

---

## API Reference

### events_since

```json
{"jsonrpc": "2.0", "id": 1, "method": "events_since", "params": {"version": 0}}
```

Returns all events since version 0. Use returned `version` for next poll.

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "version": 42,
    "events": [
      {"version": 41, "event": {"type": "stage_started", "stage": "train", ...}},
      {"version": 42, "event": {"type": "log_line", "stage": "train", "line": "Hello", ...}}
    ]
  }
}
```

**Filtering logs:** To get logs for a specific stage, filter the events array:
```python
logs = [e["event"] for e in result["events"] if e["event"]["type"] == "log_line" and e["event"]["stage"] == "train"]
```

### explain

```json
{"jsonrpc": "2.0", "id": 1, "method": "explain", "params": {"stage": "train"}}
```

Returns `StageExplanation` (same type as CLI's `pivot status --explain`).

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "stage_name": "train",
    "will_run": true,
    "is_forced": false,
    "reason": "Code changed",
    "code_changes": [...],
    "dep_changes": [...],
    "param_changes": []
  }
}
```

### stage_info

```json
{"jsonrpc": "2.0", "id": 1, "method": "stage_info", "params": {"stage": "train"}}
```

Returns stage metadata: deps and outs paths.

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "name": "train",
    "deps": ["data/input.csv"],
    "outs": ["models/model.pkl"]
  }
}
```

### Existing Methods (unchanged)

- `run` — Trigger pipeline execution
- `cancel` — Cancel running execution
- `status` — Get engine state (idle/active)
- `stages` — List all stage names
