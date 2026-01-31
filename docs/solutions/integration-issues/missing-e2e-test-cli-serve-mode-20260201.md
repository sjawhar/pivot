---
module: CLI Run
date: 2026-02-01
problem_type: integration_issue
component: testing_framework
symptoms:
  - "AgentRpcSource queries return 'Method not found' error"
  - "Agents cannot subscribe to stage events"
  - "Unit tests pass but feature doesn't work end-to-end"
root_cause: missing_workflow_step
resolution_type: code_fix
severity: high
tags: [integration-testing, e2e-testing, serve-mode, agent-rpc]
---

# Troubleshooting: Components Work Individually But Fail When Wired Together

## Problem

The serve mode feature (`pivot run --watch --serve`) had unit tests for all components (`AgentRpcSource`, `AgentRpcHandler`, `AgentEventSink`) that passed, but the CLI integration was broken - status queries returned "Method not found" and agents couldn't subscribe to events.

## Environment

- Module: CLI Run / Engine
- Python Version: 3.13+
- Affected Component: `_run_serve_mode` function in `src/pivot/cli/run.py`
- Date: 2026-02-01

## Symptoms

- `AgentRpcSource` queries (status, stages) return JSON-RPC error: `{"code": -32601, "message": "Method not found"}`
- Agents connecting to the Unix socket cannot subscribe to stage events
- All unit tests pass (17 tests for agent_rpc.py)
- All integration tests for individual components pass
- Bug only discovered via PR review (Copilot comments)

## What Didn't Work

**Unit tests for components:** Each component had comprehensive tests:
- `AgentRpcSource` - tested socket creation, run/cancel commands, message handling
- `AgentRpcHandler` - tested status and stages queries
- `AgentEventSink` - tested subscribe/unsubscribe/broadcast

**Why they failed to catch the bug:** Unit tests verify components work in isolation. They don't verify the CLI correctly wires components together. The `_run_serve_mode` function created `AgentRpcSource` without passing a handler, and never added `AgentEventSink` to the engine.

## Solution

**1. Fixed the CLI integration code:**

```python
# Before (broken) - src/pivot/cli/run.py:397
eng.add_source(AgentRpcSource(socket_path=socket_path))

# After (fixed)
rpc_handler = AgentRpcHandler(engine=eng)
eng.add_source(AgentRpcSource(socket_path=socket_path, handler=rpc_handler))
```

```python
# Before (missing) - no AgentEventSink added

# After (fixed) - src/pivot/cli/run.py:405
eng.add_sink(AgentEventSink())  # Broadcast events to connected agents
```

**2. Added E2E integration test:**

```python
def test_serve_mode_cli_responds_to_status_query(tmp_path):
    """E2E test: pivot run --watch --serve creates working RPC endpoint."""
    # 1. Start serve mode as subprocess
    proc = subprocess.Popen(
        ["uv", "run", "pivot", "run", "--watch", "--serve"],
        cwd=tmp_path, ...
    )

    # 2. Wait for socket, connect, send query
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(str(socket_path))
        request = '{"jsonrpc":"2.0","method":"status","id":1}\n'
        sock.sendall(request.encode())
        response = json.loads(sock.recv(1024).decode().strip())

    # 3. Verify valid response (not "Method not found")
    assert "error" not in response
    assert response["result"]["state"] in ("idle", "active")
```

## Why This Works

**Root cause:** Missing integration test coverage. The feature implementation was incomplete because:

1. `AgentRpcSource` can operate without a handler (for run/cancel commands only), so no error occurs during initialization
2. `AgentEventSink` is optional - the engine runs fine without it
3. Unit tests tested each component's individual behavior, not the assembled system

**Why the fix works:**
1. The handler enables status/stages queries to be processed
2. The event sink enables pub-sub event broadcast to connected agents
3. The E2E test catches any future regressions by testing the complete CLI path

## Prevention

**Added rule to tests/CLAUDE.md:**

> Major features (new CLI modes, protocols, architectural components) need E2E tests that exercise the complete path—unit tests for components are insufficient since they can pass individually but fail when wired together.
>
> **E2E test pattern:**
> 1. Start the actual CLI command (subprocess if async)
> 2. Exercise through public interface (socket, HTTP, filesystem)
> 3. Verify end-to-end behavior, not just component existence

**Key insight:** When implementing a feature that involves multiple components being wired together:
- Unit tests verify components work
- E2E tests verify the system works
- Both are required

## Related Issues

- **Critical Pattern #1:** [E2E Tests Required for Major Features](../patterns/critical-patterns.md#1-e2e-tests-required-for-major-features-always-required)
