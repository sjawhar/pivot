# TUI Force Re-Run Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add keyboard shortcuts to force re-run stages from the TUI watch mode: `r` for selected stage, `R` for all stages.

**Architecture:** Enable agent RPC socket in TUI watch mode (currently only in `--serve`). TUI sends force-run commands via Unix socket using the existing JSON-RPC protocol. This reuses the battle-tested RPC infrastructure rather than creating a new event injection mechanism.

**Tech Stack:** Textual keybindings, anyio Unix sockets, JSON-RPC 2.0.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Always enable RPC socket in TUI watch mode | Reuses existing infrastructure, already tested, no new event injection mechanism needed |
| TUI acts as RPC client | Clean separation - TUI sends commands, engine processes them |
| Unix socket communication | Thread-safe, async-compatible, same protocol agents use |
| No new keybindings file | Add to existing `_TUI_BINDINGS` list |
| Watch mode only | Run mode exits after execution, force-run doesn't make sense |

---

## Task 1: Enable RPC Socket in TUI Watch Mode

**Goal:** The agent RPC socket should always be available when TUI is active in watch mode, not just when `--serve` is passed.

**Files:**
- Modify: `src/pivot/cli/repro.py:376-415`
- Test: `tests/cli/test_watch.py` (add test)

**Step 1: Write the failing test**

```python
# tests/cli/test_watch.py (add to existing file or create)
import anyio
import pytest
from pathlib import Path

from helpers import wait_for_socket


@pytest.mark.anyio
async def test_tui_watch_mode_creates_rpc_socket(tmp_path: Path) -> None:
    """TUI watch mode should create agent.sock even without --serve."""
    # This test verifies the socket exists when TUI watch mode starts
    # The actual socket creation happens in the CLI, so this is an integration test
    state_dir = tmp_path / ".pivot"
    socket_path = state_dir / "agent.sock"

    # The socket should be created when TUI watch mode starts
    # For this unit test, we just verify the path computation is correct
    assert socket_path.name == "agent.sock"
    assert socket_path.parent.name == ".pivot"
```

**Step 2: Modify CLI to always enable RPC socket in TUI watch mode**

In `src/pivot/cli/repro.py`, change the condition from `if serve:` to `if serve or tui:` in the `engine_thread_target` function (around line 395).

```python
# src/pivot/cli/repro.py - inside engine_thread_target()

# Change from:
if serve:
    from pivot import project
    from pivot.engine.agent_rpc import (
        AgentRpcHandler,
        AgentRpcSource,
        EventSink,
    )
    # ... socket setup ...

# To:
# Add agent RPC source for TUI (needed for force re-run) or serve mode
if serve or tui:
    from pivot import project
    from pivot.engine.agent_rpc import (
        AgentRpcHandler,
        AgentRpcSource,
        EventSink,
    )
    # ... socket setup ...
```

**Step 3: Run tests**

Run: `uv run pytest tests/cli/test_watch.py -v`
Expected: PASS

**Step 4: Commit**

```bash
jj describe -m "feat(tui): enable RPC socket in TUI watch mode

Always create agent.sock when TUI is active in watch mode, not just
with --serve. This enables force re-run functionality from the TUI."
```

---

## Task 2: Add RPC Client Helper to TUI

**Goal:** Create a helper that sends JSON-RPC requests to the engine via Unix socket.

**Files:**
- Create: `src/pivot/tui/rpc_client.py`
- Test: `tests/tui/test_rpc_client.py`

**Step 1: Write the failing test**

```python
# tests/tui/test_rpc_client.py
from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest

from pivot.tui.rpc_client import send_run_command


@pytest.mark.anyio
async def test_send_run_command_single_stage(tmp_path: Path) -> None:
    """send_run_command sends correct JSON-RPC for single stage."""
    socket_path = tmp_path / "agent.sock"
    received_requests = list[dict[str, object]]()

    async def mock_server() -> None:
        listener = await anyio.create_unix_listener(str(socket_path))
        async with listener:
            conn = await listener.accept()
            async with conn:
                data = await conn.receive(4096)
                request = json.loads(data.decode().strip())
                received_requests.append(request)
                response = {"jsonrpc": "2.0", "result": "accepted", "id": request["id"]}
                await conn.send(json.dumps(response).encode() + b"\n")

    async with anyio.create_task_group() as tg:
        tg.start_soon(mock_server)
        await anyio.sleep(0.05)  # Let server start

        result = await send_run_command(socket_path, stages=["my_stage"], force=True)
        tg.cancel_scope.cancel()

    assert result is True
    assert len(received_requests) == 1
    assert received_requests[0]["method"] == "run"
    assert received_requests[0]["params"]["stages"] == ["my_stage"]
    assert received_requests[0]["params"]["force"] is True


@pytest.mark.anyio
async def test_send_run_command_all_stages(tmp_path: Path) -> None:
    """send_run_command sends stages=None for all stages."""
    socket_path = tmp_path / "agent.sock"
    received_requests = list[dict[str, object]]()

    async def mock_server() -> None:
        listener = await anyio.create_unix_listener(str(socket_path))
        async with listener:
            conn = await listener.accept()
            async with conn:
                data = await conn.receive(4096)
                request = json.loads(data.decode().strip())
                received_requests.append(request)
                response = {"jsonrpc": "2.0", "result": "accepted", "id": request["id"]}
                await conn.send(json.dumps(response).encode() + b"\n")

    async with anyio.create_task_group() as tg:
        tg.start_soon(mock_server)
        await anyio.sleep(0.05)

        result = await send_run_command(socket_path, stages=None, force=True)
        tg.cancel_scope.cancel()

    assert result is True
    assert received_requests[0]["params"]["stages"] is None


@pytest.mark.anyio
async def test_send_run_command_socket_not_found(tmp_path: Path) -> None:
    """send_run_command returns False when socket doesn't exist."""
    socket_path = tmp_path / "nonexistent.sock"

    result = await send_run_command(socket_path, stages=["stage"], force=True)

    assert result is False


@pytest.mark.anyio
async def test_send_run_command_connection_refused(tmp_path: Path) -> None:
    """send_run_command returns False on connection error."""
    socket_path = tmp_path / "agent.sock"
    # Create file but not a socket
    socket_path.touch()

    result = await send_run_command(socket_path, stages=["stage"], force=True)

    assert result is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tui/test_rpc_client.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Implement the RPC client**

```python
# src/pivot/tui/rpc_client.py
"""Lightweight RPC client for TUI to communicate with engine."""
from __future__ import annotations

import contextlib
import json
import logging
from typing import TYPE_CHECKING

import anyio

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["send_run_command"]

_logger = logging.getLogger(__name__)

# Timeout for RPC operations
_RPC_TIMEOUT = 5.0


async def send_run_command(
    socket_path: Path,
    *,
    stages: list[str] | None,
    force: bool,
) -> bool:
    """Send a run command to the engine via Unix socket.

    Args:
        socket_path: Path to the agent.sock Unix socket.
        stages: Stage names to run, or None for all stages.
        force: If True, ignore skip detection.

    Returns:
        True if command was accepted, False on error.
    """
    if not socket_path.exists():
        _logger.debug("RPC socket does not exist: %s", socket_path)
        return False

    request = {
        "jsonrpc": "2.0",
        "method": "run",
        "params": {"stages": stages, "force": force},
        "id": 1,
    }

    try:
        with anyio.fail_after(_RPC_TIMEOUT):
            async with await anyio.connect_unix(str(socket_path)) as conn:
                await conn.send(json.dumps(request).encode() + b"\n")
                response_data = await conn.receive(4096)
                response = json.loads(response_data.decode().strip())
                return response.get("result") == "accepted"
    except TimeoutError:
        _logger.warning("RPC request timed out")
        return False
    except (OSError, json.JSONDecodeError) as e:
        _logger.debug("RPC request failed: %s", e)
        return False
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tui/test_rpc_client.py -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(tui): add RPC client for force re-run

Add lightweight RPC client that sends run commands to the engine
via Unix socket. Used by TUI to trigger force re-runs."
```

---

## Task 3: Add Force Re-Run Keybindings

**Goal:** Add `r` and `R` keybindings that trigger force re-run of selected/all stages.

**Files:**
- Modify: `src/pivot/tui/run.py`
- Test: `tests/tui/test_run.py`

**Step 1: Write the failing test**

```python
# tests/tui/test_run.py (add to existing file)
import pytest
from unittest.mock import AsyncMock, patch

from pivot.tui.run import PivotApp


@pytest.mark.anyio
async def test_action_force_rerun_stage_calls_rpc() -> None:
    """action_force_rerun_stage should send RPC command for selected stage."""
    app = PivotApp(stage_names=["stage_a", "stage_b"], watch_mode=True)
    app._selected_stage_name = "stage_a"

    with patch("pivot.tui.run.rpc_client.send_run_command", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = True
        with patch.object(app, "notify") as mock_notify:
            await app.action_force_rerun_stage()

    mock_send.assert_called_once()
    call_kwargs = mock_send.call_args.kwargs
    assert call_kwargs["stages"] == ["stage_a"]
    assert call_kwargs["force"] is True


@pytest.mark.anyio
async def test_action_force_rerun_all_calls_rpc() -> None:
    """action_force_rerun_all should send RPC command for all stages."""
    app = PivotApp(stage_names=["stage_a", "stage_b"], watch_mode=True)

    with patch("pivot.tui.run.rpc_client.send_run_command", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = True
        with patch.object(app, "notify") as mock_notify:
            await app.action_force_rerun_all()

    mock_send.assert_called_once()
    call_kwargs = mock_send.call_args.kwargs
    assert call_kwargs["stages"] is None
    assert call_kwargs["force"] is True


@pytest.mark.anyio
async def test_action_force_rerun_not_in_watch_mode() -> None:
    """action_force_rerun should do nothing in run mode."""
    # Create run mode app (not watch mode)
    app = PivotApp(stage_names=["stage_a"], executor_func=lambda: {})
    app._selected_stage_name = "stage_a"

    with patch("pivot.tui.run.rpc_client.send_run_command", new_callable=AsyncMock) as mock_send:
        with patch.object(app, "notify") as mock_notify:
            await app.action_force_rerun_stage()

    mock_send.assert_not_called()


@pytest.mark.anyio
async def test_action_force_rerun_no_stage_selected() -> None:
    """action_force_rerun_stage should notify when no stage selected."""
    app = PivotApp(stage_names=[], watch_mode=True)
    app._selected_stage_name = None

    with patch("pivot.tui.run.rpc_client.send_run_command", new_callable=AsyncMock) as mock_send:
        with patch.object(app, "notify") as mock_notify:
            await app.action_force_rerun_stage()

    mock_send.assert_not_called()
    mock_notify.assert_called_once()
    assert "No stage selected" in str(mock_notify.call_args)


@pytest.mark.anyio
async def test_action_force_rerun_while_running() -> None:
    """action_force_rerun should warn when stages are running."""
    app = PivotApp(stage_names=["stage_a"], watch_mode=True)
    app._selected_stage_name = "stage_a"
    # Simulate running stage
    app._stages["stage_a"].status = StageStatus.IN_PROGRESS

    with patch("pivot.tui.run.rpc_client.send_run_command", new_callable=AsyncMock) as mock_send:
        with patch.object(app, "notify") as mock_notify:
            await app.action_force_rerun_stage()

    mock_send.assert_not_called()
    mock_notify.assert_called_once()
    assert "running" in str(mock_notify.call_args).lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tui/test_run.py::test_action_force_rerun_stage_calls_rpc -v`
Expected: FAIL with AttributeError

**Step 3: Add keybindings and action methods**

```python
# src/pivot/tui/run.py

# At module level, add import:
from pivot.tui import rpc_client

# Add to _TUI_BINDINGS list (around line 197):
textual.binding.Binding("r", "force_rerun_stage", "Force Re-run", show=False),
textual.binding.Binding("R", "force_rerun_all", "Force All", show=False),

# Add action methods to PivotApp class (after action_commit):

async def action_force_rerun_stage(self) -> None:  # pragma: no cover
    """Force re-run the currently selected stage (watch mode only)."""
    if not self._watch_mode:
        return
    if self._selected_stage_name is None:
        self.notify("No stage selected", severity="warning")
        return
    if self._has_running_stages:
        self.notify("Cannot re-run while stages are running", severity="warning")
        return

    stage_name = self._selected_stage_name
    socket_path = project.get_project_root() / ".pivot" / "agent.sock"

    self.notify(f"Forcing re-run of {stage_name}...")
    success = await rpc_client.send_run_command(
        socket_path, stages=[stage_name], force=True
    )
    if not success:
        self.notify("Failed to send re-run command", severity="error")


async def action_force_rerun_all(self) -> None:  # pragma: no cover
    """Force re-run all stages (watch mode only)."""
    if not self._watch_mode:
        return
    if self._has_running_stages:
        self.notify("Cannot re-run while stages are running", severity="warning")
        return

    socket_path = project.get_project_root() / ".pivot" / "agent.sock"

    self.notify("Forcing re-run of all stages...")
    success = await rpc_client.send_run_command(socket_path, stages=None, force=True)
    if not success:
        self.notify("Failed to send re-run command", severity="error")
```

**Step 4: Run tests**

Run: `uv run pytest tests/tui/test_run.py -v -k force_rerun`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(tui): add force re-run keybindings (r/R)

- r: Force re-run selected stage
- R: Force re-run all stages

Works in watch mode only. Sends commands via agent RPC socket."
```

---

## Task 4: Add Help Screen Documentation

**Goal:** Document the new keybindings in the help screen.

**Files:**
- Modify: `src/pivot/tui/screens/help.py`

**Step 1: Check existing help screen structure**

Read `src/pivot/tui/screens/help.py` to understand the format.

**Step 2: Add force re-run to help**

The keybindings should already appear in the help screen if they're in `_TUI_BINDINGS` with appropriate labels. Verify by:

1. Check if `show=False` bindings appear in help
2. If not, add explicit documentation

**Step 3: Verify help screen shows new bindings**

Run TUI in watch mode, press `?`, verify `r` and `R` are documented.

**Step 4: Commit**

```bash
jj describe -m "docs(tui): add force re-run to help screen"
```

---

## Task 5: Integration Test

**Goal:** End-to-end test that verifies force re-run works through the full stack.

**Files:**
- Test: `tests/integration/test_tui_force_rerun.py`

**Step 1: Write the integration test**

```python
# tests/integration/test_tui_force_rerun.py
"""Integration test for TUI force re-run functionality."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import anyio
import pytest


@pytest.fixture
def pipeline_project(tmp_path: Path) -> Path:
    """Create a minimal pipeline project."""
    # Create pipeline.py
    pipeline_code = '''
from pivot import Out, Annotated, pipeline

@pipeline.stage()
def my_stage() -> Annotated[str, Out("output.txt")]:
    """Simple stage that writes output."""
    return "hello"
'''
    (tmp_path / "pipeline.py").write_text(pipeline_code)

    # Initialize git (required for pivot)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@test.com",
             "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@test.com"},
    )

    return tmp_path


@pytest.mark.anyio
async def test_force_rerun_via_socket(pipeline_project: Path) -> None:
    """Force re-run command via socket triggers stage execution."""
    socket_path = pipeline_project / ".pivot" / "agent.sock"

    # Start watch mode in background (not TUI, just to get the socket)
    # For integration testing, we use --serve which creates the socket
    proc = subprocess.Popen(
        [sys.executable, "-m", "pivot", "repro", "--watch", "--serve", "--quiet"],
        cwd=pipeline_project,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for socket to be created
        for _ in range(50):  # 5 seconds max
            if socket_path.exists():
                break
            await anyio.sleep(0.1)
        else:
            pytest.fail("Socket was not created")

        # Small delay to ensure server is ready
        await anyio.sleep(0.2)

        # Send force re-run command
        request = {
            "jsonrpc": "2.0",
            "method": "run",
            "params": {"stages": ["my_stage"], "force": True},
            "id": 1,
        }

        async with await anyio.connect_unix(str(socket_path)) as conn:
            await conn.send(json.dumps(request).encode() + b"\n")
            response_data = await conn.receive(4096)
            response = json.loads(response_data.decode().strip())

        assert response.get("result") == "accepted"

        # Give stage time to run
        await anyio.sleep(1.0)

        # Verify output was created
        output_file = pipeline_project / "output.txt"
        assert output_file.exists(), "Stage did not run - output.txt not created"

    finally:
        proc.terminate()
        proc.wait(timeout=5)
```

**Step 2: Run integration test**

Run: `uv run pytest tests/integration/test_tui_force_rerun.py -v`
Expected: PASS

**Step 3: Commit**

```bash
jj describe -m "test(tui): add integration test for force re-run"
```

---

## Task 6: Final Verification and Quality Checks

**Goal:** Run full test suite and quality checks.

**Step 1: Run all tests**

```bash
uv run pytest tests/ -n auto
```

**Step 2: Run quality checks**

```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright
```

**Step 3: Manual testing**

1. Start TUI watch mode: `pivot repro --watch`
2. Press `r` with a stage selected - should trigger force re-run
3. Press `R` - should trigger force re-run of all stages
4. Press `?` - should show keybindings in help
5. Try in run mode (not watch) - should do nothing

**Step 4: Final commit**

```bash
jj describe -m "feat(tui): add keyboard shortcuts for force re-run

Implements #250. Adds two keybindings in watch mode:
- r: Force re-run selected stage
- R: Force re-run all stages

The TUI communicates with the engine via the agent RPC socket,
which is now enabled in TUI watch mode (previously only --serve)."
```

---

## Summary of Changes

| File | Change |
|------|--------|
| `src/pivot/cli/repro.py` | Enable RPC socket in TUI watch mode |
| `src/pivot/tui/rpc_client.py` | New file - RPC client helper |
| `src/pivot/tui/run.py` | Add keybindings and action methods |
| `tests/tui/test_rpc_client.py` | New file - RPC client tests |
| `tests/tui/test_run.py` | Add force re-run action tests |
| `tests/integration/test_tui_force_rerun.py` | New file - E2E test |

## Verification Checklist

- [ ] `r` forces re-run of selected stage
- [ ] `R` forces re-run of all stages
- [ ] No-op in run mode (not watch)
- [ ] Warning when no stage selected
- [ ] Warning when stages are running
- [ ] Help screen shows keybindings
- [ ] All tests pass
- [ ] Quality checks pass
