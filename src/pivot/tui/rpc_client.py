"""Lightweight RPC client for TUI to communicate with engine."""

from __future__ import annotations

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
                # Read until newline delimiter (protocol uses newline-delimited JSON)
                buffer = b""
                while b"\n" not in buffer:
                    chunk = await conn.receive(4096)
                    if not chunk:
                        break
                    buffer += chunk
                response: dict[str, object] = json.loads(buffer.decode().strip())
                return response.get("result") == "accepted"
    except TimeoutError:
        _logger.warning("RPC request timed out")
        return False
    except (OSError, json.JSONDecodeError, anyio.EndOfStream) as e:
        _logger.debug("RPC request failed: %s", e)
        return False
