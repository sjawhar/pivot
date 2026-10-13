"""Test helpers for TUI tests."""

# pyright: reportMissingImports=false

from __future__ import annotations

import stat
from typing import TYPE_CHECKING

import anyio

if TYPE_CHECKING:
    import pathlib


async def wait_for_socket(socket_path: pathlib.Path, timeout: float = 5.0) -> None:
    """Wait for Unix socket to be created and connectable.

    Args:
        socket_path: Path to the Unix socket file.
        timeout: Maximum time to wait in seconds.

    Raises:
        TimeoutError: If socket is not created within timeout.
    """
    deadline = anyio.current_time() + timeout
    while anyio.current_time() < deadline:
        if socket_path.exists():
            try:
                mode = socket_path.stat().st_mode
                if stat.S_ISSOCK(mode):
                    connection = await anyio.connect_unix(str(socket_path))
                    await connection.aclose()
                    return
            except (OSError, anyio.BrokenResourceError, anyio.ClosedResourceError):
                pass
        await anyio.sleep(0.05)
    msg = f"Socket {socket_path} not created and connectable within {timeout}s"
    raise TimeoutError(msg)
