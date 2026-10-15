from __future__ import annotations

import socket
from typing import TYPE_CHECKING

import pytest

from tests import helpers

if TYPE_CHECKING:
    import pathlib


@pytest.mark.anyio
async def test_wait_for_socket_requires_successful_connection(tmp_path: pathlib.Path) -> None:
    """A bound Unix socket that is not listening is not ready for clients."""
    socket_path = tmp_path / "bound.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as bound_socket:
        bound_socket.bind(str(socket_path))

        with pytest.raises(TimeoutError, match="connectable"):
            await helpers.wait_for_socket(socket_path, timeout=0.1)
