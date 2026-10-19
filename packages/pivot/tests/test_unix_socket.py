from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from pivot import unix_socket

if TYPE_CHECKING:
    import pathlib


def _helper_deep_socket_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return a socket path longer than any platform's socket address, its directory made."""
    path = tmp_path.joinpath(*["nested-directory"] * 8, "agent.sock")
    path.parent.mkdir(parents=True)
    assert len(os.fsencode(path)) > 108, "the path must exceed every platform's address limit"
    return path


@pytest.mark.anyio
async def test_socket_past_the_address_limit_accepts_connections(tmp_path: pathlib.Path) -> None:
    """A listener and a client meet at the socket's own path when it is too long to bind."""
    path = _helper_deep_socket_path(tmp_path)

    async with await unix_socket.create_unix_listener(path) as listener:
        async with (
            await unix_socket.connect_unix(path) as client,
            await listener.accept() as server,
        ):
            await client.send(b"ping")
            assert await server.receive() == b"ping"

        assert path.is_socket(), "the socket is created at the path asked for"
    assert sorted(p.name for p in path.parent.iterdir()) == ["agent.sock"]


@pytest.mark.anyio
async def test_connect_to_missing_socket_past_the_address_limit_raises(
    tmp_path: pathlib.Path,
) -> None:
    """A client finds no listener at a long path the way it does at a short one."""
    path = _helper_deep_socket_path(tmp_path)

    with pytest.raises(FileNotFoundError):
        await unix_socket.connect_unix(path)
