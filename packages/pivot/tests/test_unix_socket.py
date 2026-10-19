from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING

import pytest
from anyio.abc import SocketAttribute

if TYPE_CHECKING:
    from collections.abc import Iterator

from pivot import unix_socket

# The child sets one socket method to finish its call and then hang, so the test can kill it
# while create_unix_listener or connect_unix is still running.
_KILLED_CHILD = """\
import pathlib
import socket
import sys
import time

import anyio

from pivot import unix_socket

path = pathlib.Path(sys.argv[1])
method = sys.argv[2]
original = getattr(socket.socket, method)


def _call_then_hang(self, address):
    original(self, address)
    print(method, flush=True)
    time.sleep(60)


async def main():
    listener = await unix_socket.create_unix_listener(path) if method == "connect" else None
    setattr(socket.socket, method, _call_then_hang)
    if listener is None:
        await unix_socket.create_unix_listener(path)
    else:
        await unix_socket.connect_unix(path)


anyio.run(main)
"""


@pytest.fixture
def private_tmp(monkeypatch: pytest.MonkeyPatch) -> Iterator[pathlib.Path]:
    """A temporary directory of the test's own, for tempfile here and in child processes. It is
    short, so an address under it fits however long the test's own directory is."""
    path = pathlib.Path(tempfile.mkdtemp(prefix="t", dir="/tmp"))
    monkeypatch.setenv("TMPDIR", str(path))
    monkeypatch.setattr(tempfile, "tempdir", str(path))
    yield path
    shutil.rmtree(path)


def _helper_deep_socket_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return a socket path longer than any platform's socket address, its directory made."""
    path = tmp_path.joinpath(*["nested-directory"] * 8, "agent.sock")
    path.parent.mkdir(parents=True)
    assert len(os.fsencode(path)) > 108, "the path must exceed every platform's address limit"
    return path


def _helper_entries(directory: pathlib.Path) -> list[str]:
    return sorted(str(p.relative_to(directory)) for p in directory.rglob("*"))


@pytest.mark.anyio
async def test_socket_past_the_address_limit_accepts_connections(
    tmp_path: pathlib.Path, private_tmp: pathlib.Path
) -> None:
    """A listener and a client meet at the socket's own path when it is too long to bind,
    and nothing is written outside the socket's directory."""
    path = _helper_deep_socket_path(tmp_path)

    async with await unix_socket.create_unix_listener(path) as listener:
        async with (
            await unix_socket.connect_unix(path) as client,
            await listener.accept() as server,
        ):
            await client.send(b"ping")
            assert await server.receive() == b"ping"
            assert not _helper_entries(private_tmp), "binding and connecting wrote to TMPDIR"

        assert path.is_socket(), "the socket is created at the path asked for"
    assert sorted(p.name for p in path.parent.iterdir()) == ["agent.sock"]
    assert not _helper_entries(private_tmp), "closing the sockets left entries in TMPDIR"


@pytest.mark.anyio
async def test_connect_to_missing_socket_past_the_address_limit_raises(
    tmp_path: pathlib.Path, private_tmp: pathlib.Path
) -> None:
    """A client finds no listener at a long path the way it does at a short one."""
    path = _helper_deep_socket_path(tmp_path)

    with pytest.raises(FileNotFoundError):
        await unix_socket.connect_unix(path)
    assert not _helper_entries(private_tmp), "a failed connect left entries in TMPDIR"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux's address is 108 bytes")
@pytest.mark.parametrize(
    ("length", "direct"),
    [
        pytest.param(107, True, id="longest-path-bound-as-given"),
        pytest.param(108, False, id="shortest-path-bound-through-its-directory"),
    ],
)
@pytest.mark.anyio
async def test_socket_at_the_address_limit_accepts_connections(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, length: int, direct: bool
) -> None:
    """A 107-byte path is the longest the kernel binds as given; a 108-byte one, the shortest
    it refuses, is still served at its own path. Both paths are relative to the working
    directory."""
    monkeypatch.chdir(tmp_path)
    path = pathlib.Path("x" * (length - len("/agent.sock")), "agent.sock")
    path.parent.mkdir()
    assert len(os.fsencode(path)) == length

    async with await unix_socket.create_unix_listener(path) as listener:
        async with (
            await unix_socket.connect_unix(path) as client,
            await listener.accept() as server,
        ):
            await client.send(b"ping")
            assert await server.receive() == b"ping"
        bound_as_given = listener.extra(SocketAttribute.local_address) == str(path)
        assert bound_as_given is direct, listener.extra(SocketAttribute.local_address)
        assert (tmp_path / path).is_socket(), "the socket is created at the path asked for"


@pytest.mark.anyio
async def test_symlink_branch_serves_a_relative_long_path_and_leaves_nothing_in_tmpdir(
    tmp_path: pathlib.Path, private_tmp: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The branch that has no /proc/self/fd (macOS) reaches a long relative path through a
    symlink to the socket's directory, and removes the symlink once the call returns."""
    monkeypatch.setattr(unix_socket, "_LINUX", False)
    monkeypatch.chdir(tmp_path)
    path = pathlib.Path("x" * 120, "agent.sock")
    path.parent.mkdir()

    async with await unix_socket.create_unix_listener(path) as listener:
        async with (
            await unix_socket.connect_unix(path) as client,
            await listener.accept() as server,
        ):
            await client.send(b"ping")
            assert await server.receive() == b"ping"
            assert not _helper_entries(private_tmp), "the symlink outlived the call"
        assert (tmp_path / path).is_socket(), "the socket is created at the path asked for"


@pytest.mark.parametrize("method", ["bind", "connect"])
def test_a_process_killed_inside_a_long_path_call_leaves_nothing_in_tmpdir(
    tmp_path: pathlib.Path, private_tmp: pathlib.Path, method: str
) -> None:
    """SIGKILL while create_unix_listener or connect_unix runs leaves nothing in TMPDIR."""
    path = _helper_deep_socket_path(tmp_path)
    proc = subprocess.Popen(
        [sys.executable, "-c", _KILLED_CHILD, str(path), method],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline() == f"{method}\n", f"the child did not reach {method}"
    finally:
        proc.kill()
        proc.wait()

    assert path.is_socket(), "the socket was bound at its own path"
    left = _helper_entries(private_tmp)
    assert not left, f"a process killed inside {method} left {left} in TMPDIR"
