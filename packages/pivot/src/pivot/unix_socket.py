"""Unix sockets at any path, including paths longer than the kernel accepts as an address.

A socket address (``sockaddr_un.sun_path``) holds 108 bytes on Linux and 104 on macOS,
terminating NUL included. A deeper socket is bound and reached through a short name for its
directory; the socket itself stays at its path.

On Linux that name is ``/proc/self/fd/<fd>``, a descriptor of the directory that the kernel
closes with the process, so nothing is written outside the socket's directory. Elsewhere it is
a symlink in a private temporary directory, removed when the call returns; a process killed
during the call leaves that directory behind. The non-Linux branch has not been run on macOS.

pivot and pivot-tui each carry this module, byte for byte, so that a pivot-tui release runs
beside a pivot release that has none. ``test_unix_socket.py`` checks that the copies match.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import sys
import tempfile
from typing import TYPE_CHECKING

import anyio

if TYPE_CHECKING:
    from collections.abc import Iterator

    from anyio.abc import SocketListener, SocketStream

_LINUX = sys.platform.startswith("linux")
_SUN_PATH_MAX = 108 if _LINUX else 104


@contextlib.contextmanager
def _address(path: pathlib.Path) -> Iterator[str]:
    """Yield an address the kernel accepts for the socket at ``path``."""
    if len(os.fsencode(path)) < _SUN_PATH_MAX:
        yield str(path)
        return
    if _LINUX:
        directory = os.open(path.parent, os.O_PATH | os.O_DIRECTORY)
        try:
            yield f"/proc/self/fd/{directory}/{path.name}"
        finally:
            os.close(directory)
        return
    # The directory is private (mode 0700), so no other user can swap the symlink.
    with tempfile.TemporaryDirectory(prefix="pivot-") as alias_dir:
        alias = pathlib.Path(alias_dir, "d")
        alias.symlink_to(path.parent.absolute(), target_is_directory=True)
        yield str(alias / path.name)


async def create_unix_listener(path: pathlib.Path) -> SocketListener:
    """Listen on a Unix socket created at ``path``."""
    with _address(path) as address:
        return await anyio.create_unix_listener(address)


async def connect_unix(path: pathlib.Path) -> SocketStream:
    """Connect to the Unix socket at ``path``."""
    with _address(path) as address:
        return await anyio.connect_unix(address)
