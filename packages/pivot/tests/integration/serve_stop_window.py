"""Run serve as the console script does, stop it with SIGTERM, and send SIGTERM again right after
every later change of SIGTERM's disposition to SIG_DFL or SIG_IGN.

Usage: <venv>/bin/python serve_stop_window.py <project>

A SIGTERM that arrives while the disposition is SIG_DFL kills the process with -15; one that
arrives under SIG_IGN is discarded. The probe writes one ``probe:`` line to stderr per resend.
"""

from __future__ import annotations

import os
import pathlib
import signal
import socket
import sys
import threading
import time

from pivot import cli

_SOCKET = pathlib.Path(".pivot/agent.sock")
_original_signal = signal.signal
_stopping = threading.Event()


def _signal_then_resend(signum: int, handler: signal.Handlers | object) -> signal.Handlers | object:
    previous = _original_signal(signum, handler)  # pyright: ignore[reportArgumentType] - passes through what pivot or asyncio set
    if (
        _stopping.is_set()
        and signum == signal.SIGTERM
        and handler in (signal.SIG_DFL, signal.SIG_IGN)
    ):
        sys.stderr.write(f"probe: SIGTERM sent again at {signal.Handlers(handler).name}\n")
        sys.stderr.flush()
        os.kill(os.getpid(), signal.SIGTERM)
    return previous


def _answers_status() -> bool:
    with socket.socket(socket.AF_UNIX) as client:
        try:
            client.connect(str(_SOCKET))
            client.sendall(b'{"jsonrpc": "2.0", "method": "status", "params": {}, "id": 1}\n')
            return b'"result"' in client.recv(65536)
        except OSError:
            return False


def _stop_once_serving() -> None:
    deadline = time.monotonic() + 30
    while not _answers_status() and time.monotonic() < deadline:
        time.sleep(0.1)
    time.sleep(1)
    _stopping.set()
    os.kill(os.getpid(), signal.SIGTERM)


def main() -> None:
    os.chdir(sys.argv[1])
    signal.signal = _signal_then_resend
    threading.Thread(target=_stop_once_serving, daemon=True).start()
    sys.argv = ["pivot", "repro", "--watch", "--serve"]
    cli._console_main()


if __name__ == "__main__":
    main()
