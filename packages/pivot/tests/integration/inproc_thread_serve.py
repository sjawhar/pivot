"""Run pivot's CLI in-process from a worker thread, then report whether serve answers.

Usage: <venv>/bin/python inproc_thread_serve.py <project> <main|click>
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import socket
import sys
import threading
import time

from pivot import cli


def _handlers() -> str:
    names = [
        getattr(handler, "__name__", None) or str(handler)
        for handler in (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    ]
    return f"SIGINT={names[0]} SIGTERM={names[1]}"


def _status(sock: pathlib.Path) -> str | None:
    with socket.socket(socket.AF_UNIX) as client:
        try:
            client.connect(str(sock))
            client.sendall(b'{"jsonrpc": "2.0", "method": "status", "params": {}, "id": 1}\n')
            return client.recv(65536).decode()
        except OSError:
            return None


def main() -> None:
    project, entry = pathlib.Path(sys.argv[1]), sys.argv[2]
    os.chdir(project)
    sock = pathlib.Path(".pivot/agent.sock")
    outcome = list[str]()

    def _serve() -> None:
        try:
            if entry == "main":
                sys.argv = ["pivot", "repro", "--watch", "--serve"]
                cli.main()
            else:
                cli.cli.main(["repro", "--watch", "--serve"], standalone_mode=False)
            outcome.append("returned")
        except BaseException as error:  # noqa: BLE001 - report what the embedder's thread saw
            outcome.append(f"raised {error!r}")

    print(f"before: {_handlers()}")
    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    answer = None
    deadline = time.monotonic() + 60
    while answer is None and thread.is_alive() and time.monotonic() < deadline:
        answer = _status(sock)
        if answer is None:
            time.sleep(0.1)
    print(f"serving: {_handlers()}")
    print(json.dumps({"answer": answer, "alive": thread.is_alive(), "outcome": outcome}))
    sys.stdout.flush()
    # Nothing can stop serve in another thread but the process's own exit, as on main.
    os._exit(0)


if __name__ == "__main__":
    main()
