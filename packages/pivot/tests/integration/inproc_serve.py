"""Run serve in-process, stop it, then report whether the caller can stop a new child.

Usage: <venv>/bin/python inproc_serve.py <project> <TERM|INT>
"""

from __future__ import annotations

import os
import pathlib
import signal
import socket
import subprocess
import sys
import threading
import time

from pivot import cli


def _handler_name(handler: object) -> str:
    return getattr(handler, "__name__", None) or str(handler)


def main() -> None:
    project, signame = pathlib.Path(sys.argv[1]), sys.argv[2]
    sig = getattr(signal, f"SIG{signame}")
    os.chdir(project)
    sock = pathlib.Path(".pivot/agent.sock")
    if sock.exists():
        sock.unlink()

    def _stop_when_serving() -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_UNIX) as client:
                try:
                    client.connect(str(sock))
                    client.sendall(
                        b'{"jsonrpc": "2.0", "method": "status", "params": {}, "id": 1}\n'
                    )
                    if b'"result"' in client.recv(65536):
                        break
                except OSError:
                    time.sleep(0.1)
        time.sleep(1)
        os.kill(os.getpid(), sig)

    before = tuple(
        _handler_name(signal.getsignal(signum)) for signum in (signal.SIGINT, signal.SIGTERM)
    )
    print(f"before: SIGINT={before[0]} SIGTERM={before[1]}")
    threading.Thread(target=_stop_when_serving, daemon=True).start()
    started = time.monotonic()
    try:
        cli.cli.main(["repro", "--watch", "--serve"], standalone_mode=False)
        outcome = "returned"
    except BaseException as error:  # noqa: BLE001 - report the exception an embedder observes
        outcome = f"raised {error!r}"
    print(
        f"serve {outcome} {time.monotonic() - started:.1f} s after start; socket left: {sock.exists()}"
    )
    after = tuple(
        _handler_name(signal.getsignal(signum)) for signum in (signal.SIGINT, signal.SIGTERM)
    )
    print(f"after:  SIGINT={after[0]} SIGTERM={after[1]}")
    child = subprocess.Popen(["/bin/sleep", "30"])
    time.sleep(0.2)
    ignored = next(
        line.split()[1]
        for line in pathlib.Path(f"/proc/{child.pid}/status").read_text().splitlines()
        if line.startswith("SigIgn")
    )
    child.send_signal(signal.SIGTERM)
    try:
        child.wait(timeout=2)
        print(
            f"child started afterwards: SigIgn={ignored}; SIGTERM stopped it (returncode {child.returncode})"
        )
    except subprocess.TimeoutExpired:
        print(f"child started afterwards: SigIgn={ignored}; still running 2 s after SIGTERM")
        child.kill()
        child.wait()


if __name__ == "__main__":
    main()
