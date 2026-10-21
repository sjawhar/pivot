"""Run serve in-process, as an application that embeds pivot's CLI does, and report what the
application sees.

Usage: <venv>/bin/python inproc_serve.py <project> <main|click> <TERM|INT|thread>

``main`` calls the public ``pivot.cli.main()``; ``click`` calls the Click group with
``standalone_mode=False``. With TERM or INT, serve runs on the main thread until the probe sends
itself that signal; the probe then reports its stop handlers and whether SIGTERM stops a child
it starts afterwards. With ``thread``, serve runs on a worker thread; the probe reports whether
it answers and its stop handlers, then exits with serve still running.
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import threading
import time

from pivot import cli

_SOCKET = pathlib.Path(".pivot/agent.sock")


def _handlers() -> str:
    names = [
        getattr(handler, "__name__", None) or str(handler)
        for handler in (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    ]
    return f"SIGINT={names[0]} SIGTERM={names[1]}"


def _status() -> str | None:
    with socket.socket(socket.AF_UNIX) as client:
        try:
            client.connect(str(_SOCKET))
            client.sendall(b'{"jsonrpc": "2.0", "method": "status", "params": {}, "id": 1}\n')
            return client.recv(65536).decode()
        except OSError:
            return None


def _serve(entry: str) -> str:
    """Run serve through ``entry``; return how the call ended."""
    try:
        if entry == "main":
            sys.argv = ["pivot", "repro", "--watch", "--serve"]
            cli.main()
        else:
            cli.cli.main(["repro", "--watch", "--serve"], standalone_mode=False)
    except BaseException as error:  # noqa: BLE001 - report what the embedder observes
        return f"raised {error!r}"
    return "returned"


def _serve_until_signalled(entry: str, sig: signal.Signals) -> None:
    def _signal_once_serving() -> None:
        deadline = time.monotonic() + 30
        while '"result"' not in (_status() or "") and time.monotonic() < deadline:
            time.sleep(0.1)
        time.sleep(1)
        os.kill(os.getpid(), sig)

    print(f"before: {_handlers()}")
    threading.Thread(target=_signal_once_serving, daemon=True).start()
    started = time.monotonic()
    outcome = _serve(entry)
    print(
        f"serve {outcome} {time.monotonic() - started:.1f} s after start; "
        + f"socket left: {_SOCKET.exists()}"
    )
    print(f"after: {_handlers()}")
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
        print(f"child: SigIgn={ignored}; SIGTERM stopped it ({child.returncode})")
    except subprocess.TimeoutExpired:
        print(f"child: SigIgn={ignored}; still running 2 s after SIGTERM")
        child.kill()
        child.wait()


def _serve_on_a_worker_thread(entry: str) -> None:
    outcome = list[str]()
    print(f"before: {_handlers()}")
    thread = threading.Thread(target=lambda: outcome.append(_serve(entry)), daemon=True)
    thread.start()
    answer = None
    deadline = time.monotonic() + 60
    while answer is None and thread.is_alive() and time.monotonic() < deadline:
        answer = _status()
        if answer is None:
            time.sleep(0.1)
    print(f"serving: {_handlers()}")
    print(json.dumps({"answer": answer, "alive": thread.is_alive(), "outcome": outcome}))
    sys.stdout.flush()
    # Nothing can stop serve in another thread but the process's own exit, as on main.
    os._exit(0)


def main() -> None:
    project, entry, stop = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    os.chdir(project)
    if stop == "thread":
        _serve_on_a_worker_thread(entry)
    else:
        _serve_until_signalled(entry, getattr(signal, f"SIG{stop}"))


if __name__ == "__main__":
    main()
