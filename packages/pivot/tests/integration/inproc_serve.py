"""Run serve in-process, as an application that embeds pivot's CLI does, and report what the
application sees.

Usage: <venv>/bin/python inproc_serve.py <project> <main|click> <handler> [library]

``main`` calls the public ``pivot.cli.main()``; ``click`` calls the Click group with
``standalone_mode=False``. ``handler`` is the application's own SIGTERM handler:

- ``default``: none.
- ``python``: a Python function that records each SIGTERM.
- ``c-before``: a C handler that ``library`` installed from ``LD_PRELOAD`` before Python started.
- ``c-after``: a C handler that ``library`` installs through ctypes now, after Python started.

The application also sets a signal wakeup fd. It records its signal state before serve, while
serve answers, and after serve returns. While serve answers it sends itself SIGTERM (except with
``default``, where that would kill it) and asks serve for ``status`` again, then stops serve with
Ctrl-C. Last, it checks that SIGTERM stops a child it starts. It prints one JSON object.

With ``thread`` instead of a handler, serve runs on a worker thread; the probe reports whether
it answers and its stop handlers, then exits with serve still running.
"""

from __future__ import annotations

import ctypes
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


def _handler_names() -> dict[str, str]:
    return {
        name: getattr(handler, "__name__", None) or str(handler)
        for name, handler in (
            ("SIGINT", signal.getsignal(signal.SIGINT)),
            ("SIGTERM", signal.getsignal(signal.SIGTERM)),
        )
    }


def _handlers() -> str:
    names = _handler_names()
    return f"SIGINT={names['SIGINT']} SIGTERM={names['SIGTERM']}"


def _kernel() -> dict[str, str]:
    """Return, for SIGINT and SIGTERM, whether the kernel reports each blocked, ignored and
    caught for this process (SigBlk, SigIgn and SigCgt in /proc/self/status)."""
    masks = {
        key: int(value, 16)
        for key, _, value in (
            line.partition(":")
            for line in pathlib.Path("/proc/self/status").read_text().splitlines()
        )
        if key in ("SigBlk", "SigIgn", "SigCgt")
    }
    return {
        f"{name}:{key}": "yes" if masks[key] & (1 << (signum - 1)) else "no"
        for name, signum in (("SIGINT", signal.SIGINT), ("SIGTERM", signal.SIGTERM))
        for key in ("SigBlk", "SigIgn", "SigCgt")
    }


def _state() -> dict[str, object]:
    """Return the signal state an application owns, read on the main thread."""
    wakeup_fd = signal.set_wakeup_fd(-1)
    signal.set_wakeup_fd(wakeup_fd)
    return {"handlers": _handler_names(), "kernel": _kernel(), "wakeup_fd": wakeup_fd}


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


def _serve_with_own_signal_state(entry: str, handler: str, library: str | None) -> None:
    caught = list[int]()
    if handler == "python":
        signal.signal(signal.SIGTERM, lambda signum, _frame: caught.append(signum))
    elif handler == "c-after":
        assert library is not None, "c-after needs the library's path"
        assert ctypes.CDLL(library).install_sigterm_handler() == 0, "sigaction failed"
    wakeup_reader, wakeup_writer = socket.socketpair()
    wakeup_reader.setblocking(False)
    wakeup_writer.setblocking(False)
    signal.set_wakeup_fd(wakeup_writer.fileno())
    report: dict[str, object] = {"before": _state()}

    def _stop_once_serving() -> None:
        deadline = time.monotonic() + 30
        while '"result"' not in (_status() or "") and time.monotonic() < deadline:
            time.sleep(0.1)
        time.sleep(0.5)
        report["serving"] = {"handlers": _handler_names(), "kernel": _kernel()}
        if handler != "default":
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)
            report["answers_after_sigterm"] = '"result"' in (_status() or "")
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=_stop_once_serving, daemon=True).start()
    report["outcome"] = _serve(entry)
    report["after"] = _state()
    report["socket_left"] = _SOCKET.exists()
    report["python_handler_ran"] = caught
    try:
        report["wakeup_bytes"] = sorted(set(wakeup_reader.recv(4096)))
    except BlockingIOError:
        report["wakeup_bytes"] = []
    child = subprocess.Popen(["/bin/sleep", "30"])
    time.sleep(0.2)
    child.send_signal(signal.SIGTERM)
    try:
        report["child_after_sigterm"] = child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        report["child_after_sigterm"] = "still running 2 s after SIGTERM"
        child.kill()
        child.wait()
    print(json.dumps(report), flush=True)


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
    project, entry, handler = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    library = sys.argv[4] if len(sys.argv) > 4 else None
    os.chdir(project)
    if handler == "thread":
        _serve_on_a_worker_thread(entry)
    else:
        _serve_with_own_signal_state(entry, handler, library)


if __name__ == "__main__":
    main()
