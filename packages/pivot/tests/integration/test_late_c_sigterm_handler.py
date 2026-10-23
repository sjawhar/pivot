"""An in-process caller retains a SIGTERM handler installed in C after Python starts.

It belongs in `packages/pivot/tests/integration/` and runs the interpreter running pytest, so it
also runs standalone against any tree: `<tree>/.venv/bin/python -m pytest <this file>`.

A C extension or embedding runtime (a JVM bridge or an MPI runtime) can install a SIGTERM handler
with sigaction after the interpreter starts; this test's small library does so through ctypes.
Python's ``signal.getsignal(SIGTERM)`` can still report SIG_DFL after that direct C installation.

The test stops in-process serve with Ctrl-C, then verifies that the kernel still reports SIGTERM
as caught, its handler runs, and the caller survives a SIGTERM after serve returns.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import textwrap

import pytest

from tests.integration import serve_test_support

_PYTHON = pathlib.Path(sys.executable)


_LIBRARY = r"""
#include <signal.h>
#include <string.h>
#include <unistd.h>

static void host_handler(int sig) {
    static const char message[] = "host-handler: SIGTERM\n";
    (void)sig;
    (void)!write(2, message, sizeof message - 1);
}

int install_sigterm_handler(void) {
    struct sigaction action;
    memset(&action, 0, sizeof action);
    action.sa_handler = host_handler;
    sigemptyset(&action.sa_mask);
    return sigaction(SIGTERM, &action, NULL);
}
"""

# Runs in the tree's interpreter. The `__main__` guard matters: multiprocessing's spawn start
# method re-imports this script in each child it starts.
_EMBEDDER = textwrap.dedent("""\
    import ctypes
    import os
    import pathlib
    import signal
    import socket
    import sys
    import threading
    import time


    def kernel_sigterm():
        masks = {}
        for line in pathlib.Path("/proc/self/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("SigIgn", "SigCgt"):
                masks[key] = int(value, 16)
        bit = 1 << (signal.SIGTERM - 1)
        return "ignored" if masks["SigIgn"] & bit else "caught" if masks["SigCgt"] & bit else "default"


    def answers():
        with socket.socket(socket.AF_UNIX) as client:
            try:
                client.connect(".pivot/agent.sock")
                client.sendall(b'{"jsonrpc": "2.0", "method": "status", "params": {}, "id": 1}\\n')
                return b'"result"' in client.recv(65536)
            except OSError:
                return False


    def stop_once_serving():
        deadline = time.monotonic() + 60
        while not answers() and time.monotonic() < deadline:
            time.sleep(0.1)
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGINT)


    def main():
        rc = ctypes.CDLL(sys.argv[1]).install_sigterm_handler()
        print(f"C install rc={rc}", flush=True)
        print(f"before: getsignal(SIGTERM)={signal.getsignal(signal.SIGTERM)!r} kernel SIGTERM={kernel_sigterm()}", flush=True)
        from pivot import cli

        threading.Thread(target=stop_once_serving, daemon=True).start()
        sys.argv = ["pivot", "repro", "--watch", "--serve"]
        try:
            cli.main()
            print("serve: returned", flush=True)
        except SystemExit as exc:
            print(f"serve: SystemExit({exc.code!r})", flush=True)
        print(f"after: getsignal(SIGTERM)={signal.getsignal(signal.SIGTERM)!r} kernel SIGTERM={kernel_sigterm()}", flush=True)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.5)
        print("survived SIGTERM after serve", flush=True)


    if __name__ == "__main__":
        main()
""")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self/status")
def test_inprocess_serve_leaves_a_sigterm_handler_installed_in_c_after_python_started() -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("needs a C compiler")
    root = pathlib.Path(tempfile.mkdtemp(prefix="r8l", dir="/tmp"))
    try:
        (root / "late.c").write_text(_LIBRARY)
        library = root / "late.so"
        subprocess.run(
            [compiler, "-shared", "-fPIC", "-o", str(library), str(root / "late.c")], check=True
        )
        project = root / "p"
        serve_test_support.create_noop_project(project)
        (root / "embedder.py").write_text(_EMBEDDER)
        private = root / "t"
        private.mkdir()
        proc = subprocess.run(
            [str(_PYTHON), str(root / "embedder.py"), str(library)],
            cwd=project,
            env={**os.environ, "TMPDIR": str(private), "PIVOT_CACHE_DIR": str(root / "cache")},
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        report = (
            f"exit {proc.returncode}\nstdout:\n{proc.stdout}\nstderr tail:\n{proc.stderr[-800:]}"
        )
        lines = {line.split(":", 1)[0]: line for line in proc.stdout.splitlines() if ":" in line}
        assert "kernel SIGTERM=caught" in lines.get("before", ""), (
            f"the C handler was not installed:\n{report}"
        )
        assert "serve: SystemExit(0)" in proc.stdout, (
            f"serve did not stop cleanly on Ctrl-C:\n{report}"
        )
        assert "kernel SIGTERM=caught" in lines.get("after", ""), (
            f"serve left SIGTERM without the caller's C handler:\n{report}"
        )
        assert proc.returncode == 0 and "survived SIGTERM after serve" in proc.stdout, (
            f"a SIGTERM after serve did not reach the caller's C handler:\n{report}"
        )
        assert "host-handler: SIGTERM" in proc.stderr, f"the C handler did not run:\n{report}"
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
