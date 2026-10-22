"""Red test: the `pivot` command leaves SIGTERM ignored when whatever launched it ignores it.

It belongs in `packages/pivot/tests/integration/` and runs the `pivot` script beside the
interpreter running pytest, so it also runs standalone against any tree:
`<tree>/.venv/bin/python -m pytest <this file>`.

A launcher can start `pivot repro --watch --serve` with SIGTERM ignored: `env --ignore-signal=TERM`,
`sh -c 'trap "" TERM; exec pivot ...'`, or any parent that sets SIG_IGN before it forks. A process
inherits that disposition, and Python reports it at start as `signal.getsignal(SIGTERM) is
signal.SIG_IGN`. The 11:27Z ruling on PR 493 says the console script takes a stop signal "only if
Python saw it at its default when it started"; a signal the launcher set belongs to the launcher.
Serve already keeps an inherited SIG_IGN for SIGINT (`repro.py:585`). For SIGTERM it does not: it
installs its own handler over the launcher's SIG_IGN, and the launcher's SIGTERM, which main
ignores, now stops serve.

At main (a6766ee6) serve never touches SIGTERM: the SIGTERM is ignored, serve keeps answering, and
Ctrl-C then stops it with exit 0 and no socket. The test requires exactly that. Main cannot bind a
socket path over 108 bytes, so against main give pytest a short `--basetemp`.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

import pytest

_PIVOT = pathlib.Path(sys.executable).parent / "pivot"

_PIPELINE = """\
import pathlib

from pivot.compose import Pipeline, stage


pipeline = Pipeline("test", root=pathlib.Path(__file__).parent)


@stage
def noop() -> None:
    pass


with pipeline:
    noop()
"""


def _answers(socket_path: pathlib.Path) -> bool:
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(5)
        try:
            client.connect(str(socket_path))
            client.sendall(b'{"jsonrpc": "2.0", "method": "status", "params": {}, "id": 1}\n')
            return b'"result"' in client.recv(65536)
        except OSError:
            return False


def _sigterm_state(pid: int) -> str:
    masks = {}
    for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("SigIgn", "SigCgt"):
            masks[key] = int(value, 16)
    bit = 1 << (signal.SIGTERM - 1)
    return "ignored" if masks["SigIgn"] & bit else "caught" if masks["SigCgt"] & bit else "default"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/<pid>/status")
def test_pivot_command_keeps_sigterm_ignored_when_its_launcher_ignores_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    (project / ".pivot").mkdir(parents=True)
    (project / ".git").mkdir()
    (project / "pipeline.py").write_text(_PIPELINE)
    # Short, because multiprocessing binds its manager's socket under TMPDIR.
    private_tmp = pathlib.Path(tempfile.mkdtemp(prefix="t", dir="/tmp"))
    env = {**os.environ, "TMPDIR": str(private_tmp), "PIVOT_CACHE_DIR": str(tmp_path / "cache")}
    stderr_path = tmp_path / "serve.stderr"
    socket_path = pathlib.Path(".pivot", "agent.sock")
    try:
        # The launcher ignores SIGTERM; the child inherits SIG_IGN across exec.
        previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            with stderr_path.open("wb") as stderr_file:
                proc = subprocess.Popen(
                    [str(_PIVOT), "repro", "--watch", "--serve"],
                    cwd=project,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                    start_new_session=True,
                )
        finally:
            signal.signal(signal.SIGTERM, previous)
        monkeypatch.chdir(project)
        try:
            deadline = time.monotonic() + 30
            while not _answers(socket_path) and proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            assert _answers(socket_path), f"serve did not answer (exit {proc.poll()})"
            serving = _sigterm_state(proc.pid)
            proc.send_signal(signal.SIGTERM)
            time.sleep(2)
            assert proc.poll() is None, (
                f"the launcher ignored SIGTERM, but SIGTERM stopped serve (exit {proc.returncode}); "
                + f"SIGTERM while serving was {serving}:\n{stderr_path.read_text()[-2000:]}"
            )
            assert serving == "ignored", f"serve replaced the launcher's SIG_IGN: SIGTERM {serving}"
            assert _answers(socket_path), "serve stopped answering after an ignored SIGTERM"
            proc.send_signal(signal.SIGINT)
            exit_code = proc.wait(timeout=20)
        finally:
            if proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        assert exit_code == 0, f"exit {exit_code} after Ctrl-C:\n{stderr_path.read_text()[-2000:]}"
        assert not (project / socket_path).exists(), "serve left its socket"
    finally:
        shutil.rmtree(private_tmp)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
