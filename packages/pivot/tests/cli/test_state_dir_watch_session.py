"""One `repro --watch` session keeps the state directory it decided when it started.

The Pivot process decides the state directory once for the whole command. A `core.state_dir`
edited while the session runs reaches neither its workers nor the fingerprint caches it writes
as it ends, as on main: the session's lock, run cache, secret key and caches stay where it began.
"""

from __future__ import annotations

import json
import queue
import signal
import subprocess
import threading
import time
import typing

import pytest

from tests import helpers

if typing.TYPE_CHECKING:
    import pathlib

_SECRET_PIPELINE = """\
import pydantic

from pivot import Pipeline, StageParams, stage


class ProcessParams(StageParams):
    token: pydantic.SecretStr = pydantic.SecretStr("watch-secret")


@stage
def process(value: str, params: ProcessParams) -> str:
    return value.upper() + str(len(params.token.get_secret_value()))


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str), params=ProcessParams())
"""

_PLAIN_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


class _Session:
    """A `pivot repro --watch --jsonl` process and the stage results it printed."""

    process: subprocess.Popen[str]
    lines: queue.Queue[str]
    stderr: list[str]

    def __init__(self, root: pathlib.Path, home: pathlib.Path) -> None:
        self.process = subprocess.Popen(
            [helpers.PIVOT, "repro", "--watch", "--jsonl", "--debounce", "50"],
            cwd=root,
            env=helpers.cli_env(home),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.lines = queue.Queue[str]()
        self.stderr = list[str]()
        assert self.process.stdout is not None and self.process.stderr is not None
        threading.Thread(target=self._read, args=(self.process.stdout,), daemon=True).start()
        threading.Thread(target=self._collect, args=(self.process.stderr,), daemon=True).start()

    def _read(self, stream: typing.TextIO) -> None:
        for line in stream:
            self.lines.put(line)

    def _collect(self, stream: typing.TextIO) -> None:
        for line in stream:
            self.stderr.append(line)

    def next_stage_result(self) -> tuple[str, str]:
        """The status and reason of the next `stage_complete` event."""
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                pytest.fail(f"watch exited {self.process.returncode}: {''.join(self.stderr)}")
            try:
                line = self.lines.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                event: object = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                typed = typing.cast("dict[str, typing.Any]", event)
                if typed.get("type") == "stage_complete":
                    return str(typed["status"]), str(typed["reason"])
        pytest.fail(f"no stage result in 60s: {''.join(self.stderr)}")

    def stop(self) -> int:
        """Stop the session with Ctrl-C and return its exit code."""
        self.process.send_signal(signal.SIGINT)
        try:
            return self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise

    def kill(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()


def _project(tmp_path: pathlib.Path, pipeline: str) -> tuple[pathlib.Path, pathlib.Path]:
    home = tmp_path / "home"
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: s1\n")
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("first\n")
    (root / "pipeline.py").write_text(pipeline)
    return home, root


def test_a_watch_session_runs_every_stage_in_its_own_state_dir(tmp_path: pathlib.Path) -> None:
    """After a `core.state_dir` edit, a stage with a secret param restores from the run cache.

    The worker digests the secret with the session's key in `s1`, as the Pivot process checks
    it, so an input change and its revert give "Input dependencies changed", then the run cache.
    """
    home, root = _project(tmp_path, _SECRET_PIPELINE)
    session = _Session(root, home)
    try:
        assert session.next_stage_result() == ("ran", "No previous run")
        (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: s2\n")
        (root / "data" / "raw" / "input.txt").write_text("second\n")
        second = session.next_stage_result()
        (root / "data" / "raw" / "input.txt").write_text("first\n")
        third = session.next_stage_result()
        returncode = session.stop()
    finally:
        session.kill()

    stderr = "".join(session.stderr)
    assert returncode == 0, stderr
    assert second == ("ran", "Input dependencies changed"), stderr
    assert third == ("cached", "unchanged (run cache)"), stderr
    assert (root / "s1" / "secret.key").is_file()
    assert not (root / "s2").exists(), sorted(p.name for p in (root / "s2").iterdir())


def test_a_watch_session_writes_its_caches_to_its_own_state_dir(tmp_path: pathlib.Path) -> None:
    """The fingerprint caches the session writes as it ends go to `s1`, not an edited `s2`."""
    home, root = _project(tmp_path, _PLAIN_PIPELINE)
    session = _Session(root, home)
    try:
        assert session.next_stage_result() == ("ran", "No previous run")
        (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: s2\n")
        returncode = session.stop()
    finally:
        session.kill()

    assert returncode == 0, "".join(session.stderr)
    assert (root / "s1" / "state.lmdb").is_dir()
    assert not (root / "s2").exists(), sorted(p.name for p in (root / "s2").iterdir())
