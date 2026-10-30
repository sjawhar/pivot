"""A `repro --watch` session runs in the state directory its pipeline loaded with.

The Pivot process decides the state directory for each load of the pipeline, and a worker gets
the decision with each stage. An edit of `core.state_dir` in `.pivot/config.yaml` reloads the
session, which reads the configuration again: the reloaded pipeline's runs and caches go to the
new directory, where `pivot gc` reads its locks, and what the first load produced stays where it
began. A move no edit announces (a re-pointed link) makes the session refuse every run (L3w).
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

from pivot.storage import state
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

    def refusal_or_stage_result(self, refusal: str) -> tuple[str, str] | None:
        """None once ``refusal`` is in stderr; the status and reason of a `stage_complete` event
        that comes first."""
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if refusal in " ".join("".join(self.stderr).split()):
                return None
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
        pytest.fail(f"neither a refusal nor a stage result in 60s: {''.join(self.stderr)}")

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


def test_a_watch_session_reloads_into_the_state_dir_a_config_edit_names(
    tmp_path: pathlib.Path,
) -> None:
    """L3w: an edit of `core.state_dir` in `.pivot/config.yaml` reloads the session, and the
    reloaded pipeline runs in `s2`, where `pivot gc` reads its locks. The first run's lock and
    secret key stay in `s1`; the re-run records its lock, under a secret key of its own, in `s2`.
    """
    home, root = _project(tmp_path, _SECRET_PIPELINE)
    s1_lock = root / "s1" / "stages" / "test" / "process.lock"
    s2_lock = root / "s2" / "stages" / "test" / "process.lock"
    session = _Session(root, home)
    try:
        assert session.next_stage_result() == ("ran", "No previous run")
        first = s1_lock.read_text()
        (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: s2\n")
        second = session.next_stage_result()
        returncode = session.stop()
    finally:
        session.kill()

    stderr = "".join(session.stderr)
    assert returncode == 0, stderr
    assert second[0] == "ran", (second, stderr)
    assert s1_lock.read_text() == first, "the reloaded session wrote a lock in s1"
    assert s2_lock.is_file(), stderr
    assert (root / "s1" / "secret.key").is_file()
    assert (root / "s2" / "secret.key").is_file()


def test_a_watch_session_files_each_pipelines_caches_where_it_loaded(
    tmp_path: pathlib.Path,
) -> None:
    """The fingerprint caches the session writes as it ends go to the directory each pipeline
    loaded with: the first load's to `s1`, and the reload's that the `core.state_dir` edit
    started to `s2`."""
    home, root = _project(tmp_path, _PLAIN_PIPELINE)
    session = _Session(root, home)
    try:
        assert session.next_stage_result() == ("ran", "No previous run")
        (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: s2\n")
        reran = session.next_stage_result()
        returncode = session.stop()
    finally:
        session.kill()

    stderr = "".join(session.stderr)
    assert returncode == 0, stderr
    assert reran[0] == "ran", (reran, stderr)
    for directory in ("s1", "s2"):
        with state.StateDB(root / directory, readonly=True) as db:
            assert list(db.iter_prefix(b"sm:")), f"no fingerprint cache in {directory}: {stderr}"


def test_a_watch_session_refuses_to_run_after_a_link_its_state_dir_names_moves(
    tmp_path: pathlib.Path,
) -> None:
    """L3w, the check every engine run makes: `core.state_dir` names a link to `s1`, re-pointed to
    `s2` while the session runs. No edit announces that, so nothing reloads; the next iteration
    refuses rather than record its lock in `s1` while `pivot gc` reads `s2`, naming both
    directories and how to reload, and writes no lock."""
    home, root = _project(tmp_path, _PLAIN_PIPELINE)
    (root / "s1").mkdir()
    (root / "s2").mkdir()
    (root / "state").symlink_to("s1")
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: state\n")
    s1_lock = root / "s1" / "stages" / "test" / "process.lock"
    session = _Session(root, home)
    try:
        assert session.next_stage_result() == ("ran", "No previous run")
        first = s1_lock.read_text()
        relinked = root / "state.relinked"
        relinked.symlink_to("s2")
        relinked.replace(root / "state")
        (root / "data" / "raw" / "input.txt").write_text("second\n")
        second = session.refusal_or_stage_result("its core.state_dir now names")
        time.sleep(1)  # a lock written right after the refusal would show now
        returncode = session.stop()
    finally:
        session.kill()

    stderr = "".join(session.stderr)
    flat = " ".join(stderr.split())
    assert second is None, f"the session ran after the link moved: {second}"
    assert returncode == 0, stderr
    assert "/s1," in flat and "/s2:" in flat, stderr
    assert "Save .pivot/config.yaml or pipeline.py to reload the pipeline" in flat, stderr
    assert s1_lock.read_text() == first, "the session wrote a lock in s1 after the link moved"
    assert not (root / "s2" / "stages").exists(), sorted(p.name for p in (root / "s2").iterdir())
