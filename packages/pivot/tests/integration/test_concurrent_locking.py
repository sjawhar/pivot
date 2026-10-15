"""Integration tests for concurrent execution and artifact locking.

The subprocesses use real ``pivot run`` commands. The generated pipelines use
handles, so read/write locking is exercised on ArtifactIdentity rather than
user-selected output paths.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import pathlib
import subprocess
import sys
import time
from typing import TYPE_CHECKING

from tests.conftest import init_git_repo

if TYPE_CHECKING:
    from collections.abc import Mapping

# Resolve pivot CLI binary from the same venv as the test runner.
_PIVOT_BIN = str(pathlib.Path(sys.executable).parent / "pivot")


# ---------------------------------------------------------------------------
# Pipeline templates
# ---------------------------------------------------------------------------

_PIPELINE_DISJOINT = """\
import pathlib
import time

from pivot.compose import Pipeline, stage


def record(label: str) -> None:
    with pathlib.Path("timing.log").open("a") as timing:
        timing.write(f"{label} {time.monotonic()}\\n")


def wait_for(label: str) -> None:
    # Both runs must be inside their stages at once; a run serialised behind the other
    # never sees its partner start and fails here instead of passing on timing luck.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        lines = pathlib.Path("timing.log").read_text().splitlines()
        if any(line.startswith(label + " ") for line in lines):
            return
        time.sleep(0.01)
    raise RuntimeError(f"partner {label} never started")


@stage
def stage_a() -> str:
    record("a_start")
    wait_for("b_start")
    record("a_end")
    return "hello from a"


@stage
def stage_b() -> str:
    record("b_start")
    wait_for("a_start")
    record("b_end")
    return "hello from b"


with Pipeline("test", root=pathlib.Path(__file__).parent) as pipeline:
    stage_a()
    stage_b()
"""

_PIPELINE_OVERLAPPING = """\
import os
import pathlib
import time

from pivot.compose import Pipeline, stage


@stage
def stage_a() -> str:
    time.sleep(float(os.environ.get("STAGE_A_SLEEP", "3")))
    return "hello from a"


@stage
def stage_c(dep_a: str) -> str:
    return f"got: {dep_a}"


with Pipeline("test", root=pathlib.Path(__file__).parent) as pipeline:
    a = stage_a()
    stage_c(a)
"""

_PIPELINE_CROSSED = """\
import os
import pathlib
import time

from pivot.compose import Pipeline, stage


@stage
def stage_x() -> str:
    time.sleep(float(os.environ.get("STAGE_X_SLEEP", "0.5")))
    return "hello from x"


@stage
def stage_y() -> str:
    time.sleep(float(os.environ.get("STAGE_Y_SLEEP", "0.5")))
    return "hello from y"


with Pipeline("test", root=pathlib.Path(__file__).parent) as pipeline:
    stage_x()
    stage_y()
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _helper_setup_project(tmp_path: pathlib.Path, pipeline_code: str) -> None:
    """Set up a minimal pivot project in tmp_path with given pipeline code."""
    init_git_repo(tmp_path)
    subprocess.run(
        [_PIVOT_BIN, "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "pipeline.py").write_text(pipeline_code)


def _helper_make_env(**extra: str) -> dict[str, str]:
    """Create env dict for subprocess with optional extra vars."""
    env = os.environ.copy()
    env.update(extra)
    return env


def _helper_run_pivot(
    tmp_path: pathlib.Path,
    stage: str,
    env: Mapping[str, str] | None = None,
    *,
    resolve_deps: bool = False,
) -> subprocess.Popen[bytes]:
    """Start an isolated `run` or dependency-resolving `repro` stage command."""
    qualified_stage = stage if "/" in stage else f"test/{stage}"
    command = "repro" if resolve_deps else "run"
    return subprocess.Popen(
        [_PIVOT_BIN, command, qualified_stage, "--force"],
        cwd=tmp_path,
        env=env or os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _helper_wait_for_write_lock(lock_path: pathlib.Path) -> None:
    """Wait until another process owns lock_path exclusively, instead of sleeping by guess."""
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if lock_path.exists():
            fd = os.open(lock_path, os.O_RDONLY)
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    return
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        time.sleep(0.01)
    raise TimeoutError(f"writer never acquired artifact lock {lock_path}")


def _helper_output(project: pathlib.Path, stage: str) -> pathlib.Path:
    return project / "data" / "test" / f"{stage}.txt"


def test_wait_for_write_lock_waits_until_lock_is_held(tmp_path: pathlib.Path) -> None:
    """Contention tests start their reader only after the writer owns the artifact lock."""
    lock_path = tmp_path / "lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        _helper_wait_for_write_lock(lock_path)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_disjoint_runs_proceed_in_parallel(tmp_path: pathlib.Path) -> None:
    """Two independent artifact writers run concurrently without blocking."""
    _helper_setup_project(tmp_path, _PIPELINE_DISJOINT)

    proc_a = _helper_run_pivot(tmp_path, "stage_a")
    proc_b = _helper_run_pivot(tmp_path, "stage_b")
    try:
        stdout_a, stderr_a = proc_a.communicate(timeout=90)
        stdout_b, stderr_b = proc_b.communicate(timeout=90)
    finally:
        for process in (proc_a, proc_b):
            if process.poll() is None:
                process.kill()
                process.wait()

    assert proc_a.returncode == 0, (
        f"stage_a failed:\nstdout={stdout_a.decode()}\nstderr={stderr_a.decode()}"
    )
    assert proc_b.returncode == 0, (
        f"stage_b failed:\nstdout={stdout_b.decode()}\nstderr={stderr_b.decode()}"
    )
    assert _helper_output(tmp_path, "stage_a").read_text() == "hello from a"
    assert _helper_output(tmp_path, "stage_b").read_text() == "hello from b"

    events = {
        label: float(timestamp)
        for label, timestamp in (
            line.split() for line in (tmp_path / "timing.log").read_text().splitlines()
        )
    }
    assert events["a_start"] < events["b_end"] and events["b_start"] < events["a_end"], events


def test_overlapping_runs_block_then_continue(tmp_path: pathlib.Path) -> None:
    """A consumer waits for a concurrently running producer then reads its output."""
    _helper_setup_project(tmp_path, _PIPELINE_OVERLAPPING)
    env = _helper_make_env(STAGE_A_SLEEP="3")

    proc_a = _helper_run_pivot(tmp_path, "stage_a", env=env)
    time.sleep(0.5)
    proc_c = _helper_run_pivot(tmp_path, "stage_c", env=env, resolve_deps=True)
    try:
        stdout_a, stderr_a = proc_a.communicate(timeout=30)
        stdout_c, stderr_c = proc_c.communicate(timeout=30)
    finally:
        for process in (proc_a, proc_c):
            if process.poll() is None:
                process.kill()
                process.wait()

    assert proc_a.returncode == 0, (
        f"stage_a failed:\nstdout={stdout_a.decode()}\nstderr={stderr_a.decode()}"
    )
    assert proc_c.returncode == 0, (
        f"stage_c failed:\nstdout={stdout_c.decode()}\nstderr={stderr_c.decode()}"
    )
    assert _helper_output(tmp_path, "stage_a").read_text() == "hello from a"
    assert _helper_output(tmp_path, "stage_c").read_text() == "got: hello from a"


def test_no_deadlock_deterministic_ordering(tmp_path: pathlib.Path) -> None:
    """Two disjoint writers complete concurrently without deadlock."""
    _helper_setup_project(tmp_path, _PIPELINE_CROSSED)
    env = _helper_make_env(STAGE_X_SLEEP="1", STAGE_Y_SLEEP="1")

    proc_x = _helper_run_pivot(tmp_path, "stage_x", env=env)
    proc_y = _helper_run_pivot(tmp_path, "stage_y", env=env)
    try:
        stdout_x, stderr_x = proc_x.communicate(timeout=30)
        stdout_y, stderr_y = proc_y.communicate(timeout=30)
    finally:
        for process in (proc_x, proc_y):
            if process.poll() is None:
                process.kill()
                process.wait()

    assert proc_x.returncode == 0, (
        f"stage_x failed (possible deadlock):\nstdout={stdout_x.decode()}\nstderr={stderr_x.decode()}"
    )
    assert proc_y.returncode == 0, (
        f"stage_y failed (possible deadlock):\nstdout={stdout_y.decode()}\nstderr={stderr_y.decode()}"
    )
    assert _helper_output(tmp_path, "stage_x").read_text() == "hello from x"
    assert _helper_output(tmp_path, "stage_y").read_text() == "hello from y"


def test_periodic_status_during_contention(tmp_path: pathlib.Path) -> None:
    """A lock-blocked consumer succeeds without leaking lock-wait text to console output."""
    _helper_setup_project(tmp_path, _PIPELINE_OVERLAPPING)
    env = _helper_make_env(STAGE_A_SLEEP="4")

    proc_a = _helper_run_pivot(tmp_path, "stage_a", env=env)
    stage_a_lock = tmp_path / ".pivot" / "locks" / hashlib.sha256(b"test/stage_a").hexdigest()
    _helper_wait_for_write_lock(stage_a_lock)
    proc_c = _helper_run_pivot(tmp_path, "stage_c", env=env)
    try:
        stdout_a, stderr_a = proc_a.communicate(timeout=30)
        stdout_c, stderr_c = proc_c.communicate(timeout=30)
    finally:
        for process in (proc_a, proc_c):
            if process.poll() is None:
                process.kill()
                process.wait()

    assert proc_a.returncode == 0, (
        f"stage_a failed:\nstdout={stdout_a.decode()}\nstderr={stderr_a.decode()}"
    )
    assert proc_c.returncode == 0, (
        f"stage_c failed:\nstdout={stdout_c.decode()}\nstderr={stderr_c.decode()}"
    )
    combined_c = stdout_c.decode() + stderr_c.decode()
    assert "waiting for artifact lock" not in combined_c.lower()
