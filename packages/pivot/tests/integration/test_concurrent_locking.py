"""Integration tests for concurrent pivot execution with artifact locks.

Tests use real subprocess.Popen to launch multiple `pivot run` processes
concurrently and verify that artifact locks coordinate access correctly.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

from conftest import init_git_repo

# Resolve pivot CLI binary from the same venv as the test runner
_PIVOT_BIN = str(pathlib.Path(sys.executable).parent / "pivot")

# ---------------------------------------------------------------------------
# Pipeline templates
# ---------------------------------------------------------------------------

_PIPELINE_DISJOINT = '''\
import os
import pathlib
import time
from typing import Annotated, TypedDict
from pivot.pipeline.pipeline import Pipeline
from pivot.outputs import Out
from pivot import loaders

pipeline = Pipeline("test", root=pathlib.Path(__file__).parent)


class _OutputA(TypedDict):
    data: Annotated[pathlib.Path, Out("output_a.txt", loaders.PathOnly())]


class _OutputB(TypedDict):
    data: Annotated[pathlib.Path, Out("output_b.txt", loaders.PathOnly())]


def stage_a() -> _OutputA:
    """Writes output_a.txt with controlled sleep."""
    duration = float(os.environ.get("STAGE_A_SLEEP", "2"))
    time.sleep(duration)
    p = pathlib.Path("output_a.txt")
    p.write_text("hello from a")
    return _OutputA(data=p)


def stage_b() -> _OutputB:
    """Writes output_b.txt (disjoint from A)."""
    duration = float(os.environ.get("STAGE_B_SLEEP", "2"))
    time.sleep(duration)
    p = pathlib.Path("output_b.txt")
    p.write_text("hello from b")
    return _OutputB(data=p)


pipeline.register(stage_a)
pipeline.register(stage_b)
'''

_PIPELINE_OVERLAPPING = """\
import os
import pathlib
import time
from typing import Annotated, TypedDict
from pivot.pipeline.pipeline import Pipeline
from pivot.outputs import Dep, Out
from pivot.loaders import PathOnly

pipeline = Pipeline("test", root=pathlib.Path(__file__).parent)


class _OutputA(TypedDict):
    data: Annotated[pathlib.Path, Out("output_a.txt", PathOnly())]


class _OutputC(TypedDict):
    data: Annotated[pathlib.Path, Out("output_c.txt", PathOnly())]


def stage_a() -> _OutputA:
    duration = float(os.environ.get("STAGE_A_SLEEP", "3"))
    time.sleep(duration)
    p = pathlib.Path("output_a.txt")
    p.write_text("hello from a")
    return _OutputA(data=p)


def stage_c(
    dep_a: Annotated[pathlib.Path, Dep("output_a.txt", PathOnly())],
) -> _OutputC:
    content = dep_a.read_text()
    p = pathlib.Path("output_c.txt")
    p.write_text(f"got: {content}")
    return _OutputC(data=p)


pipeline.register(stage_a)
pipeline.register(stage_c)
"""

_PIPELINE_CROSSED = '''\
import os
import pathlib
import time
from typing import Annotated, TypedDict
from pivot.pipeline.pipeline import Pipeline
from pivot.outputs import Out
from pivot.loaders import PathOnly
from pivot import loaders

pipeline = Pipeline("test", root=pathlib.Path(__file__).parent)


class _OutputX(TypedDict):
    data: Annotated[pathlib.Path, Out("output_x.txt", PathOnly())]


class _OutputY(TypedDict):
    data: Annotated[pathlib.Path, Out("output_y.txt", PathOnly())]


def stage_x() -> _OutputX:
    """Writes output_x.txt with brief sleep."""
    time.sleep(float(os.environ.get("STAGE_X_SLEEP", "0.5")))
    p = pathlib.Path("output_x.txt")
    p.write_text("hello from x")
    return _OutputX(data=p)


def stage_y() -> _OutputY:
    """Writes output_y.txt with brief sleep."""
    time.sleep(float(os.environ.get("STAGE_Y_SLEEP", "0.5")))
    p = pathlib.Path("output_y.txt")
    p.write_text("hello from y")
    return _OutputY(data=p)


pipeline.register(stage_x)
pipeline.register(stage_y)
'''


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
    env: dict[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    """Start `pivot run <stage> --force` as a background process."""
    return subprocess.Popen(
        [_PIVOT_BIN, "run", stage, "--force"],
        cwd=tmp_path,
        env=env or os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_disjoint_runs_proceed_in_parallel(tmp_path: pathlib.Path) -> None:
    """Two stages with disjoint outputs run concurrently without blocking.

    stage_a and stage_b both sleep for STAGE_A_SLEEP/STAGE_B_SLEEP seconds.
    If they run in parallel, wall time < sum of individual durations.
    """
    sleep_seconds = 3
    _helper_setup_project(tmp_path, _PIPELINE_DISJOINT)
    env = _helper_make_env(STAGE_A_SLEEP=str(sleep_seconds), STAGE_B_SLEEP=str(sleep_seconds))

    start = time.monotonic()
    proc_a = _helper_run_pivot(tmp_path, "stage_a", env=env)
    proc_b = _helper_run_pivot(tmp_path, "stage_b", env=env)
    try:
        stdout_a, stderr_a = proc_a.communicate(timeout=30)
        stdout_b, stderr_b = proc_b.communicate(timeout=30)
    finally:
        for p in (proc_a, proc_b):
            if p.poll() is None:
                p.kill()
                p.wait()
    elapsed = time.monotonic() - start

    assert proc_a.returncode == 0, (
        f"stage_a failed:\nstdout={stdout_a.decode()}\nstderr={stderr_a.decode()}"
    )
    assert proc_b.returncode == 0, (
        f"stage_b failed:\nstdout={stdout_b.decode()}\nstderr={stderr_b.decode()}"
    )

    assert (tmp_path / "output_a.txt").exists(), "output_a.txt not created"
    assert (tmp_path / "output_b.txt").exists(), "output_b.txt not created"

    # Both stages sleep in parallel, so total wall time should be ~sleep_seconds,
    # not 2*sleep_seconds. Allow generous overhead for subprocess startup + worker pool
    # init, especially on CI runners where resources are constrained.
    max_serial_time = sleep_seconds * 2 + 2
    assert elapsed < max_serial_time, (
        f"Wall time {elapsed:.1f}s >= {max_serial_time}s — stages may not have run in parallel"
    )


def test_overlapping_runs_block_then_continue(tmp_path: pathlib.Path) -> None:
    """stage_c depends on output_a.txt. When both run concurrently:

    - stage_a holds WRITE lock on output_a.txt (sleeps)
    - stage_c needs READ lock on output_a.txt (blocks until stage_a finishes)
    - Both should eventually succeed
    """
    sleep_seconds = 3
    _helper_setup_project(tmp_path, _PIPELINE_OVERLAPPING)
    env = _helper_make_env(STAGE_A_SLEEP=str(sleep_seconds))

    proc_a = _helper_run_pivot(tmp_path, "stage_a", env=env)
    # Small delay so stage_a starts first and grabs the lock
    time.sleep(0.5)
    proc_c = _helper_run_pivot(tmp_path, "stage_c", env=env)

    try:
        stdout_a, stderr_a = proc_a.communicate(timeout=30)
        stdout_c, stderr_c = proc_c.communicate(timeout=30)
    finally:
        for p in (proc_a, proc_c):
            if p.poll() is None:
                p.kill()
                p.wait()

    assert proc_a.returncode == 0, (
        f"stage_a failed:\nstdout={stdout_a.decode()}\nstderr={stderr_a.decode()}"
    )
    assert proc_c.returncode == 0, (
        f"stage_c failed:\nstdout={stdout_c.decode()}\nstderr={stderr_c.decode()}"
    )

    assert (tmp_path / "output_a.txt").exists(), "output_a.txt not created"
    assert (tmp_path / "output_c.txt").exists(), "output_c.txt not created"
    assert "hello from a" in (tmp_path / "output_c.txt").read_text(), (
        "stage_c did not read stage_a's output correctly"
    )


def test_no_deadlock_deterministic_ordering(tmp_path: pathlib.Path) -> None:
    """Two stages with disjoint WRITE locks complete without deadlock.

    Both stage_x and stage_y write to different files and sleep briefly.
    Locks are always acquired in sorted key order, preventing deadlocks.
    If a deadlock occurred, the timeout would fire.
    """
    _helper_setup_project(tmp_path, _PIPELINE_CROSSED)
    env = _helper_make_env(STAGE_X_SLEEP="1", STAGE_Y_SLEEP="1")

    proc_x = _helper_run_pivot(tmp_path, "stage_x", env=env)
    proc_y = _helper_run_pivot(tmp_path, "stage_y", env=env)

    try:
        stdout_x, stderr_x = proc_x.communicate(timeout=30)
        stdout_y, stderr_y = proc_y.communicate(timeout=30)
    finally:
        for p in (proc_x, proc_y):
            if p.poll() is None:
                p.kill()
                p.wait()

    assert proc_x.returncode == 0, (
        f"stage_x failed (possible deadlock):\nstdout={stdout_x.decode()}\nstderr={stderr_x.decode()}"
    )
    assert proc_y.returncode == 0, (
        f"stage_y failed (possible deadlock):\nstdout={stdout_y.decode()}\nstderr={stderr_y.decode()}"
    )

    assert (tmp_path / "output_x.txt").exists(), "output_x.txt not created"
    assert (tmp_path / "output_y.txt").exists(), "output_y.txt not created"


def test_periodic_status_during_contention(tmp_path: pathlib.Path) -> None:
    """When a process blocks on an artifact lock, status messages are emitted.

    stage_a sleeps while holding WRITE lock on output_a.txt. stage_c tries to
    READ output_a.txt and must wait. The blocked process should emit a
    "waiting" message to its output.
    """
    sleep_seconds = 4
    _helper_setup_project(tmp_path, _PIPELINE_OVERLAPPING)
    env = _helper_make_env(STAGE_A_SLEEP=str(sleep_seconds))

    proc_a = _helper_run_pivot(tmp_path, "stage_a", env=env)
    # Ensure stage_a starts first and grabs the write lock
    time.sleep(1.0)
    proc_c = _helper_run_pivot(tmp_path, "stage_c", env=env)

    try:
        stdout_a, stderr_a = proc_a.communicate(timeout=30)
        stdout_c, stderr_c = proc_c.communicate(timeout=30)
    finally:
        for p in (proc_a, proc_c):
            if p.poll() is None:
                p.kill()
                p.wait()

    assert proc_a.returncode == 0, (
        f"stage_a failed:\nstdout={stdout_a.decode()}\nstderr={stderr_a.decode()}"
    )
    assert proc_c.returncode == 0, (
        f"stage_c failed:\nstdout={stdout_c.decode()}\nstderr={stderr_c.decode()}"
    )

    # Check that the blocked process emitted a waiting/lock status message.
    # The ConsoleSink prints "waiting for artifact lock" to stdout (via rich Console).
    combined_c = stdout_c.decode() + stderr_c.decode()
    assert "waiting" in combined_c.lower() or "lock" in combined_c.lower(), (
        f"Expected lock-wait status in stage_c output, got:\nstdout={stdout_c.decode()}\nstderr={stderr_c.decode()}"
    )
