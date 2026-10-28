"""Integration tests for concurrent execution and artifact locking.

The subprocesses use real ``pivot run`` commands. The generated pipelines use
handles, so read/write locking is exercised on ArtifactIdentity rather than
user-selected output paths.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import subprocess
import time
from typing import TYPE_CHECKING

from tests import helpers
from tests.conftest import init_git_repo

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Mapping

# Resolve pivot CLI binary from the same venv as the test runner.


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
    pathlib.Path("stage_a_started").touch()
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


# `make` is a widely read cached stage: `slow_use` reads it for as long as the test wants (it
# waits for `release_slow_use`), `quick_use` reads it and returns at once.
_PIPELINE_SHARED_OUTPUT = """\
import pathlib
import time

from pivot.compose import Pipeline, stage


@stage
def make() -> str:
    return "made"


@stage
def slow_use(made: str) -> str:
    pathlib.Path("slow_use_started").touch()
    deadline = time.monotonic() + 120
    while not pathlib.Path("release_slow_use").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    return f"slow saw {made}"


@stage
def quick_use(made: str) -> str:
    return f"quick saw {made}"


with Pipeline("test", root=pathlib.Path(__file__).parent) as pipeline:
    made = make()
    slow_use(made)
    quick_use(made)
"""

# Loaded by every pivot process the hook directory is on PYTHONPATH for: logs each lock
# request that asks for WRITE, and each restore of an output from the cache, with the pid.
_LOCK_AND_RESTORE_LOG_HOOK = """\
import os
import pathlib

from pivot.storage import artifact_lock, store

_log = pathlib.Path(os.environ["PIVOT_TEST_HOOK_LOG"])
_acquire_many = artifact_lock.LocalFlockLockService.acquire_many
_restore_from_cache = store.WorkspaceStore.restore_from_cache


def _record(line):
    with _log.open("a") as log:
        log.write(f"{os.getpid()} {line}\\n")


def acquire_many(self, requests, on_status=None):
    written = [r["key"] for r in requests if r["mode"] is artifact_lock.LockMode.WRITE]
    if written:
        _record("write " + ",".join(written))
    return _acquire_many(self, requests, on_status)


def restore_from_cache(self, ref, output_hash, **kwargs):
    _record("restore")
    return _restore_from_cache(self, ref, output_hash, **kwargs)


artifact_lock.LocalFlockLockService.acquire_many = acquire_many
store.WorkspaceStore.restore_from_cache = restore_from_cache
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _helper_setup_project(tmp_path: pathlib.Path, pipeline_code: str) -> None:
    """Set up a minimal pivot project in tmp_path with given pipeline code."""
    init_git_repo(tmp_path)
    subprocess.run(
        [helpers.PIVOT, "init"],
        cwd=tmp_path,
        env=_helper_make_env(tmp_path),
        check=True,
        capture_output=True,
    )
    (tmp_path / "pipeline.py").write_text(pipeline_code)


def _helper_make_env(tmp_path: pathlib.Path, **extra: str) -> dict[str, str]:
    """The installed-CLI environment for a project at ``tmp_path``, with optional extra vars."""
    return helpers.cli_env(tmp_path / "home", **extra)


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
        [helpers.PIVOT, command, qualified_stage, "--force"],
        cwd=tmp_path,
        env=env or _helper_make_env(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _helper_wait_for_path(path: pathlib.Path, timeout: float = 30.0) -> None:
    """Wait until a stage has started under its artifact lock."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for {path}")


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
    env = _helper_make_env(tmp_path, STAGE_A_SLEEP="3")

    proc_a = _helper_run_pivot(tmp_path, "stage_a", env=env)
    _helper_wait_for_path(tmp_path / "stage_a_started")
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
    env = _helper_make_env(tmp_path, STAGE_X_SLEEP="1", STAGE_Y_SLEEP="1")

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
    env = _helper_make_env(tmp_path, STAGE_A_SLEEP="4")

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


def _helper_built_shared_output_project(tmp_path: pathlib.Path) -> None:
    """Build `_PIPELINE_SHARED_OUTPUT` once, so every stage is cached."""
    _helper_setup_project(tmp_path, _PIPELINE_SHARED_OUTPUT)
    (tmp_path / "release_slow_use").touch()
    built = subprocess.run(
        [helpers.PIVOT, "repro"],
        cwd=tmp_path,
        env=_helper_make_env(tmp_path),
        capture_output=True,
        timeout=180,
    )
    assert built.returncode == 0, built.stdout.decode() + built.stderr.decode()
    (tmp_path / "release_slow_use").unlink()
    (tmp_path / "slow_use_started").unlink()


def _helper_start_slow_consumer(tmp_path: pathlib.Path) -> subprocess.Popen[bytes]:
    """Run `slow_use` again in its own pivot process and wait until it is inside the stage,
    holding READ on `make`'s output for as long as the test withholds `release_slow_use`."""
    slow = _helper_run_pivot(tmp_path, "slow_use")
    _helper_wait_for_path(tmp_path / "slow_use_started", timeout=120)
    return slow


def _helper_finish(process: subprocess.Popen[bytes], timeout: float) -> tuple[str, bool]:
    """Wait up to ``timeout`` for ``process``; return its output and whether it finished."""
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return stdout.decode() + stderr.decode(), False
    return stdout.decode() + stderr.decode(), True


def test_skip_of_a_cached_stage_does_not_wait_for_its_running_consumers(
    tmp_path: pathlib.Path,
) -> None:
    """A repro whose closure holds a widely read cached stage skips it at once, though another
    pivot run is executing one of its consumers: deciding to skip only reads the output, so
    it does not wait for the READ every running consumer holds on it."""
    _helper_built_shared_output_project(tmp_path)
    slow = _helper_start_slow_consumer(tmp_path)
    try:
        quick = subprocess.Popen(
            [helpers.PIVOT, "repro", "test/quick_use"],
            cwd=tmp_path,
            env=_helper_make_env(tmp_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        quick_out, quick_finished = _helper_finish(quick, timeout=30)
    finally:
        (tmp_path / "release_slow_use").touch()
        slow_out, slow_finished = _helper_finish(slow, timeout=60)

    assert quick_finished, f"the cached repro waited for the running consumer:\n{quick_out}"
    assert quick.returncode == 0, quick_out
    assert "0 ran, 2 cached" in quick_out, quick_out
    assert slow_finished and slow.returncode == 0, slow_out
    assert _helper_output(tmp_path, "slow_use").read_text() == "slow saw made"


def test_racing_skips_restore_a_missing_cached_output_once(tmp_path: pathlib.Path) -> None:
    """Two repros that both find a cached stage's output missing restore it from the cache
    exactly once. Each waits for WRITE on the output, held off here by a running consumer's
    READ, and decides again once it has WRITE: the second finds the output the first restored."""
    _helper_built_shared_output_project(tmp_path)
    hook_dir = tmp_path / "hook"
    hook_dir.mkdir()
    (hook_dir / "sitecustomize.py").write_text(_LOCK_AND_RESTORE_LOG_HOOK)
    hook_log = tmp_path / "hook.log"
    hook_log.touch()
    env = _helper_make_env(tmp_path, PIVOT_TEST_HOOK_LOG=str(hook_log))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(hook_dir), env.get("PYTHONPATH")]))
    output = _helper_output(tmp_path, "make")

    slow = _helper_start_slow_consumer(tmp_path)
    racers = list[subprocess.Popen[bytes]]()
    try:
        output.unlink()
        racers = [
            subprocess.Popen(
                [helpers.PIVOT, "repro", "test/make"],
                cwd=tmp_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for _ in range(2)
        ]

        def _waiting_for_write() -> set[str]:
            lines = hook_log.read_text().splitlines()
            return {line.split()[0] for line in lines if line.split()[1] == "write"}

        deadline = time.monotonic() + 120
        while len(_waiting_for_write()) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(_waiting_for_write()) == 2, hook_log.read_text()
        time.sleep(1)  # both are now retrying WRITE behind the consumer's READ
    finally:
        (tmp_path / "release_slow_use").touch()
        slow_out, slow_finished = _helper_finish(slow, timeout=60)
        racer_results = [_helper_finish(racer, timeout=60) for racer in racers]

    assert slow_finished and slow.returncode == 0, slow_out
    for racer, (racer_out, finished) in zip(racers, racer_results, strict=True):
        assert finished and racer.returncode == 0, racer_out
        assert "0 ran, 1 cached" in racer_out, racer_out
    restores = [line for line in hook_log.read_text().splitlines() if line.endswith(" restore")]
    assert len(restores) == 1, hook_log.read_text()
    assert output.read_text() == "made"
