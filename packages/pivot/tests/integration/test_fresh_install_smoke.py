"""Fresh-install smoke test: build the wheel, install it into a new venv, drive the CLI.

The rest of the suite imports pivot from the workspace venv, where every dev dependency is
present. This test catches what only a user's clean install sees: undeclared runtime
dependencies, files missing from the wheel, and a broken console script. It needs uv and
registry access, so it runs only in CI, which runs the suite without uv on PATH and names uv's
executable in ``PIVOT_TEST_UV``.
"""

from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import time
import typing

import psutil
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

_PIPELINE = """\
from __future__ import annotations

import pandas as pd

from pivot import StageParams
from pivot.compose import Pipeline, stage


class ScaleParams(StageParams):
    factor: int = 10


@stage
def scale(numbers: pd.DataFrame, params: ScaleParams) -> pd.DataFrame:
    return numbers * params.factor


@stage
def total(scaled: pd.DataFrame) -> dict:
    return {"sum": int(scaled.to_numpy().sum())}


with Pipeline("smoke") as pipeline:
    numbers = pipeline.input("numbers.csv")
    scaled = scale(numbers, params=ScaleParams())
    total(scaled)
"""


_LONG_PARENT_FAN = """\
import os
import pathlib
import time

import psutil

from pivot import Pipeline, stage


@stage
def source() -> str:
    time.sleep(11)
    return "source"


@stage
def fan(value: str) -> str:
    return value


def _live_workers() -> int:
    count = 0
    for child in psutil.Process(os.getppid()).children():
        try:
            if "loky.backend.popen_loky_posix" in " ".join(child.cmdline()):
                count += 1
        except psutil.NoSuchProcess:  # exited (or a zombie) between listing and inspecting
            continue
    return count


@stage
def linger(value: str) -> str:
    # An idle worker exits after its 10 s timeout, or after a second one when it loses loky's
    # process-lock race at that moment (loky `_process_worker`), so wait up to two timeouts.
    deadline = time.monotonic() + 25
    live = _live_workers()
    while live != 1 and time.monotonic() < deadline:
        time.sleep(0.1)
        live = _live_workers()
    pathlib.Path(os.environ["PIVOT_FRESH_LIVE_WORKERS"]).write_text(str(live))
    return value


with Pipeline("fan") as pipeline:
    shared = source()
    for index in range(4):
        with pipeline.variant(str(index)):
            value = fan(shared)
            if index == 0:
                linger(value)
"""


_WORKER_LOG_HOOK = """\
import os
import pathlib
import sys


if sys.orig_argv[1:3] == ["-m", "loky.backend.popen_loky_posix"]:
    with pathlib.Path(os.environ["PIVOT_FRESH_WORKER_LOG"]).open("a") as log:
        log.write(str(os.getpid()) + "\\n")
"""


_WATCH_CAP = 4

# Four independent stages, so one run starts the whole capped pool. Each stage records its
# worker pid; the `generation` constant is what the test edits to trigger a code reload.
_WATCH_PIPELINE = """\
import os

from pivot import Pipeline, stage

generation = {generation}


@stage
def wide() -> dict:
    return {{"pid": os.getpid(), "generation": generation}}


with Pipeline("watch") as pipeline:
    for index in range({cap}):
        with pipeline.variant(str(index)):
            wide()
"""


def _live_workers(pivot: subprocess.Popen[str]) -> set[int]:
    """Return the pids of the loky workers the pivot process currently holds."""
    found = set[int]()
    try:
        children = psutil.Process(pivot.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return found
    for child in children:
        try:
            if "loky.backend.popen_loky_posix" in " ".join(child.cmdline()):
                found.add(child.pid)
        except psutil.NoSuchProcess:
            continue
    return found


def _wait_for(condition: typing.Callable[[], bool], what: str, timeout_s: float = 120) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.1)


def _clean_env() -> dict[str, str]:
    """Return ``os.environ`` without the workspace's ``VIRTUAL_ENV*`` and ``PYTHONPATH*``."""
    return {k: v for k, v in os.environ.items() if not k.startswith(("VIRTUAL_ENV", "PYTHONPATH"))}


def _run(
    args: list[str],
    cwd: pathlib.Path,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = _clean_env()
    if extra_env is not None:
        env.update(extra_env)
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=600)


def _install_wheel(tmp_path: pathlib.Path, *requirements: str) -> str:
    """Build pivot's wheel, install it and ``requirements`` into a new venv, return ``pivot``."""
    uv = os.environ.get("PIVOT_TEST_UV", "")
    if not (pathlib.Path(uv).is_file() and os.access(uv, os.X_OK)):
        pytest.fail(f"PIVOT_TEST_UV must name the uv executable; it is {uv!r}", pytrace=False)
    dist = tmp_path / "dist"
    build = _run([uv, "build", "--package", "pivot", "--wheel", "--out-dir", str(dist)], _REPO_ROOT)
    assert build.returncode == 0, build.stderr
    wheel = next(dist.glob("pivot-*.whl"))

    venv = tmp_path / "venv"
    assert _run([uv, "venv", "--python", "3.13", str(venv)], tmp_path).returncode == 0
    install = _run(
        [uv, "pip", "install", "--python", str(venv / "bin" / "python"), str(wheel), *requirements],
        tmp_path,
    )
    assert install.returncode == 0, install.stderr
    return str(venv / "bin" / "pivot")


@pytest.mark.skipif(not os.environ.get("CI"), reason="builds and installs a wheel; runs in CI")
def test_wheel_installed_in_fresh_venv_runs_init_repro_status_dag(tmp_path: pathlib.Path) -> None:
    pivot = _install_wheel(tmp_path)

    project = tmp_path / "project"
    (project / "data" / "raw").mkdir(parents=True)
    (project / "pipeline.py").write_text(_PIPELINE)
    (project / "data" / "raw" / "numbers.csv").write_text("a,b\n1,2\n3,4\n")
    assert _run(["git", "init", "-q"], project).returncode == 0

    init = _run([pivot, "init"], project)
    assert init.returncode == 0, init.stdout + init.stderr

    first = _run([pivot, "repro"], project)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "2 ran" in first.stdout + first.stderr

    second = _run([pivot, "repro"], project)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "2 cached" in second.stdout + second.stderr

    status = _run([pivot, "status"], project)
    assert status.returncode == 0, status.stdout + status.stderr
    assert "2 cached, 0 stale" in status.stdout + status.stderr

    dag = _run([pivot, "dag"], project)
    assert dag.returncode == 0, dag.stdout + dag.stderr

    assert (project / "data" / "smoke" / "total.yaml").read_text().strip() == "sum: 100"


@pytest.mark.parametrize("loky_version", ["3.5.6", "3.6.0", "3.7.0"])
@pytest.mark.skipif(not os.environ.get("CI"), reason="builds and installs a wheel; runs in CI")
def test_fresh_wheel_keeps_prestarted_fan_workers_warm(
    tmp_path: pathlib.Path, loky_version: str
) -> None:
    pivot = _install_wheel(tmp_path, f"loky=={loky_version}")

    project = tmp_path / "project"
    project.mkdir()
    (project / "pipeline.py").write_text(_LONG_PARENT_FAN)
    assert _run(["git", "init", "-q"], project).returncode == 0
    init = _run([pivot, "init"], project)
    assert init.returncode == 0, init.stdout + init.stderr
    (project / ".pivot" / "config.yaml").write_text("core:\n  max_workers: 8\n")
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(_WORKER_LOG_HOOK)
    worker_log = tmp_path / "workers.log"
    live_workers = tmp_path / "live-workers"
    run = _run(
        [pivot, "repro"],
        project,
        extra_env={
            "PYTHONPATH": str(hook),
            "PIVOT_FRESH_WORKER_LOG": str(worker_log),
            "PIVOT_FRESH_LIVE_WORKERS": str(live_workers),
        },
    )
    assert run.returncode == 0, run.stdout + run.stderr
    started = worker_log.read_text().splitlines()
    assert len(started) == 4, f"loky {loky_version} started {len(started)} workers"
    # Within two idle timeouts of the fan, only the worker running `linger` is left.
    assert live_workers.read_text() == "1", f"loky {loky_version} kept idle workers alive"


@pytest.mark.parametrize("loky_version", ["3.5.6", "3.6.0", "3.7.0"])
@pytest.mark.skipif(not os.environ.get("CI"), reason="builds and installs a wheel; runs in CI")
def test_fresh_wheel_watch_reload_replaces_workers_within_the_cap(
    tmp_path: pathlib.Path, loky_version: str
) -> None:
    """A code edit under `repro --watch` replaces the previous run's workers: none survives
    the reload, and the session never holds more workers than `core.max_workers`."""
    pivot = _install_wheel(tmp_path, f"loky=={loky_version}")
    project = tmp_path / "project"
    project.mkdir()
    pipeline_file = project / "pipeline.py"
    pipeline_file.write_text(_WATCH_PIPELINE.format(generation=0, cap=_WATCH_CAP))
    assert _run(["git", "init", "-q"], project).returncode == 0
    init = _run([pivot, "init"], project)
    assert init.returncode == 0, init.stdout + init.stderr
    (project / ".pivot" / "config.yaml").write_text(f"core:\n  max_workers: {_WATCH_CAP}\n")
    outputs = [project / "data" / "watch" / f"wide@{index}.yaml" for index in range(_WATCH_CAP)]

    def generation_done(generation: int) -> bool:
        # The session's rerun removes each output before it writes the new one, so an output can
        # vanish at any moment. Read each once; a missing output is not yet written.
        try:
            return all(f"generation: {generation}" in out.read_text() for out in outputs)
        except FileNotFoundError:
            return False

    log = tmp_path / "watch.log"
    with log.open("w") as output:
        watch = subprocess.Popen(
            [pivot, "repro", "--watch", "--force"],
            cwd=project,
            env=_clean_env(),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
    peak = 0
    try:
        _wait_for(lambda: generation_done(0), "the first run")
        previous = _live_workers(watch)
        assert previous, "the first run left no warm worker to replace"
        for generation in (1, 2):
            pipeline_file.write_text(_WATCH_PIPELINE.format(generation=generation, cap=_WATCH_CAP))

            def reloaded(generation: int = generation) -> bool:
                nonlocal peak
                peak = max(peak, len(_live_workers(watch)))
                return generation_done(generation)

            _wait_for(reloaded, f"reload {generation}")
            survivors = previous & _live_workers(watch)
            assert not survivors, (
                f"loky {loky_version}: reload {generation} left the previous run's workers "
                f"alive: {sorted(survivors)}\n{log.read_text()}"
            )
            previous = _live_workers(watch)
    finally:
        if watch.poll() is None:
            watch.send_signal(signal.SIGINT)
            try:
                watch.wait(timeout=60)
            except subprocess.TimeoutExpired:
                watch.kill()
                watch.wait()
    assert peak <= _WATCH_CAP, (
        f"loky {loky_version}: the watch session held {peak} workers, over the cap of "
        f"{_WATCH_CAP}\n{log.read_text()}"
    )
