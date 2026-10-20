"""Fresh-install smoke test: build the wheel, install it into a new venv, drive the CLI.

The rest of the suite imports pivot from the workspace venv, where every dev dependency is
present. This test catches what only a user's clean install sees: undeclared runtime
dependencies, files missing from the wheel, and a broken console script. It needs uv and
registry access, so it runs only in CI.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

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


@stage
def linger(value: str) -> str:
    time.sleep(13)
    live = [
        child
        for child in psutil.Process(os.getppid()).children()
        if "loky.backend.popen_loky_posix" in " ".join(child.cmdline())
    ]
    pathlib.Path(os.environ["PIVOT_FRESH_LIVE_WORKERS"]).write_text(str(len(live)))
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


def _run(
    args: list[str],
    cwd: pathlib.Path,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VIRTUAL_ENV", "PYTHONPATH"))}
    if extra_env is not None:
        env.update(extra_env)
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=600)


@pytest.mark.skipif(not os.environ.get("CI"), reason="builds and installs a wheel; runs in CI")
def test_wheel_installed_in_fresh_venv_runs_init_repro_status_dag(tmp_path: pathlib.Path) -> None:
    uv = shutil.which("uv")
    assert uv is not None, "CI installs uv before running tests"
    dist = tmp_path / "dist"
    build = _run([uv, "build", "--package", "pivot", "--wheel", "--out-dir", str(dist)], _REPO_ROOT)
    assert build.returncode == 0, build.stderr
    wheel = next(dist.glob("pivot-*.whl"))

    venv = tmp_path / "venv"
    assert _run([uv, "venv", "--python", "3.13", str(venv)], tmp_path).returncode == 0
    install = _run(
        [uv, "pip", "install", "--python", str(venv / "bin" / "python"), str(wheel)], tmp_path
    )
    assert install.returncode == 0, install.stderr
    pivot = str(venv / "bin" / "pivot")

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
    uv = shutil.which("uv")
    assert uv is not None, "CI installs uv before running tests"
    dist = tmp_path / "dist"
    build = _run([uv, "build", "--package", "pivot", "--wheel", "--out-dir", str(dist)], _REPO_ROOT)
    assert build.returncode == 0, build.stderr
    wheel = next(dist.glob("pivot-*.whl"))

    venv = tmp_path / "venv"
    assert _run([uv, "venv", "--python", "3.13", str(venv)], tmp_path).returncode == 0
    install = _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(venv / "bin" / "python"),
            str(wheel),
            f"loky=={loky_version}",
        ],
        tmp_path,
    )
    assert install.returncode == 0, install.stderr
    pivot = str(venv / "bin" / "pivot")

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
    # 13 s after the fan, only the worker running `linger` is left; the others timed out idle.
    assert live_workers.read_text() == "1", f"loky {loky_version} kept idle workers alive"
