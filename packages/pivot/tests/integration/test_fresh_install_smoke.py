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


def _run(args: list[str], cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VIRTUAL_ENV", "PYTHONPATH"))}
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=600)


@pytest.mark.skipif(not os.environ.get("CI"), reason="builds and installs a wheel; runs in CI")
def test_wheel_installed_in_fresh_venv_runs_init_repro_status_dag(tmp_path: pathlib.Path) -> None:
    uv = shutil.which("uv")  # searches PATH: CI installs uv, and this test runs only in CI
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
