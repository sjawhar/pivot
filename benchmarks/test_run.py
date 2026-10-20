from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import TYPE_CHECKING

import run

if TYPE_CHECKING:
    import pytest

_PIVOT = pathlib.Path(sys.executable).parent / "pivot"
_BLAS_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _helper_python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def _helper_is_gone(pid: int) -> bool:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            state = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            return True
        if state == "Z":
            return True
        time.sleep(0.05)
    return False


def test_measure_counts_cpu_of_descendants_the_command_reaps(tmp_path: pathlib.Path) -> None:
    burn = "import time\nt = time.process_time()\nwhile time.process_time() - t < 0.3: pass"
    code = f"import subprocess, sys; subprocess.run([sys.executable, '-c', {burn!r}], check=True)"

    measurement, _ = run.measure(_helper_python(code), tmp_path)

    assert measurement["exit_code"] == 0
    assert measurement["cpu_s"] >= 0.3, f"grandchild CPU missing: {measurement}"
    assert measurement["wall_s"] >= 0.3


def test_measure_kills_and_counts_processes_left_in_the_session(tmp_path: pathlib.Path) -> None:
    measurement, _ = run.measure(["sh", "-c", "sleep 60 & echo $! > pid"], tmp_path)

    assert measurement["leftover_procs"] == 1
    assert _helper_is_gone(int((tmp_path / "pid").read_text())), "leftover sleep still running"


def test_measured_command_runs_with_a_users_environment_and_records_the_machine(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in _BLAS_VARS:
        monkeypatch.setenv(name, "1")
    code = f"import json, os; print(json.dumps([k for k in {_BLAS_VARS!r} if k in os.environ]))"

    measurement, output = run.measure(_helper_python(code), tmp_path)

    assert measurement["exit_code"] == 0
    assert json.loads(output) == [], "the harness must not cap BLAS threads for pivot"
    assert measurement["cpu_count"] == os.process_cpu_count()
    assert measurement["load1"] >= 0


def test_multi_project_builds_one_task_and_one_pipeline_with_their_upstream(
    tmp_path: pathlib.Path,
) -> None:
    out_dir = tmp_path / "results"
    argv = ["--project", "multi", "--variant", f"a={_PIVOT}", "--reps", "1"]
    argv += ["--work", str(tmp_path / "work"), "--out-dir", str(out_dir)]

    assert run.main(argv) == 0

    rows = [json.loads(line) for line in (out_dir / "a.jsonl").read_text().splitlines()]
    observed = {row["scenario"]: (row["exit_code"], row["ran"], row["cached"]) for row in rows}
    assert observed == {
        # fam_00/t00_s04 with its upstream: its 5-stage task chain, fam_00/b and prep's 5 stages.
        "multi/task/cold": (0, 11, 0),
        "multi/task/warm": (0, 0, 11),
        "multi/task/touch": (0, 5, 6),
        "multi/task/edit": (0, 1, 10),
        # fam_00: b and 4 tasks of 5 stages, plus prep's 5 stages; fam_01 and fam_02 never run.
        "multi/pipeline/cold": (0, 26, 0),
        "multi/pipeline/warm": (0, 0, 26),
        "multi/pipeline/touch": (0, 5, 21),
        "multi/pipeline/edit": (0, 1, 25),
    }
    assert len(rows) == 8, "one row per command"
