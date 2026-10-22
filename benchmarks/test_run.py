from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import pytest

import run
import run_test_helpers

if TYPE_CHECKING:
    from collections.abc import Iterator

_PIVOT = pathlib.Path(sys.executable).parent / "pivot"
_RUN_PY = pathlib.Path(run.__file__)
_BLAS_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
# Stand-ins for pivot. `init` makes the project. `repro` starts a long child in its own session,
# as a pivot repro that has started its workers does, and writes the session id. On SIGINT,
# pivot's own Ctrl-C path writes the SIGINT file and _INTERRUPTIBLE exits.
_SID_FILE = "sid"
_SIGINT_FILE = "got-sigint"
_INTERRUPTIBLE = """#!/bin/sh
case "$1" in
  init) mkdir -p .pivot ;;
  repro) trap 'echo int > {sigint_file}; exit 130' INT
         sleep 600 & ps -o sid= -p $$ > {sid_file}; wait ;;
esac
"""
_QUICK = """#!/bin/sh
case "$1" in
  init) mkdir -p .pivot ;;
esac
"""


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


def _helper_session_alive(sid: int) -> list[int]:
    return [pid for pid in run.session_members(sid) if not _helper_is_gone(pid)]


def _helper_read_when_written(path: pathlib.Path) -> str:
    deadline = time.monotonic() + 20
    while not (path.exists() and path.read_text().strip()):
        assert time.monotonic() < deadline, f"{path.name} was never written"
        time.sleep(0.05)
    return path.read_text()


@pytest.fixture
def stand_in_session(tmp_path: pathlib.Path) -> Iterator[None]:
    """Kill whatever is left of the stand-in's session, whatever the test saw."""
    yield
    sid_file = tmp_path / _SID_FILE
    if sid_file.exists() and sid_file.read_text().strip():
        run_test_helpers.kill_session(int(sid_file.read_text()))


@pytest.mark.usefixtures("stand_in_session")
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=["sigint", "sigterm"])
def test_interrupted_run_stops_the_command_it_measures(
    tmp_path: pathlib.Path, signum: signal.Signals
) -> None:
    pivot = run_test_helpers.make_stand_in(tmp_path, _INTERRUPTIBLE)
    argv = [sys.executable, str(_RUN_PY), *run_test_helpers.run_argv(tmp_path, pivot)]
    harness = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sid = int(_helper_read_when_written(tmp_path / _SID_FILE))
    assert run.session_members(sid), "the measured session should be running before the signal"

    harness.send_signal(signum)

    exit_code = harness.wait(timeout=30)
    assert not _helper_session_alive(sid), "an interrupted run.py must stop what it measures"
    # A SIGKILLed stand-in writes nothing, so this fails if anything kills it before its SIGINT.
    assert (tmp_path / _SIGINT_FILE).exists(), "pivot must get SIGINT, its Ctrl-C path, first"
    # Killed by the signal (an uncaught KeyboardInterrupt), or exited 128 + signum.
    assert exit_code in (-signum, 128 + signum), exit_code


@pytest.mark.usefixtures("stand_in_session")
@pytest.mark.parametrize("second", [signal.SIGINT, signal.SIGTERM], ids=["sigint", "sigterm"])
def test_second_signal_during_the_grace_period_still_stops_the_command(
    tmp_path: pathlib.Path, second: signal.Signals
) -> None:
    pivot = run_test_helpers.make_stand_in(tmp_path, run_test_helpers.DEAF_STAND_IN)
    argv = [sys.executable, str(_RUN_PY), *run_test_helpers.run_argv(tmp_path, pivot)]
    harness = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sid = int(_helper_read_when_written(tmp_path / _SID_FILE))
    harness.send_signal(signal.SIGINT)
    # The stand-in has its SIGINT, so run.py is waiting out the grace period.
    _helper_read_when_written(tmp_path / _SIGINT_FILE)

    harness.send_signal(second)

    time.sleep(0.1)
    assert harness.poll() is None, "a second signal must not shorten the graceful stop"

    exit_code = harness.wait(timeout=30)
    assert not _helper_session_alive(sid), "a second signal must not orphan the measured session"
    # The first signal decides run.py's status; later signals only wait for its stop to finish.
    assert exit_code in (-signal.SIGINT, 128 + signal.SIGINT), exit_code


def test_main_leaves_the_callers_sigterm_handler_in_place(tmp_path: pathlib.Path) -> None:
    pivot = run_test_helpers.make_stand_in(tmp_path, _QUICK)
    before = signal.getsignal(signal.SIGTERM)
    try:
        assert run.main(run_test_helpers.run_argv(tmp_path, pivot)) == 0
        assert signal.getsignal(signal.SIGTERM) is before, "run.main() replaced the SIGTERM handler"
    finally:
        signal.signal(signal.SIGTERM, before)


def test_variants_keep_separate_caches_when_the_caller_sets_one(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path / "shared-cache"
    monkeypatch.setenv("PIVOT_CACHE_DIR", str(shared))
    argv = ["--project", "chain-10", "--variant", f"a={_PIVOT}", "--variant", f"b={_PIVOT}"]
    argv += ["--reps", "1", "--work", str(tmp_path / "work"), "--out-dir", str(tmp_path / "out")]

    assert run.main(argv) == 0

    for name in ("a", "b"):
        own = tmp_path / "work" / name / "chain-10" / ".pivot" / "cache"
        assert own.is_dir() and any(own.rglob("*")), f"variant {name} did not use its own cache"
    assert not shared.exists(), "no variant may write the caller's PIVOT_CACHE_DIR"


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
