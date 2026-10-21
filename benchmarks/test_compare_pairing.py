"""The bootstrap pairs reps by index, so drift shared by base and head cannot hide a regression.

run.py interleaves base and head within each rep, so rep i of each side ran under the same
machine state. The gate ruled for PR #501 resamples those pairs together. An unpaired bootstrap
(base and head resampled independently) puts the shared drift into the interval, and a real
slowdown on a drifting runner then passes.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import compare

if TYPE_CHECKING:
    import pathlib

    import pytest

_SCENARIO = "chain-10/pipeline/cold"


def _helper_jsonl(wall: list[float]) -> str:
    rows = [
        {
            "scenario": _SCENARIO,
            "variant": "v",
            "rep": rep,
            "wall_s": w,
            "cpu_s": 2.0,
            "exit_code": 0,
            "load1": 0.5,
            "cpu_count": 4,
            "ran": 10,
            "cached": 0,
            "leftover_procs": 0,
        }
        for rep, w in enumerate(wall)
    ]
    return "".join(json.dumps(row) + "\n" for row in rows)


def test_drift_shared_by_both_sides_does_not_hide_a_regression(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The runner slows steadily across the job; every rep of the head is 15% slower than the
    # base rep that ran beside it.
    base = [1.0 + 0.1 * rep for rep in range(12)]
    head = [round(1.15 * wall, 6) for wall in base]
    (tmp_path / "base.jsonl").write_text(_helper_jsonl(base))
    (tmp_path / "head.jsonl").write_text(_helper_jsonl(head))
    (tmp_path / "threshold.json").write_text(
        json.dumps({_SCENARIO: {"wall_pct": 10, "cpu_pct": 10}})
    )
    argv = [str(tmp_path / "base.jsonl"), str(tmp_path / "head.jsonl")]

    exit_code = compare.main([*argv, "--threshold-file", str(tmp_path / "threshold.json")])

    out = capsys.readouterr().out
    assert exit_code == 1, f"a 15% slowdown in every paired rep must fail the job:\n{out}"
    assert f"REGRESSION {_SCENARIO} wall_s" in " ".join(out.split()), out
