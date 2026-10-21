"""An interval that reaches 1 is not entirely above 1, so the metric is not a regression.

The gate ruled for PR #501 flags a metric only when the bootstrap 95% interval of the median
per-rep ratio lies entirely above 1. Here the median change is +25% against a 10% limit, but half
the reps did not change at all, so the interval's lower bound is exactly 1.
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


def test_interval_with_lower_bound_at_one_is_not_a_regression(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = [1.0] * 6
    head = [1.0, 1.0, 1.0, 1.5, 1.5, 1.5]
    assert compare.ratio_interval(base, head)[0] == 1.0, "the construction needs a bound of 1"
    (tmp_path / "base.jsonl").write_text(_helper_jsonl(base))
    (tmp_path / "head.jsonl").write_text(_helper_jsonl(head))
    (tmp_path / "threshold.json").write_text(
        json.dumps({_SCENARIO: {"wall_pct": 10, "cpu_pct": 10}})
    )
    argv = [str(tmp_path / "base.jsonl"), str(tmp_path / "head.jsonl")]

    exit_code = compare.main([*argv, "--threshold-file", str(tmp_path / "threshold.json")])

    out = capsys.readouterr().out
    assert exit_code == 0, f"an interval that reaches 1 is not entirely above 1:\n{out}"
    assert "REGRESSION" not in out, out
