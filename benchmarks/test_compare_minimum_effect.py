"""The minimum-effect check uses the change between the two medians, not the median per-rep ratio.

The gate ruled for PR #501 has two parts. The bootstrap interval is taken over the per-rep
ratios head[i] / base[i], but the minimum effect stays the change from the base's median to the
head's median, the statistic threshold.json was calibrated on. The two statistics disagree when
the reps' ratios differ across the job, so each case below passes one and fails the other. Every
per-rep ratio is above 1, so the interval lies entirely above 1 and only the minimum effect
decides.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import compare

if TYPE_CHECKING:
    import pathlib

_SCENARIO = "chain-10/pipeline/cold"
_BASE_WALL = [1.0, 2.0, 3.0, 4.0, 5.0]


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


@pytest.mark.parametrize(
    ("ratios", "expected_exit"),
    [
        # Head median 5.0 against base median 3.0 (+67%); median per-rep ratio 1.01 (+1%).
        pytest.param([5.0, 2.5, 1.01, 1.01, 1.01], 1, id="median-change-over-limit"),
        # Head median 3.06 against base median 3.0 (+2%); median per-rep ratio 1.2 (+20%).
        pytest.param([2.0, 1.3, 1.02, 1.2, 1.01], 0, id="median-change-under-limit"),
    ],
)
def test_minimum_effect_is_the_change_between_medians(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    ratios: list[float],
    expected_exit: int,
) -> None:
    head_wall = [round(b * r, 6) for b, r in zip(_BASE_WALL, ratios, strict=True)]
    (tmp_path / "base.jsonl").write_text(_helper_jsonl(_BASE_WALL))
    (tmp_path / "head.jsonl").write_text(_helper_jsonl(head_wall))
    (tmp_path / "threshold.json").write_text(
        json.dumps({_SCENARIO: {"wall_pct": 10, "cpu_pct": 10}})
    )
    argv = [str(tmp_path / "base.jsonl"), str(tmp_path / "head.jsonl")]

    exit_code = compare.main([*argv, "--threshold-file", str(tmp_path / "threshold.json")])

    out = capsys.readouterr().out
    assert exit_code == expected_exit, (
        "the limit applies to the change between the medians, not to the median per-rep ratio:\n"
        + out
    )
