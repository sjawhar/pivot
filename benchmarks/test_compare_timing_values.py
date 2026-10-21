"""A value compare.py cannot divide by or order is input it cannot compare: exit 2, one line.

compare.py's contract: exit 0 nothing regressed, 1 something regressed, 2 the files cannot be
compared. Python's json module reads the bare tokens NaN and Infinity as floats, so a result
row or a threshold file can carry them; and a zero base median leaves the change and the
head/base ratio undefined. None of these may pass the gate silently (exit 0) or crash with the
regression code (exit 1).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import compare

if TYPE_CHECKING:
    import pathlib

_SCENARIO = "chain-10/pipeline/cold"


def _helper_jsonl(wall: list[float], cpu: list[float]) -> str:
    rows = [
        {
            "scenario": _SCENARIO,
            "variant": "v",
            "rep": rep,
            "wall_s": w,
            "cpu_s": c,
            "exit_code": 0,
            "load1": 0.5,
            "cpu_count": 4,
            "ran": 10,
            "cached": 0,
            "leftover_procs": 0,
        }
        for rep, (w, c) in enumerate(zip(wall, cpu, strict=True))
    ]
    return "".join(json.dumps(row) + "\n" for row in rows)


@pytest.mark.parametrize(
    ("base", "head"),
    [
        pytest.param(
            ([1.0] * 6, [2.0] * 6),
            ([float("nan"), 1.0, float("nan"), 1.0, float("nan"), 1.0], [2.0] * 6),
            id="nan-wall-in-head",
        ),
        pytest.param(
            ([1.0] * 6, [2.0] * 6),
            ([1.0] * 6, [float("inf")] * 6),
            id="infinite-cpu-in-head",
        ),
        pytest.param(
            ([1.0] * 6, [0.0] * 6),
            ([1.0] * 6, [0.0] * 6),
            id="zero-cpu-on-both-sides",
        ),
    ],
)
def test_undividable_timing_cannot_be_compared(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    base: tuple[list[float], list[float]],
    head: tuple[list[float], list[float]],
) -> None:
    (tmp_path / "base.jsonl").write_text(_helper_jsonl(*base))
    (tmp_path / "head.jsonl").write_text(_helper_jsonl(*head))
    (tmp_path / "threshold.json").write_text(
        json.dumps({_SCENARIO: {"wall_pct": 10, "cpu_pct": 10}})
    )
    argv = [str(tmp_path / "base.jsonl"), str(tmp_path / "head.jsonl")]

    exit_code = compare.main([*argv, "--threshold-file", str(tmp_path / "threshold.json")])

    err = capsys.readouterr().err
    assert exit_code == 2, f"an undividable timing must exit 2, not {exit_code}: {err}"
    assert err.startswith("cannot compare: ") and err.count("\n") == 1, err


def test_non_finite_limit_cannot_be_compared(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A NaN limit makes every comparison against it false, so the scenario could never regress.
    (tmp_path / "base.jsonl").write_text(_helper_jsonl([1.0] * 6, [2.0] * 6))
    (tmp_path / "head.jsonl").write_text(_helper_jsonl([2.0] * 6, [2.0] * 6))
    (tmp_path / "threshold.json").write_text(
        json.dumps({_SCENARIO: {"wall_pct": float("nan"), "cpu_pct": 10}})
    )
    argv = [str(tmp_path / "base.jsonl"), str(tmp_path / "head.jsonl")]

    exit_code = compare.main([*argv, "--threshold-file", str(tmp_path / "threshold.json")])

    err = capsys.readouterr().err
    assert exit_code == 2, f"a non-finite limit must exit 2, not {exit_code}: {err}"
    assert err.startswith("cannot compare: ") and err.count("\n") == 1, err
