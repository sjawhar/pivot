from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import compare

if TYPE_CHECKING:
    import pathlib

_SCENARIO = "chain-10/pipeline/cold"
_WARM = "chain-10/pipeline/warm"


def _helper_rows(
    wall: list[float],
    cpu: list[float],
    *,
    scenario: str = _SCENARIO,
    exit_codes: list[int] | None = None,
    cpu_count: int = 4,
) -> list[dict[str, object]]:
    codes = exit_codes if exit_codes is not None else [0] * len(wall)
    return [
        {
            "scenario": scenario,
            "variant": "v",
            "rep": rep,
            "wall_s": w,
            "cpu_s": c,
            "exit_code": code,
            "load1": 0.5,
            "cpu_count": cpu_count,
            "ran": 10,
            "cached": 0,
            "leftover_procs": 0,
        }
        for rep, (w, c, code) in enumerate(zip(wall, cpu, codes, strict=True))
    ]


def _helper_compare(
    tmp_path: pathlib.Path,
    base: list[dict[str, object]],
    head: list[dict[str, object]],
    *extra: str,
    limits: dict[str, dict[str, int]] | None = None,
) -> int:
    base_path = tmp_path / "base.jsonl"
    head_path = tmp_path / "head.jsonl"
    base_path.write_text("".join(json.dumps(row) + "\n" for row in base))
    head_path.write_text("".join(json.dumps(row) + "\n" for row in head))
    if extra:
        return compare.main([str(base_path), str(head_path), *extra])
    threshold = tmp_path / "threshold.json"
    if limits is None:
        limits = {_SCENARIO: {"wall_pct": 10, "cpu_pct": 10}}
    threshold.write_text(json.dumps(limits))
    return compare.main([str(base_path), str(head_path), "--threshold-file", str(threshold)])


@pytest.mark.parametrize(
    ("head_wall", "head_cpu", "metric"),
    [
        # One patch set cut CPU at 1,000 stages while slowing wall time from 13.6 s to 24.3 s.
        pytest.param(1.8, 1.0, "wall_s", id="wall-slower-cpu-lower"),
        pytest.param(0.5, 2.5, "cpu_s", id="wall-faster-cpu-higher"),
    ],
)
def test_either_metric_over_threshold_is_a_regression(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    head_wall: float,
    head_cpu: float,
    metric: str,
) -> None:
    base = _helper_rows([1.0] * 6, [2.0] * 6)
    head = _helper_rows([head_wall] * 6, [head_cpu] * 6)

    assert _helper_compare(tmp_path, base, head) == 1, "a regression must exit 1"
    regressions = [line for line in capsys.readouterr().out.splitlines() if "REGRESSION" in line]
    assert len(regressions) == 1, f"exactly one metric regressed: {regressions}"
    assert _SCENARIO in regressions[0] and metric in regressions[0], regressions[0]


def test_each_scenario_is_held_to_its_own_threshold(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _helper_rows([1.0] * 6, [2.0] * 6) + _helper_rows([1.0] * 6, [2.0] * 6, scenario=_WARM)
    # Both scenarios' wall time rises 20%; only the warm scenario's limit is below that.
    head = _helper_rows([1.2] * 6, [2.0] * 6) + _helper_rows([1.2] * 6, [2.0] * 6, scenario=_WARM)
    limits = {
        _SCENARIO: {"wall_pct": 30, "cpu_pct": 5},
        _WARM: {"wall_pct": 10, "cpu_pct": 5},
    }

    assert _helper_compare(tmp_path, base, head, limits=limits) == 1
    regressions = [line for line in capsys.readouterr().out.splitlines() if "REGRESSION" in line]
    assert len(regressions) == 1 and _WARM in regressions[0], regressions


@pytest.mark.parametrize(
    ("head_wall", "head_cpu"),
    [
        pytest.param([1.05] * 6, [2.1] * 6, id="within-threshold"),
        pytest.param([0.5] * 6, [1.0] * 6, id="faster"),
        pytest.param([1.0, 1.0, 1.0, 1.0, 1.0, 9.0], [2.0] * 6, id="one-slow-rep"),
    ],
)
def test_head_within_threshold_of_base_medians_passes(
    tmp_path: pathlib.Path, head_wall: list[float], head_cpu: list[float]
) -> None:
    base = _helper_rows([1.0] * 6, [2.0] * 6)
    head = _helper_rows(head_wall, head_cpu)

    assert _helper_compare(tmp_path, base, head) == 0


def test_failed_run_cannot_be_compared(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _helper_rows([1.0] * 6, [2.0] * 6)
    head = _helper_rows([1.0] * 6, [2.0] * 6, exit_codes=[0, 0, 1, 0, 0, 0])

    assert _helper_compare(tmp_path, base, head) == 2
    assert "exited with code 1" in capsys.readouterr().err


def test_scenario_on_one_side_only_cannot_be_compared(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _helper_rows([1.0] * 6, [2.0] * 6) + _helper_rows([1.0] * 6, [2.0] * 6, scenario=_WARM)
    head = _helper_rows([1.0] * 6, [2.0] * 6)

    assert _helper_compare(tmp_path, base, head) == 2
    assert _WARM in capsys.readouterr().err


def test_scenario_without_threshold_cannot_be_compared(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _helper_rows([1.0] * 6, [2.0] * 6)
    head = _helper_rows([1.0] * 6, [2.0] * 6)

    assert _helper_compare(tmp_path, base, head, limits={}) == 2
    assert f"no threshold for {_SCENARIO}" in capsys.readouterr().err


def test_runs_on_different_core_counts_cannot_be_compared(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _helper_rows([1.0] * 6, [2.0] * 6, cpu_count=4)
    head = _helper_rows([1.0] * 6, [2.0] * 6, cpu_count=2)

    assert _helper_compare(tmp_path, base, head) == 2
    assert "core counts" in capsys.readouterr().err


def test_derived_threshold_is_three_times_each_scenarios_aa_spread_rounded_up(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    edit = "chain-10/pipeline/edit"
    base = (
        _helper_rows([1.0] * 6, [2.0] * 6)
        + _helper_rows([1.0] * 6, [2.0] * 6, scenario=_WARM)
        + _helper_rows([1.0] * 6, [2.0] * 6, scenario=edit)
    )
    # Spreads: cold wall 2.4%, cold cpu 0.5%, warm wall 1.1%, warm cpu 3.1% (head faster),
    # edit none at all.
    head = (
        _helper_rows([1.024] * 6, [2.01] * 6)
        + _helper_rows([1.011] * 6, [1.938] * 6, scenario=_WARM)
        + _helper_rows([1.0] * 6, [2.0] * 6, scenario=edit)
    )

    assert _helper_compare(tmp_path, base, head, "--derive-threshold") == 0
    assert json.loads(capsys.readouterr().out) == {
        _SCENARIO: {"wall_pct": 8, "cpu_pct": 2},
        _WARM: {"wall_pct": 4, "cpu_pct": 10},
        edit: {"wall_pct": 1, "cpu_pct": 1},
    }


def test_derived_threshold_uses_the_largest_spread_from_independent_aa_pairs(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first_base = tmp_path / "first-base.jsonl"
    first_head = tmp_path / "first-head.jsonl"
    second_base = tmp_path / "second-base.jsonl"
    second_head = tmp_path / "second-head.jsonl"
    first_base.write_text(
        "".join(json.dumps(row) + "\n" for row in _helper_rows([1.0] * 6, [2.0] * 6))
    )
    first_head.write_text(
        "".join(json.dumps(row) + "\n" for row in _helper_rows([1.004] * 6, [1.99] * 6))
    )
    second_base.write_text(
        "".join(json.dumps(row) + "\n" for row in _helper_rows([1.0] * 6, [2.0] * 6))
    )
    second_head.write_text(
        "".join(json.dumps(row) + "\n" for row in _helper_rows([1.018] * 6, [2.06] * 6))
    )

    assert (
        compare.main(
            [
                str(first_base),
                str(first_head),
                str(second_base),
                str(second_head),
                "--derive-threshold",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {_SCENARIO: {"wall_pct": 6, "cpu_pct": 9}}
