from __future__ import annotations

import json
import math
import pathlib

import pytest

import compare

_BENCHMARKS = pathlib.Path(__file__).parent

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


def test_median_over_limit_inside_the_noise_is_not_a_regression(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _helper_rows([1.0] * 6, [2.0] * 6)
    # The head's median wall time is 20% above the base's, past the 10% limit, but the reps
    # scatter on both sides of the base, so the paired bootstrap interval includes no change.
    head = _helper_rows([0.7, 0.8, 0.9, 1.5, 1.6, 1.7], [2.0] * 6)

    assert _helper_compare(tmp_path, base, head) == 0, "noise alone must not fail the job"
    out = capsys.readouterr().out
    assert "REGRESSION" not in out and "95% CI" in out, out


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(48, (1.02602795595269, 1.0805473106445675), id="48-reps"),
        pytest.param(47, (1.02602795595269, 1.0826918658088236), id="47-reps"),
        pytest.param(46, (1.02602795595269, 1.0786675845188924), id="46-reps"),
        pytest.param(45, (1.02602795595269, 1.0786675845188924), id="45-reps"),
        pytest.param(44, (1.02602795595269, 1.080679725163858), id="44-reps"),
        pytest.param(43, (1.02602795595269, 1.0786675845188924), id="43-reps"),
    ],
)
def test_bootstrap_bounds_are_fixed_by_the_seed(length: int, expected: tuple[float, float]) -> None:
    # Six inputs of distinct lengths draw distinct resample streams. With the seed removed,
    # none of 400 other seeds reproduced all six bounds.
    base = _TWO_CLUSTER_BASE[:length]
    head = [round(w * (1.05 + 0.07 * math.sin(3.7 * rep)), 4) for rep, w in enumerate(base)]

    assert compare.ratio_interval(base, head) == expected


def test_unequal_rep_counts_cannot_be_compared(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _helper_rows([1.0] * 6, [2.0] * 6)
    head = _helper_rows([1.0] * 5, [2.0] * 5)

    assert _helper_compare(tmp_path, base, head) == 2
    assert "rep counts" in capsys.readouterr().err


# 48 base wall times, in rep order, from run 39531104040's fan-100/pipeline/touch: two clusters,
# 28 reps at 2.57-2.92 s and 20 at 3.32-3.59 s.
_TWO_CLUSTER_BASE = [
    3.4788, 2.803, 2.6493, 2.7654, 2.6445, 2.6971, 2.6471, 3.3751,
    3.3989, 3.4298, 3.4075, 3.4097, 3.4259, 3.5547, 3.5885, 2.6731,
    2.7271, 2.6318, 2.6324, 2.5991, 2.9167, 2.594, 2.5655, 2.7168,
    2.6047, 3.3266, 3.464, 3.3441, 3.4536, 3.4816, 3.3693, 2.7162,
    2.5851, 2.6148, 2.7203, 2.6281, 2.5904, 2.6322, 2.703, 3.3857,
    3.3242, 2.5654, 2.6122, 2.6271, 2.5722, 2.7263, 3.3286, 3.4334,
]  # fmt: skip


def test_a_few_fast_head_reps_do_not_hide_a_regression(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Every rep is 10% slower except reps 0-3, which are extreme outliers on the fast side.
    head_wall = [w * (0.01 if rep < 4 else 1.10) for rep, w in enumerate(_TWO_CLUSTER_BASE)]
    base = _helper_rows(_TWO_CLUSTER_BASE, [2.0] * 48)
    head = _helper_rows(head_wall, [2.0] * 48)
    limits = {_SCENARIO: {"wall_pct": 2, "cpu_pct": 2}}

    assert _helper_compare(tmp_path, base, head, limits=limits) == 1
    assert f"REGRESSION {_SCENARIO} wall_s" in " ".join(capsys.readouterr().out.split())


def test_reps_pair_by_rep_number_not_by_file_order(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The runner drifts slower across the job; each head rep is 15% slower than its base rep.
    # Written in reverse order, the head pairs correctly only if reps are matched by number.
    base_wall = [1.0 + 0.1 * rep for rep in range(12)]
    base = _helper_rows(base_wall, [2.0] * 12)
    head = _helper_rows([round(1.15 * w, 6) for w in base_wall], [2.0] * 12)[::-1]

    assert _helper_compare(tmp_path, base, head) == 1
    assert f"REGRESSION {_SCENARIO} wall_s" in " ".join(capsys.readouterr().out.split())


@pytest.mark.parametrize(
    "reps",
    [
        pytest.param([0, 1, 2, 3, 4, 4], id="duplicated-rep"),
        pytest.param([0, 1, 2, 3, 4, 6], id="missing-rep"),
        pytest.param([0, 1, 2, 3, 4, -1], id="out-of-range-rep"),
    ],
)
def test_reps_that_do_not_pair_cannot_be_compared(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], reps: list[int]
) -> None:
    base = _helper_rows([1.0] * 6, [2.0] * 6)
    head = _helper_rows([1.0] * 6, [2.0] * 6)
    for row, rep in zip(head, reps, strict=True):
        row["rep"] = rep

    assert _helper_compare(tmp_path, base, head) == 2
    err = capsys.readouterr().err
    assert err.startswith("cannot compare: ") and err.count("\n") == 1, err


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
        edit: {"wall_pct": 2, "cpu_pct": 2},
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


def test_threshold_missing_one_metric_cannot_be_compared(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _helper_rows([1.0] * 6, [2.0] * 6)
    head = _helper_rows([1.0] * 6, [2.0] * 6)

    assert _helper_compare(tmp_path, base, head, limits={_SCENARIO: {"wall_pct": 10}}) == 2
    assert f"no threshold for {_SCENARIO} cpu_pct" in capsys.readouterr().err


@pytest.mark.parametrize(
    "missing",
    [
        pytest.param("head.jsonl", id="result-file"),
        pytest.param("threshold.json", id="threshold-file"),
    ],
)
def test_missing_file_cannot_be_compared(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], missing: str
) -> None:
    rows = "".join(json.dumps(row) + "\n" for row in _helper_rows([1.0] * 6, [2.0] * 6))
    (tmp_path / "base.jsonl").write_text(rows)
    (tmp_path / "head.jsonl").write_text(rows)
    (tmp_path / "threshold.json").write_text(
        json.dumps({_SCENARIO: {"wall_pct": 10, "cpu_pct": 10}})
    )
    (tmp_path / missing).unlink()
    argv = [str(tmp_path / "base.jsonl"), str(tmp_path / "head.jsonl")]

    assert compare.main([*argv, "--threshold-file", str(tmp_path / "threshold.json")]) == 2
    assert f"{tmp_path / missing}: no such file" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("threshold_text", "head_tail"),
    [
        pytest.param("", "", id="empty-threshold-file"),
        pytest.param('{"chain-10/pipeline/cold": 5}', "", id="threshold-not-a-mapping"),
        pytest.param(
            '{"chain-10/pipeline/cold": {"wall_pct": 10, "cpu_pct": 10}}',
            '{"scenario": "chain-10',
            id="truncated-result-line",
        ),
    ],
)
def test_malformed_input_cannot_be_compared(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    threshold_text: str,
    head_tail: str,
) -> None:
    rows = "".join(json.dumps(row) + "\n" for row in _helper_rows([1.0] * 6, [2.0] * 6))
    (tmp_path / "base.jsonl").write_text(rows)
    (tmp_path / "head.jsonl").write_text(rows + head_tail)
    (tmp_path / "threshold.json").write_text(threshold_text)
    argv = [str(tmp_path / "base.jsonl"), str(tmp_path / "head.jsonl")]

    assert compare.main([*argv, "--threshold-file", str(tmp_path / "threshold.json")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("cannot compare: ") and err.count("\n") == 1, err


@pytest.mark.parametrize(
    "calibration_text",
    [
        pytest.param("{}", id="no-samples"),
        pytest.param('{"samples": [{"sample": 1}]}', id="sample-without-changes"),
        pytest.param("not json", id="not-json"),
    ],
)
def test_malformed_calibration_cannot_be_derived(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], calibration_text: str
) -> None:
    (tmp_path / "calibration.json").write_text(calibration_text)

    assert (
        compare.main(["--derive-threshold", "--calibration", str(tmp_path / "calibration.json")])
        == 2
    )
    err = capsys.readouterr().err
    assert err.startswith("cannot compare: ") and err.count("\n") == 1, err


def _helper_jsonl(rows: list[dict[str, object]]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def _helper_first_row_edited(edit: dict[str, object], drop: tuple[str, ...] = ()) -> str:
    rows = _helper_rows([1.0] * 6, [2.0] * 6)
    rows[0] = {key: value for key, value in {**rows[0], **edit}.items() if key not in drop}
    return _helper_jsonl(rows)


_COMPARE_ARGV = ("base.jsonl", "head.jsonl", "--threshold-file", "threshold.json")
_DERIVE_ARGV = ("--derive-threshold", "--calibration", "calibration.json")


@pytest.mark.parametrize(
    ("files", "argv"),
    [
        pytest.param(
            {"head.jsonl": _helper_first_row_edited({}, drop=("cpu_s",))},
            _COMPARE_ARGV,
            id="row-without-cpu_s",
        ),
        pytest.param(
            {"head.jsonl": _helper_first_row_edited({}, drop=("cpu_count",))},
            _COMPARE_ARGV,
            id="row-without-cpu_count",
        ),
        pytest.param(
            {"head.jsonl": _helper_first_row_edited({"wall_s": "fast"})},
            _COMPARE_ARGV,
            id="non-numeric-wall_s",
        ),
        pytest.param(
            {"head.jsonl": _helper_first_row_edited({"exit_code": 1}, drop=("rep",))},
            _COMPARE_ARGV,
            id="failed-row-without-rep",
        ),
        pytest.param(
            {"head.jsonl": _helper_first_row_edited({"cpu_count": float("inf")})},
            _COMPARE_ARGV,
            id="infinite-cpu_count",
        ),
        pytest.param(
            {"head.jsonl": _helper_jsonl(_helper_rows([1.0] * 6, [2.0] * 6))[:-40]},
            _COMPARE_ARGV,
            id="truncated-last-row",
        ),
        pytest.param(
            {"threshold.json": json.dumps({_SCENARIO: {"wall_pct": "ten", "cpu_pct": 10}})},
            _COMPARE_ARGV,
            id="non-numeric-limit",
        ),
        pytest.param(
            {
                "calibration.json": json.dumps(
                    {"samples": [{"changes": {_SCENARIO: {"wall_pct": 0.1}}}]}
                )
            },
            _DERIVE_ARGV,
            id="calibration-change-without-cpu_pct",
        ),
        pytest.param(
            {"calibration.json": json.dumps({"samples": [{"changes": [0.1, 0.2]}]})},
            _DERIVE_ARGV,
            id="calibration-changes-not-a-mapping",
        ),
    ],
)
def test_every_malformed_shape_exits_2_with_one_line(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    files: dict[str, str],
    argv: tuple[str, ...],
) -> None:
    rows = _helper_jsonl(_helper_rows([1.0] * 6, [2.0] * 6))
    limits = json.dumps({_SCENARIO: {"wall_pct": 10, "cpu_pct": 10}})
    defaults = {"base.jsonl": rows, "head.jsonl": rows, "threshold.json": limits}
    for name, text in {**defaults, **files}.items():
        (tmp_path / name).write_text(text)

    exit_code = compare.main([str(tmp_path / arg) if "." in arg else arg for arg in argv])

    err = capsys.readouterr().err
    assert exit_code == 2, f"malformed input must exit 2, not {exit_code}: {err}"
    assert err.startswith("cannot compare: ") and err.count("\n") == 1, err


def test_calibration_file_records_each_sample_and_rebuilds_the_same_thresholds(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = [([1.004] * 6, [1.99] * 6), ([1.018] * 6, [2.06] * 6)]
    argv = ["--write-calibration", str(tmp_path / "calibration.json")]
    for index, (wall, cpu) in enumerate(pairs):
        base = tmp_path / f"{index}-base.jsonl"
        head = tmp_path / f"{index}-head.jsonl"
        base.write_text(
            "".join(json.dumps(row) + "\n" for row in _helper_rows([1.0] * 6, [2.0] * 6))
        )
        head.write_text("".join(json.dumps(row) + "\n" for row in _helper_rows(wall, cpu)))
        argv += ["--sample", str(index + 1), "100", "1", str(200 + index), str(base), str(head)]

    assert compare.main(argv) == 0
    samples = json.loads((tmp_path / "calibration.json").read_text())["samples"]
    assert [(s["sample"], s["run"], s["attempt"], s["job"]) for s in samples] == [
        (1, 100, 1, 200),
        (2, 100, 1, 201),
    ]
    assert (
        compare.main(["--derive-threshold", "--calibration", str(tmp_path / "calibration.json")])
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {_SCENARIO: {"wall_pct": 6, "cpu_pct": 9}}


def test_committed_calibration_rebuilds_the_committed_thresholds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calibration = _BENCHMARKS / "calibration.json"

    assert compare.main(["--derive-threshold", "--calibration", str(calibration)]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(
        (_BENCHMARKS / "threshold.json").read_text()
    ), "threshold.json must be exactly what calibration.json derives"
