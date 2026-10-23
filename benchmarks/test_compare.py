from __future__ import annotations

import json
import math
import pathlib
from typing import TYPE_CHECKING

import pytest

import compare

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

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


def _helper_jsonl(rows: list[dict[str, object]]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


_ROWS = _helper_jsonl(_helper_rows([1.0] * 6, [2.0] * 6))
_LIMITS = json.dumps({_SCENARIO: {"wall_pct": 10, "cpu_pct": 10}})
_COMPARE_ARGV = ("base.jsonl", "head.jsonl", "--threshold-file", "threshold.json")
_DERIVE_ARGV = ("--derive-threshold", "--calibration", "calibration.json")


def _helper_write(tmp_path: pathlib.Path, files: Mapping[str, str | bytes]) -> None:
    """Write valid result and threshold files under `tmp_path`, overlaid with `files`."""
    defaults = {"base.jsonl": _ROWS, "head.jsonl": _ROWS, "threshold.json": _LIMITS}
    for name, data in {**defaults, **files}.items():
        (tmp_path / name).write_bytes(data.encode() if isinstance(data, str) else data)


def _helper_main(tmp_path: pathlib.Path, argv: tuple[str, ...]) -> int:
    """Run compare.py with each file name in `argv` resolved under `tmp_path`."""
    return compare.main([str(tmp_path / arg) if "." in arg else arg for arg in argv])


def _helper_compare(
    tmp_path: pathlib.Path,
    base: list[dict[str, object]],
    head: list[dict[str, object]],
    *extra: str,
    limits: Mapping[str, Mapping[str, float]] | None = None,
) -> int:
    threshold = _LIMITS if limits is None else json.dumps(limits)
    files = {"base.jsonl": _helper_jsonl(base), "head.jsonl": _helper_jsonl(head)}
    _helper_write(tmp_path, {**files, "threshold.json": threshold})
    return _helper_main(tmp_path, ("base.jsonl", "head.jsonl", *extra) if extra else _COMPARE_ARGV)


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


def test_interval_with_lower_bound_at_one_is_not_a_regression(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The median change is +25% against a 10% limit, but half the reps did not change at all,
    # so the interval's lower bound is exactly 1, which is not entirely above 1.
    base_wall = [1.0] * 6
    head_wall = [1.0, 1.0, 1.0, 1.5, 1.5, 1.5]
    assert compare.ratio_interval(base_wall, head_wall)[0] == 1.0, "the construction needs a 1"
    base = _helper_rows(base_wall, [2.0] * 6)
    head = _helper_rows(head_wall, [2.0] * 6)

    exit_code = _helper_compare(tmp_path, base, head)

    out = capsys.readouterr().out
    assert exit_code == 0, f"an interval that reaches 1 is not entirely above 1:\n{out}"
    assert "REGRESSION" not in out, out


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
    # The interval is taken over the per-rep ratios head[i] / base[i], but the minimum effect is
    # the change from the base's median to the head's, the statistic threshold.json was
    # calibrated on. Every ratio here is above 1, so only the minimum effect decides.
    base_wall = [1.0, 2.0, 3.0, 4.0, 5.0]
    head_wall = [round(b * r, 6) for b, r in zip(base_wall, ratios, strict=True)]
    base = _helper_rows(base_wall, [2.0] * 5)
    head = _helper_rows(head_wall, [2.0] * 5)

    exit_code = _helper_compare(tmp_path, base, head)

    out = capsys.readouterr().out
    assert exit_code == expected_exit, (
        "the limit applies to the change between the medians, not to the median per-rep ratio:\n"
        + out
    )


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
    # The runner drifts slower across the job; each head rep is 15% slower than the base rep
    # that ran beside it. An unpaired bootstrap puts that shared drift into the interval and
    # passes the slowdown. Written in reverse order, the head pairs correctly only if reps are
    # matched by number.
    base_wall = [1.0 + 0.1 * rep for rep in range(12)]
    base = _helper_rows(base_wall, [2.0] * 12)
    head = _helper_rows([round(1.15 * w, 6) for w in base_wall], [2.0] * 12)[::-1]

    assert _helper_compare(tmp_path, base, head) == 1
    assert f"REGRESSION {_SCENARIO} wall_s" in " ".join(capsys.readouterr().out.split())


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


def test_threshold_missing_one_metric_cannot_be_compared(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _helper_rows([1.0] * 6, [2.0] * 6)
    head = _helper_rows([1.0] * 6, [2.0] * 6)

    assert _helper_compare(tmp_path, base, head, limits={_SCENARIO: {"wall_pct": 10}}) == 2
    assert f"no threshold for {_SCENARIO} cpu_pct" in capsys.readouterr().err


def test_runs_on_different_core_counts_cannot_be_compared(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _helper_rows([1.0] * 6, [2.0] * 6, cpu_count=4)
    head = _helper_rows([1.0] * 6, [2.0] * 6, cpu_count=2)

    assert _helper_compare(tmp_path, base, head) == 2
    assert "core counts" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param(("--derive-threshold",), id="derive-threshold"),
        pytest.param((), id="threshold-file"),
    ],
)
def test_non_string_scenario_cannot_be_compared(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], extra: tuple[str, ...]
) -> None:
    rows = [{**row, "scenario": 5} for row in _helper_rows([1.0] * 6, [2.0] * 6)]
    limits = {"5": {"wall_pct": 10, "cpu_pct": 10}}

    assert _helper_compare(tmp_path, rows, rows, *extra, limits=limits) == 2
    err = capsys.readouterr().err
    assert err.startswith("cannot compare: ") and "scenario must be a string" in err, err


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


# Two A/A runs against the same base: wall +0.4% and +1.8%, CPU -0.5% and +3%.
_AA_HEADS = [([1.004] * 6, [1.99] * 6), ([1.018] * 6, [2.06] * 6)]


def _helper_write_aa_runs(tmp_path: pathlib.Path) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """Write each run of _AA_HEADS as a BASE HEAD pair of result files; return their paths."""
    pairs = list[tuple[pathlib.Path, pathlib.Path]]()
    for index, (wall, cpu) in enumerate(_AA_HEADS):
        base, head = tmp_path / f"{index}-base.jsonl", tmp_path / f"{index}-head.jsonl"
        base.write_text(_ROWS)
        head.write_text(_helper_jsonl(_helper_rows(wall, cpu)))
        pairs.append((base, head))
    return pairs


def test_derived_threshold_uses_the_largest_spread_from_independent_aa_pairs(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [str(path) for pair in _helper_write_aa_runs(tmp_path) for path in pair]

    assert compare.main([*argv, "--derive-threshold"]) == 0
    assert json.loads(capsys.readouterr().out) == {_SCENARIO: {"wall_pct": 6, "cpu_pct": 9}}


def test_calibration_file_records_each_sample_and_rebuilds_the_same_thresholds(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--write-calibration", str(tmp_path / "calibration.json")]
    for index, (base, head) in enumerate(_helper_write_aa_runs(tmp_path)):
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


def _helper_remove(path: pathlib.Path) -> None:
    path.unlink()


def _helper_replace_with_a_directory(path: pathlib.Path) -> None:
    path.unlink()
    path.mkdir()


def _helper_cut_mid_row(path: pathlib.Path) -> None:
    path.write_text(path.read_text()[:-40])


@pytest.mark.parametrize(
    ("name", "damage", "message"),
    [
        pytest.param("head.jsonl", _helper_remove, ": no such file", id="missing-result-file"),
        pytest.param(
            "threshold.json", _helper_remove, ": no such file", id="missing-threshold-file"
        ),
        pytest.param(
            "head.jsonl",
            _helper_replace_with_a_directory,
            ": cannot read",
            id="result-path-is-a-directory",
        ),
        pytest.param(
            "threshold.json",
            _helper_replace_with_a_directory,
            ": cannot read",
            id="threshold-path-is-a-directory",
        ),
        # A writer killed mid-line leaves the last row truncated.
        pytest.param(
            "head.jsonl", _helper_cut_mid_row, ":6: not a result row", id="truncated-result-file"
        ),
    ],
)
def test_file_that_cannot_be_read_is_named_in_one_line(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
    damage: Callable[[pathlib.Path], None],
    message: str,
) -> None:
    _helper_write(tmp_path, {})
    damage(tmp_path / name)

    exit_code = _helper_main(tmp_path, _COMPARE_ARGV)

    err = capsys.readouterr().err
    assert exit_code == 2, f"{name} that cannot be read must exit 2, not {exit_code}: {err}"
    assert err.startswith(f"cannot compare: {tmp_path / name}{message}"), err
    assert err.count("\n") == 1, err


def _helper_rows_edited(
    edit: Mapping[str, object], *, drop: tuple[str, ...] = (), reps: Collection[int] = (0,)
) -> str:
    """Six valid rows as JSONL, with `edit` applied to, and `drop` removed from, rows `reps`."""
    rows = _helper_rows([1.0] * 6, [2.0] * 6)
    for rep in reps:
        rows[rep] = {key: value for key, value in {**rows[rep], **edit}.items() if key not in drop}
    return _helper_jsonl(rows)


# An integer with more digits than Python reads (sys.get_int_max_str_digits() defaults to 4300),
# which json.dumps cannot write, and one too large for a float, which it can.
_LONG_INT = "1" + "0" * 5000
_BIG_INT = 10**400
_NESTED_ROW = "[" * 100_000 + "]" * 100_000 + "\n"


def _helper_first_row(field: str, value: object) -> str:
    """Six valid rows as JSONL, with row 0's `field` set to `value`."""
    return _helper_rows_edited({field: value})


def _helper_limits(wall_pct: object) -> str:
    """A threshold file whose wall_pct limit is `wall_pct`."""
    return json.dumps({_SCENARIO: {"wall_pct": wall_pct, "cpu_pct": 10}})


def _helper_calibration(wall_pct: object) -> str:
    """A calibration file of one sample whose wall_pct change is `wall_pct`."""
    return json.dumps({"samples": [{"changes": {_SCENARIO: {"wall_pct": wall_pct, "cpu_pct": 1}}}]})


@pytest.mark.parametrize(
    ("files", "argv"),
    [
        # Result rows: a field missing, of the wrong JSON type, or unparseable.
        pytest.param(
            {"head.jsonl": _helper_rows_edited({}, drop=("cpu_s",))},
            _COMPARE_ARGV,
            id="row-without-cpu_s",
        ),
        pytest.param(
            {"head.jsonl": _helper_rows_edited({}, drop=("cpu_count",))},
            _COMPARE_ARGV,
            id="row-without-cpu_count",
        ),
        pytest.param(
            {"head.jsonl": _helper_rows_edited({"wall_s": "fast"})},
            _COMPARE_ARGV,
            id="non-numeric-wall_s",
        ),
        pytest.param(
            {"head.jsonl": _helper_rows_edited({"exit_code": 1}, drop=("rep",))},
            _COMPARE_ARGV,
            id="failed-row-without-rep",
        ),
        pytest.param(
            {"head.jsonl": _helper_rows_edited({"cpu_count": float("inf")})},
            _COMPARE_ARGV,
            id="infinite-cpu_count",
        ),
        pytest.param(
            {"head.jsonl": _ROWS + '{"scenario": "chain-10'},
            _COMPARE_ARGV,
            id="truncated-result-line",
        ),
        # run.py writes rep, exit_code and cpu_count as integers and wall_s and cpu_s as
        # numbers. Reading them with int() and float() would truncate a fraction (rep 1.9 pairs
        # as rep 1, exit code 0.9 counts as success), read the strings "1" and "2.5" as numbers,
        # and read true and false as 1 and 0; each edit sits where that would make a valid file.
        pytest.param(
            {"head.jsonl": _helper_rows_edited({"rep": 1.9}, reps=(1,))},
            _COMPARE_ARGV,
            id="fractional-rep",
        ),
        pytest.param(
            {"head.jsonl": _helper_rows_edited({"rep": "1"}, reps=(1,))},
            _COMPARE_ARGV,
            id="string-rep",
        ),
        pytest.param(
            {"head.jsonl": _helper_rows_edited({"rep": True}, reps=(1,))},
            _COMPARE_ARGV,
            id="boolean-rep",
        ),
        pytest.param(
            {"head.jsonl": _helper_rows_edited({"exit_code": 0.9})},
            _COMPARE_ARGV,
            id="fractional-exit-code",
        ),
        pytest.param(
            {"head.jsonl": _helper_rows_edited({"exit_code": False})},
            _COMPARE_ARGV,
            id="boolean-exit-code",
        ),
        pytest.param(
            {"head.jsonl": _helper_rows_edited({"cpu_count": 4.5}, reps=range(6))},
            _COMPARE_ARGV,
            id="fractional-cpu-count",
        ),
        pytest.param(
            {"head.jsonl": _helper_rows_edited({"wall_s": "2.5"})},
            _COMPARE_ARGV,
            id="string-wall",
        ),
        pytest.param(
            {"head.jsonl": _helper_rows_edited({"cpu_s": True})},
            _COMPARE_ARGV,
            id="boolean-cpu",
        ),
        # A side's reps must be 0 to n-1, once each.
        pytest.param(
            {"head.jsonl": _helper_rows_edited({"rep": 4}, reps=(5,))},
            _COMPARE_ARGV,
            id="duplicated-rep",
        ),
        pytest.param(
            {"head.jsonl": _helper_rows_edited({"rep": 6}, reps=(5,))},
            _COMPARE_ARGV,
            id="missing-rep",
        ),
        pytest.param(
            {"head.jsonl": _helper_rows_edited({"rep": -1}, reps=(5,))},
            _COMPARE_ARGV,
            id="out-of-range-rep",
        ),
        # Python's json module reads the bare tokens NaN and Infinity as floats, and a zero base
        # median leaves the change and the head/base ratio undefined.
        pytest.param(
            {
                "head.jsonl": _helper_jsonl(
                    _helper_rows([math.nan, 1.0, math.nan, 1.0, math.nan, 1.0], [2.0] * 6)
                )
            },
            _COMPARE_ARGV,
            id="nan-wall-in-head",
        ),
        pytest.param(
            {"head.jsonl": _helper_jsonl(_helper_rows([1.0] * 6, [math.inf] * 6))},
            _COMPARE_ARGV,
            id="infinite-cpu-in-head",
        ),
        pytest.param(
            {
                "base.jsonl": _helper_jsonl(_helper_rows([1.0] * 6, [0.0] * 6)),
                "head.jsonl": _helper_jsonl(_helper_rows([1.0] * 6, [0.0] * 6)),
            },
            _COMPARE_ARGV,
            id="zero-cpu-on-both-sides",
        ),
        pytest.param(
            {"head.jsonl": _helper_first_row("wall_s", _BIG_INT)},
            _COMPARE_ARGV,
            id="timing-too-large-for-a-float",
        ),
        pytest.param(
            {"head.jsonl": _helper_first_row("cpu_s", _BIG_INT)},
            ("base.jsonl", "head.jsonl", "--derive-threshold"),
            id="timing-too-large-for-a-float-derive",
        ),
        pytest.param(
            {
                "base.jsonl": _helper_jsonl(_helper_rows([1e-310] * 6, [2.0] * 6)),
                "head.jsonl": _helper_jsonl(_helper_rows([1e10] * 6, [2.0] * 6)),
            },
            ("base.jsonl", "head.jsonl", "--derive-threshold"),
            id="change-between-medians-not-finite",
        ),
        pytest.param(
            {
                "base.jsonl": _helper_jsonl(_helper_rows([1e-310] * 6, [2.0] * 6)),
                "head.jsonl": _helper_jsonl(_helper_rows([1e10] * 6, [2.0] * 6)),
            },
            _COMPARE_ARGV,
            id="change-between-medians-not-finite-compare",
        ),
        pytest.param(
            {
                "base.jsonl": _helper_jsonl(_helper_rows([1e-310] * 6, [2.0] * 6)),
                "head.jsonl": _helper_jsonl(_helper_rows([1e10] * 6, [2.0] * 6)),
            },
            ("--write-calibration", "calibration.json")
            + ("--sample", "1", "1", "1", "1", "base.jsonl", "head.jsonl"),
            id="change-between-medians-not-finite-write-calibration",
        ),
        pytest.param(
            {"head.jsonl": _ROWS.encode() + b"\xff\n"},
            _COMPARE_ARGV,
            id="result-file-not-utf8",
        ),
        pytest.param({"head.jsonl": _ROWS + _NESTED_ROW}, _COMPARE_ARGV, id="deeply-nested-row"),
        pytest.param(
            {
                "head.jsonl": _helper_jsonl(
                    _helper_rows([1.0] * 6, [2.0] * 6, scenario=f"{_SCENARIO}\nwarm")
                )
            },
            _COMPARE_ARGV,
            id="scenario-name-with-a-newline",
        ),
        # Threshold files. A NaN limit makes every comparison against it false, so the scenario
        # could never regress.
        pytest.param({"threshold.json": ""}, _COMPARE_ARGV, id="empty-threshold-file"),
        pytest.param(
            {"threshold.json": json.dumps({_SCENARIO: 5})},
            _COMPARE_ARGV,
            id="threshold-not-a-mapping",
        ),
        pytest.param(
            {"threshold.json": _helper_limits("ten")}, _COMPARE_ARGV, id="non-numeric-limit"
        ),
        pytest.param(
            {
                "head.jsonl": _helper_jsonl(_helper_rows([2.0] * 6, [2.0] * 6)),
                "threshold.json": _helper_limits(math.nan),
            },
            _COMPARE_ARGV,
            id="non-finite-limit",
        ),
        pytest.param(
            {"threshold.json": b'{"\xff": 1}'}, _COMPARE_ARGV, id="threshold-file-not-utf8"
        ),
        pytest.param(
            {"threshold.json": _helper_limits(_BIG_INT)},
            _COMPARE_ARGV,
            id="limit-too-large-for-a-float",
        ),
        pytest.param(
            # The one input json.dumps cannot write.
            {
                "threshold.json": _helper_limits(0).replace(
                    '"wall_pct": 0', f'"wall_pct": {_LONG_INT}'
                )
            },
            _COMPARE_ARGV,
            id="limit-too-long-to-read",
        ),
        # Calibration files. A change must be a finite int or float, as a result row's timing is.
        pytest.param({"calibration.json": "{}"}, _DERIVE_ARGV, id="no-samples"),
        # An empty or wrong-type `samples` derives no limits, so it is unreadable too.
        pytest.param(
            {"calibration.json": json.dumps({"samples": ""})},
            _DERIVE_ARGV,
            id="samples-empty-string",
        ),
        pytest.param(
            {"calibration.json": json.dumps({"samples": {}})},
            _DERIVE_ARGV,
            id="samples-empty-object",
        ),
        pytest.param(
            {"calibration.json": json.dumps({"samples": []})},
            _DERIVE_ARGV,
            id="samples-empty-list",
        ),
        pytest.param(
            {"calibration.json": '{"samples": [{"sample": 1}]}'},
            _DERIVE_ARGV,
            id="sample-without-changes",
        ),
        pytest.param({"calibration.json": "not json"}, _DERIVE_ARGV, id="calibration-not-json"),
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
        pytest.param(
            {"calibration.json": _helper_calibration(math.nan)},
            _DERIVE_ARGV,
            id="calibration-change-nan",
        ),
        pytest.param(
            {"calibration.json": _helper_calibration(math.inf)},
            _DERIVE_ARGV,
            id="calibration-change-infinity",
        ),
        pytest.param(
            {"calibration.json": _helper_calibration("5")},
            _DERIVE_ARGV,
            id="calibration-change-string",
        ),
        pytest.param(
            {"calibration.json": _helper_calibration(True)},
            _DERIVE_ARGV,
            id="calibration-change-boolean",
        ),
        pytest.param(
            {"calibration.json": _helper_calibration(_BIG_INT)},
            _DERIVE_ARGV,
            id="calibration-change-too-large-for-a-float",
        ),
        pytest.param(
            {"calibration.json": _helper_calibration(1e308)},
            _DERIVE_ARGV,
            id="calibration-change-too-large-for-a-threshold",
        ),
    ],
)
def test_input_that_cannot_be_compared_exits_2_with_one_line(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    files: dict[str, str | bytes],
    argv: tuple[str, ...],
) -> None:
    _helper_write(tmp_path, files)

    exit_code = _helper_main(tmp_path, argv)

    err = capsys.readouterr().err
    assert exit_code == 2, f"input that cannot be compared must exit 2, not {exit_code}: {err}"
    assert err.startswith("cannot compare: ") and err.count("\n") == 1, err
