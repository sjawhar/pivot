"""Compare benchmark result files from run.py, scenario by scenario, and derive thresholds.

Usage: compare.py BASE.jsonl HEAD.jsonl --threshold-file benchmarks/threshold.json
       compare.py A1_BASE.jsonl A1_HEAD.jsonl [A2_BASE.jsonl A2_HEAD.jsonl ...] --derive-threshold
       compare.py --write-calibration benchmarks/calibration.json
                  --sample SAMPLE RUN ATTEMPT JOB BASE.jsonl HEAD.jsonl [--sample ...]
       compare.py --derive-threshold --calibration benchmarks/calibration.json

The threshold file gives each scenario its own limits, {"<scenario>": {"wall_pct": N,
"cpu_pct": N}}, because scenarios differ in noise: a warm no-op and a cold build do not vary
alike.

With --threshold-file, a scenario's metric is a REGRESSION only when both hold: the head's median
exceeds the base's by more than that scenario's `wall_pct` or `cpu_pct` (the minimum effect),
and a bootstrap 95% interval of the median per-rep ratio head/base, pairing reps by their `rep`
number, lies entirely above 1 (the change is not the job's own noise). It exits 0 when nothing
regressed, 1 when something did, and 2 when the files cannot be compared: a file is missing or
cannot be read, is not UTF-8, or cannot be parsed (JSON nested too deeply, or an integer too long
to read, included), a field has the wrong JSON type, a timing is not a positive finite number or
is too large for a float, a limit, an A/A change or the threshold derived from it is not a finite
number, a command failed, a scenario is on one side only or lacks a limit, a side's reps are not
0 to n-1 once each, or the two ran on machines with different core counts.

An A/A run measures the same build on both sides. With --derive-threshold, it prints a threshold
file: for each scenario and metric, three times the largest absolute change between the two
medians of any A/A run that measured it, rounded up to a whole percent and at least 2. The A/A
runs are either result-file pairs or the samples of a calibration file.

--write-calibration records each A/A run's per-scenario changes, with the GitHub Actions run,
attempt and job that produced it, so the thresholds can be rebuilt after the artifacts expire.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import statistics
import sys
from typing import Any, Literal, TypedDict, TypeGuard, cast

type _Metric = Literal["wall_s", "cpu_s"]
type _LimitKey = Literal["wall_pct", "cpu_pct"]

_METRICS: tuple[tuple[_Metric, _LimitKey], ...] = (("wall_s", "wall_pct"), ("cpu_s", "cpu_pct"))
_FLOOR_PCT = 2
_BOOTSTRAP_RESAMPLES = 10_000
_BOOTSTRAP_SEED = 0

# scenario -> limit key ("wall_pct" or "cpu_pct") -> percent change from base to head median
type _Changes = dict[str, dict[str, float]]


class _Row(TypedDict):
    """The fields of a run.py result row that compare.py reads."""

    wall_s: float
    cpu_s: float
    cpu_count: int
    rep: int


class _CannotCompareError(Exception):
    pass


def _read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        raise _CannotCompareError(f"{path}: no such file") from None
    except OSError as error:
        raise _CannotCompareError(f"{path}: cannot read ({error.strerror})") from None
    except UnicodeDecodeError as error:
        raise _CannotCompareError(f"{path}: not UTF-8 text ({error})") from None


def _read_json(path: pathlib.Path) -> object:
    text = _read_text(path)
    try:
        return json.loads(text)
    except RecursionError:
        raise _CannotCompareError(f"{path}: JSON nested too deeply to read") from None
    except ValueError as error:  # a JSONDecodeError, or an integer with too many digits to read
        raise _CannotCompareError(f"{path}: not valid JSON ({error})") from None


def _field_int(raw: dict[str, object], name: str) -> int:
    value = raw[name]
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, got {value!r}")
    return value


def _field_number(raw: dict[str, object], name: str) -> float:
    value = raw[name]
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a number, got {value!r}")
    return float(value)  # pyright: ignore[reportArgumentType] - int or float, checked above


def _field_str(raw: dict[str, object], name: str) -> str:
    value = raw[name]
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {value!r}")
    return value


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    """Whether a JSON value is an int or float, not a bool, that is finite as a float."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an integer too large for a float
        return False


def _load(path: pathlib.Path) -> dict[str, list[_Row]]:
    lines = _read_text(path).splitlines()
    by_scenario = dict[str, list[_Row]]()
    for number, line in enumerate(lines, start=1):
        try:
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise TypeError(f"a result row must be a JSON object, got {parsed!r}")
            raw = cast("dict[str, object]", parsed)
            scenario = _field_str(raw, "scenario")
            exit_code = _field_int(raw, "exit_code")
            rep = _field_int(raw, "rep")
            row = _Row(
                wall_s=_field_number(raw, "wall_s"),
                cpu_s=_field_number(raw, "cpu_s"),
                cpu_count=_field_int(raw, "cpu_count"),
                rep=rep,
            )
        except RecursionError:
            raise _CannotCompareError(f"{path}:{number}: JSON nested too deeply to read") from None
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError) as error:
            raise _CannotCompareError(f"{path}:{number}: not a result row ({error!r})") from None
        if exit_code != 0:
            raise _CannotCompareError(f"{path}: {scenario} rep {rep} exited with code {exit_code}")
        for metric, _ in _METRICS:
            if not (math.isfinite(row[metric]) and row[metric] > 0):
                raise _CannotCompareError(
                    f"{path}:{number}: {metric} must be a positive finite number, got {row[metric]!r}"
                )
        by_scenario.setdefault(scenario, []).append(row)
    return by_scenario


def _by_rep(path: pathlib.Path, scenario: str, rows: list[_Row]) -> list[_Row]:
    """Return the rows in rep order; reps must be exactly 0 .. n-1, each once."""
    ordered = sorted(rows, key=lambda row: row["rep"])
    reps = [row["rep"] for row in ordered]
    if reps != list(range(len(rows))):
        raise _CannotCompareError(
            f"{path}: {scenario} reps must be 0 to {len(rows) - 1} once each, got {reps}"
        )
    return ordered


def _paired(
    base_path: pathlib.Path, head_path: pathlib.Path
) -> dict[str, dict[_Metric, tuple[list[float], list[float]]]]:
    """Return scenario -> metric -> (base values, head values), paired by the `rep` field."""
    base = _load(base_path)
    head = _load(head_path)
    if one_side := sorted(set(base) ^ set(head)):
        raise _CannotCompareError(f"scenarios on one side only: {', '.join(one_side)}")
    cores = {row["cpu_count"] for rows in (*base.values(), *head.values()) for row in rows}
    if len(cores) != 1:
        raise _CannotCompareError(f"results come from machines with core counts {sorted(cores)}")
    if unequal := [s for s in sorted(base) if len(base[s]) != len(head[s])]:
        raise _CannotCompareError(f"base and head rep counts differ for {', '.join(unequal)}")
    paired = dict[str, dict[_Metric, tuple[list[float], list[float]]]]()
    for scenario in sorted(base):
        base_rows = _by_rep(base_path, scenario, base[scenario])
        head_rows = _by_rep(head_path, scenario, head[scenario])
        paired[scenario] = {
            metric: ([row[metric] for row in base_rows], [row[metric] for row in head_rows])
            for metric, _ in _METRICS
        }
    return paired


def ratio_interval(base: list[float], head: list[float]) -> tuple[float, float]:
    """Paired bootstrap 95% interval of the median per-rep ratio head[i] / base[i].

    run.py interleaves base and head within each rep, so rep i of each side shares the
    machine's state at that moment. Each resample draws reps with replacement and takes the
    median of their ratios, so a few fast or slow head reps move one ratio each rather than
    moving the head's median from one cluster of a bimodal job to another.
    """
    ratios = [h / b for b, h in zip(base, head, strict=True)]
    rng = random.Random(_BOOTSTRAP_SEED)
    n = len(ratios)
    medians = sorted(
        statistics.median(ratios[rng.randrange(n)] for _ in range(n))
        for _ in range(_BOOTSTRAP_RESAMPLES)
    )
    return medians[int(0.025 * _BOOTSTRAP_RESAMPLES)], medians[
        int(0.975 * _BOOTSTRAP_RESAMPLES) - 1
    ]


def _change_pct(base: float, head: float) -> float:
    return (head - base) / base * 100


def _changes(base_path: pathlib.Path, head_path: pathlib.Path) -> _Changes:
    changes = dict[str, dict[str, float]]()
    for scenario, by_metric in _paired(base_path, head_path).items():
        changes[scenario] = {}
        for metric, key in _METRICS:
            base_values, head_values = by_metric[metric]
            base, head = statistics.median(base_values), statistics.median(head_values)
            change = _change_pct(base, head)
            if not math.isfinite(change):
                raise _CannotCompareError(
                    f"{scenario} {metric}: the change from {base!r} to {head!r} is not finite"
                )
            changes[scenario][key] = change
    return changes


def _derive_threshold(samples: list[_Changes]) -> dict[str, dict[str, int]]:
    largest = dict[str, dict[str, float]]()
    for changes in samples:
        for scenario, by_key in changes.items():
            seen = largest.setdefault(scenario, {key: 0.0 for _, key in _METRICS})
            for _, key in _METRICS:
                seen[key] = max(seen[key], abs(by_key[key]))
    thresholds = dict[str, dict[str, int]]()
    for scenario, by_key in sorted(largest.items()):
        thresholds[scenario] = {}
        for key, spread in by_key.items():
            limit = round(3 * spread, 6)
            if not math.isfinite(limit):
                raise _CannotCompareError(
                    f"{scenario} {key}: an A/A change of {spread!r}% is too large for a threshold"
                )
            thresholds[scenario][key] = max(_FLOOR_PCT, math.ceil(limit))
    return thresholds


def _calibration_change(by_key: dict[str, object], key: str) -> float:
    value = by_key[key]
    if not _is_finite_number(value):
        raise ValueError(f"{key} must be a finite number, got {value!r}")
    return float(value)


def _read_calibration(path: pathlib.Path) -> list[_Changes]:
    # Building typed values checks the shape: any mismatch raises one of the caught errors.
    calibration: Any = _read_json(path)
    try:
        samples: list[_Changes] = [
            {
                str(scenario): {key: _calibration_change(by_key, key) for _, key in _METRICS}
                for scenario, by_key in sample["changes"].items()
            }
            for sample in calibration["samples"]
        ]
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise _CannotCompareError(f"{path}: not a calibration file ({error!r})") from None
    return samples


def _read_limits(path: pathlib.Path, scenarios: list[str]) -> dict[str, dict[str, float]]:
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise _CannotCompareError(f"{path}: not a threshold file")
    limits = dict[str, dict[str, float]]()
    missing = list[str]()
    for scenario in scenarios:
        by_key = raw.get(scenario)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType] - JSON object
        if not isinstance(by_key, dict):
            missing += [f"{scenario} {key}" for _, key in _METRICS]
            continue
        limits[scenario] = {}
        for _, key in _METRICS:
            value = by_key.get(key)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType] - JSON object
            if value is None:
                missing.append(f"{scenario} {key}")
            elif not _is_finite_number(value):  # pyright: ignore[reportUnknownArgumentType] - JSON object
                raise _CannotCompareError(
                    f"{path}: {scenario} {key} is not a finite number: {value!r}"
                )
            else:
                limits[scenario][key] = value
    if missing:
        raise _CannotCompareError(f"no threshold for {', '.join(missing)}")
    return limits


def _calibration_samples(
    parser: argparse.ArgumentParser, specs: list[list[str]]
) -> list[dict[str, object]]:
    samples = list[dict[str, object]]()
    for sample, run, attempt, job, base, head in specs:
        try:
            source = {
                "sample": int(sample),
                "run": int(run),
                "attempt": int(attempt),
                "job": int(job),
            }
        except ValueError:
            parser.error(
                f"--sample needs integer SAMPLE RUN ATTEMPT JOB, got {sample} {run} {attempt} {job}"
            )
        changes = _changes(pathlib.Path(base), pathlib.Path(head))
        samples.append({**source, "changes": changes})
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("result_files", nargs="*", type=pathlib.Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--threshold-file", type=pathlib.Path)
    mode.add_argument("--derive-threshold", action="store_true")
    mode.add_argument("--write-calibration", type=pathlib.Path)
    parser.add_argument("--calibration", type=pathlib.Path)
    parser.add_argument(
        "--sample",
        nargs=6,
        action="append",
        default=[],
        metavar=("SAMPLE", "RUN", "ATTEMPT", "JOB", "BASE", "HEAD"),
    )
    args = parser.parse_args(argv)
    result_files: list[pathlib.Path] = args.result_files
    calibration: pathlib.Path | None = args.calibration
    sample_specs: list[list[str]] = args.sample
    if args.write_calibration:
        if result_files or calibration or not sample_specs:
            parser.error("--write-calibration takes only --sample arguments")
    elif sample_specs:
        parser.error("--sample is only for --write-calibration")
    elif calibration:
        if not args.derive_threshold or result_files:
            parser.error("--calibration is only for --derive-threshold, without result files")
    elif len(result_files) < 2 or len(result_files) % 2:
        parser.error("result files must be BASE HEAD pairs")
    elif args.threshold_file and len(result_files) != 2:
        parser.error("--threshold-file accepts exactly one BASE HEAD pair")

    try:
        if args.write_calibration:
            samples = _calibration_samples(parser, sample_specs)
            text = json.dumps({"samples": samples}, indent=2, sort_keys=True) + "\n"
            args.write_calibration.write_text(text)
            return 0
        if args.derive_threshold:
            if calibration:
                samples = _read_calibration(calibration)
            else:
                samples = [
                    _changes(base, head)
                    for base, head in zip(result_files[::2], result_files[1::2], strict=True)
                ]
            print(json.dumps(_derive_threshold(samples), indent=2, sort_keys=True))
            return 0
        paired = _paired(result_files[0], result_files[1])
        limits = _read_limits(args.threshold_file, list(paired))
    except _CannotCompareError as error:
        # One line, whatever a path or a scenario name holds.
        print(f"cannot compare: {error}".replace("\n", "\\n"), file=sys.stderr)
        return 2

    regressed = False
    for scenario, by_metric in paired.items():
        for metric, key in _METRICS:
            base_values, head_values = by_metric[metric]
            base, head = statistics.median(base_values), statistics.median(head_values)
            limit = limits[scenario][key]
            low, high = ratio_interval(base_values, head_values)
            verdict = "ok"
            if head > base * (1 + limit / 100) and low > 1:
                verdict = "REGRESSION"
                regressed = True
            print(
                f"{verdict:10} {scenario} {metric}: base {base:.3f} head {head:.3f} "
                + f"({_change_pct(base, head):+.1f}%, limit +{limit}%; "
                + f"95% CI {(low - 1) * 100:+.1f}% to {(high - 1) * 100:+.1f}%)"
            )
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main())
