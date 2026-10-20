"""Compare benchmark result files from run.py, scenario by scenario, and derive thresholds.

Usage: compare.py BASE.jsonl HEAD.jsonl --threshold-file benchmarks/threshold.json
       compare.py A1_BASE.jsonl A1_HEAD.jsonl [A2_BASE.jsonl A2_HEAD.jsonl ...] --derive-threshold
       compare.py --write-calibration benchmarks/calibration.json
                  --sample SAMPLE RUN ATTEMPT JOB BASE.jsonl HEAD.jsonl [--sample ...]
       compare.py --derive-threshold --calibration benchmarks/calibration.json

The threshold file gives each scenario its own limits, {"<scenario>": {"wall_pct": N,
"cpu_pct": N}}, because scenarios differ in noise: a warm no-op and a cold build do not vary
alike.

With --threshold-file, exits 0 when no scenario regressed, 1 when the head's median wall time
or median CPU exceeds the base's by more than that scenario's `wall_pct` or `cpu_pct`, and 2
when the files cannot be compared: a file is missing, a command failed, a scenario is on one
side only or lacks a limit, or the two ran on machines with different core counts.

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
import statistics
import sys

_METRICS = (("wall_s", "wall_pct"), ("cpu_s", "cpu_pct"))
_FLOOR_PCT = 2

# scenario -> limit key ("wall_pct" or "cpu_pct") -> percent change from base to head median
type _Changes = dict[str, dict[str, float]]


class _CannotCompareError(Exception):
    pass


def _read_json(path: pathlib.Path) -> object:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise _CannotCompareError(f"{path}: no such file") from None


def _load(path: pathlib.Path) -> dict[str, list[dict[str, object]]]:
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        raise _CannotCompareError(f"{path}: no such file") from None
    by_scenario = dict[str, list[dict[str, object]]]()
    for line in lines:
        row = json.loads(line)
        if row["exit_code"] != 0:
            raise _CannotCompareError(
                f"{path}: {row['scenario']} rep {row['rep']} exited with code {row['exit_code']}"
            )
        by_scenario.setdefault(row["scenario"], []).append(row)
    return by_scenario


def _medians(
    base_path: pathlib.Path, head_path: pathlib.Path
) -> dict[str, dict[str, tuple[float, float]]]:
    """Return scenario -> metric -> (base median, head median)."""
    base = _load(base_path)
    head = _load(head_path)
    if one_side := sorted(set(base) ^ set(head)):
        raise _CannotCompareError(f"scenarios on one side only: {', '.join(one_side)}")
    cores = {
        int(str(row["cpu_count"])) for rows in (*base.values(), *head.values()) for row in rows
    }
    if len(cores) != 1:
        raise _CannotCompareError(f"results come from machines with core counts {sorted(cores)}")
    return {
        scenario: {
            metric: (
                statistics.median(float(str(row[metric])) for row in base[scenario]),
                statistics.median(float(str(row[metric])) for row in head[scenario]),
            )
            for metric, _ in _METRICS
        }
        for scenario in sorted(base)
    }


def _change_pct(base: float, head: float) -> float:
    return (head - base) / base * 100


def _changes(medians: dict[str, dict[str, tuple[float, float]]]) -> _Changes:
    return {
        scenario: {key: _change_pct(*by_metric[metric]) for metric, key in _METRICS}
        for scenario, by_metric in medians.items()
    }


def _derive_threshold(samples: list[_Changes]) -> dict[str, dict[str, int]]:
    largest = dict[str, dict[str, float]]()
    for changes in samples:
        for scenario, by_key in changes.items():
            seen = largest.setdefault(scenario, {key: 0.0 for _, key in _METRICS})
            for _, key in _METRICS:
                seen[key] = max(seen[key], abs(by_key[key]))
    return {
        scenario: {
            key: max(_FLOOR_PCT, math.ceil(round(3 * spread, 6))) for key, spread in by_key.items()
        }
        for scenario, by_key in sorted(largest.items())
    }


def _read_calibration(path: pathlib.Path) -> list[_Changes]:
    calibration: dict[str, list[dict[str, object]]] = _read_json(path)  # pyright: ignore[reportAssignmentType] - JSON file written by --write-calibration
    return [sample["changes"] for sample in calibration["samples"]]  # pyright: ignore[reportReturnType] - see above


def _read_limits(path: pathlib.Path, scenarios: list[str]) -> dict[str, dict[str, float]]:
    limits: dict[str, dict[str, float]] = _read_json(path)  # pyright: ignore[reportAssignmentType] - JSON threshold file
    missing = [
        f"{scenario} {key}"
        for scenario in scenarios
        for _, key in _METRICS
        if scenario not in limits or key not in limits[scenario]
    ]
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
        changes = _changes(_medians(pathlib.Path(base), pathlib.Path(head)))
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
            pathlib.Path(args.write_calibration).write_text(text)
            return 0
        if args.derive_threshold:
            if calibration:
                samples = _read_calibration(calibration)
            else:
                samples = [
                    _changes(_medians(result_files[index], result_files[index + 1]))
                    for index in range(0, len(result_files), 2)
                ]
            print(json.dumps(_derive_threshold(samples), indent=2, sort_keys=True))
            return 0
        medians = _medians(result_files[0], result_files[1])
        limits = _read_limits(pathlib.Path(args.threshold_file), list(medians))
    except _CannotCompareError as error:
        print(f"cannot compare: {error}", file=sys.stderr)
        return 2

    regressed = False
    for scenario, by_metric in medians.items():
        for metric, key in _METRICS:
            base, head = by_metric[metric]
            limit = limits[scenario][key]
            verdict = "ok"
            if head > base * (1 + limit / 100):
                verdict = "REGRESSION"
                regressed = True
            print(
                f"{verdict:10} {scenario} {metric}: base {base:.3f} head {head:.3f} "
                + f"({_change_pct(base, head):+.1f}%, limit +{limit}%)"
            )
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main())
