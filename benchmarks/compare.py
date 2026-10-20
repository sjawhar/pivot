"""Compare two benchmark result files from run.py, scenario by scenario.

Usage: compare.py BASE.jsonl HEAD.jsonl --threshold-file benchmarks/threshold.json
       compare.py A1_BASE.jsonl A1_HEAD.jsonl [A2_BASE.jsonl A2_HEAD.jsonl ...] --derive-threshold

The threshold file gives each scenario its own limits, {"<scenario>": {"wall_pct": N,
"cpu_pct": N}}, because scenarios differ in noise: a warm no-op and a cold build do not vary
alike.

With --threshold-file, exits 0 when no scenario regressed, 1 when the head's median wall time
or median CPU exceeds the base's by more than that scenario's `wall_pct` or `cpu_pct`, and 2
when the files cannot be compared: a command failed, a scenario is on one side only or has no
threshold, or the two ran on machines with different core counts.

With --derive-threshold, each file pair is an A/A run (the same build on both sides). It prints
a threshold file: for each scenario and metric, three times the largest absolute change between
any pair's medians, rounded up to a whole percent and at least 1.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

_METRICS = (("wall_s", "wall_pct"), ("cpu_s", "cpu_pct"))


class _CannotCompareError(Exception):
    pass


def _load(path: pathlib.Path) -> dict[str, list[dict[str, object]]]:
    by_scenario = dict[str, list[dict[str, object]]]()
    for line in path.read_text().splitlines():
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


def _derive_threshold(
    medians_by_pair: list[dict[str, dict[str, tuple[float, float]]]],
) -> dict[str, dict[str, int]]:
    scenarios = set(medians_by_pair[0])
    if any(set(medians) != scenarios for medians in medians_by_pair[1:]):
        raise _CannotCompareError("A/A pairs have different scenarios")
    return {
        scenario: {
            key: max(
                1,
                math.ceil(
                    round(
                        3
                        * max(
                            abs(_change_pct(*medians[scenario][metric]))
                            for medians in medians_by_pair
                        ),
                        6,
                    )
                ),
            )
            for metric, key in _METRICS
        }
        for scenario in sorted(scenarios)
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("result_files", nargs="+", type=pathlib.Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--threshold-file", type=pathlib.Path)
    mode.add_argument("--derive-threshold", action="store_true")
    args = parser.parse_args(argv)
    result_files: list[pathlib.Path] = args.result_files
    if len(result_files) < 2 or len(result_files) % 2:
        parser.error("result files must be BASE HEAD pairs")
    if args.threshold_file and len(result_files) != 2:
        parser.error("--threshold-file accepts exactly one BASE HEAD pair")

    try:
        medians_by_pair = [
            _medians(result_files[index], result_files[index + 1])
            for index in range(0, len(result_files), 2)
        ]
        if args.derive_threshold:
            print(json.dumps(_derive_threshold(medians_by_pair), indent=2, sort_keys=True))
            return 0
    except _CannotCompareError as error:
        print(f"cannot compare: {error}", file=sys.stderr)
        return 2

    medians = medians_by_pair[0]
    limits: dict[str, dict[str, float]] = json.loads(pathlib.Path(args.threshold_file).read_text())
    if missing := sorted(set(medians) - set(limits)):
        print(f"cannot compare: no threshold for {', '.join(missing)}", file=sys.stderr)
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
