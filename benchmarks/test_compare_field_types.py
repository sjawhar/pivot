"""A result field of the wrong JSON type is input compare.py cannot compare: exit 2, one line.

run.py writes `rep`, `exit_code` and `cpu_count` as integers and `wall_s` and `cpu_s` as numbers.
compare.py's contract exits 2 when a file cannot be parsed, a command failed, or a side's reps are
not 0 to n-1 once each, and it already refuses a boolean limit. Converting the fields with int()
and float() instead truncates a fraction (rep 1.9 pairs as rep 1, exit code 0.9 counts as
success), reads the strings "1" and "2.5" as numbers, and reads true and false as 1 and 0, so the
row is compared as if it were valid.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import compare

if TYPE_CHECKING:
    import pathlib

_SCENARIO = "chain-10/pipeline/cold"


def _helper_rows() -> list[dict[str, object]]:
    return [
        {
            "scenario": _SCENARIO,
            "variant": "v",
            "rep": rep,
            "wall_s": 1.0,
            "cpu_s": 2.0,
            "exit_code": 0,
            "load1": 0.5,
            "cpu_count": 4,
            "ran": 10,
            "cached": 0,
            "leftover_procs": 0,
        }
        for rep in range(6)
    ]


@pytest.mark.parametrize(
    ("field", "value", "index"),
    [
        pytest.param("rep", 1.9, 1, id="fractional-rep"),
        pytest.param("rep", "1", 1, id="string-rep"),
        pytest.param("rep", True, 1, id="boolean-rep"),
        pytest.param("exit_code", 0.9, 0, id="fractional-exit-code"),
        pytest.param("exit_code", False, 0, id="boolean-exit-code"),
        pytest.param("cpu_count", 4.5, None, id="fractional-cpu-count"),
        pytest.param("wall_s", "2.5", 0, id="string-wall"),
        pytest.param("cpu_s", True, 0, id="boolean-cpu"),
    ],
)
def test_result_field_of_the_wrong_type_cannot_be_compared(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: object,
    index: int | None,
) -> None:
    head = _helper_rows()
    for number, row in enumerate(head):
        if index is None or number == index:
            row[field] = value
    (tmp_path / "base.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _helper_rows()))
    (tmp_path / "head.jsonl").write_text("".join(json.dumps(r) + "\n" for r in head))
    (tmp_path / "threshold.json").write_text(
        json.dumps({_SCENARIO: {"wall_pct": 10, "cpu_pct": 10}})
    )
    argv = [str(tmp_path / "base.jsonl"), str(tmp_path / "head.jsonl")]

    exit_code = compare.main([*argv, "--threshold-file", str(tmp_path / "threshold.json")])

    err = capsys.readouterr().err
    assert exit_code == 2, f"{field} {value!r} must not be read as a valid value"
    assert err.startswith("cannot compare: ") and err.count("\n") == 1, err
