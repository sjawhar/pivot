"""A result file compare.py cannot parse is one it cannot compare: exit 2, never 1.

compare.py's contract: exit 1 means a scenario regressed, exit 2 means the files cannot be
compared. A truncated last line (a writer killed mid-line) is a file that cannot be compared.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import compare

if TYPE_CHECKING:
    import pathlib

    import pytest

_SCENARIO = "chain-10/pipeline/cold"


def _helper_rows() -> str:
    rows = [
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
    return "".join(json.dumps(row) + "\n" for row in rows)


def test_truncated_result_file_cannot_be_compared(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = _helper_rows()
    (tmp_path / "base.jsonl").write_text(rows)
    (tmp_path / "head.jsonl").write_text(rows[:-40])
    (tmp_path / "threshold.json").write_text(
        json.dumps({_SCENARIO: {"wall_pct": 10, "cpu_pct": 10}})
    )
    argv = [str(tmp_path / "base.jsonl"), str(tmp_path / "head.jsonl")]

    assert compare.main([*argv, "--threshold-file", str(tmp_path / "threshold.json")]) == 2
    assert f"cannot compare: {tmp_path / 'head.jsonl'}" in capsys.readouterr().err
