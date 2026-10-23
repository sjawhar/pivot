"""A calibration file that derives no limits cannot be compared, whatever shape gives it no limits.

The ruled contract: an empty `samples` list is unreadable and exits 2, because it derives no
limits (compare.py's docstring: "an empty one derives no limits, and exits 2"). A list of samples
whose `changes` are all empty derives no limits either, so it is refused the same way.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import compare

if TYPE_CHECKING:
    import pathlib

_NO_CHANGES = dict[str, dict[str, float]]()


@pytest.mark.parametrize(
    "samples",
    [
        pytest.param([{"changes": _NO_CHANGES}], id="one-sample-no-changes"),
        pytest.param(
            [{"sample": 1, "run": 1, "attempt": 1, "job": 1, "changes": _NO_CHANGES}] * 2,
            id="every-sample-no-changes",
        ),
    ],
)
def test_calibration_that_derives_no_limits_cannot_be_compared(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], samples: list[object]
) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"samples": samples}))

    exit_code = compare.main(["--derive-threshold", "--calibration", str(calibration)])

    out, err = capsys.readouterr()
    assert exit_code == 2, f"a calibration that derives no limits must exit 2, not {exit_code}"
    assert out == "", f"no threshold file may be printed, got {out!r}"
    assert err.startswith("cannot compare:") and err.count("\n") == 1, err
