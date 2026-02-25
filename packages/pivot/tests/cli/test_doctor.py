from __future__ import annotations

from typing import TYPE_CHECKING

from pivot.cli import doctor

if TYPE_CHECKING:
    import pathlib


def test_check_pipeline_config_ignores_pivot_yaml(tmp_path: pathlib.Path) -> None:
    (tmp_path / "pivot.yaml").write_text("stages:\n  train:\n    python: train.py\n")

    event = doctor._check_pipeline_config(tmp_path)

    assert event["status"] == doctor.CheckStatus.WARN
    assert event["value"] == "not found"
    assert event["details"] == {"searched": ["pipeline.py"]}


def test_check_pipeline_config_accepts_pipeline_py(tmp_path: pathlib.Path) -> None:
    (tmp_path / "pipeline.py").write_text("pipeline = None\n")

    event = doctor._check_pipeline_config(tmp_path)

    assert event["status"] == doctor.CheckStatus.OK
    assert event["value"] == "pipeline.py"
