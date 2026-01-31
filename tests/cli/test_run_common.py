# tests/cli/test_run_common.py
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from helpers import register_test_stage
from pivot import stage_def
from pivot.cli import _run_common

if TYPE_CHECKING:
    import click.testing


def test_validate_stages_exist_passes_for_registered_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """validate_stages_exist passes when stages are registered."""

    class Params(stage_def.StageParams):
        pass

    def _helper_noop(params: Params) -> None:
        pass

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_noop, name="my_stage")

        # Should not raise
        _run_common.validate_stages_exist(["my_stage"])
