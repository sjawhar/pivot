from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from conftest import isolated_pivot_dir
from pivot import cli

if TYPE_CHECKING:
    from click.testing import CliRunner


_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, incremental, stage


@stage
def update(row: str, previous: dict[str, int]) -> Annotated[dict[str, int], incremental]:
    values = dict(previous)
    values[row.strip()] = len(values) + 1
    return values


with Pipeline("incremental") as pipeline:
    update(pipeline.input("row.txt"), previous=pipeline.previous())
"""

_CSV_PIPELINE = """\
from typing import Annotated

import pandas as pd

from pivot import Pipeline, incremental, loaders, stage


@stage
def keep(previous: pd.DataFrame) -> Annotated[pd.DataFrame, incremental, loaders.CSV()]:
    return previous


with Pipeline("incremental") as pipeline:
    keep(previous=pipeline.previous())
"""


def _write_pipeline_and_input() -> None:
    pathlib.Path("pipeline.py").write_text(_PIPELINE)
    raw_dir = pathlib.Path("data", "raw")
    raw_dir.mkdir(parents=True)
    (raw_dir / "row.txt").write_text("first\n")


def _make_workspace_output_uncached() -> None:
    stage_lock = pathlib.Path(".pivot", "stages", "incremental", "update.lock")
    stage_lock.unlink()


def test_uncached_incremental_flags_are_orthogonal_through_cli(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """Force does not bypass the guard; the allow flag does with or without force."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_pipeline_and_input()
        first = runner.invoke(cli.cli, ["repro"])
        assert first.exit_code == 0, first.output

        _make_workspace_output_uncached()
        forced = runner.invoke(cli.cli, ["repro", "--force"])
        assert forced.exit_code != 0
        assert "not in cache" in forced.output

        allowed = runner.invoke(cli.cli, ["repro", "--allow-uncached-incremental"])
        assert allowed.exit_code == 0, allowed.output

        _make_workspace_output_uncached()
        forced_allowed = runner.invoke(
            cli.cli, ["repro", "--force", "--allow-uncached-incremental"]
        )
        assert forced_allowed.exit_code == 0, forced_allowed.output


def test_incremental_csv_empty_first_value_loads_on_next_run(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A CSV incremental stage can pass its first empty value through twice."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path("pipeline.py").write_text(_CSV_PIPELINE)

        first = runner.invoke(cli.cli, ["repro"])
        assert first.exit_code == 0, first.output

        second = runner.invoke(cli.cli, ["repro", "--force"])
        assert second.exit_code == 0, second.output
        assert "1 ran" in second.output
