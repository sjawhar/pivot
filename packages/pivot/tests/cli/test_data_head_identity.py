"""`pivot diff` and `pivot get` on stage outputs whose lock is committed at HEAD.

The diff and get tests are lane 13b's restored tests from test_cli_data.py, carried here
without their xfail markers.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated

import pandas as pd

from pivot import cli, compose, executor, loaders

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

    import click.testing

    from conftest import GitRepo


@compose.stage
def make_csv(source: pd.DataFrame) -> Annotated[pd.DataFrame, loaders.CSV()]:
    return source


@compose.stage
def make_output() -> str:
    return "stage output content"


_CSV_REL = "data/test/make_csv.csv"


def _run_csv_stage(
    pipeline: compose.Pipeline, commit_fn: Callable[[str], str], initial: str
) -> pathlib.Path:
    raw = pipeline.root / "data" / "raw" / "source.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(initial)
    with pipeline:
        make_csv(pipeline.input("source.csv", t=pd.DataFrame))
    results = executor.run(pipeline=pipeline)
    assert results["test/make_csv"]["status"] == "ran", results
    output = pipeline.root / _CSV_REL
    assert output.read_text() == initial, "CSV round-trip should reproduce the input"
    commit_fn("Initial CSV output")
    return output


def _rewrite(path: pathlib.Path, content: str) -> None:
    path.unlink()
    path.write_text(content)


def test_diff_csv_file(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mock_discovery: compose.Pipeline,
) -> None:
    _, commit = git_repo
    csv_file = _run_csv_stage(mock_discovery, commit, "id,value\n1,10\n2,20\n")

    _rewrite(csv_file, "id,value\n1,10\n2,25\n3,30\n")

    result = runner.invoke(cli.cli, ["diff", "--no-tui", _CSV_REL])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert _CSV_REL in result.output
    assert "Rows:" in result.output
    assert "(added)" not in result.output, (
        "File existed at HEAD; it must be diffed as modified, not reported as added"
    )


def test_diff_json_output(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mock_discovery: compose.Pipeline,
) -> None:
    _, commit = git_repo
    csv_file = _run_csv_stage(mock_discovery, commit, "id,value\n1,10\n")

    _rewrite(csv_file, "id,value\n1,99\n")

    result = runner.invoke(cli.cli, ["diff", "--json", _CSV_REL])

    assert result.exit_code == 0, f"Failed: {result.output}"
    data: list[dict[str, object]] = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["path"] == _CSV_REL
    assert data[0]["old_rows"] == 1, "HEAD version (1 row) must be restored from cache"


def test_diff_no_changes_message(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mock_discovery: compose.Pipeline,
) -> None:
    _, commit = git_repo
    _run_csv_stage(mock_discovery, commit, "id,value\n1,10\n")

    result = runner.invoke(cli.cli, ["diff", "--no-tui", _CSV_REL])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "No data file changes" in result.output


def test_diff_json_empty_returns_valid_json(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mock_discovery: compose.Pipeline,
) -> None:
    _, commit = git_repo
    _run_csv_stage(mock_discovery, commit, "id,value\n1,10\n")

    result = runner.invoke(cli.cli, ["diff", "--json", _CSV_REL])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert json.loads(result.output) == []


def test_get_stage_output(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mock_discovery: compose.Pipeline,
) -> None:
    _, commit = git_repo
    with mock_discovery:
        make_output()
    results = executor.run(pipeline=mock_discovery)
    assert results["test/make_output"]["status"] == "ran", results
    output_file = mock_discovery.root / "data" / "test" / "make_output.txt"
    sha = commit("Stage output")

    output_file.unlink()

    result = runner.invoke(cli.cli, ["get", "--rev", sha[:7], "test/make_output"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Restored" in result.output
    assert output_file.read_text() == "stage output content"
