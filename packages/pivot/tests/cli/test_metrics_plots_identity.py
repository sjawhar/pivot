"""`pivot metrics` and `pivot plots` on a compositional pipeline with a lock committed at HEAD."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated

import matplotlib.figure

from pivot import cli, compose, executor

if TYPE_CHECKING:
    import pathlib

    import click.testing

    from conftest import GitRepo


@compose.stage
def _helper_score(text: str) -> Annotated[dict[str, int], compose.metric]:
    return {"length": len(text)}


@compose.stage
def _helper_chart(text: str) -> Annotated[matplotlib.figure.Figure, compose.plot]:
    fig = matplotlib.figure.Figure()
    fig.subplots().plot(range(len(text)))
    return fig


_METRIC_REL = "metrics/test/_helper_score.yaml"
_PLOT_REL = "plots/test/_helper_chart.png"


def _run_and_commit(pipeline: compose.Pipeline, git_repo: GitRepo, text: str) -> pathlib.Path:
    root, commit = git_repo
    raw = root / "data" / "raw" / "words.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(text)
    with pipeline:
        words = pipeline.input("words.txt", t=str)
        _helper_score(words)
        _helper_chart(words)
    results = executor.run(pipeline=pipeline)
    assert {r["status"] for r in results.values()} == {"ran"}, results
    commit("run")
    return raw


def _rerun_with(pipeline: compose.Pipeline, raw: pathlib.Path, text: str) -> None:
    raw.write_text(text)
    results = executor.run(pipeline=pipeline)
    assert {r["status"] for r in results.values()} == {"ran"}, results


def test_metrics_diff_without_targets_reports_changed_metric(
    runner: click.testing.CliRunner, git_repo: GitRepo, mock_discovery: compose.Pipeline
) -> None:
    raw = _run_and_commit(mock_discovery, git_repo, "abc")
    _rerun_with(mock_discovery, raw, "abcdef")

    result = runner.invoke(cli.cli, ["metrics", "diff", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == [
        {"path": _METRIC_REL, "key": "length", "old": 3, "new": 6, "change_type": "modified"}
    ]


def test_metrics_show_without_targets_labels_metric_by_workspace_path(
    runner: click.testing.CliRunner, git_repo: GitRepo, mock_discovery: compose.Pipeline
) -> None:
    _run_and_commit(mock_discovery, git_repo, "abc")

    result = runner.invoke(cli.cli, ["metrics", "show", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {_METRIC_REL: {"length": 3}}


def test_plots_diff_target_compares_against_head_hash(
    runner: click.testing.CliRunner, git_repo: GitRepo, mock_discovery: compose.Pipeline
) -> None:
    _run_and_commit(mock_discovery, git_repo, "abc")

    result = runner.invoke(cli.cli, ["plots", "diff", "--json", "test/_helper_chart"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []


def test_plots_diff_without_targets_reports_changed_plot(
    runner: click.testing.CliRunner, git_repo: GitRepo, mock_discovery: compose.Pipeline
) -> None:
    raw = _run_and_commit(mock_discovery, git_repo, "abc")
    _rerun_with(mock_discovery, raw, "abcdef")

    result = runner.invoke(cli.cli, ["plots", "diff", "--json"])

    assert result.exit_code == 0, result.output
    diffs = json.loads(result.output)
    assert [(d["path"], d["change_type"]) for d in diffs] == [(_PLOT_REL, "modified")]


def test_plots_show_without_targets_renders_plot(
    runner: click.testing.CliRunner, git_repo: GitRepo, mock_discovery: compose.Pipeline
) -> None:
    root, _ = git_repo
    _run_and_commit(mock_discovery, git_repo, "abc")

    result = runner.invoke(cli.cli, ["plots", "show", "-o", "out/index.html"])

    assert result.exit_code == 0, result.output
    html = (root / "out" / "index.html").read_text()
    assert "<p>1 plot(s)</p>" in html
    assert f"<h3>{_PLOT_REL}</h3>" in html
