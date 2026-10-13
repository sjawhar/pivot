"""Human-readable stage-output display includes both identity and Store path."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from pivot import cli, compose

if TYPE_CHECKING:
    import click.testing


class _TrainOutputs(TypedDict):
    model: str


@compose.stage
def _display_train() -> _TrainOutputs:
    return {"model": "model"}


def test_list_deps_displays_identity_and_store_path(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    with mock_discovery:
        _display_train()

    result = runner.invoke(cli.cli, ["list", "--deps"])

    assert result.exit_code == 0, result.output
    assert "test/_display_train:model → data/test/_display_train/model.txt" in result.output
