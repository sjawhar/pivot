from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from pivot import cli
from pivot.import_artifact import ImportResult

if TYPE_CHECKING:
    from click.testing import CliRunner
    from pytest_mock import MockerFixture


# =============================================================================
# Help
# =============================================================================


def test_cli_import_help_shows_usage(runner: CliRunner) -> None:
    """import command shows help with correct arguments and options."""
    result = runner.invoke(cli.cli, ["import", "--help"])
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "REPO_URL" in result.output
    assert "PATH" in result.output
    assert "--rev" in result.output
    assert "--out" in result.output
    assert "--force" in result.output
    assert "--no-download" in result.output


# =============================================================================
# Success paths
# =============================================================================


def test_cli_import_success_downloaded(runner: CliRunner, mocker: MockerFixture) -> None:
    """import command shows 'Imported' when artifact is downloaded."""
    mock_import = AsyncMock(
        return_value=ImportResult(
            pvt_path="data/model.pkl.pvt",
            data_path="data/model.pkl",
            downloaded=True,
        )
    )
    mocker.patch("pivot.import_artifact.import_artifact", mock_import)

    result = runner.invoke(cli.cli, ["import", "https://github.com/org/repo", "data/model.pkl"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Imported data/model.pkl" in result.output
    mock_import.assert_awaited_once_with(
        "https://github.com/org/repo",
        "data/model.pkl",
        rev="main",
        out=None,
        force=False,
        no_download=False,
    )


def test_cli_import_no_download(runner: CliRunner, mocker: MockerFixture) -> None:
    """import --no-download shows metadata-only message."""
    mock_import = AsyncMock(
        return_value=ImportResult(
            pvt_path="data/model.pkl.pvt",
            data_path="data/model.pkl",
            downloaded=False,
        )
    )
    mocker.patch("pivot.import_artifact.import_artifact", mock_import)

    result = runner.invoke(
        cli.cli,
        ["import", "https://github.com/org/repo", "data/model.pkl", "--no-download"],
    )

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "metadata only" in result.output
    assert "data/model.pkl.pvt" in result.output
    mock_import.assert_awaited_once_with(
        "https://github.com/org/repo",
        "data/model.pkl",
        rev="main",
        out=None,
        force=False,
        no_download=True,
    )


def test_cli_import_with_options(runner: CliRunner, mocker: MockerFixture) -> None:
    """import passes --rev, --out, --force correctly."""
    mock_import = AsyncMock(
        return_value=ImportResult(
            pvt_path="output.pkl.pvt",
            data_path="output.pkl",
            downloaded=True,
        )
    )
    mocker.patch("pivot.import_artifact.import_artifact", mock_import)

    result = runner.invoke(
        cli.cli,
        [
            "import",
            "https://github.com/org/repo",
            "data/model.pkl",
            "--rev",
            "v1.0",
            "--out",
            "output.pkl",
            "--force",
        ],
    )

    assert result.exit_code == 0, f"Failed: {result.output}"
    mock_import.assert_awaited_once_with(
        "https://github.com/org/repo",
        "data/model.pkl",
        rev="v1.0",
        out="output.pkl",
        force=True,
        no_download=False,
    )


# =============================================================================
# Error paths
# =============================================================================


def test_cli_import_missing_args(runner: CliRunner) -> None:
    """import with no arguments exits with error."""
    result = runner.invoke(cli.cli, ["import"])
    assert result.exit_code != 0, "Should fail without arguments"


def test_cli_import_missing_path_arg(runner: CliRunner) -> None:
    """import with only repo_url exits with error."""
    result = runner.invoke(cli.cli, ["import", "https://github.com/org/repo"])
    assert result.exit_code != 0, "Should fail without path argument"
