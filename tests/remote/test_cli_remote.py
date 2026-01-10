from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import click.testing
import pytest

from pivot import cli, project
from pivot.types import TransferSummary

if TYPE_CHECKING:
    import pytest_mock


@pytest.fixture
def runner() -> click.testing.CliRunner:
    """Create a CLI runner for testing."""
    return click.testing.CliRunner()


# =============================================================================
# Push Command Tests
# =============================================================================


def test_push_no_files_to_push(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Push exits early when no files to push."""
    from pivot import transfer

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Mock transfer functions
        mocker.patch.object(
            transfer, "get_default_cache_dir", return_value=tmp_path / ".pivot/cache"
        )
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mocker.MagicMock(), "test-remote"),
        )
        mocker.patch.object(transfer, "get_local_cache_hashes", return_value=set())

        result = runner.invoke(cli.cli, ["push"])

        assert result.exit_code == 0
        assert "No files to push" in result.output


def test_push_dry_run(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Push dry run shows what would be pushed."""
    from pivot import transfer

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        mocker.patch.object(
            transfer, "get_default_cache_dir", return_value=tmp_path / ".pivot/cache"
        )
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mocker.MagicMock(), "my-remote"),
        )
        mocker.patch.object(
            transfer, "get_local_cache_hashes", return_value={"abc123def456", "789xyz456def"}
        )

        result = runner.invoke(cli.cli, ["push", "--dry-run"])

        assert result.exit_code == 0
        assert "Would push 2 file(s) to 'my-remote'" in result.output


def test_push_success(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Push command transfers files and shows summary."""
    from pivot import state, transfer

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        cache_dir.mkdir(parents=True)
        project._project_root_cache = None

        mocker.patch.object(transfer, "get_default_cache_dir", return_value=cache_dir)
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mocker.MagicMock(), "origin"),
        )
        mocker.patch.object(transfer, "get_local_cache_hashes", return_value={"hash1", "hash2"})

        mock_push = mocker.patch.object(
            transfer,
            "push",
            return_value=TransferSummary(transferred=2, skipped=0, failed=0, errors=[]),
        )

        # Mock StateDB context manager
        mock_state_db = mocker.MagicMock()
        mocker.patch.object(state, "StateDB", return_value=mock_state_db)

        result = runner.invoke(cli.cli, ["push"])

        assert result.exit_code == 0
        assert "Pushed to 'origin': 2 transferred, 0 skipped, 0 failed" in result.output
        mock_push.assert_called_once()


def test_push_with_errors(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Push command shows errors when transfers fail."""
    from pivot import state, transfer

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        cache_dir.mkdir(parents=True)
        project._project_root_cache = None

        mocker.patch.object(transfer, "get_default_cache_dir", return_value=cache_dir)
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mocker.MagicMock(), "origin"),
        )
        mocker.patch.object(transfer, "get_local_cache_hashes", return_value={"hash1", "hash2"})
        mocker.patch.object(
            transfer,
            "push",
            return_value=TransferSummary(
                transferred=1, skipped=0, failed=1, errors=["Upload failed: hash2"]
            ),
        )

        mock_state_db = mocker.MagicMock()
        mocker.patch.object(state, "StateDB", return_value=mock_state_db)

        result = runner.invoke(cli.cli, ["push"])

        assert result.exit_code == 0
        assert "1 transferred" in result.output
        assert "1 failed" in result.output
        assert "Error: Upload failed: hash2" in result.output


def test_push_with_targets(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Push with targets filters to those targets."""
    from pivot import state, transfer

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        cache_dir.mkdir(parents=True)
        project._project_root_cache = None

        mocker.patch.object(transfer, "get_default_cache_dir", return_value=cache_dir)
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mocker.MagicMock(), "origin"),
        )
        mock_target_hashes = mocker.patch.object(
            transfer, "get_target_hashes", return_value={"target_hash_1"}
        )
        mocker.patch.object(
            transfer,
            "push",
            return_value=TransferSummary(transferred=1, skipped=0, failed=0, errors=[]),
        )

        mock_state_db = mocker.MagicMock()
        mocker.patch.object(state, "StateDB", return_value=mock_state_db)

        result = runner.invoke(cli.cli, ["push", "train_model"])

        assert result.exit_code == 0
        mock_target_hashes.assert_called_once_with(["train_model"], cache_dir, include_deps=False)


# =============================================================================
# Pull Command Tests
# =============================================================================


def test_pull_dry_run_with_targets(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Pull dry run shows what would be pulled for targets."""
    from pivot import transfer

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        mocker.patch.object(
            transfer, "get_default_cache_dir", return_value=tmp_path / ".pivot/cache"
        )
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mocker.MagicMock(), "my-remote"),
        )
        mocker.patch.object(transfer, "get_target_hashes", return_value={"hash1", "hash2", "hash3"})
        mocker.patch.object(transfer, "get_local_cache_hashes", return_value={"hash1"})

        result = runner.invoke(cli.cli, ["pull", "--dry-run", "train_model"])

        assert result.exit_code == 0
        assert "Would pull 2 file(s) from 'my-remote'" in result.output


def test_pull_dry_run_all(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Pull dry run without stages lists all remote files."""
    from pivot import transfer

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        mock_remote = mocker.MagicMock()

        async def mock_list_hashes() -> set[str]:
            return {"remote1", "remote2", "remote3"}

        mock_remote.list_hashes = mock_list_hashes

        mocker.patch.object(
            transfer, "get_default_cache_dir", return_value=tmp_path / ".pivot/cache"
        )
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mock_remote, "origin"),
        )
        mocker.patch.object(transfer, "get_local_cache_hashes", return_value={"remote1"})

        result = runner.invoke(cli.cli, ["pull", "--dry-run"])

        assert result.exit_code == 0
        assert "Would pull 2 file(s) from 'origin'" in result.output


def test_pull_success(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Pull command downloads files and shows summary."""
    from pivot import state, transfer

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        cache_dir.mkdir(parents=True)
        project._project_root_cache = None

        mocker.patch.object(transfer, "get_default_cache_dir", return_value=cache_dir)
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mocker.MagicMock(), "origin"),
        )

        mock_pull = mocker.patch.object(
            transfer,
            "pull",
            return_value=TransferSummary(transferred=3, skipped=1, failed=0, errors=[]),
        )

        mock_state_db = mocker.MagicMock()
        mocker.patch.object(state, "StateDB", return_value=mock_state_db)

        result = runner.invoke(cli.cli, ["pull"])

        assert result.exit_code == 0
        assert "Pulled from 'origin': 3 transferred, 1 skipped, 0 failed" in result.output
        mock_pull.assert_called_once()


def test_pull_with_errors(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Pull command shows errors when downloads fail."""
    from pivot import state, transfer

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        cache_dir.mkdir(parents=True)
        project._project_root_cache = None

        mocker.patch.object(transfer, "get_default_cache_dir", return_value=cache_dir)
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mocker.MagicMock(), "origin"),
        )
        mocker.patch.object(
            transfer,
            "pull",
            return_value=TransferSummary(
                transferred=2,
                skipped=0,
                failed=2,
                errors=[
                    "Download failed: hash1",
                    "Download failed: hash2",
                    "Error 3",
                    "Error 4",
                    "Error 5",
                    "Error 6",
                ],
            ),
        )

        mock_state_db = mocker.MagicMock()
        mocker.patch.object(state, "StateDB", return_value=mock_state_db)

        result = runner.invoke(cli.cli, ["pull"])

        assert result.exit_code == 0
        assert "2 transferred" in result.output
        assert "2 failed" in result.output
        assert "Download failed: hash1" in result.output
        assert "... and 1 more errors" in result.output


def test_pull_with_stages(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Pull with stage names downloads those stages."""
    from pivot import state, transfer

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        cache_dir.mkdir(parents=True)
        project._project_root_cache = None

        mocker.patch.object(transfer, "get_default_cache_dir", return_value=cache_dir)
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mocker.MagicMock(), "origin"),
        )
        mock_pull = mocker.patch.object(
            transfer,
            "pull",
            return_value=TransferSummary(transferred=1, skipped=0, failed=0, errors=[]),
        )

        mock_state_db = mocker.MagicMock()
        mocker.patch.object(state, "StateDB", return_value=mock_state_db)

        result = runner.invoke(cli.cli, ["pull", "train_model", "evaluate"])

        assert result.exit_code == 0
        call_args = mock_pull.call_args
        assert call_args.args[3] == "origin"
        assert call_args.args[4] == ["train_model", "evaluate"]


def test_push_exception_shows_click_error(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Push exception is wrapped in ClickException with user-friendly message."""
    from pivot import transfer

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        mocker.patch.object(
            transfer,
            "get_default_cache_dir",
            side_effect=RuntimeError("Test error"),
        )

        result = runner.invoke(cli.cli, ["push"])

        assert result.exit_code != 0
        assert "Test error" in result.output


def test_pull_exception_shows_click_error(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Pull exception is wrapped in ClickException with user-friendly message."""
    from pivot import transfer

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        mocker.patch.object(
            transfer,
            "get_default_cache_dir",
            side_effect=RuntimeError("Test error"),
        )

        result = runner.invoke(cli.cli, ["pull"])

        assert result.exit_code != 0
        assert "Test error" in result.output
