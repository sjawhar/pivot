from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from pivot import cli, project
from pivot import config as config_mod
from pivot.cli import checkout as checkout_mod
from pivot.remote import sync as transfer
from pivot.storage import state
from pivot.types import TransferSummary

if TYPE_CHECKING:
    import click.testing
    import pytest
    from pytest_mock import MockerFixture


# =============================================================================
# Push Command Tests
# =============================================================================


def test_push_no_files_to_push(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Push exits early when no files to push."""

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        monkeypatch.setattr(project, "_project_root_cache", None)

        # Mock transfer functions
        mocker.patch.object(config_mod, "get_cache_dir", return_value=tmp_path / ".pivot/cache")
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
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Push dry run shows what would be pushed."""

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(config_mod, "get_cache_dir", return_value=tmp_path / ".pivot/cache")
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
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Push command transfers files and shows summary."""

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(config_mod, "get_cache_dir", return_value=cache_dir)
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
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Push command shows errors when transfers fail."""

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(config_mod, "get_cache_dir", return_value=cache_dir)
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

        assert result.exit_code == 1, "Should exit non-zero when transfers fail"
        assert "1 transferred" in result.output
        assert "1 failed" in result.output
        assert "Error: Upload failed: hash2" in result.output


def test_push_with_targets(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Push with targets filters to those targets."""

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        state_dir = tmp_path / ".pivot"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(config_mod, "get_cache_dir", return_value=cache_dir)
        mocker.patch.object(config_mod, "get_state_dir", return_value=state_dir)
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
        mock_target_hashes.assert_called_once_with(
            ["train_model"], state_dir, include_deps=False, all_stages=None
        )


# =============================================================================
# Fetch Command Tests
# =============================================================================


def test_fetch_dry_run_with_targets(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Fetch dry run shows what would be fetched for targets."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(config_mod, "get_cache_dir", return_value=tmp_path / ".pivot/cache")
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mocker.MagicMock(), "my-remote"),
        )
        mocker.patch.object(transfer, "get_target_hashes", return_value={"hash1", "hash2", "hash3"})
        mocker.patch.object(transfer, "get_local_cache_hashes", return_value={"hash1"})

        result = runner.invoke(cli.cli, ["fetch", "--dry-run", "train_model"])

        assert result.exit_code == 0
        assert "Would fetch 2 file(s) from 'my-remote'" in result.output


def test_fetch_dry_run_all(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Fetch dry run without stages lists all remote files."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        monkeypatch.setattr(project, "_project_root_cache", None)

        mock_remote = mocker.MagicMock()

        async def mock_list_hashes() -> set[str]:
            return {"remote1", "remote2", "remote3"}

        mock_remote.list_hashes = mock_list_hashes

        mocker.patch.object(config_mod, "get_cache_dir", return_value=tmp_path / ".pivot/cache")
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mock_remote, "origin"),
        )
        mocker.patch.object(transfer, "get_local_cache_hashes", return_value={"remote1"})

        result = runner.invoke(cli.cli, ["fetch", "--dry-run"])

        assert result.exit_code == 0
        assert "Would fetch 2 file(s) from 'origin'" in result.output


def test_fetch_success(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Fetch command downloads files to cache and shows summary."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(config_mod, "get_cache_dir", return_value=cache_dir)
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

        result = runner.invoke(cli.cli, ["fetch"])

        assert result.exit_code == 0
        assert "Fetched from 'origin': 3 transferred, 1 skipped, 0 failed" in result.output
        mock_pull.assert_called_once()


def test_fetch_with_errors(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Fetch command shows errors when downloads fail."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(config_mod, "get_cache_dir", return_value=cache_dir)
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
                errors=["Download failed: hash1", "Download failed: hash2"],
            ),
        )

        mock_state_db = mocker.MagicMock()
        mocker.patch.object(state, "StateDB", return_value=mock_state_db)

        result = runner.invoke(cli.cli, ["fetch"])

        assert result.exit_code == 1, "Should exit non-zero when transfers fail"
        assert "2 transferred" in result.output
        assert "2 failed" in result.output
        assert "Download failed: hash1" in result.output


def test_fetch_exception_shows_click_error(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Fetch exception is wrapped in ClickException with user-friendly message."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(
            config_mod,
            "get_cache_dir",
            side_effect=RuntimeError("Test error"),
        )

        result = runner.invoke(cli.cli, ["fetch"])

        assert result.exit_code != 0
        assert "Test error" in result.output


# =============================================================================
# Pull Command Tests
# =============================================================================


def test_pull_dry_run_with_targets(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Pull dry run shows what would be pulled for targets."""

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(config_mod, "get_cache_dir", return_value=tmp_path / ".pivot/cache")
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
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Pull dry run without stages lists all remote files."""

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        monkeypatch.setattr(project, "_project_root_cache", None)

        mock_remote = mocker.MagicMock()

        async def mock_list_hashes() -> set[str]:
            return {"remote1", "remote2", "remote3"}

        mock_remote.list_hashes = mock_list_hashes

        mocker.patch.object(config_mod, "get_cache_dir", return_value=tmp_path / ".pivot/cache")
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mock_remote, "origin"),
        )
        mocker.patch.object(transfer, "get_local_cache_hashes", return_value={"remote1"})

        result = runner.invoke(cli.cli, ["pull", "--dry-run", "--all"])

        assert result.exit_code == 0
        assert "Would pull 2 file(s) from 'origin'" in result.output


def test_pull_success(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Pull command downloads files and shows summary."""

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(config_mod, "get_cache_dir", return_value=cache_dir)
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

        # Mock checkout to prevent actual file restoration
        mocker.patch.object(checkout_mod, "checkout")

        result = runner.invoke(cli.cli, ["pull", "output.csv"])

        assert result.exit_code == 0
        assert "Fetched from 'origin': 3 transferred, 1 skipped, 0 failed" in result.output
        mock_pull.assert_called_once()


def test_pull_with_errors(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Pull command shows errors when downloads fail."""

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(config_mod, "get_cache_dir", return_value=cache_dir)
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

        result = runner.invoke(cli.cli, ["pull", "output.csv"])

        assert result.exit_code == 1, "Should exit non-zero when transfers fail"
        assert "2 transferred" in result.output
        assert "2 failed" in result.output
        assert "Download failed: hash1" in result.output
        assert "... and 1 more errors" in result.output


def test_pull_with_stages(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Pull with stage names downloads those stages."""

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        state_dir = tmp_path / ".pivot"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(config_mod, "get_cache_dir", return_value=cache_dir)
        mocker.patch.object(config_mod, "get_state_dir", return_value=state_dir)
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

        # Mock checkout to prevent actual file restoration
        mocker.patch.object(checkout_mod, "checkout")

        result = runner.invoke(cli.cli, ["pull", "train_model", "evaluate"])

        assert result.exit_code == 0
        call_args = mock_pull.call_args
        # Function signature: pull(cache_dir, state_dir, remote, state_db, remote_name, targets, ...)
        assert call_args.args[4] == "origin"
        assert call_args.args[5] == ["train_model", "evaluate"]


def test_push_exception_shows_click_error(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Push exception is wrapped in ClickException with user-friendly message."""

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(
            config_mod,
            "get_cache_dir",
            side_effect=RuntimeError("Test error"),
        )

        result = runner.invoke(cli.cli, ["push"])

        assert result.exit_code != 0
        assert "Test error" in result.output


def test_pull_exception_shows_click_error(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Pull exception is wrapped in ClickException with user-friendly message."""

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(
            config_mod,
            "get_cache_dir",
            side_effect=RuntimeError("Test error"),
        )

        result = runner.invoke(cli.cli, ["pull", "output.csv"])

        assert result.exit_code != 0
        assert "Test error" in result.output


# =============================================================================
# Remote List Command Tests
# =============================================================================


def test_remote_list_empty(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """remote list shows no remotes when none configured."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        monkeypatch.setattr(project, "_project_root_cache", None)

        result = runner.invoke(cli.cli, ["remote", "list"])

        assert result.exit_code == 0
        assert "No remotes configured" in result.output


def test_remote_list_shows_remotes(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """remote list shows all configured remotes."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        monkeypatch.setattr(project, "_project_root_cache", None)

        # Use config set to add remotes (new CLI approach)
        runner.invoke(cli.cli, ["config", "set", "remotes.origin", "s3://bucket1/cache"])
        runner.invoke(cli.cli, ["config", "set", "remotes.backup", "s3://bucket2/backup"])

        result = runner.invoke(cli.cli, ["remote", "list"])

        assert result.exit_code == 0
        assert "origin" in result.output
        assert "s3://bucket1/cache" in result.output
        assert "backup" in result.output
        assert "s3://bucket2/backup" in result.output


def test_remote_list_shows_default(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """remote list marks default remote."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        monkeypatch.setattr(project, "_project_root_cache", None)

        # Use config set to add remote and set default
        runner.invoke(cli.cli, ["config", "set", "remotes.origin", "s3://bucket/cache"])
        runner.invoke(cli.cli, ["config", "set", "default_remote", "origin"])

        result = runner.invoke(cli.cli, ["remote", "list"])

        assert result.exit_code == 0
        assert "(default)" in result.output


# =============================================================================
# Remote Commands - Quiet Mode Tests
# =============================================================================


def test_push_quiet_mode_no_output(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Push with --quiet produces no output."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(config_mod, "get_cache_dir", return_value=cache_dir)
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mocker.MagicMock(), "origin"),
        )
        mocker.patch.object(transfer, "get_local_cache_hashes", return_value={"hash1"})
        mocker.patch.object(
            transfer,
            "push",
            return_value=TransferSummary(transferred=1, skipped=0, failed=0, errors=[]),
        )

        mock_state_db = mocker.MagicMock()
        mocker.patch.object(state, "StateDB", return_value=mock_state_db)

        result = runner.invoke(cli.cli, ["--quiet", "push"])

        assert result.exit_code == 0
        assert result.output.strip() == "", "Quiet mode should suppress output"


def test_fetch_quiet_mode_no_output(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Fetch with --quiet produces no output."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(config_mod, "get_cache_dir", return_value=cache_dir)
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mocker.MagicMock(), "origin"),
        )
        mocker.patch.object(
            transfer,
            "pull",
            return_value=TransferSummary(transferred=1, skipped=0, failed=0, errors=[]),
        )

        mock_state_db = mocker.MagicMock()
        mocker.patch.object(state, "StateDB", return_value=mock_state_db)

        result = runner.invoke(cli.cli, ["--quiet", "fetch"])

        assert result.exit_code == 0
        assert result.output.strip() == "", "Quiet mode should suppress output"


# =============================================================================
# Pull Command - No Pipeline Tests
# =============================================================================


def test_pull_no_pipeline_no_targets_fails_early(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Pull fails early when no pipeline and no targets (and not using --all).

    Without a pipeline, targets, or --all flag, pull should fail immediately
    with a clear error message suggesting --all or targets.
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        monkeypatch.setattr(project, "_project_root_cache", None)

        # Mock remote and transfer functions
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mocker.MagicMock(), "origin"),
        )
        # Mock pull to ensure it's not called (early failure should prevent it)
        mock_pull = mocker.patch.object(
            transfer,
            "pull",
            return_value=TransferSummary(transferred=0, skipped=0, failed=0, errors=[]),
        )

        # Run pull with no pipeline, no targets, no --all
        result = runner.invoke(cli.cli, ["pull"])

        # Should fail early
        assert result.exit_code != 0, f"Expected failure, got: {result.output}"
        # Error message should suggest --all or targets
        assert "--all" in result.output or "target" in result.output.lower()
        # Should not have called pull (early failure prevents fetch)
        mock_pull.assert_not_called()


def test_pull_defaults_to_only_missing_for_checkout(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Pull defaults to only_missing=True when neither --force nor --only-missing passed.

    When pull invokes checkout without explicit --force or --only-missing flags,
    it should default to only_missing=True for safety.
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(config_mod, "get_cache_dir", return_value=cache_dir)
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mocker.MagicMock(), "origin"),
        )

        mocker.patch.object(
            transfer,
            "pull",
            return_value=TransferSummary(transferred=1, skipped=0, failed=0, errors=[]),
        )

        mock_state_db = mocker.MagicMock()
        mocker.patch.object(state, "StateDB", return_value=mock_state_db)

        # Mock checkout to capture the call
        mock_checkout = mocker.patch.object(checkout_mod, "checkout")

        result = runner.invoke(cli.cli, ["pull", "output.csv"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        # Verify checkout was called with only_missing=True
        mock_checkout.assert_called_once()
        call_kwargs = mock_checkout.call_args.kwargs
        assert call_kwargs.get("only_missing") is True, (
            f"Expected only_missing=True, got {call_kwargs}"
        )


def test_pull_force_overrides_default_only_missing(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Pull with --force overrides default only_missing, sets force=True.

    When pull is invoked with --force, it should pass force=True and
    only_missing=False to checkout.
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(config_mod, "get_cache_dir", return_value=cache_dir)
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mocker.MagicMock(), "origin"),
        )

        mocker.patch.object(
            transfer,
            "pull",
            return_value=TransferSummary(transferred=1, skipped=0, failed=0, errors=[]),
        )

        mock_state_db = mocker.MagicMock()
        mocker.patch.object(state, "StateDB", return_value=mock_state_db)

        # Mock checkout to capture the call
        mock_checkout = mocker.patch.object(checkout_mod, "checkout")

        result = runner.invoke(cli.cli, ["pull", "--force", "output.csv"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        # Verify checkout was called with force=True and only_missing=False
        mock_checkout.assert_called_once()
        call_kwargs = mock_checkout.call_args.kwargs
        assert call_kwargs.get("force") is True, f"Expected force=True, got {call_kwargs}"
        assert call_kwargs.get("only_missing") is False, (
            f"Expected only_missing=False with --force, got {call_kwargs}"
        )
