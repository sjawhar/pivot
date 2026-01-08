from __future__ import annotations

from typing import TYPE_CHECKING

import click.testing

from pivot import cli, project

if TYPE_CHECKING:
    import pytest

    from conftest import GitRepo


# =============================================================================
# CLI Help Tests
# =============================================================================


def test_get_help() -> None:
    """Shows help message."""
    runner = click.testing.CliRunner()

    result = runner.invoke(cli.cli, ["get", "--help"])

    assert result.exit_code == 0
    assert "Retrieve files or stage outputs" in result.output
    assert "--rev" in result.output


# =============================================================================
# CLI Argument Validation Tests
# =============================================================================


def test_get_requires_rev(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Requires --rev option."""
    repo_path, commit = git_repo
    (repo_path / "file.txt").write_text("content")
    commit("initial")
    (repo_path / ".pivot").mkdir()

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    runner = click.testing.CliRunner()

    result = runner.invoke(cli.cli, ["get", "file.txt"])

    assert result.exit_code != 0
    assert "Missing option" in result.output or "--rev" in result.output


def test_get_requires_targets(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Requires at least one target."""
    repo_path, commit = git_repo
    (repo_path / "file.txt").write_text("content")
    commit("initial")
    (repo_path / ".pivot").mkdir()

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    runner = click.testing.CliRunner()

    result = runner.invoke(cli.cli, ["get", "--rev", "HEAD"])

    assert result.exit_code != 0
    assert "Missing argument" in result.output or "TARGETS" in result.output


# =============================================================================
# CLI Basic Functionality Tests
# =============================================================================


def test_get_git_tracked_file(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gets a git-tracked file from revision."""
    repo_path, commit = git_repo
    (repo_path / "file.txt").write_text("original content")
    sha = commit("initial")

    # Modify file
    (repo_path / "file.txt").write_text("modified")

    # Create .pivot directory
    (repo_path / ".pivot" / "cache").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    runner = click.testing.CliRunner()

    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "file.txt", "-o", str(repo_path / "restored.txt")],
    )

    assert result.exit_code == 0, result.output
    assert "Restored" in result.output
    assert (repo_path / "restored.txt").read_text() == "original content"


def test_get_git_tracked_file_with_force(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overwrites existing file with --force."""
    repo_path, commit = git_repo
    (repo_path / "file.txt").write_text("original")
    sha = commit("initial")

    # Create output file
    output_path = repo_path / "output.txt"
    output_path.write_text("existing")

    (repo_path / ".pivot" / "cache").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    runner = click.testing.CliRunner()

    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "file.txt", "-o", str(output_path), "--force"],
    )

    assert result.exit_code == 0, result.output
    assert "Restored" in result.output
    assert output_path.read_text() == "original"


def test_get_skip_existing_without_force(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skips existing files without --force."""
    repo_path, commit = git_repo
    (repo_path / "file.txt").write_text("original")
    sha = commit("initial")

    # Create output file
    output_path = repo_path / "output.txt"
    output_path.write_text("existing")

    (repo_path / ".pivot" / "cache").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    runner = click.testing.CliRunner()

    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "file.txt", "-o", str(output_path)],
    )

    assert result.exit_code == 0, result.output
    assert "Skipped" in result.output
    assert output_path.read_text() == "existing"


# =============================================================================
# CLI Error Handling Tests
# =============================================================================


def test_get_invalid_revision(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Errors on invalid revision."""
    repo_path, commit = git_repo
    (repo_path / "file.txt").write_text("content")
    commit("initial")
    (repo_path / ".pivot" / "cache").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    runner = click.testing.CliRunner()

    result = runner.invoke(
        cli.cli,
        ["get", "--rev", "invalid-revision", "file.txt"],
    )

    assert result.exit_code != 0
    assert "RevisionNotFoundError" in result.output or "Cannot resolve" in result.output


def test_get_target_not_found(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Errors when target not found at revision."""
    repo_path, commit = git_repo
    (repo_path / "file.txt").write_text("content")
    sha = commit("initial")
    (repo_path / ".pivot" / "cache").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    runner = click.testing.CliRunner()

    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "nonexistent.txt"],
    )

    assert result.exit_code != 0
    assert "TargetNotFoundError" in result.output or "not found" in result.output
