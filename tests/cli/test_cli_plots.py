from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import click.testing
import pytest

from pivot import cli, lock, outputs, registry
from pivot.types import LockData

if TYPE_CHECKING:
    import pathlib


def _setup_git_repo(tmp_path: pathlib.Path) -> None:
    """Initialize a git repo with user config."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def runner() -> click.testing.CliRunner:
    """Create a CLI runner for testing."""
    return click.testing.CliRunner()


# =============================================================================
# plots group Tests
# =============================================================================


def test_plots_group_help(runner: click.testing.CliRunner) -> None:
    """plots group should show available commands."""
    result = runner.invoke(cli.cli, ["plots", "--help"])
    assert result.exit_code == 0
    assert "show" in result.output
    assert "diff" in result.output


def test_plots_in_main_help(runner: click.testing.CliRunner) -> None:
    """plots command should appear in main help."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "plots" in result.output


# =============================================================================
# plots show Tests
# =============================================================================


def test_plots_show_help(runner: click.testing.CliRunner) -> None:
    """plots show should show its help."""
    result = runner.invoke(cli.cli, ["plots", "show", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output
    assert "--open" in result.output


def test_plots_show_no_plots(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """plots show with no registered plots should report empty."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli.cli, ["plots", "show"])
        assert result.exit_code == 0
        assert "No plots found" in result.output


def test_plots_show_creates_html(
    runner: click.testing.CliRunner, set_project_root: pathlib.Path
) -> None:
    """plots show creates HTML output file."""
    plot_file = set_project_root / "plot.png"
    plot_file.write_bytes(b"fake png data")

    def _stage_with_plot() -> None:
        pass

    registry.REGISTRY.register(
        _stage_with_plot,
        name="test_stage",
        outs=[outputs.Plot(path=str(plot_file))],
    )

    with runner.isolated_filesystem(temp_dir=set_project_root):
        result = runner.invoke(cli.cli, ["plots", "show"])
        assert result.exit_code == 0
        assert "Rendered 1 plot(s)" in result.output


def test_plots_show_custom_output_path(
    runner: click.testing.CliRunner, set_project_root: pathlib.Path
) -> None:
    """plots show with custom output path."""
    plot_file = set_project_root / "plot.png"
    plot_file.write_bytes(b"fake png data")

    def _stage_with_plot() -> None:
        pass

    registry.REGISTRY.register(
        _stage_with_plot,
        name="test_stage",
        outs=[outputs.Plot(path=str(plot_file))],
    )

    with runner.isolated_filesystem(temp_dir=set_project_root):
        result = runner.invoke(cli.cli, ["plots", "show", "-o", "custom/output.html"])
        assert result.exit_code == 0
        assert "custom/output.html" in result.output


# =============================================================================
# plots diff Tests
# =============================================================================


def test_plots_diff_help(runner: click.testing.CliRunner) -> None:
    """plots diff should show its help."""
    result = runner.invoke(cli.cli, ["plots", "diff", "--help"])
    assert result.exit_code == 0
    assert "--json" in result.output
    assert "--md" in result.output
    assert "--no-path" in result.output


def test_plots_diff_no_plots(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """plots diff with no registered plots should report empty."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli.cli, ["plots", "diff"])
        assert result.exit_code == 0
        assert "No plots found" in result.output


def test_plots_diff_json_format(
    runner: click.testing.CliRunner, set_project_root: pathlib.Path
) -> None:
    """plots diff --json returns valid JSON."""
    _setup_git_repo(set_project_root)

    plot_file = set_project_root / "plot.png"
    plot_file.write_bytes(b"fake png data")

    def _stage_with_plot() -> None:
        pass

    registry.REGISTRY.register(
        _stage_with_plot,
        name="test_stage",
        outs=[outputs.Plot(path=str(plot_file))],
    )

    with runner.isolated_filesystem(temp_dir=set_project_root):
        # Create lock file using proper API
        cache_dir = set_project_root / ".pivot" / "cache"
        stage_lock = lock.StageLock("test_stage", cache_dir)
        stage_lock.write(
            LockData(
                code_manifest={},
                params={},
                dep_hashes={},
                output_hashes={str(plot_file): {"hash": "old_hash_value"}},
            )
        )

        # Commit lock file to git
        subprocess.run(["git", "add", "."], cwd=set_project_root, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=set_project_root,
            check=True,
            capture_output=True,
        )

        result = runner.invoke(cli.cli, ["plots", "diff", "--json"])
        assert result.exit_code == 0
        # Output should be valid JSON
        try:
            parsed = json.loads(result.output)
            assert isinstance(parsed, list)
        except json.JSONDecodeError:
            pytest.fail(f"Output is not valid JSON: {result.output}")


def test_plots_diff_md_format(
    runner: click.testing.CliRunner, set_project_root: pathlib.Path
) -> None:
    """plots diff --md returns markdown table."""
    _setup_git_repo(set_project_root)

    plot_file = set_project_root / "plot.png"
    plot_file.write_bytes(b"fake png data")

    def _stage_with_plot() -> None:
        pass

    registry.REGISTRY.register(
        _stage_with_plot,
        name="test_stage",
        outs=[outputs.Plot(path=str(plot_file))],
    )

    with runner.isolated_filesystem(temp_dir=set_project_root):
        cache_dir = set_project_root / ".pivot" / "cache"
        stage_lock = lock.StageLock("test_stage", cache_dir)
        stage_lock.write(
            LockData(
                code_manifest={},
                params={},
                dep_hashes={},
                output_hashes={str(plot_file): {"hash": "old_hash_value"}},
            )
        )

        # Commit lock file to git
        subprocess.run(["git", "add", "."], cwd=set_project_root, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=set_project_root,
            check=True,
            capture_output=True,
        )

        result = runner.invoke(cli.cli, ["plots", "diff", "--md"])
        assert result.exit_code == 0
        assert "|" in result.output  # Markdown table uses pipes


def test_plots_diff_no_path_flag(
    runner: click.testing.CliRunner, set_project_root: pathlib.Path
) -> None:
    """plots diff --no-path hides path column."""
    _setup_git_repo(set_project_root)

    plot_file = set_project_root / "plot.png"
    plot_file.write_bytes(b"fake png data")

    def _stage_with_plot() -> None:
        pass

    registry.REGISTRY.register(
        _stage_with_plot,
        name="test_stage",
        outs=[outputs.Plot(path=str(plot_file))],
    )

    with runner.isolated_filesystem(temp_dir=set_project_root):
        cache_dir = set_project_root / ".pivot" / "cache"
        stage_lock = lock.StageLock("test_stage", cache_dir)
        stage_lock.write(
            LockData(
                code_manifest={},
                params={},
                dep_hashes={},
                output_hashes={str(plot_file): {"hash": "old_hash_value"}},
            )
        )

        # Commit lock file to git
        subprocess.run(["git", "add", "."], cwd=set_project_root, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=set_project_root,
            check=True,
            capture_output=True,
        )

        result = runner.invoke(cli.cli, ["plots", "diff", "--no-path"])
        assert result.exit_code == 0
        assert "Path" not in result.output
