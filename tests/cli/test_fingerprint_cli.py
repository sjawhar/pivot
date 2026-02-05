# pyright: reportUnusedFunction=false
"""Tests for pivot fingerprint CLI commands."""

import pathlib

from click.testing import CliRunner

from pivot import cli, project
from pivot.storage import state


def test_fingerprint_reset_clears_statedb_entries(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot fingerprint reset clears AST hash entries from StateDB."""
    with runner.isolated_filesystem(temp_dir=tmp_path) as isolated_dir:
        project._project_root_cache = None

        # Create .pivot in isolated dir
        pivot_dir = pathlib.Path(isolated_dir) / ".pivot"
        pivot_dir.mkdir()
        db_path = pivot_dir / "state.db"

        # Add entries to this new db
        with state.StateDB(db_path, readonly=False) as db:
            db.save_ast_hash_many(
                [
                    ("src/a.py", 1000, 100, 111, "func_a", "3.13", 1, "aaaa1111"),
                    ("src/b.py", 2000, 200, 222, "func_b", "3.13", 1, "bbbb2222"),
                ]
            )

        result = runner.invoke(cli.cli, ["fingerprint", "reset"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Cleared 2 cached fingerprint entries" in result.output

        # Verify entries are actually gone
        with state.StateDB(db_path, readonly=True) as db:
            # Entries should not be found
            assert db.get_ast_hash("src/a.py", 1000, 100, 111, "func_a", "3.13", 1) is None
            assert db.get_ast_hash("src/b.py", 2000, 200, 222, "func_b", "3.13", 1) is None


def test_fingerprint_reset_reports_zero_when_empty(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot fingerprint reset works when there are no entries to clear."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None

        # Create .pivot but don't add any entries
        pivot_dir = pathlib.Path(".pivot")
        pivot_dir.mkdir()
        db_path = pivot_dir / "state.db"

        # Initialize empty StateDB
        with state.StateDB(db_path, readonly=False):
            pass

        result = runner.invoke(cli.cli, ["fingerprint", "reset"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Cleared 0 cached fingerprint entries" in result.output


def test_fingerprint_help_shows_reset_command(runner: CliRunner) -> None:
    """pivot fingerprint --help shows the reset subcommand."""
    result = runner.invoke(cli.cli, ["fingerprint", "--help"])

    assert result.exit_code == 0
    assert "reset" in result.output
    assert "Reset cached function fingerprints" in result.output
