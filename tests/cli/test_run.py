from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

from helpers import register_test_stage
from pivot import cli, executor, loaders, outputs
from pivot.storage import cache, track

if TYPE_CHECKING:
    import click.testing


# =============================================================================
# Module-level TypedDicts and Stage Functions for annotation-based registration
# =============================================================================


class _OutputTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


def _helper_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    pathlib.Path("output.txt").write_text("done")
    return _OutputTxtOutputs(output=pathlib.Path("output.txt"))


# =============================================================================
# run --dry-run --allow-missing Tests
# =============================================================================


def test_run_dry_run_allow_missing_uses_pvt_hash(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """run --dry-run --allow-missing uses .pvt hash when dep file is missing."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # Create and run
        pathlib.Path("input.txt").write_text("data")
        register_test_stage(_helper_process, name="process")
        executor.run()

        # Track input
        input_hash = cache.hash_file(pathlib.Path("input.txt"))
        pvt_data = track.PvtData(path="input.txt", hash=input_hash, size=4)
        track.write_pvt_file(pathlib.Path("input.txt.pvt"), pvt_data)

        # Delete input (simulating CI)
        pathlib.Path("input.txt").unlink()

        result = runner.invoke(cli.cli, ["run", "--dry-run", "--allow-missing"])

        # Should show "would skip" not "Missing deps"
        assert "Missing deps" not in result.output, f"Got: {result.output}"
        assert "would skip" in result.output.lower(), f"Got: {result.output}"


def test_run_dry_run_explain_allow_missing_uses_pvt_hash(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """run --dry-run --explain --allow-missing uses .pvt hash when dep file is missing."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # Create and run
        pathlib.Path("input.txt").write_text("data")
        register_test_stage(_helper_process, name="process")
        executor.run()

        # Track input
        input_hash = cache.hash_file(pathlib.Path("input.txt"))
        pvt_data = track.PvtData(path="input.txt", hash=input_hash, size=4)
        track.write_pvt_file(pathlib.Path("input.txt.pvt"), pvt_data)

        # Delete input (simulating CI)
        pathlib.Path("input.txt").unlink()

        result = runner.invoke(cli.cli, ["run", "--dry-run", "--explain", "--allow-missing"])

        # Should NOT show error about missing deps
        assert "Missing deps" not in result.output, f"Got: {result.output}"
        assert result.exit_code == 0, f"Expected success, got: {result.output}"


def test_run_allow_missing_requires_dry_run(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """run --allow-missing without --dry-run errors."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")
        register_test_stage(_helper_process, name="process")

        result = runner.invoke(cli.cli, ["run", "--allow-missing"])

        assert result.exit_code != 0
        assert "--allow-missing" in result.output
        assert "--dry-run" in result.output


# =============================================================================
# --tui and --json Flag Tests
# =============================================================================


def test_run_tui_and_json_mutually_exclusive(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--tui and --json are mutually exclusive."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run", "--tui", "--json"])

        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()


def test_run_tui_log_requires_tui(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """--tui-log requires --tui flag."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_process, name="process")
        pathlib.Path("input.txt").write_text("data")

        result = runner.invoke(cli.cli, ["run", "--tui-log", "log.jsonl"])

        assert result.exit_code != 0
        assert "--tui-log requires --tui" in result.output


def test_run_serve_requires_tui(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """--serve requires --tui flag in watch mode."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_process, name="process")
        pathlib.Path("input.txt").write_text("data")

        result = runner.invoke(cli.cli, ["run", "--watch", "--serve"])

        assert result.exit_code != 0
        assert "--serve requires --tui" in result.output


def test_run_help_includes_tui_flag(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--tui flag appears in help text."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run", "--help"])

        assert result.exit_code == 0
        assert "--tui" in result.output
        # Help text is case-sensitive for "TUI"
        assert "TUI display" in result.output


def test_run_json_flag_accepted(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """--json flag should work without --tui (plain is now default)."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")
        register_test_stage(_helper_process, name="process")

        result = runner.invoke(cli.cli, ["run", "--json"])

        assert result.exit_code == 0
        # JSONL output should start with schema version
        assert '"type": "schema_version"' in result.output


def test_run_uses_plain_mode_by_default(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Plain text output is the default (no --tui flag needed)."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")
        register_test_stage(_helper_process, name="process")

        # Run without any display flags
        result = runner.invoke(cli.cli, ["run"])

        assert result.exit_code == 0
        # Plain mode shows stage status in simple text format
        # (not JSONL format which has "type":)
        assert '"type":' not in result.output
        # Should contain stage name in output
        assert "process" in result.output.lower()


def test_run_serve_requires_watch(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """--serve requires --watch mode."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")
        register_test_stage(_helper_process, name="process")

        result = runner.invoke(cli.cli, ["run", "--serve"])

        assert result.exit_code != 0
        assert "--serve requires --watch" in result.output


def test_run_tui_with_tui_log_validation_passes(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--tui --tui-log passes validation (log path is writable)."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")
        register_test_stage(_helper_process, name="process")

        log_path = tmp_path / "tui.jsonl"

        # This will attempt to run TUI which fails in non-TTY test environment,
        # but it should NOT fail on validation errors about --tui-log
        result = runner.invoke(cli.cli, ["run", "--tui", "--tui-log", str(log_path)])

        # Should NOT have validation errors
        assert "--tui-log requires --tui" not in result.output
        assert "Cannot write to" not in result.output
        # The log file should have been created during validation (touch())
        assert log_path.exists()
