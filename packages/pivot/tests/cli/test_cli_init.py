import os
import pathlib
import stat

import click.testing
import pytest

from pivot import cli, ignore, project
from pivot import state_dir as state_dir_mod

# --- basic initialization tests ---


def test_init_creates_pivot_directory(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code == 0
        assert pathlib.Path(".pivot").is_dir()


def test_init_creates_gitignore(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code == 0
        assert pathlib.Path(".pivot/.gitignore").exists()


_PIVOT_052_GITIGNORE = b"""\
# Cache directory (stage outputs, file hashes)
/cache/

# State database (file hashes, generation counters)
/state.lmdb/

# Artifact and restore locks (flock files for concurrent execution)
/locks/

# Agent RPC socket (left behind by TUI and --serve sessions)
/agent.sock

# Config lock (held while editing config.yaml)
/config.yaml.lock

# Generated plots gallery (pivot plots show)
/plots/

# Key for secret params' digests in lock files; never commit it
/secret.key
"""


def test_default_init_writes_the_pivot_052_ignore_bytes(tmp_path: pathlib.Path) -> None:
    """Default init must preserve the released byte-for-byte ignore contract, and its mode."""
    runner = click.testing.CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        previous_umask = os.umask(0o002)
        try:
            result = runner.invoke(cli.cli, ["init"])
        finally:
            os.umask(previous_umask)

        assert result.exit_code == 0, result.output
        gitignore = pathlib.Path(".pivot/.gitignore")
        assert gitignore.read_bytes() == _PIVOT_052_GITIGNORE
        # 0.5.2 writes it with write_text, so the umask alone sets its mode.
        assert stat.S_IMODE(gitignore.stat().st_mode) == 0o664


def test_default_config_set_keeps_the_released_config_lock_file(tmp_path: pathlib.Path) -> None:
    """Default config writes preserve the files created by Pivot 0.5.2."""
    runner = click.testing.CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        initialized = runner.invoke(cli.cli, ["init"])
        before = {
            path.relative_to(".").as_posix()
            for path in pathlib.Path(".").rglob("*")
            if path.is_file()
        }

        result = runner.invoke(cli.cli, ["config", "set", "core.max_workers", "1"])

        after = {
            path.relative_to(".").as_posix()
            for path in pathlib.Path(".").rglob("*")
            if path.is_file()
        }
        assert result.exit_code == 0, initialized.output + result.output
        assert after - before == {".pivot/config.yaml", ".pivot/config.yaml.lock"}


def test_default_init_keeps_legacy_stages_directory(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """The default state directory keeps init's established layout and output."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code == 0, result.output
        assert pathlib.Path(".pivot/stages").is_dir()
        assert "  .pivot/stages/" in result.output


@pytest.mark.parametrize(
    "expected_content",
    [
        pytest.param("/cache/", id="cache_dir"),
        pytest.param("/state.lmdb/", id="state_lmdb"),
        pytest.param("/locks/", id="artifact_locks"),
        pytest.param("/agent.sock", id="agent_socket"),
        pytest.param("/plots/", id="plots_gallery"),
        pytest.param("/secret.key", id="secret_key"),
    ],
)
def test_init_gitignore_contains_expected_entries(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, expected_content: str
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code == 0
        content = pathlib.Path(".pivot/.gitignore").read_text()
        assert expected_content in content


# --- .pivotignore tests ---


def test_init_creates_pivotignore(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """init creates .pivotignore file in project root."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code == 0
        assert pathlib.Path(".pivotignore").is_file()


@pytest.mark.parametrize(
    "expected_pattern",
    [
        pytest.param("*.pyc", id="pyc_bytecode"),
        pytest.param("__pycache__/", id="pycache_dir"),
        pytest.param(".venv/", id="venv_dir"),
        pytest.param(".git/", id="git_dir"),
        pytest.param("node_modules/", id="node_modules"),
    ],
)
def test_init_pivotignore_contains_expected_patterns(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, expected_pattern: str
) -> None:
    """init creates .pivotignore with standard patterns."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code == 0
        content = pathlib.Path(".pivotignore").read_text()
        assert expected_pattern in content


def test_init_pivotignore_uses_default_patterns(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """init creates .pivotignore containing all default patterns from ignore module."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code == 0
        content = pathlib.Path(".pivotignore").read_text()

        # Verify each non-empty, non-comment pattern from defaults is in file
        for pattern in ignore.get_default_patterns():
            if pattern and not pattern.startswith("#"):
                assert pattern in content, f"Missing default pattern: {pattern}"


def test_init_does_not_overwrite_existing_pivotignore(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """init preserves existing .pivotignore file."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        pathlib.Path(".pivotignore").write_text("# Custom patterns\n*.custom\n")

        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code == 0
        content = pathlib.Path(".pivotignore").read_text()
        assert "*.custom" in content
        assert "*.pyc" not in content  # Not overwritten with defaults


def test_init_force_does_not_overwrite_pivotignore(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """init --force still preserves existing .pivotignore."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".pivotignore").write_text("# My patterns\n*.log\n")

        result = runner.invoke(cli.cli, ["init", "--force"])

        assert result.exit_code == 0
        content = pathlib.Path(".pivotignore").read_text()
        assert "*.log" in content
        assert "*.pyc" not in content


def test_init_output_mentions_pivotignore(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """init output lists .pivotignore as created file."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code == 0
        assert ".pivotignore" in result.output


# --- already initialized tests ---


def test_init_fails_with_suggestion_when_already_initialized(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code != 0
        assert "already initialized" in result.output.lower()
        assert "--force" in result.output


# --- force flag tests ---


def test_init_force_succeeds_when_already_initialized(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["init", "--force"])

        assert result.exit_code == 0


def test_init_force_overwrites_gitignore(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".pivot/.gitignore").write_text("old content")

        runner.invoke(cli.cli, ["init", "--force"])

        content = pathlib.Path(".pivot/.gitignore").read_text()
        assert "old content" not in content
        assert "cache/" in content


def test_init_force_preserves_other_files(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".pivot/config.yaml").write_text("cache:\n  dir: /custom\n")

        runner.invoke(cli.cli, ["init", "--force"])

        config = pathlib.Path(".pivot/config.yaml")
        assert config.exists()
        assert "/custom" in config.read_text()


# --- output message tests ---


def test_init_output_contains_expected_elements(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code == 0
        assert "initialized" in result.output.lower()
        assert ".pivot/" in result.output
        assert ".gitignore" in result.output
        assert "pipeline.py" in result.output


def test_init_next_steps_suggest_repro(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """The suggested run command must work without arguments (`pivot run` needs stage names)."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code == 0
        assert "'pivot repro'" in result.output
        assert "'pivot run'" not in result.output


# --- safety checks: symlink and file-not-dir ---


def test_init_fails_when_pivot_is_symlink_to_directory(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        target = pathlib.Path("real_dir")
        target.mkdir()
        pathlib.Path(".pivot").symlink_to(target)

        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code != 0
        assert "symlink" in result.output.lower()


def test_init_fails_when_pivot_is_symlink_to_file(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        target = pathlib.Path("some_file")
        target.write_text("content")
        pathlib.Path(".pivot").symlink_to(target)

        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code != 0
        assert "symlink" in result.output.lower()


def test_init_fails_when_pivot_is_dangling_symlink(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        pathlib.Path(".pivot").symlink_to("nonexistent")

        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code != 0
        assert "symlink" in result.output.lower()


def test_init_force_still_rejects_symlink(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        target = pathlib.Path("real_dir")
        target.mkdir()
        pathlib.Path(".pivot").symlink_to(target)

        result = runner.invoke(cli.cli, ["init", "--force"])

        assert result.exit_code != 0, "--force should not bypass symlink check"
        assert "symlink" in result.output.lower()


def test_init_fails_when_pivot_is_regular_file(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        pathlib.Path(".pivot").write_text("I am a file")

        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code != 0
        assert "not a directory" in result.output.lower()


def test_init_force_still_rejects_file(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        pathlib.Path(".pivot").write_text("I am a file")

        result = runner.invoke(cli.cli, ["init", "--force"])

        assert result.exit_code != 0, "--force should not bypass file check"
        assert "not a directory" in result.output.lower()


# --- permission tests ---


def test_init_fails_with_read_only_directory(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        cwd = pathlib.Path.cwd()
        cwd.chmod(0o555)  # read + execute only
        try:
            result = runner.invoke(cli.cli, ["init"])

            assert result.exit_code != 0
        finally:
            cwd.chmod(0o755)  # restore permissions for cleanup


# --- help tests ---


def test_init_help_shows_force_option(runner: click.testing.CliRunner) -> None:
    result = runner.invoke(cli.cli, ["init", "--help"])

    assert result.exit_code == 0
    assert "--force" in result.output


# --- integration tests ---


def test_init_creates_valid_project_structure(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integration test: init creates a valid project that other commands can use."""
    monkeypatch.chdir(tmp_path)
    project._project_root_cache = None

    result = runner.invoke(cli.cli, ["init"])

    assert result.exit_code == 0

    # Verify directory structure
    pivot_dir = tmp_path / ".pivot"
    assert pivot_dir.is_dir()
    assert (pivot_dir / ".gitignore").is_file()

    # Verify project root detection works after init
    project._project_root_cache = None
    assert project.find_project_root() == tmp_path

    # Verify gitignore content
    gitignore_content = (pivot_dir / ".gitignore").read_text()
    assert "cache/" in gitignore_content
    assert "state.lmdb/" in gitignore_content


# --- quiet mode tests ---


def test_init_quiet_produces_no_output(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot --quiet init produces no output."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        result = runner.invoke(cli.cli, ["--quiet", "init"])

        assert result.exit_code == 0
        assert result.output.strip() == "", "Quiet mode should suppress output"
        assert pathlib.Path(".pivot").is_dir(), "Should still create .pivot directory"


def test_init_quiet_still_creates_files(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot --quiet init still creates all expected files."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        result = runner.invoke(cli.cli, ["--quiet", "init"])

        assert result.exit_code == 0
        assert pathlib.Path(".pivot").is_dir()
        assert pathlib.Path(".pivot/.gitignore").exists()


# --- gitignore overwrite warning tests ---


def test_init_force_warns_when_overwriting_custom_gitignore(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot init --force warns when overwriting custom .gitignore."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".pivot/.gitignore").write_text("# Custom gitignore\nmy_custom_entry/\n")

        result = runner.invoke(cli.cli, ["init", "--force"])

        assert result.exit_code == 0
        assert "warning" in result.output.lower(), "Should warn about overwriting"


def test_init_force_no_warning_for_identical_gitignore(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot init --force doesn't warn when .gitignore is identical."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        pathlib.Path(".pivot").mkdir()
        # Write the exact same content that init would write
        pathlib.Path(".pivot/.gitignore").write_text(state_dir_mod.STATE_GITIGNORE_CONTENT)

        result = runner.invoke(cli.cli, ["init", "--force"])

        assert result.exit_code == 0
        assert "warning" not in result.output.lower(), "Should not warn for identical content"


def test_init_force_quiet_suppresses_warning(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot --quiet init --force suppresses overwrite warning."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".pivot/.gitignore").write_text("# Custom content")

        result = runner.invoke(cli.cli, ["--quiet", "init", "--force"])

        assert result.exit_code == 0
        assert result.output.strip() == "", "Quiet mode should suppress warning"


# --- not initialized tests ---


def test_repro_fails_without_pivot_init(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Running pivot repro without .pivot should error with helpful message."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        result = runner.invoke(cli.cli, ["repro"])

        assert result.exit_code == 1
        assert "No .pivot directory found" in result.output
        assert "pivot init" in result.output


def test_list_fails_without_pivot_init(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Running pivot list without .pivot should error with helpful message."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        project._project_root_cache = None
        result = runner.invoke(cli.cli, ["list"])

        assert result.exit_code == 1
        assert "No .pivot directory found" in result.output
