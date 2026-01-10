import pathlib

import click.testing
import pytest

from pivot import cli, project


@pytest.fixture
def runner() -> click.testing.CliRunner:
    return click.testing.CliRunner()


# --- basic initialization tests ---


def test_init_creates_pivot_directory(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code == 0
        assert pathlib.Path(".pivot").is_dir()


def test_init_creates_gitignore(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code == 0
        assert pathlib.Path(".pivot/.gitignore").exists()


@pytest.mark.parametrize(
    "expected_content",
    [
        "cache/",
        "state.db",
        "state.lmdb/",
        "config.yaml.lock",
    ],
)
def test_init_gitignore_contains_expected_entries(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, expected_content: str
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code == 0
        content = pathlib.Path(".pivot/.gitignore").read_text()
        assert expected_content in content


# --- already initialized tests ---


def test_init_fails_with_suggestion_when_already_initialized(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
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
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["init", "--force"])

        assert result.exit_code == 0


def test_init_force_overwrites_gitignore(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
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
        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code == 0
        assert "initialized" in result.output.lower()
        assert ".pivot/" in result.output
        assert ".gitignore" in result.output
        assert "pivot.yaml" in result.output


# --- safety checks: symlink and file-not-dir ---


def test_init_fails_when_pivot_is_symlink_to_directory(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
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
        pathlib.Path(".pivot").symlink_to("nonexistent")

        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code != 0
        assert "symlink" in result.output.lower()


def test_init_force_still_rejects_symlink(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
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
        pathlib.Path(".pivot").write_text("I am a file")

        result = runner.invoke(cli.cli, ["init"])

        assert result.exit_code != 0
        assert "not a directory" in result.output.lower()


def test_init_force_still_rejects_file(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").write_text("I am a file")

        result = runner.invoke(cli.cli, ["init", "--force"])

        assert result.exit_code != 0, "--force should not bypass file check"
        assert "not a directory" in result.output.lower()


# --- permission tests ---


def test_init_fails_with_read_only_directory(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
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
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integration test: init creates a valid project that other commands can use."""
    monkeypatch.chdir(tmp_path)
    project._project_root_cache = None

    runner = click.testing.CliRunner()
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
