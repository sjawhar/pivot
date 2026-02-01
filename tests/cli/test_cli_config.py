import json
import pathlib

import click.testing
import pytest

from pivot import cli

# --- config list tests ---


def test_config_list_shows_defaults(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["config", "list"])

        assert result.exit_code == 0
        assert "cache.dir" in result.output
        assert "core.max_workers" in result.output


def test_config_list_shows_local_override(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".pivot/config.yaml").write_text("cache:\n  dir: /custom\n")

        result = runner.invoke(cli.cli, ["config", "list"])

        assert result.exit_code == 0
        assert "/custom" in result.output
        assert "local" in result.output


def test_config_list_json_output(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["config", "list", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "cache" in data
        assert "core" in data


def test_config_list_local_only(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".pivot/config.yaml").write_text("cache:\n  dir: /local\n")

        result = runner.invoke(cli.cli, ["config", "list", "--local"])

        assert result.exit_code == 0
        assert "/local" in result.output


# --- config get tests ---


def test_config_get_returns_value(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".pivot/config.yaml").write_text("cache:\n  dir: /custom\n")

        result = runner.invoke(cli.cli, ["config", "get", "cache.dir"])

        assert result.exit_code == 0
        assert "/custom" in result.output


def test_config_get_returns_default(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["config", "get", "core.max_workers"])

        assert result.exit_code == 0
        assert "-2" in result.output
        assert "default" in result.output


def test_config_get_json_output(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["config", "get", "cache.dir", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "key" in data
        assert "value" in data
        assert "source" in data


def test_config_get_unknown_key_errors(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["config", "get", "unknown.key"])

        assert result.exit_code != 0
        assert "unknown" in result.output.lower()


# --- config set tests ---


def test_config_set_creates_value(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["config", "set", "cache.dir", "/custom"])

        assert result.exit_code == 0
        assert "cache.dir" in result.output

        config_file = pathlib.Path(".pivot/config.yaml")
        assert config_file.exists()
        assert "/custom" in config_file.read_text()


def test_config_set_validates_value(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["config", "set", "core.max_workers", "not-a-number"])

        assert result.exit_code != 0
        assert "integer" in result.output.lower() or "invalid" in result.output.lower()


def test_config_set_coerces_string_to_int(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["config", "set", "core.max_workers", "4"])

        assert result.exit_code == 0

        config_file = pathlib.Path(".pivot/config.yaml")
        content = config_file.read_text()
        assert "max_workers: 4" in content


def test_config_set_global_flag(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_config = tmp_path / ".config" / "pivot" / "config.yaml"

    from pivot.config import io

    monkeypatch.setattr(io, "get_global_config_path", lambda: global_config)

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["config", "set", "--global", "display.precision", "3"])

        assert result.exit_code == 0
        assert global_config.exists()


def test_config_set_checkout_mode_comma_separated(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["config", "set", "cache.checkout_mode", "symlink,copy"])

        assert result.exit_code == 0

        config_file = pathlib.Path(".pivot/config.yaml")
        content = config_file.read_text()
        assert "symlink" in content
        assert "copy" in content


def test_config_set_remote(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["config", "set", "remotes.origin", "s3://my-bucket/cache"])

        assert result.exit_code == 0

        config_file = pathlib.Path(".pivot/config.yaml")
        content = config_file.read_text()
        assert "origin" in content
        assert "s3://my-bucket/cache" in content


def test_config_set_invalid_remote_url(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["config", "set", "remotes.origin", "not-an-s3-url"])

        assert result.exit_code != 0
        assert "s3://" in result.output.lower()


# --- config unset tests ---


def test_config_unset_removes_value(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".pivot/config.yaml").write_text("cache:\n  dir: /custom\n")

        result = runner.invoke(cli.cli, ["config", "unset", "cache.dir"])

        assert result.exit_code == 0
        assert "removed" in result.output.lower() or "unset" in result.output.lower()

        config_file = pathlib.Path(".pivot/config.yaml")
        content = config_file.read_text()
        assert "/custom" not in content


def test_config_unset_missing_key(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["config", "unset", "cache.dir"])

        assert result.exit_code == 0
        assert "not found" in result.output.lower() or "not set" in result.output.lower()


def test_config_unset_global_flag(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_config = tmp_path / ".config" / "pivot" / "config.yaml"
    global_config.parent.mkdir(parents=True)
    global_config.write_text("display:\n  precision: 3\n")

    from pivot.config import io

    monkeypatch.setattr(io, "get_global_config_path", lambda: global_config)

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["config", "unset", "--global", "display.precision"])

        assert result.exit_code == 0

        content = global_config.read_text()
        assert "precision" not in content


# --- config help tests ---


def test_config_help_shows_subcommands(runner: click.testing.CliRunner) -> None:
    result = runner.invoke(cli.cli, ["config", "--help"])

    assert result.exit_code == 0
    assert "list" in result.output
    assert "get" in result.output
    assert "set" in result.output
    assert "unset" in result.output


def test_config_set_help_shows_options(runner: click.testing.CliRunner) -> None:
    result = runner.invoke(cli.cli, ["config", "set", "--help"])

    assert result.exit_code == 0
    assert "--global" in result.output


def test_config_list_help_shows_options(runner: click.testing.CliRunner) -> None:
    result = runner.invoke(cli.cli, ["config", "list", "--help"])

    assert result.exit_code == 0
    assert "--json" in result.output
    assert "--local" in result.output or "--global" in result.output


# --- config get --global/--local tests ---


def test_config_get_global_flag(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """config get --global reads from global config only."""
    # Set up global config path before entering isolated filesystem
    global_dir = tmp_path / "global"
    global_dir.mkdir()
    global_config = global_dir / "config.yaml"
    global_config.write_text("display:\n  precision: 7\n")

    from pivot import config
    from pivot.config import io

    # Patch both locations since CLI uses `config.get_global_config_path`
    monkeypatch.setattr(io, "get_global_config_path", lambda: global_config)
    monkeypatch.setattr(config, "get_global_config_path", lambda: global_config)

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        # Set local value (different from global)
        local_config = pathlib.Path(".pivot/config.yaml")
        local_config.write_text("display:\n  precision: 3\n")

        result = runner.invoke(cli.cli, ["config", "get", "--global", "display.precision"])

        assert result.exit_code == 0
        assert "7" in result.output  # Global value, not local


def test_config_get_local_flag(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
) -> None:
    """config get --local reads from local config only."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        # Set local value
        local_config = pathlib.Path(".pivot/config.yaml")
        local_config.write_text("display:\n  precision: 3\n")

        result = runner.invoke(cli.cli, ["config", "get", "--local", "display.precision"])

        assert result.exit_code == 0
        assert "3" in result.output  # Local value


def test_config_get_global_local_mutually_exclusive(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
) -> None:
    """config get --global --local should error."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(
            cli.cli, ["config", "get", "--global", "--local", "display.precision"]
        )

        assert result.exit_code != 0
        assert "Cannot use both" in result.output


def test_config_get_help_shows_scope_options(runner: click.testing.CliRunner) -> None:
    """config get --help should show --global and --local options."""
    result = runner.invoke(cli.cli, ["config", "get", "--help"])

    assert result.exit_code == 0
    assert "--global" in result.output
    assert "--local" in result.output
