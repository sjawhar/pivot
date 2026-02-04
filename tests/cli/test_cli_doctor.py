import json
import pathlib
import subprocess

import click.testing
import pytest

from pivot import project
from pivot.cli import cli
from pivot.cli import doctor as doctor_module

# =============================================================================
# Human-Readable Output Tests
# =============================================================================


def test_doctor_basic_output(
    runner: click.testing.CliRunner, set_project_root: pathlib.Path
) -> None:
    """pivot doctor shows human-readable output."""
    result = runner.invoke(cli, ["doctor"])

    assert result.exit_code == 0
    assert "Pivot Environment Check" in result.output
    assert "Python version" in result.output
    assert "Project root" in result.output


def test_doctor_shows_all_checks_passing(
    runner: click.testing.CliRunner, set_project_root: pathlib.Path
) -> None:
    """pivot doctor shows all checks passing when environment is correct."""
    (set_project_root / "pivot.yaml").write_text("stages:\n  test:\n    cmd: echo hi\n")

    result = runner.invoke(cli, ["doctor"])

    assert result.exit_code == 0
    assert "[OK]" in result.output
    assert "All checks passed" in result.output


def test_doctor_shows_pipeline_config_warning(
    runner: click.testing.CliRunner, set_project_root: pathlib.Path
) -> None:
    """pivot doctor warns when no pipeline config exists."""
    result = runner.invoke(cli, ["doctor"])

    assert result.exit_code == 0
    assert "[WARN]" in result.output


def test_doctor_exits_1_on_error(
    runner: click.testing.CliRunner, set_project_root: pathlib.Path
) -> None:
    """pivot doctor exits 1 when errors are found."""
    # Create a malformed pivot.yaml to trigger a parse error
    (set_project_root / "pivot.yaml").write_text("stages:\n  - invalid: yaml: content\n")

    result = runner.invoke(cli, ["doctor"])

    assert result.exit_code == 1, "Should exit with code 1 when errors are found"
    assert "[ERROR]" in result.output or "error" in result.output.lower()


# =============================================================================
# JSON Output Tests
# =============================================================================


def test_doctor_json_output(
    runner: click.testing.CliRunner, set_project_root: pathlib.Path
) -> None:
    """pivot doctor --json outputs JSONL format."""
    result = runner.invoke(cli, ["doctor", "--json"])

    assert result.exit_code == 0
    lines = [line for line in result.output.strip().split("\n") if line]
    assert len(lines) >= 3, "Should have at least schema, checks, and summary"

    # Verify each line is valid JSON
    events = [json.loads(line) for line in lines]

    # First event should be schema version
    assert events[0]["type"] == "schema_version"
    assert events[0]["version"] == 1


def test_doctor_json_includes_all_checks(
    runner: click.testing.CliRunner, set_project_root: pathlib.Path
) -> None:
    """pivot doctor --json includes all check events."""
    result = runner.invoke(cli, ["doctor", "--json"])

    events = [json.loads(line) for line in result.output.strip().split("\n") if line]
    check_names = [e["name"] for e in events if e["type"] == "check"]

    assert "python_version" in check_names
    assert "project_root" in check_names
    assert "pipeline_config" in check_names
    assert "cache_directory" in check_names
    assert "git_repository" in check_names


def test_doctor_json_ends_with_summary(
    runner: click.testing.CliRunner, set_project_root: pathlib.Path
) -> None:
    """pivot doctor --json ends with summary event."""
    result = runner.invoke(cli, ["doctor", "--json"])

    events = [json.loads(line) for line in result.output.strip().split("\n") if line]

    summary = events[-1]
    assert summary["type"] == "summary"
    assert "passed" in summary
    assert "warnings" in summary
    assert "errors" in summary


# =============================================================================
# Remote Connectivity Tests
# =============================================================================


def test_doctor_remote_flag_checks_remotes(
    runner: click.testing.CliRunner, set_project_root: pathlib.Path
) -> None:
    """pivot doctor --remote checks configured remotes."""
    result = runner.invoke(cli, ["doctor", "--remote"])

    assert result.exit_code == 0
    # No remotes configured should show OK with "none configured"
    assert "remote" in result.output.lower() or "none configured" in result.output.lower()


def test_doctor_remote_json_includes_remote_checks(
    runner: click.testing.CliRunner, set_project_root: pathlib.Path
) -> None:
    """pivot doctor --remote --json includes remote check events."""
    result = runner.invoke(cli, ["doctor", "--remote", "--json"])

    events = [json.loads(line) for line in result.output.strip().split("\n") if line]
    check_names = [e["name"] for e in events if e["type"] == "check"]

    # Should have at least one remote check (even if just "none configured")
    assert any("remote" in name for name in check_names)


# =============================================================================
# Check Function Unit Tests
# =============================================================================


def test_check_python_version_returns_ok() -> None:
    """_check_python_version returns OK status."""
    result = doctor_module._check_python_version()

    assert result["type"] == "check"
    assert result["name"] == "python_version"
    assert result["status"] == doctor_module.CheckStatus.OK
    assert "." in result["value"]  # Should be version string like "3.13.0"


def test_check_project_root_returns_path_when_found(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_check_project_root returns OK with path when found."""
    (tmp_path / ".pivot").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project, "_project_root_cache", None)

    event, root = doctor_module._check_project_root()

    assert event["status"] == doctor_module.CheckStatus.OK
    assert root is not None
    assert str(tmp_path) in event["value"]


def test_check_pipeline_config_ok_with_yaml(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_check_pipeline_config returns OK when pivot.yaml exists."""
    (tmp_path / "pivot.yaml").write_text("stages:\n  test:\n    cmd: echo\n")

    result = doctor_module._check_pipeline_config(tmp_path)

    assert result["status"] == doctor_module.CheckStatus.OK
    assert result["value"] == "pivot.yaml"
    assert result["details"] is not None
    assert result["details"]["stages"] == 1


def test_check_pipeline_config_ok_with_pipeline_py(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_check_pipeline_config returns OK when pipeline.py exists."""
    (tmp_path / "pipeline.py").write_text("# pipeline")

    result = doctor_module._check_pipeline_config(tmp_path)

    assert result["status"] == doctor_module.CheckStatus.OK
    assert result["value"] == "pipeline.py"


def test_check_pipeline_config_warn_when_missing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_check_pipeline_config returns WARN when no config exists."""

    result = doctor_module._check_pipeline_config(tmp_path)

    assert result["status"] == doctor_module.CheckStatus.WARN
    assert result["value"] == "not found"


def test_check_pipeline_config_handles_scalar_yaml(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_check_pipeline_config handles scalar YAML (not a dict)."""
    (tmp_path / "pivot.yaml").write_text("just a string")

    result = doctor_module._check_pipeline_config(tmp_path)

    # Should not crash - should return OK with 0 stages
    assert result["status"] == doctor_module.CheckStatus.OK
    assert result["details"] is not None
    assert result["details"]["stages"] == 0


def test_check_cache_directory_ok_when_exists_and_writable(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_check_cache_directory returns OK when cache exists and is writable."""
    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)

    # Set up project root so config.get_cache_dir() works correctly
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result = doctor_module._check_cache_directory(tmp_path)

    assert result["status"] == doctor_module.CheckStatus.OK
    assert result["details"] is not None
    assert result["details"]["exists"] is True
    assert result["details"]["writable"] is True


def test_check_cache_directory_ok_when_not_exists(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_check_cache_directory returns OK when cache doesn't exist yet."""
    # Don't create cache directory

    # Set up project root so config.get_cache_dir() works correctly
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result = doctor_module._check_cache_directory(tmp_path)

    assert result["status"] == doctor_module.CheckStatus.OK
    assert result["details"] is not None
    assert result["details"]["exists"] is False


def test_check_git_repository_ok_when_in_repo(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_check_git_repository returns OK when in git repo."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)

    result = doctor_module._check_git_repository(tmp_path)

    assert result["status"] == doctor_module.CheckStatus.OK
    assert result["value"] == "found"
    assert result["details"] is not None
    assert "branch" in result["details"]


def test_check_git_repository_warn_when_not_in_repo(tmp_path: pathlib.Path) -> None:
    """_check_git_repository returns WARN when not in git repo."""
    result = doctor_module._check_git_repository(tmp_path)

    assert result["status"] == doctor_module.CheckStatus.WARN
    assert "not found" in result["value"] or "not installed" in result["value"]


# =============================================================================
# URL Sanitization Tests
# =============================================================================


def test_sanitize_url_preserves_url_without_password() -> None:
    """_sanitize_url preserves URLs without passwords."""
    url = "s3://my-bucket/prefix"
    assert doctor_module._sanitize_url(url) == url


def test_sanitize_url_removes_password() -> None:
    """_sanitize_url removes password from URL."""
    url = "s3://user:secret@my-bucket/prefix"
    result = doctor_module._sanitize_url(url)

    assert "secret" not in result
    assert "user@" in result


def test_sanitize_url_preserves_port() -> None:
    """_sanitize_url preserves port number."""
    url = "http://user:pass@host:8080/path"
    result = doctor_module._sanitize_url(url)

    assert "pass" not in result
    assert ":8080" in result


# =============================================================================
# Skipped Check Tests
# =============================================================================


def test_skipped_check_creates_error_event() -> None:
    """_skipped_check creates an error event."""
    result = doctor_module._skipped_check("test_check")

    assert result["type"] == "check"
    assert result["name"] == "test_check"
    assert result["status"] == doctor_module.CheckStatus.ERROR
    assert result["value"] == "skipped"


def test_skipped_check_includes_reason() -> None:
    """_skipped_check includes reason in details."""
    result = doctor_module._skipped_check("test_check", "custom reason")

    assert result["details"] is not None
    assert result["details"]["reason"] == "custom reason"
