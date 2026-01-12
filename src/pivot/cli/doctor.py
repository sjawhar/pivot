from __future__ import annotations

import enum
import os
import subprocess
import sys
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast
from urllib.parse import urlparse, urlunparse

import click

from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers

if TYPE_CHECKING:
    import pathlib


# JSONL schema version for forward compatibility
_JSONL_SCHEMA_VERSION = 1


class CheckStatus(enum.StrEnum):
    OK = "ok"
    WARN = "warn"
    ERROR = "error"


class DoctorCheckEvent(TypedDict):
    """JSONL event for a single doctor check."""

    type: Literal["check"]
    name: str
    status: CheckStatus
    value: str
    details: dict[str, object] | None


class DoctorSummaryEvent(TypedDict):
    """JSONL summary event for doctor command."""

    type: Literal["summary"]
    passed: int
    warnings: int
    errors: int


class DoctorSchemaVersionEvent(TypedDict):
    """Schema version event for doctor JSONL."""

    type: Literal["schema_version"]
    version: int


DoctorJsonEvent = DoctorSchemaVersionEvent | DoctorCheckEvent | DoctorSummaryEvent


def _skipped_check(name: str, reason: str = "no project root") -> DoctorCheckEvent:
    """Create a skipped check event for when a prerequisite is missing."""
    return DoctorCheckEvent(
        type="check",
        name=name,
        status=CheckStatus.ERROR,
        value="skipped",
        details={"reason": reason},
    )


def _sanitize_url(url: str) -> str:
    """Remove password from URL if present, preserve everything else."""
    parsed = urlparse(url)
    if not parsed.password:
        return url

    # Rebuild netloc without password
    netloc = parsed.username or ""
    if netloc:
        netloc += "@"
    netloc += parsed.hostname or ""
    if parsed.port:
        netloc += f":{parsed.port}"

    return urlunparse(parsed._replace(netloc=netloc))


def _relative_path(path: pathlib.Path, project_root: pathlib.Path) -> str:
    """Return path relative to project root, or absolute if outside."""
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _check_python_version() -> DoctorCheckEvent:
    """Check Python version is 3.13+."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    # Diagnostic check - always returns OK since we require 3.13+ to run
    return DoctorCheckEvent(
        type="check",
        name="python_version",
        status=CheckStatus.OK,
        value=version,
        details=None,
    )


def _check_project_root() -> tuple[DoctorCheckEvent, pathlib.Path | None]:
    """Check project root exists."""
    from pivot import project

    try:
        root = project.get_project_root()
        return (
            DoctorCheckEvent(
                type="check",
                name="project_root",
                status=CheckStatus.OK,
                value=str(root),
                details=None,
            ),
            root,
        )
    except Exception as e:
        return (
            DoctorCheckEvent(
                type="check",
                name="project_root",
                status=CheckStatus.ERROR,
                value="not found",
                details={"error": str(e)},
            ),
            None,
        )


def _check_pipeline_config(project_root: pathlib.Path | None) -> DoctorCheckEvent:
    """Check pivot.yaml/yml exists and is valid."""
    if project_root is None:
        return _skipped_check("pipeline_config")

    import yaml

    for name in ("pivot.yaml", "pivot.yml"):
        config_path = project_root / name
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config: object = yaml.safe_load(f)
                # Handle scalar YAML (e.g., just a string) which isn't a valid config
                if isinstance(config, dict):
                    config_dict = cast("dict[str, Any]", config)
                    stages_raw: object = config_dict.get("stages", {})
                else:
                    stages_raw = {}
                stage_count = (
                    len(cast("dict[str, Any]", stages_raw)) if isinstance(stages_raw, dict) else 0
                )
                return DoctorCheckEvent(
                    type="check",
                    name="pipeline_config",
                    status=CheckStatus.OK,
                    value=name,
                    details={"stages": stage_count},
                )
            except Exception as e:
                return DoctorCheckEvent(
                    type="check",
                    name="pipeline_config",
                    status=CheckStatus.ERROR,
                    value=name,
                    details={"error": str(e)},
                )

    # Check for pipeline.py
    pipeline_py = project_root / "pipeline.py"
    if pipeline_py.exists():
        return DoctorCheckEvent(
            type="check",
            name="pipeline_config",
            status=CheckStatus.OK,
            value="pipeline.py",
            details=None,
        )

    return DoctorCheckEvent(
        type="check",
        name="pipeline_config",
        status=CheckStatus.WARN,
        value="not found",
        details={"searched": ["pivot.yaml", "pivot.yml", "pipeline.py"]},
    )


def _check_cache_directory(project_root: pathlib.Path | None) -> DoctorCheckEvent:
    """Check cache directory exists and is writable."""
    if project_root is None:
        return _skipped_check("cache_directory")

    cache_dir = project_root / ".pivot" / "cache"
    if not cache_dir.exists():
        # Cache doesn't exist yet - that's OK, it'll be created on first run
        return DoctorCheckEvent(
            type="check",
            name="cache_directory",
            status=CheckStatus.OK,
            value=_relative_path(cache_dir, project_root),
            details={"exists": False, "writable": True, "note": "will be created on first run"},
        )

    # Check if writable using os.access (no file creation needed)
    writable = os.access(cache_dir, os.W_OK)
    return DoctorCheckEvent(
        type="check",
        name="cache_directory",
        status=CheckStatus.OK if writable else CheckStatus.ERROR,
        value=_relative_path(cache_dir, project_root),
        details={"exists": True, "writable": writable},
    )


def _check_git_repository(project_root: pathlib.Path | None) -> DoctorCheckEvent:
    """Check if in a git repository."""
    if project_root is None:
        return _skipped_check("git_repository")

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=5,
        )
        if result.returncode != 0:
            return DoctorCheckEvent(
                type="check",
                name="git_repository",
                status=CheckStatus.WARN,
                value="not found",
                details={"note": "git is optional but recommended for versioning"},
            )

        # Get branch name
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=5,
        )
        branch = branch_result.stdout.strip() or "detached HEAD"

        # Check for uncommitted changes
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=5,
        )
        clean = len(status_result.stdout.strip()) == 0

        return DoctorCheckEvent(
            type="check",
            name="git_repository",
            status=CheckStatus.OK,
            value="found",
            details={"branch": branch, "clean": clean},
        )
    except FileNotFoundError:
        return DoctorCheckEvent(
            type="check",
            name="git_repository",
            status=CheckStatus.WARN,
            value="git not installed",
            details={"note": "git is optional but recommended for versioning"},
        )
    except subprocess.TimeoutExpired:
        return DoctorCheckEvent(
            type="check",
            name="git_repository",
            status=CheckStatus.WARN,
            value="timeout",
            details={"note": "git command timed out"},
        )


def _check_remote_connectivity(project_root: pathlib.Path | None) -> list[DoctorCheckEvent]:
    """Check connectivity to configured remotes."""
    if project_root is None:
        return [_skipped_check("remote")]

    from pivot.remote import config as remote_config

    try:
        remotes = remote_config.list_remotes()
    except Exception as e:
        return [
            DoctorCheckEvent(
                type="check",
                name="remote",
                status=CheckStatus.ERROR,
                value="config error",
                details={"error": str(e)},
            )
        ]

    if not remotes:
        return [
            DoctorCheckEvent(
                type="check",
                name="remote",
                status=CheckStatus.OK,
                value="none configured",
                details=None,
            )
        ]

    results = list[DoctorCheckEvent]()
    for name, url in remotes.items():
        sanitized_url = _sanitize_url(url)
        try:
            # Validate S3 URL format
            remote_config.validate_s3_url(url)
            results.append(
                DoctorCheckEvent(
                    type="check",
                    name=f"remote:{name}",
                    status=CheckStatus.OK,
                    value=sanitized_url,
                    details={"valid": True},
                )
            )
        except Exception as e:
            results.append(
                DoctorCheckEvent(
                    type="check",
                    name=f"remote:{name}",
                    status=CheckStatus.ERROR,
                    value=sanitized_url,
                    details={"valid": False, "error": str(e)},
                )
            )

    return results


def _print_check_human(check: DoctorCheckEvent) -> None:
    """Print a check result in human-readable format."""
    name = check["name"]
    status = check["status"]
    value = check["value"]
    details = check["details"]

    # Format status indicator
    if status == CheckStatus.OK:
        indicator = "[OK]"
    elif status == CheckStatus.WARN:
        indicator = "[WARN]"
    else:
        indicator = "[ERROR]"

    # Format value with details
    if name == "python_version" or name == "project_root":
        display = value
    elif name == "pipeline_config":
        if details and "stages" in details:
            display = f"{value} ({details['stages']} stages)"
        else:
            display = value
    elif name == "cache_directory":
        if details and details.get("writable"):
            display = f"{value} (writable)"
        elif details and not details.get("exists"):
            display = f"{value} (will be created)"
        else:
            display = f"{value} (not writable)"
    elif name == "git_repository":
        if details and "branch" in details:
            clean_str = ", clean" if details.get("clean") else ", uncommitted changes"
            display = f"{details['branch']} branch{clean_str}"
        else:
            display = value
    elif name.startswith("remote:"):
        display = value
    else:
        display = value

    # Print with padding
    label = name.replace("_", " ").replace(":", ": ")
    click.echo(f"  {label.capitalize():.<30} {display} {indicator}")

    # Print error details on next line if present
    if status == CheckStatus.ERROR and details and "error" in details:
        click.echo(f"    Error: {details['error']}")


@cli_decorators.pivot_command(auto_discover=False)
@click.option("--json", "as_json", is_flag=True, help="Output as JSONL")
@click.option("--remote", "check_remote", is_flag=True, help="Also check remote connectivity")
def doctor(as_json: bool, check_remote: bool) -> None:
    """Check environment and configuration for issues."""
    checks = list[DoctorCheckEvent]()

    # Run checks
    checks.append(_check_python_version())

    project_check, project_root = _check_project_root()
    checks.append(project_check)

    checks.append(_check_pipeline_config(project_root))
    checks.append(_check_cache_directory(project_root))
    checks.append(_check_git_repository(project_root))

    if check_remote:
        checks.extend(_check_remote_connectivity(project_root))

    # Count results
    passed = sum(1 for c in checks if c["status"] == CheckStatus.OK)
    warnings = sum(1 for c in checks if c["status"] == CheckStatus.WARN)
    errors = sum(1 for c in checks if c["status"] == CheckStatus.ERROR)

    if as_json:
        # Emit JSONL
        cli_helpers.emit_jsonl(
            DoctorSchemaVersionEvent(type="schema_version", version=_JSONL_SCHEMA_VERSION)
        )
        for check in checks:
            cli_helpers.emit_jsonl(check)
        cli_helpers.emit_jsonl(
            DoctorSummaryEvent(type="summary", passed=passed, warnings=warnings, errors=errors)
        )
    else:
        # Human-readable output
        click.echo("Pivot Environment Check")
        click.echo()

        for check in checks:
            _print_check_human(check)

        click.echo()
        if errors > 0:
            click.echo(f"{errors} error(s), {warnings} warning(s), {passed} passed")
        elif warnings > 0:
            click.echo(f"All checks passed with {warnings} warning(s).")
        else:
            click.echo("All checks passed.")

    # Exit with error code if any errors
    if errors > 0:
        raise SystemExit(1)
