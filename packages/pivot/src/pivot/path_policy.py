from __future__ import annotations

import enum
import logging
import os
import pathlib
from typing import Literal, TypedDict

from pivot import exceptions

SymlinkAction = Literal["error", "warn", "allow"]

logger = logging.getLogger(__name__)


class PathType(enum.StrEnum):
    """Classification of paths for policy enforcement."""

    DEP = "dependency"
    OUT = "output"
    CWD = "working directory"
    VAR = "vars file"
    CLI_OUTPUT = "CLI output"


class PathPolicy(TypedDict):
    """Policy configuration for a path type."""

    allow_absolute: bool
    symlink_escape_action: SymlinkAction


POLICIES: dict[PathType, PathPolicy] = {
    PathType.DEP: PathPolicy(
        allow_absolute=True,
        symlink_escape_action="warn",
    ),
    PathType.OUT: PathPolicy(
        allow_absolute=False,
        symlink_escape_action="error",
    ),
    PathType.CWD: PathPolicy(
        allow_absolute=False,
        symlink_escape_action="error",
    ),
    PathType.VAR: PathPolicy(
        allow_absolute=False,
        symlink_escape_action="error",
    ),
    PathType.CLI_OUTPUT: PathPolicy(
        allow_absolute=False,
        symlink_escape_action="error",
    ),
}


class PathValidationResult(TypedDict):
    """Result of path validation."""

    valid: bool
    normalized_path: pathlib.Path | None
    error: str | None
    warnings: list[str]


def has_path_traversal(path: str) -> bool:
    """Check if path contains traversal components (..)."""
    return ".." in pathlib.Path(path).parts


def validate_path_syntax(path: str) -> str | None:
    """Validate path has no injection characters. Returns error message or None."""
    if "\x00" in path:
        return "contains null byte"
    if "\n" in path or "\r" in path:
        return "contains newline character"
    if has_path_traversal(path):
        return "contains path traversal (..)"
    return None


def validate_path(
    path: str,
    path_type: PathType,
    base_dir: pathlib.Path,
    *,
    check_exists: bool = False,
) -> PathValidationResult:
    """Validate a path according to its type's policy.

    Args:
        path: The path string to validate
        path_type: Classification determining which policy applies
        base_dir: Directory that relative paths are resolved against and must stay within
        check_exists: If True, also check symlink resolution (requires path to exist)

    Returns:
        PathValidationResult with validation outcome
    """
    policy = POLICIES[path_type]
    warnings = list[str]()

    # Syntax validation (always applies)
    syntax_error = validate_path_syntax(path)
    if syntax_error:
        return PathValidationResult(
            valid=False,
            normalized_path=None,
            error=f"{path_type.value} path {syntax_error}: {path!r}",
            warnings=warnings,
        )

    # Normalize path to absolute
    if os.path.isabs(path):
        normalized = pathlib.Path(path)
    else:
        normalized = pathlib.Path(os.path.normpath(base_dir / path))

    # Check if path is within base_dir (logical path check, no symlink resolution)
    is_within_base = _is_within(normalized, base_dir)

    if not is_within_base:
        # Path is outside base_dir
        if not policy["allow_absolute"]:
            return PathValidationResult(
                valid=False,
                normalized_path=None,
                error=f"{path_type.value} path '{path}' resolves outside base directory",
                warnings=warnings,
            )
        # Allowed (deps only) - warn about reproducibility
        warnings.append(f"Absolute {path_type.value} path may break reproducibility: {path}")

    # Symlink resolution (only if path exists and check_exists=True)
    if check_exists and normalized.exists():
        try:
            real_path = normalized.resolve()
            real_base = base_dir.resolve()
            if not _is_within(real_path, real_base):
                msg = f"{path_type.value} path resolves outside base via symlink: {path!r} -> {real_path}"
                match policy["symlink_escape_action"]:
                    case "error":
                        return PathValidationResult(
                            valid=False,
                            normalized_path=None,
                            error=msg,
                            warnings=warnings,
                        )
                    case "warn":
                        warnings.append(msg)
                    case "allow":
                        pass
        except OSError as e:
            warnings.append(f"Could not resolve symlink for {path_type.value}: {path!r} ({e})")

    return PathValidationResult(
        valid=True,
        normalized_path=normalized,
        error=None,
        warnings=warnings,
    )


def require_valid_path(
    path: str,
    path_type: PathType,
    base_dir: pathlib.Path,
    *,
    check_exists: bool = False,
    context: str = "",
) -> pathlib.Path:
    """Validate path and raise SecurityValidationError if invalid.

    Args:
        path: The path string to validate
        path_type: Classification determining which policy applies
        base_dir: Directory that relative paths are resolved against
        check_exists: If True, also check symlink resolution
        context: Optional context for error messages (e.g., stage name)

    Returns:
        Normalized path if valid

    Raises:
        SecurityValidationError: If path validation fails
    """
    result = validate_path(path, path_type, base_dir, check_exists=check_exists)

    for warning in result["warnings"]:
        prefix = f"{context}: " if context else ""
        logger.warning(f"{prefix}{warning}")

    if not result["valid"]:
        prefix = f"{context}: " if context else ""
        raise exceptions.SecurityValidationError(f"{prefix}{result['error']}")

    # This is guaranteed to be set when valid=True
    assert result["normalized_path"] is not None
    return result["normalized_path"]


def _is_within(path: pathlib.Path, root: pathlib.Path) -> bool:
    """Check if path is within root directory."""
    try:
        # Use absolute paths for comparison to handle edge cases
        path.absolute().relative_to(root.absolute())
        return True
    except ValueError:
        return False
