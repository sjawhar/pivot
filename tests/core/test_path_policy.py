from __future__ import annotations

import pathlib

import pytest

from pivot import exceptions, path_policy

# --- validate_path_syntax tests ---


@pytest.mark.parametrize(
    ("path", "expected_error"),
    [
        ("normal/path.txt", None),
        ("path\x00with\x00null", "contains null byte"),
        ("path\nwith\nnewline", "contains newline character"),
        ("path\rwith\rcarriage", "contains newline character"),
        ("../escape/path", "contains path traversal (..)"),
        ("path/../escape", "contains path traversal (..)"),
        ("path/to/../file", "contains path traversal (..)"),
        ("..hidden", None),
        ("hidden..", None),
        ("..", "contains path traversal (..)"),
    ],
)
def test_validate_path_syntax(path: str, expected_error: str | None) -> None:
    result = path_policy.validate_path_syntax(path)
    assert result == expected_error


# --- has_path_traversal tests ---


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("a/b/c", False),
        ("../a", True),
        ("a/../b", True),
        ("a/b/..", True),
        ("..", True),
        ("..hidden", False),
        ("file..", False),
        ("a..b", False),
    ],
)
def test_has_path_traversal(path: str, expected: bool) -> None:
    result = path_policy.has_path_traversal(path)
    assert result == expected


# --- validate_path tests for different PathTypes ---


def test_validate_path_relative_within_base(tmp_path: pathlib.Path) -> None:
    """Relative paths within base_dir should be valid for all path types."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    for path_type in path_policy.PathType:
        result = path_policy.validate_path("subdir/file.txt", path_type, tmp_path)
        assert result["valid"], f"Should be valid for {path_type}"
        assert result["normalized_path"] == tmp_path / "subdir" / "file.txt"
        assert result["error"] is None


def test_validate_path_absolute_outside_base_dep_allowed(tmp_path: pathlib.Path) -> None:
    """DEP allows absolute paths outside base_dir with warning."""
    outside_path = "/some/external/path.txt"

    result = path_policy.validate_path(outside_path, path_policy.PathType.DEP, tmp_path)

    assert result["valid"]
    assert len(result["warnings"]) == 1
    assert "may break reproducibility" in result["warnings"][0]


@pytest.mark.parametrize(
    "path_type",
    [
        path_policy.PathType.OUT,
        path_policy.PathType.CWD,
        path_policy.PathType.VAR,
        path_policy.PathType.CLI_OUTPUT,
    ],
)
def test_validate_path_absolute_outside_base_rejected(
    tmp_path: pathlib.Path,
    path_type: path_policy.PathType,
) -> None:
    """OUT/CWD/VAR/CLI_OUTPUT reject paths outside base_dir."""
    outside_path = "/some/external/path.txt"

    result = path_policy.validate_path(outside_path, path_type, tmp_path)

    assert not result["valid"]
    assert "resolves outside base directory" in (result["error"] or "")


def test_validate_path_absolute_within_base_allowed(tmp_path: pathlib.Path) -> None:
    """Absolute paths within base_dir are allowed for all path types."""
    file_path = tmp_path / "data.csv"
    file_path.touch()

    for path_type in path_policy.PathType:
        result = path_policy.validate_path(str(file_path), path_type, tmp_path)
        assert result["valid"], f"Absolute path within base should be valid for {path_type}"


def test_validate_path_syntax_error_null_byte(tmp_path: pathlib.Path) -> None:
    """Null bytes in path should fail validation."""
    result = path_policy.validate_path("file\x00.txt", path_policy.PathType.OUT, tmp_path)

    assert not result["valid"]
    assert "contains null byte" in (result["error"] or "")


def test_validate_path_syntax_error_traversal(tmp_path: pathlib.Path) -> None:
    """Path traversal should fail validation."""
    result = path_policy.validate_path("../escape.txt", path_policy.PathType.OUT, tmp_path)

    assert not result["valid"]
    assert "contains path traversal" in (result["error"] or "")


# --- Symlink escape tests ---


def test_validate_path_symlink_escape_error(tmp_path: pathlib.Path) -> None:
    """OUT type should error when symlink escapes base_dir."""
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "secret.txt"
    outside_file.write_text("secret")

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    symlink = project_dir / "link.txt"
    symlink.symlink_to(outside_file)

    result = path_policy.validate_path(
        "link.txt",
        path_policy.PathType.OUT,
        project_dir,
        check_exists=True,
    )

    assert not result["valid"]
    assert "resolves outside base via symlink" in (result["error"] or "")


def test_validate_path_symlink_escape_warn(tmp_path: pathlib.Path) -> None:
    """DEP type should warn when symlink escapes base_dir."""
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "data.csv"
    outside_file.write_text("data")

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    symlink = project_dir / "link.csv"
    symlink.symlink_to(outside_file)

    result = path_policy.validate_path(
        "link.csv",
        path_policy.PathType.DEP,
        project_dir,
        check_exists=True,
    )

    assert result["valid"]
    assert len(result["warnings"]) == 1
    assert "resolves outside base via symlink" in result["warnings"][0]


def test_validate_path_symlink_within_base_ok(tmp_path: pathlib.Path) -> None:
    """Symlinks that stay within base_dir should be fine."""
    target_file = tmp_path / "target.txt"
    target_file.write_text("content")
    symlink = tmp_path / "link.txt"
    symlink.symlink_to(target_file)

    result = path_policy.validate_path(
        "link.txt",
        path_policy.PathType.OUT,
        tmp_path,
        check_exists=True,
    )

    assert result["valid"]
    assert len(result["warnings"]) == 0


def test_validate_path_check_exists_false_skips_symlink_check(tmp_path: pathlib.Path) -> None:
    """When check_exists=False, symlink resolution is skipped."""
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "secret.txt"
    outside_file.write_text("secret")

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    symlink = project_dir / "link.txt"
    symlink.symlink_to(outside_file)

    result = path_policy.validate_path(
        "link.txt",
        path_policy.PathType.OUT,
        project_dir,
        check_exists=False,
    )

    assert result["valid"], "Should not check symlink when check_exists=False"


# --- require_valid_path tests ---


def test_require_valid_path_returns_normalized(tmp_path: pathlib.Path) -> None:
    """require_valid_path returns the normalized path on success."""
    result = path_policy.require_valid_path(
        "subdir/file.txt",
        path_policy.PathType.OUT,
        tmp_path,
    )

    assert result == tmp_path / "subdir" / "file.txt"


def test_require_valid_path_raises_on_invalid(tmp_path: pathlib.Path) -> None:
    """require_valid_path raises SecurityValidationError on invalid path."""
    with pytest.raises(exceptions.SecurityValidationError) as exc_info:
        path_policy.require_valid_path(
            "../escape.txt",
            path_policy.PathType.OUT,
            tmp_path,
        )

    assert "contains path traversal" in str(exc_info.value)


def test_require_valid_path_with_context(tmp_path: pathlib.Path) -> None:
    """require_valid_path includes context in error message."""
    with pytest.raises(exceptions.SecurityValidationError) as exc_info:
        path_policy.require_valid_path(
            "../escape.txt",
            path_policy.PathType.OUT,
            tmp_path,
            context="my_stage",
        )

    assert "my_stage:" in str(exc_info.value)


def test_require_valid_path_logs_warnings(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """require_valid_path logs warnings but still returns valid path."""
    outside_path = "/some/external/data.csv"

    result = path_policy.require_valid_path(
        outside_path,
        path_policy.PathType.DEP,
        tmp_path,
    )

    assert result == pathlib.Path(outside_path)
    assert "may break reproducibility" in caplog.text


# --- _is_within tests ---


def test_is_within_true_for_subpath(tmp_path: pathlib.Path) -> None:
    """_is_within returns True for paths within root."""
    subpath = tmp_path / "subdir" / "file.txt"
    assert path_policy._is_within(subpath, tmp_path)


def test_is_within_false_for_outside_path(tmp_path: pathlib.Path) -> None:
    """_is_within returns False for paths outside root."""
    outside = tmp_path.parent / "other"
    assert not path_policy._is_within(outside, tmp_path)


def test_is_within_true_for_root_itself(tmp_path: pathlib.Path) -> None:
    """_is_within returns True when path equals root."""
    assert path_policy._is_within(tmp_path, tmp_path)


# --- Policy coverage tests ---


def test_all_path_types_have_policies() -> None:
    """Every PathType should have a corresponding policy."""
    for path_type in path_policy.PathType:
        assert path_type in path_policy.POLICIES, f"Missing policy for {path_type}"


def test_policy_structure() -> None:
    """Policies should have required fields."""
    for path_type, policy in path_policy.POLICIES.items():
        assert "allow_absolute" in policy, f"{path_type} missing allow_absolute"
        assert "symlink_escape_action" in policy, f"{path_type} missing symlink_escape_action"
        assert policy["symlink_escape_action"] in ("error", "warn", "allow")
