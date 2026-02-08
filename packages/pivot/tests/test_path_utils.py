"""Tests for path_utils module."""

import os
from pathlib import Path

from pivot import path_utils


def test_preserve_trailing_slash_with_slash() -> None:
    assert path_utils.preserve_trailing_slash("foo/", "foo") == "foo/"


def test_preserve_trailing_slash_without_slash() -> None:
    assert path_utils.preserve_trailing_slash("foo", "foo") == "foo"


def test_preserve_trailing_slash_already_has_slash() -> None:
    assert path_utils.preserve_trailing_slash("foo/", "foo/") == "foo/"


def test_preserve_trailing_slash_with_double_slash() -> None:
    """preserve_trailing_slash handles paths with multiple trailing slashes."""
    # Should preserve single slash even if original has multiple
    assert path_utils.preserve_trailing_slash("foo//", "foo") == "foo/"


def test_preserve_trailing_slash_normalized_empty() -> None:
    """preserve_trailing_slash handles empty normalized path."""
    # Empty normalized should get slash if original had it
    assert path_utils.preserve_trailing_slash("foo/", "") == "/"


def test_preserve_trailing_slash_both_empty() -> None:
    """preserve_trailing_slash handles both paths being empty."""
    assert path_utils.preserve_trailing_slash("", "") == ""


def test_preserve_trailing_slash_original_slash_normalized_has_slash() -> None:
    """preserve_trailing_slash is idempotent when normalized already has slash."""
    # If normalized already has slash, don't add another
    assert path_utils.preserve_trailing_slash("bar/", "foo/") == "foo/"


def test_canonicalize_artifact_path_relative(tmp_path: Path) -> None:
    """Relative path becomes absolute from base."""
    result = path_utils.canonicalize_artifact_path("data/input.csv", tmp_path)
    assert result == str(tmp_path / "data" / "input.csv")
    assert os.path.isabs(result)


def test_canonicalize_artifact_path_absolute(tmp_path: Path) -> None:
    """Absolute path stays absolute, gets normalized."""
    abs_input = str(tmp_path / "data" / ".." / "data" / "input.csv")
    result = path_utils.canonicalize_artifact_path(abs_input, tmp_path)
    assert result == str(tmp_path / "data" / "input.csv")


def test_canonicalize_artifact_path_trailing_slash(tmp_path: Path) -> None:
    """Trailing slash preserved for directory paths."""
    result = path_utils.canonicalize_artifact_path("outputs/", tmp_path)
    assert result.endswith("/")
    assert result == str(tmp_path / "outputs") + "/"


def test_canonicalize_artifact_path_no_trailing_slash(tmp_path: Path) -> None:
    """Non-directory paths don't get trailing slash."""
    result = path_utils.canonicalize_artifact_path("data/input.csv", tmp_path)
    assert not result.endswith("/")


def test_canonicalize_artifact_path_dotdot_normalized(tmp_path: Path) -> None:
    """Parent traversal is collapsed."""
    result = path_utils.canonicalize_artifact_path("sub/../data/input.csv", tmp_path)
    assert result == str(tmp_path / "data" / "input.csv")
