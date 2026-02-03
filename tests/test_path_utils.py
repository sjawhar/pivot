"""Tests for path_utils module."""

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
