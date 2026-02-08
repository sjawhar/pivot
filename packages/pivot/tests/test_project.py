"""Tests for pivot.project module."""

import pathlib

import pytest

from pivot import project


def test_normalize_path_with_custom_base(tmp_path: pathlib.Path) -> None:
    """Relative path resolved from custom base."""
    custom_base = tmp_path / "custom" / "base"
    custom_base.mkdir(parents=True)

    result = project.normalize_path("foo/bar.txt", base=custom_base)

    assert result == custom_base / "foo" / "bar.txt"


def test_normalize_path_windows_backslash(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows backslash paths are normalized to POSIX: foo\\bar -> foo/bar."""
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result = project.normalize_path("foo\\bar")

    assert result == tmp_path / "foo" / "bar"


def test_normalize_path_mixed_separators(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mixed separators are normalized: foo\\bar/baz -> foo/bar/baz."""
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result = project.normalize_path("foo\\bar/baz")

    assert result == tmp_path / "foo" / "bar" / "baz"


def test_normalize_path_preserves_symlinks(tmp_path: pathlib.Path) -> None:
    """Symlink paths are preserved, not resolved to target."""
    # Create actual directory and a symlink to it
    actual_dir = tmp_path / "actual"
    actual_dir.mkdir()
    (actual_dir / "file.txt").write_text("content")

    symlink_dir = tmp_path / "link"
    symlink_dir.symlink_to(actual_dir)

    # normalize_path should preserve the symlink path, not resolve to actual
    result = project.normalize_path("link/file.txt", base=tmp_path)

    assert result == tmp_path / "link" / "file.txt"
    # Verify we didn't resolve to target
    assert "actual" not in str(result)


def test_normalize_path_collapses_dotdot(tmp_path: pathlib.Path) -> None:
    """Parent directory references are collapsed: foo/../bar -> bar."""
    result = project.normalize_path("foo/../bar", base=tmp_path)

    assert result == tmp_path / "bar"


def test_normalize_path_absolute_path_unchanged(tmp_path: pathlib.Path) -> None:
    """Absolute paths are not affected by base parameter."""
    custom_base = tmp_path / "custom"
    custom_base.mkdir()
    absolute_path = tmp_path / "absolute" / "path.txt"

    result = project.normalize_path(absolute_path, base=custom_base)

    # Should be the absolute path (normalized), not relative to custom_base
    assert result == tmp_path / "absolute" / "path.txt"


def test_normalize_path_default_base_is_project_root(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When base is None, project root is used (preserving existing behavior)."""
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result = project.normalize_path("relative/path.txt")

    assert result == tmp_path / "relative" / "path.txt"
