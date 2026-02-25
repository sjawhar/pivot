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


def test_resolve_path_for_comparison_wraps_permission_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _helper_raise_permission(_: str) -> pathlib.Path:
        raise PermissionError("no access")

    monkeypatch.setattr(project, "resolve_path", _helper_raise_permission)

    with pytest.raises(PermissionError, match="input 'data.csv'"):
        project.resolve_path_for_comparison("data.csv", "input")


def test_resolve_path_for_comparison_falls_back_for_stage_output_file_missing(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _helper_raise_not_found(_: str) -> pathlib.Path:
        raise FileNotFoundError()

    monkeypatch.setattr(project, "resolve_path", _helper_raise_not_found)

    result = project.resolve_path_for_comparison("outs/model.pkl", "stage output")
    assert result == project.normalize_path("outs/model.pkl")


def test_resolve_path_for_comparison_reraises_not_found_for_non_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _helper_raise_not_found(_: str) -> pathlib.Path:
        raise FileNotFoundError()

    monkeypatch.setattr(project, "resolve_path", _helper_raise_not_found)

    with pytest.raises(FileNotFoundError):
        project.resolve_path_for_comparison("deps/input.csv", "dependency")


def test_try_resolve_path_returns_none_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def _helper_raise_oserror(_: str) -> pathlib.Path:
        raise OSError("bad filesystem")

    monkeypatch.setattr(project, "resolve_path", _helper_raise_oserror)

    assert project.try_resolve_path("data.csv") is None


def test_to_relative_path_returns_input_for_relative(tmp_path: pathlib.Path) -> None:
    result = project.to_relative_path("already/relative.txt", base=tmp_path)
    assert result == "already/relative.txt"


def test_to_absolute_path_with_base_and_absolute_passthrough(tmp_path: pathlib.Path) -> None:
    relative = project.to_absolute_path("data/file.txt", base=tmp_path)
    absolute_input = tmp_path / "already_abs.txt"
    absolute = project.to_absolute_path(str(absolute_input), base=tmp_path)

    assert relative == tmp_path / "data" / "file.txt"
    assert absolute == absolute_input
