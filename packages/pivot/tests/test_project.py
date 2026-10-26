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


def _helper_linked_project(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """A project at ``<tmp>/project`` holding ``data/d``, and a symlink ``<tmp>/link`` to it."""
    root = tmp_path / "project"
    (root / "data" / "d").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    return root, link


def test_relative_within_takes_a_path_typed_through_a_symlink_to_the_root(
    tmp_path: pathlib.Path,
) -> None:
    """A path under a symlink to the root is placed under the root, members not yet on disk
    included."""
    root, link = _helper_linked_project(tmp_path)

    assert project.relative_within(link / "data" / "d" / "new.csv", root) == pathlib.Path(
        "data/d/new.csv"
    )
    assert project.relative_within(link, root) == pathlib.Path(".")
    assert project.to_relative_path(link / "data" / "d", root) == "data/d"


def test_relative_within_takes_a_path_typed_through_a_symlink_into_the_root(
    tmp_path: pathlib.Path,
) -> None:
    """A symlink outside the root to a directory inside it places a path at that directory."""
    root, _ = _helper_linked_project(tmp_path)
    into = tmp_path / "into"
    into.symlink_to(root / "data", target_is_directory=True)

    assert project.relative_within(into / "d" / "x.csv", root) == pathlib.Path("data/d/x.csv")


def test_relative_within_keeps_a_symlink_inside_the_root_unresolved(
    tmp_path: pathlib.Path,
) -> None:
    """A symlink inside the root keeps its own path, as normalize_path keeps it."""
    root, link = _helper_linked_project(tmp_path)
    (root / "alias").symlink_to(root / "data", target_is_directory=True)

    assert project.relative_within(link / "alias" / "d", root) == pathlib.Path("alias/d")
    assert project.relative_within(root / "alias" / "d", root) == pathlib.Path("alias/d")


@pytest.mark.parametrize(
    "outside",
    [
        pytest.param("outside.csv", id="beside-the-root"),
        pytest.param("link/../outside.csv", id="dotdot-through-the-link"),
        pytest.param("link/da\0ta", id="nul-byte"),
    ],
)
def test_relative_within_refuses_a_path_outside_the_root(
    tmp_path: pathlib.Path, outside: str
) -> None:
    root, _ = _helper_linked_project(tmp_path)
    (tmp_path / "outside.csv").write_text("x")

    assert project.relative_within(tmp_path / outside, root) is None


def test_relative_within_refuses_traversal_through_a_symlink_inside_the_root(
    tmp_path: pathlib.Path,
) -> None:
    """A lexically contained path can still leave through an in-project symlink and ``..``."""
    root, _ = _helper_linked_project(tmp_path)
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)

    assert project.relative_within(root / "escape" / ".." / "outside.csv", root) is None
