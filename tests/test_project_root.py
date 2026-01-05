"""Tests for project root detection.

Tests the ability to find project root by searching for .pivot or .git directories,
and resolve paths relative to the project root.

Logic: Stop at first directory (walking up from cwd) that contains either .pivot or .git.
"""

import contextlib
from pathlib import Path

import pytest

from pivot import project

# --- Tests for find_project_root() ---


@pytest.mark.parametrize(
    ("directories", "work_dir", "expected_root"),
    [
        # Find marker from project root
        ([".git"], ".", None),
        ([".pivot"], ".", None),
        # Find marker from subdirectory
        ([".git", "src/pivot/nested"], "src/pivot/nested", None),
        ([".pivot", "src/pivot/nested"], "src/pivot/nested", None),
        # Stop at closer marker
        ([".git", "subproject/.pivot"], "subproject", "subproject"),
        # Both marker types in same directory
        ([".git", ".pivot"], ".", None),
        # No markers - fallback to cwd
        (["no_markers"], "no_markers", "no_markers"),
        # Nested repositories - stop at inner
        ([".git", "inner/.git"], "inner", "inner"),
    ],
)
def test_find_project_root(
    tmp_path: Path, directories: list[str], work_dir: str, expected_root: str | None
) -> None:
    """Should find project root by walking up to marker directory."""
    for directory in directories:
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)

    with contextlib.chdir(tmp_path / work_dir):
        root = project.find_project_root()
        expected = tmp_path if expected_root is None else tmp_path / expected_root
        assert root == expected


def test_find_git_root_at_filesystem_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Should handle reaching filesystem root without finding .git."""
    monkeypatch.setattr(Path, "cwd", lambda: Path("/"))

    root = project.find_project_root()
    assert root == Path("/"), "Should return root (fallback to cwd)"


# --- Tests for get_project_root() caching ---


def test_get_project_root_caches_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should cache result after first call."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    with contextlib.chdir(tmp_path):
        monkeypatch.setattr(project, "_project_root_cache", None)

        root1 = project.get_project_root()
        assert root1 == tmp_path

        root2 = project.get_project_root()
        assert root2 == tmp_path
        assert root2 is root1, "Should return same cached object"


def test_get_project_root_respects_cached_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should return cached value even if cwd changes."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    subdir = tmp_path / "subdir"
    subdir.mkdir()

    with contextlib.chdir(tmp_path):
        monkeypatch.setattr(project, "_project_root_cache", None)
        root1 = project.get_project_root()

    with contextlib.chdir(subdir):
        root2 = project.get_project_root()
        assert root2 == tmp_path, "Should return cached root, not re-search"
        assert root2 is root1


# --- Tests for resolve_path() ---


@pytest.mark.parametrize(
    ("input_path", "expected_relative"),
    [
        # Relative paths
        ("data/input.csv", "data/input.csv"),
        # Paths with parent references
        ("data/../models/model.pkl", "models/model.pkl"),
        # Paths with redundant slashes
        ("data//input.csv", "data/input.csv"),
        # Empty path
        ("", "."),
        # Dot path
        (".", "."),
    ],
)
def test_resolve_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, input_path: str, expected_relative: str
) -> None:
    """Should resolve paths relative to project root, not cwd."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    # Work from subdirectory to prove resolution is from project root
    subdir = tmp_path / "src" / "pivot"
    subdir.mkdir(parents=True)

    with contextlib.chdir(subdir):
        monkeypatch.setattr(project, "_project_root_cache", None)

        resolved = project.resolve_path(input_path)
        expected = (tmp_path / expected_relative).resolve()
        assert resolved == expected


def test_resolve_absolute_path_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should return absolute paths unchanged (already absolute)."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    with contextlib.chdir(tmp_path):
        monkeypatch.setattr(project, "_project_root_cache", None)

        absolute_path = "/tmp/output.csv"
        resolved = project.resolve_path(absolute_path)
        assert resolved == Path(absolute_path).resolve()
