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


# --- Tests for normalize_path() ---


def test_normalize_path_preserves_symlinks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should preserve symlinks in path (not resolve to real path)."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    # Create real directory and symlink to it
    real_dir = tmp_path / "real_data"
    real_dir.mkdir()
    symlink_dir = tmp_path / "data"
    symlink_dir.symlink_to(real_dir)

    # Create file inside the real directory
    data_file = real_dir / "input.csv"
    data_file.write_text("test")

    with contextlib.chdir(tmp_path):
        monkeypatch.setattr(project, "_project_root_cache", None)

        # normalize_path should preserve the symlink
        normalized = project.normalize_path("data/input.csv")
        assert normalized == tmp_path / "data" / "input.csv", "Should preserve symlink in path"

        # resolve_path should follow the symlink
        resolved = project.resolve_path("data/input.csv")
        assert resolved == tmp_path / "real_data" / "input.csv", (
            "resolve_path should follow symlink"
        )


def test_normalize_path_allows_outside_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should allow paths outside project root (caller validates if needed)."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    # Create symlink pointing outside project
    outside_dir = tmp_path.parent / "outside_project"
    outside_dir.mkdir(exist_ok=True)
    symlink = tmp_path / "link_to_outside"
    symlink.symlink_to(outside_dir)

    with contextlib.chdir(tmp_path):
        monkeypatch.setattr(project, "_project_root_cache", None)

        # Should not raise - returns normalized path preserving symlink
        normalized = project.normalize_path("link_to_outside/file.csv")
        assert normalized == tmp_path / "link_to_outside" / "file.csv"


def test_normalize_path_handles_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should handle absolute paths correctly."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    # Create symlink with absolute path
    real_dir = tmp_path / "real_data"
    real_dir.mkdir()
    symlink_dir = tmp_path / "data"
    symlink_dir.symlink_to(real_dir)

    with contextlib.chdir(tmp_path):
        monkeypatch.setattr(project, "_project_root_cache", None)

        # Absolute path with symlink should be preserved
        abs_path = str(tmp_path / "data" / "file.csv")
        normalized = project.normalize_path(abs_path)
        assert normalized == tmp_path / "data" / "file.csv"


def test_normalize_path_from_subdirectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should resolve paths relative to project root, not cwd."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    # Create symlink at project root
    real_dir = tmp_path / "real_data"
    real_dir.mkdir()
    symlink_dir = tmp_path / "data"
    symlink_dir.symlink_to(real_dir)

    # Work from subdirectory
    subdir = tmp_path / "src"
    subdir.mkdir()

    with contextlib.chdir(subdir):
        monkeypatch.setattr(project, "_project_root_cache", None)

        # Should resolve from project root (tmp_path), preserving symlink
        normalized = project.normalize_path("data/file.csv")
        assert normalized == tmp_path / "data" / "file.csv"


# --- Tests for contains_symlink_in_path() ---


def test_contains_symlink_detects_symlinked_directory(tmp_path: Path) -> None:
    """Should detect when path goes through symlinked directory."""
    # Create real directory and symlink to it
    real_dir = tmp_path / "real_data"
    real_dir.mkdir()
    symlink_dir = tmp_path / "data"
    symlink_dir.symlink_to(real_dir)

    # Check file inside symlinked directory
    file_path = symlink_dir / "file.csv"
    assert project.contains_symlink_in_path(file_path, tmp_path), (
        "Should detect symlinked directory"
    )


def test_contains_symlink_no_symlinks(tmp_path: Path) -> None:
    """Should return False when path contains no symlinks."""
    # Regular directory path
    regular_dir = tmp_path / "data"
    regular_dir.mkdir()
    file_path = regular_dir / "file.csv"

    assert not project.contains_symlink_in_path(file_path, tmp_path), (
        "Should return False for regular paths"
    )


def test_contains_symlink_nested_symlinks(tmp_path: Path) -> None:
    """Should detect symlinks in nested paths."""
    # Create nested structure with symlink in middle
    real_dir = tmp_path / "real_data"
    real_dir.mkdir()
    symlink_dir = tmp_path / "data"
    symlink_dir.symlink_to(real_dir)

    # Create subdirectory inside symlinked dir
    subdir = real_dir / "subdir"
    subdir.mkdir()

    # Check file in nested structure
    file_path = symlink_dir / "subdir" / "file.csv"
    assert project.contains_symlink_in_path(file_path, tmp_path), (
        "Should detect symlink in nested path"
    )


def test_contains_symlink_symlinked_file(tmp_path: Path) -> None:
    """Should detect when the file itself is a symlink."""
    # Create regular directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create real file and symlink to it
    real_file = tmp_path / "real_file.csv"
    real_file.write_text("test")
    symlink_file = data_dir / "file.csv"
    symlink_file.symlink_to(real_file)

    assert project.contains_symlink_in_path(symlink_file, tmp_path), "Should detect symlinked file"


def test_contains_symlink_stops_at_base(tmp_path: Path) -> None:
    """Should stop checking at base directory, not check base itself."""
    # Create symlink as base
    real_dir = tmp_path / "real_base"
    real_dir.mkdir()
    symlink_base = tmp_path / "base"
    symlink_base.symlink_to(real_dir)

    # Create file inside real directory
    file_path = real_dir / "file.csv"

    # Should not check the base itself
    assert not project.contains_symlink_in_path(file_path, symlink_base), (
        "Should not check base directory for symlinks"
    )


# --- Tests for resolve_path_for_comparison() ---


def test_resolve_path_for_comparison_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should resolve existing file normally."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    test_file = tmp_path / "data.csv"
    test_file.write_text("test")

    with contextlib.chdir(tmp_path):
        monkeypatch.setattr(project, "_project_root_cache", None)
        result = project.resolve_path_for_comparison("data.csv", "dependency")
        assert result == test_file.resolve()


def test_resolve_path_for_comparison_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should resolve missing files without raising (Path.resolve() doesn't raise)."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    with contextlib.chdir(tmp_path):
        monkeypatch.setattr(project, "_project_root_cache", None)
        # Missing file - still resolves (Path.resolve() doesn't raise for missing files)
        result = project.resolve_path_for_comparison("missing.csv", "dependency")
        # Returns resolved path even though file doesn't exist
        assert result == tmp_path / "missing.csv"
