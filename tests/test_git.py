from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from pivot import git, project

if TYPE_CHECKING:
    import pathlib

    import pytest


# =============================================================================
# read_file_from_head Tests
# =============================================================================


def test_read_file_from_head_no_git_repo(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns None when not in a git repo."""
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result = git.read_file_from_head("somefile.txt")

    assert result is None


def test_read_file_from_head_file_not_in_head(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns None when file doesn't exist in HEAD."""
    # Create a git repo with initial commit
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True
    )
    (tmp_path / "other.txt").write_text("content")
    subprocess.run(["git", "add", "other.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True
    )

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result = git.read_file_from_head("nonexistent.txt")

    assert result is None


def test_read_file_from_head_returns_content(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns file content from HEAD."""
    # Create a git repo with committed file
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True
    )
    (tmp_path / "file.txt").write_text("committed content")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True
    )

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result = git.read_file_from_head("file.txt")

    assert result == b"committed content"


def test_read_file_from_head_uncommitted_changes_not_visible(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns committed content, not uncommitted changes."""
    # Create a git repo with committed file
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True
    )
    (tmp_path / "file.txt").write_text("original content")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True
    )

    # Modify file but don't commit
    (tmp_path / "file.txt").write_text("modified content")

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result = git.read_file_from_head("file.txt")

    assert result == b"original content"


def test_read_file_from_head_subdirectory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Can read files in subdirectories."""
    # Create a git repo with file in subdirectory
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True
    )
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.txt").write_text("nested content")
    subprocess.run(
        ["git", "add", "subdir/nested.txt"], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True
    )

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result = git.read_file_from_head("subdir/nested.txt")

    assert result == b"nested content"


def test_read_file_from_head_empty_repo(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns None for empty repo (no commits)."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result = git.read_file_from_head("file.txt")

    assert result is None


# =============================================================================
# read_files_from_head Tests
# =============================================================================


def test_read_files_from_head_empty_list(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns empty dict for empty path list."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result = git.read_files_from_head([])

    assert result == {}


def test_read_files_from_head_multiple_files(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns content for multiple files."""
    # Create a git repo with multiple files
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True
    )
    (tmp_path / "file1.txt").write_text("content1")
    (tmp_path / "file2.txt").write_text("content2")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True
    )

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result = git.read_files_from_head(["file1.txt", "file2.txt", "missing.txt"])

    assert result == {"file1.txt": b"content1", "file2.txt": b"content2"}


def test_read_files_from_head_no_git_repo(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns empty dict when not in a git repo."""
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result = git.read_files_from_head(["file.txt"])

    assert result == {}


# =============================================================================
# resolve_revision Tests
# =============================================================================


def test_resolve_revision_no_git_repo(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns None when not in a git repo."""
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result = git.resolve_revision("HEAD")

    assert result is None


def test_resolve_revision_with_branch(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolves branch name to commit SHA."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True
    )
    (tmp_path / "file.txt").write_text("content")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True
    )

    # Get actual commit SHA
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    expected_sha = result.stdout.strip()

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    sha = git.resolve_revision("master")
    # master might be main on some systems, try both
    if sha is None:
        sha = git.resolve_revision("main")

    assert sha is not None
    assert sha == expected_sha


def test_resolve_revision_with_short_sha(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolves short SHA prefix to full commit SHA."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True
    )
    (tmp_path / "file.txt").write_text("content")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True
    )

    # Get actual commit SHA
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    full_sha = result.stdout.strip()
    short_sha = full_sha[:7]

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    sha = git.resolve_revision(short_sha)

    assert sha is not None
    assert sha == full_sha


def test_resolve_revision_invalid(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns None for invalid revision."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True
    )
    (tmp_path / "file.txt").write_text("content")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True
    )

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result = git.resolve_revision("nonexistent-branch")

    assert result is None


# =============================================================================
# read_file_from_revision Tests
# =============================================================================


def test_read_file_from_revision_returns_content(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns file content from specified revision."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True
    )
    (tmp_path / "file.txt").write_text("original content")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "first"], cwd=tmp_path, check=True, capture_output=True)

    # Get first commit SHA
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    first_sha = result.stdout.strip()[:7]

    # Make second commit with modified content
    (tmp_path / "file.txt").write_text("modified content")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "second"], cwd=tmp_path, check=True, capture_output=True)

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    # Read from first commit
    content = git.read_file_from_revision("file.txt", first_sha)

    assert content == b"original content"


def test_read_file_from_revision_file_not_found(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns None when file doesn't exist at revision."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True
    )
    (tmp_path / "file.txt").write_text("content")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True
    )

    # Get SHA
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    sha = result.stdout.strip()[:7]

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    content = git.read_file_from_revision("nonexistent.txt", sha)

    assert content is None


def test_read_file_from_revision_invalid_rev(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns None for invalid revision."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True
    )
    (tmp_path / "file.txt").write_text("content")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True
    )

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    content = git.read_file_from_revision("file.txt", "invalid-rev")

    assert content is None


# =============================================================================
# read_files_from_revision Tests
# =============================================================================


def test_read_files_from_revision_multiple_files(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns content for multiple files from revision."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True
    )
    (tmp_path / "file1.txt").write_text("content1")
    (tmp_path / "file2.txt").write_text("content2")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True
    )

    # Get SHA
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    sha = result.stdout.strip()[:7]

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result_files = git.read_files_from_revision(["file1.txt", "file2.txt", "missing.txt"], sha)

    assert result_files == {"file1.txt": b"content1", "file2.txt": b"content2"}


def test_read_files_from_revision_empty_list(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns empty dict for empty path list."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result = git.read_files_from_revision([], "HEAD")

    assert result == {}
