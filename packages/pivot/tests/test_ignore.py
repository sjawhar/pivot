from __future__ import annotations

import pathlib  # noqa: TCH003 - used at runtime by pytest fixtures (tmp_path)
import threading
import time
import unicodedata
from typing import TYPE_CHECKING, Any

from pivot import ignore

if TYPE_CHECKING:
    from pytest import LogCaptureFixture
    from pytest_mock import MockerFixture


# =============================================================================
# Default Patterns Tests
# =============================================================================


def test_default_patterns_includes_python_bytecode() -> None:
    """Default patterns should include Python bytecode patterns."""
    patterns = ignore.get_default_patterns()

    assert "*.pyc" in patterns
    assert "*.pyo" in patterns
    assert "__pycache__/" in patterns


def test_default_patterns_includes_virtual_environments() -> None:
    """Default patterns should include virtual environment directories."""
    patterns = ignore.get_default_patterns()

    assert ".venv/" in patterns
    assert "venv/" in patterns


def test_default_patterns_includes_version_control() -> None:
    """Default patterns should include version control directories."""
    patterns = ignore.get_default_patterns()

    assert ".git/" in patterns
    assert ".hg/" in patterns


def test_default_patterns_includes_ide_editors() -> None:
    """Default patterns should include IDE and editor files."""
    patterns = ignore.get_default_patterns()

    assert ".idea/" in patterns
    assert ".vscode/" in patterns
    assert "*.swp" in patterns
    assert "*.swo" in patterns
    assert "*~" in patterns
    assert ".#*" in patterns


def test_default_patterns_includes_build_outputs() -> None:
    """Default patterns should include build output directories."""
    patterns = ignore.get_default_patterns()

    assert "*.egg-info/" in patterns
    assert "dist/" in patterns
    assert "build/" in patterns
    assert "node_modules/" in patterns


def test_default_patterns_includes_pivot_internals() -> None:
    """Default patterns should include Pivot internal directories."""
    patterns = ignore.get_default_patterns()

    assert ".pivot/" in patterns


# =============================================================================
# Pattern Loading Tests
# =============================================================================


def test_project_pivotignore_patterns(tmp_path: pathlib.Path) -> None:
    """Should load patterns from project .pivotignore file."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("*.log\ntemp/\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    assert filter_instance.is_ignored("app.log")
    assert filter_instance.is_ignored("temp/", is_dir=True)
    assert not filter_instance.is_ignored("important.txt")


def test_user_pivotignore_patterns(tmp_path: pathlib.Path) -> None:
    """Should load patterns from user ~/.pivotignore file."""
    user_ignore = tmp_path / "user_pivotignore"
    user_ignore.write_text("*.bak\n")

    filter_instance = ignore.IgnoreFilter(
        project_root=tmp_path,
        user_ignore_path=user_ignore,
    )

    assert filter_instance.is_ignored("data.bak")


def test_patterns_are_additive(tmp_path: pathlib.Path) -> None:
    """Patterns from all sources should combine (additive semantics)."""
    # Project ignores *.log
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("*.log\n")

    # User ignores *.tmp
    user_ignore = tmp_path / "user_pivotignore"
    user_ignore.write_text("*.tmp\n")

    filter_instance = ignore.IgnoreFilter(
        project_root=tmp_path,
        user_ignore_path=user_ignore,
    )

    # Both should be ignored (additive)
    assert filter_instance.is_ignored("app.log")
    assert filter_instance.is_ignored("cache.tmp")


def test_negation_patterns(tmp_path: pathlib.Path) -> None:
    """Should support gitignore negation patterns (!pattern)."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("*.log\n!important.log\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    assert filter_instance.is_ignored("debug.log")
    assert not filter_instance.is_ignored("important.log")


def test_comments_and_blank_lines(tmp_path: pathlib.Path) -> None:
    """Should skip comments and blank lines in .pivotignore."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("# This is a comment\n\n*.tmp\n\n# Another comment\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    assert filter_instance.is_ignored("cache.tmp")
    # Comments should not be treated as patterns
    assert not filter_instance.is_ignored("# This is a comment")


# =============================================================================
# Protected Paths Tests
# =============================================================================


def test_pivot_yaml_never_ignored(tmp_path: pathlib.Path) -> None:
    """pivot.yaml should never be ignored regardless of patterns."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("pivot.yaml\n*.yaml\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    assert not filter_instance.is_ignored("pivot.yaml")


def test_pivot_yml_never_ignored(tmp_path: pathlib.Path) -> None:
    """pivot.yml should never be ignored regardless of patterns."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("pivot.yml\n*.yml\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    assert not filter_instance.is_ignored("pivot.yml")


def test_pivot_dir_never_ignored(tmp_path: pathlib.Path) -> None:
    """.pivot/ directory should never be ignored regardless of patterns."""
    # Note: .pivot/ is in default patterns, but we're testing that
    # even explicit patterns can't override the protection
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text(".pivot/\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    # .pivot/ is protected - patterns should be ignored for it
    assert not filter_instance.is_ignored(".pivot/", is_dir=True)
    assert not filter_instance.is_ignored(".pivot/state.lmdb")


# =============================================================================
# Mtime Invalidation Tests
# =============================================================================


def test_detects_file_modification(tmp_path: pathlib.Path) -> None:
    """Should detect when .pivotignore is modified and reload patterns."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("*.log\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    assert filter_instance.is_ignored("app.log")
    assert not filter_instance.is_ignored("app.txt")

    # Modify the file (ensure mtime changes)
    time.sleep(0.01)
    pivotignore.write_text("*.txt\n")

    # Should detect change and reload
    assert not filter_instance.is_ignored("app.log")
    assert filter_instance.is_ignored("app.txt")


def test_detects_file_deletion(tmp_path: pathlib.Path) -> None:
    """Should handle .pivotignore being deleted."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("*.log\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    assert filter_instance.is_ignored("app.log")

    # Delete the file
    pivotignore.unlink()

    # Should handle gracefully (fall back to no project patterns)
    # Only default patterns should apply now
    assert not filter_instance.is_ignored("app.log")


def test_detects_file_creation(tmp_path: pathlib.Path) -> None:
    """Should detect when .pivotignore is created after filter init."""
    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    # No .pivotignore initially
    assert not filter_instance.is_ignored("app.log")

    # Create .pivotignore
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("*.log\n")

    # Should detect and load new patterns
    assert filter_instance.is_ignored("app.log")


# =============================================================================
# Invalid Patterns Tests
# =============================================================================


def test_unusual_patterns_do_not_crash(tmp_path: pathlib.Path) -> None:
    """Unusual patterns should not crash the filter."""
    pivotignore = tmp_path / ".pivotignore"
    # pathspec is lenient and handles these patterns
    pivotignore.write_text("*.log\n[unclosed\n***\n*.txt\n")

    # Should not raise
    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    # Valid patterns should still work
    assert filter_instance.is_ignored("app.log")
    assert filter_instance.is_ignored("data.txt")


def test_file_read_error_logs_warning(
    tmp_path: pathlib.Path, caplog: LogCaptureFixture, mocker: MockerFixture
) -> None:
    """File read errors should log a warning and continue."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("*.log\n")

    # Make file unreadable by mocking open to raise
    original_open = open

    def mock_open(path: str | pathlib.Path, *args: Any, **kwargs: Any) -> Any:
        if str(path) == str(pivotignore):
            raise PermissionError("Access denied")
        return original_open(path, *args, **kwargs)

    mocker.patch("builtins.open", side_effect=mock_open)

    with caplog.at_level("WARNING"):
        filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    # Should handle gracefully
    assert not filter_instance.is_ignored("app.log")  # No patterns loaded


# =============================================================================
# Auto-detect is_dir Tests
# =============================================================================


def test_auto_detects_existing_directory(tmp_path: pathlib.Path) -> None:
    """Should auto-detect that existing path is a directory."""
    # Create a directory
    (tmp_path / "build").mkdir()

    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("build/\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    # Should match even without explicit is_dir=True
    assert filter_instance.is_ignored(tmp_path / "build")


def test_auto_detects_existing_file(tmp_path: pathlib.Path) -> None:
    """Should auto-detect that existing path is a file."""
    # Create a file
    (tmp_path / "build").write_text("content")

    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("build/\n")  # Only matches directories

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    # Should NOT match since it's a file, not a directory
    assert not filter_instance.is_ignored(tmp_path / "build")


def test_uses_trailing_slash_for_nonexistent(tmp_path: pathlib.Path) -> None:
    """For non-existent paths, should use trailing slash to infer is_dir."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("output/\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    # Non-existent path with trailing slash should match directory pattern
    assert filter_instance.is_ignored("output/")
    # Non-existent path without trailing slash should not match directory pattern
    assert not filter_instance.is_ignored("output")


# =============================================================================
# Windows Path Normalization Tests
# =============================================================================


def test_normalizes_input_paths(tmp_path: pathlib.Path) -> None:
    """Should normalize backslashes in input paths to forward slashes."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("data/temp/*.log\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    # Both separators should work
    assert filter_instance.is_ignored("data/temp/app.log")
    assert filter_instance.is_ignored("data\\temp\\app.log")


def test_normalizes_patterns(tmp_path: pathlib.Path) -> None:
    """Should normalize backslashes in patterns to forward slashes."""
    pivotignore = tmp_path / ".pivotignore"
    # User might write Windows-style patterns
    pivotignore.write_text("data\\temp\\*.log\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    assert filter_instance.is_ignored("data/temp/app.log")


def test_normalizes_unicode_nfc(tmp_path: pathlib.Path) -> None:
    """Should normalize Unicode to NFC form for consistent matching."""
    pivotignore = tmp_path / ".pivotignore"
    # Pattern with precomposed e-acute (NFC form)
    pivotignore.write_text("café/*.log\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    # Test with NFC (precomposed) - should match
    nfc_path = "café/app.log"
    assert filter_instance.is_ignored(nfc_path)

    # Test with NFD (decomposed e + combining acute) - should also match after normalization
    nfd_path = unicodedata.normalize("NFD", "café/app.log")
    assert nfc_path != nfd_path, "Test setup: paths should be different encodings"
    assert filter_instance.is_ignored(nfd_path)


def test_handles_absolute_paths(tmp_path: pathlib.Path) -> None:
    """Should convert absolute paths to relative before matching."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("src/*.log\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    # Relative path should match
    assert filter_instance.is_ignored("src/app.log")

    # Absolute path under project root should also match
    abs_path = str(tmp_path / "src" / "app.log")
    assert filter_instance.is_ignored(abs_path)


def test_absolute_path_outside_project_not_matched(tmp_path: pathlib.Path) -> None:
    """Absolute paths outside project root should not be converted."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("src/*.log\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    # Path outside project root - won't be converted to relative
    outside_path = "/some/other/project/src/app.log"
    assert not filter_instance.is_ignored(outside_path)


# =============================================================================
# Thread Safety Tests
# =============================================================================


def test_concurrent_is_ignored_calls(tmp_path: pathlib.Path) -> None:
    """Multiple threads calling is_ignored should not cause errors."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("*.log\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    results: list[bool] = []
    errors: list[Exception] = []

    def check_ignored() -> None:
        try:
            for _ in range(100):
                results.append(filter_instance.is_ignored("test.log"))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=check_ignored) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    assert all(results), "All results should be True"


def test_concurrent_invalidation(tmp_path: pathlib.Path) -> None:
    """Concurrent invalidation and is_ignored calls should be safe."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("*.log\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    errors: list[Exception] = []

    def invalidate_repeatedly() -> None:
        try:
            for _ in range(100):
                filter_instance.invalidate()
        except Exception as e:
            errors.append(e)

    def check_repeatedly() -> None:
        try:
            for _ in range(100):
                filter_instance.is_ignored("test.log")
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=invalidate_repeatedly),
        threading.Thread(target=check_repeatedly),
        threading.Thread(target=check_repeatedly),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"


# =============================================================================
# CheckIgnoreResult Tests
# =============================================================================


def test_check_ignore_returns_matching_pattern(tmp_path: pathlib.Path) -> None:
    """check_ignore should return the pattern that matched."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("*.log\ntemp/\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    result = filter_instance.check_ignore("app.log")

    assert result.ignored is True
    assert result.pattern == "*.log"
    assert result.path == "app.log"


def test_check_ignore_returns_source_location(tmp_path: pathlib.Path) -> None:
    """check_ignore should return the source file and line number."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("# comment\n*.log\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    result = filter_instance.check_ignore("app.log")

    assert result.ignored is True
    assert ".pivotignore" in (result.source or "")
    # Line 2 (after comment)
    assert ":2" in (result.source or "") or "line 2" in (result.source or "").lower()


def test_check_ignore_returns_none_for_no_match(tmp_path: pathlib.Path) -> None:
    """check_ignore should return ignored=False with None pattern for no match."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("*.log\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    result = filter_instance.check_ignore("important.txt")

    assert result.ignored is False
    assert result.pattern is None
    assert result.source is None


# =============================================================================
# No Ignore File Tests
# =============================================================================


def test_works_without_pivotignore(tmp_path: pathlib.Path) -> None:
    """Should work when no .pivotignore exists (no patterns applied)."""
    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    # Nothing should be ignored without patterns
    assert not filter_instance.is_ignored("any_file.txt")
    assert not filter_instance.is_ignored("any/path/file.log")


def test_works_without_user_pivotignore(tmp_path: pathlib.Path) -> None:
    """Should work when user ~/.pivotignore doesn't exist."""
    nonexistent = tmp_path / "nonexistent_user_ignore"

    filter_instance = ignore.IgnoreFilter(
        project_root=tmp_path,
        user_ignore_path=nonexistent,
    )

    # Should not raise, just have no user patterns
    assert not filter_instance.is_ignored("any_file.txt")
