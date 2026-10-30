from __future__ import annotations

import os
import pathlib  # noqa: TCH003 - used at runtime by pytest fixtures (tmp_path)
import subprocess
import threading
import time
import unicodedata
from typing import TYPE_CHECKING, Any

import pytest

from pivot import ignore
from tests import helpers

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


def test_user_level_pivotignore_is_not_read(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the project's .pivotignore counts: a ~/.pivotignore would change dependency hashes
    on one machine and not on another."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".pivotignore").write_text("*.bak\n")
    monkeypatch.setenv("HOME", str(home))
    project = tmp_path / "project"
    project.mkdir()
    (project / ".pivotignore").write_text("*.log\n")

    filter_instance = ignore.IgnoreFilter(project_root=project)

    assert filter_instance.is_ignored("app.log")
    assert not filter_instance.is_ignored("data.bak"), "~/.pivotignore must not apply"


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


def test_pipeline_py_never_ignored(tmp_path: pathlib.Path) -> None:
    """pipeline.py should never be ignored regardless of patterns."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("pipeline.py\n*.py\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    assert not filter_instance.is_ignored("pipeline.py")


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
    # git never matches "[unclosed"; "***" is a "**" that matches every name
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
# Backslashes (POSIX names, gitignore escapes)
# =============================================================================


def test_backslash_in_a_path_is_part_of_its_name(tmp_path: pathlib.Path) -> None:
    """A backslash in a checked path is part of a name, never a separator (Pivot is POSIX)."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("data/temp/*.log\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    assert filter_instance.is_ignored("data/temp/app.log")
    assert not filter_instance.is_ignored("data\\temp\\app.log")
    assert not filter_instance.is_ignored(str(tmp_path / "data\\temp\\app.log"))


def test_backslash_in_a_pattern_escapes_the_next_character_as_in_gitignore(
    tmp_path: pathlib.Path,
) -> None:
    """A pattern's backslash escapes the character after it, as in a .gitignore."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("a\\\\b.csv\n\\#notes.txt\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    assert filter_instance.is_ignored("a\\b.csv")
    assert not filter_instance.is_ignored("a/b.csv")
    assert filter_instance.is_ignored("#notes.txt")
    assert filter_instance.is_ignored("sub/#notes.txt")


def test_trailing_backslash_is_part_of_a_file_name(tmp_path: pathlib.Path) -> None:
    """A name ending in a backslash is a file, not a directory written with a separator."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("tmp*/\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    assert filter_instance.is_ignored("tmp/")
    assert not filter_instance.is_ignored("tmp\\")


def test_pattern_ending_in_a_lone_backslash_is_skipped_as_in_gitignore(
    tmp_path: pathlib.Path, caplog: LogCaptureFixture
) -> None:
    """gitignore never matches a pattern ending in a lone backslash; Pivot skips it and warns."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("build\\\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)
    with caplog.at_level("WARNING"):
        ignored = filter_instance.is_ignored("build/x.txt")

    assert not ignored
    assert "Invalid pattern on line 1 of .pivotignore: 'build\\'" in caplog.text


def test_gitignore_backslash_escapes_are_kept_in_patterns(
    tmp_path: pathlib.Path,
) -> None:
    """Patterns retain gitignore backslash escapes."""
    project_ignore = tmp_path / ".pivotignore"
    project_ignore.write_text("\\!keep\ntrailing\\ \ndata\\raw\\*\n\\#user.txt\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    assert filter_instance.is_ignored("sub/!keep")
    assert filter_instance.is_ignored("trailing ")
    assert not filter_instance.is_ignored("data/raw/file.txt")
    assert filter_instance.is_ignored("dataraw*")
    assert filter_instance.is_ignored("sub/#user.txt")


def test_unicode_paths_match_as_git_matches_them(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """git compares a path's bytes with the pattern's, so a decomposed path is another name
    than a precomposed pattern; on macOS git precomposes the path first, so Pivot does too."""
    pivotignore = tmp_path / ".pivotignore"
    pivotignore.write_text("café/*.log\n")
    nfc_path = "café/app.log"
    nfd_path = unicodedata.normalize("NFD", nfc_path)
    assert nfc_path != nfd_path, "Test setup: paths should be different encodings"

    monkeypatch.setattr(ignore, "_PRECOMPOSE", False)
    assert ignore.IgnoreFilter(project_root=tmp_path).is_ignored(nfc_path)
    assert not ignore.IgnoreFilter(project_root=tmp_path).is_ignored(nfd_path)

    monkeypatch.setattr(ignore, "_PRECOMPOSE", True)
    assert ignore.IgnoreFilter(project_root=tmp_path).is_ignored(nfd_path)


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


# =============================================================================
# gitignore's parent-directory rule
# =============================================================================


def test_negation_cannot_reinclude_a_file_inside_an_ignored_directory(
    tmp_path: pathlib.Path,
) -> None:
    """As in gitignore, a file under an ignored directory stays ignored whatever follows, and
    check_ignore names the pattern that ignores the directory."""
    (tmp_path / ".pivotignore").write_text("build/\n!build/keep.txt\n")

    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)
    result = filter_instance.check_ignore("build/keep.txt")

    assert filter_instance.is_ignored("build/keep.txt"), "build/ is ignored, so is its file"
    assert (result.ignored, result.pattern, result.source) == (True, "build/", ".pivotignore:1")


# =============================================================================
# Member filters (directory dependencies)
# =============================================================================


def test_member_filter_matches_the_members_path_inside_the_project(
    tmp_path: pathlib.Path,
) -> None:
    """A member is matched as ``<directory>/<relpath>``, so anchored patterns work."""
    (tmp_path / ".pivotignore").write_text("*.pyc\n__pycache__/\n/src/lane/scratch/\n")

    member_filter = ignore.IgnoreFilter(project_root=tmp_path).member_filter("src/lane")

    assert member_filter is not None
    assert member_filter("__pycache__", True)
    assert member_filter("x.cpython-312.pyc", False)
    assert member_filter("scratch", True), "the anchored pattern names src/lane/scratch/"
    assert not member_filter("grade.py", False)
    assert not member_filter("scratch", False), "a directory pattern names no file"


def test_member_filter_takes_an_absolute_directory_under_the_project(
    tmp_path: pathlib.Path,
) -> None:
    (tmp_path / ".pivotignore").write_text("/inputs/tmp/\n")

    member_filter = ignore.IgnoreFilter(project_root=tmp_path).member_filter(tmp_path / "inputs")

    assert member_filter is not None
    assert member_filter("tmp", True)


@pytest.mark.parametrize(
    "content",
    [pytest.param(None, id="no-file"), pytest.param("# only a comment\n\n", id="no-pattern")],
)
def test_member_filter_is_none_without_patterns(
    tmp_path: pathlib.Path, content: str | None
) -> None:
    """With nothing to leave out, the walk gets no filter and costs what it did."""
    if content is not None:
        (tmp_path / ".pivotignore").write_text(content)

    assert ignore.IgnoreFilter(project_root=tmp_path).member_filter("inputs") is None


@pytest.mark.parametrize(
    "directory",
    [pytest.param("build", id="the-directory"), pytest.param("build/gen", id="inside-it")],
)
def test_member_filter_is_none_for_an_ignored_directory_and_warns_once(
    tmp_path: pathlib.Path, caplog: LogCaptureFixture, directory: str
) -> None:
    """Leaving out every file of an input would hide every change to it, so a directory that
    .pivotignore ignores (itself or a directory above it) keeps every file, with one warning."""
    (tmp_path / ".pivotignore").write_text("*.pyc\nbuild/\n")
    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    with caplog.at_level("WARNING", logger="pivot.ignore"):
        first = filter_instance.member_filter(directory)
        second = filter_instance.member_filter(directory)

    assert (first, second) == (None, None)
    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 1, warnings
    assert f"Input directory {directory} is ignored by .pivotignore" in warnings[0]
    assert "'build/' at .pivotignore:2" in warnings[0]


def test_member_filter_never_ignores_a_protected_path(tmp_path: pathlib.Path) -> None:
    (tmp_path / ".pivotignore").write_text("*.py\n")

    member_filter = ignore.IgnoreFilter(project_root=tmp_path).member_filter(".")

    assert member_filter is not None
    assert not member_filter("pipeline.py", False), "pipeline.py is protected"
    assert member_filter("stages.py", False)


def test_member_filter_compares_names_as_bytes_as_git_does(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """git compares bytes: a decomposed name is not the precomposed pattern, except on macOS,
    where git precomposes the names it reads (core.precomposeunicode)."""
    (tmp_path / ".pivotignore").write_text("café.log\n")
    decomposed = unicodedata.normalize("NFD", "café.log")

    monkeypatch.setattr(ignore, "_PRECOMPOSE", False)
    bytewise = ignore.IgnoreFilter(project_root=tmp_path).member_filter("inputs")
    assert bytewise is not None
    assert bytewise("café.log", False)
    assert not bytewise(decomposed, False), "another byte sequence is another name"

    monkeypatch.setattr(ignore, "_PRECOMPOSE", True)
    precomposing = ignore.IgnoreFilter(project_root=tmp_path).member_filter("inputs")
    assert precomposing is not None
    assert precomposing(decomposed, False), "macOS git precomposes the name it reads"


@pytest.mark.parametrize(
    ("relpath", "is_dir", "expected"),
    [
        pytest.param("pkg/__pycache__/m.cpython-312.pyc", False, True, id="under-ignored-dir"),
        pytest.param("pkg/__pycache__", True, True, id="the-ignored-dir"),
        pytest.param("pkg/m.py", False, False, id="kept-file"),
        pytest.param("build/keep.txt", False, True, id="negation-under-ignored-dir"),
        pytest.param("pkg/m.pyc", False, True, id="ignored-file"),
    ],
)
def test_member_ignored_checks_every_directory_above(
    tmp_path: pathlib.Path, relpath: str, is_dir: bool, expected: bool
) -> None:
    """``member_ignored`` decides what the pruning walk decides for a path met on its own."""
    (tmp_path / ".pivotignore").write_text("*.pyc\n__pycache__/\nbuild/\n!build/keep.txt\n")
    member_filter = ignore.IgnoreFilter(project_root=tmp_path).member_filter("inputs")
    assert member_filter is not None

    assert ignore.member_ignored(member_filter, relpath, is_dir=is_dir) is expected


# =============================================================================
# git's per-pattern rules (wildmatch), where a looser matcher drops a file git keeps
# =============================================================================


@pytest.mark.parametrize(
    ("patterns", "kept", "ignored"),
    [
        pytest.param(
            "logs/**\n!logs/.gitkeep\n",
            ["logs/.gitkeep"],
            ["logs/today.log", "logs/old/day.log"],
            id="dir-starstar-with-negation",
        ),
        pytest.param(
            "**/foo/**\n!**/foo/keep.txt\n",
            ["inputs/foo/keep.txt", "inputs/a/foo/keep.txt"],
            ["inputs/foo/other.txt"],
            id="leading-and-trailing-starstar",
        ),
        pytest.param(
            "/inputs/foo/**\n!/inputs/foo/keep.txt\n",
            ["inputs/foo/keep.txt"],
            ["inputs/foo/x/keep.txt", "inputs/foo/other.txt"],
            id="anchored-starstar",
        ),
        pytest.param(
            "a/**/b\n", ["x/a/b", "a/c"], ["a/b", "a/x/b", "a/x/y/b"], id="middle-starstar"
        ),
        pytest.param(
            "a/**/**/**/b\n",
            ["x/a/b", "a/c"],
            ["a/b", "a/x/b", "a/x/y/b"],
            id="consecutive-starstar-components-match-as-one",
        ),
        pytest.param(
            "*.txt\na**/**/b\n!a**/**/keep.txt\n",
            ["x/a/b", "a/keep.txt", "a/x/keep.txt", "ax/y/keep.txt", "a/c"],
            ["a/b", "a/x/b", "a/x/y/b", "ax/b", "ax/y/b", "x/keep.txt", "a/x/other.txt"],
            id="non-whole-star-before-a-whole-starstar",
        ),
        pytest.param(
            "a/**\\/b\n",
            ["a/b", "x/a/b"],
            ["a/x/b", "a/x/y/b"],
            id="starstar-before-an-escaped-slash-is-a-whole-component",
        ),
        pytest.param("a**b\n", ["a/b", "a/x/b"], ["ab", "axb", "d/axxb"], id="starstar-as-star"),
        pytest.param("***\n", [], ["anything", "deep/er/file"], id="star-run-matches-all"),
        pytest.param("sub/\n!sub/\n", ["sub/a.txt"], [], id="negated-dir-pattern"),
        pytest.param("[!a]*.bin\n", ["a.bin"], ["c.bin", "d/b.bin"], id="negated-class"),
        pytest.param("[]x].txt\n", ["y.txt"], ["].txt", "x.txt"], id="bracket-first-member"),
        pytest.param("[[:digit:]]*\n", ["a1"], ["1a", "d/9"], id="named-class"),
        pytest.param("[z-a]\n", ["a"], ["z"], id="reversed-range-keeps-its-first-byte"),
        pytest.param("*.txt  \n", ["a.txt  "], ["a.txt"], id="trailing-spaces-trimmed"),
        pytest.param("*.txt\\ \n", ["a.txt"], ["a.txt "], id="escaped-trailing-space"),
        pytest.param("\\!x\n", ["x"], ["!x"], id="escaped-bang"),
        pytest.param(
            "a[[:space:]]b\n", ["a\vb", "a\fb"], ["a b", "a\tb"], id="space-class-is-gits"
        ),
        pytest.param("[a-c]\n", ["d", "ab"], ["a", "b", "c"], id="range-holds-both-ends"),
        pytest.param("**/a[!x]b\n", ["a/b", "axb"], ["ayb", "d/a b"], id="class-never-a-slash"),
        pytest.param("**/a?b\n", ["a/b", "ab"], ["ayb", "d/a b"], id="question-never-a-slash"),
        pytest.param(
            "*.txt\n!data/raw/src/docs**\n",
            ["data/raw/src/docsx.txt", "data/raw/src/docs.txt"],
            [
                "data/raw/src/docs/notes.txt",
                "data/raw/src/docs/sub/deep.txt",
                "data/raw/src/other.txt",
                "data/raw/src/sub/docs.txt",
            ],
            id="starstar-after-a-literal-prefix-is-a-star",
        ),
        pytest.param(
            "foo**/bar\n",
            ["x/foo/bar", "foo/baz", "foo/x/bar", "foobar"],
            ["foo/bar", "foox/bar"],
            id="starstar-slash-after-a-literal-prefix-is-a-star",
        ),
    ],
)
def test_patterns_decide_as_git_decides(
    tmp_path: pathlib.Path, patterns: str, kept: list[str], ignored: list[str]
) -> None:
    """Each case is git 2.52's own answer, checked here with ``git ls-files``: ``X/**`` names
    what is inside X, never X itself, so a later negation brings a file inside it back. On a
    git older than 2.52 the check against git is left out for a case holding a line that git
    matched otherwise before 2.52 (``helpers.git_differs_before_2_52``)."""
    project = tmp_path / "project"
    for relpath in [*kept, *ignored]:
        (project / relpath).parent.mkdir(parents=True, exist_ok=True)
        (project / relpath).write_text("x\n")
    (project / ".pivotignore").write_text(patterns)
    filter_instance = ignore.IgnoreFilter(project_root=project)

    if helpers.git_version() >= (2, 52) or not any(
        helpers.git_differs_before_2_52(line) for line in patterns.encode().split(b"\n")
    ):
        assert _helper_git_kept(tmp_path, project, patterns) == set(kept), "the case is git's"
    assert [path for path in kept if filter_instance.is_ignored(path, is_dir=False)] == []
    assert [path for path in ignored if not filter_instance.is_ignored(path, is_dir=False)] == []


def test_starstar_after_a_literal_prefix_is_not_a_whole_component(tmp_path: pathlib.Path) -> None:
    """git's own case, t0008 "** not confused by matching leading prefix" (git commit
    1940a02dc1, git 2.52): `foo**/bar` ignores foo/bar, not foobar. git gives wildmatch the
    literal prefix's last byte, so the `**` after `o` is a `*`; before 2.52 it gave wildmatch
    `**/bar` and `bar`, a whole `**/` matching no directory, and ignored foobar too."""
    (tmp_path / ".pivotignore").write_text("foo**/bar\n")
    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    assert [
        path for path in ("foobar", "foo/bar") if filter_instance.check_ignore(path).ignored
    ] == ["foo/bar"]


def _helper_git_kept(home: pathlib.Path, project: pathlib.Path, patterns: str) -> set[str]:
    """The files under ``project`` that git keeps with ``patterns`` as its root .gitignore: no
    global or system git configuration is read."""
    (project / ".gitignore").write_text(patterns + "/.gitignore\n/.pivotignore\n")
    env = {
        **os.environ,
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    subprocess.run(["git", "init", "-q"], cwd=project, env=env, check=True)
    listed = subprocess.run(
        ["git", "ls-files", "-o", "--exclude-standard", "-z"],
        cwd=project,
        env=env,
        check=True,
        capture_output=True,
    ).stdout
    return {os.fsdecode(path) for path in listed.split(b"\0") if path}


def test_a_pivotignore_that_is_not_utf8_is_read_as_bytes(
    tmp_path: pathlib.Path, caplog: LogCaptureFixture
) -> None:
    """git reads the file as bytes, so does Pivot: no byte makes it fail, and a byte that is not
    UTF-8 matches the same byte in a name."""
    (tmp_path / ".pivotignore").write_bytes(b"*.pyc\n\xff\xfe*.tmp\n")
    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)
    odd_name = os.fsdecode(b"\xff\xfeX.tmp")

    with caplog.at_level("WARNING", logger="pivot.ignore"):
        member_filter = filter_instance.member_filter("inputs")
        result = filter_instance.check_ignore(f"inputs/{odd_name}")

    assert member_filter is not None
    assert member_filter("x.pyc", False)
    assert member_filter(odd_name, False), "the undecodable bytes match themselves"
    assert not member_filter("X.tmp", False)
    assert (result.ignored, result.pattern, result.source) == (
        True,
        "\\xff\\xfe*.tmp",
        ".pivotignore:2",
    ), "check-ignore shows the line with its odd bytes escaped"
    assert caplog.records == []


def test_patterns_a_regex_would_backtrack_on_match_quickly(tmp_path: pathlib.Path) -> None:
    """git's wildmatch gives up as soon as a "*" cannot match by taking the rest of the text
    (WM_ABORT_ALL), so no pattern tries every way of splitting a path. A regular expression for
    the same patterns does: eight "**/" on a path 30 levels deep took 12.8 s that way, and
    `*-*-*-*-*-*.bak` on 200 names of 60 hyphen-separated parts 38 s (git: 44 and 5 ms)."""
    (tmp_path / ".pivotignore").write_text("a/**/**/**/**/**/**/**/**/z\n*-*-*-*-*-*.bak\n")
    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)
    member_filter = filter_instance.member_filter(".")
    assert member_filter is not None
    deep = "/".join(["a", *["d"] * 28, "y"])
    names = ["-".join(f"p{part}" for part in range(60)) + suffix for suffix in (".txt", "")]

    started = time.perf_counter()
    deep_ignored = ignore.member_ignored(member_filter, deep, is_dir=False)
    names_ignored = [member_filter(name, False) for name in names * 100]
    elapsed = time.perf_counter() - started

    assert not deep_ignored
    assert not any(names_ignored)
    assert member_filter("/".join(["a", *["d"] * 28, "z"]), False), "the pattern still matches"
    assert member_filter("p0-p1-p2-p3-p4-p5.bak", False)
    assert elapsed < 1.0, f"{elapsed:.2f} s for 1 deep path and 200 long names"


def test_a_long_run_of_starstar_components_matches_quickly(tmp_path: pathlib.Path) -> None:
    """256 consecutive "**/" match what one "**/" matches, and are merged into one first: kept
    as they are, each one more is another level of trying every split of the path, and one
    `repro` took 587 s on such a negation."""
    (tmp_path / ".pivotignore").write_text("a/" + "**/" * 256 + "z\n")
    member_filter = ignore.IgnoreFilter(project_root=tmp_path).member_filter(".")
    assert member_filter is not None
    deep = "/".join(["a", *["d"] * 28, "y"])

    started = time.perf_counter()
    deep_ignored = ignore.member_ignored(member_filter, deep, is_dir=False)
    elapsed = time.perf_counter() - started

    assert not deep_ignored
    assert member_filter("/".join(["a", *["d"] * 28, "z"]), False)
    assert member_filter("a/z", False)
    assert elapsed < 1.0, f"{elapsed:.2f} s for 1 deep path"


def test_a_negation_of_257_starstar_components_keeps_what_git_keeps(
    tmp_path: pathlib.Path, caplog: LogCaptureFixture
) -> None:
    """`!` and 257 "**/" before `keep.txt` is `!**/keep.txt` once they are merged: git keeps
    `sub/keep.txt`, and so does Pivot, with no warning."""
    patterns = "*.txt\n!" + "**/" * 257 + "keep.txt\n"
    project = tmp_path / "project"
    for relpath in ("keep.txt", "sub/keep.txt", "sub/other.txt"):
        (project / relpath).parent.mkdir(parents=True, exist_ok=True)
        (project / relpath).write_text("x\n")
    (project / ".pivotignore").write_text(patterns)
    filter_instance = ignore.IgnoreFilter(project_root=project)

    with caplog.at_level("WARNING", logger="pivot.ignore"):
        kept = {
            relpath
            for relpath in ("keep.txt", "sub/keep.txt", "sub/other.txt")
            if not filter_instance.is_ignored(relpath, is_dir=False)
        }

    assert kept == {"keep.txt", "sub/keep.txt"}
    assert _helper_git_kept(tmp_path / "home", project, patterns) == kept, "the case is git's"
    assert caplog.records == []


_NEGATION_WARNING = (
    "Negation on line {line} of .pivotignore ('{text}') has 257 runs of '*' that try more than "
    + "one place; Pivot matches at most 256, so it takes it to match every member and no input "
    + "directory. Shorten it."
)
_PATTERN_WARNING = (
    "Pattern on line {line} of .pivotignore ('{text}') has 257 runs of '*' that try more than "
    + "one place; Pivot matches at most 256, so it takes it to match no member and every input "
    + "directory, which is hashed whole. Shorten it."
)


def test_a_negation_too_long_to_match_matches_every_member(
    tmp_path: pathlib.Path, caplog: LogCaptureFixture
) -> None:
    """A negation with more than 256 runs of "*" is taken to match every member, with a
    warning: it can then keep a file git leaves out (a rerun too many) but never leave out one
    git keeps (a stale stage). A trailing "/" still limits it to directories."""
    long_negation = "!" + "*a" * 257
    (tmp_path / "every").mkdir()
    (tmp_path / "every" / ".pivotignore").write_text(f"*.txt\nbuild/\n{long_negation}\n")
    (tmp_path / "dirs").mkdir()
    (tmp_path / "dirs" / ".pivotignore").write_text(f"*.txt\nbuild/\n{long_negation}/\n")

    with caplog.at_level("WARNING", logger="pivot.ignore"):
        every = ignore.IgnoreFilter(project_root=tmp_path / "every")
        every_ignored = (
            every.is_ignored("a.txt", is_dir=False),
            every.is_ignored("build", is_dir=True),
        )
        dirs = ignore.IgnoreFilter(project_root=tmp_path / "dirs")
        dirs_ignored = (
            dirs.is_ignored("a.txt", is_dir=False),
            dirs.is_ignored("build", is_dir=True),
        )

    assert every_ignored == (False, False), "it matches every file and directory"
    assert dirs_ignored == (True, False), "with a trailing '/', every directory and no file"
    assert [record.getMessage() for record in caplog.records] == [
        _NEGATION_WARNING.format(line=3, text=long_negation + suffix) for suffix in ("", "/")
    ]


def test_an_over_limit_negation_unignores_no_input_directory(tmp_path: pathlib.Path) -> None:
    """`build/` ignores build and every path below it. A directory-only negation over the limit
    matches every *member*, but decides neither that an input directory nor that a directory
    above one is not ignored, so an input at build or at build/sub gets no filter and is hashed
    whole; one elsewhere still gets a filter."""
    (tmp_path / ".pivotignore").write_text("build/\n*.txt\n!" + "*a" * 257 + "/\n")
    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    assert filter_instance.member_filter("build") is None
    assert filter_instance.member_filter("build/sub") is None, "the parent build is ignored"
    assert filter_instance.member_filter("src") is not None


def test_a_pattern_too_long_to_match_matches_no_member(
    tmp_path: pathlib.Path, caplog: LogCaptureFixture
) -> None:
    """Any other line with more than 256 runs of "*" matches no member, with a warning: leaving
    it out there can only keep files git leaves out."""
    (tmp_path / ".pivotignore").write_text("*a" * 257 + "\n*.txt\n")

    with caplog.at_level("WARNING", logger="pivot.ignore"):
        filter_instance = ignore.IgnoreFilter(project_root=tmp_path)
        long_ignored = filter_instance.is_ignored("xa", is_dir=False)

    assert not long_ignored
    assert filter_instance.is_ignored("notes.txt", is_dir=False)
    assert [record.getMessage() for record in caplog.records] == [
        _PATTERN_WARNING.format(line=1, text="*a" * 257)
    ]


def test_an_over_limit_pattern_ignores_every_input_directory(tmp_path: pathlib.Path) -> None:
    """A positive line over the limit matches no member but every input directory, so every
    input is hashed whole. Dropping it there would filter an input git ignores by it: here git
    ignores the 258-byte input directory a×128/a×129 (so grade.py in it too), and `*.txt` would
    then leave out its counted.txt."""
    deep = "a" * 128 + "/" + "a" * 129
    patterns = "*a" * 128 + "/" + "*a" * 129 + "\n*.txt\n"
    project = tmp_path / "project"
    (project / deep).mkdir(parents=True)
    (project / deep / "counted.txt").write_text("x\n")
    (project / deep / "grade.py").write_text("x\n")
    (project / "kept.py").write_text("x\n")
    (project / ".pivotignore").write_text(patterns)
    filter_instance = ignore.IgnoreFilter(project_root=project)

    git_kept = _helper_git_kept(tmp_path / "home", project, patterns)
    assert "kept.py" in git_kept
    assert f"{deep}/grade.py" not in git_kept, "git ignores the input directory by the line"
    assert not filter_instance.is_ignored(deep, is_dir=True), "no member: as before"
    assert filter_instance.member_filter(deep) is None, "an input directory: hashed whole"
    assert filter_instance.member_filter("src") is None, "every input directory"


def test_the_limit_is_256_runs_of_star_that_try_more_than_one_place(
    tmp_path: pathlib.Path, caplog: LogCaptureFixture
) -> None:
    """256 runs of "*" that each try every place are matched as git matches them; 257 are not,
    and the warning quotes the line."""
    at_limit, over_limit = "*a" * 256, "*a" * 257
    for name, line in (("at", at_limit), ("over", over_limit)):
        (tmp_path / name).mkdir()
        (tmp_path / name / ".pivotignore").write_text(line + "\n")

    with caplog.at_level("WARNING", logger="pivot.ignore"):
        at = ignore.IgnoreFilter(project_root=tmp_path / "at")
        at_ignored = (at.is_ignored("a" * 256, is_dir=False), at.is_ignored("a" * 255))
        at_warnings = [record.getMessage() for record in caplog.records]
        over = ignore.IgnoreFilter(project_root=tmp_path / "over")
        over_ignored = over.is_ignored("a" * 257, is_dir=False)

    assert at_ignored == (True, False), "256 runs: matched as git matches them"
    assert at_warnings == []
    assert not over_ignored, "257 runs: it matches no member"
    assert [record.getMessage() for record in caplog.records] == [
        _PATTERN_WARNING.format(line=1, text=over_limit)
    ]


def test_runs_of_star_that_never_recurse_do_not_count(
    tmp_path: pathlib.Path, caplog: LogCaptureFixture
) -> None:
    """A lone "*" right before "/" matches the rest of one name, and a run at the end returns
    at once: neither makes wildmatch recurse, so `*/` 257 times then `x`, and a line ending in
    one more `*`, are matched as git matches them, with no warning."""
    (tmp_path / ".pivotignore").write_text("*/" * 257 + "x\nkeep/" + "*/" * 300 + "*\n")

    with caplog.at_level("WARNING", logger="pivot.ignore"):
        filter_instance = ignore.IgnoreFilter(project_root=tmp_path)
        deep_x = filter_instance.is_ignored("d/" * 257 + "x", is_dir=False)
        short_x = filter_instance.is_ignored("d/" * 256 + "x", is_dir=False)
        deep_keep = filter_instance.is_ignored("keep/" + "d/" * 300 + "f", is_dir=False)

    assert (deep_x, short_x, deep_keep) == (True, False, True)
    assert caplog.records == []


def test_a_nul_byte_ends_a_line_as_in_git(tmp_path: pathlib.Path) -> None:
    """git reads each line as a C string, so a NUL ends it: `!a.log<NUL>junk` is `!a.log`."""
    (tmp_path / ".pivotignore").write_bytes(b"*.log\n!a.log\0junk\n")
    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    assert not filter_instance.is_ignored("a.log", is_dir=False)
    assert filter_instance.is_ignored("b.log", is_dir=False)


def test_check_ignore_never_ignores_the_project_root(tmp_path: pathlib.Path) -> None:
    """The project root is never ignored, as git's own walk never tests it. This differs from
    `git check-ignore .`, which matches the root, as an empty name, against `*` (git 2.43 and
    2.52 print `.gitignore:1:*	.`), though not against `.*`."""
    for patterns in ("*\n", ".*\n"):
        (tmp_path / ".pivotignore").write_text(patterns)
        filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

        assert not filter_instance.check_ignore(".").ignored, patterns
        assert not filter_instance.check_ignore(tmp_path).ignored, patterns
        assert filter_instance.check_ignore(".hidden").ignored, patterns


def test_check_ignore_tests_a_symlink_to_a_directory_as_a_file(tmp_path: pathlib.Path) -> None:
    """git, and the hash walk, take a symlink as what it is, not as what it points to: `cache/`
    names a directory, so a link named `cache` is not ignored, whatever it links to."""
    (tmp_path / ".pivotignore").write_text("cache/\n")
    (tmp_path / "real").mkdir()
    (tmp_path / "cache").symlink_to(tmp_path / "real", target_is_directory=True)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "cache").mkdir()
    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    assert not filter_instance.check_ignore("cache").ignored
    assert filter_instance.check_ignore("sub/cache").ignored
    assert "cache" in _helper_git_kept(tmp_path / "home", tmp_path, "cache/\n"), "git keeps it"


def test_check_ignore_takes_a_relative_path_from_the_project_root(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unlike `git check-ignore`, a relative path is taken from the project root, not from the
    working directory, for matching and for whether it is a directory."""
    (tmp_path / ".pivotignore").write_text("/data/raw/src/\n")
    (tmp_path / "data" / "raw" / "src").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "data")
    filter_instance = ignore.IgnoreFilter(project_root=tmp_path)

    assert filter_instance.check_ignore("data/raw/src").ignored
    assert not filter_instance.check_ignore("raw/src").ignored
