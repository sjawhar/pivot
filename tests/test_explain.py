"""Tests for pivot.explain module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygtrie
import pytest

from pivot import explain
from pivot.storage import cache

if TYPE_CHECKING:
    import pathlib


# =============================================================================
# hash_artifact() Tests
# =============================================================================


def test_hash_artifact_existing_file(tmp_path: pathlib.Path) -> None:
    """Hash existing file from disk."""
    f = tmp_path / "data.txt"
    f.write_text("content")

    result = explain.hash_artifact(f)

    assert result is not None
    assert result["source"] == "disk"
    assert result["hash"] == cache.hash_file(f)


def test_hash_artifact_missing_file_no_fallback(tmp_path: pathlib.Path) -> None:
    """Missing file without allow_missing returns None."""
    f = tmp_path / "missing.txt"

    result = explain.hash_artifact(f)

    assert result is None


def test_hash_artifact_missing_file_with_pvt_fallback(tmp_path: pathlib.Path) -> None:
    """Missing file with allow_missing uses tracked hash from .pvt."""
    f = tmp_path / "missing.txt"
    tracked_trie: pygtrie.Trie[str] = pygtrie.Trie()
    tracked_trie[f.parts] = "abc123"

    result = explain.hash_artifact(f, allow_missing=True, tracked_trie=tracked_trie)

    assert result is not None
    assert result["source"] == "pvt"
    assert result["hash"] == "abc123"


def test_hash_artifact_missing_no_tracked_returns_none(tmp_path: pathlib.Path) -> None:
    """Missing file not in tracked trie returns None even with allow_missing."""
    f = tmp_path / "missing.txt"
    tracked_trie: pygtrie.Trie[str] = pygtrie.Trie()
    # Don't add f to trie

    result = explain.hash_artifact(f, allow_missing=True, tracked_trie=tracked_trie)

    assert result is None


def test_hash_artifact_existing_directory(tmp_path: pathlib.Path) -> None:
    """Hash existing directory from disk."""
    d = tmp_path / "dir"
    d.mkdir()
    (d / "a.txt").write_text("aaa")
    (d / "b.txt").write_text("bbb")

    result = explain.hash_artifact(d)

    assert result is not None
    assert result["source"] == "disk"
    dir_hash, _ = cache.hash_directory(d)
    assert result["hash"] == dir_hash


def test_hash_artifact_allow_missing_false_no_trie(tmp_path: pathlib.Path) -> None:
    """Missing file with allow_missing=False and no trie returns None."""
    f = tmp_path / "missing.txt"

    result = explain.hash_artifact(f, allow_missing=False, tracked_trie=None)

    assert result is None


def test_hash_artifact_allow_missing_true_no_trie(tmp_path: pathlib.Path) -> None:
    """Missing file with allow_missing=True but no trie returns None."""
    f = tmp_path / "missing.txt"

    result = explain.hash_artifact(f, allow_missing=True, tracked_trie=None)

    assert result is None


def test_hash_artifact_existing_file_ignores_trie(tmp_path: pathlib.Path) -> None:
    """Existing file uses disk hash, not trie value."""
    f = tmp_path / "data.txt"
    f.write_text("real content")
    tracked_trie: pygtrie.Trie[str] = pygtrie.Trie()
    tracked_trie[f.parts] = "stale_hash_from_pvt"

    result = explain.hash_artifact(f, allow_missing=True, tracked_trie=tracked_trie)

    assert result is not None
    assert result["source"] == "disk"
    assert result["hash"] == cache.hash_file(f)
    assert result["hash"] != "stale_hash_from_pvt"


def test_hash_artifact_unreadable_file_falls_back_to_pvt(tmp_path: pathlib.Path) -> None:
    """Unreadable file with allow_missing uses tracked hash from .pvt."""
    f = tmp_path / "unreadable.txt"
    f.write_text("content")
    f.chmod(0o000)  # Make unreadable

    # Skip if we can still read (e.g., running as root)
    try:
        f.read_bytes()
        f.chmod(0o644)  # Restore for cleanup
        pytest.skip("Cannot test PermissionError - file still readable (may be running as root)")
    except PermissionError:
        pass  # Good, we can test the fallback

    tracked_trie: pygtrie.Trie[str] = pygtrie.Trie()
    tracked_trie[f.parts] = "fallback_hash"

    try:
        result = explain.hash_artifact(f, allow_missing=True, tracked_trie=tracked_trie)

        assert result is not None, "Should fall back to pvt when file unreadable"
        assert result["source"] == "pvt"
        assert result["hash"] == "fallback_hash"
    finally:
        f.chmod(0o644)  # Restore permissions for cleanup


def test_hash_artifact_unreadable_file_no_fallback_returns_none(tmp_path: pathlib.Path) -> None:
    """Unreadable file without allow_missing returns None."""
    f = tmp_path / "unreadable.txt"
    f.write_text("content")
    f.chmod(0o000)

    # Skip if we can still read (e.g., running as root)
    try:
        f.read_bytes()
        f.chmod(0o644)
        pytest.skip("Cannot test PermissionError - file still readable (may be running as root)")
    except PermissionError:
        pass

    try:
        result = explain.hash_artifact(f, allow_missing=False)
        assert result is None, "Unreadable file without fallback should return None"
    finally:
        f.chmod(0o644)
