"""Tests for skip.check_stage — unified skip detection."""

from __future__ import annotations

from typing import Any

from pivot import skip
from pivot.types import ArtifactIdentity, DirHash, FileHash, HashInfo, LockData


def _id(producer: str, key: str | None = None) -> ArtifactIdentity:
    return ArtifactIdentity(producer, key)


def _helper_make_lock_data(
    *,
    code_manifest: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    dep_hashes: dict[ArtifactIdentity, HashInfo] | None = None,
    output_hashes: dict[ArtifactIdentity, HashInfo] | None = None,
) -> LockData:
    return LockData(
        code_manifest=code_manifest or {"func:main": "abc123"},
        params=params or {"lr": 0.01},
        dep_hashes=dep_hashes or {_id("input"): FileHash(hash="hash_a")},
        output_hashes=output_hashes or {_id("stage", "output"): FileHash(hash="hash_out")},
    )


# =============================================================================
# No lock data (first run)
# =============================================================================


def test_no_lock_data_returns_changed() -> None:
    result = skip.check_stage(
        lock_data=None,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={_id("input"): FileHash(hash="hash_a")},
        out_paths=[_id("stage", "output")],
    )
    assert result["changed"] is True
    assert "No previous run" in result["reason"]


# =============================================================================
# Fast mode: short-circuit behavior
# =============================================================================


def test_fast_code_changed() -> None:
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "DIFFERENT"},
        params={"lr": 0.01},
        dep_hashes={_id("input"): FileHash(hash="hash_a")},
        out_paths=[_id("stage", "output")],
    )
    assert result["changed"] is True
    assert "Code changed" in result["reason"]


def test_fast_params_changed() -> None:
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.99},
        dep_hashes={_id("input"): FileHash(hash="hash_a")},
        out_paths=[_id("stage", "output")],
    )
    assert result["changed"] is True
    assert "Params changed" in result["reason"]


def test_fast_deps_changed() -> None:
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={_id("input"): FileHash(hash="DIFFERENT")},
        out_paths=[_id("stage", "output")],
    )
    assert result["changed"] is True
    assert "dependencies changed" in result["reason"]


def test_fast_out_paths_changed() -> None:
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={_id("input"): FileHash(hash="hash_a")},
        out_paths=[_id("stage", "output"), _id("stage", "new_output")],
    )
    assert result["changed"] is True
    assert "Output paths changed" in result["reason"]


def test_fast_nothing_changed() -> None:
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={_id("input"): FileHash(hash="hash_a")},
        out_paths=[_id("stage", "output")],
    )
    assert result["changed"] is False
    assert result["reason"] == ""


def test_fast_force_returns_changed() -> None:
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={_id("input"): FileHash(hash="hash_a")},
        out_paths=[_id("stage", "output")],
        force=True,
    )
    assert result["changed"] is True
    assert "forced" in result["reason"]


# =============================================================================
# Explain mode: exhaustive comparisons
# =============================================================================


def test_explain_returns_all_changes() -> None:
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "DIFFERENT"},
        params={"lr": 0.99},
        dep_hashes={_id("input"): FileHash(hash="DIFFERENT")},
        out_paths=[_id("stage", "output")],
        explain=True,
    )
    assert result["changed"] is True
    assert len(result.get("code_changes", [])) > 0
    assert len(result.get("param_changes", [])) > 0
    assert len(result.get("dep_changes", [])) > 0


def test_explain_nothing_changed() -> None:
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={_id("input"): FileHash(hash="hash_a")},
        out_paths=[_id("stage", "output")],
        explain=True,
    )
    assert result["changed"] is False
    assert result.get("code_changes", []) == []
    assert result.get("param_changes", []) == []
    assert result.get("dep_changes", []) == []


def test_explain_force_with_no_changes() -> None:
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={_id("input"): FileHash(hash="hash_a")},
        out_paths=[_id("stage", "output")],
        explain=True,
        force=True,
    )
    assert result["changed"] is True
    assert "forced" in result["reason"]
    assert result.get("code_changes") == []
    assert result.get("param_changes") == []
    assert result.get("dep_changes") == []


# =============================================================================
# Short-circuit verification: fast mode doesn't populate detail fields
# =============================================================================


def test_fast_mode_no_detail_fields_on_short_circuit() -> None:
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "DIFFERENT"},
        params={"lr": 0.01},
        dep_hashes={_id("input"): FileHash(hash="hash_a")},
        out_paths=[_id("stage", "output")],
    )
    assert result["changed"] is True
    assert result.get("param_changes") is None
    assert result.get("dep_changes") is None


# =============================================================================
# Edge cases
# =============================================================================


def test_new_dep_added() -> None:
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={
            _id("input"): {"hash": "hash_a"},
            _id("extra"): FileHash(hash="hash_b"),
        },
        out_paths=[_id("stage", "output")],
        explain=True,
    )
    assert result["changed"] is True
    dep_changes = result.get("dep_changes", [])
    added = [c for c in dep_changes if c["change_type"] == "added"]
    assert len(added) == 1


def test_dep_removed() -> None:
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={},
        out_paths=[_id("stage", "output")],
        explain=True,
    )
    assert result["changed"] is True
    dep_changes = result.get("dep_changes", [])
    removed = [c for c in dep_changes if c["change_type"] == "removed"]
    assert len(removed) == 1


def test_code_function_added() -> None:
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123", "func:helper": "def456"},
        params={"lr": 0.01},
        dep_hashes={_id("input"): FileHash(hash="hash_a")},
        out_paths=[_id("stage", "output")],
        explain=True,
    )
    assert result["changed"] is True
    code_changes = result.get("code_changes", [])
    added = [c for c in code_changes if c["change_type"] == "added"]
    assert len(added) == 1
    assert added[0]["key"] == "func:helper"


def test_group_dep_ignores_unaccessed_keys() -> None:
    lock_data = _helper_make_lock_data(
        dep_hashes={
            _id("group"): DirHash(
                hash="group_hash",
                manifest=[
                    {"relpath": "x", "hash": "hash_x", "size": 1, "isexec": False},
                ],
            ),
        },
    )
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={
            _id("group"): DirHash(
                hash="group_hash",
                manifest=[
                    {"relpath": "x", "hash": "hash_x", "size": 1, "isexec": False},
                    {"relpath": "y", "hash": "hash_y_changed", "size": 1, "isexec": False},
                ],
            )
        },
        out_paths=[_id("stage", "output")],
        explain=True,
    )
    assert result["changed"] is False
    dep_changes = result.get("dep_changes", [])
    assert dep_changes == []


def test_group_dep_detects_accessed_key_change() -> None:
    lock_data = _helper_make_lock_data(
        dep_hashes={
            _id("group"): DirHash(
                hash="group_hash",
                manifest=[
                    {"relpath": "x", "hash": "hash_x", "size": 1, "isexec": False},
                ],
            ),
        },
    )
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={
            _id("group"): DirHash(
                hash="group_hash",
                manifest=[
                    {"relpath": "x", "hash": "hash_x_changed", "size": 1, "isexec": False},
                ],
            )
        },
        out_paths=[_id("stage", "output")],
        explain=True,
    )
    assert result["changed"] is True
    dep_changes = result.get("dep_changes", [])
    assert dep_changes[0]["identity"] == _id("group")


# =============================================================================
# ArtifactIdentity-keyed dep_hashes
# =============================================================================


def test_diff_dep_hashes_identity_keyed() -> None:
    """diff_dep_hashes works with ArtifactIdentity-keyed dicts and returns ArtifactIdentity."""
    old: dict[ArtifactIdentity, HashInfo] = {
        _id("input"): FileHash(hash="aaa"),
        _id("features", "emb"): FileHash(hash="bbb"),
    }
    new: dict[ArtifactIdentity, HashInfo] = {
        _id("input"): FileHash(hash="aaa_changed"),
        _id("features", "emb"): FileHash(hash="bbb"),
        _id("extra"): FileHash(hash="ccc"),
    }
    changes = skip.diff_dep_hashes(old, new)
    assert len(changes) == 2, f"Expected 2 changes, got {changes}"
    # Modified
    modified = [c for c in changes if c["change_type"] == "modified"]
    assert len(modified) == 1
    assert modified[0]["identity"] == _id("input")
    assert isinstance(modified[0]["identity"], ArtifactIdentity)
    # Added
    added = [c for c in changes if c["change_type"] == "added"]
    assert len(added) == 1
    assert added[0]["identity"] == _id("extra")


def test_check_stage_identity_keyed_dep_hashes() -> None:
    """check_stage accepts ArtifactIdentity-keyed dep_hashes and out_paths."""
    lock_data = LockData(
        code_manifest={"func:main": "abc"},
        params={},
        dep_hashes={_id("input"): FileHash(hash="aaa")},
        output_hashes={_id("stage", "out"): FileHash(hash="bbb")},
    )
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc"},
        params={},
        dep_hashes={_id("input"): FileHash(hash="aaa")},
        out_paths=[_id("stage", "out")],
    )
    assert result["changed"] is False


def test_check_stage_identity_keyed_dep_change_detected() -> None:
    """check_stage detects changes in ArtifactIdentity-keyed dep_hashes."""
    lock_data = LockData(
        code_manifest={"func:main": "abc"},
        params={},
        dep_hashes={_id("input"): FileHash(hash="aaa")},
        output_hashes={_id("stage", "out"): FileHash(hash="bbb")},
    )
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc"},
        params={},
        dep_hashes={_id("input"): FileHash(hash="CHANGED")},
        out_paths=[_id("stage", "out")],
        explain=True,
    )
    assert result["changed"] is True
    dep_changes = result.get("dep_changes", [])
    assert len(dep_changes) == 1
    assert dep_changes[0]["identity"] == _id("input")
