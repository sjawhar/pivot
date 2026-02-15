"""Tests for skip.check_stage — unified skip detection."""

from __future__ import annotations

from typing import Any

from pivot import skip
from pivot.types import DirHash, FileHash, HashInfo, LockData


def _identity(producer: str, key: str | None) -> str:
    return producer if key is None else f"{producer}:{key}"


def _helper_make_lock_data(
    *,
    code_manifest: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    dep_hashes: dict[str, HashInfo] | None = None,
    output_hashes: dict[str, HashInfo] | None = None,
) -> LockData:
    return LockData(
        code_manifest=code_manifest or {"func:main": "abc123"},
        params=params or {"lr": 0.01},
        dep_hashes=dep_hashes or {_identity("input", None): FileHash(hash="hash_a")},
        output_hashes=output_hashes or {_identity("stage", "output"): FileHash(hash="hash_out")},
    )


# =============================================================================
# No lock data (first run)
# =============================================================================


def test_no_lock_data_returns_changed() -> None:
    result = skip.check_stage(
        lock_data=None,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={_identity("input", None): FileHash(hash="hash_a")},
        out_paths=[_identity("stage", "output")],
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
        dep_hashes={_identity("input", None): FileHash(hash="hash_a")},
        out_paths=[_identity("stage", "output")],
    )
    assert result["changed"] is True
    assert "Code changed" in result["reason"]


def test_fast_params_changed() -> None:
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.99},
        dep_hashes={_identity("input", None): FileHash(hash="hash_a")},
        out_paths=[_identity("stage", "output")],
    )
    assert result["changed"] is True
    assert "Params changed" in result["reason"]


def test_fast_deps_changed() -> None:
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={_identity("input", None): FileHash(hash="DIFFERENT")},
        out_paths=[_identity("stage", "output")],
    )
    assert result["changed"] is True
    assert "dependencies changed" in result["reason"]


def test_fast_out_paths_changed() -> None:
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={_identity("input", None): FileHash(hash="hash_a")},
        out_paths=[_identity("stage", "output"), _identity("stage", "new_output")],
    )
    assert result["changed"] is True
    assert "Output paths changed" in result["reason"]


def test_fast_nothing_changed() -> None:
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={_identity("input", None): FileHash(hash="hash_a")},
        out_paths=[_identity("stage", "output")],
    )
    assert result["changed"] is False
    assert result["reason"] == ""


def test_fast_force_returns_changed() -> None:
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={_identity("input", None): FileHash(hash="hash_a")},
        out_paths=[_identity("stage", "output")],
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
        dep_hashes={_identity("input", None): FileHash(hash="DIFFERENT")},
        out_paths=[_identity("stage", "output")],
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
        dep_hashes={_identity("input", None): FileHash(hash="hash_a")},
        out_paths=[_identity("stage", "output")],
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
        dep_hashes={"/data/input.csv": FileHash(hash="hash_a")},
        out_paths=["/data/output.csv"],
        explain=True,
        force=True,
    )
    assert result["changed"] is True
    assert "forced" in result["reason"]
    # Explain mode still computes diffs even when forced
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
        dep_hashes={"/data/input.csv": FileHash(hash="hash_a")},
        out_paths=["/data/output.csv"],
    )
    assert result["changed"] is True
    # Fast mode short-circuits at code — param/dep changes not computed
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
            _identity("input", None): {"hash": "hash_a"},
            _identity("extra", None): FileHash(hash="hash_b"),
        },
        out_paths=[_identity("stage", "output")],
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
        out_paths=[_identity("stage", "output")],
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
        dep_hashes={_identity("input", None): FileHash(hash="hash_a")},
        out_paths=[_identity("stage", "output")],
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
            _identity("group", None): DirHash(
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
            _identity("group", None): DirHash(
                hash="group_hash",
                manifest=[
                    {"relpath": "x", "hash": "hash_x", "size": 1, "isexec": False},
                    {"relpath": "y", "hash": "hash_y_changed", "size": 1, "isexec": False},
                ],
            )
        },
        out_paths=[_identity("stage", "output")],
        explain=True,
    )
    assert result["changed"] is False
    assert result["dep_changes"] == []


def test_group_dep_detects_accessed_key_change() -> None:
    lock_data = _helper_make_lock_data(
        dep_hashes={
            _identity("group", None): DirHash(
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
            _identity("group", None): DirHash(
                hash="group_hash",
                manifest=[
                    {"relpath": "x", "hash": "hash_x_changed", "size": 1, "isexec": False},
                ],
            )
        },
        out_paths=[_identity("stage", "output")],
        explain=True,
    )
    assert result["changed"] is True
    assert result["dep_changes"][0]["identity"] == _identity("group", None)
