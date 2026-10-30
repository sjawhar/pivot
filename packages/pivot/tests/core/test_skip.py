"""Tests for skip.check_stage — unified skip detection."""

from __future__ import annotations

from typing import Any

import pytest

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


def test_group_dep_detects_added_key() -> None:
    lock_data = _helper_make_lock_data(
        dep_hashes={
            _id("group"): DirHash(hash="group_hash", size=1, nfiles=1),
        },
    )
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={_id("group"): DirHash(hash="group_hash_with_y", size=2, nfiles=2)},
        out_paths=[_id("stage", "output")],
        explain=True,
    )
    assert result["changed"] is True
    dep_changes = result.get("dep_changes", [])
    assert dep_changes[0]["identity"] == _id("group")


def test_group_dep_detects_accessed_key_change() -> None:
    lock_data = _helper_make_lock_data(
        dep_hashes={
            _id("group"): DirHash(hash="group_hash", size=1, nfiles=1),
        },
    )
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={_id("group"): DirHash(hash="group_hash_changed", size=1, nfiles=1)},
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


def test_explain_reports_output_paths_change_as_stale() -> None:
    """Status must call a stage stale when only its declared outputs changed, as repro does."""
    lock_data = _helper_make_lock_data(output_hashes={_id("stage", "old"): FileHash(hash="h")})
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={_id("input"): FileHash(hash="hash_a")},
        out_paths=[_id("stage", "new")],
        explain=True,
    )
    assert result["changed"] is True
    assert result["reason"] == "Output paths changed"


def test_explain_reports_added_and_removed_params() -> None:
    """A param that appears or disappears is a change, not only a param whose value moved."""
    lock_data = _helper_make_lock_data(params={"lr": 0.01, "old_only": 1})
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01, "new_only": 2},
        dep_hashes={_id("input"): FileHash(hash="hash_a")},
        out_paths=[_id("stage", "output")],
        explain=True,
    )
    assert result["changed"] is True
    assert "param_changes" in result, "explain=True must report param changes"
    changes = {c["key"]: c["change_type"] for c in result["param_changes"]}
    assert changes == {"new_only": "added", "old_only": "removed"}


def test_explain_unchanged_stage_has_empty_reason() -> None:
    """Callers print their own 'unchanged' label only when the reason is empty."""
    result = skip.check_stage(
        lock_data=_helper_make_lock_data(),
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={_id("input"): FileHash(hash="hash_a")},
        out_paths=[_id("stage", "output")],
        explain=True,
    )
    assert result["changed"] is False
    assert result["reason"] == ""


# =============================================================================
# Rerun marks (uncovered import, edited source) carried by pivot commit
# =============================================================================

_UNCOVERED_MARK = f"{skip.UNCOVERED_IMPORT_KEY_PREFIX}helpers, helpers.lib"
_EDITED_MARK = f"{skip.EDITED_SOURCE_KEY_PREFIX}helpers"


@pytest.mark.parametrize("mark", [_UNCOVERED_MARK, _EDITED_MARK])
def test_carry_rerun_marks_keeps_marker_when_static_fingerprint_is_unchanged(mark: str) -> None:
    fingerprint = {"self:count": "aa", "module:helpers": ""}

    carried = skip.carry_rerun_marks({**fingerprint, mark: ""}, fingerprint)

    assert carried == {**fingerprint, mark: ""}
    assert skip.rerun_mark_reason(carried) is not None


@pytest.mark.parametrize("mark", [_UNCOVERED_MARK, _EDITED_MARK])
def test_carry_rerun_marks_drops_marker_after_a_code_edit(mark: str) -> None:
    previous = {"self:count": "aa", mark: ""}
    fingerprint = {"self:count": "bb"}

    assert skip.carry_rerun_marks(previous, fingerprint) == fingerprint


def test_carry_rerun_marks_without_a_marked_lock_returns_the_fingerprint() -> None:
    fingerprint = {"self:count": "aa"}

    assert skip.carry_rerun_marks(None, fingerprint) == fingerprint
    assert skip.carry_rerun_marks(dict(fingerprint), fingerprint) == fingerprint


@pytest.mark.parametrize(
    ("mark", "reason"),
    [
        (
            _UNCOVERED_MARK,
            "Previous run imported modules its fingerprint does not cover: helpers, helpers.lib",
        ),
        (_EDITED_MARK, "Previous run imported modules edited after pivot loaded them: helpers"),
    ],
)
def test_marked_lock_reruns_with_its_reason_and_never_from_the_run_cache(
    mark: str, reason: str
) -> None:
    """A lock with either mark never matches its fingerprint, names why, and is never satisfied
    from the run cache."""
    fingerprint = {"self:count": "aa"}
    lock_data = LockData(
        code_manifest={**fingerprint, mark: ""},
        params={},
        dep_hashes={},
        output_hashes={},
    )

    for explain in (False, True):
        decision = skip.check_stage(lock_data, fingerprint, {}, {}, [], explain=explain)

        assert decision["changed"] is True
        assert decision["reason"] == reason
        assert decision["run_cache_allowed"] is False


# =============================================================================
# Objects the previous run read by hash, now in neither the cache nor the remote
# =============================================================================

_UNAVAILABLE = "Read objects missing from the cache: 0123456789abcdef (no remote is configured)"


@pytest.mark.parametrize("explain", [False, True])
def test_a_current_lock_whose_read_objects_are_unavailable_is_changed(explain: bool) -> None:
    lock_data = _helper_make_lock_data()

    decision = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={_id("input"): FileHash(hash="hash_a")},
        out_paths=[_id("stage", "output")],
        explain=explain,
        unavailable_reads=_UNAVAILABLE,
    )

    assert decision["changed"] is True
    assert decision["reason"] == _UNAVAILABLE


@pytest.mark.parametrize("explain", [False, True])
def test_a_code_change_is_named_before_unavailable_read_objects(explain: bool) -> None:
    lock_data = _helper_make_lock_data()

    decision = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "DIFFERENT"},
        params={"lr": 0.01},
        dep_hashes={_id("input"): FileHash(hash="hash_a")},
        out_paths=[_id("stage", "output")],
        explain=explain,
        unavailable_reads=_UNAVAILABLE,
    )

    assert decision["changed"] is True
    assert decision["reason"] == "Code changed"
