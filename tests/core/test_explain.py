"""Tests for explain module - detailed change explanations."""

from pathlib import Path
from typing import ClassVar

import pydantic
import pytest

from pivot import explain
from pivot.storage import lock
from pivot.types import (
    ChangeType,
    CodeChange,
    DepChange,
    HashInfo,
    LockData,
    ParamChange,
    StageExplanation,
)

# =============================================================================
# diff_code_manifests tests
# =============================================================================


@pytest.mark.parametrize(
    ("old", "new", "expected_key", "expected_old_hash", "expected_new_hash", "expected_type"),
    [
        pytest.param(
            {"func:helper": "abc123"},
            {"func:helper": "def456"},
            "func:helper",
            "abc123",
            "def456",
            ChangeType.MODIFIED,
            id="modified",
        ),
        pytest.param(
            {},
            {"func:new_helper": "abc123"},
            "func:new_helper",
            None,
            "abc123",
            ChangeType.ADDED,
            id="added",
        ),
        pytest.param(
            {"func:old_helper": "abc123"},
            {},
            "func:old_helper",
            "abc123",
            None,
            ChangeType.REMOVED,
            id="removed",
        ),
    ],
)
def test_diff_code_change(
    old: dict[str, str],
    new: dict[str, str],
    expected_key: str,
    expected_old_hash: str | None,
    expected_new_hash: str | None,
    expected_type: ChangeType,
) -> None:
    """diff_code_manifests detects added/modified/removed code components."""
    changes = explain.diff_code_manifests(old, new)
    assert len(changes) == 1
    assert changes[0] == CodeChange(
        key=expected_key,
        old_hash=expected_old_hash,
        new_hash=expected_new_hash,
        change_type=expected_type,
    )


def test_diff_code_multiple_changes() -> None:
    """Detects multiple simultaneous code changes."""
    old = {"func:a": "hash_a", "func:b": "hash_b"}
    new = {"func:a": "hash_a_new", "func:c": "hash_c"}

    changes = explain.diff_code_manifests(old, new)

    keys: dict[str, CodeChange] = {c["key"]: c for c in changes}
    assert len(keys) == 3
    assert keys["func:a"]["change_type"] == "modified"
    assert keys["func:b"]["change_type"] == "removed"
    assert keys["func:c"]["change_type"] == "added"


def test_diff_code_unchanged() -> None:
    """Returns empty list when code unchanged."""
    manifest = {"func:helper": "abc123", "self:stage": "def456"}

    changes = explain.diff_code_manifests(manifest, manifest)

    assert changes == []


# =============================================================================
# diff_params tests
# =============================================================================


@pytest.mark.parametrize(
    ("old", "new", "expected_key", "expected_old_value", "expected_new_value", "expected_type"),
    [
        pytest.param(
            {"learning_rate": 0.01},
            {"learning_rate": 0.001},
            "learning_rate",
            0.01,
            0.001,
            ChangeType.MODIFIED,
            id="modified",
        ),
        pytest.param({}, {"batch_size": 32}, "batch_size", None, 32, ChangeType.ADDED, id="added"),
        pytest.param({"epochs": 10}, {}, "epochs", 10, None, ChangeType.REMOVED, id="removed"),
    ],
)
def test_diff_params_change(
    old: dict[str, object],
    new: dict[str, object],
    expected_key: str,
    expected_old_value: object,
    expected_new_value: object,
    expected_type: ChangeType,
) -> None:
    """diff_params detects added/modified/removed params."""
    changes = explain.diff_params(old, new)
    assert len(changes) == 1
    assert changes[0] == ParamChange(
        key=expected_key,
        old_value=expected_old_value,
        new_value=expected_new_value,
        change_type=expected_type,
    )


def test_diff_params_nested_changed() -> None:
    """Detects changes in nested params."""
    old = {"model": {"hidden_size": 256}}
    new = {"model": {"hidden_size": 512}}

    changes = explain.diff_params(old, new)

    assert len(changes) == 1
    assert changes[0]["key"] == "model"
    assert changes[0]["change_type"] == "modified"


def test_diff_params_unchanged() -> None:
    """Returns empty list when params unchanged."""
    params = {"learning_rate": 0.01, "epochs": 10}

    changes = explain.diff_params(params, params)

    assert changes == []


# =============================================================================
# diff_dep_hashes tests
# =============================================================================


@pytest.mark.parametrize(
    ("old", "new", "expected_path", "expected_old_hash", "expected_new_hash", "expected_type"),
    [
        pytest.param(
            {"data.csv": {"hash": "abc123"}},
            {"data.csv": {"hash": "def456"}},
            "data.csv",
            "abc123",
            "def456",
            ChangeType.MODIFIED,
            id="modified",
        ),
        pytest.param(
            {},
            {"new_data.csv": {"hash": "abc123"}},
            "new_data.csv",
            None,
            "abc123",
            ChangeType.ADDED,
            id="added",
        ),
        pytest.param(
            {"old_data.csv": {"hash": "abc123"}},
            {},
            "old_data.csv",
            "abc123",
            None,
            ChangeType.REMOVED,
            id="removed",
        ),
    ],
)
def test_diff_deps_change(
    old: dict[str, HashInfo],
    new: dict[str, HashInfo],
    expected_path: str,
    expected_old_hash: str | None,
    expected_new_hash: str | None,
    expected_type: ChangeType,
) -> None:
    """diff_dep_hashes detects added/modified/removed dependencies."""
    changes = explain.diff_dep_hashes(old, new)
    assert len(changes) == 1
    assert changes[0] == DepChange(
        path=expected_path,
        old_hash=expected_old_hash,
        new_hash=expected_new_hash,
        change_type=expected_type,
    )


def test_diff_deps_directory_with_manifest() -> None:
    """Handles directory dependencies with manifests."""
    old: dict[str, HashInfo] = {
        "data_dir": {
            "hash": "tree_abc",
            "manifest": [{"relpath": "a.csv", "hash": "h1", "size": 10}],
        }
    }
    new: dict[str, HashInfo] = {
        "data_dir": {
            "hash": "tree_def",
            "manifest": [{"relpath": "a.csv", "hash": "h2", "size": 10}],
        }
    }

    changes = explain.diff_dep_hashes(old, new)

    assert len(changes) == 1
    assert changes[0]["path"] == "data_dir"
    assert changes[0]["old_hash"] == "tree_abc"
    assert changes[0]["new_hash"] == "tree_def"


def test_diff_deps_unchanged() -> None:
    """Returns empty list when deps unchanged."""
    dep_hashes: dict[str, HashInfo] = {"data.csv": {"hash": "abc123"}}

    changes = explain.diff_dep_hashes(dep_hashes, dep_hashes)

    assert changes == []


# =============================================================================
# get_stage_explanation tests
# =============================================================================


def test_get_stage_explanation_no_lock(tmp_path: Path) -> None:
    """Returns 'No previous run' when no lock file exists."""
    result = explain.get_stage_explanation(
        stage_name="new_stage",
        fingerprint={"self:new_stage": "abc123"},
        deps=[],
        params_instance=None,
        overrides=None,
        cache_dir=tmp_path,
    )

    assert result == StageExplanation(
        stage_name="new_stage",
        will_run=True,
        is_forced=False,
        reason="No previous run",
        code_changes=[],
        param_changes=[],
        dep_changes=[],
    )


def test_get_stage_explanation_unchanged(tmp_path: Path) -> None:
    """Returns will_run=False when stage unchanged."""
    stage_lock = lock.StageLock("unchanged_stage", tmp_path)
    stage_lock.write(
        LockData(
            code_manifest={"self:unchanged_stage": "abc123"},
            params={},
            dep_hashes={},
            output_hashes={},
        )
    )

    result = explain.get_stage_explanation(
        stage_name="unchanged_stage",
        fingerprint={"self:unchanged_stage": "abc123"},
        deps=[],
        params_instance=None,
        overrides=None,
        cache_dir=tmp_path,
    )

    assert result["will_run"] is False
    assert result["reason"] == ""
    assert result["code_changes"] == []
    assert result["param_changes"] == []
    assert result["dep_changes"] == []


def test_get_stage_explanation_code_changed(tmp_path: Path) -> None:
    """Returns detailed code changes when code differs."""
    stage_lock = lock.StageLock("code_stage", tmp_path)
    stage_lock.write(
        LockData(
            code_manifest={"self:code_stage": "old_hash", "func:helper": "helper_old"},
            params={},
            dep_hashes={},
            output_hashes={},
        )
    )

    result = explain.get_stage_explanation(
        stage_name="code_stage",
        fingerprint={"self:code_stage": "new_hash", "func:helper": "helper_old"},
        deps=[],
        params_instance=None,
        overrides=None,
        cache_dir=tmp_path,
    )

    assert result["will_run"] is True
    assert result["reason"] == "Code changed"
    assert len(result["code_changes"]) == 1
    assert result["code_changes"][0]["key"] == "self:code_stage"
    assert result["code_changes"][0]["change_type"] == "modified"


def test_get_stage_explanation_params_changed(tmp_path: Path) -> None:
    """Returns detailed param changes when params differ."""

    class TrainParams(pydantic.BaseModel):
        learning_rate: float = 0.01

    stage_lock = lock.StageLock("param_stage", tmp_path)
    stage_lock.write(
        LockData(
            code_manifest={"self:param_stage": "abc"},
            params={"learning_rate": 0.01},
            dep_hashes={},
            output_hashes={},
        )
    )

    overrides = {"param_stage": {"learning_rate": 0.001}}
    result = explain.get_stage_explanation(
        stage_name="param_stage",
        fingerprint={"self:param_stage": "abc"},
        deps=[],
        params_instance=TrainParams(),
        overrides=overrides,
        cache_dir=tmp_path,
    )

    assert result["will_run"] is True
    assert result["reason"] == "Params changed"
    assert len(result["param_changes"]) == 1
    assert result["param_changes"][0]["key"] == "learning_rate"
    assert result["param_changes"][0]["old_value"] == 0.01
    assert result["param_changes"][0]["new_value"] == 0.001


def test_get_stage_explanation_deps_changed(tmp_path: Path) -> None:
    """Returns detailed dep changes when dependencies differ."""
    from pivot import project

    data_file = tmp_path / "data.csv"
    data_file.write_text("id,value\n1,10\n")
    normalized_path = str(project.normalize_path(str(data_file)))

    stage_lock = lock.StageLock("dep_stage", tmp_path)
    stage_lock.write(
        LockData(
            code_manifest={"self:dep_stage": "abc"},
            params={},
            dep_hashes={normalized_path: {"hash": "old_data_hash"}},
            output_hashes={},
        )
    )

    result = explain.get_stage_explanation(
        stage_name="dep_stage",
        fingerprint={"self:dep_stage": "abc"},
        deps=[str(data_file)],
        params_instance=None,
        overrides=None,
        cache_dir=tmp_path,
    )

    assert result["will_run"] is True
    assert "dep" in result["reason"].lower() or "input" in result["reason"].lower()
    assert len(result["dep_changes"]) == 1
    assert result["dep_changes"][0]["old_hash"] == "old_data_hash"
    assert result["dep_changes"][0]["change_type"] == "modified"


def test_get_stage_explanation_multiple_changes(tmp_path: Path) -> None:
    """Returns all changes when multiple things differ."""
    from pivot import project

    class Params(pydantic.BaseModel):
        epochs: int = 10

    data_file = tmp_path / "input.csv"
    data_file.write_text("x,y\n1,2\n")
    normalized_path = str(project.normalize_path(str(data_file)))

    stage_lock = lock.StageLock("multi_stage", tmp_path)
    stage_lock.write(
        LockData(
            code_manifest={"self:multi_stage": "old_code"},
            params={"epochs": 5},
            dep_hashes={normalized_path: {"hash": "old_hash"}},
            output_hashes={},
        )
    )

    result = explain.get_stage_explanation(
        stage_name="multi_stage",
        fingerprint={"self:multi_stage": "new_code"},
        deps=[str(data_file)],
        params_instance=Params(),
        overrides=None,
        cache_dir=tmp_path,
    )

    assert result["will_run"] is True
    assert len(result["code_changes"]) >= 1, "Should have code changes"
    assert len(result["param_changes"]) >= 1, "Should have param changes"
    assert len(result["dep_changes"]) >= 1, "Should have dep changes"


def test_get_stage_explanation_missing_deps(tmp_path: Path) -> None:
    """Returns will_run=True with reason when deps are missing."""
    stage_lock = lock.StageLock("missing_deps_stage", tmp_path)
    stage_lock.write(
        LockData(
            code_manifest={"self:missing_deps_stage": "abc"},
            params={},
            dep_hashes={},
            output_hashes={},
        )
    )

    result = explain.get_stage_explanation(
        stage_name="missing_deps_stage",
        fingerprint={"self:missing_deps_stage": "abc"},
        deps=["nonexistent.csv"],
        params_instance=None,
        overrides=None,
        cache_dir=tmp_path,
    )

    assert result["will_run"] is True
    assert "missing" in result["reason"].lower()


@pytest.mark.parametrize(
    "invalid_params_yaml",
    [
        {"stage": {"unknown_field": "value"}},  # Unknown field
    ],
)
def test_get_stage_explanation_invalid_params(
    tmp_path: Path, invalid_params_yaml: dict[str, dict[str, str]]
) -> None:
    """Returns will_run=True with reason when params.yaml is invalid."""

    class StrictParams(pydantic.BaseModel):
        model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="forbid")
        learning_rate: float = 0.01

    stage_lock = lock.StageLock("stage", tmp_path)
    stage_lock.write(
        LockData(
            code_manifest={"self:stage": "abc"},
            params={"learning_rate": 0.01},
            dep_hashes={},
            output_hashes={},
        )
    )

    result = explain.get_stage_explanation(
        stage_name="stage",
        fingerprint={"self:stage": "abc"},
        deps=[],
        params_instance=StrictParams(),
        overrides=invalid_params_yaml,
        cache_dir=tmp_path,
    )

    assert result["will_run"] is True
    assert "invalid" in result["reason"].lower() or "error" in result["reason"].lower()


# =============================================================================
# get_stage_explanation force=True tests
# =============================================================================


@pytest.mark.parametrize(
    ("stage_name", "create_lock"),
    [
        pytest.param("no_lock_stage", False, id="no_lock"),
        pytest.param("unchanged_stage", True, id="unchanged"),
    ],
)
def test_get_stage_explanation_force_without_changes(
    tmp_path: Path, stage_name: str, create_lock: bool
) -> None:
    """Force=True with no actual changes shows 'forced' reason and is_forced=True."""
    fingerprint = {f"self:{stage_name}": "abc123"}

    if create_lock:
        stage_lock = lock.StageLock(stage_name, tmp_path)
        stage_lock.write(
            LockData(
                code_manifest=fingerprint,
                params={},
                dep_hashes={},
                output_hashes={},
            )
        )

    result = explain.get_stage_explanation(
        stage_name=stage_name,
        fingerprint=fingerprint,
        deps=[],
        params_instance=None,
        overrides=None,
        cache_dir=tmp_path,
        force=True,
    )

    assert result == StageExplanation(
        stage_name=stage_name,
        will_run=True,
        is_forced=True,
        reason="forced",
        code_changes=[],
        param_changes=[],
        dep_changes=[],
    )


def test_get_stage_explanation_force_with_code_changes(tmp_path: Path) -> None:
    """Force with code changes shows code changes but is_forced=True."""
    stage_lock = lock.StageLock("code_stage", tmp_path)
    stage_lock.write(
        LockData(
            code_manifest={"self:code_stage": "old_hash"},
            params={},
            dep_hashes={},
            output_hashes={},
        )
    )

    result = explain.get_stage_explanation(
        stage_name="code_stage",
        fingerprint={"self:code_stage": "new_hash"},
        deps=[],
        params_instance=None,
        overrides=None,
        cache_dir=tmp_path,
        force=True,
    )

    assert result["will_run"] is True
    assert result["is_forced"] is True
    assert result["reason"] == "Code changed", "Code changes take precedence over 'forced' reason"
    assert len(result["code_changes"]) == 1
    assert result["code_changes"][0]["key"] == "self:code_stage"


def test_get_stage_explanation_force_with_missing_deps(tmp_path: Path) -> None:
    """Force with missing deps shows missing deps but is_forced=True."""
    stage_lock = lock.StageLock("dep_stage", tmp_path)
    stage_lock.write(
        LockData(
            code_manifest={"self:dep_stage": "abc"},
            params={},
            dep_hashes={},
            output_hashes={},
        )
    )

    result = explain.get_stage_explanation(
        stage_name="dep_stage",
        fingerprint={"self:dep_stage": "abc"},
        deps=["nonexistent.csv"],
        params_instance=None,
        overrides=None,
        cache_dir=tmp_path,
        force=True,
    )

    assert result["will_run"] is True
    assert result["is_forced"] is True
    assert "missing" in result["reason"].lower()
