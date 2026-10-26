"""Tests for detailed stage change explanations."""

from __future__ import annotations

import logging
import pathlib
import shutil
from typing import TYPE_CHECKING, Annotated, ClassVar, TypedDict

import pydantic
import pygtrie
import pytest

from pivot import (
    StageParams,
    cli,
    compose,
    config,
    exceptions,
    executor,
    explain,
    loaders,
    project,
    skip,
    status,
)
from pivot.storage import listing, lock, track
from pivot.storage import store as store_mod
from pivot.types import (
    ArtifactIdentity,
    ArtifactRef,
    ArtifactTag,
    DirHash,
    HashInfo,
    LockData,
    StageExplanation,
)

if TYPE_CHECKING:
    import click.testing

# xxhash64 of the bytes each fixture file holds, worked out by hand from the bytes.
_BETA_HASH = "f5ee2990398e98c4"  # b"beta"
_A_CSV_HASH = "c104cf7901c31c75"  # b"a\n1\n"
_B_CSV_HASH = "7700509cbd39f780"  # b"b\n2\n"


class _UpstreamParams(StageParams):
    value: int = 1


@compose.stage
def _upstream_stage_a(input_file: str, params: _UpstreamParams) -> str:
    return f"{params.value}:{input_file}"


@compose.stage
def _upstream_stage_b(a_output: str) -> str:
    return a_output


@compose.stage
def _directory_producer() -> Annotated[dict[str, str], compose.directory]:
    return {"a.txt": "alpha", "sub/b.txt": "beta"}


@compose.stage
def _one_file_consumer(part: str) -> str:
    return part.upper()


def _build_upstream_pipeline(root: pathlib.Path, value: int) -> compose.Pipeline:
    with compose.Pipeline("test", root=root) as pipeline:
        input_file = pipeline.input("input.txt")
        a_output = _upstream_stage_a(
            input_file,  # pyright: ignore[reportArgumentType] - pipeline construction passes an ArtifactHandle
            params=_UpstreamParams(value=value),
        )
        _upstream_stage_b(a_output)
    return pipeline


def _input_ref(name: str) -> ArtifactRef:
    return ArtifactRef(
        identity=ArtifactIdentity(name, None),
        format=loaders.Text(),
        python_type=str,
        tag=ArtifactTag.INPUT,
    )


def _store(tmp_path: pathlib.Path) -> store_mod.WorkspaceStore:
    return store_mod.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="test",
        input_bindings={"data.csv": "data.csv"},
        cache_dir=tmp_path / ".pivot" / "cache",
    )


def _lock(state_dir: pathlib.Path, stage_name: str) -> LockData | None:
    """The stage's lock as status reads it before explaining."""
    return lock.StageLock(stage_name, lock.get_stages_dir(state_dir)).read()


def test_get_stage_explanation_with_allow_missing_stale_pvt(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Detects staleness when the tracked hash differs from the stage lock."""
    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project, "_project_root_cache", None)
    data_path = tmp_path / "data.csv"
    normalized_path = str(project.normalize_path(data_path))
    dep = _input_ref("data.csv")
    tracked_files: dict[str, track.PvtData] = {
        normalized_path: track.PvtData(path="data.csv", hash="2222222222222222", size=100)
    }
    tracked_trie: pygtrie.Trie[str] = pygtrie.Trie()
    tracked_trie[pathlib.Path(normalized_path).parts] = normalized_path
    lock.StageLock("pvt_stage", lock.get_stages_dir(state_dir)).write(
        LockData(
            code_manifest={"self:pvt_stage": "abc"},
            params={},
            dep_hashes={dep.identity: {"hash": "1111111111111111"}},
            output_hashes={},
        )
    )

    result = explain.get_stage_explanation(
        stage_name="pvt_stage",
        fingerprint={"self:pvt_stage": "abc"},
        deps={"data": dep},
        outs=[],
        params_instance=None,
        overrides=None,
        state_dir=state_dir,
        store=_store(tmp_path),
        lock_data=_lock(state_dir, "pvt_stage"),
        allow_missing=True,
        tracked_files=tracked_files,
        tracked_trie=tracked_trie,
    )

    assert result["will_run"] is True
    assert "dep" in result["reason"].lower() or "input" in result["reason"].lower()


def _write_lock(
    state_dir: pathlib.Path,
    stage_name: str,
    code_manifest: dict[str, str],
    *,
    params: dict[str, object] | None = None,
    dep_hashes: dict[ArtifactIdentity, HashInfo] | None = None,
) -> None:
    lock.StageLock(stage_name, lock.get_stages_dir(state_dir)).write(
        LockData(
            code_manifest=code_manifest,
            params={} if params is None else params,
            dep_hashes={} if dep_hashes is None else dep_hashes,
            output_hashes={},
        )
    )


@pytest.mark.parametrize(
    ("old", "new", "expected_key", "expected_old_hash", "expected_new_hash", "expected_type"),
    [
        pytest.param(
            {"func:helper": "abc123"},
            {"func:helper": "def456"},
            "func:helper",
            "abc123",
            "def456",
            "modified",
            id="modified",
        ),
        pytest.param(
            {},
            {"func:new_helper": "abc123"},
            "func:new_helper",
            None,
            "abc123",
            "added",
            id="added",
        ),
        pytest.param(
            {"func:old_helper": "abc123"},
            {},
            "func:old_helper",
            "abc123",
            None,
            "removed",
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
    expected_type: str,
) -> None:
    """diff_code_manifests detects added, modified, and removed components."""
    changes = skip.diff_code_manifests(old, new)

    assert len(changes) == 1
    assert changes[0]["key"] == expected_key
    assert changes[0]["old_hash"] == expected_old_hash
    assert changes[0]["new_hash"] == expected_new_hash
    assert changes[0]["change_type"] == expected_type


def test_diff_code_multiple_changes() -> None:
    """Detects multiple simultaneous code changes."""
    old = {"func:a": "hash_a", "func:b": "hash_b"}
    new = {"func:a": "hash_a_new", "func:c": "hash_c"}

    changes = skip.diff_code_manifests(old, new)

    keys = {change["key"]: change for change in changes}
    assert len(keys) == 3
    assert keys["func:a"]["change_type"] == "modified"
    assert keys["func:b"]["change_type"] == "removed"
    assert keys["func:c"]["change_type"] == "added"


def test_diff_code_unchanged() -> None:
    """Returns no changes when a code manifest is unchanged."""
    manifest = {"func:helper": "abc123", "self:stage": "def456"}

    assert skip.diff_code_manifests(manifest, manifest) == []


@pytest.mark.parametrize(
    ("old", "new", "expected_key", "expected_old_value", "expected_new_value", "expected_type"),
    [
        pytest.param(
            {"learning_rate": 0.01},
            {"learning_rate": 0.001},
            "learning_rate",
            0.01,
            0.001,
            "modified",
            id="modified",
        ),
        pytest.param({}, {"batch_size": 32}, "batch_size", None, 32, "added", id="added"),
        pytest.param({"epochs": 10}, {}, "epochs", 10, None, "removed", id="removed"),
    ],
)
def test_diff_params_change(
    old: dict[str, object],
    new: dict[str, object],
    expected_key: str,
    expected_old_value: object,
    expected_new_value: object,
    expected_type: str,
) -> None:
    """diff_params detects added, modified, and removed parameters."""
    changes = skip.diff_params(old, new)

    assert len(changes) == 1
    assert changes[0]["key"] == expected_key
    assert changes[0]["old_value"] == expected_old_value
    assert changes[0]["new_value"] == expected_new_value
    assert changes[0]["change_type"] == expected_type


def test_diff_params_nested_changed() -> None:
    """Detects a changed nested parameter value."""
    changes = skip.diff_params(
        {"model": {"hidden_size": 256}},
        {"model": {"hidden_size": 512}},
    )

    assert len(changes) == 1
    assert changes[0]["key"] == "model"
    assert changes[0]["change_type"] == "modified"


def test_diff_params_unchanged() -> None:
    """Returns no changes when parameters are unchanged."""
    params = {"learning_rate": 0.01, "epochs": 10}

    assert skip.diff_params(params, params) == []


def test_diff_params_treats_nan_as_unchanged() -> None:
    """A recorded NaN is not equal to a fresh NaN in Python; the canonical text is."""
    assert skip.diff_params({"ceiling": float("nan")}, {"ceiling": float("nan")}) == []
    assert skip.params_equal({"c": [float("nan")]}, {"c": [float("nan")]})
    assert not skip.params_equal({"c": float("nan")}, {"c": 1.0})


@pytest.mark.parametrize(
    ("old", "new", "expected_identity", "expected_old_hash", "expected_new_hash", "expected_type"),
    [
        pytest.param(
            {"data.csv": {"hash": "abc123"}},
            {"data.csv": {"hash": "def456"}},
            ArtifactIdentity("data.csv", None),
            "abc123",
            "def456",
            "modified",
            id="modified",
        ),
        pytest.param(
            {},
            {"new_data.csv": {"hash": "abc123"}},
            ArtifactIdentity("new_data.csv", None),
            None,
            "abc123",
            "added",
            id="added",
        ),
        pytest.param(
            {"old_data.csv": {"hash": "abc123"}},
            {},
            ArtifactIdentity("old_data.csv", None),
            "abc123",
            None,
            "removed",
            id="removed",
        ),
    ],
)
def test_diff_deps_change(
    old: dict[str, HashInfo],
    new: dict[str, HashInfo],
    expected_identity: ArtifactIdentity,
    expected_old_hash: str | None,
    expected_new_hash: str | None,
    expected_type: str,
) -> None:
    """diff_dep_hashes detects added, modified, and removed dependencies."""
    old_by_identity = {ArtifactIdentity(path, None): hash_info for path, hash_info in old.items()}
    new_by_identity = {ArtifactIdentity(path, None): hash_info for path, hash_info in new.items()}
    changes = skip.diff_dep_hashes(old_by_identity, new_by_identity)

    assert len(changes) == 1
    assert changes[0]["identity"] == expected_identity
    assert changes[0]["old_hash"] == expected_old_hash
    assert changes[0]["new_hash"] == expected_new_hash
    assert changes[0]["change_type"] == expected_type


def test_diff_deps_directory_compares_listing_hashes() -> None:
    """A directory dependency changes when the hash of its listing changes."""
    identity = ArtifactIdentity("data_dir", None)
    old: dict[ArtifactIdentity, HashInfo] = {identity: DirHash(hash="tree_abc", size=10, nfiles=1)}
    new: dict[ArtifactIdentity, HashInfo] = {identity: DirHash(hash="tree_def", size=10, nfiles=1)}

    changes = skip.diff_dep_hashes(old, new)

    assert len(changes) == 1
    assert changes[0]["identity"] == identity
    assert changes[0]["old_hash"] == "tree_abc"
    assert changes[0]["new_hash"] == "tree_def"


def test_diff_deps_unchanged() -> None:
    """Returns no changes when dependency hashes are unchanged."""
    identity = ArtifactIdentity("data.csv", None)
    dep_hashes: dict[ArtifactIdentity, HashInfo] = {identity: {"hash": "abc123"}}

    assert skip.diff_dep_hashes(dep_hashes, dep_hashes) == []


def test_get_stage_explanation_no_lock(tmp_path: pathlib.Path) -> None:
    """Returns 'No previous run' when no lock file exists."""
    result = explain.get_stage_explanation(
        stage_name="new_stage",
        fingerprint={"self:new_stage": "abc123"},
        deps={},
        outs=[],
        params_instance=None,
        overrides=None,
        state_dir=tmp_path / ".pivot",
        store=_store(tmp_path),
        lock_data=None,
    )

    assert result == StageExplanation(
        stage_name="new_stage",
        will_run=True,
        is_forced=False,
        reason="No previous run",
        code_changes=[],
        param_changes=[],
        dep_changes=[],
        upstream_stale=[],
    )


def test_get_stage_explanation_unchanged(tmp_path: pathlib.Path) -> None:
    """Returns cached when the stage lock matches current inputs."""
    state_dir = tmp_path / ".pivot"
    manifest = {"self:unchanged_stage": "abc123"}
    _write_lock(state_dir, "unchanged_stage", manifest)

    result = explain.get_stage_explanation(
        stage_name="unchanged_stage",
        fingerprint=manifest,
        deps={},
        outs=[],
        params_instance=None,
        overrides=None,
        state_dir=state_dir,
        store=_store(tmp_path),
        lock_data=_lock(state_dir, "unchanged_stage"),
    )

    assert result["will_run"] is False
    assert result["reason"] == ""
    assert result["code_changes"] == []
    assert result["param_changes"] == []
    assert result["dep_changes"] == []


def test_get_stage_explanation_without_state_db_has_no_side_effects(
    tmp_path: pathlib.Path,
) -> None:
    """Does not create state.lmdb when explain falls back to hash comparisons."""
    state_dir = tmp_path / ".pivot"
    _write_lock(state_dir, "unchanged_stage", {"self:unchanged_stage": "abc123"})

    result = explain.get_stage_explanation(
        stage_name="unchanged_stage",
        fingerprint={"self:unchanged_stage": "abc123"},
        deps={},
        outs=[],
        params_instance=None,
        overrides=None,
        state_dir=state_dir,
        store=_store(tmp_path),
        lock_data=_lock(state_dir, "unchanged_stage"),
    )

    assert result["will_run"] is False
    assert not (state_dir / "state.lmdb").exists()


def _output_ref(producer: str, key: str, tag: ArtifactTag) -> ArtifactRef:
    return ArtifactRef(
        identity=ArtifactIdentity(producer, key),
        format=loaders.YAML(),
        python_type=dict,
        tag=tag,
    )


def _explain_with_outputs(
    tmp_path: pathlib.Path, outs: list[ArtifactRef], edit: dict[ArtifactRef, str | None]
) -> StageExplanation:
    """Lock `outs` at their current bytes, apply `edit` (None deletes), and explain the stage."""
    state_dir = tmp_path / ".pivot"
    store = _store(tmp_path)
    output_hashes = dict[ArtifactIdentity, HashInfo]()
    for ref in outs:
        path = store.resolve_display_path(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value: 1\n")
        output_hashes[ref.identity] = executor.worker.hash_output(path)
    manifest = {"self:summarize": "abc123"}
    lock.StageLock("s", lock.get_stages_dir(state_dir)).write(
        LockData(code_manifest=manifest, params={}, dep_hashes={}, output_hashes=output_hashes)
    )
    for ref, text in edit.items():
        path = store.resolve_display_path(ref)
        if text is None:
            path.unlink()
        else:
            path.write_text(text)

    return explain.get_stage_explanation(
        stage_name="s",
        fingerprint=manifest,
        deps={},
        outs=outs,
        params_instance=None,
        overrides=None,
        state_dir=state_dir,
        store=store,
        lock_data=_lock(state_dir, "s"),
    )


def test_get_stage_explanation_edited_metric_is_not_missing_from_cache(
    tmp_path: pathlib.Path,
) -> None:
    """A hand-edited metric is uncached by design, so pull cannot restore it: say what changed."""
    table = _output_ref("s", "table", ArtifactTag.DATA)
    stats = _output_ref("s", "stats", ArtifactTag.METRIC)

    result = _explain_with_outputs(tmp_path, [stats, table], {stats: "value: 999\n"})

    assert result["will_run"] is True
    assert result["reason"] != explain.OUTPUTS_MISSING_REASON
    assert result["reason"] == "Metric output changed: s:stats"


def test_get_stage_explanation_deleted_metric_is_reported_missing(tmp_path: pathlib.Path) -> None:
    """A deleted metric names the metric, not the cache."""
    stats = _output_ref("s", "stats", ArtifactTag.METRIC)

    result = _explain_with_outputs(tmp_path, [stats], {stats: None})

    assert result["will_run"] is True
    assert result["reason"] == "Metric output missing: s:stats"


def test_get_stage_explanation_missing_cacheable_output_keeps_cache_reason(
    tmp_path: pathlib.Path,
) -> None:
    """A cacheable output pull could restore keeps the cache reason, even beside a metric edit."""
    table = _output_ref("s", "table", ArtifactTag.DATA)
    stats = _output_ref("s", "stats", ArtifactTag.METRIC)

    result = _explain_with_outputs(tmp_path, [stats, table], {stats: "value: 999\n", table: None})

    assert result["will_run"] is True
    assert result["reason"] == explain.OUTPUTS_MISSING_REASON


def test_get_stage_explanation_code_changed(tmp_path: pathlib.Path) -> None:
    """Returns detailed code changes when code differs."""
    state_dir = tmp_path / ".pivot"
    _write_lock(
        state_dir,
        "code_stage",
        {"self:code_stage": "old_hash", "func:helper": "helper_old"},
    )

    result = explain.get_stage_explanation(
        stage_name="code_stage",
        fingerprint={"self:code_stage": "new_hash", "func:helper": "helper_old"},
        deps={},
        outs=[],
        params_instance=None,
        overrides=None,
        state_dir=state_dir,
        store=_store(tmp_path),
        lock_data=_lock(state_dir, "code_stage"),
    )

    assert result["will_run"] is True
    assert result["reason"] == "Code changed"
    assert len(result["code_changes"]) == 1
    assert result["code_changes"][0]["key"] == "self:code_stage"
    assert result["code_changes"][0]["change_type"] == "modified"


def test_get_stage_explanation_params_changed(tmp_path: pathlib.Path) -> None:
    """Returns detailed parameter changes when parameters differ."""

    class TrainParams(pydantic.BaseModel):
        learning_rate: float = 0.01

    state_dir = tmp_path / ".pivot"
    _write_lock(
        state_dir,
        "param_stage",
        {"self:param_stage": "abc"},
        params={"learning_rate": 0.01},
    )
    result = explain.get_stage_explanation(
        stage_name="param_stage",
        fingerprint={"self:param_stage": "abc"},
        deps={},
        outs=[],
        params_instance=TrainParams(),
        overrides={"param_stage": {"learning_rate": 0.001}},
        state_dir=state_dir,
        store=_store(tmp_path),
        lock_data=_lock(state_dir, "param_stage"),
    )

    assert result["will_run"] is True
    assert result["reason"] == "Params changed"
    assert len(result["param_changes"]) == 1
    assert result["param_changes"][0]["key"] == "learning_rate"
    assert result["param_changes"][0]["old_value"] == 0.01
    assert result["param_changes"][0]["new_value"] == 0.001


def test_get_stage_explanation_deps_changed(tmp_path: pathlib.Path) -> None:
    """Returns detailed dependency changes when an input hash differs."""
    data_path = tmp_path / "data.csv"
    data_path.write_text("id,value\n1,10\n")
    dep = _input_ref("data.csv")
    state_dir = tmp_path / ".pivot"
    _write_lock(
        state_dir,
        "dep_stage",
        {"self:dep_stage": "abc"},
        dep_hashes={dep.identity: {"hash": "0d0d0d0d0d0d0d0d"}},
    )

    result = explain.get_stage_explanation(
        stage_name="dep_stage",
        fingerprint={"self:dep_stage": "abc"},
        deps={"input": dep},
        outs=[],
        params_instance=None,
        overrides=None,
        state_dir=state_dir,
        store=_store(tmp_path),
        lock_data=_lock(state_dir, "dep_stage"),
    )

    assert result["will_run"] is True
    assert "dep" in result["reason"].lower() or "input" in result["reason"].lower()
    assert len(result["dep_changes"]) == 1
    assert result["dep_changes"][0]["old_hash"] == "0d0d0d0d0d0d0d0d"
    assert result["dep_changes"][0]["change_type"] == "modified"


def test_get_stage_explanation_multiple_changes(tmp_path: pathlib.Path) -> None:
    """Returns all code, parameter, and dependency changes."""

    class Params(pydantic.BaseModel):
        epochs: int = 10

    data_path = tmp_path / "data.csv"
    data_path.write_text("x,y\n1,2\n")
    dep = _input_ref("data.csv")
    state_dir = tmp_path / ".pivot"
    _write_lock(
        state_dir,
        "multi_stage",
        {"self:multi_stage": "old_code"},
        params={"epochs": 5},
        dep_hashes={dep.identity: {"hash": "0000000000000001"}},
    )

    result = explain.get_stage_explanation(
        stage_name="multi_stage",
        fingerprint={"self:multi_stage": "new_code"},
        deps={"input": dep},
        outs=[],
        params_instance=Params(),
        overrides=None,
        state_dir=state_dir,
        store=_store(tmp_path),
        lock_data=_lock(state_dir, "multi_stage"),
    )

    assert result["will_run"] is True
    assert len(result["code_changes"]) >= 1
    assert len(result["param_changes"]) >= 1
    assert len(result["dep_changes"]) >= 1


def test_get_stage_explanation_missing_deps(tmp_path: pathlib.Path) -> None:
    """Returns will_run when an input dependency is missing."""
    dep = _input_ref("data.csv")
    state_dir = tmp_path / ".pivot"
    _write_lock(state_dir, "missing_deps_stage", {"self:missing_deps_stage": "abc"})

    result = explain.get_stage_explanation(
        stage_name="missing_deps_stage",
        fingerprint={"self:missing_deps_stage": "abc"},
        deps={"input": dep},
        outs=[],
        params_instance=None,
        overrides=None,
        state_dir=state_dir,
        store=_store(tmp_path),
        lock_data=_lock(state_dir, "missing_deps_stage"),
    )

    assert result["will_run"] is True
    assert "missing" in result["reason"].lower()


def test_get_stage_explanation_invalid_params(tmp_path: pathlib.Path) -> None:
    """Returns will_run with details when parameter overrides are invalid."""

    class StrictParams(pydantic.BaseModel):
        model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="forbid")
        learning_rate: float = 0.01

    state_dir = tmp_path / ".pivot"
    _write_lock(
        state_dir,
        "stage",
        {"self:stage": "abc"},
        params={"learning_rate": 0.01},
    )

    result = explain.get_stage_explanation(
        stage_name="stage",
        fingerprint={"self:stage": "abc"},
        deps={},
        outs=[],
        params_instance=StrictParams(),
        overrides={"stage": {"unknown_field": "value"}},
        state_dir=state_dir,
        store=_store(tmp_path),
        lock_data=_lock(state_dir, "stage"),
    )

    assert result["will_run"] is True
    assert "invalid" in result["reason"].lower() or "error" in result["reason"].lower()


def test_get_stage_explanation_invalid_params_does_not_echo_a_secret(
    tmp_path: pathlib.Path,
) -> None:
    class SecretParams(pydantic.BaseModel):
        pin: pydantic.Secret[int] = pydantic.Secret[int](1)

    state_dir = tmp_path / ".pivot"
    _write_lock(state_dir, "stage", {"self:stage": "abc"}, params={"pin": "x"})

    result = explain.get_stage_explanation(
        stage_name="stage",
        fingerprint={"self:stage": "abc"},
        deps={},
        outs=[],
        params_instance=SecretParams(),
        overrides={"stage": {"pin": "99x999"}},
        state_dir=state_dir,
        store=_store(tmp_path),
        lock_data=_lock(state_dir, "stage"),
    )

    assert result["will_run"] is True
    assert "Invalid params.yaml" in result["reason"]
    assert "pin" in result["reason"]
    assert "99x999" not in result["reason"]


@pytest.mark.parametrize(
    ("stage_name", "create_lock"),
    [
        pytest.param("no_lock_stage", False, id="no_lock"),
        pytest.param("unchanged_stage", True, id="unchanged"),
    ],
)
def test_get_stage_explanation_force_without_changes(
    tmp_path: pathlib.Path, stage_name: str, create_lock: bool
) -> None:
    """Force=True reports a forced run when there are no other changes."""
    state_dir = tmp_path / ".pivot"
    fingerprint = {f"self:{stage_name}": "abc123"}
    if create_lock:
        _write_lock(state_dir, stage_name, fingerprint)

    result = explain.get_stage_explanation(
        stage_name=stage_name,
        fingerprint=fingerprint,
        deps={},
        outs=[],
        params_instance=None,
        overrides=None,
        state_dir=state_dir,
        store=_store(tmp_path),
        lock_data=_lock(state_dir, stage_name),
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
        upstream_stale=[],
    )


def test_get_stage_explanation_force_with_code_changes(tmp_path: pathlib.Path) -> None:
    """Code changes remain the primary explanation for a forced run."""
    state_dir = tmp_path / ".pivot"
    _write_lock(state_dir, "code_stage", {"self:code_stage": "old_hash"})

    result = explain.get_stage_explanation(
        stage_name="code_stage",
        fingerprint={"self:code_stage": "new_hash"},
        deps={},
        outs=[],
        params_instance=None,
        overrides=None,
        state_dir=state_dir,
        store=_store(tmp_path),
        lock_data=_lock(state_dir, "code_stage"),
        force=True,
    )

    assert result["will_run"] is True
    assert result["is_forced"] is True
    assert result["reason"] == "Code changed"
    assert len(result["code_changes"]) == 1
    assert result["code_changes"][0]["key"] == "self:code_stage"


def test_get_stage_explanation_force_with_missing_deps(tmp_path: pathlib.Path) -> None:
    """Missing dependencies remain the primary explanation for a forced run."""
    dep = _input_ref("data.csv")
    state_dir = tmp_path / ".pivot"
    _write_lock(state_dir, "dep_stage", {"self:dep_stage": "abc"})

    result = explain.get_stage_explanation(
        stage_name="dep_stage",
        fingerprint={"self:dep_stage": "abc"},
        deps={"input": dep},
        outs=[],
        params_instance=None,
        overrides=None,
        state_dir=state_dir,
        store=_store(tmp_path),
        lock_data=_lock(state_dir, "dep_stage"),
        force=True,
    )

    assert result["will_run"] is True
    assert result["is_forced"] is True
    assert "missing" in result["reason"].lower()


def test_get_pipeline_explanations_upstream_propagation(
    mock_discovery: compose.Pipeline,
) -> None:
    """get_pipeline_explanations propagates staleness to downstream stages."""
    input_path = mock_discovery.root / "data" / "raw" / "input.txt"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("data")

    initial_pipeline = _build_upstream_pipeline(mock_discovery.root, value=1)
    executor.run(pipeline=initial_pipeline)

    pipeline = _build_upstream_pipeline(mock_discovery.root, value=2)
    explanations = status.get_pipeline_explanations(
        stages=None,
        single_stage=False,
        all_stages=pipeline.snapshot(),
        pipeline=pipeline,
    )

    stage_b_exp = next(
        (
            explanation
            for explanation in explanations
            if explanation["stage_name"] == "test/_upstream_stage_b"
        ),
        None,
    )
    assert stage_b_exp is not None, "test/_upstream_stage_b should be in explanations"
    assert stage_b_exp["will_run"] is True, (
        "test/_upstream_stage_b should run due to upstream staleness"
    )
    assert "upstream_stale" in stage_b_exp, "test/_upstream_stage_b should have upstream_stale"
    assert "test/_upstream_stage_a" in stage_b_exp["upstream_stale"], (
        "test/_upstream_stage_b should list test/_upstream_stage_a as stale upstream"
    )


# =============================================================================
# A producer's directory output: a consumer's one file, hashed from the producer's listing
# =============================================================================


class _ProducedDirectory(TypedDict):
    """What status hands explain for a run producer whose directory output is gone."""

    locks: dict[str, LockData | None]
    deps: list[ArtifactRef]
    stages_dirs: dict[str, pathlib.Path]
    file_identity: ArtifactIdentity
    committed_listing: pathlib.Path
    cache_dir: pathlib.Path


def _run_directory_pipeline(root: pathlib.Path) -> _ProducedDirectory:
    """Run a directory producer and a consumer of its file ``sub/b.txt``, then delete the
    directory, as a fresh clone has it before ``pivot pull``: no file is left to hash."""
    with compose.Pipeline("test", root=root) as pipeline:
        _one_file_consumer(_directory_producer()["sub/b.txt"])
    executor.run(pipeline=pipeline)
    shutil.rmtree(root / "data" / "test" / "_directory_producer")
    all_stages = pipeline.snapshot()
    state_dir = config.get_state_dir()
    (dep,) = all_stages["test/_one_file_consumer"]["deps"].values()
    assert dep.directory is not None, "the consumer reads one file of the directory output"
    stages_dirs = {name: explain.stages_dir(info, state_dir) for name, info in all_stages.items()}
    return _ProducedDirectory(
        locks=explain.read_locks(all_stages, all_stages, state_dir),
        deps=[dep],
        stages_dirs=stages_dirs,
        file_identity=dep.identity,
        committed_listing=listing.output_listing_path(
            stages_dirs["test/_directory_producer"], dep.directory
        ),
        cache_dir=config.get_cache_dir() / "files",
    )


def test_producer_output_hashes_reads_a_file_hash_from_the_committed_listing(
    mock_discovery: compose.Pipeline,
) -> None:
    """With the directory gone and no cache, the producer's committed listing gives the hash of
    the file a consumer reads, and adds no file nothing reads."""
    produced = _run_directory_pipeline(mock_discovery.root)

    hashes = explain.producer_output_hashes(
        produced["locks"], produced["deps"], produced["stages_dirs"], None
    )

    assert hashes[produced["file_identity"]] == {"hash": _BETA_HASH}
    assert ArtifactIdentity("test/_directory_producer", "a.txt") not in hashes, (
        "a file no dependency reads is not added"
    )


def test_producer_output_hashes_reads_the_cache_object_without_a_committed_listing(
    mock_discovery: compose.Pipeline,
) -> None:
    """A listing missing from the commit is read from the cache object at the directory hash."""
    produced = _run_directory_pipeline(mock_discovery.root)
    produced["committed_listing"].unlink()

    hashes = explain.producer_output_hashes(
        produced["locks"], produced["deps"], produced["stages_dirs"], produced["cache_dir"]
    )

    assert hashes[produced["file_identity"]] == {"hash": _BETA_HASH}


def test_producer_output_hashes_leaves_the_file_to_hash_when_no_listing_exists(
    mock_discovery: compose.Pipeline,
) -> None:
    """With neither the committed listing nor a cache, the file is left for explain to hash."""
    produced = _run_directory_pipeline(mock_discovery.root)
    produced["committed_listing"].unlink()
    shutil.rmtree(produced["cache_dir"])

    hashes = explain.producer_output_hashes(
        produced["locks"], produced["deps"], produced["stages_dirs"], produced["cache_dir"]
    )

    assert produced["file_identity"] not in hashes


def test_producer_output_hashes_refuses_a_committed_listing_edited_after_the_run(
    mock_discovery: compose.Pipeline,
) -> None:
    """A committed listing whose bytes no longer hash to the lock's record is refused."""
    produced = _run_directory_pipeline(mock_discovery.root)
    committed = produced["committed_listing"]
    committed.write_bytes(committed.read_bytes().replace(_BETA_HASH.encode(), b"0" * 16))

    with pytest.raises(exceptions.CorruptListingError, match="its bytes hash to"):
        explain.producer_output_hashes(
            produced["locks"], produced["deps"], produced["stages_dirs"], produced["cache_dir"]
        )


# Removed: annotation-style pipeline re-registration is not part of the compositional API.


# =============================================================================
# A tracked directory: a member's hash from the directory's listing
# =============================================================================

# corpus.pvt as pivot 0.5.9 wrote it: its files inline, its hash over their JSON.
_OLD_CORPUS_PVT = f"""\
path: corpus
hash: 9f3a1c2b4d5e6f70
size: 8
num_files: 2
manifest:
- relpath: a.csv
  hash: {_A_CSV_HASH}
  size: 4
  isexec: false
- relpath: sub/b.csv
  hash: {_B_CSV_HASH}
  size: 4
  isexec: false
"""


class _TrackedCorpus(TypedDict):
    """What ``pivot status --allow-missing`` hands explain for the tracked ``corpus``."""

    tracked_files: dict[str, track.PvtData]
    tracked_trie: pygtrie.Trie[str]
    corpus: pathlib.Path
    cache_dir: pathlib.Path


def _discover_corpus(root: pathlib.Path) -> _TrackedCorpus:
    """Discover the tracked paths under ``root`` as ``pivot status --allow-missing`` does."""
    tracked_files, tracked_trie = status._discover_tracked_files(allow_missing=True)
    assert tracked_files is not None and tracked_trie is not None
    return _TrackedCorpus(
        tracked_files=tracked_files,
        tracked_trie=tracked_trie,
        corpus=project.normalize_path(root / "corpus"),
        cache_dir=config.get_cache_dir() / "files",
    )


def _track_corpus(root: pathlib.Path, runner: click.testing.CliRunner) -> _TrackedCorpus:
    """``pivot track corpus`` over a.csv and sub/b.csv, then delete the directory, as a fresh
    clone has it before ``pivot pull``: only a listing can give a member's hash."""
    corpus = root / "corpus"
    (corpus / "sub").mkdir(parents=True)
    (corpus / "a.csv").write_bytes(b"a\n1\n")
    (corpus / "sub" / "b.csv").write_bytes(b"b\n2\n")
    result = runner.invoke(cli.cli, ["track", "corpus"])
    assert result.exit_code == 0, result.output
    shutil.rmtree(corpus)
    return _discover_corpus(root)


def test_find_tracked_hash_exact_match() -> None:
    """Finds the hash for an exact tracked-file match."""
    tracked_files: dict[str, track.PvtData] = {
        "/project/data.csv": track.PvtData(path="data.csv", hash="abc123", size=100)
    }
    tracked_trie: pygtrie.Trie[str] = pygtrie.Trie()
    tracked_trie[("/", "project", "data.csv")] = "/project/data.csv"

    result = explain._find_tracked_hash(
        pathlib.Path("/project/data.csv"), tracked_files, tracked_trie, None
    )

    assert result is not None
    assert result["hash"] == "abc123"


def test_find_tracked_hash_reads_a_member_from_the_committed_listing(
    mock_discovery: compose.Pipeline, runner: click.testing.CliRunner
) -> None:
    """With the directory gone and no cache, ``corpus.pvt.listing`` gives a member's hash."""
    tracked = _track_corpus(mock_discovery.root, runner)

    result = explain._find_tracked_hash(
        tracked["corpus"] / "sub" / "b.csv", tracked["tracked_files"], tracked["tracked_trie"], None
    )

    assert result == {"hash": _B_CSV_HASH}


def test_find_tracked_hash_reads_a_member_from_the_cache_without_a_committed_listing(
    mock_discovery: compose.Pipeline, runner: click.testing.CliRunner
) -> None:
    """A listing missing from the commit is read from the cache object at the directory hash."""
    tracked = _track_corpus(mock_discovery.root, runner)
    (mock_discovery.root / "corpus.pvt.listing").unlink()

    result = explain._find_tracked_hash(
        tracked["corpus"] / "sub" / "b.csv",
        tracked["tracked_files"],
        tracked["tracked_trie"],
        tracked["cache_dir"],
    )

    assert result == {"hash": _B_CSV_HASH}


def test_find_tracked_hash_has_no_hash_for_a_member_when_no_listing_exists(
    mock_discovery: compose.Pipeline, runner: click.testing.CliRunner
) -> None:
    """With neither the committed listing nor a cache, a member has no tracked hash, so explain
    falls back to the stage lock's hash."""
    tracked = _track_corpus(mock_discovery.root, runner)
    (mock_discovery.root / "corpus.pvt.listing").unlink()
    shutil.rmtree(tracked["cache_dir"])

    assert (
        explain._find_tracked_hash(
            tracked["corpus"] / "sub" / "b.csv",
            tracked["tracked_files"],
            tracked["tracked_trie"],
            tracked["cache_dir"],
        )
        is None
    )


def test_find_tracked_hash_not_tracked() -> None:
    """Returns None for an untracked path."""
    assert (
        explain._find_tracked_hash(
            pathlib.Path("/project/untracked.csv"),
            {},
            pygtrie.Trie(),
            None,
        )
        is None
    )


def test_find_tracked_hash_not_in_listing(
    mock_discovery: compose.Pipeline, runner: click.testing.CliRunner
) -> None:
    """Returns None for a file inside a tracked directory that its listing does not hold."""
    tracked = _track_corpus(mock_discovery.root, runner)

    assert (
        explain._find_tracked_hash(
            tracked["corpus"] / "sub" / "c.csv",
            tracked["tracked_files"],
            tracked["tracked_trie"],
            tracked["cache_dir"],
        )
        is None
    )


def test_find_tracked_hash_refuses_a_committed_listing_edited_after_tracking(
    mock_discovery: compose.Pipeline, runner: click.testing.CliRunner
) -> None:
    """A committed listing whose bytes no longer hash to the .pvt's record is refused, naming it."""
    tracked = _track_corpus(mock_discovery.root, runner)
    committed = mock_discovery.root / "corpus.pvt.listing"
    committed.write_bytes(
        committed.read_bytes().replace(_B_CSV_HASH.encode(), _A_CSV_HASH.encode())
    )

    with pytest.raises(exceptions.CorruptListingError, match=r"corpus\.pvt\.listing: its bytes"):
        explain._find_tracked_hash(
            tracked["corpus"] / "sub" / "b.csv",
            tracked["tracked_files"],
            tracked["tracked_trie"],
            tracked["cache_dir"],
        )


def test_find_tracked_hash_reads_a_member_from_an_old_directory_pvt_with_one_warning(
    mock_discovery: compose.Pipeline, caplog: pytest.LogCaptureFixture
) -> None:
    """A directory .pvt the previous release wrote lists its files inline: explain reads a
    member from it, warns once with the command that converts it, and rewrites nothing."""
    pvt_path = mock_discovery.root / "corpus.pvt"
    pvt_path.write_text(_OLD_CORPUS_PVT)

    with caplog.at_level(logging.WARNING, logger="pivot.storage.track"):
        tracked = _discover_corpus(mock_discovery.root)
        result = explain._find_tracked_hash(
            tracked["corpus"] / "sub" / "b.csv",
            tracked["tracked_files"],
            tracked["tracked_trie"],
            tracked["cache_dir"],
        )

    assert result == {"hash": _B_CSV_HASH}
    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 1, warnings
    assert str(pvt_path) in warnings[0] and "pivot track --force" in warnings[0], warnings
    assert pvt_path.read_text() == _OLD_CORPUS_PVT, "an old .pvt is never rewritten"
    assert not (mock_discovery.root / "corpus.pvt.listing").exists(), "explain writes no listing"


def test_get_stage_explanation_with_allow_missing_uses_pvt_hash(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uses a tracked hash when --allow-missing cannot read the input file."""
    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project, "_project_root_cache", None)
    data_path = tmp_path / "data.csv"
    normalized_path = str(project.normalize_path(data_path))
    dep = _input_ref("data.csv")
    _write_lock(
        state_dir,
        "pvt_stage",
        {"self:pvt_stage": "abc"},
        dep_hashes={dep.identity: {"hash": "1234123412341234"}},
    )
    tracked_files: dict[str, track.PvtData] = {
        normalized_path: track.PvtData(path="data.csv", hash="1234123412341234", size=100)
    }
    tracked_trie: pygtrie.Trie[str] = pygtrie.Trie()
    tracked_trie[pathlib.Path(normalized_path).parts] = normalized_path

    result = explain.get_stage_explanation(
        stage_name="pvt_stage",
        fingerprint={"self:pvt_stage": "abc"},
        deps={"data": dep},
        outs=[],
        params_instance=None,
        overrides=None,
        state_dir=state_dir,
        store=_store(tmp_path),
        lock_data=_lock(state_dir, "pvt_stage"),
        allow_missing=True,
        tracked_files=tracked_files,
        tracked_trie=tracked_trie,
    )

    assert "missing" not in result["reason"].lower()
    assert result["will_run"] is False


def test_get_stage_explanation_allow_missing_untracked_still_fails(
    tmp_path: pathlib.Path,
) -> None:
    """Still reports missing dependencies when no tracked or locked hash exists."""
    dep = _input_ref("data.csv")
    state_dir = tmp_path / ".pivot"
    _write_lock(state_dir, "stage", {"self:stage": "abc"})

    result = explain.get_stage_explanation(
        stage_name="stage",
        fingerprint={"self:stage": "abc"},
        deps={"data": dep},
        outs=[],
        params_instance=None,
        overrides=None,
        state_dir=state_dir,
        store=_store(tmp_path),
        lock_data=_lock(state_dir, "stage"),
        allow_missing=True,
        tracked_files={},
        tracked_trie=pygtrie.Trie(),
    )

    assert result["will_run"] is True
    assert "missing" in result["reason"].lower()


def test_get_stage_explanation_allow_missing_uses_lock_file_hash(
    tmp_path: pathlib.Path,
) -> None:
    """Uses the locked hash when --allow-missing has no tracked-file manifest."""
    dep = _input_ref("data.csv")
    state_dir = tmp_path / ".pivot"
    _write_lock(
        state_dir,
        "lock_fallback_stage",
        {"self:lock_fallback_stage": "abc"},
        dep_hashes={dep.identity: {"hash": "4321432143214321"}},
    )

    result = explain.get_stage_explanation(
        stage_name="lock_fallback_stage",
        fingerprint={"self:lock_fallback_stage": "abc"},
        deps={"data": dep},
        outs=[],
        params_instance=None,
        overrides=None,
        state_dir=state_dir,
        store=_store(tmp_path),
        lock_data=_lock(state_dir, "lock_fallback_stage"),
        allow_missing=True,
        tracked_files={},
        tracked_trie=pygtrie.Trie(),
    )

    assert "missing" not in result["reason"].lower()
    assert result["will_run"] is False
