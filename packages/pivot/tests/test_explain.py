from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, cast

import pydantic
import pygtrie
import pytest

from pivot import explain, skip, types
from pivot.executor import worker
from pivot.storage import lock, state, track

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _write_lock(
    stage_name: str,
    state_dir: pathlib.Path,
    *,
    dep_hashes: dict[types.ArtifactIdentity, types.HashInfo] | None = None,
) -> None:
    stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
    stage_lock.write(
        types.LockData(
            code_manifest={"self:stage": "fp1"},
            params={"lr": 0.1},
            dep_hashes=dep_hashes or {types.ArtifactIdentity("dep.txt", None): {"hash": "h1"}},
            output_hashes={types.ArtifactIdentity(stage_name, None): {"hash": "out1"}},
            merkle_id=None,
        )
    )


def test_tracked_lookup_helpers_handle_exact_nested_and_missing() -> None:
    tracked_trie: pygtrie.Trie[str] = pygtrie.Trie()
    tracked_trie[pathlib.Path("/project/data").parts] = "/project/data"
    manifest = [{"relpath": "nested/file.csv", "hash": "nestedhash", "size": 1, "isexec": False}]
    tracked_files = cast(
        "dict[str, track.PvtData]",
        {
            "/project/data": {
                "path": "data",
                "hash": "dirhash",
                "size": 10,
                "manifest": manifest,
            }
        },
    )

    assert explain._find_tracked_ancestor(
        pathlib.Path("/project/data"), tracked_trie
    ) == pathlib.Path(  # noqa: SLF001
        "/project/data"
    )
    assert explain._find_tracked_hash(  # noqa: SLF001
        pathlib.Path("/project/data"),
        tracked_files,
        tracked_trie,
    ) == {"hash": "dirhash", "manifest": manifest}
    assert explain._find_tracked_hash(  # noqa: SLF001
        pathlib.Path("/project/data/nested/file.csv"),
        tracked_files,
        tracked_trie,
    ) == {"hash": "nestedhash"}
    assert (
        explain._find_tracked_hash(pathlib.Path("/project/other.csv"), tracked_files, tracked_trie)  # noqa: SLF001
        is None
    )


def test_stage_explanation_returns_no_previous_run_without_lock(tmp_path: pathlib.Path) -> None:
    explanation = explain.get_stage_explanation(
        stage_name="train",
        fingerprint={"self:stage": "fp1"},
        deps=["dep.txt"],
        outs_paths=["train"],
        params_instance=None,
        overrides=None,
        state_dir=tmp_path / ".pivot",
        force=False,
    )

    assert explanation["will_run"] is True
    assert explanation["reason"] == "No previous run"


def test_stage_explanation_returns_invalid_params_reason(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    state_dir = tmp_path / ".pivot"
    _write_lock("train", state_dir)

    class _ParamsModel(pydantic.BaseModel):
        lr: float

    with pytest.raises(pydantic.ValidationError) as excinfo:
        _ParamsModel.model_validate({})

    mocker.patch.object(
        explain.parameters,
        "get_effective_params",
        autospec=True,
        side_effect=excinfo.value,
    )

    explanation = explain.get_stage_explanation(
        stage_name="train",
        fingerprint={"self:stage": "fp1"},
        deps=["dep.txt"],
        outs_paths=["train"],
        params_instance=None,
        overrides=None,
        state_dir=state_dir,
    )

    assert explanation["will_run"] is True
    assert explanation["reason"].startswith("Invalid params.yaml:")


def test_stage_explanation_uses_generation_skip_when_available(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    state_dir = tmp_path / ".pivot"
    _write_lock("train", state_dir)
    state_db_path = state_dir / "state.db"
    with state.StateDB(state_db_path):
        pass

    mocker.patch.object(
        explain.parameters,
        "get_effective_params",
        autospec=True,
        return_value={"lr": 0.1},
    )
    mocker.patch.object(worker, "can_skip_via_generation", autospec=True, return_value=True)

    explanation = explain.get_stage_explanation(
        stage_name="train",
        fingerprint={"self:stage": "fp1"},
        deps=["dep.txt"],
        outs_paths=["train"],
        params_instance=None,
        overrides=None,
        state_dir=state_dir,
    )

    assert explanation["will_run"] is False
    assert explanation["reason"] == ""


def test_stage_explanation_reports_missing_and_unreadable_deps(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    state_dir = tmp_path / ".pivot"
    _write_lock(
        "train",
        state_dir,
        dep_hashes={types.ArtifactIdentity("upstream", "data"): {"hash": "from_lock"}},
    )
    mocker.patch.object(
        explain.parameters,
        "get_effective_params",
        autospec=True,
        return_value={"lr": 0.1},
    )
    mocker.patch.object(
        worker,
        "hash_dependencies",
        autospec=True,
        return_value=({}, ["missing.txt"], ["bad.txt"], None),
    )
    mocker.patch.object(skip, "diff_code_manifests", autospec=True, return_value=[])
    mocker.patch.object(skip, "diff_params", autospec=True, return_value=[])
    mocker.patch.object(explain.project, "to_relative_path", autospec=True, side_effect=lambda p: p)

    explanation = explain.get_stage_explanation(
        stage_name="train",
        fingerprint={"self:stage": "fp1"},
        deps=["upstream:data", "missing.txt"],
        outs_paths=["train"],
        params_instance=None,
        overrides=None,
        state_dir=state_dir,
    )

    assert explanation["will_run"] is True
    assert "Missing deps: missing.txt" in explanation["reason"]
    assert "Unreadable deps: bad.txt" in explanation["reason"]


def test_stage_explanation_uses_skip_decision_for_final_reason(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    state_dir = tmp_path / ".pivot"
    dep_file = tmp_path / "dep.txt"
    dep_file.write_text("x")
    _write_lock("train", state_dir)
    mocker.patch.object(
        explain.parameters,
        "get_effective_params",
        autospec=True,
        return_value={"lr": 0.1},
    )
    mocker.patch.object(
        worker,
        "hash_dependencies",
        autospec=True,
        return_value=({"dep.txt": {"hash": "h1"}}, [], [], None),
    )
    mocker.patch.object(
        skip,
        "check_stage",
        autospec=True,
        return_value={
            "changed": False,
            "reason": "",
            "code_changes": [],
            "param_changes": [],
            "dep_changes": [],
        },
    )

    explanation = explain.get_stage_explanation(
        stage_name="train",
        fingerprint={"self:stage": "fp1"},
        deps=[str(dep_file)],
        outs_paths=["train"],
        params_instance=None,
        overrides=None,
        state_dir=state_dir,
        force=False,
        allow_missing=False,
    )

    assert explanation == types.StageExplanation(
        stage_name="train",
        will_run=False,
        is_forced=False,
        reason="",
        code_changes=[],
        param_changes=[],
        dep_changes=[],
        upstream_stale=[],
    )
