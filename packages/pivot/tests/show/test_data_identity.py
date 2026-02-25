from __future__ import annotations

import inspect
import pathlib
from typing import TYPE_CHECKING

from pivot import git, loaders, project, types
from pivot.cli import helpers as cli_helpers
from pivot.registry import RegistryStageInfo
from pivot.show import common as show_common
from pivot.show import data as data_module
from pivot.storage import store as store_mod
from pivot.types import DataFileFormat, StorageLockData

if TYPE_CHECKING:
    import pytest


def _make_artifact_ref(
    producer: str,
    key: str | None,
    tag: types.ArtifactTag,
    fmt: loaders.Writer[object] | loaders.Reader[object] | loaders.Loader[object, object],
) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(producer=producer, key=key),
        format=fmt,
        python_type=pathlib.Path,
        tag=tag,
    )


def _make_stage_info(name: str, outs: list[types.ArtifactRef]) -> RegistryStageInfo:
    def _stage_func() -> None:
        pass

    return RegistryStageInfo(
        func=_stage_func,
        name=name,
        deps={},
        outs=outs,
        params=None,
        mutex=[],
        variant=None,
        signature=inspect.signature(_stage_func),
        fingerprint={"_code": "fake_hash"},
        params_arg_name=None,
        state_dir=None,
        collection_params={},
    )


def test_data_rel_path_uses_workspace_store(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace_store = store_mod.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="eval",
        input_bindings={},
    )
    monkeypatch.setattr(cli_helpers, "get_workspace_store", lambda: workspace_store)
    ref = _make_artifact_ref("train", None, types.ArtifactTag.DATA, loaders.CSV())

    rel_path = data_module._data_rel_path(ref, tmp_path)

    assert rel_path == "data/eval/train.csv"
    assert rel_path != "train"


def test_get_data_outputs_from_stages_detects_format_from_ref(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    monkeypatch.setattr(cli_helpers, "get_workspace_store", lambda: None)

    ref = _make_artifact_ref("train", None, types.ArtifactTag.DATA, loaders.JSON())
    stage_info = {"train": _make_stage_info("train", [ref])}

    monkeypatch.setattr(cli_helpers, "list_stages", lambda: ["train"])
    monkeypatch.setattr(cli_helpers, "get_stage", lambda name: stage_info[name])

    assert data_module.detect_format(pathlib.Path("train")) == DataFileFormat.UNKNOWN

    result = data_module.get_data_outputs_from_stages()

    assert result == {"train": "train"}


def test_get_data_outputs_from_stages_uses_store_paths(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    workspace_store = store_mod.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="eval",
        input_bindings={},
    )
    monkeypatch.setattr(cli_helpers, "get_workspace_store", lambda: workspace_store)

    ref = _make_artifact_ref("train", None, types.ArtifactTag.DATA, loaders.CSV())
    stage_info = {"train": _make_stage_info("train", [ref])}

    monkeypatch.setattr(cli_helpers, "list_stages", lambda: ["train"])
    monkeypatch.setattr(cli_helpers, "get_stage", lambda name: stage_info[name])

    result = data_module.get_data_outputs_from_stages()

    assert result == {"train": "data/eval/train.csv"}


def test_extract_output_hashes_from_lock_uses_identity_keys() -> None:
    lock_data: StorageLockData = {
        "code_manifest": {},
        "params": {},
        "deps": [],
        "outs": [
            {"key": "train", "hash": "deadbeef", "tag": "data"},
            {"key": "train:model", "hash": "beadfeed", "tag": "data"},
        ],
    }

    result = show_common.extract_output_hashes_from_lock(lock_data)

    assert result == {"train": "deadbeef", "train:model": "beadfeed"}


def test_get_data_hashes_from_head_uses_identity_keys_and_store_paths(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    workspace_store = store_mod.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="eval",
        input_bindings={},
    )
    monkeypatch.setattr(cli_helpers, "get_workspace_store", lambda: workspace_store)

    ref = _make_artifact_ref("train", None, types.ArtifactTag.DATA, loaders.CSV())
    stage_info = {"train": _make_stage_info("train", [ref])}

    monkeypatch.setattr(cli_helpers, "list_stages", lambda: ["train"])
    monkeypatch.setattr(cli_helpers, "get_stage", lambda name: stage_info[name])

    lock_content = """
code_manifest: {}
params: {}
deps: []
outs:
  - key: train
    hash: deadbeef
    tag: data
"""

    monkeypatch.setattr(
        git,
        "read_files_from_head",
        lambda paths: {".pivot/stages/train.lock": lock_content},
    )

    result = data_module.get_data_hashes_from_head()

    assert result == {"data/eval/train.csv": "deadbeef"}
