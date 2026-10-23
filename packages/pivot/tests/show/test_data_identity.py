# pyright: reportMissingImports=false
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
        incremental_inputs={},
        params=None,
        resources={},
        variant=None,
        signature=inspect.signature(_stage_func),
        fingerprint={"_code": "fake_hash"},
        params_arg_name=None,
        state_dir=None,
        collection_params={},
        no_fingerprint=False,
        code_deps=[],
        input_paths={},
        unsafe_fingerprint=False,
    )


def _use_store(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    workspace_store = store_mod.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="eval",
        input_bindings={},
    )
    monkeypatch.setattr(cli_helpers, "get_workspace_store", lambda: workspace_store)


def _register(monkeypatch: pytest.MonkeyPatch, ref: types.ArtifactRef) -> None:
    stage_info = {"train": _make_stage_info("train", [ref])}

    def _get_stage(name: str) -> RegistryStageInfo:
        return stage_info[name]

    monkeypatch.setattr(cli_helpers, "list_stages", lambda: ["train"])
    monkeypatch.setattr(cli_helpers, "get_stage", _get_stage)


def _mock_head_files(monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes]) -> None:
    def _read_files_from_head(paths: list[str]) -> dict[str, bytes]:
        return files

    monkeypatch.setattr(git, "read_files_from_head", _read_files_from_head)


def test_get_data_hashes_from_head_detects_format_from_ref_loader(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A data output is diffable by its loader's format, independent of its identity string."""
    _use_store(tmp_path, monkeypatch)
    _register(
        monkeypatch, _make_artifact_ref("train", None, types.ArtifactTag.DATA, loaders.JSON())
    )
    _mock_head_files(monkeypatch, {})

    assert data_module.detect_format(pathlib.Path("train")) == DataFileFormat.UNKNOWN

    result = data_module.get_data_hashes_from_head()

    assert result == {"data/train.json": None}


def test_extract_output_hashes_from_lock_uses_identity_keys() -> None:
    lock_data: StorageLockData = {
        "code_manifest": {},
        "params": {},
        "deps": [],
        "outs": [
            {"key": None, "hash": "deadbeef"},
            {"key": "model", "hash": "beadfeed"},
        ],
    }

    result = show_common.extract_output_hashes_from_lock(lock_data, "train")

    assert result == {"train": "deadbeef", "train:model": "beadfeed"}


def test_get_data_hashes_from_head_uses_identity_keys_and_store_paths(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_store(tmp_path, monkeypatch)
    _register(monkeypatch, _make_artifact_ref("train", None, types.ArtifactTag.DATA, loaders.CSV()))

    lock_content = b"""
code_manifest: {}
params: {}
deps: []
outs:
  - key: null
    hash: deadbeef00000000
    tag: data
"""

    _mock_head_files(monkeypatch, {".pivot/stages/train.lock": lock_content})

    result = data_module.get_data_hashes_from_head()

    assert result == {"data/train.csv": "deadbeef00000000"}
