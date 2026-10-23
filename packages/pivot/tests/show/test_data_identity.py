# pyright: reportMissingImports=false
from __future__ import annotations

import inspect
import pathlib

import pytest

from pivot import exceptions, git, loaders, project, types
from pivot.cli import helpers as cli_helpers
from pivot.registry import RegistryStageInfo
from pivot.show import common as show_common
from pivot.show import data as data_module
from pivot.show import selection as selection_module
from pivot.storage import store as store_mod
from pivot.types import DataFileFormat, StorageLockData


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
    monkeypatch.setattr(cli_helpers, "get_all_stages", lambda: stage_info)


def _mock_head_files(monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes]) -> None:
    def _read_files_from_head(paths: list[str]) -> dict[str, bytes]:
        return files

    monkeypatch.setattr(git, "read_files_from_head", _read_files_from_head)


def _head_hashes() -> dict[str, str | None]:
    selection = selection_module.select_data_files(["."])
    assert selection is not None
    return selection["head_hashes"]


def test_select_data_files_detects_format_from_ref_loader(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A data output is diffable by its loader's format, independent of its identity string."""
    _use_store(tmp_path, monkeypatch)
    _register(
        monkeypatch, _make_artifact_ref("train", None, types.ArtifactTag.DATA, loaders.JSON())
    )
    _mock_head_files(monkeypatch, {})

    assert data_module.detect_format(pathlib.Path("train")) == DataFileFormat.UNKNOWN

    result = _head_hashes()

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


def test_select_data_files_expands_directory_manifest(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory output records one head hash per comparable file in its manifest."""
    _use_store(tmp_path, monkeypatch)
    _register(
        monkeypatch,
        _make_artifact_ref("train", None, types.ArtifactTag.DIRECTORY, loaders.CSV()),
    )
    lock_content = b"""
code_manifest: {}
params: {}
deps: []
outs:
  - key: null
    hash: deadbeef00000000
    manifest:
      - relpath: first.csv
        hash: abcd123400000000
        size: 4
        isexec: false
      - relpath: notes.yaml
        hash: beadfeed00000000
        size: 5
        isexec: false
"""
    _mock_head_files(monkeypatch, {".pivot/stages/train.lock": lock_content})

    result = _head_hashes()

    assert result == {"data/train/first.csv": "abcd123400000000"}


def test_select_data_files_refuses_a_manifest_path_outside_its_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed lock whose directory manifest leaves the directory is refused, naming the lock
    and the path, before any file is read through it."""
    _use_store(tmp_path, monkeypatch)
    _register(
        monkeypatch,
        _make_artifact_ref("train", None, types.ArtifactTag.DIRECTORY, loaders.CSV()),
    )
    lock_content = b"""
code_manifest: {}
params: {}
deps: []
outs:
  - key: null
    hash: deadbeef00000000
    manifest:
      - relpath: ../../secret.csv
        hash: abcd123400000000
        size: 4
        isexec: false
"""
    _mock_head_files(monkeypatch, {".pivot/stages/train.lock": lock_content})

    with pytest.raises(exceptions.SecurityValidationError) as excinfo:
        _head_hashes()

    assert ".pivot/stages/train.lock" in str(excinfo.value)
    assert "outs[0].manifest[0].relpath: '../../secret.csv'" in str(excinfo.value)


def test_select_data_files_expands_an_unlocked_path_directory_from_the_workspace(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path output that is a directory on disk, with no HEAD lock, compares its data files."""
    _use_store(tmp_path, monkeypatch)
    _register(
        monkeypatch, _make_artifact_ref("train", None, types.ArtifactTag.DATA, loaders.PathOnly())
    )
    directory = tmp_path / "data" / "train.dat"
    directory.mkdir(parents=True)
    (directory / "rows.csv").write_text("id\n1\n")
    _mock_head_files(monkeypatch, {})

    selection = selection_module.select_data_files(["data/train.dat/rows.csv"])
    assert selection is not None
    assert selection["head_hashes"] == {}
    assert list(selection["workspace_paths"]) == ["data/train.dat/rows.csv"]


def test_select_data_files_reads_a_path_directory_output_as_a_directory_from_its_lock(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DATA-tagged output whose HEAD lock records a manifest is a directory output, even
    when the workspace no longer holds it."""
    _use_store(tmp_path, monkeypatch)
    _register(
        monkeypatch, _make_artifact_ref("train", None, types.ArtifactTag.DATA, loaders.PathOnly())
    )
    lock_content = b"""
code_manifest: {}
params: {}
deps: []
outs:
  - key: null
    hash: deadbeef00000000
    manifest:
      - relpath: rows.csv
        hash: abcd123400000000
        size: 4
        isexec: false
"""
    _mock_head_files(monkeypatch, {".pivot/stages/train.lock": lock_content})

    selection = selection_module.select_data_files(["data/train.dat/rows.csv"])
    assert selection is not None
    assert selection["head_hashes"] == {"data/train.dat/rows.csv": "abcd123400000000"}
    assert selection["workspace_paths"] == {}
