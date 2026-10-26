# pyright: reportMissingImports=false, reportImplicitRelativeImport=false
from __future__ import annotations

import inspect
import pathlib
from typing import TYPE_CHECKING

import pytest
import yaml

from pivot import exceptions, loaders, project, types
from pivot.cli import helpers as cli_helpers
from pivot.registry import RegistryStageInfo
from pivot.show import common as show_common
from pivot.show import data as data_module
from pivot.show import selection as selection_module
from pivot.storage import listing, lock
from pivot.storage import store as store_mod
from pivot.types import DataFileFormat, StorageLockData

if TYPE_CHECKING:
    from conftest import GitRepo

_LOCK_PATH = ".pivot/stages/train.lock"
_LISTING_PATH = ".pivot/stages/train=.listing"  # train's unkeyed output, beside its lock


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


def _commit_head(git_repo: GitRepo, files: dict[str, bytes]) -> None:
    """Commit ``files``, keyed by project-relative path, as git HEAD."""
    root, commit = git_repo
    for rel_path, content in files.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    commit("head")


def _directory_head(listing_bytes: bytes, *, size: int, nfiles: int) -> dict[str, bytes]:
    """The HEAD files of train's directory output: its lock, which records the directory by the
    hash of ``listing_bytes``, its size and its file count, and that listing beside the lock."""
    lock_content = yaml.safe_dump(
        {
            "schema_version": lock.SCHEMA_VERSION,
            "code_manifest": {},
            "params": {},
            "deps": [],
            "outs": [
                {
                    "key": None,
                    "listing": listing.digest(listing_bytes),
                    "size": size,
                    "nfiles": nfiles,
                }
            ],
        }
    )
    return {_LOCK_PATH: lock_content.encode(), _LISTING_PATH: listing_bytes}


def _head_hashes() -> dict[str, str | None]:
    selection = selection_module.select_data_files(["."])
    assert selection is not None
    return selection["head_hashes"]


@pytest.mark.usefixtures("git_repo")
def test_select_data_files_detects_format_from_ref_loader(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A data output is diffable by its loader's format, independent of its identity string."""
    _use_store(tmp_path, monkeypatch)
    _register(
        monkeypatch, _make_artifact_ref("train", None, types.ArtifactTag.DATA, loaders.JSON())
    )

    assert data_module.detect_format(pathlib.Path("train")) == DataFileFormat.UNKNOWN

    result = _head_hashes()

    assert result == {"data/train.json": None}


def test_extract_output_hashes_from_lock_uses_identity_keys() -> None:
    lock_data: StorageLockData = {
        "schema_version": lock.SCHEMA_VERSION,
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


def test_select_data_files_expands_a_directory_from_its_committed_listing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, git_repo: GitRepo
) -> None:
    """A directory output records one head hash per comparable file its committed listing
    names."""
    _use_store(tmp_path, monkeypatch)
    _register(
        monkeypatch,
        _make_artifact_ref("train", None, types.ArtifactTag.DIRECTORY, loaders.CSV()),
    )
    _commit_head(
        git_repo,
        _directory_head(
            b"abcd123400000000 4 - first.csv\nbeadfeed00000000 5 - notes.yaml\n",
            size=9,
            nfiles=2,
        ),
    )

    result = _head_hashes()

    assert result == {"data/train/first.csv": "abcd123400000000"}


@pytest.mark.parametrize(
    ("listing_bytes", "refusal"),
    [
        pytest.param(
            b"abcd123400000000 4 - ../../secret.csv\n",
            "an empty, '.' or '..' segment",
            id="traversal",
        ),
        pytest.param(
            b'abcd123400000000 4 - "secret\\000.csv"\n',
            "a NUL byte",
            id="nul",
        ),
    ],
)
def test_select_data_files_refuses_an_unsafe_path_in_a_committed_listing(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    git_repo: GitRepo,
    listing_bytes: bytes,
    refusal: str,
) -> None:
    """A committed listing whose member path leaves its directory or holds a NUL byte is refused,
    naming the listing and the path's fault, before any file is read through it. The listing
    hashes to the directory hash its lock records, so the path check, not the hash check,
    refuses it."""
    _use_store(tmp_path, monkeypatch)
    _register(
        monkeypatch,
        _make_artifact_ref("train", None, types.ArtifactTag.DIRECTORY, loaders.CSV()),
    )
    _commit_head(git_repo, _directory_head(listing_bytes, size=4, nfiles=1))

    with pytest.raises(exceptions.CorruptListingError) as excinfo:
        _head_hashes()

    assert f"{_LISTING_PATH} at HEAD: line 1: {refusal}" in str(excinfo.value)


@pytest.mark.usefixtures("git_repo")
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

    selection = selection_module.select_data_files(["data/train.dat/rows.csv"])
    assert selection is not None
    assert selection["head_hashes"] == {}
    assert list(selection["workspace_paths"]) == ["data/train.dat/rows.csv"]


def test_select_data_files_reads_a_path_directory_output_as_a_directory_from_its_lock(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, git_repo: GitRepo
) -> None:
    """A DATA-tagged output whose HEAD lock records a directory is a directory output, its
    members read from the listing committed beside the lock, even when the workspace no longer
    holds it."""
    _use_store(tmp_path, monkeypatch)
    _register(
        monkeypatch, _make_artifact_ref("train", None, types.ArtifactTag.DATA, loaders.PathOnly())
    )
    _commit_head(git_repo, _directory_head(b"abcd123400000000 4 - rows.csv\n", size=4, nfiles=1))

    selection = selection_module.select_data_files(["data/train.dat/rows.csv"])
    assert selection is not None
    assert selection["head_hashes"] == {"data/train.dat/rows.csv": "abcd123400000000"}
    assert selection["workspace_paths"] == {}
