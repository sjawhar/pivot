"""Orphaned lock detection scans the state directories the engine writes locks and listings to."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from pivot import config, loaders, status
from pivot.registry import RegistryStageInfo
from pivot.storage import cache, lock
from pivot.types import ArtifactIdentity, ArtifactRef, ArtifactTag, LockData

if TYPE_CHECKING:
    import pathlib

    import pytest

    from pivot.registry import PipelineLike as Pipeline
    from pivot.storage.listing import HashedDirectory


def _stage_info(
    name: str,
    state_dir: pathlib.Path | None,
    *,
    deps: dict[str, ArtifactRef] | None = None,
    outs: list[ArtifactRef] | None = None,
) -> RegistryStageInfo:
    def _stage_func() -> None:
        return None

    return RegistryStageInfo(
        func=_stage_func,
        name=name,
        deps=deps if deps is not None else {},
        outs=outs if outs is not None else [],
        incremental_inputs={},
        params=None,
        resources={},
        variant=None,
        signature=inspect.signature(_stage_func),
        fingerprint=None,
        params_arg_name=None,
        state_dir=state_dir,
        collection_params={},
        no_fingerprint=False,
        code_deps=[],
        input_paths={},
        unsafe_fingerprint=False,
    )


def _directory_out(stage_name: str, key: str) -> ArtifactRef:
    return ArtifactRef(
        identity=ArtifactIdentity(stage_name, key),
        format=loaders.Text(),
        python_type=dict,
        tag=ArtifactTag.DIRECTORY,
    )


def _input_dep(name: str) -> ArtifactRef:
    return ArtifactRef(
        identity=ArtifactIdentity(name, None),
        format=loaders.PathOnly(),
        python_type=str,
        tag=ArtifactTag.INPUT,
    )


def _hashed_tree(root: pathlib.Path) -> HashedDirectory:
    """A real one-file directory, hashed as a stage hashes a directory output or input."""
    root.mkdir(parents=True)
    (root / "part.csv").write_text("a,b\n")
    return cache.hash_directory(root)


def _write_lock(
    state_dir: pathlib.Path,
    stage_name: str,
    hashed: HashedDirectory,
    *,
    output_keys: tuple[str, ...] = (),
    input_names: tuple[str, ...] = (),
) -> None:
    """Write a stage's lock, and the listing of each directory output and input it records,
    with the writer a run uses. An input without a path binding lives at data/raw/<name>."""
    record = hashed["record"]
    lock.StageLock(stage_name, lock.get_stages_dir(state_dir)).write(
        LockData(
            code_manifest={},
            params={},
            dep_hashes={ArtifactIdentity(name, None): record for name in input_names},
            output_hashes={ArtifactIdentity(stage_name, key): record for key in output_keys},
        ),
        hashed_dirs={record["hash"]: hashed},
        input_paths={ArtifactIdentity(name, None): f"data/raw/{name}" for name in input_names},
    )


def test_find_orphaned_lock_files_scans_each_stage_state_dir(
    set_project_root: pathlib.Path,
    test_pipeline: Pipeline,
) -> None:
    """Each stage's state dir is scanned; a registered output's or input's listing is kept."""
    hashed = _hashed_tree(set_project_root / "tree")
    main_state_dir = set_project_root / ".pivot"
    sub_state_dir = set_project_root / "sub" / ".pivot"
    all_stages = {
        "main/train": _stage_info("main/train", main_state_dir),
        "sub/train": _stage_info(
            "sub/train",
            sub_state_dir,
            deps={"corpus": _input_dep("corpus")},
            outs=[_directory_out("sub/train", "images")],
        ),
    }
    _write_lock(main_state_dir, "main/train", hashed)
    _write_lock(
        sub_state_dir, "sub/train", hashed, output_keys=("images",), input_names=("corpus",)
    )
    _write_lock(
        sub_state_dir,
        "sub/renamed_away",
        hashed,
        output_keys=("images",),
        input_names=("old_corpus",),
    )

    assert status.find_orphaned_lock_files(all_stages, test_pipeline) == [
        "=inputs/data%2Fraw%2Fold_corpus.listing",
        "sub/renamed_away",
        "sub/renamed_away=images.listing",
    ]


def test_find_orphaned_lock_files_includes_default_dir_with_only_custom_stages(
    set_project_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """A custom stage dir does not suppress cleanup of stale locks and listings in the default."""
    hashed = _hashed_tree(set_project_root / "tree")
    default_state_dir = set_project_root / ".pivot"
    custom_state_dir = set_project_root / "custom" / ".pivot"
    monkeypatch.setattr(config, "locate_state_dir", lambda: default_state_dir)
    all_stages = {
        "custom/train": _stage_info(
            "custom/train", custom_state_dir, outs=[_directory_out("custom/train", "images")]
        )
    }
    _write_lock(custom_state_dir, "custom/train", hashed, output_keys=("images",))
    _write_lock(default_state_dir, "deleted/train", hashed, output_keys=("images",))

    assert status.find_orphaned_lock_files(all_stages, test_pipeline) == [
        "deleted/train",
        "deleted/train=images.listing",
    ]


def test_find_orphaned_lock_files_detects_a_stage_moved_to_another_state_dir(
    set_project_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """A stage lock left in its former state directory no longer represents its registered stage."""
    hashed = _hashed_tree(set_project_root / "tree")
    default_state_dir = set_project_root / ".pivot"
    custom_state_dir = set_project_root / "custom" / ".pivot"
    monkeypatch.setattr(config, "locate_state_dir", lambda: default_state_dir)
    all_stages = {
        "train": _stage_info("train", custom_state_dir, outs=[_directory_out("train", "images")])
    }
    _write_lock(custom_state_dir, "train", hashed, output_keys=("images",))
    _write_lock(default_state_dir, "train", hashed, output_keys=("images",))

    assert status.find_orphaned_lock_files(all_stages, test_pipeline) == [
        "train",
        "train=images.listing",
    ]


def test_find_orphaned_lock_files_scans_default_dir_without_stages(
    set_project_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """Deleting every stage does not hide stale locks and listings in the default directory."""
    hashed = _hashed_tree(set_project_root / "tree")
    default_state_dir = set_project_root / ".pivot"
    monkeypatch.setattr(config, "locate_state_dir", lambda: default_state_dir)
    _write_lock(
        default_state_dir,
        "deleted/train",
        hashed,
        output_keys=("images",),
        input_names=("corpus",),
    )

    assert status.find_orphaned_lock_files({}, test_pipeline) == [
        "=inputs/data%2Fraw%2Fcorpus.listing",
        "deleted/train",
        "deleted/train=images.listing",
    ]
