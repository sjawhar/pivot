# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false, reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportUnusedParameter=false, reportUnusedCallResult=false
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from pivot import loaders, types
from pivot.remote import sync
from pivot.storage import lock

if TYPE_CHECKING:
    from pivot.registry import RegistryStageInfo


def _helper_ref(
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


def _make_stage_info(
    name: str,
    outs: list[types.ArtifactRef],
    *,
    state_dir: pathlib.Path | None = None,
) -> RegistryStageInfo:
    info: RegistryStageInfo = {  # type: ignore[assignment] - test helper
        "func": lambda: None,
        "name": name,
        "deps": {},
        "outs": outs,
        "incremental_inputs": {},
        "params": None,
        "resources": {},
        "variant": None,
        "signature": None,
        "fingerprint": None,
        "params_arg_name": None,
        "state_dir": state_dir,
        "collection_params": {},
        "no_fingerprint": False,
        "code_deps": [],
        "input_paths": {},
    }
    return info


def _write_stage_lock(
    stage_name: str,
    state_dir: pathlib.Path,
    *,
    output_hashes: dict[types.ArtifactIdentity, types.HashInfo],
) -> None:
    stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
    stage_lock.write(
        types.LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes=output_hashes,
        )
    )


def test_get_target_hashes_resolves_identity_key(tmp_path: pathlib.Path) -> None:
    stage_name = "train"
    data_ref = _helper_ref(stage_name, "model", types.ArtifactTag.DATA, loaders.CSV())
    metric_ref = _helper_ref(stage_name, "metrics", types.ArtifactTag.METRIC, loaders.JSON())
    stage_info = _make_stage_info(stage_name, [data_ref, metric_ref], state_dir=tmp_path)

    _write_stage_lock(
        stage_name,
        tmp_path,
        output_hashes={
            types.ArtifactIdentity(stage_name, "model"): types.FileHash(hash="abc1230000000000"),
            types.ArtifactIdentity(stage_name, "metrics"): types.FileHash(hash="def4560000000000"),
        },
    )

    hashes = sync.get_target_hashes(
        ["train:model"],
        tmp_path,
        include_deps=False,
        all_stages={stage_name: stage_info},
    )

    assert hashes == {"abc1230000000000"}


def test_get_target_hashes_resolves_bare_stage_name_and_key(tmp_path: pathlib.Path) -> None:
    """A bare suffix resolves before remote sync treats the target as a workspace path."""
    stage_name = "smoke/train"
    data_ref = _helper_ref(stage_name, "model", types.ArtifactTag.DATA, loaders.CSV())
    stage_info = _make_stage_info(stage_name, [data_ref], state_dir=tmp_path)
    _write_stage_lock(
        stage_name,
        tmp_path,
        output_hashes={data_ref.identity: types.FileHash(hash="aabbccddeeff0011")},
    )

    bare_stage_hashes = sync.get_target_hashes(
        ["train"], tmp_path, all_stages={stage_name: stage_info}
    )
    bare_key_hashes = sync.get_target_hashes(
        ["train:model"], tmp_path, all_stages={stage_name: stage_info}
    )
    qualified_key_hashes = sync.get_target_hashes(
        ["smoke/train:model"], tmp_path, all_stages={stage_name: stage_info}
    )

    assert bare_stage_hashes == {"aabbccddeeff0011"}
    assert bare_key_hashes == {"aabbccddeeff0011"}
    assert qualified_key_hashes == {"aabbccddeeff0011"}


def test_get_file_hash_from_stages_matches_identity_key(tmp_path: pathlib.Path) -> None:
    stage_name = "train"
    data_ref = _helper_ref(stage_name, "model", types.ArtifactTag.DATA, loaders.CSV())
    stage_info = _make_stage_info(stage_name, [data_ref], state_dir=tmp_path)

    _write_stage_lock(
        stage_name,
        tmp_path,
        output_hashes={
            types.ArtifactIdentity(stage_name, "model"): types.FileHash(hash="abc1230000000000"),
        },
    )

    out_hash = sync._get_file_hash_from_stages(
        "train:model",
        tmp_path,
        all_stages={stage_name: stage_info},
    )

    assert out_hash == types.FileHash(hash="abc1230000000000")


def test_get_target_hashes_empty_targets(tmp_path: pathlib.Path) -> None:
    """Empty target list returns empty set."""
    hashes = sync.get_target_hashes([], tmp_path)
    assert hashes == set()


def test_get_target_hashes_skips_metric_outputs(tmp_path: pathlib.Path) -> None:
    """Stages with only METRIC outputs should be skipped."""
    stage_name = "eval"
    metric_ref = _helper_ref(stage_name, "score", types.ArtifactTag.METRIC, loaders.JSON())
    stage_info = _make_stage_info(stage_name, [metric_ref], state_dir=tmp_path)

    _write_stage_lock(
        stage_name,
        tmp_path,
        output_hashes={
            types.ArtifactIdentity(stage_name, "score"): types.FileHash(hash="3e7c123000000000"),
        },
    )

    hashes = sync.get_target_hashes(
        [stage_name],
        tmp_path,
        all_stages={stage_name: stage_info},
    )

    # Metrics are not cached, so should be empty
    assert hashes == set()


def test_get_target_hashes_multiple_outputs_per_stage(tmp_path: pathlib.Path) -> None:
    """Stage with multiple outputs returns all non-metric hashes."""
    stage_name = "train"
    data_ref = _helper_ref(stage_name, "model", types.ArtifactTag.DATA, loaders.CSV())
    metric_ref = _helper_ref(stage_name, "loss", types.ArtifactTag.METRIC, loaders.JSON())
    stage_info = _make_stage_info(stage_name, [data_ref, metric_ref], state_dir=tmp_path)

    _write_stage_lock(
        stage_name,
        tmp_path,
        output_hashes={
            types.ArtifactIdentity(stage_name, "model"): types.FileHash(hash="da7a123000000000"),
            types.ArtifactIdentity(stage_name, "loss"): types.FileHash(hash="3e7c456000000000"),
        },
    )

    hashes = sync.get_target_hashes(
        [stage_name],
        tmp_path,
        all_stages={stage_name: stage_info},
    )

    # Only data output, not metric
    assert hashes == {"da7a123000000000"}


def test_get_target_hashes_identity_key_target(tmp_path: pathlib.Path) -> None:
    """Identity key target (stage:key) resolves to specific output."""
    stage_name = "train"
    data_ref = _helper_ref(stage_name, "model", types.ArtifactTag.DATA, loaders.CSV())
    stage_info = _make_stage_info(stage_name, [data_ref], state_dir=tmp_path)

    _write_stage_lock(
        stage_name,
        tmp_path,
        output_hashes={
            types.ArtifactIdentity(stage_name, "model"): types.FileHash(hash="abc1230000000000"),
        },
    )

    hashes = sync.get_target_hashes(
        ["train:model"],
        tmp_path,
        all_stages={stage_name: stage_info},
    )

    assert hashes == {"abc1230000000000"}


def test_get_target_hashes_include_deps(tmp_path: pathlib.Path) -> None:
    """include_deps=True includes dependency hashes."""
    stage_name = "train"
    data_ref = _helper_ref(stage_name, None, types.ArtifactTag.DATA, loaders.CSV())
    stage_info = _make_stage_info(stage_name, [data_ref], state_dir=tmp_path)

    _write_stage_lock(
        stage_name,
        tmp_path,
        output_hashes={
            types.ArtifactIdentity(stage_name, None): types.FileHash(hash="0123000000000000"),
        },
    )

    # Manually write lock with deps
    stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(tmp_path))
    stage_lock.write(
        types.LockData(
            code_manifest={},
            params={},
            dep_hashes={
                types.ArtifactIdentity("source", None): types.FileHash(hash="de45600000000000"),
            },
            output_hashes={
                types.ArtifactIdentity(stage_name, None): types.FileHash(hash="0123000000000000"),
            },
        )
    )

    hashes = sync.get_target_hashes(
        [stage_name],
        tmp_path,
        include_deps=True,
        all_stages={stage_name: stage_info},
    )

    assert "0123000000000000" in hashes
    assert "de45600000000000" in hashes
