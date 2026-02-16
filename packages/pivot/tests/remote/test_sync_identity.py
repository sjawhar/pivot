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
        "params": None,
        "mutex": [],
        "variant": None,
        "signature": None,
        "fingerprint": None,
        "params_arg_name": None,
        "state_dir": state_dir,
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
            types.ArtifactIdentity(stage_name, "model"): types.FileHash(hash="abc123"),
            types.ArtifactIdentity(stage_name, "metrics"): types.FileHash(hash="def456"),
        },
    )

    hashes = sync.get_target_hashes(
        ["train:model"],
        tmp_path,
        include_deps=False,
        all_stages={stage_name: stage_info},
    )

    assert hashes == {"abc123"}


def test_get_file_hash_from_stages_matches_identity_key(tmp_path: pathlib.Path) -> None:
    stage_name = "train"
    data_ref = _helper_ref(stage_name, "model", types.ArtifactTag.DATA, loaders.CSV())
    stage_info = _make_stage_info(stage_name, [data_ref], state_dir=tmp_path)

    _write_stage_lock(
        stage_name,
        tmp_path,
        output_hashes={
            types.ArtifactIdentity(stage_name, "model"): types.FileHash(hash="abc123"),
        },
    )

    out_hash = sync._get_file_hash_from_stages(
        "train:model",
        tmp_path,
        all_stages={stage_name: stage_info},
    )

    assert out_hash == types.FileHash(hash="abc123")
