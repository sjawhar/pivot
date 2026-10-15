# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false, reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportUnusedParameter=false, reportUnusedCallResult=false
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from pivot import loaders, types
from pivot.remote import sync
from pivot.storage import lock, track

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
    """include_deps=True includes the hashes of upstream stage outputs the stage depends on."""
    stage_name = "train"
    data_ref = _helper_ref(stage_name, None, types.ArtifactTag.DATA, loaders.CSV())
    source_ref = _helper_ref("source", None, types.ArtifactTag.DATA, loaders.CSV())
    stage_info = _make_stage_info(stage_name, [data_ref], state_dir=tmp_path)
    source_info = _make_stage_info("source", [source_ref], state_dir=tmp_path)

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
        all_stages={stage_name: stage_info, "source": source_info},
    )

    assert "0123000000000000" in hashes
    assert "de45600000000000" in hashes


def _helper_write_lock_with_deps(
    stage_name: str,
    state_dir: pathlib.Path,
    *,
    dep_hashes: dict[types.ArtifactIdentity, types.HashInfo],
    output_hashes: dict[types.ArtifactIdentity, types.HashInfo],
) -> None:
    lock.StageLock(stage_name, lock.get_stages_dir(state_dir)).write(
        types.LockData(
            code_manifest={},
            params={},
            dep_hashes=dep_hashes,
            output_hashes=output_hashes,
        )
    )


def test_get_target_hashes_include_deps_skips_raw_inputs(set_project_root: pathlib.Path) -> None:
    """Raw inputs (git-tracked, produced by no stage) are never cached, so pull must not fetch them."""
    scale_ref = _helper_ref("smoke/scale", None, types.ArtifactTag.DATA, loaders.CSV())
    total_ref = _helper_ref("smoke/total", None, types.ArtifactTag.DATA, loaders.YAML())
    all_stages = {
        "smoke/scale": _make_stage_info("smoke/scale", [scale_ref], state_dir=set_project_root),
        "smoke/total": _make_stage_info("smoke/total", [total_ref], state_dir=set_project_root),
    }
    _helper_write_lock_with_deps(
        "smoke/total",
        set_project_root,
        dep_hashes={
            types.ArtifactIdentity("smoke/scale", None): types.FileHash(hash="5ca1ed0000000000"),
            types.ArtifactIdentity("numbers.csv", None): types.FileHash(hash="0a0a0a0000000000"),
        },
        output_hashes={
            types.ArtifactIdentity("smoke/total", None): types.FileHash(hash="7074a10000000000"),
        },
    )

    hashes = sync.get_target_hashes(
        ["smoke/total"], set_project_root, include_deps=True, all_stages=all_stages
    )

    assert hashes == {"7074a10000000000", "5ca1ed0000000000"}, (
        "Should fetch the stage's output and its upstream stage output, not the raw input"
    )


def test_get_target_hashes_include_deps_keeps_pvt_tracked_inputs(
    set_project_root: pathlib.Path,
) -> None:
    """A `pivot track` input is in the local cache, so push sends it and pull fetches it."""
    total_ref = _helper_ref("smoke/total", None, types.ArtifactTag.DATA, loaders.YAML())
    all_stages = {
        "smoke/total": _make_stage_info("smoke/total", [total_ref], state_dir=set_project_root),
    }
    raw_dir = set_project_root / "data" / "raw"
    raw_dir.mkdir(parents=True)
    track.write_pvt_file(
        raw_dir / "numbers.csv.pvt",
        track.PvtData(path="numbers.csv", hash="7cacced000000000", size=4),
    )
    _helper_write_lock_with_deps(
        "smoke/total",
        set_project_root,
        dep_hashes={
            types.ArtifactIdentity("numbers.csv", None): types.FileHash(hash="7cacced000000000"),
        },
        output_hashes={
            types.ArtifactIdentity("smoke/total", None): types.FileHash(hash="7074a10000000000"),
        },
    )

    hashes = sync.get_target_hashes(
        ["smoke/total"], set_project_root, include_deps=True, all_stages=all_stages
    )

    assert hashes == {"7074a10000000000", "7cacced000000000"}


def test_get_target_hashes_include_deps_skips_imported_inputs(
    set_project_root: pathlib.Path,
) -> None:
    """An input brought in with `pivot import` is written to the workspace, never cached or
    pushed to this project's remote, so pull must not fetch it."""
    total_ref = _helper_ref("smoke/total", None, types.ArtifactTag.DATA, loaders.YAML())
    all_stages = {
        "smoke/total": _make_stage_info("smoke/total", [total_ref], state_dir=set_project_root),
    }
    raw_dir = set_project_root / "data" / "raw"
    raw_dir.mkdir(parents=True)
    track.write_pvt_file(
        raw_dir / "numbers.csv.pvt",
        track.PvtData(
            path="numbers.csv",
            hash="1a1a1a0000000000",
            size=4,
            source=track.ImportSource(
                repo="https://github.com/org/upstream",
                rev="main",
                rev_lock="0123456789abcdef0123456789abcdef01234567",
                stage="up/numbers",
                path="up/numbers",
                remote="s3://upstream-bucket/cache",
            ),
        ),
    )
    _helper_write_lock_with_deps(
        "smoke/total",
        set_project_root,
        dep_hashes={
            types.ArtifactIdentity("numbers.csv", None): types.FileHash(hash="1a1a1a0000000000"),
        },
        output_hashes={
            types.ArtifactIdentity("smoke/total", None): types.FileHash(hash="7074a10000000000"),
        },
    )

    hashes = sync.get_target_hashes(
        ["smoke/total"], set_project_root, include_deps=True, all_stages=all_stages
    )

    assert hashes == {"7074a10000000000"}, "An imported input's blob is not in this remote"


def test_get_target_hashes_include_deps_skips_metric_outputs_of_upstream(
    set_project_root: pathlib.Path,
) -> None:
    """An upstream Metric output is git-tracked, not cached, so it is not fetched."""
    eval_ref = _helper_ref("p/eval", "score", types.ArtifactTag.METRIC, loaders.JSON())
    report_ref = _helper_ref("p/report", None, types.ArtifactTag.DATA, loaders.YAML())
    all_stages = {
        "p/eval": _make_stage_info("p/eval", [eval_ref], state_dir=set_project_root),
        "p/report": _make_stage_info("p/report", [report_ref], state_dir=set_project_root),
    }
    _helper_write_lock_with_deps(
        "p/report",
        set_project_root,
        dep_hashes={
            types.ArtifactIdentity("p/eval", "score"): types.FileHash(hash="3e7a1c0000000000"),
        },
        output_hashes={
            types.ArtifactIdentity("p/report", None): types.FileHash(hash="4e9047000000000a"),
        },
    )

    hashes = sync.get_target_hashes(
        ["p/report"], set_project_root, include_deps=True, all_stages=all_stages
    )

    assert hashes == {"4e9047000000000a"}
