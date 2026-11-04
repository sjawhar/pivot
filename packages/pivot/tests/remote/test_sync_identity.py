# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false, reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportUnusedParameter=false, reportUnusedCallResult=false
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pytest

from pivot import exceptions, loaders, types
from pivot.remote import sync
from pivot.storage import cache, listing, lock

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
    input_paths: dict[str, str] | None = None,
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
        "input_paths": input_paths or {},
        "unsafe_fingerprint": False,
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

    out_blobs = sync._get_file_hash_from_stages(
        "train:model",
        tmp_path,
        tmp_path / "cache" / "files",
        all_stages={stage_name: stage_info},
    )

    assert out_blobs == {"abc1230000000000"}


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


def test_get_target_hashes_names_reads_but_no_dependency(tmp_path: pathlib.Path) -> None:
    """A stage target names its own outputs and the objects its lock lists under `reads`, which
    a rerun reads, but not the upstream outputs it depends on: those are targets of their own.
    A pull of it fetches that set and a push of it sends the same set."""
    stage_name = "train"
    data_ref = _helper_ref(stage_name, None, types.ArtifactTag.DATA, loaders.CSV())
    source_ref = _helper_ref("source", None, types.ArtifactTag.DATA, loaders.CSV())
    stage_info = _make_stage_info(stage_name, [data_ref], state_dir=tmp_path)
    source_info = _make_stage_info("source", [source_ref], state_dir=tmp_path)
    lock.StageLock(stage_name, lock.get_stages_dir(tmp_path)).write(
        types.LockData(
            code_manifest={},
            params={},
            dep_hashes={
                types.ArtifactIdentity("source", None): types.FileHash(hash="de45600000000000"),
            },
            output_hashes={
                types.ArtifactIdentity(stage_name, None): types.FileHash(hash="0123000000000000"),
            },
            reads=["7ead000000000000"],
        )
    )

    hashes = sync.get_target_hashes(
        [stage_name],
        tmp_path,
        all_stages={stage_name: stage_info, "source": source_info},
    )

    assert hashes == {"0123000000000000", "7ead000000000000"}


_MEMBER_RECORDED_HASH = "7cacced000000000"
"""The hash the stage's own ``value.txt.pvt`` member records; no cache or remote holds it."""


def _helper_project_with_a_stage_member(
    root: pathlib.Path,
) -> tuple[dict[str, RegistryStageInfo], pathlib.Path]:
    """A stage `b/tree` whose directory output holds `value.txt` and a `value.txt.pvt` the stage
    wrote (BET-412's shape: a `.pvt`-shaped file inside the stage's own output), recorded as the
    stage's member by the listing committed beside its lock. Returns the stages (with the
    consumer `smoke/total`) and the cache root."""
    cache_dir = root / ".pivot" / "cache"
    tree = root / "data" / "b" / "tree"
    tree.mkdir(parents=True)
    value = tree / "value.txt"
    value.write_text("1")
    member = tree / "value.txt.pvt"
    member.write_text(f"path: value.txt\nhash: {_MEMBER_RECORDED_HASH}\nsize: 1\n")
    members = listing.DirListing(
        files=[
            types.DirManifestEntry(
                relpath=entry.name,
                hash=cache.hash_file(entry)[0],
                size=entry.stat().st_size,
                isexec=False,
            )
            for entry in sorted(tree.iterdir())
        ],
        empty_dirs=[],
    )
    data = listing.encode(members)
    record = listing.record_of(members, data)
    tree_ref = _helper_ref("b/tree", None, types.ArtifactTag.DIRECTORY, loaders.YAML())
    # Only the listing committed beside the lock records the members; the cache holds no copy
    lock.StageLock("b/tree", lock.get_stages_dir(root)).write(
        types.LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes={tree_ref.identity: record},
        ),
        hashed_dirs={
            record["hash"]: listing.HashedDirectory(record=record, listing=members, data=data)
        },
    )
    total_ref = _helper_ref("smoke/total", None, types.ArtifactTag.DATA, loaders.YAML())
    return {
        "b/tree": _make_stage_info("b/tree", [tree_ref], state_dir=root),
        "smoke/total": _make_stage_info("smoke/total", [total_ref], state_dir=root),
    }, cache_dir


def test_get_target_hashes_refuses_a_path_target_named_only_by_a_stages_own_pvt_member(
    set_project_root: pathlib.Path,
) -> None:
    """`data/b/tree/value.txt` is a file inside a stage output that the stage's own
    `value.txt.pvt` names, not a tracked path, so it is refused as an unknown target rather
    than resolving to the hash that member records."""
    all_stages, cache_dir = _helper_project_with_a_stage_member(set_project_root)

    with pytest.raises(exceptions.UnknownTargetError) as raised:
        sync.get_target_hashes(
            ["data/b/tree/value.txt"],
            set_project_root,
            all_stages=all_stages,
            cache_dir=cache_dir,
        )

    assert raised.value.targets == ["data/b/tree/value.txt"]
