from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pivot import loaders, types
from pivot.executor import worker

if TYPE_CHECKING:
    import multiprocessing as mp
    import pathlib

    from pivot.storage import store as store_mod
    from pivot.types import OutputMessage


def _make_artifact_ref(
    producer: str,
    key: str | None,
    *,
    tag: types.ArtifactTag,
    loader: loaders.Reader[object] | loaders.Writer[object] | loaders.Loader[object, object],
    python_type: type,
) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(producer=producer, key=key),
        format=loader,
        python_type=python_type,
        tag=tag,
    )


def _stage_uppercase(data: str) -> str:
    return data.upper()


def _stage_access_group(group: dict[str, str]) -> str:
    return group["a.txt"]


def _make_worker_stage_info(
    func: types.StageFunc,
    tmp_path: pathlib.Path,
    *,
    deps: dict[str, types.ArtifactRef],
    outs: list[types.ArtifactRef],
    store_spec: store_mod.StoreSpec,
) -> worker.WorkerStageInfo:
    state_dir = tmp_path / ".pivot"
    state_dir.mkdir(exist_ok=True)
    return worker.WorkerStageInfo(
        func=func,
        fingerprint={"self:test": "abc123"},
        deps=deps,
        outs=outs,
        store_spec=store_spec,
        signature=None,
        params=None,
        variant=None,
        overrides={},
        checkout_modes=[],
        run_id="test_run",
        force=False,
        no_commit=True,
        params_arg_name=None,
        project_root=tmp_path,
        state_dir=state_dir,
    )


def test_tracked_dict_records_accessed_keys() -> None:
    tracked = worker.TrackedDict({"a": "one", "b": "two"})

    assert tracked.accessed_keys == frozenset(), "Should be empty before access"

    assert tracked["a"] == "one"
    assert tracked.accessed_keys == frozenset({"a"}), "Should record accessed key"

    assert tracked["b"] == "two"
    assert tracked.accessed_keys == frozenset({"a", "b"}), "Should track all accessed keys"


def test_execute_stage_uses_store_spec_for_io(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    (tmp_path / "input.txt").write_text("hello")

    deps = {
        "data": _make_artifact_ref(
            "input",
            None,
            tag=types.ArtifactTag.DATA,
            loader=loaders.Text(),
            python_type=str,
        )
    }
    outs = [
        _make_artifact_ref(
            "stage",
            None,
            tag=types.ArtifactTag.DATA,
            loader=loaders.Text(),
            python_type=str,
        )
    ]
    store_spec = cast(
        "store_mod.StoreSpec",
        cast(
            "object",
            {
                "kind": "workspace",
                "cache_dir": str(worker_env),
                "project_root": str(tmp_path),
                "pipeline_name": "pipe",
                "input_bindings": {"input": "input.txt"},
            },
        ),
    )

    stage_info = _make_worker_stage_info(
        _stage_uppercase,
        tmp_path,
        deps=deps,
        outs=outs,
        store_spec=store_spec,
    )

    result = worker.execute_stage("stage", stage_info, worker_env, output_queue)

    output_path = tmp_path / "data" / "pipe" / "stage.txt"
    assert output_path.exists(), "Output should be written via Store"
    assert output_path.read_text() == "HELLO"
    assert result["status"] == types.StageStatus.RAN


def test_execute_stage_records_accessed_group_keys(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    group_dir = tmp_path / "data" / "pipe" / "upstream"
    group_dir.mkdir(parents=True)
    (group_dir / "a.txt").write_text("alpha")
    (group_dir / "b.txt").write_text("beta")

    deps = {
        "group": _make_artifact_ref(
            "upstream",
            None,
            tag=types.ArtifactTag.DIRECTORY,
            loader=loaders.Text(),
            python_type=dict,
        )
    }
    store_spec = cast(
        "store_mod.StoreSpec",
        cast(
            "object",
            {
                "kind": "workspace",
                "cache_dir": str(worker_env),
                "project_root": str(tmp_path),
                "pipeline_name": "pipe",
                "input_bindings": {},
            },
        ),
    )

    stage_info = _make_worker_stage_info(
        _stage_access_group,
        tmp_path,
        deps=deps,
        outs=[],
        store_spec=store_spec,
    )

    result = worker.execute_stage("stage", stage_info, worker_env, output_queue)

    assert result["status"] == types.StageStatus.RAN
    assert "accessed_dep_keys" in result
    assert result["accessed_dep_keys"] == {"group": {"a.txt"}}, "Should record accessed group keys"
