# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false, reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportUnusedParameter=false, reportUnusedCallResult=false, reportImplicitRelativeImport=false
from __future__ import annotations

import inspect
import pathlib
import typing
from typing import TYPE_CHECKING

from pivot import cli, config, loaders, path_utils, project, types
from pivot.cli import checkout as checkout_mod
from pivot.cli import helpers as cli_helpers
from pivot.registry import PipelineLike, RegistryStageInfo
from pivot.storage import cache, lock, store, track

if TYPE_CHECKING:
    import click.testing

    from pivot.compose import Pipeline


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


def _helper_stage_info(
    name: str,
    outs: list[types.ArtifactRef],
    state_dir: pathlib.Path | None,
) -> RegistryStageInfo:
    def _stage_func() -> None:
        return None

    return RegistryStageInfo(
        func=_stage_func,
        name=name,
        deps={},
        outs=outs,
        params=None,
        mutex=[],
        variant=None,
        signature=inspect.signature(_stage_func),
        fingerprint=None,
        params_arg_name=None,
        state_dir=state_dir,
        collection_params={},
        no_fingerprint=False,
    )


def _helper_register_stage(
    pipeline: PipelineLike,
    name: str,
    outs: list[types.ArtifactRef],
    state_dir: pathlib.Path | None = None,
) -> None:
    stage_info = _helper_stage_info(name, outs, state_dir or pipeline.state_dir)
    compose_pipeline = typing.cast("Pipeline", pipeline)
    compose_pipeline._registry._stages[name] = stage_info


def _helper_write_lock(
    stage_name: str,
    state_dir: pathlib.Path,
    output_hashes: dict[types.ArtifactIdentity, types.HashInfo],
) -> None:
    stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
    stage_lock.write(
        types.LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes=output_hashes,
            merkle_id=None,
        )
    )


def _helper_cache_file(path: pathlib.Path, content: str) -> types.HashInfo:
    cache_dir = config.get_cache_dir() / "files"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    output_hash = cache.save_to_cache(path, cache_dir, checkout_mode=config.CheckoutMode.COPY)
    if path.exists():
        path.unlink()
    return output_hash


def _helper_cache_directory(path: pathlib.Path) -> types.HashInfo:
    cache_dir = config.get_cache_dir() / "files"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.mkdir(parents=True, exist_ok=True)
    (path / "part-1.txt").write_text("first")
    (path / "part-2.txt").write_text("second")
    output_hash = cache.save_to_cache(path, cache_dir, checkout_mode=config.CheckoutMode.COPY)
    if path.exists():
        cache.remove_output(path)
    return output_hash


def test_get_stage_output_info_uses_store_paths(mock_discovery: PipelineLike) -> None:
    store_instance = cli_helpers.get_workspace_store()
    assert store_instance is not None

    ref = _helper_ref("train", None, types.ArtifactTag.DATA, loaders.CSV())
    _helper_register_stage(mock_discovery, "train", [ref])
    output_hash = types.FileHash(hash="a" * 16)
    _helper_write_lock("train", mock_discovery.state_dir, {ref.identity: output_hash})

    result = checkout_mod._get_stage_output_info()
    expected_path = str(store_instance.resolve_display_path(ref))

    assert result == {expected_path: output_hash}
    canonical = path_utils.canonicalize_artifact_path(
        types.identity_key(ref.identity), project.get_project_root()
    )
    assert expected_path != canonical


def test_checkout_identity_target_restores_store_path(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    store_instance = cli_helpers.get_workspace_store()
    assert store_instance is not None

    ref = _helper_ref("train", None, types.ArtifactTag.DATA, loaders.CSV())
    _helper_register_stage(mock_discovery, "train", [ref])

    output_path = store_instance.resolve_display_path(ref)
    output_hash = _helper_cache_file(output_path, "id,value\n1,2\n")
    _helper_write_lock("train", mock_discovery.state_dir, {ref.identity: output_hash})

    result = runner.invoke(cli.cli, ["checkout", "train"])

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert output_path.read_text() == "id,value\n1,2\n"


def test_checkout_no_targets_restores_all_outputs(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    store_instance = cli_helpers.get_workspace_store()
    assert store_instance is not None

    ref_train = _helper_ref("train", None, types.ArtifactTag.DATA, loaders.CSV())
    ref_eval = _helper_ref("evaluate", None, types.ArtifactTag.DATA, loaders.JSON())
    _helper_register_stage(mock_discovery, "train", [ref_train])
    _helper_register_stage(mock_discovery, "evaluate", [ref_eval])

    output_path_train = store_instance.resolve_display_path(ref_train)
    output_hash_train = _helper_cache_file(output_path_train, "id,value\n1,2\n")
    _helper_write_lock("train", mock_discovery.state_dir, {ref_train.identity: output_hash_train})

    output_path_eval = store_instance.resolve_display_path(ref_eval)
    output_hash_eval = _helper_cache_file(output_path_eval, '{"score": 0.9}\n')
    _helper_write_lock("evaluate", mock_discovery.state_dir, {ref_eval.identity: output_hash_eval})

    result = runner.invoke(cli.cli, ["checkout"])

    assert result.exit_code == 0, result.output
    assert output_path_train.exists()
    assert output_path_eval.exists()


def test_checkout_pvt_target_restores_file(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    data_dir = project.get_project_root() / "data"
    data_path = data_dir / "raw.csv"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path.write_text("id,value\n1,2\n")

    output_hash = cache.save_to_cache(
        data_path, config.get_cache_dir() / "files", checkout_mode=config.CheckoutMode.COPY
    )
    data_path.unlink()

    pvt_path = track.get_pvt_path(data_path)
    track.write_pvt_file(
        pvt_path,
        track.PvtData(path="raw.csv", hash=output_hash["hash"], size=10),
    )

    result = runner.invoke(cli.cli, ["checkout", str(pvt_path)])

    assert result.exit_code == 0, result.output
    assert data_path.exists()
    assert data_path.read_text() == "id,value\n1,2\n"


def test_checkout_unknown_identity_target_raises(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    ref = _helper_ref("train", "model", types.ArtifactTag.DATA, loaders.CSV())
    _helper_register_stage(mock_discovery, "train", [ref])

    result = runner.invoke(cli.cli, ["checkout", "train:nope"])

    assert result.exit_code != 0
    assert "has no cached outputs" in result.output
    assert "pivot repro train" in result.output


def test_checkout_directory_output_restores_contents(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    store_instance = cli_helpers.get_workspace_store()
    assert store_instance is not None

    ref = _helper_ref("bundle", None, types.ArtifactTag.DIRECTORY, loaders.PathOnly())
    _helper_register_stage(mock_discovery, "bundle", [ref])

    output_path = store_instance.resolve_display_path(ref)
    output_hash = _helper_cache_directory(output_path)
    _helper_write_lock("bundle", mock_discovery.state_dir, {ref.identity: output_hash})

    result = runner.invoke(cli.cli, ["checkout", "bundle"])

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert (output_path / "part-1.txt").read_text() == "first"
    assert (output_path / "part-2.txt").read_text() == "second"


def test_checkout_all_uses_merged_pipeline_paths(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    from conftest import isolated_pivot_dir

    with isolated_pivot_dir(runner, tmp_path) as project_root:
        pipeline_a = project_root / "pipe_a"
        pipeline_b = project_root / "pipe_b"
        pipeline_a.mkdir(parents=True, exist_ok=True)
        pipeline_b.mkdir(parents=True, exist_ok=True)

        (pipeline_a / ".pivot").mkdir()
        (pipeline_b / ".pivot").mkdir()

        pipeline_a_code = """\
from __future__ import annotations

import inspect
import pathlib

from pivot import loaders, types
from pivot.compose import Pipeline
from pivot.registry import RegistryStageInfo


def stage_func() -> None:
    return None


pipeline = Pipeline("pipe_a", root=pathlib.Path(__file__).parent)

pipeline._registry.add_existing(RegistryStageInfo(
    func=stage_func,
    name="train",
    deps={},
    outs=[
        types.ArtifactRef(
            identity=types.ArtifactIdentity("train", None),
            format=loaders.CSV(),
            python_type=pathlib.Path,
            tag=types.ArtifactTag.DATA,
        )
    ],
    params=None,
    mutex=[],
    variant=None,
    signature=inspect.signature(stage_func),
    fingerprint=None,
    params_arg_name=None,
    state_dir=pipeline.state_dir,
    collection_params={},
    no_fingerprint=False,
))
"""
        (pipeline_a / "pipeline.py").write_text(pipeline_a_code)

        pipeline_b_code = """\
from __future__ import annotations

import inspect
import pathlib

from pivot import loaders, types
from pivot.compose import Pipeline
from pivot.registry import RegistryStageInfo


def stage_func() -> None:
    return None


pipeline = Pipeline("pipe_b", root=pathlib.Path(__file__).parent)

pipeline._registry.add_existing(RegistryStageInfo(
    func=stage_func,
    name="evaluate",
    deps={},
    outs=[
        types.ArtifactRef(
            identity=types.ArtifactIdentity("evaluate", None),
            format=loaders.JSON(),
            python_type=pathlib.Path,
            tag=types.ArtifactTag.DATA,
        )
    ],
    params=None,
    mutex=[],
    variant=None,
    signature=inspect.signature(stage_func),
    fingerprint=None,
    params_arg_name=None,
    state_dir=pipeline.state_dir,
    collection_params={},
    no_fingerprint=False,
))
"""
        (pipeline_b / "pipeline.py").write_text(pipeline_b_code)

        merged_store = store.WorkspaceStore(
            project_root=project_root, pipeline_name="all", input_bindings={}
        )
        ref = _helper_ref("train", None, types.ArtifactTag.DATA, loaders.CSV())
        output_path = merged_store.resolve_display_path(ref)
        output_hash = _helper_cache_file(output_path, "id,value\n1,2\n")

        lock_dir = pipeline_a / ".pivot"
        _helper_write_lock("train", lock_dir, {ref.identity: output_hash})

        result = runner.invoke(cli.cli, ["checkout", "--all", "train"])

        assert result.exit_code == 0, result.output
        assert output_path.exists()
        assert output_path.name == "train.csv"


def test_checkout_stage_identity_no_lock_data_gives_helpful_error(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    """Stage identity key with no lock file data gives a helpful error."""
    ref = _helper_ref("train", None, types.ArtifactTag.DATA, loaders.CSV())
    _helper_register_stage(mock_discovery, "train", [ref])
    # Do NOT write lock data -- simulate stage that hasn't been run

    result = runner.invoke(cli.cli, ["checkout", "train"])

    assert result.exit_code != 0
    assert "has no cached outputs" in result.output
    assert "pivot repro train" in result.output


def test_checkout_stage_identity_with_lock_data_restores(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    """Stage identity key with lock file data restores outputs successfully."""
    store_instance = cli_helpers.get_workspace_store()
    assert store_instance is not None

    ref = _helper_ref("train", None, types.ArtifactTag.DATA, loaders.CSV())
    _helper_register_stage(mock_discovery, "train", [ref])

    output_path = store_instance.resolve_display_path(ref)
    output_hash = _helper_cache_file(output_path, "id,value\n1,2\n")
    _helper_write_lock("train", mock_discovery.state_dir, {ref.identity: output_hash})

    result = runner.invoke(cli.cli, ["checkout", "train"])

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert output_path.read_text() == "id,value\n1,2\n"
