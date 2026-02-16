# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false, reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportUnusedParameter=false, reportUnusedCallResult=false
from __future__ import annotations

import pathlib
import pickle
from typing import cast

import pytest

from pivot import loaders, types
from pivot.storage import store


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


def test_cache_store_round_trip(tmp_path: pathlib.Path) -> None:
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / ".pivot"
    cache_store = store.CacheStore(cache_dir=cache_dir, state_db_path=state_dir)

    ref = _helper_ref("stage", None, types.ArtifactTag.DATA, loaders.Text())
    output_path = cache_store.prepare_output(ref)
    output_path.write_text("hello cache")

    cache_store.commit(ref, output_path)
    checkout_path = cache_store.checkout(ref)

    assert checkout_path.read_text() == "hello cache"
    assert cache_store.exists(ref) is True


def test_cache_store_prepare_output_commit_pattern(tmp_path: pathlib.Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_store = store.CacheStore(cache_dir=cache_dir, state_db_path=None)

    ref = _helper_ref("stage", "output", types.ArtifactTag.DATA, loaders.Text())
    output_path = cache_store.prepare_output(ref)
    output_path.write_text("content")

    output_hash = cache_store.commit(ref, output_path)
    assert len(output_hash) == 16
    assert cache_store.checkout(ref).read_text() == "content"


def test_cache_store_pickle_round_trip(tmp_path: pathlib.Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_store = store.CacheStore(cache_dir=cache_dir, state_db_path=None)
    ref = _helper_ref("stage", None, types.ArtifactTag.DATA, loaders.Text())
    output_path = cache_store.prepare_output(ref)
    output_path.write_text("pickle cache")
    cache_store.commit(ref, output_path)

    restored = pickle.loads(pickle.dumps(cache_store))
    assert restored.checkout(ref).read_text() == "pickle cache"


def test_cache_store_hash_artifact_types(tmp_path: pathlib.Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_store = store.CacheStore(cache_dir=cache_dir, state_db_path=None)

    file_ref = _helper_ref("stage", None, types.ArtifactTag.DATA, loaders.Text())
    file_path = cache_store.prepare_output(file_ref)
    file_path.write_text("hash me")
    cache_store.commit(file_ref, file_path)

    file_hash = cache_store.hash_artifact(file_ref)
    assert "hash" in file_hash
    assert "manifest" not in file_hash

    dir_ref = _helper_ref("stage", "dir", types.ArtifactTag.DIRECTORY, loaders.PathOnly())
    dir_path = cache_store.prepare_output(dir_ref)
    (dir_path / "a.txt").write_text("a")
    cache_store.commit(dir_ref, dir_path)

    dir_hash = cache_store.hash_artifact(dir_ref)
    assert "hash" in dir_hash
    assert "manifest" in dir_hash


def test_workspace_store_round_trip(tmp_path: pathlib.Path) -> None:
    workspace_store = store.WorkspaceStore(
        project_root=tmp_path, pipeline_name="pipe", input_bindings={}
    )
    ref = _helper_ref("stage", None, types.ArtifactTag.DATA, loaders.Text())
    output_path = workspace_store.prepare_output(ref)
    output_path.write_text("hello workspace")

    workspace_store.commit(ref, output_path)
    checkout_path = workspace_store.checkout(ref)

    assert checkout_path.read_text() == "hello workspace"
    assert workspace_store.exists(ref) is True


def test_workspace_store_prepare_output_commit_pattern(tmp_path: pathlib.Path) -> None:
    workspace_store = store.WorkspaceStore(
        project_root=tmp_path, pipeline_name="pipe", input_bindings={}
    )
    ref = _helper_ref("stage", "out", types.ArtifactTag.DATA, loaders.Text())
    output_path = workspace_store.prepare_output(ref)
    output_path.write_text("content")

    output_hash = workspace_store.commit(ref, output_path)
    assert len(output_hash) == 16
    assert workspace_store.checkout(ref).read_text() == "content"


def test_workspace_store_pickle_round_trip(tmp_path: pathlib.Path) -> None:
    workspace_store = store.WorkspaceStore(
        project_root=tmp_path, pipeline_name="pipe", input_bindings={}
    )
    ref = _helper_ref("stage", None, types.ArtifactTag.DATA, loaders.Text())
    output_path = workspace_store.prepare_output(ref)
    output_path.write_text("pickle workspace")
    workspace_store.commit(ref, output_path)

    restored = pickle.loads(pickle.dumps(workspace_store))
    assert restored.checkout(ref).read_text() == "pickle workspace"


@pytest.mark.parametrize(
    ("tag", "key", "expected"),
    [
        pytest.param(
            types.ArtifactTag.DATA,
            None,
            "data/pipe/stage.txt",
            id="data-single",
        ),
        pytest.param(
            types.ArtifactTag.DATA,
            "out",
            "data/pipe/stage/out.txt",
            id="data-multi",
        ),
        pytest.param(
            types.ArtifactTag.METRIC,
            "score",
            "metrics/pipe/stage/score.json",
            id="metric",
        ),
        pytest.param(
            types.ArtifactTag.PLOT,
            "loss",
            "plots/pipe/stage/loss.png",
            id="plot",
        ),
        pytest.param(
            types.ArtifactTag.DIRECTORY,
            "dir",
            "data/pipe/stage/dir",
            id="directory",
        ),
    ],
)
def test_workspace_store_path_resolution(
    tmp_path: pathlib.Path,
    tag: types.ArtifactTag,
    key: str | None,
    expected: str,
) -> None:
    fmt: loaders.Writer[object] | loaders.Reader[object] | loaders.Loader[object, object]
    if tag == types.ArtifactTag.METRIC:
        fmt = cast("loaders.Writer[object]", loaders.JSON[dict[str, object]]())
    elif tag == types.ArtifactTag.PLOT:
        fmt = cast("loaders.Writer[object]", loaders.MatplotlibFigure())
    elif tag == types.ArtifactTag.DIRECTORY:
        fmt = cast("loaders.Reader[object]", loaders.PathOnly())
    else:
        fmt = cast("loaders.Writer[object]", loaders.Text())

    workspace_store = store.WorkspaceStore(
        project_root=tmp_path, pipeline_name="pipe", input_bindings={}
    )
    ref = _helper_ref("stage", key, tag, fmt)

    resolved = workspace_store.prepare_output(ref)
    assert resolved == tmp_path / expected


def test_workspace_store_input_bindings(tmp_path: pathlib.Path) -> None:
    workspace_store = store.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="pipe",
        input_bindings={"custom": "data/external/custom.csv"},
    )
    custom_ref = _helper_ref("custom", None, types.ArtifactTag.DATA, loaders.Text())
    default_ref = _helper_ref("rawdata", None, types.ArtifactTag.DATA, loaders.Text())

    assert workspace_store.checkout(custom_ref) == tmp_path / "data/external/custom.csv"
    assert workspace_store.checkout(default_ref) == tmp_path / "data/raw/rawdata"


def test_workspace_store_collision_detection(tmp_path: pathlib.Path) -> None:
    workspace_store = store.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="pipe",
        input_bindings={"foo": "data/raw/bar"},
    )
    foo_ref = _helper_ref("foo", None, types.ArtifactTag.DATA, loaders.Text())
    bar_ref = _helper_ref("bar", None, types.ArtifactTag.DATA, loaders.Text())

    _ = workspace_store.checkout(foo_ref)
    with pytest.raises(ValueError, match="Collision"):
        _ = workspace_store.checkout(bar_ref)


def test_workspace_store_resolves_external_input(tmp_path: pathlib.Path) -> None:
    (tmp_path / "data" / "external").mkdir(parents=True)
    (tmp_path / "data" / "external" / "ext.jsonl").write_text("[]")

    workspace_store = store.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="test",
        input_bindings={"ext.jsonl": "data/external/ext.jsonl"},
    )
    ref = types.ArtifactRef(
        identity=types.ArtifactIdentity("ext.jsonl", None),
        format=loaders.DataFrameJSONL(),
        python_type=list,
        tag=types.ArtifactTag.DATA,
    )

    assert workspace_store.exists(ref)


def test_store_spec_round_trip(tmp_path: pathlib.Path) -> None:
    spec: store.StoreSpec = {
        "kind": "workspace",
        "cache_dir": str(tmp_path / "cache"),
        "project_root": str(tmp_path),
        "pipeline_name": "pipe",
        "input_bindings": {"raw": "data/raw/raw"},
    }

    round_trip = pickle.loads(pickle.dumps(spec))
    rebuilt = store.store_from_spec(round_trip)
    assert isinstance(rebuilt, store.WorkspaceStore)


def test_store_spec_cache_round_trip(tmp_path: pathlib.Path) -> None:
    spec: store.StoreSpec = {
        "kind": "cache",
        "cache_dir": str(tmp_path / "cache"),
        "project_root": str(tmp_path),
        "pipeline_name": "pipe",
        "input_bindings": {},
    }

    round_trip = pickle.loads(pickle.dumps(spec))
    rebuilt = store.store_from_spec(round_trip)
    assert isinstance(rebuilt, store.CacheStore)
