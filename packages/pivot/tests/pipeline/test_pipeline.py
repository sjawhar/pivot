from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any

import pytest

from pivot import discovery, loaders, project, registry, types
from pivot.pipeline import pipeline as pipeline_mod

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _helper_stage_func() -> None:
    return None


def _artifact_ref(identity_key: str) -> types.ArtifactRef:
    identity = types.identity_from_key(identity_key)
    return types.ArtifactRef(
        identity=identity,
        format=loaders.PathOnly(),
        python_type=str,
        tag=types.ArtifactTag.DATA,
    )


def _stage_info(
    name: str,
    *,
    deps: list[str] | None = None,
    outs: list[str] | None = None,
    func: Any = _helper_stage_func,
    state_dir: pathlib.Path | None = None,
) -> registry.RegistryStageInfo:
    dep_keys = deps or []
    out_keys = outs or []
    return registry.RegistryStageInfo(
        func=func,
        name=name,
        deps={f"d{i}": _artifact_ref(dep) for i, dep in enumerate(dep_keys)},
        outs=[_artifact_ref(out) for out in out_keys],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        params_arg_name=None,
        state_dir=state_dir,
        collection_params={},
    )


def test_rejects_invalid_pipeline_name(tmp_path: pathlib.Path) -> None:
    with pytest.raises(pipeline_mod.PipelineConfigError, match="cannot be empty"):
        pipeline_mod.Pipeline("", root=tmp_path)

    with pytest.raises(pipeline_mod.PipelineConfigError, match="Invalid pipeline name"):
        pipeline_mod.Pipeline("1bad", root=tmp_path)


def test_include_rejects_self_include(tmp_path: pathlib.Path) -> None:
    pipeline = pipeline_mod.Pipeline("self_ref", root=tmp_path)

    with pytest.raises(pipeline_mod.PipelineConfigError, match="cannot include itself"):
        pipeline.include(pipeline)


def test_include_merges_input_bindings(tmp_path: pathlib.Path) -> None:
    child = pipeline_mod.Pipeline("child", root=tmp_path)
    child.set_input_bindings({"ext.jsonl": "data/external/ext.jsonl"})

    parent = pipeline_mod.Pipeline("parent", root=tmp_path)
    parent.include(child)

    assert parent.input_bindings["ext.jsonl"] == "data/external/ext.jsonl"


def test_include_prefixes_collisions_and_rewrites_internal_refs(tmp_path: pathlib.Path) -> None:
    parent = pipeline_mod.Pipeline("parent", root=tmp_path)
    parent._registry.add_existing(  # noqa: SLF001
        _stage_info(
            "train",
            outs=[types.identity_key(types.ArtifactIdentity("train", "model"))],
        )
    )
    parent.set_input_bindings({"keep": "keep.txt"})

    child = pipeline_mod.Pipeline("child", root=tmp_path)
    child._registry.add_existing(  # noqa: SLF001
        _stage_info(
            "train",
            deps=[types.identity_key(types.ArtifactIdentity("prep", "data"))],
            outs=[types.identity_key(types.ArtifactIdentity("train", "model"))],
        )
    )
    child._registry.add_existing(  # noqa: SLF001
        _stage_info(
            "prep",
            outs=[types.identity_key(types.ArtifactIdentity("prep", "data"))],
        )
    )
    child.set_input_bindings({"keep": "other.txt", "new": "new.txt"})

    parent.include(child)

    assert sorted(parent.list_stages()) == ["child/prep", "child/train", "train"]
    prefixed_train = parent.get("child/train")
    dep_identity = next(iter(prefixed_train["deps"].values())).identity
    out_identity = prefixed_train["outs"][0].identity
    assert dep_identity == types.ArtifactIdentity("child/prep", "data")
    assert out_identity == types.ArtifactIdentity("child/train", "model")
    assert parent.input_bindings["keep"] == "keep.txt"
    assert parent.input_bindings["new"] == "new.txt"


def test_load_pipeline_uses_cache(mocker: MockerFixture, tmp_path: pathlib.Path) -> None:
    config_path = tmp_path / "pipeline.py"
    loaded: dict[pathlib.Path, pipeline_mod.Pipeline | None] = {config_path: None}

    mock = mocker.patch.object(discovery, "load_pipeline_from_path", autospec=True)
    assert pipeline_mod._load_pipeline(config_path, loaded) is None  # noqa: SLF001
    mock.assert_not_called()

    loaded.clear()
    pipeline = pipeline_mod.Pipeline("cached", root=tmp_path)
    mock.return_value = pipeline
    assert pipeline_mod._load_pipeline(config_path, loaded) is pipeline  # noqa: SLF001
    assert pipeline_mod._load_pipeline(config_path, loaded) is pipeline  # noqa: SLF001
    mock.assert_called_once_with(config_path)


def test_find_pipeline_dir_for_stage_falls_back_to_state_dir(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    state_dir = tmp_path / "sub" / ".pivot"
    info = _stage_info("x", state_dir=state_dir)
    mocker.patch.object(pipeline_mod.inspect, "getfile", side_effect=TypeError("no source"))

    found = pipeline_mod._find_pipeline_dir_for_stage(info, tmp_path)  # noqa: SLF001

    assert found == "sub"


def test_find_pipeline_dir_returns_none_when_no_derivation(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    info = _stage_info("x", state_dir=pathlib.Path("/outside/.pivot"))
    mocker.patch.object(pipeline_mod.inspect, "getfile", side_effect=TypeError("no source"))

    found = pipeline_mod._find_pipeline_dir_for_stage(info, tmp_path)  # noqa: SLF001

    assert found is None


def test_find_producer_helpers_cover_missing_and_scan_paths(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    dep_path = str(tmp_path / "shared" / "dataset.csv")
    pipeline_a = pipeline_mod.Pipeline("a", root=tmp_path / "a")
    pipeline_b = pipeline_mod.Pipeline("b", root=tmp_path / "b")
    pipeline_b._registry.add_existing(_stage_info("producer", outs=[dep_path]))  # noqa: SLF001

    assert pipeline_mod._find_producer_in_pipeline(dep_path, pipeline_a) is None  # noqa: SLF001

    p1 = tmp_path / "a" / "pipeline.py"
    p2 = tmp_path / "b" / "pipeline.py"
    mocker.patch.object(
        discovery,
        "find_pipeline_paths_for_dependency",
        autospec=True,
        return_value=[p1, p2],
    )
    mocker.patch.object(
        discovery,
        "load_pipeline_from_path",
        autospec=True,
        side_effect=lambda path: None if path == p1 else pipeline_b,
    )

    loaded: dict[pathlib.Path, pipeline_mod.Pipeline | None] = {}
    traversed = pipeline_mod._find_producer_via_traversal(dep_path, tmp_path, loaded)  # noqa: SLF001
    scanned = pipeline_mod._find_producer_via_scan(dep_path, [p1, p2], loaded)  # noqa: SLF001

    assert traversed is not None and traversed[1] == "b"
    assert scanned is not None and scanned[0]["name"] == "producer"


def test_find_producer_via_index_returns_none_for_unusable_hints(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    loaded: dict[pathlib.Path, pipeline_mod.Pipeline | None] = {}

    outside_dep = "/tmp/not-in-project.csv"
    assert pipeline_mod._find_producer_via_index(outside_dep, tmp_path, loaded) is None  # noqa: SLF001

    dep_path = tmp_path / "in_project.csv"
    hint_file = tmp_path / ".pivot" / "cache" / "outputs" / "in_project.csv"
    hint_file.parent.mkdir(parents=True)
    hint_file.write_text("hint")
    mocker.patch.object(discovery, "find_config_in_dir", autospec=True, return_value=None)
    assert pipeline_mod._find_producer_via_index(str(dep_path), tmp_path, loaded) is None  # noqa: SLF001


def test_find_producer_via_index_reads_hint_file_and_validates(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    dep_path = tmp_path / "data" / "input.csv"
    dep_path.parent.mkdir(parents=True)
    hint_file = tmp_path / ".pivot" / "cache" / "outputs" / "data" / "input.csv"
    hint_file.parent.mkdir(parents=True)
    hint_file.write_text("pipelines/a")

    producer_pipeline = pipeline_mod.Pipeline("ext", root=tmp_path / "pipelines" / "a")
    producer_pipeline._registry.add_existing(  # noqa: SLF001
        _stage_info("build", outs=[str(dep_path)])
    )

    config_path = tmp_path / "pipelines" / "a" / "pipeline.py"
    mocker.patch.object(discovery, "find_config_in_dir", autospec=True, return_value=config_path)
    mocker.patch.object(
        discovery,
        "load_pipeline_from_path",
        autospec=True,
        return_value=producer_pipeline,
    )

    loaded: dict[pathlib.Path, pipeline_mod.Pipeline | None] = {}
    result = pipeline_mod._find_producer_via_index(str(dep_path), tmp_path, loaded)  # noqa: SLF001

    assert result is not None
    assert result[0]["name"] == "build"
    assert result[1] == "ext"


def test_pipeline_core_methods_delegate_to_registry(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    pipeline = pipeline_mod.Pipeline("core", root=tmp_path)
    stage = _stage_info("stage", outs=[str(tmp_path / "out.txt")])
    pipeline._registry.add_existing(stage)  # noqa: SLF001

    registry_mock = mocker.patch.object(pipeline, "_registry", autospec=True)
    registry_mock.list_stages.return_value = ["stage"]
    registry_mock.get.return_value = stage
    registry_mock.ensure_fingerprint.return_value = {"f": "1"}
    snapshot = {"stage": stage}
    registry_mock.snapshot.return_value = snapshot
    dag_obj = object()
    registry_mock.build_dag.return_value = dag_obj

    assert pipeline.state_dir == tmp_path / ".pivot"
    assert pipeline.list_stages() == ["stage"]
    assert pipeline.get("stage") == stage
    assert pipeline.get_stage("stage") == stage
    assert pipeline.ensure_fingerprint("stage") == {"f": "1"}
    assert pipeline.snapshot() == snapshot

    mocker.patch.object(pipeline, "resolve_external_dependencies", autospec=True)
    mocker.patch.object(pipeline, "_write_output_index", autospec=True)
    assert pipeline.build_dag() is dag_obj

    pipeline.invalidate_dag_cache()
    pipeline.restore(snapshot)
    pipeline.clear()
    assert pipeline._external_deps_resolved is False  # noqa: SLF001


def test_init_without_caller_context_raises(
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoBackFrame:
        f_back = None

    class NoFileBack:
        f_globals: dict[str, object] = {}

    class NoFileFrame:
        f_back = NoFileBack()

    mocker.patch.object(pipeline_mod.inspect, "currentframe", return_value=NoBackFrame())
    with pytest.raises(RuntimeError, match="Cannot determine caller frame"):
        pipeline_mod.Pipeline("x")

    monkeypatch.setattr(pipeline_mod.inspect, "currentframe", lambda: NoFileFrame())
    with pytest.raises(RuntimeError, match="Provide an explicit root"):
        pipeline_mod.Pipeline("y")


def test_include_uses_numeric_suffix_for_prefixed_collisions(tmp_path: pathlib.Path) -> None:
    parent = pipeline_mod.Pipeline("parent", root=tmp_path)
    parent._registry.add_existing(_stage_info("train"))  # noqa: SLF001
    parent._registry.add_existing(_stage_info("child/train"))  # noqa: SLF001

    child = pipeline_mod.Pipeline("child", root=tmp_path)
    child._registry.add_existing(_stage_info("train"))  # noqa: SLF001

    parent.include(child)

    assert "child/train_2" in parent.list_stages()


def test_resolve_external_dependencies_adds_producer_from_scan(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    dep_path = tmp_path / "shared" / "dataset.csv"
    consumer = pipeline_mod.Pipeline("consumer", root=tmp_path)
    consumer._registry.add_existing(  # noqa: SLF001
        _stage_info(
            "consume",
            deps=[str(dep_path)],
            outs=[types.identity_key(types.ArtifactIdentity("consume", "result"))],
        )
    )

    producer_pipeline = pipeline_mod.Pipeline("ext", root=tmp_path / "ext")
    producer_pipeline._registry.add_existing(  # noqa: SLF001
        _stage_info("produce", outs=[str(dep_path)])
    )

    scan_path = tmp_path / "ext" / "pipeline.py"
    mocker.patch.object(project, "get_project_root", autospec=True, return_value=tmp_path)
    mocker.patch.object(
        discovery,
        "find_pipeline_paths_for_dependency",
        autospec=True,
        return_value=[],
    )
    mocker.patch.object(discovery, "find_config_in_dir", autospec=True, return_value=None)
    mocker.patch.object(discovery, "glob_all_pipelines", autospec=True, return_value=[scan_path])
    mocker.patch.object(
        discovery,
        "load_pipeline_from_path",
        autospec=True,
        return_value=producer_pipeline,
    )

    consumer.resolve_external_dependencies()

    assert "consume" in consumer.list_stages()
    assert "produce" in consumer.list_stages()


def test_resolve_external_dependencies_handles_fast_paths_and_collisions(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    resolved = pipeline_mod.Pipeline("resolved", root=tmp_path)
    resolved._external_deps_resolved = True  # noqa: SLF001
    resolved.resolve_external_dependencies()

    no_work = pipeline_mod.Pipeline("no_work", root=tmp_path)
    no_work._registry.add_existing(_stage_info("solo", outs=[str(tmp_path / "data.csv")]))  # noqa: SLF001
    mocker.patch.object(project, "get_project_root", autospec=True, return_value=tmp_path)
    no_work.resolve_external_dependencies()
    assert no_work._external_deps_resolved  # noqa: SLF001

    dep_path = tmp_path / "needs.csv"
    target = pipeline_mod.Pipeline("target", root=tmp_path)
    target._registry.add_existing(_stage_info("consume", deps=[str(dep_path)]))  # noqa: SLF001
    target._registry.add_existing(_stage_info("produce"))  # noqa: SLF001

    external = pipeline_mod.Pipeline("ext", root=tmp_path / "ext")
    external._registry.add_existing(_stage_info("produce", outs=[str(dep_path)]))  # noqa: SLF001

    scan_path = tmp_path / "ext" / "pipeline.py"
    mocker.patch.object(
        discovery,
        "find_pipeline_paths_for_dependency",
        autospec=True,
        return_value=[],
    )
    mocker.patch.object(discovery, "find_config_in_dir", autospec=True, return_value=None)
    mocker.patch.object(discovery, "glob_all_pipelines", autospec=True, return_value=[scan_path])
    mocker.patch.object(discovery, "load_pipeline_from_path", autospec=True, return_value=external)

    target.resolve_external_dependencies()

    assert target.list_stages() == ["consume", "produce"]


def test_write_output_index_writes_project_relative_entries(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    pipeline = pipeline_mod.Pipeline("idx", root=tmp_path)
    out_path = tmp_path / "data" / "output.csv"
    outside = pathlib.Path("/tmp/outside.csv")
    pipeline._registry.add_existing(  # noqa: SLF001
        _stage_info("stage", outs=[str(out_path), str(outside)])
    )

    source_file = tmp_path / "sub" / "stage_file.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("def f():\n    return 1\n")
    config_path = tmp_path / "sub" / "pipeline.py"
    config_path.write_text("pipeline = None\n")

    mocker.patch.object(project, "get_project_root", autospec=True, return_value=tmp_path)
    mocker.patch.object(
        pipeline_mod.inspect, "getfile", autospec=True, return_value=str(source_file)
    )
    mocker.patch.object(
        discovery,
        "find_config_in_dir",
        autospec=True,
        side_effect=lambda p: config_path if p == source_file.parent else None,
    )

    pipeline._write_output_index()  # noqa: SLF001

    index_entry = tmp_path / ".pivot" / "cache" / "outputs" / "data" / "output.csv"
    assert index_entry.read_text() == "sub"
    assert pipeline._external_deps_resolved  # noqa: SLF001


def test_write_output_index_handles_failures_gracefully(
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    pipeline = pipeline_mod.Pipeline("idx_fail", root=tmp_path)
    pipeline._registry.add_existing(  # noqa: SLF001
        _stage_info("stage", outs=[str(tmp_path / "out.txt")], state_dir=tmp_path / ".pivot")
    )

    mocker.patch.object(project, "get_project_root", side_effect=RuntimeError("boom"))
    pipeline._write_output_index()  # noqa: SLF001

    monkeypatch.setattr(project, "get_project_root", lambda: tmp_path)
    mocker.patch.object(pipeline_mod.inspect, "getfile", side_effect=TypeError("missing source"))
    mocker.patch.object(pathlib.Path, "write_text", side_effect=OSError("nope"))
    pipeline._write_output_index()  # noqa: SLF001


def test_init_uses_caller_file_when_root_not_provided(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    class FakeBack:
        f_globals = {"__file__": str(tmp_path / "nested" / "pipeline.py")}

    class FakeFrame:
        f_back = FakeBack()

    mocker.patch.object(
        pipeline_mod.inspect, "currentframe", autospec=True, return_value=FakeFrame()
    )

    pipeline = pipeline_mod.Pipeline("auto_root")

    assert pipeline.root == (tmp_path / "nested").resolve()
