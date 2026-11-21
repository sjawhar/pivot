# pyright: reportMissingImports=false
"""Tests for the bipartite artifact-stage graph."""

from __future__ import annotations

import pickle
import random
from typing import TYPE_CHECKING

import networkx as nx
import pytest

from pivot import exceptions, loaders
from pivot.engine import graph, types
from pivot.executor import core as executor_core
from pivot.registry import RegistryStageInfo
from pivot.storage import store as store_mod
from pivot.types import ArtifactIdentity, ArtifactRef, ArtifactTag, OnError
from tests import helpers

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet
    from pathlib import Path


def _artifact_ref(identity: ArtifactIdentity) -> ArtifactRef:
    return ArtifactRef(
        identity=identity,
        format=loaders.PathOnly(),
        python_type=str,
        tag=ArtifactTag.DATA,
    )


def _create_stage(
    name: str,
    deps: list[ArtifactIdentity],
    outs: list[ArtifactIdentity],
) -> RegistryStageInfo:
    """Create a stage dict for testing."""
    return RegistryStageInfo(
        func=lambda: None,
        name=name,
        deps={f"_{i}": _artifact_ref(dep) for i, dep in enumerate(deps)},
        outs=[_artifact_ref(out) for out in outs],
        incremental_inputs={},
        params=None,
        resources={},
        variant=None,
        signature=None,
        fingerprint=dict[str, str](),
        params_arg_name=None,
        state_dir=None,
        collection_params={},
        no_fingerprint=False,
        code_deps=[],
        input_paths={},
        unsafe_fingerprint=False,
    )


# --- Node naming tests ---


def test_artifact_node_creates_prefixed_string() -> None:
    """artifact_node creates 'artifact:' prefixed string."""
    node = graph.artifact_node(ArtifactIdentity("source", None))
    assert node == "artifact:source"


def test_stage_node_creates_prefixed_string() -> None:
    """stage_node creates 'stage:' prefixed string."""
    node = graph.stage_node("train")
    assert node == "stage:train"


def test_parse_node_extracts_type_and_value() -> None:
    """parse_node extracts NodeType and value from prefixed string."""
    node_type, value = graph.parse_node("artifact:source:input.csv")
    assert node_type == types.NodeType.ARTIFACT
    assert value == "source:input.csv"

    node_type, value = graph.parse_node("stage:train")
    assert node_type == types.NodeType.STAGE
    assert value == "train"


# --- Graph building tests ---


@pytest.mark.usefixtures("clean_registry")
def test_build_graph_simple_chain(tmp_path: Path) -> None:
    """Build bipartite graph for simple chain: input -> A -> intermediate -> B -> output."""
    input_identity = ArtifactIdentity("source", "input.csv")
    intermediate = ArtifactIdentity("stage_a", "intermediate.csv")
    output_identity = ArtifactIdentity("stage_b", "output.csv")

    # Create stages dict directly for isolated graph test
    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [intermediate]),
        "stage_b": _create_stage("stage_b", [intermediate], [output_identity]),
    }

    g = graph.build_graph(stages)

    # Check we have both stage and artifact nodes
    stage_nodes = [n for n in g.nodes() if g.nodes[n]["type"] == types.NodeType.STAGE]
    artifact_nodes = [n for n in g.nodes() if g.nodes[n]["type"] == types.NodeType.ARTIFACT]

    assert len(stage_nodes) == 2
    assert len(artifact_nodes) == 3  # input, intermediate, output

    # Check edges: artifact -> stage (consumed by) and stage -> artifact (produces)
    assert g.has_edge(graph.artifact_node(input_identity), graph.stage_node("stage_a"))
    assert g.has_edge(graph.stage_node("stage_a"), graph.artifact_node(intermediate))
    assert g.has_edge(graph.artifact_node(intermediate), graph.stage_node("stage_b"))
    assert g.has_edge(graph.stage_node("stage_b"), graph.artifact_node(output_identity))


@pytest.mark.usefixtures("clean_registry")
def test_build_graph_diamond(tmp_path: Path) -> None:
    """Build bipartite graph for diamond pattern.

    input -> preprocess -> clean
          -> features -> feats
    clean + feats -> train -> model
    """
    input_identity = ArtifactIdentity("source", "input.csv")
    clean = ArtifactIdentity("preprocess", "clean.csv")
    feats = ArtifactIdentity("features", "feats.csv")
    model = ArtifactIdentity("train", "model.pkl")

    stages = {
        "preprocess": _create_stage("preprocess", [input_identity], [clean]),
        "features": _create_stage("features", [input_identity], [feats]),
        "train": _create_stage("train", [clean, feats], [model]),
    }

    g = graph.build_graph(stages)

    # Both preprocess and features consume input
    assert g.has_edge(graph.artifact_node(input_identity), graph.stage_node("preprocess"))
    assert g.has_edge(graph.artifact_node(input_identity), graph.stage_node("features"))

    # Train consumes both clean and feats
    assert g.has_edge(graph.artifact_node(clean), graph.stage_node("train"))
    assert g.has_edge(graph.artifact_node(feats), graph.stage_node("train"))


@pytest.mark.usefixtures("clean_registry")
def test_build_graph_empty() -> None:
    """Build graph with no stages returns empty graph."""
    g = graph.build_graph({})
    assert len(g.nodes()) == 0
    assert len(g.edges()) == 0


# --- Query function tests ---


@pytest.mark.usefixtures("clean_registry")
def test_get_consumers_returns_dependent_stages(tmp_path: Path) -> None:
    """get_consumers returns stages that depend on an artifact."""
    input_identity = ArtifactIdentity("source", "input.csv")
    out_a = ArtifactIdentity("stage_a", "a.csv")
    out_b = ArtifactIdentity("stage_b", "b.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [out_a]),
        "stage_b": _create_stage("stage_b", [input_identity], [out_b]),
    }

    g = graph.build_graph(stages)
    consumers = graph.get_consumers(g, input_identity)

    assert set(consumers) == {"stage_a", "stage_b"}


@pytest.mark.usefixtures("clean_registry")
def test_get_consumers_returns_empty_for_unknown_path(tmp_path: Path) -> None:
    """get_consumers returns empty list for unknown path."""
    g = graph.build_graph({})
    consumers = graph.get_consumers(g, ArtifactIdentity("unknown", "artifact"))
    assert consumers == []


@pytest.mark.usefixtures("clean_registry")
def test_get_producer_returns_producing_stage(tmp_path: Path) -> None:
    """get_producer returns the stage that produces an artifact."""
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }

    g = graph.build_graph(stages)
    producer = graph.get_producer(g, output_identity)

    assert producer == "stage_a"


@pytest.mark.usefixtures("clean_registry")
def test_get_producer_returns_none_for_input_artifact(tmp_path: Path) -> None:
    """get_producer returns None for artifacts that are inputs (not produced by any stage)."""
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }

    g = graph.build_graph(stages)
    producer = graph.get_producer(g, input_identity)

    assert producer is None


@pytest.mark.usefixtures("clean_registry")
def test_get_downstream_stages(tmp_path: Path) -> None:
    """get_downstream_stages returns all transitively downstream stages."""
    input_identity = ArtifactIdentity("source", "input.csv")
    intermediate = ArtifactIdentity("stage_a", "intermediate.csv")
    output_identity = ArtifactIdentity("stage_b", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [intermediate]),
        "stage_b": _create_stage("stage_b", [intermediate], [output_identity]),
    }

    g = graph.build_graph(stages)
    downstream = graph.get_downstream_stages(g, "stage_a")

    assert set(downstream) == {"stage_b"}


@pytest.mark.usefixtures("clean_registry")
def test_get_downstream_stages_empty_for_leaf(tmp_path: Path) -> None:
    """get_downstream_stages returns empty for leaf stage."""
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }

    g = graph.build_graph(stages)
    downstream = graph.get_downstream_stages(g, "stage_a")

    assert downstream == []


# --- Incremental update tests ---


@pytest.mark.usefixtures("clean_registry")
def test_update_stage_adds_new_dep(tmp_path: Path) -> None:
    """update_stage adds new dependency edges."""
    input_a = ArtifactIdentity("source", "a.csv")
    input_b = ArtifactIdentity("source", "b.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    # Initial: stage_a depends on input_a
    stages = {
        "stage_a": _create_stage("stage_a", [input_a], [output_identity]),
    }
    g = graph.build_graph(stages)

    assert graph.get_consumers(g, input_a) == ["stage_a"]
    assert graph.get_consumers(g, input_b) == []

    # Update: stage_a now also depends on input_b
    new_info = _create_stage("stage_a", [input_a, input_b], [output_identity])
    graph.update_stage(g, "stage_a", new_info)

    assert set(graph.get_consumers(g, input_a)) == {"stage_a"}
    assert set(graph.get_consumers(g, input_b)) == {"stage_a"}


@pytest.mark.usefixtures("clean_registry")
def test_update_stage_removes_old_dep(tmp_path: Path) -> None:
    """update_stage removes old dependency edges and orphaned artifacts."""
    input_a = ArtifactIdentity("source", "a.csv")
    input_b = ArtifactIdentity("source", "b.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    # Initial: stage_a depends on both inputs
    stages = {
        "stage_a": _create_stage("stage_a", [input_a, input_b], [output_identity]),
    }
    g = graph.build_graph(stages)

    # Update: stage_a now only depends on input_a
    new_info = _create_stage("stage_a", [input_a], [output_identity])
    graph.update_stage(g, "stage_a", new_info)

    assert graph.get_consumers(g, input_a) == ["stage_a"]
    assert graph.get_consumers(g, input_b) == []

    # input_b should be removed from graph (orphaned)
    assert graph.artifact_node(input_b) not in g


@pytest.mark.usefixtures("clean_registry")
def test_update_stage_preserves_shared_artifacts(tmp_path: Path) -> None:
    """update_stage doesn't remove artifacts used by other stages."""
    shared_input = ArtifactIdentity("source", "shared.csv")
    out_a = ArtifactIdentity("stage_a", "a.csv")
    out_b = ArtifactIdentity("stage_b", "b.csv")

    # Both stages depend on shared_input
    stages = {
        "stage_a": _create_stage("stage_a", [shared_input], [out_a]),
        "stage_b": _create_stage("stage_b", [shared_input], [out_b]),
    }
    g = graph.build_graph(stages)

    # Update stage_a to have no deps - shared_input should remain (used by stage_b)
    new_info = _create_stage("stage_a", [], [out_a])
    graph.update_stage(g, "stage_a", new_info)

    # shared_input still in graph
    assert graph.artifact_node(shared_input) in g
    assert graph.get_consumers(g, shared_input) == ["stage_b"]


# --- get_stage_dag tests ---


@pytest.mark.usefixtures("clean_registry")
def test_get_stage_dag_extracts_stage_only_graph(tmp_path: Path) -> None:
    """get_stage_dag returns stage-only DAG from bipartite graph."""
    input_identity = ArtifactIdentity("source", "input.csv")
    cleaned = ArtifactIdentity("preprocess", "cleaned.csv")
    model = ArtifactIdentity("train", "model.pkl")

    stages = {
        "preprocess": _create_stage("preprocess", [input_identity], [cleaned]),
        "train": _create_stage("train", [cleaned], [model]),
    }
    bipartite = graph.build_graph(stages)

    # Extract stage-only DAG
    stage_dag = graph.get_stage_dag(bipartite)

    # Should have stage nodes (not artifact:... or stage:... prefixed)
    assert "preprocess" in stage_dag
    assert "train" in stage_dag

    # Should NOT have artifact nodes or prefixed stage nodes
    for node in stage_dag.nodes():
        assert not node.startswith("artifact:")
        assert not node.startswith("stage:")

    # Edge direction: consumer -> producer (for DFS postorder execution)
    # train depends on preprocess, so edge goes train -> preprocess
    assert stage_dag.has_edge("train", "preprocess")


# --- get_artifact_consumers tests ---


@pytest.mark.usefixtures("clean_registry")
def test_get_artifact_consumers_returns_direct_and_downstream(tmp_path: Path) -> None:
    """get_artifact_consumers returns stages that depend on artifact."""
    # Build graph: input.csv -> preprocess -> cleaned.csv -> train -> model.pkl
    input_identity = ArtifactIdentity("source", "input.csv")
    cleaned = ArtifactIdentity("preprocess", "cleaned.csv")
    model = ArtifactIdentity("train", "model.pkl")

    stages = {
        "preprocess": _create_stage("preprocess", [input_identity], [cleaned]),
        "train": _create_stage("train", [cleaned], [model]),
    }
    g = graph.build_graph(stages)

    # Input change should affect both preprocess AND train
    consumers = graph.get_artifact_consumers(g, input_identity, include_downstream=True)

    assert "preprocess" in consumers
    assert "train" in consumers  # Downstream of preprocess

    # Without downstream, only direct consumers
    direct = graph.get_artifact_consumers(g, input_identity, include_downstream=False)

    assert "preprocess" in direct
    assert "train" not in direct


@pytest.mark.usefixtures("clean_registry")
def test_get_artifact_consumers_returns_empty_for_unknown_path(tmp_path: Path) -> None:
    """get_artifact_consumers returns empty list for unknown artifact."""
    g = graph.build_graph({})
    consumers = graph.get_artifact_consumers(g, ArtifactIdentity("unknown", "artifact"))
    assert consumers == []


# --- Validation tests ---


def test_build_graph_raises_on_cycle(tmp_path: Path) -> None:
    """build_graph raises CyclicGraphError when graph has cycles."""
    file_a = ArtifactIdentity("stage_a", "a.csv")
    file_b = ArtifactIdentity("stage_b", "b.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [file_b], [file_a]),
        "stage_b": _create_stage("stage_b", [file_a], [file_b]),
    }

    with pytest.raises(exceptions.CyclicGraphError, match="Circular dependency"):
        graph.build_graph(stages)  # Cycles always checked


def test_build_graph_raises_on_missing_dependency(tmp_path: Path) -> None:
    """validate_dependency_sources raises DependencyNotFoundError for missing deps."""
    missing_dep = ArtifactIdentity("stage_b", "missing.csv")

    stages = {
        "stage_a": _create_stage(
            "stage_a", [missing_dep], [ArtifactIdentity("stage_a", "out.csv")]
        ),
        "stage_b": _create_stage("stage_b", [], [ArtifactIdentity("stage_b", "other.csv")]),
    }

    errors = graph.validate_dependency_sources(stages)
    assert len(errors) == 1


def test_validate_dependency_sources_reports_missing_pipeline_input_path(
    set_project_root: Path,
) -> None:
    missing_input = ArtifactIdentity("extra.csv", None)
    stages = {
        "demo/combine": _create_stage(
            "demo/combine",
            [missing_input],
            [ArtifactIdentity("demo/combine", None)],
        )
    }
    stages["demo/combine"]["deps"]["_0"].tag = ArtifactTag.INPUT
    store = store_mod.WorkspaceStore(
        project_root=set_project_root,
        pipeline_name="demo",
        input_bindings={"extra.csv": "missing/extra.csv"},
    )

    errors = graph.validate_dependency_sources(stages, store)

    assert len(errors) == 1
    assert str(errors[0]) == (
        "Stage 'demo/combine' depends on input 'extra.csv' "
        "(path missing/extra.csv), which does not exist"
    )
    assert errors[0].input_path == set_project_root / "missing" / "extra.csv"


@pytest.mark.parametrize(
    ("all_pipelines", "on_error", "runnable"),
    [
        pytest.param(True, OnError.KEEP_GOING, {"ok/a"}, id="all-keep-going"),
        pytest.param(True, OnError.FAIL, None, id="all-fail"),
        pytest.param(False, OnError.KEEP_GOING, None, id="one-keep-going"),
    ],
)
def test_get_runnable_stages_leaves_out_stages_downstream_of_a_missing_dependency(
    set_project_root: Path,
    all_pipelines: bool,
    on_error: OnError,
    runnable: set[str] | None,
) -> None:
    """Only a keep-going run across all pipelines runs the stages a missing dependency
    leaves runnable, leaving out the broken stage and every stage downstream of it; any other
    run stops."""
    stages = {
        "ok/a": _create_stage("ok/a", [], [ArtifactIdentity("ok/a", None)]),
        "bad/b": _create_stage(
            "bad/b", [ArtifactIdentity("missing.csv", None)], [ArtifactIdentity("bad/b", None)]
        ),
        "bad/c": _create_stage(
            "bad/c", [ArtifactIdentity("bad/b", None)], [ArtifactIdentity("bad/c", None)]
        ),
    }
    store = store_mod.WorkspaceStore(
        project_root=set_project_root, pipeline_name="bad", input_bindings={}
    )

    found = graph.get_runnable_stages(stages, store, all_pipelines=all_pipelines, on_error=on_error)

    assert (None if found.stages is None else set(found.stages)) == runnable
    assert [error.stage for error in found.errors] == ["bad/b"]
    [error] = found.errors
    assert error.input_path == set_project_root / "data" / "missing.csv.dat"
    assert str(error) == (
        "Stage 'bad/b' depends on input 'missing.csv' (path data/missing.csv.dat), "
        + "which does not exist"
    )
    assert found.left_out == ({} if runnable is None else {"bad/b": error, "bad/c": error})


@pytest.mark.parametrize(
    ("all_pipelines", "on_error", "watching", "leave_out"),
    [
        pytest.param(True, OnError.KEEP_GOING, False, True, id="all-keep-going"),
        pytest.param(True, OnError.KEEP_GOING, True, False, id="all-keep-going-watching"),
        pytest.param(True, OnError.FAIL, False, False, id="all-fail"),
        pytest.param(False, OnError.KEEP_GOING, False, False, id="one-keep-going"),
    ],
)
def test_missing_tracked_file_leave_out_mode_is_an_unwatched_keep_going_run_of_all_pipelines(
    all_pipelines: bool, on_error: OnError, watching: bool, leave_out: bool
) -> None:
    """Leave-out mode, where a missing tracked file leaves out only the stages that read it, is a
    run across all pipelines that keeps going past failures outside a watch session: the engine
    and the dry run both take it from ``is_leave_out_mode``."""
    found = graph.is_leave_out_mode(
        all_pipelines=all_pipelines, on_error=on_error, watching=watching
    )

    assert found is leave_out


def _helper_input_stage(name: str, inputs: list[str]) -> RegistryStageInfo:
    """A stage that reads the pipeline inputs named ``inputs``, in that order, and writes one
    output."""
    info = _create_stage(
        name, [ArtifactIdentity(input_name, None) for input_name in inputs], [_output(name)]
    )
    for dep in info["deps"].values():
        dep.tag = ArtifactTag.INPUT
    return info


def _output(stage_name: str) -> ArtifactIdentity:
    """The one output of stage ``stage_name``."""
    return ArtifactIdentity(stage_name, None)


def _helper_store(root: Path, input_bindings: dict[str, str]) -> store_mod.WorkspaceStore:
    return store_mod.WorkspaceStore(
        project_root=root, pipeline_name="p", input_bindings=input_bindings
    )


def _helper_missing_tracked(root: Path, *paths: str) -> list[str]:
    """Track each of ``paths`` (relative to ``root``) by its .pvt alone, with no data and no
    cache object, and return the tracked files the pre-flight finds missing, as it gives them.
    The directory holding each path exists, since it holds the .pvt."""
    for path in paths:
        helpers.track_without_data(root / path)
    return executor_core.find_missing_tracked_files(root).missing.paths


def _helper_runnable(
    stages: dict[str, RegistryStageInfo],
    store: store_mod.WorkspaceStore,
    missing_tracked: list[str],
    restored_tracked: Mapping[str, AbstractSet[tuple[str, ...]]] | None = None,
) -> graph.RunnableStages:
    """``get_runnable_stages`` of a keep-going run across all pipelines."""
    return graph.get_runnable_stages(
        stages,
        store,
        all_pipelines=True,
        on_error=OnError.KEEP_GOING,
        missing_tracked=missing_tracked,
        restored_tracked=restored_tracked,
    )


def test_get_runnable_stages_leaves_out_a_stage_reading_a_directory_holding_a_missing_tracked_file(
    set_project_root: Path,
) -> None:
    """A dependency on an existing directory that holds a missing tracked file leaves its stage
    out, with one error naming the stage, the directory and the file."""
    missing = _helper_missing_tracked(set_project_root, "data/dir/x.csv")
    stages = {
        "ok/a": _create_stage("ok/a", [], [_output("ok/a")]),
        "p/read": _helper_input_stage("p/read", ["dir"]),
    }

    found = _helper_runnable(stages, _helper_store(set_project_root, {"dir": "data/dir"}), missing)

    assert found.stages is not None
    assert set(found.stages) == {"ok/a"}
    [error] = found.errors
    assert error.stage == "p/read"
    assert error.input_path == set_project_root / "data" / "dir" / "x.csv"
    assert error.directory == set_project_root / "data" / "dir"
    assert str(error) == (
        "Stage 'p/read' depends on directory 'dir' (path data/dir), which holds tracked file "
        + "'data/dir/x.csv' that does not exist"
    )
    assert found.left_out == {"p/read": error}


def test_get_runnable_stages_reports_each_missing_tracked_file_a_directory_holds(
    set_project_root: Path,
) -> None:
    """A directory holding two missing tracked files gives its stage one error per file."""
    missing = _helper_missing_tracked(set_project_root, "data/dir/y.csv", "data/dir/x.csv")
    stages = {"p/read": _helper_input_stage("p/read", ["dir"])}

    found = _helper_runnable(stages, _helper_store(set_project_root, {"dir": "data/dir"}), missing)

    assert found.stages == {}
    assert [error.stage for error in found.errors] == ["p/read", "p/read"]
    assert [error.input_path for error in found.errors] == [
        set_project_root / "data" / "dir" / "x.csv",
        set_project_root / "data" / "dir" / "y.csv",
    ]
    assert [error.directory for error in found.errors] == [set_project_root / "data" / "dir"] * 2
    assert [str(error) for error in found.errors] == [
        "Stage 'p/read' depends on directory 'dir' (path data/dir), which holds tracked file "
        + f"'data/dir/{name}' that does not exist"
        for name in ("x.csv", "y.csv")
    ]
    assert found.left_out == {"p/read": found.errors[0]}


def test_get_runnable_stages_reports_a_missing_tracked_file_read_directly_and_by_its_directory_once(
    set_project_root: Path,
) -> None:
    """A stage that reads a missing tracked file and the directory holding it gets the error of
    the file's own dependency alone, though it lists the directory first."""
    missing = _helper_missing_tracked(set_project_root, "data/dir/x.csv")
    stages = {"p/read": _helper_input_stage("p/read", ["dir", "x"])}
    store = _helper_store(set_project_root, {"dir": "data/dir", "x": "data/dir/x.csv"})

    found = _helper_runnable(stages, store, missing)

    assert found.stages == {}
    [error] = found.errors
    assert error.stage == "p/read"
    assert error.input_path == set_project_root / "data" / "dir" / "x.csv"
    assert error.directory is None
    assert str(error) == (
        "Stage 'p/read' depends on input 'x' (path data/dir/x.csv), which does not exist"
    )
    assert found.left_out == {"p/read": error}


def test_get_runnable_stages_reports_dependencies_inside_one_missing_tracked_file_once_per_stage(
    set_project_root: Path,
) -> None:
    """Two dependencies of one stage inside one missing tracked directory give that stage one
    error, the first; another stage reading inside it gets its own."""
    missing = _helper_missing_tracked(set_project_root, "data/tracked")
    stages = {
        "p/read": _helper_input_stage("p/read", ["a", "b"]),
        "p/other": _helper_input_stage("p/other", ["b"]),
    }
    store = _helper_store(set_project_root, {"a": "data/tracked/a.csv", "b": "data/tracked/b.csv"})

    found = _helper_runnable(stages, store, missing)

    assert found.stages == {}
    assert [error.stage for error in found.errors] == ["p/read", "p/other"]
    assert [error.input_path for error in found.errors] == [
        set_project_root / "data" / "tracked" / "a.csv",
        set_project_root / "data" / "tracked" / "b.csv",
    ]
    assert [str(error) for error in found.errors] == [
        "Stage 'p/read' depends on input 'a' (path data/tracked/a.csv), which does not exist",
        "Stage 'p/other' depends on input 'b' (path data/tracked/b.csv), which does not exist",
    ]
    assert found.left_out == {"p/read": found.errors[0], "p/other": found.errors[1]}


def test_get_runnable_stages_reports_missing_untracked_inputs_beside_a_missing_tracked_file(
    set_project_root: Path,
) -> None:
    """Missing untracked inputs keep one error each, beside the error of a directory that holds
    a missing tracked file."""
    missing = _helper_missing_tracked(set_project_root, "data/dir/x.csv")
    stages = {"p/read": _helper_input_stage("p/read", ["gone", "lost", "dir"])}
    store = _helper_store(
        set_project_root, {"gone": "data/gone.csv", "lost": "data/lost.csv", "dir": "data/dir"}
    )

    found = _helper_runnable(stages, store, missing)

    assert found.stages == {}
    assert [error.stage for error in found.errors] == ["p/read"] * 3
    assert [error.input_path for error in found.errors] == [
        set_project_root / "data" / "gone.csv",
        set_project_root / "data" / "lost.csv",
        set_project_root / "data" / "dir" / "x.csv",
    ]
    assert [str(error) for error in found.errors] == [
        "Stage 'p/read' depends on input 'gone' (path data/gone.csv), which does not exist",
        "Stage 'p/read' depends on input 'lost' (path data/lost.csv), which does not exist",
        "Stage 'p/read' depends on directory 'dir' (path data/dir), which holds tracked file "
        + "'data/dir/x.csv' that does not exist",
    ]
    assert found.left_out == {"p/read": found.errors[0]}


def test_get_runnable_stages_finds_no_missing_tracked_file_in_a_directory_a_stage_produces(
    set_project_root: Path,
) -> None:
    """A dependency on a directory another stage produces is that stage's output, not a
    directory on disk: a missing tracked file inside it leaves its reader in."""
    made = ArtifactIdentity("p/make", None)
    missing = _helper_missing_tracked(set_project_root, "data/p/make/x.csv")
    stages = {
        "p/make": _create_stage("p/make", [], [made]),
        "p/use": _create_stage("p/use", [made], [_output("p/use")]),
    }
    stages["p/make"]["outs"][0].tag = ArtifactTag.DIRECTORY
    stages["p/use"]["deps"]["_0"].tag = ArtifactTag.DIRECTORY
    store = _helper_store(set_project_root, {})
    assert store.exists(stages["p/use"]["deps"]["_0"])

    found = _helper_runnable(stages, store, missing)

    assert found.stages is not None
    assert set(found.stages) == {"p/make", "p/use"}
    assert found.errors == []
    assert found.left_out == {}


def test_get_runnable_stages_decides_containment_by_path_parts(set_project_root: Path) -> None:
    """A missing tracked file ``data-backup/x.csv`` is not inside a dependency on ``data``."""
    missing = _helper_missing_tracked(set_project_root, "data-backup/x.csv")
    (set_project_root / "data").mkdir()
    stages = {"p/read": _helper_input_stage("p/read", ["data"])}

    found = _helper_runnable(stages, _helper_store(set_project_root, {"data": "data"}), missing)

    assert found.stages is not None
    assert set(found.stages) == {"p/read"}
    assert found.errors == []
    assert found.left_out == {}


@pytest.mark.parametrize(
    ("binding", "kept"),
    [
        pytest.param("data/tracked", False, id="the-restored-directory"),
        pytest.param("data/tracked/x.csv", False, id="a-file-of-its-listing"),
        pytest.param("data/tracked/sub", False, id="a-subdirectory-holding-its-files"),
        pytest.param("data/x.csv", False, id="a-restored-file"),
        pytest.param("data/tracked/inner/deep/z.csv", False, id="a-file-of-a-nested-directory"),
        pytest.param("data/tracked/cat.png", True, id="a-file-its-listing-lacks"),
        pytest.param("data/tracked/inner/cat.png", True, id="a-file-no-listing-holds"),
        pytest.param("data/x.csv/inner", True, id="inside-a-restored-file"),
    ],
)
def test_get_runnable_stages_counts_a_path_the_restore_creates_as_present(
    set_project_root: Path, binding: str, kept: bool
) -> None:
    """A dry run with --checkout-missing counts a missing dependency as present when the restore
    creates it: a restorable tracked file or directory, a file the directory's listing holds, or
    a subdirectory holding such files, also when a restorable directory lies inside another and
    its own listing holds the file. Any other missing dependency keeps its error, as the run
    still finds it missing after the restore; a stage reading a missing tracked file the cache
    cannot restore is left out either way."""
    [gone] = _helper_missing_tracked(set_project_root, "data/z.csv")
    tracked = set_project_root / "data" / "tracked"
    restored = {
        str(set_project_root / "data" / "x.csv"): set[tuple[str, ...]](),
        str(tracked): {("x.csv",), ("sub",), ("sub", "y.csv")},
        str(tracked / "inner"): {("deep",), ("deep", "z.csv")},
    }
    stages = {
        "p/read": _helper_input_stage("p/read", ["x"]),
        "p/other": _helper_input_stage("p/other", ["z"]),
    }
    store = _helper_store(set_project_root, {"x": binding, "z": "data/z.csv"})

    found = _helper_runnable(stages, store, [gone], restored_tracked=restored)

    assert found.stages is not None
    assert set(found.stages) == (set() if kept else {"p/read"})
    assert [(error.stage, error.input_path) for error in found.errors] == [
        *([("p/read", set_project_root / binding)] if kept else []),
        ("p/other", set_project_root / "data" / "z.csv"),
    ]
    assert str(found.errors[-1]) == (
        "Stage 'p/other' depends on input 'z' (path data/z.csv), which does not exist"
    )


def test_get_runnable_stages_leaves_out_a_stage_reading_a_restored_directory_holding_a_missing_file(
    set_project_root: Path,
) -> None:
    """A dry run with --checkout-missing counts the restorable directory images as present, so
    it finds what the run finds in images once it restored it: extra.bin, which a hand-edited
    .pvt tracks and neither images' listing nor any cache restores, leaves p/scan out with the
    error of the directory that holds it. p/scan's errors follow its dependencies' order, as the
    run's do: images, then the existing other. p/png reads a file the restore creates and is
    kept."""
    raw = set_project_root / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "extra.pvt").write_text("path: images/extra.bin\nhash: 0123456789abcdef\nsize: 10\n")
    missing = _helper_missing_tracked(set_project_root, "data/raw/other/y.bin")
    assert missing == [str(raw / "images" / "extra.bin"), str(raw / "other" / "y.bin")]
    stages = {
        "p/scan": _helper_input_stage("p/scan", ["images", "other"]),
        "p/png": _helper_input_stage("p/png", ["png"]),
    }
    store = _helper_store(
        set_project_root,
        {"images": "data/raw/images", "other": "data/raw/other", "png": "data/raw/images/a.png"},
    )

    found = _helper_runnable(
        stages, store, missing, restored_tracked={str(raw / "images"): {("a.png",)}}
    )

    assert found.stages is not None
    assert set(found.stages) == {"p/png"}
    assert [(error.stage, error.input_path, error.directory) for error in found.errors] == [
        ("p/scan", raw / "images" / "extra.bin", raw / "images"),
        ("p/scan", raw / "other" / "y.bin", raw / "other"),
    ]
    assert str(found.errors[0]) == (
        "Stage 'p/scan' depends on directory 'images' (path data/raw/images), which holds "
        + "tracked file 'data/raw/images/extra.bin' that does not exist"
    )
    assert found.left_out == {"p/scan": found.errors[0]}


def test_get_runnable_stages_counts_a_restored_path_reached_through_a_symlink_as_present(
    set_project_root: Path,
) -> None:
    """A dependency on ``alias/x``, where ``alias`` links to ``inputs/sub`` and the restore
    creates ``inputs/sub/x``, counts as present: the resolved path decides too."""
    sub = set_project_root / "data" / "inputs" / "sub"
    sub.mkdir(parents=True)
    (set_project_root / "data" / "alias").symlink_to(sub)
    stages = {"p/read": _helper_input_stage("p/read", ["x"])}
    store = _helper_store(set_project_root, {"x": "data/alias/x"})

    found = _helper_runnable(stages, store, [], restored_tracked={str(sub / "x"): set()})
    without = _helper_runnable(stages, store, [], restored_tracked={str(sub / "y"): set()})

    assert found.stages is not None
    assert set(found.stages) == {"p/read"}
    assert found.errors == []
    assert [error.input_path for error in without.errors] == [
        set_project_root / "data" / "alias" / "x"
    ]


def test_get_runnable_stages_leaves_out_a_stage_reading_a_symlink_to_a_directory_holding_it(
    set_project_root: Path,
) -> None:
    """A dependency on ``link``, a symlink to the directory ``real`` that holds a missing tracked
    file, reads that file: its stage is left out with one error naming the directory as the
    dependency gives it and the file."""
    missing = _helper_missing_tracked(set_project_root, "data/raw/real/f.txt")
    (set_project_root / "data" / "raw" / "real" / "present.txt").write_text("present")
    (set_project_root / "data" / "raw" / "link").symlink_to("real")
    stages = {"p/scan": _helper_input_stage("p/scan", ["link"])}

    found = _helper_runnable(
        stages, _helper_store(set_project_root, {"link": "data/raw/link"}), missing
    )

    assert found.stages == {}
    [error] = found.errors
    assert error.stage == "p/scan"
    assert error.input_path == set_project_root / "data" / "raw" / "real" / "f.txt"
    assert error.directory == set_project_root / "data" / "raw" / "link"
    assert str(error) == (
        "Stage 'p/scan' depends on directory 'link' (path data/raw/link), which holds tracked "
        + "file 'data/raw/real/f.txt' that does not exist"
    )
    assert found.left_out == {"p/scan": error}


def test_get_runnable_stages_reports_a_missing_tracked_file_reached_through_a_symlink_once(
    set_project_root: Path,
) -> None:
    """Two dependencies of one stage inside one missing tracked directory, one of them through
    the symlink ``alias`` to it, give that stage one error, the first."""
    missing = _helper_missing_tracked(set_project_root, "data/tracked")
    (set_project_root / "data" / "alias").symlink_to("tracked")
    stages = {"p/read": _helper_input_stage("p/read", ["a", "b"])}
    store = _helper_store(set_project_root, {"a": "data/alias/a.csv", "b": "data/tracked/b.csv"})

    found = _helper_runnable(stages, store, missing)

    assert found.stages == {}
    [error] = found.errors
    assert error.stage == "p/read"
    assert error.input_path == set_project_root / "data" / "alias" / "a.csv"
    assert found.left_out == {"p/read": error}


def test_get_runnable_stages_reports_a_missing_tracked_file_a_fifo_stands_in_once_per_stage(
    set_project_root: Path,
) -> None:
    """A FIFO where the tracked file data/dir/x.csv belongs exists on disk, but the pre-flight
    counts the file missing (``track.present_kind``), and that list decides: a stage reading
    the file is left out with the error a missing file gives, and not again for the directory
    that holds it; a stage reading the symlink ``alias`` to it reads it too, by its resolved
    path. Neither the pre-flight's view nor the check opens the FIFO."""
    fifo = set_project_root / "data" / "dir" / "x.csv"
    helpers.track_without_data(fifo)
    helpers.occupy(fifo, "fifo")
    (set_project_root / "data" / "alias").symlink_to("dir/x.csv")
    missing = helpers.call_without_waiting(
        lambda: executor_core.find_missing_tracked_files(set_project_root).missing.paths,
        lambda: helpers.free_fifo(fifo),
    )
    stages = {
        "ok/a": _create_stage("ok/a", [], [_output("ok/a")]),
        "p/read": _helper_input_stage("p/read", ["x", "dir"]),
        "p/link": _helper_input_stage("p/link", ["alias"]),
    }
    store = _helper_store(
        set_project_root, {"x": "data/dir/x.csv", "dir": "data/dir", "alias": "data/alias"}
    )

    found = helpers.call_without_waiting(
        lambda: _helper_runnable(stages, store, missing), lambda: helpers.free_fifo(fifo)
    )

    assert missing == [str(fifo)]
    assert found.stages is not None
    assert set(found.stages) == {"ok/a"}
    assert [str(error) for error in found.errors] == [
        "Stage 'p/read' depends on input 'x' (path data/dir/x.csv), which does not exist",
        "Stage 'p/link' depends on input 'alias' (path data/alias), which does not exist",
    ]
    assert [error.directory for error in found.errors] == [None, None]
    assert found.left_out == {"p/read": found.errors[0], "p/link": found.errors[1]}


def test_get_runnable_stages_leaves_out_downstream_stages_with_their_first_upstream_error(
    set_project_root: Path,
) -> None:
    """Each stage downstream of broken stages is left out with the first error, in error order,
    of a stage upstream of it; two levels down too."""
    missing = _helper_missing_tracked(set_project_root, "data/dir/x.csv")
    stages = {
        "ok/a": _create_stage("ok/a", [], [_output("ok/a")]),
        "p/read": _helper_input_stage("p/read", ["gone"]),
        "p/scan": _helper_input_stage("p/scan", ["dir"]),
        "p/mid": _create_stage("p/mid", [_output("p/scan"), _output("p/read")], [_output("p/mid")]),
        "p/end": _create_stage("p/end", [_output("p/mid")], [_output("p/end")]),
    }
    store = _helper_store(set_project_root, {"gone": "data/gone.csv", "dir": "data/dir"})

    found = _helper_runnable(stages, store, missing)

    assert found.stages is not None
    assert set(found.stages) == {"ok/a"}
    assert [error.stage for error in found.errors] == ["p/read", "p/scan"]
    read_error, scan_error = found.errors
    assert read_error.input_path == set_project_root / "data" / "gone.csv"
    assert scan_error.input_path == set_project_root / "data" / "dir" / "x.csv"
    assert found.left_out == {
        "p/read": read_error,
        "p/scan": scan_error,
        "p/mid": read_error,
        "p/end": read_error,
    }


def test_dependency_not_found_error_on_a_directory_survives_pickling(
    set_project_root: Path,
) -> None:
    """The error of a directory holding a missing tracked file keeps its directory, its file
    and its message across a pickle round trip."""
    missing = _helper_missing_tracked(set_project_root, "data/dir/x.csv")
    stages = {"p/read": _helper_input_stage("p/read", ["dir"])}
    [original] = _helper_runnable(
        stages, _helper_store(set_project_root, {"dir": "data/dir"}), missing
    ).errors

    restored = pickle.loads(pickle.dumps(original))

    assert isinstance(restored, exceptions.DependencyNotFoundError)
    assert restored.stage == "p/read"
    assert restored.directory == set_project_root / "data" / "dir"
    assert restored.input_path == set_project_root / "data" / "dir" / "x.csv"
    assert (
        str(restored)
        == str(original)
        == (
            "Stage 'p/read' depends on directory 'dir' (path data/dir), which holds tracked file "
            + "'data/dir/x.csv' that does not exist"
        )
    )


def test_build_graph_allows_missing_when_validate_false(tmp_path: Path) -> None:
    """build_graph does not validate dependency existence."""
    output_identity = ArtifactIdentity("stage_a", "output.csv")
    missing_dep = ArtifactIdentity("external", "missing.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [missing_dep], [output_identity]),
    }

    # Should not raise
    g = graph.build_graph(stages)
    assert "stage:stage_a" in g


# --- get_upstream_stages tests ---


@pytest.mark.usefixtures("clean_registry")
def test_get_upstream_stages_returns_producing_stages(tmp_path: Path) -> None:
    """get_upstream_stages returns stages that produce inputs for a stage."""
    input_identity = ArtifactIdentity("source", "input.csv")
    cleaned = ArtifactIdentity("preprocess", "cleaned.csv")
    features = ArtifactIdentity("extract", "features.csv")
    model = ArtifactIdentity("train", "model.pkl")

    stages = {
        "preprocess": _create_stage("preprocess", [input_identity], [cleaned]),
        "extract": _create_stage("extract", [input_identity], [features]),
        "train": _create_stage("train", [cleaned, features], [model]),
    }

    g = graph.build_graph(stages)
    upstream = graph.get_upstream_stages(g, "train")

    # train depends on outputs from preprocess and extract
    assert set(upstream) == {"preprocess", "extract"}


@pytest.mark.usefixtures("clean_registry")
def test_get_upstream_stages_empty_for_root_stage(tmp_path: Path) -> None:
    """get_upstream_stages returns empty list for stage with no upstream dependencies."""
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }

    g = graph.build_graph(stages)
    upstream = graph.get_upstream_stages(g, "stage_a")

    # stage_a has no upstream stages (only external input)
    assert upstream == []


@pytest.mark.usefixtures("clean_registry")
def test_get_upstream_stages_empty_for_unknown_stage(tmp_path: Path) -> None:
    """get_upstream_stages returns empty list for unknown stage."""
    g = graph.build_graph({})
    upstream = graph.get_upstream_stages(g, "unknown_stage")
    assert upstream == []


# --- get_execution_order with single_stage tests ---


@pytest.mark.usefixtures("clean_registry")
def test_get_execution_order_single_stage_mode(tmp_path: Path) -> None:
    """get_execution_order with single_stage=True returns only requested stages."""
    input_identity = ArtifactIdentity("source", "input.csv")
    intermediate = ArtifactIdentity("stage_a", "intermediate.csv")
    output_identity = ArtifactIdentity("stage_b", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [intermediate]),
        "stage_b": _create_stage("stage_b", [intermediate], [output_identity]),
    }

    bipartite = graph.build_graph(stages)
    stage_dag = graph.get_stage_dag(bipartite)

    # Single stage mode - should return only stage_b, NOT its dependency stage_a
    order = graph.get_execution_order(stage_dag, stages=["stage_b"], single_stage=True)

    assert order == ["stage_b"]
    assert "stage_a" not in order


@pytest.mark.usefixtures("clean_registry")
def test_get_execution_order_single_stage_preserves_order(tmp_path: Path) -> None:
    """get_execution_order with single_stage=True preserves input order."""
    input_identity = ArtifactIdentity("source", "input.csv")
    out_a = ArtifactIdentity("stage_a", "a.csv")
    out_b = ArtifactIdentity("stage_b", "b.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [out_a]),
        "stage_b": _create_stage("stage_b", [input_identity], [out_b]),
    }

    bipartite = graph.build_graph(stages)
    stage_dag = graph.get_stage_dag(bipartite)

    # Request in specific order - should be preserved
    order = graph.get_execution_order(stage_dag, stages=["stage_b", "stage_a"], single_stage=True)

    assert order == ["stage_b", "stage_a"]


# --- Additional edge case tests ---


@pytest.mark.usefixtures("clean_registry")
def test_get_producer_returns_none_for_unknown_path(tmp_path: Path) -> None:
    """get_producer returns None for completely unknown artifact path."""
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }

    g = graph.build_graph(stages)
    producer = graph.get_producer(g, ArtifactIdentity("unknown", "artifact"))

    assert producer is None


@pytest.mark.usefixtures("clean_registry")
def test_get_downstream_stages_empty_for_unknown_stage(tmp_path: Path) -> None:
    """get_downstream_stages returns empty list for unknown stage."""
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }

    g = graph.build_graph(stages)
    downstream = graph.get_downstream_stages(g, "unknown_stage")

    assert downstream == []


@pytest.mark.usefixtures("clean_registry")
def test_update_stage_adds_new_out(tmp_path: Path) -> None:
    """update_stage adds new output edges."""
    input_identity = ArtifactIdentity("source", "input.csv")
    out_a = ArtifactIdentity("stage_a", "a.csv")
    out_b = ArtifactIdentity("stage_a", "b.csv")

    # Initial: stage_a produces only out_a
    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [out_a]),
    }
    g = graph.build_graph(stages)

    assert graph.get_producer(g, out_a) == "stage_a"
    assert graph.get_producer(g, out_b) is None

    # Update: stage_a now also produces out_b
    new_info = _create_stage("stage_a", [input_identity], [out_a, out_b])
    graph.update_stage(g, "stage_a", new_info)

    assert graph.get_producer(g, out_a) == "stage_a"
    assert graph.get_producer(g, out_b) == "stage_a"


@pytest.mark.usefixtures("clean_registry")
def test_update_stage_removes_old_out(tmp_path: Path) -> None:
    """update_stage removes old output edges and orphaned artifacts."""
    input_identity = ArtifactIdentity("source", "input.csv")
    out_a = ArtifactIdentity("stage_a", "a.csv")
    out_b = ArtifactIdentity("stage_a", "b.csv")

    # Initial: stage_a produces both outputs
    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [out_a, out_b]),
    }
    g = graph.build_graph(stages)

    # Update: stage_a now only produces out_a
    new_info = _create_stage("stage_a", [input_identity], [out_a])
    graph.update_stage(g, "stage_a", new_info)

    assert graph.get_producer(g, out_a) == "stage_a"
    assert graph.get_producer(g, out_b) is None

    # out_b should be removed from graph (orphaned)
    assert graph.artifact_node(out_b) not in g


# --- Error path tests ---


@pytest.mark.usefixtures("clean_registry")
def test_cycle_detection_error_message_format(tmp_path: Path) -> None:
    """Cycle error message contains affected stage names."""
    stages = {
        "stage_a": _create_stage(
            "stage_a",
            [ArtifactIdentity("stage_b", "b.csv")],
            [ArtifactIdentity("stage_a", "a.csv")],
        ),
        "stage_b": _create_stage(
            "stage_b",
            [ArtifactIdentity("stage_a", "a.csv")],
            [ArtifactIdentity("stage_b", "b.csv")],
        ),
    }

    try:
        graph.build_graph(stages)
        pytest.fail("Should have raised CyclicGraphError")
    except exceptions.CyclicGraphError as e:
        # Error message should contain stage names
        assert "stage_a" in str(e) or "stage_b" in str(e)
        assert "Circular dependency" in str(e)


def test_get_execution_order_unknown_stage_raises_error(tmp_path: Path) -> None:
    """get_execution_order raises StageNotFoundError for unknown stages."""
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }

    bipartite = graph.build_graph(stages)
    stage_dag = graph.get_stage_dag(bipartite)

    # Should raise StageNotFoundError, not raw NetworkXError
    with pytest.raises(exceptions.StageNotFoundError, match="unknown_stage"):
        graph.get_execution_order(stage_dag, stages=["unknown_stage"])


def test_get_execution_order_mixed_known_unknown_stages(tmp_path: Path) -> None:
    """get_execution_order raises StageNotFoundError with all unknown stages."""
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }

    bipartite = graph.build_graph(stages)
    stage_dag = graph.get_stage_dag(bipartite)

    # Should raise with both unknown stages listed
    with pytest.raises(exceptions.StageNotFoundError, match="unknown"):
        graph.get_execution_order(stage_dag, stages=["stage_a", "unknown1", "unknown2"])


def test_get_execution_order_with_stages_returns_subgraph_order(tmp_path: Path) -> None:
    """get_execution_order with stages returns dependencies in correct order."""
    # Build diamond: input -> A, B -> C
    input_identity = ArtifactIdentity("source", "input.csv")
    a_out = ArtifactIdentity("stage_a", "a.csv")
    b_out = ArtifactIdentity("stage_b", "b.csv")
    c_out = ArtifactIdentity("stage_c", "c.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [a_out]),
        "stage_b": _create_stage("stage_b", [input_identity], [b_out]),
        "stage_c": _create_stage("stage_c", [a_out, b_out], [c_out]),
    }

    bipartite = graph.build_graph(stages)
    stage_dag = graph.get_stage_dag(bipartite)

    # Request only stage_c - should include its dependencies (a and b)
    order = graph.get_execution_order(stage_dag, stages=["stage_c"])

    assert "stage_a" in order
    assert "stage_b" in order
    assert "stage_c" in order
    # C must come after A and B
    assert order.index("stage_c") > order.index("stage_a")
    assert order.index("stage_c") > order.index("stage_b")


# --- extract_graph_view tests ---


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_empty() -> None:
    """extract_graph_view on empty graph returns empty lists."""
    g = graph.build_graph({})
    view = graph.extract_graph_view(g)

    assert view["stages"] == []
    assert view["artifacts"] == []
    assert view["stage_edges"] == []
    assert view["artifact_edges"] == []


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_single_stage(tmp_path: Path) -> None:
    """extract_graph_view extracts stage and artifact from single-stage graph."""
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    assert view["stages"] == ["stage_a"]
    assert set(view["artifacts"]) == {"source:input.csv", "stage_a:output.csv"}
    # Single stage with no downstream — no stage edges
    assert view["stage_edges"] == []
    # Artifact edges: input -> output (through stage_a)
    assert ("source:input.csv", "stage_a:output.csv") in view["artifact_edges"]


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_linear_chain(tmp_path: Path) -> None:
    """extract_graph_view extracts correct edges for a linear chain."""
    input_identity = ArtifactIdentity("source", "input.csv")
    intermediate = ArtifactIdentity("stage_a", "intermediate.csv")
    output_identity = ArtifactIdentity("stage_b", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [intermediate]),
        "stage_b": _create_stage("stage_b", [intermediate], [output_identity]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    assert set(view["stages"]) == {"stage_a", "stage_b"}
    assert set(view["artifacts"]) == {
        "source:input.csv",
        "stage_a:intermediate.csv",
        "stage_b:output.csv",
    }
    # stage_a -> stage_b (producer -> consumer, data-flow direction)
    assert ("stage_a", "stage_b") in view["stage_edges"]
    # artifact edges: input -> intermediate, intermediate -> output
    assert ("source:input.csv", "stage_a:intermediate.csv") in view["artifact_edges"]
    assert ("stage_a:intermediate.csv", "stage_b:output.csv") in view["artifact_edges"]


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_diamond(tmp_path: Path) -> None:
    """extract_graph_view handles diamond DAG correctly."""
    input_identity = ArtifactIdentity("source", "input.csv")
    clean = ArtifactIdentity("preprocess", "clean.csv")
    feats = ArtifactIdentity("features", "feats.csv")
    model = ArtifactIdentity("train", "model.pkl")

    stages = {
        "preprocess": _create_stage("preprocess", [input_identity], [clean]),
        "features": _create_stage("features", [input_identity], [feats]),
        "train": _create_stage("train", [clean, feats], [model]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    assert set(view["stages"]) == {"preprocess", "features", "train"}
    # Stage edges (producer -> consumer)
    assert ("preprocess", "train") in view["stage_edges"]
    assert ("features", "train") in view["stage_edges"]


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_stage_with_multiple_outputs(tmp_path: Path) -> None:
    """extract_graph_view deduplicates stage edges when multiple artifacts connect stages.

    Stage A produces [file1.csv, file2.csv], Stage B consumes both.
    Should create ONE stage edge (A->B) despite two artifact paths.
    """
    input_identity = ArtifactIdentity("source", "input.csv")
    file1 = ArtifactIdentity("stage_a", "file1.csv")
    file2 = ArtifactIdentity("stage_a", "file2.csv")
    output_identity = ArtifactIdentity("stage_b", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [file1, file2]),
        "stage_b": _create_stage("stage_b", [file1, file2], [output_identity]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    assert set(view["stages"]) == {"stage_a", "stage_b"}

    # Stage edges: A->B appears ONCE (deduplicated despite two artifact connections)
    stage_edges = view["stage_edges"]
    assert stage_edges.count(("stage_a", "stage_b")) == 1, (
        "Stage edges should be deduplicated when multiple artifacts connect same stages"
    )
    assert len(stage_edges) == 1, f"Expected exactly 1 stage edge, got {len(stage_edges)}"

    # Artifact edges: input -> file1, input -> file2, file1 -> output, file2 -> output
    artifact_edges = set(view["artifact_edges"])
    assert ("source:input.csv", "stage_a:file1.csv") in artifact_edges
    assert ("source:input.csv", "stage_a:file2.csv") in artifact_edges
    assert ("stage_a:file1.csv", "stage_b:output.csv") in artifact_edges
    assert ("stage_a:file2.csv", "stage_b:output.csv") in artifact_edges
    assert len(artifact_edges) == 4


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_external_input_artifacts(tmp_path: Path) -> None:
    """extract_graph_view handles external inputs (no producer) correctly.

    External input files should appear in artifacts but have no incoming edges.
    """
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    # Check external input has NO incoming artifact edges
    artifact_edges = view["artifact_edges"]
    for _src, dst in artifact_edges:
        assert dst != "source:input.csv", (
            "External input source:input.csv should not be a destination in artifact edges"
        )

    # But it SHOULD have outgoing edges
    assert ("source:input.csv", "stage_a:output.csv") in artifact_edges


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_no_spurious_edges(tmp_path: Path) -> None:
    """extract_graph_view should not create edges between unconnected stages."""
    input_a = ArtifactIdentity("source", "input_a.csv")
    input_b = ArtifactIdentity("source", "input_b.csv")
    output_a = ArtifactIdentity("stage_a", "output_a.csv")
    output_b = ArtifactIdentity("stage_b", "output_b.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_a], [output_a]),
        "stage_b": _create_stage("stage_b", [input_b], [output_b]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    # No stage edges should exist (disconnected components)
    assert view["stage_edges"] == []

    # Only artifact edges within each component
    artifact_edges = set(view["artifact_edges"])
    assert ("source:input_a.csv", "stage_a:output_a.csv") in artifact_edges
    assert ("source:input_b.csv", "stage_b:output_b.csv") in artifact_edges

    # No cross-edges between components
    assert ("source:input_a.csv", "stage_b:output_b.csv") not in artifact_edges
    assert ("source:input_b.csv", "stage_a:output_a.csv") not in artifact_edges

    # Exact count
    assert len(artifact_edges) == 2


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_complex_diamond_with_edge_verification(tmp_path: Path) -> None:
    """extract_graph_view with strict edge verification for complex diamond.

    Verifies exact edge count and directionality for diamond pattern.
    """
    input_identity = ArtifactIdentity("source", "input.csv")
    clean = ArtifactIdentity("preprocess", "clean.csv")
    feats = ArtifactIdentity("features", "feats.csv")
    model = ArtifactIdentity("train", "model.pkl")

    stages = {
        "preprocess": _create_stage("preprocess", [input_identity], [clean]),
        "features": _create_stage("features", [input_identity], [feats]),
        "train": _create_stage("train", [clean, feats], [model]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    # Verify stage edges with exact counts and no reverse edges
    stage_edges = view["stage_edges"]
    assert stage_edges.count(("preprocess", "train")) == 1
    assert stage_edges.count(("features", "train")) == 1
    assert ("train", "preprocess") not in stage_edges
    assert ("train", "features") not in stage_edges

    # Total stage edges should be exactly 2
    assert len(stage_edges) == 2

    # Verify artifact edges
    artifact_edges = set(view["artifact_edges"])
    # From input to intermediates
    assert ("source:input.csv", "preprocess:clean.csv") in artifact_edges
    assert ("source:input.csv", "features:feats.csv") in artifact_edges
    # From intermediates to model
    assert ("preprocess:clean.csv", "train:model.pkl") in artifact_edges
    assert ("features:feats.csv", "train:model.pkl") in artifact_edges

    # NO reverse edges
    assert ("preprocess:clean.csv", "source:input.csv") not in artifact_edges
    assert ("train:model.pkl", "preprocess:clean.csv") not in artifact_edges

    # Exact count
    assert len(artifact_edges) == 4


# --- Width tests ---


def _helper_stage_dag(dependencies: dict[str, list[str]]) -> nx.DiGraph[str]:
    """Stage DAG in which each stage consumes the output of every stage it lists."""
    stages = {
        name: _create_stage(
            name,
            [ArtifactIdentity(dep, "out.csv") for dep in deps],
            [ArtifactIdentity(name, "out.csv")],
        )
        for name, deps in dependencies.items()
    }
    return graph.get_stage_dag(graph.build_graph(stages))


@pytest.mark.parametrize(
    ("dependencies", "run", "width"),
    [
        pytest.param({"a": [], "b": ["a"], "c": ["b"], "d": ["c"]}, None, 1, id="chain"),
        pytest.param(
            {"src": [], **{f"m{i}": ["src"] for i in range(5)}}, None, 5, id="fan-out-of-5"
        ),
        pytest.param(
            {"top": [], "left": ["top"], "right": ["top"], "bottom": ["left", "right"]},
            None,
            2,
            id="diamond",
        ),
        # Levels {s0, s1} and {s2, s3}: the widest level is 2, but s0, s2 and s3 run together.
        pytest.param(
            {"s0": [], "s1": [], "s2": ["s1"], "s3": ["s1"]},
            None,
            3,
            id="widest-level-undercounts",
        ),
        # `pivot run a b` orders b after a only through x, which is not in the run.
        pytest.param({"a": [], "x": ["a"], "b": ["x"]}, ["a", "b"], 2, id="path-outside-run"),
        pytest.param({"a": [], "x": ["a"], "b": ["x"]}, ["b", "a", "x"], 1, id="path-inside-run"),
        pytest.param({"a": [], "b": []}, ["a", "a"], 1, id="repeated-stage"),
    ],
)
def test_get_width_is_the_most_stages_that_can_run_at_once(
    dependencies: dict[str, list[str]], run: list[str] | None, width: int
) -> None:
    stage_dag = _helper_stage_dag(dependencies)
    stages = list(dependencies) if run is None else run
    assert graph.get_width(stage_dag, stages, limit=len(stages) + 1) == width
    assert graph.get_width(stage_dag, stages, limit=1) == 1


def _helper_two_dimensional_order(count: int, spread: float) -> nx.DiGraph[str]:
    """Stage DAG of a 2-dimensional order: a stage depends on each earlier stage whose second
    coordinate is also smaller, kept to the direct dependencies. Its topological levels are
    narrower than its width, so the search for the width has work to do."""
    rng = random.Random(count)
    xs = sorted(rng.random() * count for _ in range(count))
    points = [(x, x + spread * rng.random()) for x in xs]
    stage_dag: nx.DiGraph[str] = nx.DiGraph()
    for i in range(count):
        stage_dag.add_node(f"s{i:03d}")
    for b in range(count):
        highest = float("-inf")
        for a in range(b - 1, -1, -1):
            if points[a][0] + spread < highest:
                break
            if highest < points[a][1] < points[b][1]:
                stage_dag.add_edge(f"s{b:03d}", f"s{a:03d}")
                highest = points[a][1]
    return stage_dag


def _helper_reinserted(stage_dag: nx.DiGraph[str], rng: random.Random) -> nx.DiGraph[str]:
    """The same stage DAG with its stages and edges added in a shuffled order, so it lists each
    stage's dependencies in another order."""
    reinserted: nx.DiGraph[str] = nx.DiGraph()
    nodes, edges = list(stage_dag.nodes), list(stage_dag.edges)
    for node in rng.sample(nodes, len(nodes)):
        reinserted.add_node(node)
    for consumer, producer in rng.sample(edges, len(edges)):
        reinserted.add_edge(consumer, producer)
    return reinserted


@pytest.mark.parametrize(
    ("count", "spread"),
    [
        pytest.param(12, 6.0, id="12-stages"),
        pytest.param(30, 12.0, id="30-stages"),
        pytest.param(60, 20.0, id="60-stages"),
        pytest.param(140, 64.0, id="140-stages"),
    ],
)
def test_get_width_does_not_depend_on_the_order_of_stages(
    count: int, spread: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One run gives one pool size whatever the order of its stages and of the stage DAG's stages
    and edges, at every step budget from none up to one every order's search finishes within."""
    stage_dag = _helper_two_dimensional_order(count, spread)
    stages = sorted(stage_dag.nodes())
    finished = graph.get_width(stage_dag, stages, limit=count + 1)
    orders = [(stage_dag, stages)]
    for seed in range(4):
        rng = random.Random(seed)
        orders.append((_helper_reinserted(stage_dag, rng), rng.sample(stages, count)))

    for budget in range(graph._AUGMENT_STEPS + 1):
        monkeypatch.setattr(graph, "_AUGMENT_STEPS", budget)
        widths = {graph.get_width(dag, run, limit=count + 1) for dag, run in orders}
        assert len(widths) == 1, f"with {budget} steps, orders of one run gave {sorted(widths)}"
        if widths == {finished}:
            break
