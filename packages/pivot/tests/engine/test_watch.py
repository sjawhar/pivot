"""Tests for the WatchCoordinator."""

from __future__ import annotations

import pathlib

import networkx as nx

from pivot import project
from pivot import types as pivot_types
from pivot.engine import graph as engine_graph
from pivot.engine.types import NodeType, StageExecutionState
from pivot.engine.watch import WatchCoordinator


def _helper_build_graph(
    stages: dict[str, dict[str, list[str]]],
) -> tuple[nx.DiGraph[str], dict[pathlib.Path, pivot_types.ArtifactIdentity]]:
    """Build a minimal identity graph and its filesystem path index."""
    graph: nx.DiGraph[str] = nx.DiGraph()
    identities = dict[pathlib.Path, pivot_types.ArtifactIdentity]()

    def _identity(path_str: str, producer: str | None = None) -> pivot_types.ArtifactIdentity:
        path = project.normalize_path(path_str)
        if path not in identities:
            identities[path] = pivot_types.ArtifactIdentity(
                producer or f"input/{path.name}",
                None,
            )
        return identities[path]

    for name, info in stages.items():
        stage = engine_graph.stage_node(name)
        graph.add_node(stage, type=NodeType.STAGE)
        for dep in info.get("deps", []):
            artifact = engine_graph.artifact_node(_identity(dep))
            graph.add_node(artifact, type=NodeType.ARTIFACT)
            graph.add_edge(artifact, stage)
        for out in info.get("outs", []):
            artifact = engine_graph.artifact_node(_identity(out, producer=name))
            graph.add_node(artifact, type=NodeType.ARTIFACT)
            graph.add_edge(stage, artifact)
    return graph, identities


def _helper_state_map(
    state: StageExecutionState,
) -> dict[str, StageExecutionState]:
    """Create a defaultdict-like lookup that returns a fixed state for any stage."""

    class _FixedState(dict[str, StageExecutionState]):
        def __missing__(self, key: str) -> StageExecutionState:
            return state

    return _FixedState()


# =============================================================================
# Affected Stage Computation
# =============================================================================


def test_affected_stages_returns_consumers_and_downstream() -> None:
    """Changed input affects its consumer and all transitive downstream stages."""
    graph, artifact_paths = _helper_build_graph(
        {
            "extract": {"deps": ["/data/raw.csv"], "outs": ["/data/clean.csv"]},
            "train": {"deps": ["/data/clean.csv"], "outs": ["/models/model.pkl"]},
            "evaluate": {"deps": ["/models/model.pkl"], "outs": ["/results/metrics.json"]},
        }
    )
    coord = WatchCoordinator(graph, artifact_paths)

    affected = coord.get_affected_stages([pathlib.Path("/data/raw.csv")])
    assert set(affected) == {"extract", "train", "evaluate"}, (
        "all downstream stages should be affected"
    )


def test_affected_stages_unknown_path_returns_empty() -> None:
    """Path not in graph returns no affected stages."""
    graph, artifact_paths = _helper_build_graph(
        {
            "stage_a": {"deps": ["/input.csv"], "outs": ["/output.csv"]},
        }
    )
    coord = WatchCoordinator(graph, artifact_paths)

    assert coord.get_affected_stages([pathlib.Path("/unknown.csv")]) == []


def test_affected_stages_deduplicates() -> None:
    """Multiple paths affecting the same stage are deduplicated."""
    graph, artifact_paths = _helper_build_graph(
        {
            "stage_a": {"deps": ["/input1.csv", "/input2.csv"], "outs": ["/output.csv"]},
        }
    )
    coord = WatchCoordinator(graph, artifact_paths)

    affected = coord.get_affected_stages(
        [
            pathlib.Path("/input1.csv"),
            pathlib.Path("/input2.csv"),
        ]
    )
    assert affected == ["stage_a"], "should deduplicate"


def test_affected_output_path_does_not_resolve_to_an_input(
    tmp_path: pathlib.Path,
) -> None:
    """An output symlink must not gain consumers of the input it resolves to."""
    input_path = tmp_path / "input.txt"
    input_path.write_text("input")
    output_path = tmp_path / "output.txt"
    output_path.symlink_to(input_path)
    graph, artifact_paths = _helper_build_graph(
        {
            "source": {"outs": [str(output_path)]},
            "consume_output": {"deps": [str(output_path)]},
            "consume_input": {"deps": [str(input_path)]},
        }
    )
    coord = WatchCoordinator(graph, artifact_paths)

    assert coord.get_affected_stages([output_path]) == ["consume_output"]


def test_physical_output_event_uses_symlinked_output_identity(
    tmp_path: pathlib.Path,
) -> None:
    """A physical output event retains its producer and does not route through raw aliases."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(real_dir, target_is_directory=True)
    output_path = linked_dir / "out.txt"
    physical_output = real_dir / "out.txt"
    physical_output.write_text("output")
    raw_alias = tmp_path / "raw-alias.txt"
    raw_alias.symlink_to(physical_output)
    graph, artifact_paths = _helper_build_graph(
        {
            "source": {"outs": [str(output_path)]},
            "consume_output": {"deps": [str(output_path)]},
            "consume_input": {"deps": [str(raw_alias)]},
        }
    )
    coord = WatchCoordinator(graph, artifact_paths)

    assert coord.get_producer(physical_output) == "source"
    assert (
        coord.should_filter_path(
            physical_output,
            get_stage_state=_helper_state_map(StageExecutionState.RUNNING).__getitem__,
        )
        is True
    )
    assert coord.get_affected_stages([physical_output]) == ["consume_output"]


# =============================================================================
# Path Filtering
# =============================================================================


def test_should_filter_path_true_for_preparing_and_running_producer() -> None:
    """Paths produced by a PREPARING, WAITING_ON_LOCK, or RUNNING stage are filtered."""
    graph, artifact_paths = _helper_build_graph(
        {
            "stage_a": {"deps": ["/input.csv"], "outs": ["/output.csv"]},
        }
    )
    coord = WatchCoordinator(graph, artifact_paths)
    output = pathlib.Path("/output.csv")

    for state in (
        StageExecutionState.PREPARING,
        StageExecutionState.WAITING_ON_LOCK,
        StageExecutionState.RUNNING,
    ):
        state_map = _helper_state_map(state)
        assert coord.should_filter_path(output, get_stage_state=state_map.__getitem__) is True, (
            f"should filter when producer is {state.name}"
        )


def test_should_filter_path_false_for_non_executing_producer() -> None:
    """Paths produced by PENDING, BLOCKED, READY, or COMPLETED stages are NOT filtered."""
    graph, artifact_paths = _helper_build_graph(
        {
            "stage_a": {"deps": ["/input.csv"], "outs": ["/output.csv"]},
        }
    )
    coord = WatchCoordinator(graph, artifact_paths)
    output = pathlib.Path("/output.csv")

    for state in (
        StageExecutionState.PENDING,
        StageExecutionState.BLOCKED,
        StageExecutionState.READY,
        StageExecutionState.COMPLETED,
    ):
        state_map = _helper_state_map(state)
        assert coord.should_filter_path(output, get_stage_state=state_map.__getitem__) is False, (
            f"should NOT filter when producer is {state.name}"
        )


def test_should_filter_path_false_for_input_artifact() -> None:
    """Input artifacts (no producer) should never be filtered."""
    graph, artifact_paths = _helper_build_graph(
        {
            "stage_a": {"deps": ["/input.csv"], "outs": ["/output.csv"]},
        }
    )
    coord = WatchCoordinator(graph, artifact_paths)

    state_map = _helper_state_map(StageExecutionState.RUNNING)
    assert (
        coord.should_filter_path(pathlib.Path("/input.csv"), get_stage_state=state_map.__getitem__)
        is False
    )


# =============================================================================
# Worker Restart Policy
# =============================================================================


def test_should_restart_workers_true_when_parallel() -> None:
    """Worker restart recommended when parallel mode is enabled."""
    coord = WatchCoordinator(nx.DiGraph(), {})
    assert coord.should_restart_workers(parallel=True) is True


def test_should_restart_workers_false_when_not_parallel() -> None:
    """Worker restart not recommended in sequential mode."""
    coord = WatchCoordinator(nx.DiGraph(), {})
    assert coord.should_restart_workers(parallel=False) is False
