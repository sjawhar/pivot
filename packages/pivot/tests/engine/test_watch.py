"""Tests for the WatchCoordinator."""

from __future__ import annotations

import pathlib

import networkx as nx

from pivot.engine import graph as engine_graph
from pivot.engine.types import NodeType, StageExecutionState
from pivot.engine.watch import WatchCoordinator


def _helper_build_graph(
    stages: dict[str, dict[str, list[str]]],
) -> nx.DiGraph[str]:
    """Build a minimal bipartite graph for testing.

    Args:
        stages: Mapping of stage_name -> {"deps": [paths], "outs": [paths]}
    """
    g: nx.DiGraph[str] = nx.DiGraph()
    for name, info in stages.items():
        stage = engine_graph.stage_node(name)
        g.add_node(stage, type=NodeType.STAGE)
        for dep in info.get("deps", []):
            art = engine_graph.artifact_node(pathlib.Path(dep))
            g.add_node(art, type=NodeType.ARTIFACT)
            g.add_edge(art, stage)
        for out in info.get("outs", []):
            art = engine_graph.artifact_node(pathlib.Path(out))
            g.add_node(art, type=NodeType.ARTIFACT)
            g.add_edge(stage, art)
    return g


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
    g = _helper_build_graph(
        {
            "extract": {"deps": ["/data/raw.csv"], "outs": ["/data/clean.csv"]},
            "train": {"deps": ["/data/clean.csv"], "outs": ["/models/model.pkl"]},
            "evaluate": {"deps": ["/models/model.pkl"], "outs": ["/results/metrics.json"]},
        }
    )
    coord = WatchCoordinator(graph=g)

    affected = coord.get_affected_stages([pathlib.Path("/data/raw.csv")])
    assert set(affected) == {"extract", "train", "evaluate"}, (
        "all downstream stages should be affected"
    )


def test_affected_stages_unknown_path_returns_empty() -> None:
    """Path not in graph returns no affected stages."""
    g = _helper_build_graph(
        {
            "stage_a": {"deps": ["/input.csv"], "outs": ["/output.csv"]},
        }
    )
    coord = WatchCoordinator(graph=g)

    assert coord.get_affected_stages([pathlib.Path("/unknown.csv")]) == []


def test_affected_stages_deduplicates() -> None:
    """Multiple paths affecting the same stage are deduplicated."""
    g = _helper_build_graph(
        {
            "stage_a": {"deps": ["/input1.csv", "/input2.csv"], "outs": ["/output.csv"]},
        }
    )
    coord = WatchCoordinator(graph=g)

    affected = coord.get_affected_stages(
        [
            pathlib.Path("/input1.csv"),
            pathlib.Path("/input2.csv"),
        ]
    )
    assert affected == ["stage_a"], "should deduplicate"


# =============================================================================
# Path Filtering
# =============================================================================


def test_should_filter_path_true_for_preparing_and_running_producer() -> None:
    """Paths produced by a PREPARING, WAITING_ON_LOCK, or RUNNING stage are filtered."""
    g = _helper_build_graph(
        {
            "stage_a": {"deps": ["/input.csv"], "outs": ["/output.csv"]},
        }
    )
    coord = WatchCoordinator(graph=g)
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
    g = _helper_build_graph(
        {
            "stage_a": {"deps": ["/input.csv"], "outs": ["/output.csv"]},
        }
    )
    coord = WatchCoordinator(graph=g)
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
    g = _helper_build_graph(
        {
            "stage_a": {"deps": ["/input.csv"], "outs": ["/output.csv"]},
        }
    )
    coord = WatchCoordinator(graph=g)

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
    coord = WatchCoordinator(graph=nx.DiGraph())
    assert coord.should_restart_workers(parallel=True) is True


def test_should_restart_workers_false_when_not_parallel() -> None:
    """Worker restart not recommended in sequential mode."""
    coord = WatchCoordinator(graph=nx.DiGraph())
    assert coord.should_restart_workers(parallel=False) is False
