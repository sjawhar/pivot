"""Stages added to a running schedule: the instances a template expands into mid-run.

``engine_graph.add_stages`` grows the run's bipartite graph, and ``Scheduler.extend`` adds the
new stages to a schedule already under way, keeping every finished stage's state.
"""

from __future__ import annotations

import networkx as nx
import pytest

from pivot import loaders, types
from pivot.engine import graph as engine_graph
from pivot.engine.scheduler import Scheduler
from pivot.engine.types import NodeType, StageExecutionState
from pivot.registry import RegistryStageInfo

_COMPLETED = StageExecutionState.COMPLETED


def _helper_graph(names: list[str], edges: list[tuple[str, str]]) -> nx.DiGraph[str]:
    g: nx.DiGraph[str] = nx.DiGraph()
    for name in names:
        g.add_node(engine_graph.stage_node(name), type=NodeType.STAGE)
    _helper_add_edges(g, edges)
    return g


def _helper_add_edges(g: nx.DiGraph[str], edges: list[tuple[str, str]]) -> None:
    for src, dst in edges:
        for name in (src, dst):
            g.add_node(engine_graph.stage_node(name), type=NodeType.STAGE)
        art = f"artifact:{src}__{dst}"
        g.add_node(art, type=NodeType.ARTIFACT)
        g.add_edge(engine_graph.stage_node(src), art)
        g.add_edge(art, engine_graph.stage_node(dst))


def _helper_scheduler(names: list[str], g: nx.DiGraph[str], *, cpu: int = 4) -> Scheduler:
    scheduler = Scheduler()
    scheduler.initialize(
        names,
        g,
        stage_resources={name: {"cpu": 1} for name in names},
        resource_capacity={"cpu": cpu},
    )
    return scheduler


def _helper_complete(scheduler: Scheduler, stage: str, *, failed: bool = False) -> list[str]:
    scheduler.set_state(stage, _COMPLETED)
    ready, _ = scheduler.on_stage_completed(stage, failed=failed)
    return ready


def _helper_extend(
    scheduler: Scheduler, g: nx.DiGraph[str], stages: list[str]
) -> list[tuple[str, StageExecutionState]]:
    return scheduler.extend(stages, g, stage_resources={name: {"cpu": 1} for name in stages})


# =============================================================================
# Scheduler.extend
# =============================================================================


def test_stages_added_after_their_producer_finished_are_ready_in_order() -> None:
    g = _helper_graph(["plan"], [])
    scheduler = _helper_scheduler(["plan"], g)
    _helper_complete(scheduler, "plan")
    _helper_add_edges(g, [("plan", "cut@a"), ("plan", "cut@b"), ("cut@a", "build@a")])

    states = _helper_extend(scheduler, g, ["cut@a", "cut@b", "build@a"])

    assert states == [
        ("cut@a", StageExecutionState.READY),
        ("cut@b", StageExecutionState.READY),
        ("build@a", StageExecutionState.PENDING),
    ]
    assert list(scheduler.stage_states) == ["plan", "cut@a", "cut@b", "build@a"]
    assert scheduler.can_start("cut@a")
    assert not scheduler.can_start("build@a")
    assert _helper_complete(scheduler, "cut@a") == ["build@a"]


def test_a_stage_added_while_another_upstream_runs_waits_for_it() -> None:
    g = _helper_graph(["plan", "config"], [])
    scheduler = _helper_scheduler(["plan", "config"], g)
    scheduler.set_state("config", StageExecutionState.RUNNING)
    _helper_complete(scheduler, "plan")
    _helper_add_edges(g, [("plan", "cut@a"), ("config", "cut@a")])

    assert _helper_extend(scheduler, g, ["cut@a"]) == [("cut@a", StageExecutionState.PENDING)]

    assert _helper_complete(scheduler, "config") == ["cut@a"]


def test_a_stage_added_under_a_failed_upstream_is_blocked_with_its_chain() -> None:
    g = _helper_graph(["plan", "config"], [])
    scheduler = _helper_scheduler(["plan", "config"], g)
    _helper_complete(scheduler, "config", failed=True)
    _helper_complete(scheduler, "plan")
    _helper_add_edges(g, [("plan", "cut@a"), ("config", "cut@a"), ("cut@a", "build@a")])

    states = _helper_extend(scheduler, g, ["cut@a", "build@a"])

    assert states == [
        ("cut@a", StageExecutionState.BLOCKED),
        ("build@a", StageExecutionState.BLOCKED),
    ]


def test_a_later_failure_upstream_blocks_the_added_chain() -> None:
    g = _helper_graph(["root", "plan", "config"], [("root", "config")])
    scheduler = _helper_scheduler(["root", "plan", "config"], g)
    _helper_complete(scheduler, "plan")
    _helper_add_edges(g, [("plan", "cut@a"), ("config", "cut@a"), ("cut@a", "build@a")])
    _helper_extend(scheduler, g, ["cut@a", "build@a"])

    scheduler.set_state("root", _COMPLETED)
    _, blocked = scheduler.on_stage_completed("root", failed=True)

    assert {name for name, _ in blocked} == {"config", "cut@a", "build@a"}


def test_extend_refuses_a_request_for_a_resource_with_no_capacity() -> None:
    g = _helper_graph(["plan"], [])
    scheduler = _helper_scheduler(["plan"], g)
    _helper_add_edges(g, [("plan", "cut@a")])

    with pytest.raises(ValueError, match="no capacity"):
        scheduler.extend(["cut@a"], g, stage_resources={"cut@a": {"gpu": 1}})


def test_extend_refuses_resources_that_do_not_name_the_new_stages() -> None:
    g = _helper_graph(["plan"], [])
    scheduler = _helper_scheduler(["plan"], g)
    _helper_add_edges(g, [("plan", "cut@a")])

    with pytest.raises(ValueError, match="missing"):
        scheduler.extend(["cut@a"], g, stage_resources={})


def test_an_added_stage_holds_its_resources_like_any_other() -> None:
    g = _helper_graph(["plan"], [])
    scheduler = _helper_scheduler(["plan"], g, cpu=1)
    _helper_complete(scheduler, "plan")
    _helper_add_edges(g, [("plan", "cut@a"), ("plan", "cut@b")])
    _helper_extend(scheduler, g, ["cut@a", "cut@b"])

    scheduler.acquire_resources("cut@a")

    assert not scheduler.can_start("cut@b"), "one cpu slot, held by cut@a"
    scheduler.release_resources("cut@a")
    assert scheduler.can_start("cut@b")


# =============================================================================
# engine_graph.add_stages
# =============================================================================


def _helper_ref(identity: types.ArtifactIdentity) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=identity, format=loaders.YAML(), python_type=dict, tag=types.ArtifactTag.DATA
    )


def _helper_noop() -> None:
    pass


def _helper_stage(name: str, deps: dict[str, types.ArtifactIdentity]) -> RegistryStageInfo:
    return RegistryStageInfo(
        func=_helper_noop,
        name=name,
        deps={dep: _helper_ref(identity) for dep, identity in deps.items()},
        outs=[_helper_ref(types.ArtifactIdentity(name, None))],
        params=None,
        resources={},
        variant=None,
        signature=None,
        fingerprint=None,
        params_arg_name=None,
        state_dir=None,
        collection_params={},
        no_fingerprint=False,
        code_deps=[],
        input_paths={},
        incremental_inputs={},
        unsafe_fingerprint=False,
    )


def test_add_stages_grows_a_graph_to_the_one_build_graph_makes() -> None:
    first = {"a": _helper_stage("a", {})}
    later = {"b": _helper_stage("b", {"x": types.ArtifactIdentity("a", None)})}
    g = engine_graph.build_graph(first)

    engine_graph.add_stages(g, later)

    whole = engine_graph.build_graph({**first, **later})
    assert set(g.nodes) == set(whole.nodes)
    assert set(g.edges) == set(whole.edges)
    assert engine_graph.get_upstream_stages(g, "b") == ["a"]
