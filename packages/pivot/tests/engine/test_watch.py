"""Tests for the WatchCoordinator."""

from __future__ import annotations

import pathlib

import anyio
import networkx as nx
import pytest

from pivot import project
from pivot import types as pivot_types
from pivot.engine import graph as engine_graph
from pivot.engine import sources
from pivot.engine.types import InputEvent, NodeType, StageExecutionState
from pivot.engine.watch import WatchCoordinator
from pivot.storage import store as store_mod


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


def test_affected_stages_keep_a_backslash_in_a_name_apart_from_a_separator() -> None:
    """A change to ``a\\b.csv`` (one file name) does not affect the consumer of ``a/b.csv``."""
    graph, artifact_paths = _helper_build_graph(
        {
            "use_name": {"deps": ["/data/a\\b.csv"], "outs": ["/data/name.out"]},
            "use_nested": {"deps": ["/data/a/b.csv"], "outs": ["/data/nested.out"]},
        }
    )
    coord = WatchCoordinator(graph, artifact_paths)

    assert coord.get_affected_stages([pathlib.Path("/data/a\\b.csv")]) == ["use_name"]
    assert coord.get_affected_stages([pathlib.Path("/data/a/b.csv")]) == ["use_nested"]


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


@pytest.mark.parametrize(
    "event_dir",
    [pytest.param("real", id="physical"), pytest.param("linked", id="logical")],
)
def test_output_event_under_symlinked_parent_uses_output_identity(
    tmp_path: pathlib.Path, event_dir: str
) -> None:
    """Either spelling of an output event retains its producer and skips raw-input aliases."""
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
    event_path = tmp_path / event_dir / "out.txt"

    assert coord.get_producer(event_path) == "source"
    assert (
        coord.should_filter_path(
            event_path,
            get_stage_state=_helper_state_map(StageExecutionState.RUNNING).__getitem__,
        )
        is True
    )
    assert coord.get_affected_stages([event_path]) == ["consume_output"]


@pytest.mark.anyio
async def test_watchfiles_physical_output_event_uses_symlinked_output_identity(
    tmp_path: pathlib.Path,
) -> None:
    """Whatever spelling watchfiles reports for the output, it routes as the output."""
    real_dir = tmp_path / "real"
    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(real_dir, target_is_directory=True)
    real_dir.mkdir()
    output_path = linked_dir / "out.txt"
    physical_output = real_dir / "out.txt"
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
    source = sources.FilesystemSource(watch_paths=[tmp_path], debounce_ms=50)
    send, receive = anyio.create_memory_object_stream[InputEvent](1)

    async with anyio.create_task_group() as tg:
        tg.start_soon(source.run, send)
        await anyio.wait_all_tasks_blocked()
        output_path.write_text("output")
        with anyio.fail_after(10):
            event = await receive.receive()
        assert event["type"] == "data_artifact_changed"
        output_events = [
            pathlib.Path(event_path)
            for event_path in event["paths"]
            if pathlib.Path(event_path).name == "out.txt"
        ]
        assert output_events, f"watchfiles reported no output path: {event['paths']}"
        running = _helper_state_map(StageExecutionState.RUNNING)
        for event_path in output_events:
            assert coord.get_producer(event_path) == "source", event_path
            assert coord.should_filter_path(event_path, get_stage_state=running.__getitem__), (
                event_path
            )
            assert coord.get_affected_stages([event_path]) == ["consume_output"], event_path
        tg.cancel_scope.cancel()


def test_symlink_event_refreshes_retargeted_input_alias(
    tmp_path: pathlib.Path,
) -> None:
    """A symlink event refreshes aliases before an edit arrives through its new target."""
    first_target = tmp_path / "first"
    second_target = tmp_path / "second"
    first_target.mkdir()
    second_target.mkdir()
    first_input = first_target / "a.txt"
    second_input = second_target / "a.txt"
    first_input.write_text("first")
    second_input.write_text("second")
    alias = tmp_path / "alias"
    alias.symlink_to(first_target, target_is_directory=True)
    graph, artifact_paths = _helper_build_graph(
        {"aliasdir": {"deps": [str(alias)], "outs": [str(tmp_path / "output.txt")]}}
    )
    coord = WatchCoordinator(graph, artifact_paths)

    assert coord.get_affected_stages([first_input]) == ["aliasdir"]

    alias.unlink()
    alias.symlink_to(second_target, target_is_directory=True)
    assert coord.get_affected_stages([alias]) == ["aliasdir"]

    assert coord.get_affected_stages([second_input]) == ["aliasdir"]


def _helper_ignore_project(
    root: pathlib.Path, *, linked: bool
) -> tuple[WatchCoordinator, pathlib.Path, pathlib.Path]:
    """A project whose stage ``lane`` reads input directory ``tasks/server`` (local, or a
    symlink to a directory outside the project) and writes directory ``data/p/lane``, read
    by ``report``. Returns the planner, the input's real directory and the output."""
    (root / ".pivotignore").write_text("*.pyc\n__pycache__/\n/tasks/server/scratch/\n")
    (root / "tasks").mkdir()
    source = root.parent / f"{root.name}-elsewhere" if linked else root / "tasks" / "server"
    source.mkdir()
    if linked:
        (root / "tasks" / "server").symlink_to(source, target_is_directory=True)
    output = root / "data" / "p" / "lane"
    output.mkdir(parents=True)
    server = pivot_types.ArtifactIdentity("server", None)
    built = pivot_types.ArtifactIdentity("p/lane", None)
    graph: nx.DiGraph[str] = nx.DiGraph()
    for stage, deps, outs in (("p/lane", [server], [built]), ("p/report", [built], [])):
        graph.add_node(engine_graph.stage_node(stage), type=NodeType.STAGE)
        for dep in deps:
            graph.add_node(engine_graph.artifact_node(dep), type=NodeType.ARTIFACT)
            graph.add_edge(engine_graph.artifact_node(dep), engine_graph.stage_node(stage))
        for out in outs:
            graph.add_node(engine_graph.artifact_node(out), type=NodeType.ARTIFACT)
            graph.add_edge(engine_graph.stage_node(stage), engine_graph.artifact_node(out))
    artifact_paths = {
        project.normalize_path(root / "tasks" / "server"): server,
        project.normalize_path(output): built,
    }
    workspace = store_mod.WorkspaceStore(root, "p", {"server": "tasks/server"})
    return WatchCoordinator(graph, artifact_paths, workspace), source, output


@pytest.mark.parametrize("linked", [False, True], ids=["local", "symlinked-from-outside"])
def test_affected_stages_skip_a_change_pivotignore_names_inside_an_input_directory(
    set_project_root: pathlib.Path, linked: bool
) -> None:
    """The input's hash leaves out what .pivotignore names, so a change there runs nothing;
    a change to any other file of the input runs its consumers."""
    coord, source, _ = _helper_ignore_project(set_project_root, linked=linked)
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "grade.cpython-312.pyc").write_bytes(b"pyc")
    (source / "scratch").mkdir()
    (source / "scratch" / "notes.txt").write_text("notes")
    (source / "grade.py").write_text("grade")
    logical = set_project_root / "tasks" / "server"

    for base in {source, logical}:
        assert coord.get_affected_stages([base / "__pycache__" / "grade.cpython-312.pyc"]) == []
        assert coord.get_affected_stages([base / "__pycache__"]) == []
        assert coord.get_affected_stages([base / "stray.pyc"]) == [], "a deleted ignored file"
        assert coord.get_affected_stages([base / "scratch" / "notes.txt"]) == []
        assert coord.get_affected_stages([base / "grade.py"]) == ["p/lane", "p/report"]


def test_affected_stages_keep_every_change_inside_a_directory_output(
    set_project_root: pathlib.Path,
) -> None:
    """A dependency on an upstream directory output is its whole hash, so .pivotignore leaves
    no change inside it out."""
    coord, _, output = _helper_ignore_project(set_project_root, linked=False)
    (output / "x.pyc").write_bytes(b"pyc")

    assert coord.get_affected_stages([output / "x.pyc"]) == ["p/report"]


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
