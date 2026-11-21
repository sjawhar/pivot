"""Bipartite artifact-stage graph built on NetworkX."""

from __future__ import annotations

import collections
import pathlib
from typing import TYPE_CHECKING, NamedTuple, TypedDict

import networkx as nx

from pivot import project
from pivot import types as pivot_types
from pivot.engine.types import NodeType
from pivot.storage import path_forms

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping
    from collections.abc import Set as AbstractSet

    import pygtrie

    from pivot.exceptions import DependencyNotFoundError
    from pivot.registry import RegistryStageInfo
    from pivot.storage.store import Store

__all__ = [
    "artifact_node",
    "stage_node",
    "build_graph",
    "validate_dependency_sources",
    "skips_missing_dependencies",
    "is_leave_out_mode",
    "RunnableStages",
    "get_runnable_stages",
    "tracked_overlap",
    "get_consumers",
    "get_producer",
    "get_downstream_stages",
    "get_upstream_stages",
    "get_stage_dag",
    "get_width",
    "update_stage",
    "get_artifact_consumers",
    "get_execution_order",
    "GraphView",
    "extract_graph_view",
]


class GraphView(TypedDict):
    """Pre-extracted graph data for rendering.

    Decouples renderers from the internal bipartite graph representation.
    All node identifiers are plain strings (stage names, artifact identities)
    with no encoding prefixes.

    Edge direction is data-flow: producer -> consumer / input -> output.
    """

    stages: list[str]
    artifacts: list[str]
    stage_edges: list[tuple[str, str]]
    artifact_edges: list[tuple[str, str]]


def _coerce_identity(
    identity: pivot_types.ArtifactIdentity | pathlib.Path | str,
) -> pivot_types.ArtifactIdentity:
    if isinstance(identity, pivot_types.ArtifactIdentity):
        return identity
    if isinstance(identity, pathlib.Path):
        return _path_identity(str(identity))
    return _path_identity(identity)


def artifact_node(identity: pivot_types.ArtifactIdentity | pathlib.Path | str) -> str:
    """Create artifact node ID from identity."""
    resolved = _coerce_identity(identity)
    return f"artifact:{pivot_types.identity_key(resolved)}"


def stage_node(name: str) -> str:
    """Create stage node ID from name."""
    return f"stage:{name}"


def parse_node(node: str) -> tuple[NodeType, str]:
    """Extract NodeType and value from node ID.

    Splits on the first colon to preserve identity strings with colons.
    """
    prefix, value = node.split(":", 1)
    return NodeType(prefix), value


def _path_identity(path: str) -> pivot_types.ArtifactIdentity:
    return pivot_types.ArtifactIdentity(path, None)


def _output_identities(
    out: pivot_types.ArtifactRef,
) -> list[pivot_types.ArtifactIdentity]:
    return [out.identity]


def _dep_identity(dep: pivot_types.ArtifactRef) -> pivot_types.ArtifactIdentity:
    """The artifact node a dependency hangs from: one file of a directory is its directory."""
    return pivot_types.source_identity(dep)


def _build_outputs_map(
    stages: dict[str, RegistryStageInfo],
) -> dict[pivot_types.ArtifactIdentity, str]:
    """Build mapping from output identity to stage name."""
    outputs_map = dict[pivot_types.ArtifactIdentity, str]()
    for stage_name, stage_info in stages.items():
        for out in stage_info["outs"]:
            for identity in _output_identities(out):
                outputs_map[identity] = stage_name
    return outputs_map


def _check_acyclic(g: nx.DiGraph[str]) -> None:
    """Check graph for cycles, raise if found."""
    from pivot import exceptions

    try:
        cycle = nx.find_cycle(g, orientation="original")
    except nx.NetworkXNoCycle:
        return

    # Extract stage names from cycle for error message
    # Cycle is a list of (from_node, to_node, direction) tuples
    stages_in_cycle = list[str]()
    for from_node, _, _ in cycle:
        node_type, name = parse_node(from_node)
        if node_type == NodeType.STAGE and name not in stages_in_cycle:
            stages_in_cycle.append(name)

    if not stages_in_cycle:
        # Fallback if cycle is artifact-only (shouldn't happen in valid graph)
        # Extract readable names from the cycle edges
        nodes_in_cycle = list[str]()
        for from_node, _, _ in cycle:
            _, from_name = parse_node(from_node)
            if from_name not in nodes_in_cycle:
                nodes_in_cycle.append(from_name)
        stages_in_cycle = nodes_in_cycle if nodes_in_cycle else ["<unknown>"]

    raise exceptions.CyclicGraphError(
        f"Circular dependency detected: {' -> '.join(stages_in_cycle)}"
    )


def validate_dependency_sources(
    stages: dict[str, RegistryStageInfo],
    store: Store | None = None,
    *,
    present: list[tuple[str, pivot_types.ArtifactRef]] | None = None,
    created: Callable[[pathlib.Path], bool] | None = None,
) -> list[DependencyNotFoundError]:
    """Check that every dependency is either produced by a stage or exists on disk.

    Collects ALL errors instead of failing on the first.

    Returns a list of :class:`~pivot.exceptions.DependencyNotFoundError` for
    unresolvable deps.  With *store=None*, only structural checks run (wrong
    output key on a known stage); external-dep existence is silently skipped.
    With *present*, each dependency that no stage produces and that exists on disk is
    appended to it with its stage, in iteration order (``get_runnable_stages`` reads these for
    the missing tracked files each is, lies inside or holds). With *created*, a dependency
    missing from disk whose path it accepts counts as one that exists: a restore creates that
    path (``get_runnable_stages``' ``restored_tracked``, a dry run's view of
    ``--checkout-missing``).
    """
    from pivot import exceptions

    outputs_map = _build_outputs_map(stages)
    stage_names = set(stages.keys())
    errors = list[exceptions.DependencyNotFoundError]()

    for stage_name, info in stages.items():
        for dep in info["deps"].values():
            dep_identity = _dep_identity(dep)
            if dep_identity in outputs_map:
                continue
            if dep_identity.producer in stage_names:
                errors.append(
                    exceptions.DependencyNotFoundError(
                        stage=stage_name,
                        dep=pivot_types.identity_key(dep_identity),
                        available_outputs=[pivot_types.identity_key(out) for out in outputs_map],
                    )
                )
                continue
            if store is None:
                continue
            if store.exists(dep):
                if present is not None:
                    present.append((stage_name, dep))
                continue
            try:
                input_path = store.checkout(dep)
            except FileNotFoundError:
                input_path = None
                input_path_display = None
            else:
                if created is not None and created(input_path):
                    if present is not None:
                        present.append((stage_name, dep))
                    continue
                input_path_display = project.to_relative_path(input_path)
            errors.append(
                exceptions.DependencyNotFoundError(
                    stage=stage_name,
                    dep=pivot_types.identity_key(dep_identity),
                    available_outputs=[pivot_types.identity_key(out) for out in outputs_map],
                    input_path=input_path,
                    input_path_display=input_path_display,
                )
            )

    return errors


def build_graph(
    stages: dict[str, RegistryStageInfo],
) -> nx.DiGraph[str]:
    """Build bipartite artifact-stage graph from stage definitions.

    Args:
        stages: Dict mapping stage name to RegistryStageInfo.
    Returns:
        Directed graph where:
        - Nodes are either artifacts (identities) or stages (functions)
        - Edges go: artifact -> stage (consumed by) and stage -> artifact (produces)

    Raises:
        CyclicGraphError: If graph contains cycles (always checked)
    """
    g: nx.DiGraph[str] = nx.DiGraph()
    add_stages(g, stages)

    # Always check for cycles - a cyclic graph is never valid
    _check_acyclic(g)

    return g


def add_stages(g: nx.DiGraph[str], stages: dict[str, RegistryStageInfo]) -> None:
    """Add each stage of ``stages`` to bipartite graph ``g``, with its dependency and output
    edges, as ``build_graph`` does. A run adds a template's instances this way (``Pipeline.map``):
    nothing in the graph depends on them, so they close no cycle."""
    for stage_name, info in stages.items():
        stage = stage_node(stage_name)
        g.add_node(stage, type=NodeType.STAGE)

        # Deps: artifact -> stage
        for dep in info["deps"].values():
            dep_identity = _dep_identity(dep)
            artifact = artifact_node(dep_identity)
            g.add_node(artifact, type=NodeType.ARTIFACT)
            g.add_edge(artifact, stage)

        # Outs: stage -> artifact
        for out in info["outs"]:
            for identity in _output_identities(out):
                artifact = artifact_node(identity)
                g.add_node(artifact, type=NodeType.ARTIFACT)
                g.add_edge(stage, artifact)


def get_consumers(g: nx.DiGraph[str], identity: pivot_types.ArtifactIdentity) -> list[str]:
    """Get stages that depend on this artifact.

    Args:
        g: The bipartite graph.
        identity: Artifact identity.

    Returns:
        List of stage names that consume this artifact.
    """
    node = artifact_node(identity)
    if node not in g:
        return []
    return [parse_node(n)[1] for n in g.successors(node) if g.nodes[n]["type"] == NodeType.STAGE]


def get_producer(g: nx.DiGraph[str], identity: pivot_types.ArtifactIdentity) -> str | None:
    """Get the stage that produces this artifact.

    Args:
        g: The bipartite graph.
        identity: Artifact identity.

    Returns:
        Stage name that produces this artifact, or None if it's an input.
    """
    node = artifact_node(identity)
    if node not in g:
        return None
    for pred in g.predecessors(node):
        if g.nodes[pred]["type"] == NodeType.STAGE:
            return parse_node(pred)[1]
    return None


def get_downstream_stages(g: nx.DiGraph[str], stage_name: str) -> list[str]:
    """Get all stages transitively downstream of this one.

    Args:
        g: The bipartite graph.
        stage_name: Name of the stage.

    Returns:
        List of stage names that transitively depend on this stage's outputs.
    """
    node = stage_node(stage_name)
    if node not in g:
        return []

    downstream = list[str]()
    for descendant in nx.descendants(g, node):
        if g.nodes[descendant]["type"] == NodeType.STAGE:
            downstream.append(parse_node(descendant)[1])
    return downstream


def skips_missing_dependencies(*, all_pipelines: bool, on_error: pivot_types.OnError) -> bool:
    """Whether a run leaves out stages with a missing dependency instead of stopping: one
    across all pipelines that keeps going past failures."""
    return all_pipelines and on_error == pivot_types.OnError.KEEP_GOING


def is_leave_out_mode(
    *, all_pipelines: bool, on_error: pivot_types.OnError, watching: bool
) -> bool:
    """Whether a run is in leave-out mode: one that skips missing dependencies
    (``skips_missing_dependencies``) outside a watch or serve session (``watching``: the engine
    has a filesystem source). Such a run also leaves out each stage that reads a tracked file
    still missing after the restore, instead of stopping on it, and each target whose stage it
    leaves out (``expansion.keep_targets``). The run (``Engine._orchestrate_execution``) and its
    dry run and ``--explain`` (``pivot repro``) both decide the mode here."""
    return (
        skips_missing_dependencies(all_pipelines=all_pipelines, on_error=on_error) and not watching
    )


class RunnableStages(NamedTuple):
    """The stages a run executes, with the missing dependencies found
    (``get_runnable_stages``)."""

    stages: dict[str, RegistryStageInfo] | None
    """The stages the run executes; None when the run stops at a missing dependency."""
    errors: list[DependencyNotFoundError]
    """Each missing dependency found."""
    left_out: dict[str, DependencyNotFoundError]
    """Each stage a run that skips missing dependencies leaves out, with its cause: the first
    error of ``errors`` whose stage is that stage or upstream of it. Empty in any other run."""
    present: list[tuple[str, pivot_types.ArtifactRef]]
    """Each dependency that no stage produces and that exists, or that the restore creates, with
    its stage (``validate_dependency_sources``' ``present``), of every stage, left out or not:
    for a caller to read them without checking the disk again (a dry run's notice,
    ``status.restore_limits``)."""


def get_runnable_stages(
    stages: dict[str, RegistryStageInfo],
    store: Store,
    *,
    all_pipelines: bool,
    on_error: pivot_types.OnError,
    missing_tracked: Collection[str] = (),
    restored_tracked: Mapping[str, AbstractSet[tuple[str, ...]]] | None = None,
) -> RunnableStages:
    """Return the stages a run executes, with the missing dependencies found.

    A run across all pipelines that keeps going past failures leaves out each stage with a
    missing dependency and every stage downstream of it. Any other run stops at a missing
    dependency, and the stages are None.

    ``missing_tracked`` holds the tracked files still missing, as the absolute normalized paths
    discovery gives (``executor.core.verify_tracked_files``, by ``storage.track.present_kind``):
    the one source of which tracked paths are missing, whatever the store's existence test says
    of them (a socket or a FIFO at a tracked path exists, but the path is missing). A stage reads
    such a file when a dependency is the file, lies inside it, or is an existing directory that
    holds it, and it gets one error per such file. ``restored_tracked`` maps each missing tracked
    path that a restore brings back to every path the restore creates inside it, as its parts
    relative to that path (``executor.core.find_missing_tracked_files``, for a dry run with
    ``--checkout-missing``): a missing dependency the restore creates, or that holds a path the
    restore creates, counts as one that exists, as the run finds it after its restore
    (``_restore_creates``), so a directory among them that holds a missing tracked file leaves
    its stage out too. ``RunnableStages.present`` holds the dependencies found on disk or
    created by the restore.
    """
    from pivot import exceptions

    # The dependencies on disk, or created by the restore, read for the missing tracked files
    # each is, lies inside or holds
    present = list[tuple[str, pivot_types.ArtifactRef]]()
    errors = validate_dependency_sources(
        stages,
        store=store,
        present=present,
        created=_restore_creates(restored_tracked) if restored_tracked else None,
    )
    if missing_tracked:
        errors = _tracked_file_errors(errors, present, store, missing_tracked)
    if not errors:
        return RunnableStages(stages, errors, {}, present)
    if not skips_missing_dependencies(all_pipelines=all_pipelines, on_error=on_error):
        return RunnableStages(None, errors, {}, present)
    graph = build_graph(stages)
    left_out = dict[str, exceptions.DependencyNotFoundError]()
    for error in errors:
        if error.stage in left_out:
            # An earlier error left it out, and every stage downstream of it with it
            continue
        for name in (error.stage, *get_downstream_stages(graph, error.stage)):
            # A stage's cause is the first error that leaves it out
            left_out.setdefault(name, error)
    return RunnableStages(
        {name: info for name, info in stages.items() if name not in left_out},
        errors,
        left_out,
        present,
    )


def _tracked_file_errors(
    errors: list[DependencyNotFoundError],
    present: list[tuple[str, pivot_types.ArtifactRef]],
    store: Store,
    missing_tracked: Collection[str],
) -> list[DependencyNotFoundError]:
    """``validate_dependency_sources``' ``errors`` with the missing tracked files counted, so a
    stage has one error per missing tracked file it reads.

    ``validate_dependency_sources`` reports each dependency that is a missing tracked file or
    lies inside one and does not exist: of a stage's errors for one tracked file, the first
    stays. A dependency of ``present`` (one no stage produces, found on disk or created by a
    restore, with its stage) that is a missing tracked file or lies inside one (an entry is
    there, but neither a regular file nor a directory, so ``storage.track.present_kind`` counts
    the file missing: a socket, a FIFO) gets the message ``validate_dependency_sources`` gives a
    missing one, without an output suggestion (the tracked file's data is the fix), unless the
    stage has one for that file; and one that is a directory holding missing tracked files gets
    one error per such file the stage has no error for yet. Each path counts in both its forms
    (``storage.path_forms``), as a watch run waits for a missing input (``Engine``'s
    ``_MissingDependencyRun.awaits``): a dependency that reaches a missing tracked file through a
    symlink reads it. A dependency is or lies inside the outermost missing tracked file that is
    or holds its normalized form, else its resolved form (``path_forms.enclosing``), and holds
    each one that lies inside either form (``path_forms.held``)."""
    from pivot import exceptions

    missing = path_forms.forms_trie(missing_tracked)
    # The stage and missing tracked file of each error kept
    reported = set[tuple[str, str]]()
    kept = list[exceptions.DependencyNotFoundError]()
    for error in errors:
        if error.input_path is not None:
            input_forms = path_forms.forms(project.normalize_path(error.input_path))
            tracked = path_forms.enclosing(missing, *input_forms)
            if tracked is not None:
                if (error.stage, tracked) in reported:
                    continue
                reported.add((error.stage, tracked))
        kept.append(error)

    # The missing tracked file each present dependency is or lies inside, and those it holds,
    # sorted, by its normalized path
    found_by_path = dict[pathlib.Path, tuple[str | None, list[str]]]()
    for stage_name, dep in present:
        input_path = store.checkout(dep)
        path = project.normalize_path(input_path)
        if path not in found_by_path:
            forms = path_forms.forms(path)
            found_by_path[path] = (
                path_forms.enclosing(missing, *forms),
                path_forms.held(missing, *forms),
            )
        enclosing, held = found_by_path[path]
        if enclosing is not None and (stage_name, enclosing) not in reported:
            reported.add((stage_name, enclosing))
            kept.append(
                exceptions.DependencyNotFoundError(
                    stage=stage_name,
                    dep=pivot_types.identity_key(_dep_identity(dep)),
                    # The tracked file's data is the fix, not an output name it resembles
                    available_outputs=None,
                    input_path=input_path,
                    input_path_display=project.to_relative_path(input_path),
                )
            )
        for tracked in held:
            if (stage_name, tracked) in reported:
                continue
            reported.add((stage_name, tracked))
            kept.append(
                exceptions.DependencyNotFoundError(
                    stage=stage_name,
                    dep=pivot_types.identity_key(_dep_identity(dep)),
                    # The dependency exists, so no output name it resembles is the fix
                    available_outputs=None,
                    input_path=pathlib.Path(tracked),
                    input_path_display=project.to_relative_path(tracked),
                    directory=path,
                    directory_display=project.to_relative_path(path),
                )
            )
    return kept


def _restore_creates(
    restored: Mapping[str, AbstractSet[tuple[str, ...]]],
) -> Callable[[pathlib.Path], bool]:
    """Whether the restore of the missing tracked paths of ``restored`` creates a path, or a
    path inside it. ``restored`` maps each such path to every path its restore creates inside
    it, as its parts relative to that path (``executor.core.MissingTrackedFiles.restored``).

    A path counts in both its forms (``storage.path_forms.forms``), and so does each restored
    path (``path_forms.forms_trie``): nothing under a missing path exists, so a path inside it
    resolves to the same parts inside its resolved form. The trie of the restored paths is built,
    resolving each of them once, when the first path is asked about: a dry run with no missing
    dependency resolves none. Path parts decide containment, never a string prefix.
    """
    roots: pygtrie.Trie[str] | None = None

    def creates(path: pathlib.Path) -> bool:
        nonlocal roots
        if roots is None:
            roots = path_forms.forms_trie(restored)
        for form in path_forms.forms(project.normalize_path(path)):
            parts = form.parts
            # A restored path, or a directory holding one
            if parts in roots or roots.has_subtrie(parts):
                return True
            # A path inside a restored path whose restore creates it
            for depth in range(1, len(parts)):
                root = parts[:depth]
                if root in roots and parts[depth:] in restored[roots[root]]:
                    return True
        return False

    return creates


def tracked_overlap(tracked: Collection[str]) -> Callable[[pathlib.Path], str | None]:
    """For the absolute normalized tracked paths ``tracked``, the function that gives the one an
    absolute normalized path is or lies inside, else the first, sorted, of those it holds, else
    None: ``storage.path_forms.overlapping`` of both forms of the path, each tracked path keyed
    by both of its own (``path_forms.forms_trie``), as ``get_runnable_stages`` counts a missing
    tracked file (``_tracked_file_errors``). So "is or lies inside" by either form comes before
    "holds" by either: the tracked path that is or holds the path, by either form, is the one
    whose restore brings that path back. The trie of ``tracked`` is built here, once."""
    trie = path_forms.forms_trie(tracked)

    def overlap(path: pathlib.Path) -> str | None:
        return path_forms.overlapping(trie, *path_forms.forms(path))

    return overlap


def update_stage(g: nx.DiGraph[str], stage_name: str, new_info: RegistryStageInfo) -> None:
    """Incrementally update graph when a stage's definition changes.

    Efficiently diffs current and new deps/outs, adding and removing edges
    as needed. Removes orphaned artifact nodes (no longer connected to any stage).

    Args:
        g: The bipartite graph to modify in place.
        stage_name: Name of the stage to update.
        new_info: New stage definition from registry.
    """
    stage = stage_node(stage_name)

    # Get current deps and outs from graph
    current_deps = {
        pivot_types.identity_from_key(parse_node(n)[1])
        for n in g.predecessors(stage)
        if g.nodes[n]["type"] == NodeType.ARTIFACT
    }
    current_outs = {
        pivot_types.identity_from_key(parse_node(n)[1])
        for n in g.successors(stage)
        if g.nodes[n]["type"] == NodeType.ARTIFACT
    }

    # Get new deps and outs from info
    new_deps = {_dep_identity(ref) for ref in new_info["deps"].values()}
    new_outs = {ref.identity for ref in new_info["outs"]}

    # Remove old deps
    for removed_dep in current_deps - new_deps:
        artifact = artifact_node(removed_dep)
        g.remove_edge(artifact, stage)
        if g.degree(artifact) == 0:
            g.remove_node(artifact)

    # Add new deps
    for added_dep in new_deps - current_deps:
        artifact = artifact_node(added_dep)
        if artifact not in g:
            g.add_node(artifact, type=NodeType.ARTIFACT)
        g.add_edge(artifact, stage)

    # Remove old outs
    for removed_out in current_outs - new_outs:
        artifact = artifact_node(removed_out)
        g.remove_edge(stage, artifact)
        if g.degree(artifact) == 0:
            g.remove_node(artifact)

    # Add new outs
    for added_out in new_outs - current_outs:
        artifact = artifact_node(added_out)
        if artifact not in g:
            g.add_node(artifact, type=NodeType.ARTIFACT)
        g.add_edge(stage, artifact)


def get_upstream_stages(g: nx.DiGraph[str], stage_name: str) -> list[str]:
    """Get stages whose outputs are consumed by this stage."""
    node = stage_node(stage_name)
    if node not in g:
        return []

    upstream = list[str]()
    for artifact in g.predecessors(node):
        if g.nodes[artifact]["type"] != NodeType.ARTIFACT:
            continue
        for producer in g.predecessors(artifact):
            if g.nodes[producer]["type"] == NodeType.STAGE:
                upstream.append(parse_node(producer)[1])
    return upstream


def get_stage_dag(g: nx.DiGraph[str]) -> nx.DiGraph[str]:
    """Extract stage-only DAG from bipartite graph.

    Returns a DAG with edges from consumer to producer. This allows
    get_execution_order() to work correctly with dfs_postorder_nodes traversal.
    """
    stage_dag: nx.DiGraph[str] = nx.DiGraph()

    for node in g.nodes():
        if g.nodes[node]["type"] == NodeType.STAGE:
            stage_name = parse_node(node)[1]
            stage_dag.add_node(stage_name)

    for node in g.nodes():
        if g.nodes[node]["type"] != NodeType.STAGE:
            continue
        stage_name = parse_node(node)[1]

        for artifact in g.successors(node):
            if g.nodes[artifact]["type"] != NodeType.ARTIFACT:
                continue
            for consumer in g.successors(artifact):
                if g.nodes[consumer]["type"] == NodeType.STAGE:
                    consumer_name = parse_node(consumer)[1]
                    # Edge from consumer to producer (for DFS postorder execution)
                    stage_dag.add_edge(consumer_name, stage_name)

    return stage_dag


def get_width(stage_dag: nx.DiGraph[str], stages: list[str], *, limit: int) -> int:
    """Return the most of ``stages`` that can run at once, or ``limit`` if that is fewer.

    The scheduler starts a stage once the stages of the run it consumes from have finished,
    so stages can run together exactly when no chain of run stages connects any two of them.
    The largest such set is the stage count minus a maximum matching that pairs each stage
    with one stage it transitively depends on (Dilworth's theorem, by Konig's).

    The widest topological level is such a set, so once it holds ``limit`` stages the matching
    is skipped. Below ``limit`` the result is exact, unless the search for the matching runs
    out of steps; then it is an upper bound on the width, never fewer than the stages that can
    run at once. The stages are numbered from their names and dependencies alone, so the result
    does not depend on the order of ``stages``.

    Args:
        stage_dag: Stage-only DAG (from get_stage_dag)
        stages: The stages of the run; dependencies through stages outside it do not count
        limit: The most workers the pool may have
    """
    names = sorted(set(stages))
    count = len(names)
    index = dict(zip(names, range(count), strict=True))
    upstream = [[index[up] for up in stage_dag.successors(name) if up in index] for name in names]
    downstream = [list[int]() for _ in names]
    for i, producers in enumerate(upstream):
        for j in producers:
            downstream[j].append(i)
    # Producers first, in an order fixed by the names and dependencies alone; a stage's level is
    # one more than its deepest dependency's.
    waiting = [len(producers) for producers in upstream]
    topological = [i for i, pending in enumerate(waiting) if pending == 0]
    level = [0] * count
    for i in topological:
        deeper = level[i] + 1
        for k in downstream[i]:
            if level[k] < deeper:
                level[k] = deeper
            waiting[k] -= 1
            if not waiting[k]:
                topological.append(k)
    # Stages on one level, counted from the producers or from the last consumers, never reach
    # each other, so the widest level is a lower bound on the width.
    remaining = [0] * count
    for i in reversed(topological):
        deeper = remaining[i] + 1
        for j in upstream[i]:
            if remaining[j] < deeper:
                remaining[j] = deeper
    widest = max(
        max(collections.Counter(level).values(), default=0),
        max(collections.Counter(remaining).values(), default=0),
    )
    if widest >= limit:
        return limit

    # Renumber consumers first, so stage i depends only on stages numbered above i. Each
    # stage lists its dependencies deepest first, the ones a chain of stages most likely runs
    # through, ties by number.
    consumers_first = topological[::-1]
    number = [0] * count
    for position, i in enumerate(consumers_first):
        number[i] = position
    direct = [list[int]() for _ in names]
    # A stable sort keeps equal levels in consumers-first order, that is, by number.
    for j in sorted(consumers_first, key=level.__getitem__, reverse=True):
        numbered = number[j]
        for i in downstream[j]:
            direct[number[i]].append(numbered)
    order = range(count - 1, -1, -1)  # producers first

    # The stages a matching leaves unmatched are an upper bound on the width.
    owner = [-1] * count  # the stage each dependency is matched to
    unmatched = list[int]()
    for i in reversed(order):
        for j in direct[i]:
            if owner[j] == -1:
                owner[j] = i
                break
        else:
            unmatched.append(i)
    if len(unmatched) == widest:
        return widest
    matched_bits = bytearray((count + 7) // 8)
    for j, stage in enumerate(owner):
        if stage != -1:
            matched_bits[j >> 3] |= 1 << (j & 7)
    taken = int.from_bytes(matched_bits, "little")  # bitset of matched dependencies

    # Bitset of the run stages each stage transitively depends on, built from each
    # dependency's own set with the dependency added.
    reach = [0] * len(names)
    closed = [0] * len(names)
    for i in order:
        bits = 0
        for j in direct[i]:
            bits |= closed[j]
        reach[i] = bits
        closed[i] = bits | (1 << i)
    still_unmatched = list[int]()
    for i in unmatched:
        free_bits = reach[i] & ~taken
        if free_bits:
            free = (free_bits & -free_bits).bit_length() - 1
            owner[free] = i
            taken |= 1 << free
        else:
            still_unmatched.append(i)

    # Augment in rounds. Within a round no search explores a dependency an earlier search
    # explored. That can hide an augmenting path once a search succeeds, so rounds repeat until
    # one matches no stage: the matching did not change during that round, so no stage it left
    # unmatched has an augmenting path, and the matching is maximum. The searches share a
    # budget of steps; once it runs out, the stages still unmatched bound the width from above,
    # so the pool may be larger than the width but never smaller.
    budget = _AUGMENT_STEPS
    unmatched_count = len(still_unmatched)
    while unmatched_count > widest:
        seen = 0
        left = list[int]()
        for start in still_unmatched:
            matched, seen, budget = _augment(start, reach, owner, seen, budget)
            if matched:
                unmatched_count -= 1
                if unmatched_count == widest:
                    return widest
            elif budget == 0:
                return min(unmatched_count, limit)
            else:
                left.append(start)
        if len(left) == len(still_unmatched):
            break
        still_unmatched = left
    return min(unmatched_count, limit)


# Steps of augmenting-path search `get_width` takes before it settles for an upper bound. A
# count, not a clock, so one graph and limit always give one size.
_AUGMENT_STEPS = 10_000
# Steps per stage, dependency and requested resource that `Scheduler.resource_width`'s replay
# takes beyond `_AUGMENT_STEPS` before it settles for its limit, `get_width`'s pool size.
_REPLAY_STEPS_PER_INPUT = 8


def _augment(
    start: int, reach: list[int], owner: list[int], seen: int, budget: int
) -> tuple[bool, int, int]:
    """Match ``start`` along an augmenting path, if one exists, without recursion.

    ``seen`` holds dependencies known to lead to no augmenting path. Returns whether ``start``
    was matched, ``seen`` with the dependencies this search explored, and the steps left of
    ``budget``; a search that runs out of steps returns unmatched with none left.
    """
    path = [start]  # stages on the path; path[k + 1] owns dependencies[k]
    dependencies = list[int]()
    while path:
        if budget == 0:
            return False, seen, 0
        budget -= 1
        candidates = reach[path[-1]] & ~seen
        if not candidates:
            path.pop()
            if dependencies:
                dependencies.pop()
            continue
        lowest = candidates & -candidates
        seen |= lowest
        dependency = lowest.bit_length() - 1
        dependencies.append(dependency)
        if owner[dependency] == -1:
            for stage, matched in zip(path, dependencies, strict=True):
                owner[matched] = stage
            return True, seen, budget
        path.append(owner[dependency])
    return False, seen, budget


def extract_graph_view(g: nx.DiGraph[str]) -> GraphView:
    """Extract a renderer-friendly view from the bipartite graph.

    Walks the bipartite graph, collecting stage names, artifact identities,
    and derived edges without exposing the internal node encoding.

    Edge semantics (data-flow direction):
    - stage_edges: (producer_stage, consumer_stage)
    - artifact_edges: (input_artifact, output_artifact)

    Args:
        g: Bipartite artifact-stage graph from build_graph().

    Returns:
        GraphView with plain-string nodes and edges.
    """
    stages = list[str]()
    artifacts = list[str]()
    stage_edges_set = set[tuple[str, str]]()
    artifact_edges_set = set[tuple[str, str]]()

    # Collect nodes by type
    for node in g.nodes():
        node_type, value = parse_node(node)
        if node_type == NodeType.STAGE:
            stages.append(value)
        else:
            artifacts.append(value)

    # Derive stage-to-stage edges (producer -> consumer)
    # Walk: stage -> artifact (produces) -> stage (consumes)
    # Use set to deduplicate edges
    for node in g.nodes():
        if g.nodes[node]["type"] != NodeType.STAGE:
            continue
        _, producer_name = parse_node(node)
        for art_succ in g.successors(node):
            if g.nodes[art_succ]["type"] != NodeType.ARTIFACT:
                continue
            for consumer_node in g.successors(art_succ):
                if g.nodes[consumer_node]["type"] != NodeType.STAGE:
                    continue
                _, consumer_name = parse_node(consumer_node)
                stage_edges_set.add((producer_name, consumer_name))

    # Derive artifact-to-artifact edges (input -> output)
    # Walk: artifact -> stage (consumes) -> artifact (produces)
    # Use set to deduplicate edges (multiple stages can create same artifact flow)
    for node in g.nodes():
        if g.nodes[node]["type"] != NodeType.ARTIFACT:
            continue
        _, input_path = parse_node(node)
        for stage_succ in g.successors(node):
            if g.nodes[stage_succ]["type"] != NodeType.STAGE:
                continue
            for output_node in g.successors(stage_succ):
                if g.nodes[output_node]["type"] != NodeType.ARTIFACT:
                    continue
                _, output_path = parse_node(output_node)
                artifact_edges_set.add((input_path, output_path))

    return GraphView(
        stages=sorted(stages),
        artifacts=sorted(artifacts),
        stage_edges=sorted(stage_edges_set),
        artifact_edges=sorted(artifact_edges_set),
    )


def get_artifact_consumers(
    g: nx.DiGraph[str],
    identity: pivot_types.ArtifactIdentity,
    include_downstream: bool = True,
) -> list[str]:
    """Get all stages affected by a change to this artifact.

    Args:
        g: The bipartite graph.
        identity: Artifact identity.
        include_downstream: If True, include transitive dependents.

    Returns:
        Sorted list of stage names that would be affected (deterministic order).
    """
    direct = get_consumers(g, identity)
    if not direct:
        return []

    if not include_downstream:
        return sorted(direct)

    all_affected = set(direct)
    for stage in direct:
        downstream = get_downstream_stages(g, stage)
        all_affected.update(downstream)

    return sorted(all_affected)


def get_execution_order(
    graph: nx.DiGraph[str],
    stages: list[str] | None = None,
    single_stage: bool = False,
) -> list[str]:
    """Get execution order using DFS postorder traversal.

    Args:
        graph: Stage-only DAG (from get_stage_dag)
        stages: Optional target stages to execute (default: all stages)
        single_stage: If True, run only the specified stages without dependencies.
            Stages are executed in the order provided, not DAG order.

    Returns:
        List of stage names in execution order (dependencies first, unless single_stage)
    """
    if stages:
        if single_stage:
            return stages
        subgraph = _get_subgraph(graph, stages)
        return list(nx.dfs_postorder_nodes(subgraph))

    return list(nx.dfs_postorder_nodes(graph))


def _get_subgraph(graph: nx.DiGraph[str], source_stages: list[str]) -> nx.DiGraph[str]:
    """Get subgraph containing sources and all their dependencies.

    Raises:
        StageNotFoundError: If any source stage is not in the graph.
    """
    from pivot import exceptions

    # Validate all stages exist before traversing
    graph_nodes = set(graph.nodes())
    unknown = [s for s in source_stages if s not in graph_nodes]
    if unknown:
        raise exceptions.StageNotFoundError(unknown, available_stages=list(graph_nodes))

    nodes = set[str]()
    for stage in source_stages:
        nodes.update(nx.dfs_postorder_nodes(graph, stage))
    return graph.subgraph(nodes)
