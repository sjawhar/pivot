"""Property-based tests for lock files, artifact identities, the stage DAG and input hashing."""

from __future__ import annotations

import pathlib
import string
import tempfile
from unittest import mock

import hypothesis
import networkx as nx
import pytest
import yaml
from hypothesis import strategies as st

from pivot import compose, exceptions, loaders, run_history
from pivot.engine import graph as engine_graph
from pivot.engine.scheduler import Scheduler
from pivot.registry import RegistryStageInfo
from pivot.storage import listing, lock
from pivot.types import (
    ArtifactIdentity,
    ArtifactRef,
    ArtifactTag,
    DepEntry,
    DirManifestEntry,
    FileHash,
    HashInfo,
    LockData,
    identity_from_key,
    identity_key,
)

_SETTINGS = hypothesis.settings(max_examples=100, deadline=None, derandomize=True, database=None)

_hashes = st.text(alphabet="0123456789abcdef", min_size=16, max_size=16)
_segment = st.text(alphabet=string.ascii_letters + string.digits + "_-@.", min_size=1).filter(
    lambda s: s not in {".", ".."}
)
_producers = st.lists(_segment, min_size=1, max_size=3).map("/".join)
_keys = st.none() | _segment
_identities = st.builds(ArtifactIdentity, _producers, _keys)


# =============================================================================
# Lock files: schema v3 round trip and malformed input
# =============================================================================

_manifest_entries = st.builds(
    DirManifestEntry,
    relpath=_segment,
    hash=_hashes,
    size=st.integers(min_value=0, max_value=2**40),
    isexec=st.booleans(),
)
_directory_listings: st.SearchStrategy[listing.DirListing] = st.builds(
    listing.DirListing,
    files=st.lists(_manifest_entries, max_size=3, unique_by=lambda entry: entry["relpath"]),
    empty_dirs=st.just([]),
)
_param_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(),
    lambda children: st.lists(children, max_size=3)
    | st.dictionaries(st.text(), children, max_size=3),
    max_leaves=8,
)


def _hashed_directory(members: listing.DirListing) -> listing.HashedDirectory:
    data = listing.encode(members)
    return listing.HashedDirectory(
        record=listing.record_of(members, data),
        listing=members,
        data=data,
    )


@st.composite
def _hash_info_and_listing(
    draw: st.DrawFn,
) -> tuple[HashInfo, listing.HashedDirectory | None]:
    if draw(st.booleans()):
        return FileHash(hash=draw(_hashes)), None
    hashed = _hashed_directory(draw(_directory_listings))
    return hashed["record"], hashed


@st.composite
def _lock_data(draw: st.DrawFn) -> tuple[str, LockData, dict[str, listing.HashedDirectory]]:
    stage_name = draw(_producers)
    hashed_dirs = dict[str, listing.HashedDirectory]()

    def take_hash() -> HashInfo:
        hash_info, hashed = draw(_hash_info_and_listing())
        if hashed is not None:
            hashed_dirs[hashed["record"]["hash"]] = hashed
        return hash_info

    out_keys = draw(st.lists(_keys, max_size=3, unique=True))
    data = LockData(
        code_manifest=draw(st.dictionaries(st.text(min_size=1), _hashes, max_size=4)),
        params=draw(st.dictionaries(st.text(min_size=1), _param_values, max_size=4)),
        dep_hashes={
            identity: take_hash()
            for identity in draw(st.lists(_identities, max_size=4, unique=True))
        },
        output_hashes={ArtifactIdentity(stage_name, key): take_hash() for key in out_keys},
    )
    return stage_name, data, hashed_dirs


@_SETTINGS
@hypothesis.given(_lock_data())
def test_lock_write_then_read_returns_the_written_data(
    case: tuple[str, LockData, dict[str, listing.HashedDirectory]],
) -> None:
    stage_name, data, hashed_dirs = case
    with tempfile.TemporaryDirectory() as tmp:
        stage_lock = lock.StageLock(stage_name, pathlib.Path(tmp))
        stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
        stage_lock.write(data, hashed_dirs=hashed_dirs)
        assert stage_lock.read() == data


_member_names = st.lists(st.text(max_size=6), min_size=1, max_size=3).map("/".join)


@_SETTINGS
@hypothesis.given(_member_names, st.one_of(st.none(), _member_names))
def test_lock_write_never_produces_a_lock_read_refuses(relpath: str, empty_dir: str | None) -> None:
    """Write and read apply one rule to recorded names: whatever a directory output records,
    the write either refuses it or writes a lock that reads back as written."""
    identity = ArtifactIdentity("p/stage", "o")
    members = listing.DirListing(
        files=[
            DirManifestEntry(
                relpath=relpath,
                hash="00112233445566ff",
                size=1,
                isexec=False,
            )
        ],
        empty_dirs=[] if empty_dir is None else [empty_dir],
    )
    hashed = _hashed_directory(members)
    data = LockData(
        code_manifest={},
        params={},
        dep_hashes={},
        output_hashes={identity: hashed["record"]},
        output_paths={identity: "data/p/stage/o"},
    )
    with tempfile.TemporaryDirectory() as tmp:
        stage_lock = lock.StageLock("p/stage", pathlib.Path(tmp))
        try:
            stage_lock.write(data, hashed_dirs={hashed["record"]["hash"]: hashed})
        except exceptions.InvalidPathError:
            assert not stage_lock.path.exists()
            return
        assert stage_lock.read() == data


@_SETTINGS
@hypothesis.given(st.binary(max_size=200))
def test_lock_read_treats_arbitrary_bytes_as_missing_or_valid(content: bytes) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stage_lock = lock.StageLock("p/stage", pathlib.Path(tmp))
        stage_lock.path.parent.mkdir(parents=True)
        stage_lock.path.write_bytes(content)
        result = stage_lock.read()
        assert result is None or isinstance(result["code_manifest"], dict)


@pytest.mark.parametrize(
    "document",
    [
        pytest.param(
            {
                "code_manifest": {},
                "params": {},
                "deps": [{"producer": ["p/a"], "key": None, "hash": "x"}],
                "outs": [],
            },
            id="list-producer",
        ),
        pytest.param(
            {
                "code_manifest": {},
                "params": {},
                "deps": [],
                "outs": [{"key": {"k": 1}, "hash": "x", "tag": "data"}],
            },
            id="mapping-key",
        ),
    ],
)
def test_lock_read_treats_wrongly_typed_identity_fields_as_missing(
    document: dict[str, object], tmp_path: pathlib.Path
) -> None:
    stage_lock = lock.StageLock("p/stage", tmp_path)
    stage_lock.path.parent.mkdir(parents=True)
    stage_lock.path.write_text(yaml.safe_dump(document))
    assert stage_lock.read() is None


@pytest.mark.parametrize(
    ("code_manifest", "params"),
    [
        pytest.param({}, [], id="params-list"),
        pytest.param({}, 3, id="params-int"),
        pytest.param(["self:x"], {}, id="code-manifest-list"),
    ],
)
def test_lock_read_treats_non_mapping_manifest_or_params_as_missing(
    code_manifest: object, params: object, tmp_path: pathlib.Path
) -> None:
    stage_lock = lock.StageLock("p/stage", tmp_path)
    stage_lock.path.parent.mkdir(parents=True)
    document = {"code_manifest": code_manifest, "params": params, "deps": [], "outs": []}
    stage_lock.path.write_text(yaml.safe_dump(document))
    assert stage_lock.read() is None


# =============================================================================
# Identities and names
# =============================================================================


@_SETTINGS
@hypothesis.given(_identities)
def test_identity_key_round_trips_through_identity_from_key(identity: ArtifactIdentity) -> None:
    assert identity_from_key(identity_key(identity)) == identity


_name_candidates = st.text(
    alphabet=string.ascii_letters + string.digits + "_@./-: \\\x00é", max_size=30
) | st.text(max_size=8)


@_SETTINGS
@hypothesis.given(_name_candidates)
def test_pipeline_and_lock_file_accept_exactly_the_same_names(name: str) -> None:
    """Any name a Pipeline accepts can name its lock file, and the reverse."""
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        try:
            compose.Pipeline(name, root=root)
            pipeline_ok = True
        except ValueError:
            pipeline_ok = False
        try:
            lock.StageLock(name, root / "stages")
            lock_ok = True
        except ValueError:
            lock_ok = False
    assert pipeline_ok == lock_ok, f"{name!r}: pipeline={pipeline_ok} lock={lock_ok}"


# =============================================================================
# DAG construction
# =============================================================================


def _stage(name: str, deps: list[str], out: str) -> RegistryStageInfo:
    def ref(producer: str) -> ArtifactRef:
        return ArtifactRef(
            identity=ArtifactIdentity(producer, None),
            format=loaders.PathOnly(),
            python_type=str,
            tag=ArtifactTag.DATA,
        )

    return RegistryStageInfo(
        func=_noop,
        name=name,
        deps={f"_{i}": ref(dep) for i, dep in enumerate(deps)},
        outs=[ref(out)],
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


def _noop() -> None:
    return None


@st.composite
def _acyclic_edges(draw: st.DrawFn) -> tuple[int, list[set[int]]]:
    """Stage i may only depend on stages < i, so every generated graph is acyclic."""
    count = draw(st.integers(min_value=1, max_value=8))
    deps = [
        draw(st.sets(st.integers(0, i - 1), max_size=i)) if i else set[int]() for i in range(count)
    ]
    return count, deps


def _stages_from_edges(count: int, deps: list[set[int]]) -> dict[str, RegistryStageInfo]:
    # Shuffle-free but non-sorted names so DFS order cannot coincide with name order.
    return {
        f"s{(i * 5) % 11}": _stage(
            f"s{(i * 5) % 11}", [f"out{j}" for j in sorted(deps[i])], f"out{i}"
        )
        for i in range(count)
    }


@_SETTINGS
@hypothesis.given(_acyclic_edges())
def test_execution_order_runs_every_producer_before_its_consumers(
    case: tuple[int, list[set[int]]],
) -> None:
    count, deps = case
    stages = _stages_from_edges(count, deps)
    order = engine_graph.get_execution_order(
        engine_graph.get_stage_dag(engine_graph.build_graph(stages))
    )
    assert sorted(order) == sorted(stages), "every stage runs exactly once"
    position = {name: index for index, name in enumerate(order)}
    for i in range(count):
        for j in deps[i]:
            assert position[f"s{(j * 5) % 11}"] < position[f"s{(i * 5) % 11}"]


@_SETTINGS
@hypothesis.given(_acyclic_edges(), st.data())
def test_adding_a_back_edge_is_reported_as_a_cycle(
    case: tuple[int, list[set[int]]], data: st.DataObject
) -> None:
    count, deps = case
    hypothesis.assume(count >= 2)
    consumer = data.draw(st.integers(1, count - 1))
    producer = data.draw(st.integers(0, consumer - 1))
    deps = [set(d) for d in deps]
    deps[consumer].add(producer)
    deps[producer].add(consumer)  # producer now also consumes the consumer's output
    with pytest.raises(exceptions.CyclicGraphError, match="Circular dependency"):
        engine_graph.build_graph(_stages_from_edges(count, deps))


@_SETTINGS
@hypothesis.given(_acyclic_edges(), st.data())
def test_width_is_the_largest_set_of_run_stages_none_of_which_reaches_another(
    case: tuple[int, list[set[int]]], data: st.DataObject
) -> None:
    """The width matches networkx's largest antichain of the run's stages, for a whole pipeline
    and for any subset of its stages, as `pivot run` names them, up to the pool's limit."""
    count, deps = case
    stages = _stages_from_edges(count, deps)
    stage_dag = engine_graph.get_stage_dag(engine_graph.build_graph(stages))
    run = data.draw(st.lists(st.sampled_from(sorted(stages)), min_size=1, unique=True))
    limit = data.draw(st.integers(1, count + 1))
    largest = max(len(antichain) for antichain in nx.antichains(stage_dag.subgraph(run)))
    assert engine_graph.get_width(stage_dag, run, limit=limit) == min(largest, limit)


@_SETTINGS
@hypothesis.given(_acyclic_edges(), st.data())
def test_width_cut_short_never_undersizes_the_pool(
    case: tuple[int, list[set[int]]], data: st.DataObject
) -> None:
    """When the search for the width runs out of steps, the size it settles for is never below
    the stages that can run at once, nor above the limit."""
    count, deps = case
    stages = _stages_from_edges(count, deps)
    stage_dag = engine_graph.get_stage_dag(engine_graph.build_graph(stages))
    run = sorted(stages)
    limit = data.draw(st.integers(1, count + 1))
    steps = data.draw(st.integers(0, 20))
    largest = max(len(antichain) for antichain in nx.antichains(stage_dag.subgraph(run)))
    with mock.patch.object(engine_graph, "_AUGMENT_STEPS", steps):
        width = engine_graph.get_width(stage_dag, run, limit=limit)
    assert min(largest, limit) <= width <= limit


def _full_replay_width(
    order: list[str],
    upstream: dict[str, set[str]],
    requests: dict[str, dict[str, int]],
    capacity: dict[str, int],
) -> int:
    """Scheduler.resource_width's replay with neither its heaps nor its early stop: start every
    ready stage that fits, in stage order, then finish the stage that started first."""
    waiting = {stage: set(upstream[stage]) for stage in order}
    counts = dict.fromkeys(capacity, 0)
    pending = list(order)
    running = list[str]()
    widest = 0
    while pending or running:
        for stage in list(pending):
            fits = all(
                counts[resource] + amount <= capacity[resource]
                for resource, amount in requests[stage].items()
            )
            if waiting[stage] or not fits:
                continue
            for resource, amount in requests[stage].items():
                counts[resource] += amount
            running.append(stage)
            pending.remove(stage)
        widest = max(widest, len(running))
        if not running:
            break
        stage = running.pop(0)
        for resource, amount in requests[stage].items():
            counts[resource] -= amount
        for consumer in order:
            waiting[consumer].discard(stage)
    return widest


@_SETTINGS
@hypothesis.given(_acyclic_edges(), st.data())
def test_resource_width_matches_a_full_replay(
    case: tuple[int, list[set[int]]], data: st.DataObject
) -> None:
    """The heaps and the early stop change only the cost: on any DAG, with stages sharing small
    resources, resource_width equals a full replay of the same admission order, up to the
    limit."""
    count, deps = case
    stages = _stages_from_edges(count, deps)
    graph = engine_graph.build_graph(stages)
    order = engine_graph.get_execution_order(engine_graph.get_stage_dag(graph))
    capacity = {
        "cpu": data.draw(st.integers(1, count + 1)),
        "a": data.draw(st.integers(1, 3)),
        "b": data.draw(st.integers(1, 3)),
    }
    requests = dict[str, dict[str, int]]()
    for stage in order:
        request = {"cpu": data.draw(st.integers(1, capacity["cpu"]))}
        for resource in data.draw(st.sets(st.sampled_from(["a", "b"]))):
            request[resource] = data.draw(st.integers(1, capacity[resource]))
        requests[stage] = request
    limit = data.draw(st.integers(1, count + 1))
    scheduler = Scheduler()
    scheduler.initialize(order, graph, stage_resources=requests, resource_capacity=capacity)
    upstream = {
        stage: {
            producer
            for producer in engine_graph.get_upstream_stages(graph, stage)
            if producer in order
        }
        for stage in order
    }

    expected = min(_full_replay_width(order, upstream, requests, capacity), limit)
    assert scheduler.resource_width(limit=limit) == expected


# =============================================================================
# Run-cache input hash
# =============================================================================

# One entry per producer: a stage has either a single unkeyed output or keyed outputs, never
# both, so (producer, None) and (producer, "k") never meet in one stage's deps.
_dep_entries = st.lists(
    st.builds(DepEntry, producer=_producers, key=_keys, hash=_hashes),
    max_size=4,
    unique_by=lambda d: d["producer"],
)


@_SETTINGS
@hypothesis.given(
    st.dictionaries(st.text(min_size=1), _hashes, max_size=4),
    st.dictionaries(st.text(min_size=1), _param_values, min_size=1, max_size=4),
    _dep_entries,
    st.data(),
)
def test_input_hash_ignores_ordering_and_tracks_param_values(
    manifest: dict[str, str],
    params: dict[str, object],
    deps: list[DepEntry],
    data: st.DataObject,
) -> None:
    out_specs = [("out", True)]
    base = run_history.compute_input_hash(manifest, params, deps, out_specs)

    reordered = run_history.compute_input_hash(
        dict(reversed(manifest.items())), dict(reversed(params.items())), deps[::-1], out_specs
    )
    assert reordered == base, "key and dependency order must not change the run-cache key"

    key = data.draw(st.sampled_from(sorted(params)))
    new_value = data.draw(_param_values.filter(lambda v: v != params[key]))
    changed = run_history.compute_input_hash(manifest, {**params, key: new_value}, deps, out_specs)
    assert changed != base, "a changed param value must change the run-cache key"
