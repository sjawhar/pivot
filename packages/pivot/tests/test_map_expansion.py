"""Expanding mapped stages outside a run (``status.expand_current``), the rule every command uses.

A template expands only when its producer is current by the explain path, and then from the
listing the producer's lock records: the committed listing, else the cache's object. These tests
run a real pipeline in a temporary project, then expand as a command would.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Annotated, NoReturn, TypedDict

import pytest
import yaml

from pivot import compose, config, executor, expansion, loaders, stage_def, status
from pivot.remote import config as remote_config
from pivot.storage import cache
from pivot.types import Recovery

if TYPE_CHECKING:
    import pathlib


def _stage_plan(
    plan: dict[str, int],
) -> Annotated[dict[str, dict[str, int]], compose.directory, loaders.YAML[dict[str, int]]()]:
    return {f"{name}.yaml": {"value": value} for name, value in plan.items()}


class _CutParams(stage_def.StageParams):
    factor: int = 1


def _stage_cut(entry: dict[str, int], params: _CutParams) -> dict[str, int]:
    return {"cut": entry["value"] * params.factor}


def _stage_build(cut: dict[str, int]) -> dict[str, int]:
    return {"built": cut["cut"] + 1000}


def _stage_twin(entry: dict[str, int]) -> dict[str, int]:
    return {"twin": entry["value"]}


class _Fields(TypedDict):
    _stage_y: dict[str, int]


def _stage_fields(entry: dict[str, int]) -> _Fields:
    return {"_stage_y": entry}


def _stage_y() -> dict[str, int]:
    return {"value": 0}


def _write_plan(root: pathlib.Path, plan: dict[str, int]) -> None:
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (root / "data" / "raw" / "plan.yaml").write_text(yaml.safe_dump(plan))


def _pipeline(root: pathlib.Path) -> compose.Pipeline:
    with compose.Pipeline("p", root=root) as pipeline:
        plan = compose.stage(_stage_plan)(pipeline.input("plan.yaml", t=dict[str, int]))
        cut = compose.stage(_stage_cut)(pipeline.map(plan), params=_CutParams())
        compose.stage(_stage_build)(cut)
    return pipeline


@pytest.fixture
def ran(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[pathlib.Path, compose.Pipeline]:
    """A project whose pipeline ran over keys a and b, in a fresh pipeline object."""
    root = set_project_root
    monkeypatch.chdir(root)
    _write_plan(root, {"a": 1, "b": 2})
    executor.run(pipeline=_pipeline(root))
    return root, _pipeline(root)


_CUTS = ["p/_stage_cut@a", "p/_stage_cut@b"]
_BUILDS = ["p/_stage_build@a", "p/_stage_build@b"]


def _empty_cache(root: pathlib.Path) -> None:
    for directory, _dirs, names in os.walk(root / ".pivot" / "cache" / "files"):
        os.chmod(directory, 0o755)
        for name in names:
            os.unlink(os.path.join(directory, name))


def test_a_current_producer_expands_every_template(
    ran: tuple[pathlib.Path, compose.Pipeline],
) -> None:
    _, pipeline = ran

    result, known = status.expand_current(pipeline)

    assert result == expansion.Expansion(
        expanded={"p/_stage_cut": _CUTS, "p/_stage_build": _BUILDS}, not_expanded={}, listings={}
    )
    assert sorted(pipeline.list_stages()) == sorted(["p/_stage_plan", *_CUTS, *_BUILDS])
    assert list(known) == ["p/_stage_plan"], "the explanations it read, for the caller to reuse"


def test_expanding_again_replaces_the_last_expansion(
    ran: tuple[pathlib.Path, compose.Pipeline],
) -> None:
    root, pipeline = ran
    status.expand_current(pipeline)
    _write_plan(root, {"a": 1, "b": 2, "c": 3})

    result, _ = status.expand_current(pipeline)

    assert result["expanded"] == {}, "the producer is stale now"
    assert pipeline.list_stages() == ["p/_stage_plan"]


def test_a_stale_producer_leaves_its_templates_not_expanded_naming_why(
    ran: tuple[pathlib.Path, compose.Pipeline],
) -> None:
    root, pipeline = ran
    _write_plan(root, {"a": 1, "b": 3})

    result, _ = status.expand_current(pipeline)
    rows = status.not_expanded_status(result)

    assert result["expanded"] == {}
    assert result["not_expanded"]["p/_stage_cut"].startswith("waits for 'p/_stage_plan': ")
    assert [(row["name"], row["status"]) for row in rows] == [
        ("p/_stage_cut", "not_expanded"),
        ("p/_stage_build", "not_expanded"),
    ]
    assert rows[0]["reason"].startswith(status.NOT_EXPANDED_REASON_PREFIX)


def test_keys_come_from_the_recorded_listing_not_the_workspace(
    ran: tuple[pathlib.Path, compose.Pipeline],
) -> None:
    """The producer's directory is gone from the workspace but its outputs are in the cache, so
    it is current, and its keys come from the listing its lock records."""
    root, pipeline = ran
    plan_dir = root / "data" / "p" / "_stage_plan"
    for name in os.listdir(plan_dir):
        os.unlink(plan_dir / name)
    os.rmdir(plan_dir)

    result, _ = status.expand_current(pipeline)

    assert result["expanded"]["p/_stage_cut"] == _CUTS


def test_outputs_missing_from_the_cache_count_as_current_only_with_a_remote(
    ran: tuple[pathlib.Path, compose.Pipeline], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, pipeline = ran
    _empty_cache(root)
    plan_dir = root / "data" / "p" / "_stage_plan"
    for name in os.listdir(plan_dir):
        os.unlink(plan_dir / name)

    without_remote, _ = status.expand_current(pipeline)
    monkeypatch.setattr(remote_config, "list_remotes", lambda: {"origin": "s3://bucket/cache"})
    with_remote, _ = status.expand_current(pipeline)

    assert "Outputs missing from cache" in without_remote["not_expanded"]["p/_stage_cut"]
    assert with_remote["expanded"]["p/_stage_cut"] == _CUTS, "restorable from the remote"


def test_a_listing_neither_committed_nor_cached_is_named_for_pull_to_fetch(
    ran: tuple[pathlib.Path, compose.Pipeline], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, pipeline = ran
    monkeypatch.setattr(remote_config, "list_remotes", lambda: {"origin": "s3://bucket/cache"})
    producer = pipeline.get_stage("p/_stage_plan")
    record = expansion.recorded_directory(
        producer["outs"][0].identity,
        expansion.stage_lock(producer, config.locate_state_dir()).read(),
    )
    assert record is not None
    _empty_cache(root)
    for listing in (root / ".pivot" / "stages" / "p").glob("_stage_plan=*.listing"):
        listing.unlink()

    result, _ = status.expand_current(pipeline)

    assert "the listing of 'p/_stage_plan' is missing" in result["not_expanded"]["p/_stage_cut"]
    assert result["listings"] == {"p/_stage_cut": record}
    assert result["not_expanded"]["p/_stage_build"] == (
        "mapped over the instances of 'p/_stage_cut', which is not expanded"
    )


def test_every_template_over_a_missing_listing_names_it_for_pull_to_fetch(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two templates mapped over one producer's files both name its missing listing, the one
    that reads it and the one that reuses that read, so a pull of either fetches it."""
    root = set_project_root
    monkeypatch.chdir(root)
    _write_plan(root, {"a": 1, "b": 2})

    def twin_pipeline() -> compose.Pipeline:
        with compose.Pipeline("p", root=root) as pipeline:
            plan = compose.stage(_stage_plan)(pipeline.input("plan.yaml", t=dict[str, int]))
            compose.stage(_stage_cut)(pipeline.map(plan), params=_CutParams())
            compose.stage(_stage_twin)(pipeline.map(plan))
        return pipeline

    executor.run(pipeline=twin_pipeline())
    pipeline = twin_pipeline()
    monkeypatch.setattr(remote_config, "list_remotes", lambda: {"origin": "s3://bucket/cache"})
    producer = pipeline.get_stage("p/_stage_plan")
    record = expansion.recorded_directory(
        producer["outs"][0].identity,
        expansion.stage_lock(producer, config.locate_state_dir()).read(),
    )
    assert record is not None
    _empty_cache(root)
    for listing in (root / ".pivot" / "stages" / "p").glob("_stage_plan=*.listing"):
        listing.unlink()

    result, _ = status.expand_current(pipeline)

    assert result["listings"] == {"p/_stage_cut": record, "p/_stage_twin": record}


def test_a_missing_listing_names_the_run_that_writes_it_again(
    ran: tuple[pathlib.Path, compose.Pipeline],
) -> None:
    """A producer whose listing neither its stage directory nor the cache holds leaves its
    templates unexpanded, in a command and in a run, saying `pivot repro --force` fixes it."""
    root, pipeline = ran
    producer = pipeline.get_stage("p/_stage_plan")
    record = expansion.recorded_directory(
        producer["outs"][0].identity,
        expansion.stage_lock(producer, config.locate_state_dir()).read(),
    )
    assert record is not None
    cached = cache.resolve_cache_object_path(
        config.locate_cache_dir() / "files", record["hash"], notice=False
    )
    assert cached is not None
    cached.unlink()
    for listing in (root / ".pivot" / "stages" / "p").glob("_stage_plan=*.listing"):
        listing.unlink()

    result, _ = status.expand_current(pipeline)
    run = executor.run(pipeline=_pipeline(root))

    remedy = "`pivot repro --force p/_stage_plan` writes it again"
    assert remedy in result["not_expanded"]["p/_stage_cut"], result["not_expanded"]
    assert remedy in run["p/_stage_cut"]["reason"], run["p/_stage_cut"]
    assert run["p/_stage_cut"]["reason"] == result["not_expanded"]["p/_stage_cut"]


def test_a_refused_key_reads_the_same_in_a_command_and_in_a_run(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = set_project_root
    monkeypatch.chdir(root)
    _write_plan(root, {"a b": 1})
    run = executor.run(pipeline=_pipeline(root))
    pipeline = _pipeline(root)

    result, _ = status.expand_current(pipeline)

    reason = run["p/_stage_cut"]["reason"]
    assert reason.startswith("cannot expand over the files of 'p/_stage_plan': file 'a b.yaml'")
    assert result["not_expanded"]["p/_stage_cut"] == reason


def test_a_refused_expansion_leaves_its_template_and_its_chain_not_expanded(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Instance `p/_stage_fields@b`'s field `_stage_y` and pipeline `p/_stage_fields@b`'s stage
    `_stage_y` would both write `data/p/_stage_fields@b/_stage_y.yaml`."""
    root = set_project_root
    monkeypatch.chdir(root)
    _write_plan(root, {"a": 1, "b": 2})
    with compose.Pipeline("p", root=root) as mapped:
        plan = compose.stage(_stage_plan)(mapped.input("plan.yaml", t=dict[str, int]))
        fields = compose.stage(_stage_fields)(mapped.map(plan))
        compose.stage(_stage_build)(fields["_stage_y"])
    with compose.Pipeline("p/_stage_fields@b", root=root) as static:
        compose.stage(_stage_y)()
    pipeline = compose.Pipeline("all", root=root)
    pipeline.include(mapped)
    pipeline.include(static)
    executor.run(pipeline=pipeline, stages=["p/_stage_plan"])

    result, _ = status.expand_current(pipeline)

    assert "both write" in result["not_expanded"]["p/_stage_fields"]
    assert "p/_stage_build" in result["not_expanded"]
    assert sorted(pipeline.list_stages()) == ["p/_stage_fields@b/_stage_y", "p/_stage_plan"]


def test_a_pipeline_without_templates_reads_nothing(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = set_project_root
    with compose.Pipeline("p", root=root) as pipeline:
        compose.stage(_stage_build)(pipeline.input("cut.yaml", t=dict[str, int]))

    def _no_explanation(*_args: object, **_kwargs: object) -> NoReturn:
        pytest.fail("no explanation is read")

    monkeypatch.setattr(status, "_explanations_for", _no_explanation)

    assert status.expand_current(pipeline) == (
        expansion.Expansion(expanded={}, not_expanded={}, listings={}),
        {},
    )


def test_a_producer_counts_as_current_through_pull_only_by_its_recovery() -> None:
    """With a remote configured, a producer is current when pull brings it up to date: its
    explanation carries a recovery, and so does each stale stage above it. A stage whose
    cacheable output is missing beside an edited metric reads the cache reason but has no
    recovery, so neither it nor a stage below it counts."""
    rows = [
        ("p/plan", Recovery.PULL_OUTPUTS, list[str]()),
        ("p/edited", None, list[str]()),
        ("p/below_plan", Recovery.UPSTREAM, ["p/plan"]),
        ("p/below_edited", Recovery.UPSTREAM, ["p/edited"]),
        ("p/reads", Recovery.PULL_READS, list[str]()),
    ]

    assert status.pullable_stages(rows) == {"p/plan", "p/below_plan", "p/reads"}
