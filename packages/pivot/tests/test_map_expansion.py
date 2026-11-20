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

from pivot import compose, config, executor, expansion, loaders, project, stage_def, status
from pivot.config import io as config_io
from pivot.remote import config as remote_config
from pivot.storage import cache
from pivot.types import Recovery, StageStatus

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

    from pivot.types import DirHash


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


def _stage_final(build: dict[str, int]) -> dict[str, int]:
    return {"final": build["built"] * 2}


def _stage_report(final: dict[str, int]) -> dict[str, int]:
    return {"report": final["final"] + 1}


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


def _chain(pipeline: compose.Pipeline) -> dict[str, int]:
    """Add to ``pipeline`` _stage_plan over input plan.yaml, _stage_cut mapped over its files and
    _stage_build over _stage_cut's instances; return _stage_build's output."""
    plan = compose.stage(_stage_plan)(pipeline.input("plan.yaml", t=dict[str, int]))
    cut = compose.stage(_stage_cut)(pipeline.map(plan), params=_CutParams())
    return compose.stage(_stage_build)(cut)


def _pipeline(root: pathlib.Path) -> compose.Pipeline:
    with compose.Pipeline("p", root=root) as pipeline:
        _chain(pipeline)
    return pipeline


def _composed(root: pathlib.Path) -> compose.Pipeline:
    """Pipeline report, whose _stage_final is mapped over the instances of pipeline p's
    _stage_build (``_chain``): walking report's handles reaches p/_stage_build before
    p/_stage_cut, which it is mapped over, and that before p/_stage_plan."""
    with compose.Pipeline("p", root=root) as prep:
        build = _chain(prep)
    with compose.Pipeline("report", root=root) as report:
        compose.stage(_stage_final)(build)
    return report


def _reentered(root: pathlib.Path) -> compose.Pipeline:
    """Pipeline alpha, _stage_plan and _stage_cut mapped over its files; then pipeline beta,
    _stage_build mapped over alpha/_stage_cut's instances and _stage_final over beta/
    _stage_build's; then alpha, re-entered with `with`, _stage_report mapped over beta/
    _stage_final's. Alpha registers beta's stages, which it reads, before its own: it lists
    beta/_stage_build before alpha/_stage_cut, which beta/_stage_build is mapped over."""
    with compose.Pipeline("alpha", root=root) as alpha:
        plan = compose.stage(_stage_plan)(alpha.input("plan.yaml", t=dict[str, int]))
        cut = compose.stage(_stage_cut)(alpha.map(plan), params=_CutParams())
    with compose.Pipeline("beta", root=root):
        final = compose.stage(_stage_final)(compose.stage(_stage_build)(cut))
    with alpha:
        compose.stage(_stage_report)(final)
    return alpha


def _nested(root: pathlib.Path) -> compose.Pipeline:
    """Pipeline alpha, _stage_plan, then _stage_cut and _stage_twin mapped over its files,
    whose `with` block holds pipeline beta's, _stage_build mapped over alpha/_stage_cut's
    instances, and then maps _stage_final over beta/_stage_build's. Alpha registers
    beta/_stage_build, which it reads, before its own stages, alpha/_stage_cut among them."""
    with compose.Pipeline("alpha", root=root) as alpha:
        plan = compose.stage(_stage_plan)(alpha.input("plan.yaml", t=dict[str, int]))
        cut = compose.stage(_stage_cut)(alpha.map(plan), params=_CutParams())
        compose.stage(_stage_twin)(alpha.map(plan))
        with compose.Pipeline("beta", root=root):
            build = compose.stage(_stage_build)(cut)
        compose.stage(_stage_final)(build)
    return alpha


def _every_instance(templates: list[str]) -> list[str]:
    """Each instance of each of mapped stages ``templates``, of keys a and b."""
    return [f"{template}@{key}" for template in templates for key in "ab"]


def _built(
    root: pathlib.Path, build: Callable[[pathlib.Path], compose.Pipeline], *, reverse: bool
) -> compose.Pipeline:
    """``build(root)``, the stages of other pipelines that it reads registered in the reverse of
    the order ``Pipeline._upstream_closure`` gives them when ``reverse``."""
    if not reverse:
        return build(root)
    original = compose.Pipeline._upstream_closure

    def reversed_closure(
        pipeline: compose.Pipeline,
    ) -> list[tuple[compose.Pipeline, compose._StageNode]]:
        return original(pipeline)[::-1]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(compose.Pipeline, "_upstream_closure", reversed_closure)
        return build(root)


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
_FINALS = ["report/_stage_final@a", "report/_stage_final@b"]


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
        expanded={"p/_stage_cut": _CUTS, "p/_stage_build": _BUILDS},
        listed={"p/_stage_cut": _CUTS, "p/_stage_build": _BUILDS},
        not_expanded={},
        listings={},
        refused=set[str](),
    )
    assert sorted(pipeline.list_stages()) == sorted(["p/_stage_plan", *_CUTS, *_BUILDS])
    assert list(known) == ["p/_stage_plan"], "the explanations it read, for the caller to reuse"


def test_a_template_reached_through_another_pipeline_expands_after_the_one_it_is_mapped_over(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_composed``: report's walk reaches p/_stage_build before p/_stage_cut, which it is
    mapped over. With p/_stage_plan current every template expands, each after the one it is
    mapped over, and the pipeline lists each after the one it reads; pivot 0.10.4 listed
    p/_stage_build first and, taking the templates in that order, left it and
    report/_stage_final not expanded."""
    root = set_project_root
    monkeypatch.chdir(root)
    _write_plan(root, {"a": 1, "b": 2})
    executor.run(pipeline=_composed(root))
    pipeline = _composed(root)

    result, _ = status.expand_current(pipeline)

    assert result["not_expanded"] == {}
    assert result["expanded"] == {
        "p/_stage_cut": _CUTS,
        "p/_stage_build": _BUILDS,
        "report/_stage_final": _FINALS,
    }
    assert sorted(pipeline.list_stages()) == sorted(["p/_stage_plan", *_CUTS, *_BUILDS, *_FINALS])
    assert pipeline.list_templates() == ["p/_stage_cut", "p/_stage_build", "report/_stage_final"]


@pytest.mark.parametrize(
    ("build", "reverse", "listed"),
    [
        pytest.param(
            _reentered,
            False,
            ["beta/_stage_build", "beta/_stage_final", "alpha/_stage_cut", "alpha/_stage_report"],
            id="re_entered",
        ),
        pytest.param(
            _reentered,
            True,
            ["beta/_stage_final", "beta/_stage_build", "alpha/_stage_cut", "alpha/_stage_report"],
            id="re_entered-reversed",
        ),
        pytest.param(
            _nested,
            False,
            ["beta/_stage_build", "alpha/_stage_cut", "alpha/_stage_twin", "alpha/_stage_final"],
            id="nested",
        ),
    ],
)
def test_a_pipeline_expands_a_mapped_stage_it_lists_before_the_one_it_is_mapped_over(
    set_project_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    build: Callable[[pathlib.Path], compose.Pipeline],
    reverse: bool,
    listed: list[str],
) -> None:
    """``_reentered`` and ``_nested``: alpha lists ``listed``, beta/_stage_build before
    alpha/_stage_cut, which it is mapped over, and, with ``_reentered``'s closure reversed
    (``_built``), beta/_stage_final before beta/_stage_build too. With alpha/_stage_plan current
    every mapped stage expands, each after the one it is mapped over, as the run expands them,
    and the result holds them in the order alpha lists them; pivot 0.10.4 took them in that
    order and left each of beta's stages, and every stage mapped over their instances, not
    expanded."""
    root = set_project_root
    monkeypatch.chdir(root)
    _write_plan(root, {"a": 1, "b": 2})
    executor.run(pipeline=_built(root, build, reverse=reverse))
    pipeline = _built(root, build, reverse=reverse)

    result, _ = status.expand_current(pipeline)

    assert pipeline.list_templates() == listed
    assert result["not_expanded"] == {}
    assert list(result["expanded"].items()) == [(name, _every_instance([name])) for name in listed]
    assert sorted(pipeline.list_stages()) == sorted(["alpha/_stage_plan", *_every_instance(listed)])


def _run_built(
    root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    build: Callable[[pathlib.Path], compose.Pipeline],
    *,
    reverse: bool,
) -> tuple[compose.Pipeline, dict[str, StageStatus]]:
    """``build`` in a new project at ``root``, registered as ``_built`` registers it with
    ``reverse``, and the status of each stage of its run."""
    for marker in (".pivot", ".git"):
        (root / marker).mkdir(parents=True)
    monkeypatch.setattr(project, "_project_root_cache", root)
    config_io.clear_config_cache()
    monkeypatch.chdir(root)
    _write_plan(root, {"a": 1, "b": 2})
    pipeline = _built(root, build, reverse=reverse)
    results = executor.run(pipeline=pipeline)
    return pipeline, {name: StageStatus(result["status"]) for name, result in results.items()}


def _stages_dir_files(root: pathlib.Path) -> dict[str, bytes]:
    """Each file in the stages directory of the project at ``root`` (locks, listings), by its
    path there."""
    stages_dir = root / ".pivot" / "stages"
    return {
        path.relative_to(stages_dir).as_posix(): path.read_bytes()
        for path in sorted(stages_dir.rglob("*"))
        if path.is_file()
    }


def test_the_registration_order_moves_no_stage_name_fingerprint_or_lock_byte(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_composed``, its stages of p registered in the reverse of the closure's order
    (``_built``), which is pivot 0.10.4's order, and in this one's, each run in a project of its
    own: both runs run every stage and instance, of the same names, with the same fingerprints,
    and leave the same locks and listings, byte for byte. So registering a stage after the
    stages it reads re-keys no lock of a composed pipeline."""
    backward, backward_ran = _run_built(tmp_path / "backward", monkeypatch, _composed, reverse=True)
    walked, walked_ran = _run_built(tmp_path / "walked", monkeypatch, _composed, reverse=False)

    assert backward.list_templates() == ["p/_stage_build", "p/_stage_cut", "report/_stage_final"]
    assert walked.list_templates() == ["p/_stage_cut", "p/_stage_build", "report/_stage_final"]
    every = ["p/_stage_plan", *_CUTS, *_BUILDS, *_FINALS]
    assert backward_ran == walked_ran == dict.fromkeys(every, StageStatus.RAN)
    assert sorted(backward.list_stages()) == sorted(walked.list_stages()) == sorted(every)
    assert {name: backward.ensure_fingerprint(name) for name in every} == {
        name: walked.ensure_fingerprint(name) for name in every
    }
    backward_files = _stages_dir_files(tmp_path / "backward")
    assert sorted(path for path in backward_files if path.endswith(".lock")) == sorted(
        f"{name}.lock" for name in every
    )
    assert backward_files == _stages_dir_files(tmp_path / "walked")


def test_the_order_a_re_entered_pipeline_lists_moves_no_name_fingerprint_or_lock_byte(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_reentered``, which lists beta's stages before alpha/_stage_cut, which
    beta/_stage_build is mapped over, registered in its own order and with its closure reversed
    (``_built``, which lists beta/_stage_final first), each run in a project of its own: both
    runs run every stage and instance, of the same names, with the same fingerprints, and leave
    the same locks and listings, byte for byte. So the order a pipeline lists its mapped stages
    in, which ``expansion.expand_current`` no longer follows, keys no lock."""
    backward, backward_ran = _run_built(
        tmp_path / "backward", monkeypatch, _reentered, reverse=True
    )
    walked, walked_ran = _run_built(tmp_path / "walked", monkeypatch, _reentered, reverse=False)

    assert backward.list_templates() == [
        "beta/_stage_final",
        "beta/_stage_build",
        "alpha/_stage_cut",
        "alpha/_stage_report",
    ]
    walked_templates = [
        "beta/_stage_build",
        "beta/_stage_final",
        "alpha/_stage_cut",
        "alpha/_stage_report",
    ]
    assert walked.list_templates() == walked_templates
    every = ["alpha/_stage_plan", *_every_instance(walked_templates)]
    assert backward_ran == walked_ran == dict.fromkeys(every, StageStatus.RAN)
    assert sorted(backward.list_stages()) == sorted(walked.list_stages()) == sorted(every)
    assert {name: backward.ensure_fingerprint(name) for name in every} == {
        name: walked.ensure_fingerprint(name) for name in every
    }
    backward_files = _stages_dir_files(tmp_path / "backward")
    assert sorted(path for path in backward_files if path.endswith(".lock")) == sorted(
        f"{name}.lock" for name in every
    )
    assert backward_files == _stages_dir_files(tmp_path / "walked")


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
    assert result["refused"] == set[str](), "a template waiting for its producer is not refused"
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


def _remove_plan_listing(root: pathlib.Path) -> None:
    for listing in (root / ".pivot" / "stages" / "p").glob("_stage_plan=*.listing"):
        listing.unlink()


def _plan_record(pipeline: compose.Pipeline) -> DirHash:
    """The record p/_stage_plan's lock keeps for its directory output."""
    producer = pipeline.get_stage("p/_stage_plan")
    record = expansion.recorded_directory(
        producer["outs"][0].identity,
        expansion.stage_lock(producer, config.locate_state_dir()).read(),
    )
    assert record is not None
    return record


def _corrupt_plan_listing(root: pathlib.Path) -> None:
    for listing in (root / ".pivot" / "stages" / "p").glob("_stage_plan=*.listing"):
        listing.write_bytes(listing.read_bytes() + b"junk\n")


def _keep_plan_listing(_root: pathlib.Path) -> None:
    pass


@pytest.mark.parametrize(
    ("plan", "change", "refused", "listed"),
    [
        pytest.param({"a": 1, "b": 2}, _remove_plan_listing, set[str](), {"p/_stage_cut"}),
        pytest.param({"a": 1, "b": 2}, _corrupt_plan_listing, set[str](), set[str]()),
        pytest.param({"a b": 1}, _keep_plan_listing, {"p/_stage_cut"}, set[str]()),
    ],
    ids=["listing_removed", "listing_corrupt", "key_names_no_stage"],
)
def test_a_producer_current_only_through_pull_excuses_only_a_listing_that_cannot_be_read(
    set_project_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    plan: dict[str, int],
    change: Callable[[pathlib.Path], None],
    refused: set[str],
    listed: set[str],
) -> None:
    """p/_stage_plan ran over ``plan``; then its outputs left the cache and the workspace, and a
    remote is configured, so it is current only because a pull would bring it up to date
    (``pulled``). A run never pulls: it runs p/_stage_plan again and expands from the listing
    that run writes. So a listing ``change`` leaves unreadable, removed (``listings`` names it,
    for a pull to fetch) or corrupt, leaves p/_stage_cut not expanded and not refused; a
    readable listing whose key cannot name a stage still refuses it."""
    root = set_project_root
    monkeypatch.chdir(root)
    _write_plan(root, plan)
    executor.run(pipeline=_pipeline(root))
    pipeline = _pipeline(root)
    monkeypatch.setattr(remote_config, "list_remotes", lambda: {"origin": "s3://bucket/cache"})
    record = _plan_record(pipeline)
    _empty_cache(root)
    plan_dir = root / "data" / "p" / "_stage_plan"
    for name in os.listdir(plan_dir):
        os.unlink(plan_dir / name)
    change(root)

    result, _ = status.expand_current(pipeline)

    assert result["expanded"] == {}
    assert list(result["not_expanded"]) == ["p/_stage_cut", "p/_stage_build"]
    assert result["refused"] == refused
    assert result["listings"] == dict.fromkeys(listed, record)


def test_a_listing_neither_committed_nor_cached_is_named_for_pull_to_fetch(
    ran: tuple[pathlib.Path, compose.Pipeline], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, pipeline = ran
    monkeypatch.setattr(remote_config, "list_remotes", lambda: {"origin": "s3://bucket/cache"})
    record = _plan_record(pipeline)
    _empty_cache(root)
    _remove_plan_listing(root)

    result, _ = status.expand_current(pipeline)

    assert "the listing of 'p/_stage_plan' is missing" in result["not_expanded"]["p/_stage_cut"]
    assert result["listings"] == {"p/_stage_cut": record}
    assert result["not_expanded"]["p/_stage_build"] == (
        "mapped over the instances of 'p/_stage_cut', which is not expanded"
    )
    assert result["refused"] == {"p/_stage_cut"}, (
        "the template over the missing listing is refused, its chain only waits"
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
    record = _plan_record(pipeline)
    _empty_cache(root)
    _remove_plan_listing(root)

    result, _ = status.expand_current(pipeline)

    assert result["listings"] == {"p/_stage_cut": record, "p/_stage_twin": record}


def test_a_missing_listing_names_the_run_that_writes_it_again(
    ran: tuple[pathlib.Path, compose.Pipeline],
) -> None:
    """A producer whose listing neither its stage directory nor the cache holds leaves its
    templates unexpanded, in a command and in a run, saying `pivot repro --force` fixes it."""
    root, pipeline = ran
    record = _plan_record(pipeline)
    cached = cache.resolve_cache_object_path(
        config.locate_cache_dir() / "files", record["hash"], notice=False
    )
    assert cached is not None
    cached.unlink()
    _remove_plan_listing(root)

    result, _ = status.expand_current(pipeline)
    run = executor.run(pipeline=_pipeline(root))

    remedy = "`pivot repro --force p/_stage_plan` writes it again"
    assert remedy in result["not_expanded"]["p/_stage_cut"], result["not_expanded"]
    assert remedy in run["p/_stage_cut"]["reason"], run["p/_stage_cut"]
    assert run["p/_stage_cut"]["reason"] == result["not_expanded"]["p/_stage_cut"]
    assert result["refused"] == {"p/_stage_cut"}


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
    assert result["refused"] == {"p/_stage_cut"}


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
    assert result["refused"] == {"p/_stage_fields"}
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
        expansion.Expansion(
            expanded={}, listed={}, not_expanded={}, listings={}, refused=set[str]()
        ),
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
