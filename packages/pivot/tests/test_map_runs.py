"""Runs of a pipeline mapped over a directory output (``Pipeline.map``).

The producer writes one file per key. A run that finds the producer current (it ran, or was
skipped with its output present) adds one instance of each mapped stage per key and runs them
in the same run, each depending on its own key's file only. A run targeted at the producer
expands nothing. Each test runs a real pipeline in a temporary project; the results the run
returns are its record of which stage ran and which was cached.
"""

from __future__ import annotations

import pathlib
import time
from typing import TYPE_CHECKING, Annotated, TypedDict

import pytest
import yaml

from pivot import compose, config, exceptions, executor, loaders, stage_def
from pivot.engine import engine as engine_mod
from pivot.engine.types import RunRequested
from pivot.storage import state as state_mod
from pivot.types import OnError, StageStatus

if TYPE_CHECKING:
    from collections.abc import Callable

    from pivot.executor.core import ExecutionSummary


# =============================================================================
# Stage functions
# =============================================================================


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


def _stage_failing_plan(
    plan: dict[str, int],
) -> Annotated[dict[str, dict[str, int]], compose.directory, loaders.YAML[dict[str, int]]()]:
    raise RuntimeError("plan failed on purpose")


def _stage_spaced_plan(
    plan: dict[str, int],
) -> Annotated[dict[str, dict[str, int]], compose.directory, loaders.YAML[dict[str, int]]()]:
    return {"a b.yaml": {"value": 1}}


def _stage_static() -> dict[str, int]:
    return {"static": 1}


class _FragileParams(stage_def.StageParams):
    # A file to create before failing, so a stage that waits for the failure can go on
    marker: str = ""


def _stage_fragile_cut(entry: dict[str, int], params: _FragileParams) -> dict[str, int]:
    if entry["value"] < 0:
        if params.marker:
            pathlib.Path(params.marker).touch()
        raise RuntimeError("negative value")
    return {"cut": entry["value"]}


class _LateParams(stage_def.StageParams):
    marker: str = ""


def _stage_late(params: _LateParams) -> dict[str, int]:
    """Finish only once `_stage_fragile_cut` has failed, leaving the engine time to record it."""
    marker = pathlib.Path(params.marker)
    deadline = time.monotonic() + 60
    while not marker.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"{marker} never appeared")
        time.sleep(0.05)
    time.sleep(2)
    return {"late": 1}


def _stage_join(cut: dict[str, int], late: dict[str, int]) -> dict[str, int]:
    return {"joined": cut["cut"] + late["late"]}


def _stage_broken() -> dict[str, int]:
    raise RuntimeError("broken on purpose")


class _Fields(TypedDict):
    _stage_y: dict[str, int]


def _stage_fields(entry: dict[str, int]) -> _Fields:
    return {"_stage_y": entry}


def _stage_y() -> dict[str, int]:
    return {"value": 0}


# =============================================================================
# Helpers
# =============================================================================


def _project(root: pathlib.Path, monkeypatch: pytest.MonkeyPatch, plan: dict[str, int]) -> None:
    monkeypatch.chdir(root)
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    _write_plan(root, plan)


def _write_plan(root: pathlib.Path, plan: dict[str, int]) -> None:
    (root / "data" / "raw" / "plan.yaml").write_text(yaml.safe_dump(plan))


def _plan_input(pipeline: compose.Pipeline) -> dict[str, int]:
    return pipeline.input("plan.yaml", t=dict[str, int])


def _chain_pipeline(
    root: pathlib.Path,
    plan_stage: Callable[[dict[str, int]], dict[str, dict[str, int]]] = _stage_plan,
) -> compose.Pipeline:
    with compose.Pipeline("p", root=root) as pipeline:
        plan = compose.stage(plan_stage)(_plan_input(pipeline))
        cut = compose.stage(_stage_cut)(pipeline.map(plan), params=_CutParams())
        compose.stage(_stage_build)(cut)
    return pipeline


def _statuses(results: dict[str, ExecutionSummary]) -> dict[str, StageStatus]:
    return {name: StageStatus(result["status"]) for name, result in results.items()}


_RAN = StageStatus.RAN
_CACHED = StageStatus.CACHED
_ALL = ["a", "b", "c"]


def _chains(keys: list[str], status: StageStatus) -> dict[str, StageStatus]:
    return {
        **{f"p/_stage_cut@{key}": status for key in keys},
        **{f"p/_stage_build@{key}": status for key in keys},
    }


def _read(root: pathlib.Path, stage: str) -> object:
    return yaml.safe_load((root / "data" / "p" / f"{stage}.yaml").read_text())


# =============================================================================
# Expansion within one run
# =============================================================================


def test_one_run_runs_the_producer_and_every_chain(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = set_project_root
    _project(root, monkeypatch, {"a": 1, "b": 2, "c": 3})
    pipeline = _chain_pipeline(root)

    results = executor.run(pipeline=pipeline)

    assert _statuses(results) == {"p/_stage_plan": _RAN, **_chains(_ALL, _RAN)}, results
    assert _read(root, "_stage_cut@b") == {"cut": 2}
    assert _read(root, "_stage_build@c") == {"built": 1003}


def test_a_second_run_expands_from_the_skipped_producer_and_caches_every_chain(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = set_project_root
    _project(root, monkeypatch, {"a": 1, "b": 2, "c": 3})
    pipeline = _chain_pipeline(root)
    executor.run(pipeline=pipeline)

    results = executor.run(pipeline=pipeline)

    assert _statuses(results) == {"p/_stage_plan": _CACHED, **_chains(_ALL, _CACHED)}, results


def test_an_edit_to_one_keys_input_reruns_only_that_chain(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = set_project_root
    _project(root, monkeypatch, {"a": 1, "b": 2, "c": 3})
    pipeline = _chain_pipeline(root)
    executor.run(pipeline=pipeline)
    _write_plan(root, {"a": 1, "b": 20, "c": 3})

    results = executor.run(pipeline=pipeline)

    assert _statuses(results) == {
        "p/_stage_plan": _RAN,
        **_chains(["a", "c"], _CACHED),
        **_chains(["b"], _RAN),
    }, results
    assert _read(root, "_stage_build@b") == {"built": 1020}


def test_a_new_key_runs_the_producer_and_the_new_chain_only(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = set_project_root
    _project(root, monkeypatch, {"a": 1, "b": 2, "c": 3})
    pipeline = _chain_pipeline(root)
    executor.run(pipeline=pipeline)
    _write_plan(root, {"a": 1, "b": 2, "c": 3, "d": 4})

    results = executor.run(pipeline=pipeline)

    assert _statuses(results) == {
        "p/_stage_plan": _RAN,
        **_chains(_ALL, _CACHED),
        **_chains(["d"], _RAN),
    }, results
    assert _read(root, "_stage_build@d") == {"built": 1004}


def test_params_for_one_key_rerun_that_instance_after_a_skipped_producer(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The producer is cached by the coordinator, so no worker has started when the instances
    are added; the changed instance still runs, in the same run."""
    root = set_project_root
    _project(root, monkeypatch, {"a": 1, "b": 2, "c": 3})
    pipeline = _chain_pipeline(root)
    executor.run(pipeline=pipeline)
    (root / "params.yaml").write_text("p/_stage_cut@b:\n  factor: 10\n")

    results = executor.run(pipeline=pipeline)

    assert _statuses(results) == {
        "p/_stage_plan": _CACHED,
        **_chains(["a", "c"], _CACHED),
        **_chains(["b"], _RAN),
    }, results
    assert _read(root, "_stage_cut@b") == {"cut": 20}


def test_params_for_the_template_reach_every_instance(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = set_project_root
    _project(root, monkeypatch, {"a": 1, "b": 2})
    (root / "params.yaml").write_text("p/_stage_cut:\n  factor: 3\n")
    pipeline = _chain_pipeline(root)

    executor.run(pipeline=pipeline)

    assert _read(root, "_stage_cut@a") == {"cut": 3}
    assert _read(root, "_stage_cut@b") == {"cut": 6}


def test_a_no_commit_run_expands_too(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = set_project_root
    _project(root, monkeypatch, {"a": 1, "b": 2})
    pipeline = _chain_pipeline(root)

    results = executor.run(pipeline=pipeline, no_commit=True)

    assert _statuses(results) == {"p/_stage_plan": _RAN, **_chains(["a", "b"], _RAN)}, results


# =============================================================================
# Targets
# =============================================================================


def test_a_run_targeted_at_the_producer_expands_nothing(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = set_project_root
    _project(root, monkeypatch, {"a": 1, "b": 2})
    pipeline = _chain_pipeline(root)

    results = executor.run(pipeline=pipeline, stages=["p/_stage_plan"])

    assert _statuses(results) == {"p/_stage_plan": _RAN}, results
    assert pipeline.list_stages() == ["p/_stage_plan"]
    assert not (root / "data" / "p" / "_stage_cut@a.yaml").exists()


def test_a_run_targeted_at_a_template_runs_its_upstream_and_expands_it(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = set_project_root
    _project(root, monkeypatch, {"a": 1, "b": 2})
    pipeline = _chain_pipeline(root)

    results = executor.run(pipeline=pipeline, stages=["p/_stage_build"])

    assert _statuses(results) == {"p/_stage_plan": _RAN, **_chains(["a", "b"], _RAN)}, results


def test_a_run_targeted_at_an_instance_runs_that_instances_chain_only(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run reaches the instance through its template, and its chained templates expand over
    that key alone."""
    root = set_project_root
    _project(root, monkeypatch, {"a": 1, "b": 2, "c": 3})
    pipeline = _chain_pipeline(root)

    results = executor.run(pipeline=pipeline, stages=["p/_stage_build@b"])

    assert _statuses(results) == {"p/_stage_plan": _RAN, **_chains(["b"], _RAN)}, results
    assert pipeline.instances_of("p/_stage_cut") == ["p/_stage_cut@b"]
    assert not (root / "data" / "p" / "_stage_cut@a.yaml").exists()


def test_a_forced_run_of_an_instance_forces_no_other_instance(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = set_project_root
    _project(root, monkeypatch, {"a": 1, "b": 2, "c": 3})
    pipeline = _chain_pipeline(root)
    executor.run(pipeline=pipeline)

    results = executor.run(
        pipeline=pipeline, stages=["p/_stage_cut@a", "p/_stage_build@c"], force=True
    )

    assert _statuses(results) == {
        "p/_stage_plan": _RAN,
        "p/_stage_cut@a": _RAN,
        "p/_stage_cut@c": _RAN,
        "p/_stage_build@c": _RAN,
    }, results


def test_a_template_target_beside_its_instance_expands_every_key(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`p/_stage_build` needs every instance of `p/_stage_cut`, whatever else is named."""
    root = set_project_root
    _project(root, monkeypatch, {"a": 1, "b": 2})
    pipeline = _chain_pipeline(root)

    results = executor.run(pipeline=pipeline, stages=["p/_stage_cut@a", "p/_stage_build"])

    assert _statuses(results) == {"p/_stage_plan": _RAN, **_chains(["a", "b"], _RAN)}, results


@pytest.mark.parametrize(
    ("target", "reason"),
    [
        pytest.param(
            "p/_stage_cut@zzz", "Unknown stage(s): p/_stage_cut@zzz", id="the-named-instance"
        ),
        pytest.param(
            "p/_stage_build@zzz",
            "Unknown stage(s): p/_stage_cut@zzz, needed by the target p/_stage_build@zzz",
            id="an-instance-downstream",
        ),
    ],
)
def test_a_named_key_the_producer_writes_no_file_for_fails_the_template(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch, target: str, reason: str
) -> None:
    """The producer ran in this run, so only the run can tell the key is not in the plan: it
    fails the first template over that key as `pivot status` refuses the name, naming the
    target the run was asked for when that is an instance further down the chain."""
    root = set_project_root
    _project(root, monkeypatch, {"a": 1})
    pipeline = _chain_pipeline(root)

    results = executor.run(pipeline=pipeline, stages=[target], on_error="keep_going")

    statuses = _statuses(results)
    assert statuses["p/_stage_plan"] == _RAN
    assert statuses["p/_stage_cut"] == StageStatus.FAILED, results
    assert results["p/_stage_cut"]["reason"] == reason
    assert "p/_stage_build" not in target or statuses["p/_stage_build"] == StageStatus.BLOCKED
    assert pipeline.instances_of("p/_stage_cut") == []


def test_a_single_stage_run_of_a_template_is_refused(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = set_project_root
    _project(root, monkeypatch, {"a": 1})
    pipeline = _chain_pipeline(root)

    with pytest.raises(exceptions.PipelineConfigError, match="pivot repro p/_stage_cut"):
        executor.run(pipeline=pipeline, stages=["p/_stage_cut"], single_stage=True)


def test_a_restart_after_a_reload_keeps_its_mapped_targets(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reloaded pipeline lists no template or instance among its stages; a run request
    replayed after the reload keeps them, and drops only the names the reload removed."""
    root = set_project_root
    _project(root, monkeypatch, {"a": 1})
    request = RunRequested(
        type="run_requested",
        stages=["p/_stage_cut", "p/_stage_build@a", "p/_stage_gone"],
        force=False,
        reason="test",
        single_stage=False,
        parallel=True,
        max_workers=None,
        resource_capacity={},
        no_commit=False,
        on_error=OnError.FAIL,
        cache_dir=None,
        allow_uncached_incremental=False,
        checkout_missing=False,
    )
    engine = engine_mod.Engine(pipeline=_chain_pipeline(root))

    kept = engine._drop_removed_stages(request)

    assert kept is not None
    assert kept["stages"] == ["p/_stage_cut", "p/_stage_build@a"]


def test_each_run_expands_afresh(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key that left the plan has no instance in the next run (its lock stays: part P3)."""
    root = set_project_root
    _project(root, monkeypatch, {"a": 1, "b": 2})
    pipeline = _chain_pipeline(root)
    executor.run(pipeline=pipeline)
    _write_plan(root, {"a": 1})

    results = executor.run(pipeline=pipeline)

    assert _statuses(results) == {"p/_stage_plan": _RAN, **_chains(["a"], _CACHED)}, results
    assert pipeline.instances_of("p/_stage_cut") == ["p/_stage_cut@a"]


@pytest.mark.parametrize(
    "targets",
    [
        pytest.param(None, id="every-stage"),
        pytest.param(["p/_stage_build@a"], id="an-instance"),
        pytest.param(["p/_stage_cut", "p/_stage_build@b"], id="a-template-and-an-instance"),
    ],
)
def test_the_run_history_records_the_targets_as_given(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch, targets: list[str] | None
) -> None:
    """A target naming an instance or a template is recorded as given, never as the template
    the run reached it through; a run of every stage records every stage it ran, the producer
    and each instance, and no template."""
    root = set_project_root
    _project(root, monkeypatch, {"a": 1, "b": 2})

    results = executor.run(pipeline=_chain_pipeline(root), stages=targets)

    with state_mod.StateDB(config.get_state_dir()) as state_db:
        (run,) = state_db.list_runs(limit=1)
    assert run["targeted_stages"] == (run["execution_order"] if targets is None else targets)
    assert sorted(run["execution_order"]) == sorted(results), run


# =============================================================================
# Failures
# =============================================================================


def test_an_instance_output_conflicting_with_a_static_stage_fails_the_template(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Instance `p/_stage_fields@b`'s field `_stage_y` and pipeline `p/_stage_fields@b`'s stage
    `_stage_y` would both write `data/p/_stage_fields@b/_stage_y.yaml`."""
    root = set_project_root
    _project(root, monkeypatch, {"a": 1, "b": 2})
    with compose.Pipeline("p", root=root) as mapped:
        plan = compose.stage(_stage_plan)(_plan_input(mapped))
        compose.stage(_stage_fields)(mapped.map(plan))
    with compose.Pipeline("p/_stage_fields@b", root=root) as static:
        compose.stage(_stage_y)()
    pipeline = compose.Pipeline("all", root=root)
    pipeline.include(mapped)
    pipeline.include(static)

    results = executor.run(pipeline=pipeline, on_error="keep_going")

    statuses = _statuses(results)
    assert statuses["p/_stage_plan"] == _RAN, results
    assert statuses["p/_stage_fields@b/_stage_y"] == _RAN, "the static stage ran"
    assert statuses["p/_stage_fields"] == StageStatus.FAILED, results
    assert "both write" in results["p/_stage_fields"]["reason"], results
    assert "p/_stage_fields@a" not in results, "the failed expansion added no instance"


def test_a_blocked_instance_names_its_own_failed_upstream(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`p/_stage_broken` fails first; `p/_stage_build@a` is blocked by `p/_stage_fragile_cut@a`,
    the stage above it that failed, and says so."""
    root = set_project_root
    _project(root, monkeypatch, {"a": -1, "b": 2})
    with compose.Pipeline("p", root=root) as pipeline:
        compose.stage(_stage_broken)()
        plan = compose.stage(_stage_plan)(_plan_input(pipeline))
        cut = compose.stage(_stage_fragile_cut)(pipeline.map(plan), params=_FragileParams())
        compose.stage(_stage_build)(cut)

    results = executor.run(pipeline=pipeline, on_error="keep_going")

    assert _statuses(results) == {
        "p/_stage_broken": StageStatus.FAILED,
        "p/_stage_plan": _RAN,
        "p/_stage_fragile_cut@a": StageStatus.FAILED,
        "p/_stage_fragile_cut@b": _RAN,
        "p/_stage_build@a": StageStatus.BLOCKED,
        "p/_stage_build@b": _RAN,
    }, results
    assert results["p/_stage_build@a"]["reason"] == "upstream 'p/_stage_fragile_cut@a' failed"


def test_an_instance_added_under_a_failed_upstream_names_it(
    set_project_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """`p/_stage_join` also waits for `p/_stage_late`, which finishes only once
    `p/_stage_fragile_cut@a` has failed, so the run adds `p/_stage_join@a` already blocked. It
    names `p/_stage_fragile_cut@a`, not `p/_stage_broken`, the run's first failure."""
    root = set_project_root
    _project(root, monkeypatch, {"a": -1, "b": 2})
    marker = str(tmp_path_factory.mktemp("marker") / "failed")
    with compose.Pipeline("p", root=root) as pipeline:
        compose.stage(_stage_broken)()
        plan = compose.stage(_stage_plan)(_plan_input(pipeline))
        cut = compose.stage(_stage_fragile_cut)(
            pipeline.map(plan), params=_FragileParams(marker=marker)
        )
        late = compose.stage(_stage_late)(params=_LateParams(marker=marker))
        compose.stage(_stage_join)(cut, late)

    results = executor.run(pipeline=pipeline, on_error="keep_going", max_workers=4)

    assert _statuses(results) == {
        "p/_stage_broken": StageStatus.FAILED,
        "p/_stage_plan": _RAN,
        "p/_stage_late": _RAN,
        "p/_stage_fragile_cut@a": StageStatus.FAILED,
        "p/_stage_fragile_cut@b": _RAN,
        "p/_stage_join@a": StageStatus.BLOCKED,
        "p/_stage_join@b": _RAN,
    }, results
    assert results["p/_stage_join@a"]["reason"] == "upstream 'p/_stage_fragile_cut@a' failed"


def test_a_failed_expansion_starts_no_other_stage_under_fail_fast(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both mapped stages become ready when the producer finishes, and the engine expands them
    in one pass: `p/_stage_cut` adds its instances, then `p/_stage_fields` fails (its instance
    `p/_stage_fields@b` would write the output of pipeline `p/_stage_fields@b`'s stage).
    Fail-fast then starts nothing more, so `p/_stage_cut`'s instances are blocked, not run."""
    root = set_project_root
    _project(root, monkeypatch, {"a": 1, "b": 2})
    with compose.Pipeline("p", root=root) as mapped:
        plan = compose.stage(_stage_plan)(_plan_input(mapped))
        compose.stage(_stage_cut)(mapped.map(plan), params=_CutParams())
        compose.stage(_stage_fields)(mapped.map(plan))
    with compose.Pipeline("p/_stage_fields@b", root=root) as static:
        compose.stage(_stage_y)()
    pipeline = compose.Pipeline("all", root=root)
    pipeline.include(mapped)
    pipeline.include(static)

    results = executor.run(pipeline=pipeline, on_error="fail")

    assert _statuses(results) == {
        "p/_stage_plan": _RAN,
        "p/_stage_fields@b/_stage_y": _RAN,
        "p/_stage_fields": StageStatus.FAILED,
        "p/_stage_cut@a": StageStatus.BLOCKED,
        "p/_stage_cut@b": StageStatus.BLOCKED,
    }, results
    assert results["p/_stage_cut@a"]["reason"] == "upstream 'p/_stage_fields' failed"
    assert not (root / "data" / "p" / "_stage_cut@a.yaml").exists()


def test_a_key_that_cannot_name_a_stage_fails_the_template_naming_the_file(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = set_project_root
    _project(root, monkeypatch, {"a": 1})
    pipeline = _chain_pipeline(root, _stage_spaced_plan)

    results = executor.run(pipeline=pipeline, on_error="keep_going")

    statuses = _statuses(results)
    assert statuses["p/_stage_spaced_plan"] == _RAN, results
    assert statuses["p/_stage_cut"] == StageStatus.FAILED, results
    assert "'a b.yaml'" in results["p/_stage_cut"]["reason"]
    assert statuses["p/_stage_build"] == StageStatus.BLOCKED, "the chained template is blocked"


def test_a_failed_producer_blocks_its_templates(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = set_project_root
    _project(root, monkeypatch, {"a": 1})
    pipeline = _chain_pipeline(root, _stage_failing_plan)

    results = executor.run(pipeline=pipeline, on_error="keep_going")

    assert _statuses(results) == {
        "p/_stage_failing_plan": StageStatus.FAILED,
        "p/_stage_cut": StageStatus.BLOCKED,
        "p/_stage_build": StageStatus.BLOCKED,
    }, results


def test_a_template_mapped_alongside_a_static_stage_runs_beside_it(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = set_project_root
    _project(root, monkeypatch, {"a": 1})
    with compose.Pipeline("p", root=root) as pipeline:
        plan = compose.stage(_stage_plan)(_plan_input(pipeline))
        compose.stage(_stage_cut)(pipeline.map(plan), params=_CutParams())
        compose.stage(_stage_static)()

    results = executor.run(pipeline=pipeline)

    assert _statuses(results) == {
        "p/_stage_plan": _RAN,
        "p/_stage_cut@a": _RAN,
        "p/_stage_static": _RAN,
    }, results
