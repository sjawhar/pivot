"""Runs of a pipeline mapped over a directory output (``Pipeline.map``).

The producer writes one file per key. A run that finds the producer current (it ran, or was
skipped with its output present) adds one instance of each mapped stage per key and runs them
in the same run, each depending on its own key's file only. A run targeted at the producer
expands nothing. Each test runs a real pipeline in a temporary project; the results the run
returns are its record of which stage ran and which was cached.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import pytest
import yaml

from pivot import compose, exceptions, executor, loaders, stage_def
from pivot.types import StageStatus

if TYPE_CHECKING:
    import pathlib
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


def test_a_single_stage_run_of_a_template_is_refused(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = set_project_root
    _project(root, monkeypatch, {"a": 1})
    pipeline = _chain_pipeline(root)

    with pytest.raises(exceptions.PipelineConfigError, match="pivot repro p/_stage_cut"):
        executor.run(pipeline=pipeline, stages=["p/_stage_cut"], single_stage=True)


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


# =============================================================================
# Failures
# =============================================================================


def test_an_instance_output_conflicting_with_a_static_stage_fails_the_template(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = set_project_root
    _project(root, monkeypatch, {"a": 1, "b": 2})
    with compose.Pipeline("p", root=root) as pipeline:
        plan = compose.stage(_stage_plan)(_plan_input(pipeline))
        compose.stage(_stage_cut)(pipeline.map(plan), params=_CutParams())
        with pipeline.variant("b"):
            compose.stage(_stage_cut)(plan["b.yaml"], params=_CutParams())

    results = executor.run(pipeline=pipeline, on_error="keep_going")

    statuses = _statuses(results)
    assert statuses["p/_stage_plan"] == _RAN, results
    assert statuses["p/_stage_cut@b"] == _RAN, "the static stage ran"
    assert statuses["p/_stage_cut"] == StageStatus.FAILED, results
    assert "'p/_stage_cut@b' already registered" in results["p/_stage_cut"]["reason"]
    assert "p/_stage_cut@a" not in results, "the failed expansion added no instance"


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
