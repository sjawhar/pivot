"""Public executor behavior using compositional Pipeline stages.

The old file tested the removed annotation/register API and implementation helpers
that were replaced by store-based worker execution. This suite keeps the same
observable executor guarantees: ordering, cache state, force, errors, streams,
locks, generated outputs, and targeting.
"""

from __future__ import annotations

import atexit
import pathlib  # noqa: TC003 - compose resolves postponed stage annotations at runtime
import sys
import time
from typing import TYPE_CHECKING, Annotated, Any, cast

import pytest

from pivot import compose, exceptions, executor, loaders, project
from pivot.executor import core as executor_core
from tests import helpers

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_mock import MockerFixture

    from pivot.registry import PipelineLike as Pipeline

_JSON = Annotated[dict[str, int], loaders.JSON[dict[str, int]]()]


def _upper(data: str) -> str:
    return data.upper()


def _result(data: str) -> str:
    return f"Result: {data}"


def _lower(data: str) -> str:
    return data.lower()


def _identity(data: str) -> str:
    return data


def _uppercase_path(data: pathlib.Path) -> str:
    return data.read_text().upper()


def _directory_count(directory: pathlib.Path) -> str:
    return str(len(list(directory.iterdir())))


def _json_stage() -> _JSON:
    return {"value": 42}


def _failing() -> str:
    raise RuntimeError("intentional failure")


def _independent_a() -> str:
    return "a"


def _independent_b() -> str:
    return "b"


def _downstream(value: str) -> str:
    return f"downstream:{value}"


def _sys_exit_one() -> str:
    sys.exit(1)


def _sys_exit_zero() -> str:
    sys.exit(0)


def _keyboard_interrupt() -> str:
    raise KeyboardInterrupt


def _writes_streams() -> str:
    print("stdout line")
    print("stderr line", file=sys.stderr)
    return "done"


def _partial_stream() -> str:
    print("partial", end="")
    return "done"


def _sleep_a() -> str:
    time.sleep(0.4)
    return "a"


def _sleep_b() -> str:
    time.sleep(0.4)
    return "b"


def _timed(label: str, *, fail: bool = False) -> str:
    """Record a timing interval in the worker cwd for resource scheduling checks."""
    timing_path = pathlib.Path("resource_timing.log")
    with timing_path.open("a") as timing:
        timing.write(f"start {label} {time.monotonic()}\n")
    time.sleep(0.25)
    with timing_path.open("a") as timing:
        timing.write(f"end {label} {time.monotonic()}\n")
    if fail:
        raise RuntimeError(f"{label} failed")
    return label


def _resource_a() -> str:
    return _timed("a")


def _resource_b() -> str:
    return _timed("b")


def _resource_c(value: str) -> str:
    _ = value
    return _timed("c")


def _resource_fail() -> str:
    return _timed("fail", fail=True)


def _timing_intervals(root: pathlib.Path) -> dict[str, tuple[float, float]]:
    events = dict[str, dict[str, float]]()
    for line in (root / "resource_timing.log").read_text().splitlines():
        event, label, timestamp = line.split()
        events.setdefault(label, {})[event] = float(timestamp)
    return {label: (times["start"], times["end"]) for label, times in events.items()}


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _resource_stage(
    resources: dict[str, int | str], func: Callable[..., object]
) -> Callable[..., object]:
    """Declare a future Task 19 resources stage without statically assuming its signature."""
    stage = cast("Any", compose.stage)
    return cast("Callable[..., object]", stage(resources=resources)(func))


def _resource_pipeline(
    root: pathlib.Path,
    registrations: list[
        tuple[dict[str, int | str] | None, Callable[..., object], tuple[object, ...]]
    ],
) -> compose.Pipeline:
    pipeline = compose.Pipeline("resources", root=root)
    with pipeline:
        for resources, func, args in registrations:
            stage = (
                _resource_stage(resources, func) if resources is not None else compose.stage(func)
            )
            stage(*args)
    return pipeline


def _output(stage: str) -> pathlib.Path:
    return helpers.get_output_path(stage)


def _prepare(test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    root = test_pipeline.root
    monkeypatch.chdir(root)
    monkeypatch.setattr(project, "get_project_root", lambda: root)
    (root / ".git").mkdir(exist_ok=True)
    return root


def test_simple_pipeline_runs_in_order(
    test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stages execute in handle dependency order and persist the final value."""
    root = _prepare(test_pipeline, monkeypatch)
    (root / "input.txt").write_text("hello")
    step1 = helpers.register_test_stage(_upper, name="step1", deps={"data": "input.txt"})
    helpers.register_test_stage(_result, name="step2", deps={"data": step1})

    results = executor.run(pipeline=test_pipeline)

    assert results["test/step1"]["status"] == "ran"
    assert results["test/step2"]["status"] == "ran"
    assert _output("step2").read_text() == "Result: HELLO"


def test_unchanged_stages_are_skipped(
    test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged code and inputs are cached on the second run."""
    root = _prepare(test_pipeline, monkeypatch)
    (root / "input.txt").write_text("hello")
    helpers.register_test_stage(_upper, name="process", deps={"data": "input.txt"})

    assert executor.run(pipeline=test_pipeline)["test/process"]["status"] == "ran"
    assert executor.run(pipeline=test_pipeline)["test/process"]["status"] == "cached"


def test_input_change_triggers_rerun(
    test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing an input value invalidates its consumer."""
    root = _prepare(test_pipeline, monkeypatch)
    input_path = root / "input.txt"
    input_path.write_text("first")
    helpers.register_test_stage(_upper, name="process", deps={"data": "input.txt"})

    assert executor.run(pipeline=test_pipeline)["test/process"]["status"] == "ran"
    input_path.write_text("second")
    assert executor.run(pipeline=test_pipeline)["test/process"]["status"] == "ran"
    assert _output("process").read_text() == "SECOND"


def test_downstream_runs_when_upstream_is_forced(
    test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed upstream input invalidates its downstream consumer."""
    root = _prepare(test_pipeline, monkeypatch)
    input_path = root / "input.txt"
    input_path.write_text("first")
    upstream = helpers.register_test_stage(_upper, name="upstream", deps={"data": "input.txt"})
    helpers.register_test_stage(_downstream, name="downstream", deps={"value": upstream})

    executor.run(pipeline=test_pipeline)
    input_path.write_text("second")
    results = executor.run(pipeline=test_pipeline)
    assert results["test/upstream"]["status"] == "ran"
    assert results["test/downstream"]["status"] == "ran"


def test_run_specific_stage(test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch) -> None:
    """A specific qualified stage runs without unrelated stages."""
    _prepare(test_pipeline, monkeypatch)
    helpers.register_test_stage(_independent_a, name="a")
    helpers.register_test_stage(_independent_b, name="b")

    results = executor.run(stages=["test/a"], single_stage=True, pipeline=test_pipeline)

    assert set(results) == {"test/a"}
    assert results["test/a"]["status"] == "ran"


def test_missing_dependency_raises_error(
    test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing plain input errors before its stage executes."""
    _prepare(test_pipeline, monkeypatch)
    helpers.register_test_stage(_uppercase_path, name="process", deps={"data": "missing.txt"})

    with pytest.raises(exceptions.DependencyNotFoundError):
        executor.run(pipeline=test_pipeline)


def test_nonexistent_stage_raises_error(
    test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selecting a missing stage raises StageNotFoundError."""
    _prepare(test_pipeline, monkeypatch)

    with pytest.raises(exceptions.StageNotFoundError):
        executor.run(stages=["test/missing"], pipeline=test_pipeline)


def test_on_error_fail_stops_on_first_failure(
    test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default failure mode blocks later independent stages."""
    _prepare(test_pipeline, monkeypatch)
    helpers.register_test_stage(_failing, name="failing")
    helpers.register_test_stage(_independent_b, name="independent")

    results = executor.run(pipeline=test_pipeline, on_error="fail", parallel=False)

    assert results["test/failing"]["status"] == "failed"
    assert results["test/independent"]["status"] == "blocked"


def test_on_error_keep_going_continues_independent_stages(
    test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """keep_going still executes independent stages after a failure."""
    _prepare(test_pipeline, monkeypatch)
    helpers.register_test_stage(_failing, name="failing")
    helpers.register_test_stage(_independent_b, name="independent")

    results = executor.run(pipeline=test_pipeline, on_error="keep_going")

    assert results["test/failing"]["status"] == "failed"
    assert results["test/independent"]["status"] == "ran"


def test_on_error_keep_going_skips_downstream_of_failed(
    test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """keep_going marks a failed producer's dependent stage skipped."""
    _prepare(test_pipeline, monkeypatch)
    failed = helpers.register_test_stage(_failing, name="failing")
    helpers.register_test_stage(_downstream, name="dependent", deps={"value": failed})

    results = executor.run(pipeline=test_pipeline, on_error="keep_going")

    assert results["test/failing"]["status"] == "failed"
    assert results["test/dependent"]["status"] == "blocked"


def test_invalid_on_error_raises_value_error(test_pipeline: Pipeline) -> None:
    """Unsupported error policy is rejected at the public executor boundary."""
    with pytest.raises(ValueError, match="Invalid on_error mode"):
        executor.run(pipeline=test_pipeline, on_error="unknown")


@pytest.mark.parametrize(
    ("func", "name"),
    [
        (_sys_exit_one, "exit_one"),
        (_sys_exit_zero, "exit_zero"),
        (_keyboard_interrupt, "interrupt"),
    ],
)
def test_stage_signal_exits_return_failed(
    func: Callable[..., object], name: str, test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sys.exit and KeyboardInterrupt inside workers are stage failures, not harness exits."""
    _prepare(test_pipeline, monkeypatch)
    helpers.register_test_stage(func, name=name)

    assert executor.run(pipeline=test_pipeline)[f"test/{name}"]["status"] == "failed"


@pytest.mark.xfail(
    strict=True,
    reason="red test for Task 2: compose directory inputs are hashed as files",
)
def test_directory_dependency_hashed_and_runs(
    test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory plain input is hashed and supplied as a Path."""
    root = _prepare(test_pipeline, monkeypatch)
    directory = root / "directory"
    directory.mkdir()
    (directory / "file.txt").write_text("content")
    helpers.register_test_stage(_directory_count, name="count", deps={"directory": "directory"})

    results = executor.run(pipeline=test_pipeline)
    assert results["test/count"]["status"] == "ran"
    assert _output("count").read_text() == "1"


def test_parallel_false_runs_sequentially(
    test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """parallel=False has non-overlapping stage timing, not merely a long wall clock."""
    root = _prepare(test_pipeline, monkeypatch)
    helpers.register_test_stage(_resource_a, name="a")
    helpers.register_test_stage(_resource_b, name="b")

    executor.run(pipeline=test_pipeline, parallel=False)

    intervals = _timing_intervals(root)
    assert not _overlap(intervals["a"], intervals["b"]), intervals
    assert intervals["a"][1] <= intervals["b"][0] or intervals["b"][1] <= intervals["a"][0]


def test_executor_saves_outputs_to_cache_and_lock(
    test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Successful output has a cache object and identity-keyed lock entry."""
    root = _prepare(test_pipeline, monkeypatch)
    helpers.register_test_stage(_json_stage, name="json")

    executor.run(pipeline=test_pipeline)

    assert _output("json").read_text() == '{\n  "value": 42\n}'
    lock_path = root / ".pivot/stages/test/json.lock"
    assert lock_path.exists()
    assert "key:" in lock_path.read_text()
    assert list((root / ".pivot/cache/files").rglob("*"))


def test_executor_restores_missing_outputs_on_skip(
    test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A skipped stage restores its missing output from cache."""
    _prepare(test_pipeline, monkeypatch)
    helpers.register_test_stage(_json_stage, name="json")
    output = _output("json")

    executor.run(pipeline=test_pipeline)
    output.unlink()
    result = executor.run(pipeline=test_pipeline)["test/json"]

    assert result["status"] == "cached"
    assert output.read_text() == '{\n  "value": 42\n}'


def test_force_runs_unchanged_stage(
    test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """force bypasses normal cache skip logic."""
    _prepare(test_pipeline, monkeypatch)
    helpers.register_test_stage(_independent_a, name="stage")
    executor.run(pipeline=test_pipeline)

    assert executor.run(pipeline=test_pipeline, force=True)["test/stage"]["status"] == "ran"


def test_force_with_specific_stage(
    test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forcing a target includes its handle dependencies but excludes unrelated stages."""
    root = _prepare(test_pipeline, monkeypatch)
    (root / "input.txt").write_text("data")
    step1 = helpers.register_test_stage(_upper, name="step1", deps={"data": "input.txt"})
    helpers.register_test_stage(_downstream, name="step2", deps={"value": step1})
    helpers.register_test_stage(_independent_a, name="other")

    executor.run(pipeline=test_pipeline)
    results = executor.run(stages=["test/step2"], force=True, pipeline=test_pipeline)

    assert results["test/step1"]["status"] == "ran"
    assert results["test/step2"]["status"] == "ran"
    assert "test/other" not in results


def test_force_with_single_stage(test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch) -> None:
    """single_stage force executes only the selected stage with existing upstream output."""
    root = _prepare(test_pipeline, monkeypatch)
    (root / "input.txt").write_text("data")
    step1 = helpers.register_test_stage(_upper, name="step1", deps={"data": "input.txt"})
    helpers.register_test_stage(_downstream, name="step2", deps={"value": step1})

    executor.run(pipeline=test_pipeline)
    results = executor.run(
        stages=["test/step2"], single_stage=True, force=True, pipeline=test_pipeline
    )

    assert "test/step1" not in results
    assert results["test/step2"]["status"] == "ran"


def test_ensure_cleanup_registered_registers_atexit(mocker: MockerFixture) -> None:
    """Cleanup callback identity is registered once and cache reset in finally."""
    executor_core._ensure_cleanup_registered.cache_clear()
    try:
        mock_register = mocker.patch.object(atexit, "register")
        executor_core._ensure_cleanup_registered()
        mock_register.assert_called_once_with(executor_core._cleanup_worker_pool)

        executor_core._ensure_cleanup_registered()
        mock_register.assert_called_once()
    finally:
        executor_core._ensure_cleanup_registered.cache_clear()
        executor_core._ensure_cleanup_registered()


def _resource_root(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project, "get_project_root", lambda: tmp_path)
    (tmp_path / ".git").mkdir(exist_ok=True)
    return tmp_path


def _assert_serial(root: pathlib.Path, first: str, second: str) -> None:
    intervals = _timing_intervals(root)
    assert not _overlap(intervals[first], intervals[second]), intervals


# mutex=["gpu"] -> resources={"gpu": 1}
@pytest.mark.xfail(strict=True, reason="red test for Task 19: stage resources replace mutex")
def test_mutex_prevents_concurrent_execution(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shared GPU capacity serializes two otherwise independent stages."""
    root = _resource_root(tmp_path, monkeypatch)
    stage_a = _resource_stage({"gpu": 1}, _resource_a)
    stage_b = _resource_stage({"gpu": 1}, _resource_b)
    pipeline = compose.Pipeline("resources", root=root)
    with pipeline:
        stage_a()
        stage_b()

    results = executor.run(pipeline=pipeline, max_workers=2)
    assert results["resources/_resource_a"]["status"] == "ran"
    assert results["resources/_resource_b"]["status"] == "ran"
    _assert_serial(root, "a", "b")


# mutex=["gpu"] -> resources={"gpu": 1}
@pytest.mark.xfail(strict=True, reason="red test for Task 19: stage resources replace mutex")
def test_mutex_releases_on_completion(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resource is released after a successful owner completes."""
    root = _resource_root(tmp_path, monkeypatch)
    stage_a = _resource_stage({"gpu": 1}, _resource_a)
    stage_b = _resource_stage({"gpu": 1}, _resource_b)
    pipeline = compose.Pipeline("resources", root=root)
    with pipeline:
        stage_a()
        stage_b()

    executor.run(pipeline=pipeline, max_workers=2)
    intervals = _timing_intervals(root)
    assert intervals["b"][0] >= intervals["a"][1], intervals


# mutex=["gpu"] -> resources={"gpu": 1}
@pytest.mark.xfail(strict=True, reason="red test for Task 19: stage resources replace mutex")
def test_mutex_releases_on_failure(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A resource is released after its owner fails so keep-going work proceeds."""
    root = _resource_root(tmp_path, monkeypatch)
    failing = _resource_stage({"gpu": 1}, _resource_fail)
    succeeding = _resource_stage({"gpu": 1}, _resource_b)
    pipeline = compose.Pipeline("resources", root=root)
    with pipeline:
        failing()
        succeeding()

    results = executor.run(pipeline=pipeline, max_workers=2, on_error="keep_going")
    assert results["resources/_resource_fail"]["status"] == "failed"
    assert results["resources/_resource_b"]["status"] == "ran"
    intervals = _timing_intervals(root)
    assert intervals["b"][0] >= intervals["fail"][1], intervals


# mutex=["gpu", "cpu"] -> resources={"gpu": 1, "cpu": 1}
@pytest.mark.xfail(strict=True, reason="red test for Task 19: stage resources replace mutex")
def test_multiple_mutex_groups_per_stage(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage reserves every requested resource, not just the first one."""
    root = _resource_root(tmp_path, monkeypatch)
    stage_a = _resource_stage({"gpu": 1, "cpu": 1}, _resource_a)
    stage_b = _resource_stage({"cpu": 1}, _resource_b)
    pipeline = compose.Pipeline("resources", root=root)
    with pipeline:
        stage_a()
        stage_b()

    executor.run(pipeline=pipeline, max_workers=2)
    _assert_serial(root, "a", "b")


# mutex=["gpu"] -> resources={"gpu": 1}
@pytest.mark.xfail(strict=True, reason="red test for Task 19: stage resources replace mutex")
def test_mutex_with_dependencies(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resource scheduling preserves both handle dependency and resource order."""
    root = _resource_root(tmp_path, monkeypatch)
    stage_a = _resource_stage({"gpu": 1}, _resource_a)
    stage_c = _resource_stage({"gpu": 1}, _resource_c)
    pipeline = compose.Pipeline("resources", root=root)
    with pipeline:
        a = stage_a()
        stage_c(a)

    results = executor.run(pipeline=pipeline, max_workers=2)
    assert results["resources/_resource_a"]["status"] == "ran"
    assert results["resources/_resource_c"]["status"] == "ran"


def test_no_mutex_stages_unaffected(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stages with no resource request still execute normally."""
    root = _resource_root(tmp_path, monkeypatch)
    pipeline = compose.Pipeline("plain", root=root)
    with pipeline:
        compose.stage(_resource_a)()
        compose.stage(_resource_b)()

    results = executor.run(pipeline=pipeline, max_workers=2)
    assert results["plain/_resource_a"]["status"] == "ran"
    assert results["plain/_resource_b"]["status"] == "ran"
    intervals = _timing_intervals(root)
    assert set(intervals) == {"a", "b"}


# mutex=["gpu"] -> resources={"gpu": 1}
@pytest.mark.xfail(strict=True, reason="red test for Task 19: stage resources replace mutex")
def test_mutex_names_are_case_insensitive(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resource names normalize case before capacity matching."""
    root = _resource_root(tmp_path, monkeypatch)
    stage_a = _resource_stage({"GPU": 1}, _resource_a)
    stage_b = _resource_stage({"gpu": 1}, _resource_b)
    pipeline = compose.Pipeline("resources", root=root)
    with pipeline:
        stage_a()
        stage_b()

    executor.run(pipeline=pipeline, max_workers=2)
    _assert_serial(root, "a", "b")


# mutex=["*"] -> resources={"cpu": "all"}
@pytest.mark.xfail(strict=True, reason="red test for Task 19: stage resources replace mutex")
def test_exclusive_mutex_runs_alone(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full CPU reservation excludes a simultaneous ordinary CPU reservation."""
    root = _resource_root(tmp_path, monkeypatch)
    exclusive = _resource_stage({"cpu": "all"}, _resource_a)
    normal = _resource_stage({"cpu": 1}, _resource_b)
    pipeline = compose.Pipeline("resources", root=root)
    with pipeline:
        exclusive()
        normal()

    executor.run(pipeline=pipeline, max_workers=2)
    _assert_serial(root, "a", "b")
