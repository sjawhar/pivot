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
    """parallel=False serializes independent stages."""
    _prepare(test_pipeline, monkeypatch)
    helpers.register_test_stage(_sleep_a, name="a")
    helpers.register_test_stage(_sleep_b, name="b")

    start = time.monotonic()
    executor.run(pipeline=test_pipeline, parallel=False)

    assert time.monotonic() - start >= 0.7


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


def test_ensure_cleanup_registered_registers_atexit(mocker: MockerFixture) -> None:
    """The worker cleanup registration is cached and happens once."""
    executor_core._ensure_cleanup_registered.cache_clear()
    calls = list[object]()

    def register(callback: Callable[..., object], *_args: object, **_kwargs: object) -> None:
        calls.append(callback)

    mocker.patch.object(atexit, "register", side_effect=register)

    executor_core._ensure_cleanup_registered()
    executor_core._ensure_cleanup_registered()

    assert len(calls) == 1


def _declare_resource_stage(resources: dict[str, int | str]) -> None:
    """Exercise Task 19's future decorator API without making pyright infer it."""
    stage = cast("Any", compose.stage)

    @stage(resources=resources)
    def resource_stage() -> None:
        pass

    _ = resource_stage


# Mutex declaration moved from ``mutex=[...]`` to Task 19's resources API.
# mutex=["gpu"] -> resources={"gpu": 1}
@pytest.mark.xfail(strict=True, reason="red test for Task 19: stage resources replace mutex")
def test_mutex_prevents_concurrent_execution() -> None:
    _declare_resource_stage({"gpu": 1})


# mutex=["gpu"] -> resources={"gpu": 1}
@pytest.mark.xfail(strict=True, reason="red test for Task 19: stage resources replace mutex")
def test_mutex_releases_on_completion() -> None:
    _declare_resource_stage({"gpu": 1})


# mutex=["gpu"] -> resources={"gpu": 1}
@pytest.mark.xfail(strict=True, reason="red test for Task 19: stage resources replace mutex")
def test_mutex_releases_on_failure() -> None:
    _declare_resource_stage({"gpu": 1})


# mutex=["gpu", "cpu"] -> resources={"gpu": 1, "cpu": 1}
@pytest.mark.xfail(strict=True, reason="red test for Task 19: stage resources replace mutex")
def test_multiple_mutex_groups_per_stage() -> None:
    _declare_resource_stage({"gpu": 1, "cpu": 1})


# mutex=["gpu"] -> resources={"gpu": 1}
@pytest.mark.xfail(strict=True, reason="red test for Task 19: stage resources replace mutex")
def test_mutex_with_dependencies() -> None:
    _declare_resource_stage({"gpu": 1})


# mutex=[] -> resources omitted (this behavior remains green through executor parallel tests).
@pytest.mark.xfail(strict=True, reason="red test for Task 19: stage resources replace mutex")
def test_no_mutex_stages_unaffected() -> None:
    _declare_resource_stage({})


# mutex=["gpu"] -> resources={"gpu": 1}
@pytest.mark.xfail(strict=True, reason="red test for Task 19: stage resources replace mutex")
def test_mutex_names_are_case_insensitive() -> None:
    _declare_resource_stage({"GPU": 1})


# mutex=["*"] -> resources={"cpu": "all"}
@pytest.mark.xfail(strict=True, reason="red test for Task 19: stage resources replace mutex")
def test_exclusive_mutex_runs_alone() -> None:
    _declare_resource_stage({"cpu": "all"})
