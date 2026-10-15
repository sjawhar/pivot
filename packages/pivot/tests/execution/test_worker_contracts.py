"""Worker behaviors a stage author or the coordinator relies on, through execute_stage and helpers.

Covers what reaches the output queue (logs, stdout/stderr, state changes), the reasons a run or
skip reports, params overrides, artifact locking, deferred StateDB writes, seeding and nested
parallelism, and the value plumbing between the store and the stage function.
"""

from __future__ import annotations

import array
import contextlib
import dataclasses
import fcntl
import io
import logging
import os
import pathlib
import queue
import random
import re
import signal
import subprocess
import sys
import termios
import textwrap
import threading
import time
from typing import TYPE_CHECKING, cast, override

import joblib.parallel
import numpy
import pytest

from pivot import exceptions, loaders, stage_def, types
from pivot import project as pivot_project
from pivot.executor import worker
from pivot.storage import artifact_lock, cache, lock, state
from pivot.storage import store as store_mod
from pivot.types import (
    FileHash,
    LockData,
    LogMessage,
    OutputMessage,
    OutputMessageKind,
    StageResult,
    StageStatus,
)

if TYPE_CHECKING:
    import multiprocessing as mp
    from collections.abc import Callable

_STAGE = "skip/consume"


def _ref(
    producer: str,
    key: str | None = None,
    *,
    tag: types.ArtifactTag = types.ArtifactTag.DATA,
    loader: loaders.Reader[object]
    | loaders.Writer[object]
    | loaders.Loader[object, object]
    | None = None,
) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(producer=producer, key=key),
        format=loaders.Text() if loader is None else loader,
        python_type=str,
        tag=tag,
    )


def _stage_info(
    func: Callable[..., object],
    project: pathlib.Path,
    worker_env: pathlib.Path,
    *,
    deps: dict[str, types.ArtifactRef] | None = None,
    outs: list[types.ArtifactRef] | None = None,
    bindings: dict[str, str] | None = None,
    no_commit: bool = True,
    force: bool = False,
) -> worker.WorkerStageInfo:
    return worker.WorkerStageInfo(
        func=func,
        fingerprint={"self:consume": "0123456789abcdef"},
        deps={"data": _ref("raw", tag=types.ArtifactTag.INPUT)} if deps is None else deps,
        outs=[_ref(_STAGE)] if outs is None else outs,
        store_spec=cast(
            "store_mod.StoreSpec",
            cast(
                "object",
                {
                    "kind": "workspace",
                    "cache_dir": str(worker_env / "files"),
                    "project_root": str(project),
                    "pipeline_name": "skip",
                    "input_bindings": {"raw": "raw.txt"} if bindings is None else bindings,
                    "checkout_modes": cache.DEFAULT_CHECKOUT_MODE_ORDER.copy(),
                },
            ),
        ),
        signature=None,
        params=None,
        variant=None,
        overrides={},
        run_id="test_run",
        force=force,
        no_commit=no_commit,
        params_arg_name=None,
        project_root=project,
        state_dir=project / ".pivot",
        collection_params={},
        incremental_inputs={},
    )


def _drain(output_queue: mp.Queue[OutputMessage]) -> list[OutputMessage]:
    messages = list[OutputMessage]()
    while True:
        try:
            messages.append(output_queue.get(timeout=0.5))
        except queue.Empty:
            return messages


def _log_lines(messages: list[OutputMessage]) -> list[tuple[str, str, bool]]:
    return [
        (message["stage"], message["line"], message["is_stderr"])
        for message in messages
        if message is not None and message["kind"] == OutputMessageKind.LOG
    ]


def _output_path(project: pathlib.Path) -> pathlib.Path:
    return project / "data" / f"{_STAGE}.txt"


@pytest.fixture
def project(worker_env: pathlib.Path, tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "raw.txt").write_text("hello")
    return tmp_path


def _upper(data: str) -> str:
    return data.upper()


# =============================================================================
# What reaches the output queue
# =============================================================================


def test_logging_goes_to_queue_and_non_console_handlers_stay(
    output_queue: mp.Queue[OutputMessage],
) -> None:
    root = logging.getLogger()
    console = logging.StreamHandler(sys.stderr)
    to_file = logging.StreamHandler(io.StringIO())
    root.addHandler(console)
    root.addHandler(to_file)
    try:
        with worker._queue_logging(_STAGE, output_queue):
            assert console not in root.handlers, "console output would corrupt the TUI"
            assert to_file in root.handlers
            logging.getLogger("pivot.test").warning("disk %s", "full")
        assert console in root.handlers, "the console handler comes back afterwards"
    finally:
        root.removeHandler(console)
        root.removeHandler(to_file)

    assert _drain(output_queue) == [
        LogMessage(
            kind=OutputMessageKind.LOG, stage=_STAGE, line="WARNING: disk full", is_stderr=True
        )
    ]


def _prints(data: str) -> str:
    print(f"out {data}")
    print("err", file=sys.stderr)
    return data


def test_stage_stdout_and_stderr_are_captured_apart(
    project: pathlib.Path, worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    result = worker.execute_stage(
        _STAGE, _stage_info(_prints, project, worker_env), worker_env, output_queue
    )

    assert result["status"] == StageStatus.RAN, result["reason"]
    assert result["output_lines"] == [("out hello", False), ("err", True)]
    assert _log_lines(_drain(output_queue)) == [
        (_STAGE, "out hello", False),
        (_STAGE, "err", True),
    ]


def _returns_value(data: str) -> str:
    return data


def _returns_none(data: str) -> None:
    _ = data


@pytest.mark.parametrize(
    ("func", "warned"),
    [pytest.param(_returns_value, True, id="value"), pytest.param(_returns_none, False, id="none")],
)
def test_stage_without_outputs_warns_only_when_it_discards_a_value(
    project: pathlib.Path,
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    func: Callable[[str], object],
    warned: bool,
) -> None:
    info = _stage_info(func, project, worker_env, outs=[])

    result = worker.execute_stage(_STAGE, info, worker_env, output_queue)

    assert result["status"] == StageStatus.RAN, result["reason"]
    warning = f"WARNING: Stage '{_STAGE}' returned value but has no Out annotation - discarding"
    assert ((_STAGE, warning, True) in _log_lines(_drain(output_queue))) is warned


def test_worker_waits_for_artifact_lock_the_coordinator_holds(
    project: pathlib.Path, worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    """The coordinator locks outputs under .pivot/locks (engine.py:1627); the worker must too."""
    info = _stage_info(_upper, project, worker_env)
    held = artifact_lock.LocalFlockLockService(project / ".pivot" / "locks").acquire_many(
        [artifact_lock.LockRequest(key=_STAGE, mode=artifact_lock.LockMode.WRITE)]
    )
    results = list[StageResult]()
    runner = threading.Thread(
        target=lambda: results.append(worker.execute_stage(_STAGE, info, worker_env, output_queue))
    )
    runner.start()
    states = list[str]()
    deadline = time.monotonic() + 10
    try:
        while "waiting_on_lock" not in states and time.monotonic() < deadline:
            try:
                message = output_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if message is not None and message["kind"] == OutputMessageKind.STATE:
                states.append(message["state"])
        assert "waiting_on_lock" in states, "the worker must block on the held output lock"
        assert not _output_path(project).exists(), "nothing is written while the lock is held"
    finally:
        held.release()
        runner.join(timeout=30)
    states.extend(
        message["state"]
        for message in _drain(output_queue)
        if message is not None and message["kind"] == OutputMessageKind.STATE
    )

    assert results[0]["status"] == StageStatus.RAN, results[0]["reason"]
    assert "running" in states[states.index("waiting_on_lock") :]


# =============================================================================
# Params
# =============================================================================


class _FactorParams(stage_def.StageParams):
    factor: int = 2


def _describe_factor(data: str, params: _FactorParams) -> str:
    return f"{data}:{params.factor}"


def test_params_yaml_override_reaches_the_stage(
    project: pathlib.Path, worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    info = _stage_info(_describe_factor, project, worker_env)
    info["params"] = _FactorParams()
    info["params_arg_name"] = "params"
    info["overrides"] = {_STAGE: {"factor": 5}}

    result = worker.execute_stage(_STAGE, info, worker_env, output_queue)

    assert result["status"] == StageStatus.RAN, result["reason"]
    assert _output_path(project).read_text() == "hello:5"


def test_invalid_params_yaml_override_fails_the_stage(
    project: pathlib.Path, worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    info = _stage_info(_describe_factor, project, worker_env)
    info["params"] = _FactorParams()
    info["params_arg_name"] = "params"
    info["overrides"] = {_STAGE: {"factor": "many"}}

    result = worker.execute_stage(_STAGE, info, worker_env, output_queue)

    assert result["status"] == StageStatus.FAILED
    assert "params.yaml" in result["reason"], "the reason must point at the file to fix"


def _reports_project_root(data: str) -> str:
    _ = data
    return str(pivot_project.get_project_root())


def test_stage_sees_its_own_project_root_inside_another_project(
    tmp_path: pathlib.Path,
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reused worker may hold a previous project's root; walking up would find the outer one."""
    inner = tmp_path / "inner"
    (inner / ".pivot").mkdir(parents=True)
    (inner / "raw.txt").write_text("hello")
    monkeypatch.setattr(pivot_project, "_project_root_cache", tmp_path)

    result = worker.execute_stage(
        _STAGE, _stage_info(_reports_project_root, inner, worker_env), worker_env, output_queue
    )

    assert result["status"] == StageStatus.RAN, result["reason"]
    assert _output_path(inner).read_text() == str(inner)


# =============================================================================
# Reasons a run or skip reports
# =============================================================================


def _run_and_apply(
    info: worker.WorkerStageInfo, worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> StageResult:
    """Run the stage and apply its deferred writes, as the coordinator does."""
    result = worker.execute_stage(_STAGE, info, worker_env, output_queue)
    if "deferred_writes" in result:
        with state.StateDB(info["state_dir"]) as db:
            db.apply_deferred_writes(
                _STAGE,
                [types.identity_key(out.identity) for out in info["outs"]],
                result["deferred_writes"],
            )
    return result


def _record_dep_generation(project: pathlib.Path) -> None:
    """Record a generation for the dependency, as the coordinator does for a produced artifact."""
    with state.StateDB(project / ".pivot") as db:
        db.record_dep_generations(_STAGE, {"raw": db.increment_generation("raw")})


def _evict_output(project: pathlib.Path, worker_env: pathlib.Path) -> None:
    """Delete the output from the workspace and from the cache."""
    lock_data = lock.StageLock(_STAGE, lock.get_stages_dir(project / ".pivot")).read()
    assert lock_data is not None
    output_hash = lock_data["output_hashes"][types.ArtifactIdentity(producer=_STAGE, key=None)]
    _output_path(project).unlink()
    cache.get_cache_path(worker_env / "files", output_hash["hash"]).unlink()


def test_generation_skip_reports_the_run_cache_key(
    project: pathlib.Path, worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    info = _stage_info(_upper, project, worker_env, no_commit=False)
    first = _run_and_apply(info, worker_env, output_queue)
    _record_dep_generation(project)

    second = _run_and_apply(info, worker_env, output_queue)

    assert first["status"] == StageStatus.RAN, first["reason"]
    assert (second["status"], second["reason"]) == (StageStatus.CACHED, "unchanged (generation)")
    assert second["input_hash"] is not None
    assert second["input_hash"] == first["input_hash"]


def test_forced_run_of_unchanged_stage_reports_forced(
    project: pathlib.Path, worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    _run_and_apply(
        _stage_info(_upper, project, worker_env, no_commit=False), worker_env, output_queue
    )

    forced = worker.execute_stage(
        _STAGE, _stage_info(_upper, project, worker_env, force=True), worker_env, output_queue
    )

    assert (forced["status"], forced["reason"]) == (StageStatus.RAN, "forced")


def test_run_cache_restores_an_earlier_input_version(
    project: pathlib.Path, worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    raw = project / "raw.txt"
    commit = _stage_info(_upper, project, worker_env, no_commit=False)
    for content in ("v1", "v2"):
        raw.write_text(content)
        assert _run_and_apply(commit, worker_env, output_queue)["status"] == StageStatus.RAN

    raw.write_text("v1")
    committed = _run_and_apply(commit, worker_env, output_queue)
    assert (committed["status"], committed["reason"]) == (
        StageStatus.CACHED,
        "unchanged (run cache)",
    )
    assert _output_path(project).read_text() == "V1"

    raw.write_text("v2")
    uncommitted = worker.execute_stage(
        _STAGE, _stage_info(_upper, project, worker_env), worker_env, output_queue
    )
    assert (uncommitted["status"], uncommitted["reason"]) == (
        StageStatus.CACHED,
        "unchanged (run cache)",
    )
    assert _output_path(project).read_text() == "V2"


@pytest.mark.parametrize(
    "generations",
    [
        pytest.param(True, id="generation-tier-passes"),
        pytest.param(False, id="lock-comparison-passes"),
    ],
)
def test_unchanged_stage_with_unrestorable_output_reruns(
    project: pathlib.Path,
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    generations: bool,
) -> None:
    info = _stage_info(_upper, project, worker_env, no_commit=False)
    _run_and_apply(info, worker_env, output_queue)
    if generations:
        _record_dep_generation(project)
    _evict_output(project, worker_env)

    result = worker.execute_stage(_STAGE, info, worker_env, output_queue)

    assert (result["status"], result["reason"]) == (StageStatus.RAN, "outputs missing from cache")
    assert _output_path(project).read_text() == "HELLO"


def test_missing_and_unreadable_dependencies_are_named(
    project: pathlib.Path, worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    deps = {
        "a": _ref("raw", tag=types.ArtifactTag.INPUT),
        "b": _ref("other", tag=types.ArtifactTag.INPUT),
    }
    bindings = {"raw": "raw.txt", "other": "other.txt"}
    (project / "raw.txt").unlink()

    missing = worker.execute_stage(
        _STAGE,
        _stage_info(_upper, project, worker_env, deps=deps, bindings=bindings),
        worker_env,
        output_queue,
    )

    (project / "raw.txt").write_text("a")
    (project / "other.txt").write_text("b")
    for name in ("raw.txt", "other.txt"):
        (project / name).chmod(0)
    try:
        unreadable = worker.execute_stage(
            _STAGE,
            _stage_info(_upper, project, worker_env, deps=deps, bindings=bindings),
            worker_env,
            output_queue,
        )
    finally:
        for name in ("raw.txt", "other.txt"):
            (project / name).chmod(0o644)

    assert (missing["status"], missing["reason"]) == (
        StageStatus.FAILED,
        "missing deps: raw, other",
    )
    assert (unreadable["status"], unreadable["reason"]) == (
        StageStatus.FAILED,
        "unreadable deps: raw, other",
    )


def _exits(data: str) -> str:
    sys.exit(3)


def _interrupted(data: str) -> str:
    raise KeyboardInterrupt


@pytest.mark.parametrize(
    ("func", "reason"),
    [
        pytest.param(_exits, "Stage called sys.exit(3)", id="sys-exit"),
        pytest.param(_interrupted, "KeyboardInterrupt", id="keyboard-interrupt"),
    ],
)
def test_stage_exit_is_a_failure_with_a_reason(
    project: pathlib.Path,
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    func: Callable[[str], str],
    reason: str,
) -> None:
    result = worker.execute_stage(
        _STAGE, _stage_info(func, project, worker_env), worker_env, output_queue
    )

    assert (result["status"], result["reason"]) == (StageStatus.FAILED, reason)


def _returns_empty_dict(data: str) -> dict[str, str]:
    _ = data
    return {}


def _returns_int_key(data: str) -> dict[int, str]:
    return {1: data}


_DIRECTORY_OUT = _ref(_STAGE, tag=types.ArtifactTag.DIRECTORY)


@pytest.mark.parametrize(
    ("func", "outs", "cause"),
    [
        pytest.param(_returns_none, [_ref(_STAGE)], "returned None", id="none-for-an-output"),
        pytest.param(
            _returns_value,
            [_ref(_STAGE, "a"), _ref(_STAGE, "b")],
            "mapping",
            id="str-for-several-outputs",
        ),
        pytest.param(_returns_value, [_DIRECTORY_OUT], "dict", id="str-for-a-directory"),
        pytest.param(_returns_empty_dict, [_DIRECTORY_OUT], "non-empty", id="empty-directory"),
        pytest.param(_returns_int_key, [_DIRECTORY_OUT], "strings", id="non-string-member-name"),
    ],
)
def test_return_value_that_does_not_fit_the_outputs_fails_with_the_cause(
    project: pathlib.Path,
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    func: Callable[[str], object],
    outs: list[types.ArtifactRef],
    cause: str,
) -> None:
    result = worker.execute_stage(
        _STAGE, _stage_info(func, project, worker_env, outs=outs), worker_env, output_queue
    )

    assert result["status"] == StageStatus.FAILED
    error = result["reason"].splitlines()[-1]
    assert _STAGE in error, "the error must name the stage"
    assert cause in error, "the error must say what was wrong with the return value"
    assert not _output_path(project).exists()


@dataclasses.dataclass(frozen=True)
class _WritesNothing(loaders.Writer[object]):
    """A writer with a bug: it never creates the file it is given."""

    @override
    def save(self, data: object, path: pathlib.Path) -> None:
        _ = data, path


@pytest.mark.parametrize(
    "no_commit", [pytest.param(True, id="no-commit"), pytest.param(False, id="commit")]
)
def test_output_the_writer_never_created_fails_the_stage(
    project: pathlib.Path,
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    no_commit: bool,
) -> None:
    info = _stage_info(
        _upper,
        project,
        worker_env,
        outs=[_ref(_STAGE, loader=_WritesNothing())],
        no_commit=no_commit,
    )

    result = worker.execute_stage(_STAGE, info, worker_env, output_queue)

    expected = project.resolve() / "data" / f"{_STAGE}.dat"
    assert (result["status"], result["reason"]) == (
        StageStatus.FAILED,
        f"Stage did not produce output: {expected}",
    )


def test_unreadable_state_database_fails_the_stage_without_a_run_cache_key(
    project: pathlib.Path, worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    """A failure before dependency hashing reports input_hash None (types.py StageResult)."""
    (project / ".pivot" / "state.lmdb").write_text("not a database")

    result = worker.execute_stage(
        _STAGE, _stage_info(_upper, project, worker_env), worker_env, output_queue
    )

    assert result["status"] == StageStatus.FAILED
    assert result["reason"].splitlines()[-1].startswith("lmdb.Error: ")
    assert result["input_hash"] is None
    assert not _output_path(project).exists()


def test_worker_defers_file_hash_cache_writes_to_the_coordinator(
    project: pathlib.Path, worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    """Workers open StateDB read-only; the coordinator applies their file-hash entries.

    Entries cover the dependency the worker hashed and the output it committed.
    """
    raw = project.resolve() / "raw.txt"

    result = worker.execute_stage(
        _STAGE, _stage_info(_upper, project, worker_env, no_commit=False), worker_env, output_queue
    )

    assert result["status"] == StageStatus.RAN, result["reason"]
    with state.StateDB(project / ".pivot") as db:
        assert db.get(raw, raw.stat()) is None, "the worker must not write to StateDB"
    assert "deferred_writes" in result
    deferred = result["deferred_writes"]
    assert "file_hash_entries" in deferred
    output = project.resolve() / "data" / "skip" / "consume.txt"
    assert [entry[0] for entry in deferred["file_hash_entries"]] == [str(raw), str(output)]


# =============================================================================
# Determinism and nested parallelism
# =============================================================================


def test_stage_randomness_is_seeded_with_zero() -> None:
    """Same code and inputs give the same outputs across runs and across Pivot versions."""
    saved_random = random.getstate()
    saved_numpy = numpy.random.get_state()
    try:
        random.seed(0)
        numpy.random.seed(0)
        expected = (random.random(), numpy.random.random())
        random.seed(12345)
        numpy.random.seed(12345)

        worker._set_deterministic_seeds()

        assert (random.random(), numpy.random.random()) == expected
    finally:
        random.setstate(saved_random)
        numpy.random.set_state(saved_numpy)


def _active_backend() -> str:
    return type(joblib.parallel.get_active_backend()[0]).__name__


@pytest.mark.parametrize(
    ("setting", "backend"),
    [
        pytest.param(None, "ThreadingBackend", id="default"),
        pytest.param("processes", "LokyBackend", id="processes"),
        pytest.param("threads", "ThreadingBackend", id="other-value"),
    ],
)
def test_nested_parallelism_backend_follows_the_environment(
    monkeypatch: pytest.MonkeyPatch, setting: str | None, backend: str
) -> None:
    if setting is None:
        monkeypatch.delenv("PIVOT_NESTED_PARALLELISM", raising=False)
    else:
        monkeypatch.setenv("PIVOT_NESTED_PARALLELISM", setting)

    assert worker._execute_with_joblib_protection(_active_backend, {}) == backend


def _add(a: int, b: int) -> int:
    return a + b


def test_stage_runs_unprotected_when_joblib_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """joblib is a dev dependency only (packages/pivot/pyproject.toml), so users may lack it."""
    monkeypatch.setitem(sys.modules, "joblib", None)

    assert worker._execute_with_joblib_protection(_add, {"a": 2, "b": 3}) == 5


# =============================================================================
# Values between the store and the stage
# =============================================================================


def _join_group(group: dict[str, str]) -> str:
    return ",".join(f"{key}={value}" for key, value in sorted(group.items()))


def test_directory_dependency_loads_nested_files_by_relative_path(
    project: pathlib.Path, worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    group = project / "data" / "upstream"
    (group / "sub").mkdir(parents=True)
    (group / "a.txt").write_text("alpha")
    (group / "sub" / "b.txt").write_text("beta")
    info = _stage_info(
        _join_group,
        project,
        worker_env,
        deps={"group": _ref("upstream", tag=types.ArtifactTag.DIRECTORY)},
    )

    result = worker.execute_stage(_STAGE, info, worker_env, output_queue)

    assert result["status"] == StageStatus.RAN, result["reason"]
    assert _output_path(project).read_text() == "a.txt=alpha,sub/b.txt=beta"


def _deep_directory(data: str) -> dict[str, str]:
    return {"x/y/z.txt": data}


def test_directory_output_creates_nested_member_directories(
    project: pathlib.Path, worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    info = _stage_info(
        _deep_directory,
        project,
        worker_env,
        outs=[_ref(_STAGE, tag=types.ArtifactTag.DIRECTORY)],
    )

    result = worker.execute_stage(_STAGE, info, worker_env, output_queue)

    assert result["status"] == StageStatus.RAN, result["reason"]
    assert (project / "data" / _STAGE / "x" / "y" / "z.txt").read_text() == "hello"


def test_every_missing_output_key_is_reported() -> None:
    outs = [_ref(_STAGE, "a"), _ref(_STAGE, "b"), _ref(_STAGE, "c")]

    with pytest.raises(KeyError) as exc_info:
        worker._resolve_output_values({"b": 1}, outs, _STAGE)

    message = str(exc_info.value)
    assert "'a'" in message and "'c'" in message, "every missing key must be named"


def test_undeclared_return_keys_are_warned_and_dropped(caplog: pytest.LogCaptureFixture) -> None:
    outs = [_ref(_STAGE, "a"), _ref(_STAGE, "b")]

    with caplog.at_level(logging.WARNING, logger="pivot.executor.worker"):
        values = worker._resolve_output_values({"a": 1, "b": 2, "extra": 3}, outs, _STAGE)

    assert {ref.identity.key: value for ref, value in values.items()} == {"a": 1, "b": 2}
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "extra" in caplog.records[0].getMessage(), "the warning must name the dropped key"


def _path_ref() -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(producer=_STAGE, key=None),
        format=loaders.PathOnly(),
        python_type=pathlib.Path,
        tag=types.ArtifactTag.DATA,
    )


def test_path_output_must_be_a_path(tmp_path: pathlib.Path) -> None:
    with pytest.raises(
        TypeError,
        match=re.escape(
            f"Stage '{_STAGE}' returned str; a PathOnly output must return pathlib.Path"
        ),
    ):
        worker._reject_path_inside_project(_path_ref(), "not a path", tmp_path)


def test_path_output_checks_dependencies_through_symlinks(tmp_path: pathlib.Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    produced = external / "result.bin"
    produced.write_bytes(b"result")
    unrelated = tmp_path / "unrelated.bin"
    unrelated.write_bytes(b"input")
    link = tmp_path / "link"
    link.symlink_to(external, target_is_directory=True)

    worker._reject_path_inside_project(_path_ref(), produced, project_root, [unrelated])
    with pytest.raises(exceptions.StageDefinitionError) as exc_info:
        worker._reject_path_inside_project(_path_ref(), produced, project_root, [link])
    message = str(exc_info.value)
    assert _STAGE in message
    assert str(produced) in message and str(link) in message, "both paths must be named"
    assert produced.read_bytes() == b"result"


def test_path_output_inside_the_project_tells_the_author_where_to_write(
    tmp_path: pathlib.Path,
) -> None:
    produced = tmp_path / "result.bin"
    produced.write_bytes(b"result")

    with pytest.raises(exceptions.StageDefinitionError) as exc_info:
        worker._reject_path_inside_project(_path_ref(), produced, tmp_path)

    message = str(exc_info.value)
    assert _STAGE in message and str(produced) in message
    assert "tempfile.mkdtemp()" in message, "the error must say where to write instead"
    assert produced.read_bytes() == b"result"


def test_list_and_tuple_dependencies_are_rebuilt_in_index_order() -> None:
    kwargs: dict[str, object] = {
        "x": 1,
        "items[2]": "a",
        "items[10]": "m",
        "items[0]": "z",
        "p[1]": "q",
        "p[0]": "p",
    }
    collection_params = {"items": "list", "p": "tuple", "no_list": "list", "no_tuple": "tuple"}

    rebuilt = worker._reconstruct_list_kwargs(kwargs, collection_params)

    assert rebuilt == {
        "x": 1,
        "items": ["z", "a", "m"],
        "p": ("p", "q"),
        "no_list": [],
        "no_tuple": (),
    }
    assert type(rebuilt["p"]) is tuple
    assert type(rebuilt["no_tuple"]) is tuple
    assert type(rebuilt["items"]) is list
    assert type(rebuilt["no_list"]) is list


def _incremental(key: str, loader: loaders.Loader[object, object]) -> types.ArtifactRef:
    return _ref(_STAGE, key, tag=types.ArtifactTag.INCREMENTAL, loader=loader)


def test_first_run_gives_every_incremental_input_its_empty_value(tmp_path: pathlib.Path) -> None:
    store = store_mod.WorkspaceStore(tmp_path, "skip", {}, cache_dir=tmp_path / "cache")
    first = _incremental(
        "a", cast("loaders.Loader[object, object]", loaders.JSON[dict[str, int]]())
    )
    second = _incremental("b", cast("loaders.Loader[object, object]", loaders.Text()))

    values = worker._prepare_incremental_inputs(
        _STAGE,
        [first, second],
        {"prev_a": first.identity, "prev_b": second.identity},
        None,
        store,
        tmp_path / ".pivot",
    )

    assert values == {"prev_a": {}, "prev_b": ""}


@pytest.mark.parametrize(
    "recorded",
    [pytest.param(False, id="not-in-the-lock"), pytest.param(True, id="not-in-the-cache")],
)
def test_incremental_input_that_cannot_be_restored_names_the_fix(
    tmp_path: pathlib.Path, recorded: bool
) -> None:
    store = store_mod.WorkspaceStore(tmp_path, "skip", {}, cache_dir=tmp_path / "cache")
    ref = _incremental("a", cast("loaders.Loader[object, object]", loaders.Text()))
    state_dir = tmp_path / ".pivot"
    lock_path = lock.StageLock(_STAGE, lock.get_stages_dir(state_dir)).path
    output_hashes = {ref.identity: FileHash(hash="0123456789abcdef")} if recorded else {}
    lock_data = LockData(code_manifest={}, params={}, dep_hashes={}, output_hashes=output_hashes)

    with pytest.raises(exceptions.CacheRestoreError) as exc_info:
        worker._prepare_incremental_inputs(
            _STAGE, [ref], {"prev": ref.identity}, lock_data, store, state_dir
        )

    assert str(exc_info.value) == (
        f"Cache missing for incremental output '{_STAGE}:a'. Run 'pivot pull' to restore it, "
        + f"or remove '{lock_path}' and rerun to reset this incremental output."
    )


# =============================================================================
# Output capture primitives
# =============================================================================


def test_ring_buffer_counts_every_dropped_line() -> None:
    ring = worker._OutputRingBuffer(max_lines=2)
    for index in range(5):
        ring.append(str(index), False)

    assert ring.snapshot() == [("[3 earlier lines truncated]", False), ("3", False), ("4", False)]


def test_queue_writer_joins_partial_writes_and_restarts_after_flush(
    output_queue: mp.Queue[OutputMessage],
) -> None:
    ring = worker._OutputRingBuffer()
    with worker._QueueWriter(_STAGE, output_queue, is_stderr=False, ring_buffer=ring) as writer:
        writer.write("ab")
        writer.write("c\n")
        writer.write("partial")
        writer.flush()
        writer.write("next\n")

    assert ring.snapshot() == [("abc", False), ("partial", False), ("next", False)]
    assert _log_lines(_drain(output_queue)) == [
        (_STAGE, "abc", False),
        (_STAGE, "partial", False),
        (_STAGE, "next", False),
    ]


def test_queue_writer_reads_subprocess_bytes_until_end_of_file(
    output_queue: mp.Queue[OutputMessage],
) -> None:
    """Invalid UTF-8 is replaced, and closing the writer does not wait on its reader thread."""
    ring = worker._OutputRingBuffer()
    writer = worker._QueueWriter(_STAGE, output_queue, is_stderr=False, ring_buffer=ring)
    writer.__enter__()
    try:
        subprocess.run(
            ["/bin/sh", "-c", r"printf 'ok\n\377\n'"], stdout=writer.fileno(), check=True
        )
    finally:
        started = time.monotonic()
        writer.__exit__(None, None, None)
        elapsed = time.monotonic() - started

    assert ring.snapshot() == [("ok", False), ("\ufffd", False)]
    assert elapsed < 4.0, "the reader thread must stop at end of file"


def test_queue_writer_keeps_a_character_split_across_pipe_reads(
    output_queue: mp.Queue[OutputMessage],
) -> None:
    """A subprocess's non-ASCII output must reach the log intact, wherever the pipe splits it."""
    euro = "€".encode()
    ring = worker._OutputRingBuffer()
    writer = worker._QueueWriter(_STAGE, output_queue, is_stderr=False, ring_buffer=ring)
    writer.__enter__()
    try:
        write_fd = writer.fileno()
        _ = os.write(write_fd, b"abc" + euro[:1])
        # Second write only after the reader has taken the first one, so the split is a real
        # read boundary rather than whatever the scheduler happens to produce.
        unread = array.array("i", [1])
        deadline = time.monotonic() + 10
        while unread[0]:
            assert time.monotonic() < deadline, "the reader thread never read the first bytes"
            time.sleep(0.01)
            _ = fcntl.ioctl(write_fd, termios.FIONREAD, unread)
        _ = os.write(write_fd, euro[1:] + b"\n")
    finally:
        writer.__exit__(None, None, None)

    assert _log_lines(_drain(output_queue)) == [(_STAGE, "abc€", False)]
    assert ring.snapshot() == [("abc€", False)]


def test_queue_writer_reports_a_truncated_character_at_eof(
    output_queue: mp.Queue[OutputMessage],
) -> None:
    """Output that ends mid-character is still logged (as U+FFFD), not silently dropped."""
    ring = worker._OutputRingBuffer()
    writer = worker._QueueWriter(_STAGE, output_queue, is_stderr=False, ring_buffer=ring)
    writer.__enter__()
    try:
        _ = os.write(writer.fileno(), b"abc" + "€".encode()[:1])
    finally:
        writer.__exit__(None, None, None)

    assert _log_lines(_drain(output_queue)) == [(_STAGE, "abc\ufffd", False)]


_LEAKED_CHILD_SCRIPT = textwrap.dedent(
    """
    import pathlib, queue, subprocess, sys
    from pivot.executor import worker

    writer = worker._QueueWriter(
        "s", queue.Queue(), is_stderr=False, ring_buffer=worker._OutputRingBuffer()
    )
    writer.__enter__()
    child = subprocess.Popen(
        ["sleep", "60"],
        stdin=subprocess.DEVNULL,
        stdout=writer.fileno(),
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pathlib.Path(sys.argv[1]).write_text(str(child.pid))
    writer.__exit__(None, None, None)
    """
)


def test_background_child_holding_stage_stdout_does_not_keep_the_worker_alive(
    tmp_path: pathlib.Path,
) -> None:
    """A stage may leave a process running that inherited its stdout; the worker still exits."""
    pid_file = tmp_path / "child.pid"
    try:
        subprocess.run(
            [sys.executable, "-c", _LEAKED_CHILD_SCRIPT, str(pid_file)],
            check=True,
            capture_output=True,
            timeout=30,
        )
    finally:
        if pid_file.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text()), signal.SIGKILL)


# =============================================================================
# Dependency hashing
# =============================================================================


def test_hash_dependencies_hashes_a_path_list(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("content")

    hashes, missing, unreadable, _ = worker.hash_dependencies([str(path)])

    assert hashes == {str(path): FileHash(hash=cache.hash_file(path, None)[0])}
    assert (missing, unreadable) == ([], [])
