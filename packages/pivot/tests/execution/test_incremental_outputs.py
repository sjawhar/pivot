import collections.abc
import pathlib
import shutil
import stat
import tempfile
from typing import Any

from pivot import loaders, types
from pivot.executor import worker
from pivot.storage import cache, listing, lock


def _incremental_tag() -> types.ArtifactTag:
    tag = getattr(types.ArtifactTag, "INCREMENTAL", None)
    assert tag is not None, "ArtifactTag.INCREMENTAL must identify incremental outputs"
    return tag


def _increment(previous: dict[str, int]) -> dict[str, int]:
    return {"count": previous.get("count", 0) + 1}


_incremental_invocations = list[int]()


def _counting_increment(previous: dict[str, int]) -> dict[str, int]:
    _incremental_invocations.append(1)
    return {"count": previous.get("count", 0) + 1}


def _increment_from_input(data: str, previous: dict[str, int]) -> dict[str, int]:
    _ = data
    return {"count": previous.get("count", 0) + 1}


def _mutate_previous_file(previous: pathlib.Path) -> pathlib.Path:
    assert previous.exists()
    assert not previous.is_symlink()
    previous.write_text(previous.read_text() + " updated")
    result_dir = pathlib.Path(tempfile.mkdtemp())
    result = result_dir / "result.txt"
    result.write_text(previous.read_text())
    return result


def _mutate_previous_directory(previous: pathlib.Path) -> pathlib.Path:
    assert previous.is_dir()
    assert not previous.is_symlink()
    assert (previous / "seed.txt").read_text() == "seed"
    assert (previous / "nested" / "existing.txt").read_text() == "nested seed"
    (previous / "seed.txt").write_text("seed updated")
    (previous / "nested" / "existing.txt").write_text("nested seed updated")
    (previous / "nested" / "next.txt").write_text("next")
    result_dir = pathlib.Path(tempfile.mkdtemp()) / "result"
    shutil.copytree(previous, result_dir)
    return result_dir


def _copy_previous_directory(previous: pathlib.Path) -> pathlib.Path:
    assert previous.is_dir()
    assert (previous / "script.sh").stat().st_mode & stat.S_IXUSR
    result_dir = pathlib.Path(tempfile.mkdtemp()) / "result"
    shutil.copytree(previous, result_dir)
    return result_dir


def _output_ref(
    name: str,
    fmt: loaders.Reader[Any] | loaders.Writer[Any] | loaders.Loader[Any, Any],
) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(producer=f"incremental/{name}", key=None),
        format=fmt,
        python_type=dict,
        tag=_incremental_tag(),
    )


def _worker_info(
    func: collections.abc.Callable[..., Any],
    tmp_path: pathlib.Path,
    output: types.ArtifactRef,
    *,
    deps: dict[str, types.ArtifactRef] | None = None,
    input_bindings: dict[str, str] | None = None,
    force: bool = False,
) -> worker.WorkerStageInfo:
    return worker.WorkerStageInfo(
        func=func,
        fingerprint={"self:test": "abc123"},
        deps=deps or {},
        outs=[output],
        store_spec={
            "kind": "workspace",
            "cache_dir": str(tmp_path / ".pivot" / "cache"),
            "project_root": str(tmp_path),
            "pipeline_name": "incremental",
            "input_bindings": input_bindings or {},
            "checkout_modes": [cache.CheckoutMode.COPY],
        },
        signature=None,
        params=None,
        variant=None,
        overrides={},
        run_id="test-run",
        force=force,
        no_commit=False,
        params_arg_name=None,
        project_root=tmp_path,
        state_dir=tmp_path / ".pivot",
        state_dir_resolution=tmp_path / ".pivot",
        collection_params={},
        incremental_inputs={"previous": output.identity},
        unsafe_fingerprint=False,
        no_fingerprint=False,
        loaded_sources_token="",
    )


def _output_path(tmp_path: pathlib.Path, name: str, suffix: str = "json") -> pathlib.Path:
    return tmp_path / "data" / "incremental" / f"{name}.{suffix}"


def test_incremental_first_run_injects_loader_empty_and_forced_run_restores_previous_copy(
    worker_env: pathlib.Path,
    tmp_path: pathlib.Path,
    output_queue: Any,
) -> None:
    """First execution gets JSON's empty value; a forced rerun receives its cached result."""
    output = _output_ref("cache", loaders.JSON[dict[str, int]]())
    first_info = _worker_info(_increment, tmp_path, output)

    first = worker.execute_stage("incremental/cache", first_info, worker_env, output_queue)

    assert first["status"] is types.StageStatus.RAN
    assert loaders.JSON[dict[str, int]]().load(_output_path(tmp_path, "cache")) == {"count": 1}

    second_info = _worker_info(_increment, tmp_path, output, force=True)
    second = worker.execute_stage("incremental/cache", second_info, worker_env, output_queue)

    assert second["status"] is types.StageStatus.RAN
    assert loaders.JSON[dict[str, int]]().load(_output_path(tmp_path, "cache")) == {"count": 2}


def test_unchanged_incremental_stage_is_cached_and_restores_missing_workspace_output(
    worker_env: pathlib.Path,
    tmp_path: pathlib.Path,
    output_queue: Any,
) -> None:
    """An unchanged incremental stage skips user code and restores its output on demand."""
    _incremental_invocations.clear()
    output = _output_ref("cached", loaders.JSON[dict[str, int]]())
    info = _worker_info(_counting_increment, tmp_path, output)

    first = worker.execute_stage("incremental/cached", info, worker_env, output_queue)
    second = worker.execute_stage("incremental/cached", info, worker_env, output_queue)

    output_path = _output_path(tmp_path, "cached")
    output_path.unlink()
    restored = worker.execute_stage("incremental/cached", info, worker_env, output_queue)

    assert first["status"] is types.StageStatus.RAN
    assert second["status"] is types.StageStatus.CACHED
    assert restored["status"] is types.StageStatus.CACHED
    assert _incremental_invocations == [1]
    assert loaders.JSON[dict[str, int]]().load(output_path) == {"count": 1}


def test_incremental_stage_never_restores_an_older_run_cache_entry(
    worker_env: pathlib.Path,
    tmp_path: pathlib.Path,
    output_queue: Any,
) -> None:
    """Returning an input to an earlier value still advances from the latest incremental cache."""
    input_path = tmp_path / "data" / "raw" / "input.txt"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("one")
    input_ref = types.ArtifactRef(
        identity=types.ArtifactIdentity(producer="input", key=None),
        format=loaders.Text(),
        python_type=str,
        tag=types.ArtifactTag.INPUT,
    )
    output = _output_ref("history", loaders.JSON[dict[str, int]]())
    info = _worker_info(
        _increment_from_input,
        tmp_path,
        output,
        deps={"data": input_ref},
        input_bindings={"input": "data/raw/input.txt"},
    )

    assert (
        worker.execute_stage("incremental/history", info, worker_env, output_queue)["status"]
        is types.StageStatus.RAN
    )
    input_path.write_text("two")
    assert (
        worker.execute_stage("incremental/history", info, worker_env, output_queue)["status"]
        is types.StageStatus.RAN
    )
    input_path.write_text("one")

    third = worker.execute_stage("incremental/history", info, worker_env, output_queue)

    assert third["status"] is types.StageStatus.RAN
    assert loaders.JSON[dict[str, int]]().load(_output_path(tmp_path, "history")) == {"count": 3}


def test_incremental_missing_cache_reports_recovery_steps(
    worker_env: pathlib.Path,
    tmp_path: pathlib.Path,
    output_queue: Any,
) -> None:
    """A lock without its cached incremental artifact fails rather than restarting silently."""
    output = _output_ref("missing", loaders.JSON[dict[str, int]]())
    initial_info = _worker_info(_increment, tmp_path, output)
    initial = worker.execute_stage("incremental/missing", initial_info, worker_env, output_queue)
    assert initial["status"] is types.StageStatus.RAN

    stage_lock = lock.StageLock("incremental/missing", lock.get_stages_dir(tmp_path / ".pivot"))
    lock_data = stage_lock.read()
    assert lock_data is not None
    output_hash = lock_data["output_hashes"][output.identity]
    cache.remove_output(cache.get_cache_path(worker_env, output_hash["hash"]))
    _output_path(tmp_path, "missing").unlink()

    result = worker.execute_stage(
        "incremental/missing",
        _worker_info(_increment, tmp_path, output, force=True),
        worker_env,
        output_queue,
    )

    assert result["status"] is types.StageStatus.FAILED
    assert "Cache missing for incremental output 'incremental/missing'" in result["reason"]
    assert "pivot pull" in result["reason"]
    assert str(stage_lock.path) in result["reason"]


def test_incremental_file_restores_a_writable_copy_without_mutating_the_cache(
    worker_env: pathlib.Path,
    tmp_path: pathlib.Path,
    output_queue: Any,
) -> None:
    """Path-only incremental files can be edited before a stage replaces them."""
    output = _output_ref("file", loaders.PathOnly())
    output_path = _output_path(tmp_path, "file", "dat")
    output_path.parent.mkdir(parents=True)
    output_path.write_text("seed")
    output_hash = cache.save_to_cache(output_path, worker_env)
    cache.remove_output(output_path)
    stage_lock = lock.StageLock("incremental/file", lock.get_stages_dir(tmp_path / ".pivot"))
    stage_lock.write(
        types.LockData(
            code_manifest={"self:test": "abc123"},
            params={},
            dep_hashes={},
            output_hashes={output.identity: output_hash},
        )
    )

    result = worker.execute_stage(
        "incremental/file",
        _worker_info(_mutate_previous_file, tmp_path, output, force=True),
        worker_env,
        output_queue,
    )

    assert result["status"] is types.StageStatus.RAN
    assert output_path.read_text() == "seed updated"
    assert cache.get_cache_path(worker_env, output_hash["hash"]).read_text() == "seed"


def test_incremental_directory_restores_a_writable_copy_without_mutating_the_cache(
    worker_env: pathlib.Path,
    tmp_path: pathlib.Path,
    output_queue: Any,
) -> None:
    """Path-only incremental directories restore as copies, including nested writable paths."""
    output = _output_ref("directory", loaders.PathOnly())
    output_path = _output_path(tmp_path, "directory", "dat")
    output_path.mkdir(parents=True)
    (output_path / "seed.txt").write_text("seed")
    (output_path / "nested").mkdir()
    (output_path / "nested" / "existing.txt").write_text("nested seed")
    output_hash = cache.save_to_cache(output_path, worker_env)
    cache.remove_output(output_path)
    stage_lock = lock.StageLock("incremental/directory", lock.get_stages_dir(tmp_path / ".pivot"))
    stage_lock.write(
        types.LockData(
            code_manifest={"self:test": "abc123"},
            params={},
            dep_hashes={},
            output_hashes={output.identity: output_hash},
        ),
        cache_dir=worker_env,
    )

    result = worker.execute_stage(
        "incremental/directory",
        _worker_info(_mutate_previous_directory, tmp_path, output, force=True),
        worker_env,
        output_queue,
    )

    assert result["status"] is types.StageStatus.RAN
    assert output_path.joinpath("seed.txt").read_text() == "seed updated"
    assert output_path.joinpath("nested", "existing.txt").read_text() == "nested seed updated"
    assert output_path.joinpath("nested", "next.txt").read_text() == "next"
    assert types.is_dir_hash(output_hash)
    members = listing.read(None, output_hash, worker_env)
    cache_files = {
        entry["relpath"]: cache.get_cache_path(worker_env, entry["hash"])
        for entry in members["files"]
    }
    assert cache_files["seed.txt"].read_text() == "seed"
    assert cache_files["nested/existing.txt"].read_text() == "nested seed"
    assert "nested/next.txt" not in cache_files


def test_incremental_directory_copy_restore_preserves_executable_mode(
    worker_env: pathlib.Path,
    tmp_path: pathlib.Path,
    output_queue: Any,
) -> None:
    """COPY restoration preserves executable files before an incremental stage runs."""
    output = _output_ref("executable", loaders.PathOnly())
    output_path = _output_path(tmp_path, "executable", "dat")
    output_path.mkdir(parents=True)
    script = output_path / "script.sh"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    output_hash = cache.save_to_cache(output_path, worker_env)
    cache.remove_output(output_path)
    lock.StageLock("incremental/executable", lock.get_stages_dir(tmp_path / ".pivot")).write(
        types.LockData(
            code_manifest={"self:test": "abc123"},
            params={},
            dep_hashes={},
            output_hashes={output.identity: output_hash},
        ),
        cache_dir=worker_env,
    )

    result = worker.execute_stage(
        "incremental/executable",
        _worker_info(_copy_previous_directory, tmp_path, output, force=True),
        worker_env,
        output_queue,
    )

    assert result["status"] is types.StageStatus.RAN
    assert output_path.joinpath("script.sh").stat().st_mode & stat.S_IXUSR
