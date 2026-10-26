"""Worker skip primitives against a real WorkspaceStore, StateDB and cache on disk.

Covers the decisions the worker makes before it runs a stage: the generation tier
(``can_skip_via_generation``), verifying and restoring recorded outputs
(``restore_outputs_to_hashes``), dependency hashing, the run-cache key, and the generation bump
a completed run asks the coordinator to apply.
"""

from __future__ import annotations

import os
import typing
from typing import TYPE_CHECKING, cast

import pytest

from pivot import loaders, run_history, types
from pivot.executor import worker
from pivot.storage import cache, state
from pivot.storage import store as store_mod
from pivot.types import DepEntry, DirHash, FileHash, LockData

if TYPE_CHECKING:
    import multiprocessing as mp
    import pathlib
    from collections.abc import Generator

    from pivot.types import HashInfo, OutputMessage

_STAGE = "skip/consume"
_FINGERPRINT = {"self:consume": "0123456789abcdef"}
_OUTPUT_HASH = FileHash(hash="feedfacefeedface")


def _ref(
    producer: str,
    key: str | None = None,
    *,
    tag: types.ArtifactTag = types.ArtifactTag.DATA,
) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(producer=producer, key=key),
        format=loaders.Text(),
        python_type=str,
        tag=tag,
    )


@pytest.fixture
def project(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / ".pivot" / "cache" / "files").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def state_db(project: pathlib.Path) -> Generator[state.StateDB]:
    with state.StateDB(project / ".pivot") as db:
        yield db


def _workspace(project: pathlib.Path, bindings: dict[str, str]) -> store_mod.WorkspaceStore:
    return store_mod.WorkspaceStore(
        project_root=project,
        pipeline_name="skip",
        input_bindings=bindings,
        cache_dir=project / ".pivot" / "cache" / "files",
    )


# =============================================================================
# Generation tier
# =============================================================================


class _GenerationCase:
    """A stage that ran once on ``raw.txt``: lock written, generations recorded, stat cached."""

    store: store_mod.WorkspaceStore
    state_db: state.StateDB
    raw: pathlib.Path
    dep: types.ArtifactRef
    out: types.ArtifactRef
    lock_data: LockData

    def __init__(self, project: pathlib.Path, state_db: state.StateDB) -> None:
        self.store = _workspace(project, {"raw": "raw.txt"})
        self.state_db = state_db
        self.raw = project / "raw.txt"
        self.raw.write_text("input-v1")
        self.dep = _ref("raw", tag=types.ArtifactTag.INPUT)
        self.out = _ref(_STAGE)
        dep_hash, _ = cache.hash_file(self.raw, state_db)
        generation = state_db.increment_generation("raw")
        state_db.record_dep_generations(_STAGE, {"raw": generation})
        self.lock_data = LockData(
            code_manifest=dict(_FINGERPRINT),
            params={},
            dep_hashes={self.dep.identity: FileHash(hash=dep_hash)},
            output_hashes={self.out.identity: _OUTPUT_HASH},
        )

    def can_skip(
        self,
        *,
        fingerprint: dict[str, str] = _FINGERPRINT,
        params: dict[str, object] | None = None,
        outs: list[str] | None = None,
        deps: dict[str, types.ArtifactRef] | None = None,
    ) -> bool:
        return worker.can_skip_via_generation(
            _STAGE,
            fingerprint,
            {"data": self.dep} if deps is None else deps,
            [_STAGE] if outs is None else outs,
            {} if params is None else params,
            self.lock_data,
            self.state_db,
            self.store,
        )


@pytest.fixture
def generation_case(project: pathlib.Path, state_db: state.StateDB) -> _GenerationCase:
    return _GenerationCase(project, state_db)


def test_generation_tier_skips_unchanged_stage(generation_case: _GenerationCase) -> None:
    assert generation_case.can_skip() is True


def test_generation_tier_skips_unchanged_stage_without_dependencies(
    generation_case: _GenerationCase,
) -> None:
    generation_case.lock_data["dep_hashes"] = {}
    assert generation_case.can_skip(deps={}) is True


def test_generation_tier_catches_rewrite_the_stat_cache_cannot_see(
    generation_case: _GenerationCase,
) -> None:
    """Same size, inode and mtime: only the bumped generation shows the producer rewrote it."""
    before = generation_case.raw.stat()
    generation_case.raw.write_text("input-v2")
    os.utime(generation_case.raw, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert generation_case.raw.stat().st_ino == before.st_ino
    generation_case.state_db.increment_generation("raw")

    assert generation_case.can_skip() is False


@pytest.mark.parametrize(
    "change",
    [
        "code",
        "params",
        "outputs",
        "dep-missing-from-lock",
        "dep-locked-as-directory",
        "dep-deleted",
        "dep-content",
        "no-recorded-generations",
    ],
)
def test_generation_tier_defers_to_full_check_when_anything_changed(
    generation_case: _GenerationCase, change: str
) -> None:
    case = generation_case
    match change:
        case "code":
            assert case.can_skip(fingerprint={"self:consume": "changedchangedch"}) is False
        case "params":
            assert case.can_skip(params={"rate": 0.1}) is False
        case "outputs":
            assert case.can_skip(outs=[_STAGE, f"{_STAGE}:extra"]) is False
        case "dep-missing-from-lock":
            case.lock_data["dep_hashes"] = {}
            assert case.can_skip() is False
        case "dep-locked-as-directory":
            case.lock_data["dep_hashes"] = {
                case.dep.identity: DirHash(hash="0123456789abcdef", manifest=[])
            }
            assert case.can_skip() is False
        case "dep-deleted":
            case.raw.unlink()
            assert case.can_skip() is False
        case "dep-content":
            case.raw.write_text("input-v2 with a different size")
            assert case.can_skip() is False
        case "no-recorded-generations":
            assert (
                worker.can_skip_via_generation(
                    "skip/never-ran",
                    _FINGERPRINT,
                    {"data": case.dep},
                    [_STAGE],
                    {},
                    case.lock_data,
                    case.state_db,
                    case.store,
                )
                is False
            )
        case _:
            raise AssertionError(change)


# =============================================================================
# Verifying and restoring recorded outputs
# =============================================================================


class _EvictedCacheStore(store_mod.WorkspaceStore):
    """A workspace whose cache entry disappears between the existence check and the restore."""

    @typing.override
    def restore_from_cache(
        self,
        ref: types.ArtifactRef,
        output_hash: types.HashInfo,
        *,
        state_dir: pathlib.Path | None = None,
        checkout_modes: list[cache.CheckoutMode] | None = None,
    ) -> bool:
        return False


def _restore(
    store: store_mod.Store,
    outs: list[types.ArtifactRef],
    hashes: dict[types.ArtifactIdentity, HashInfo],
    state_db: state.StateDB,
) -> worker.OutputRestoreResult:
    return worker.restore_outputs_to_hashes(store, outs, hashes, state_db, report_changes=True)


def _cached_output(
    project: pathlib.Path, store: store_mod.WorkspaceStore, ref: types.ArtifactRef, content: str
) -> tuple[pathlib.Path, FileHash]:
    """Write an output, commit it to the cache, and return its path and hash."""
    path = store.prepare_output(ref)
    path.write_text(content)
    output_hash = store.commit(ref, path)
    assert not types.is_dir_hash(output_hash)
    return path, output_hash


def test_restore_reports_matching_outputs_available_without_writing(
    project: pathlib.Path, state_db: state.StateDB
) -> None:
    store = _workspace(project, {})
    ref = _ref(_STAGE)
    _, file_hash = _cached_output(project, store, ref, "out")

    result = _restore(store, [ref], {ref.identity: file_hash}, state_db)

    assert result == worker.OutputRestoreResult(
        available=True,
        outputs_restored=False,
        unavailable=[],
        empty_dirs_unrecorded=[],
        to_restore=[],
    )


def test_restore_verifies_directory_output(project: pathlib.Path, state_db: state.StateDB) -> None:
    store = _workspace(project, {})
    ref = _ref(_STAGE, tag=types.ArtifactTag.DIRECTORY)
    directory = store.prepare_output(ref)
    (directory / "a.txt").write_text("a")
    output_hash = cache.hash_directory(directory, None)

    result = _restore(store, [ref], {ref.identity: output_hash}, state_db)

    assert result == worker.OutputRestoreResult(
        available=True,
        outputs_restored=False,
        unavailable=[],
        empty_dirs_unrecorded=[],
        to_restore=[],
    )


def test_restore_brings_back_deleted_output_from_cache(
    project: pathlib.Path, state_db: state.StateDB
) -> None:
    store = _workspace(project, {})
    ref = _ref(_STAGE)
    path, file_hash = _cached_output(project, store, ref, "cached out")
    path.unlink()

    result = _restore(store, [ref], {ref.identity: file_hash}, state_db)

    assert result == worker.OutputRestoreResult(
        available=True,
        outputs_restored=True,
        unavailable=[],
        empty_dirs_unrecorded=[],
        to_restore=[ref],
    )
    assert path.read_text() == "cached out"


def test_restore_reports_output_without_recorded_hash_unavailable(
    project: pathlib.Path, state_db: state.StateDB
) -> None:
    store = _workspace(project, {})
    recorded = _ref(_STAGE, "recorded")
    unrecorded = _ref(_STAGE, "unrecorded")
    _, file_hash = _cached_output(project, store, recorded, "out")

    result = _restore(store, [unrecorded, recorded], {recorded.identity: file_hash}, state_db)

    assert result == worker.OutputRestoreResult(
        available=False,
        outputs_restored=False,
        unavailable=[unrecorded],
        empty_dirs_unrecorded=[],
        to_restore=[],
    )


def test_restore_reports_output_missing_from_cache_unavailable(
    project: pathlib.Path, state_db: state.StateDB
) -> None:
    store = _workspace(project, {})
    ref = _ref(_STAGE)
    path = store.prepare_output(ref)
    path.write_text("never cached")
    file_hash, _ = cache.hash_file(path, None)
    path.unlink()

    result = _restore(store, [ref], {ref.identity: FileHash(hash=file_hash)}, state_db)

    assert result == worker.OutputRestoreResult(
        available=False,
        outputs_restored=False,
        unavailable=[ref],
        empty_dirs_unrecorded=[],
        to_restore=[],
    )
    assert not path.exists()


def test_restore_reports_failed_restore_unavailable(
    project: pathlib.Path, state_db: state.StateDB
) -> None:
    store = _EvictedCacheStore(
        project_root=project,
        pipeline_name="skip",
        input_bindings={},
        cache_dir=project / ".pivot" / "cache" / "files",
    )
    ref = _ref(_STAGE)
    path, file_hash = _cached_output(project, store, ref, "out")
    path.unlink()

    result = _restore(store, [ref], {ref.identity: file_hash}, state_db)

    assert result == worker.OutputRestoreResult(
        available=False,
        outputs_restored=False,
        unavailable=[ref],
        empty_dirs_unrecorded=[],
        to_restore=[ref],
    )


def test_check_lists_every_unavailable_output_and_writes_nothing(
    project: pathlib.Path, state_db: state.StateDB
) -> None:
    """``pivot status`` (explain.py) checks with restore=False and names every unavailable output.

    A cached output counts as available without being restored, and the check names it as one
    a restore would write; a restore stops at the first unavailable output instead.
    """
    store = _workspace(project, {})
    restorable = _ref(_STAGE, "restorable")
    unrecorded = _ref(_STAGE, "unrecorded")
    uncached = _ref(_STAGE, "uncached")
    restorable_path, restorable_hash = _cached_output(project, store, restorable, "cached")
    restorable_path.unlink()
    uncached_path = store.prepare_output(uncached)
    uncached_path.write_text("never cached")
    uncached_hash = FileHash(hash=cache.hash_file(uncached_path, None)[0])
    uncached_path.unlink()
    outs = [unrecorded, restorable, uncached]
    hashes = {restorable.identity: restorable_hash, uncached.identity: uncached_hash}

    checked = worker.restore_outputs_to_hashes(
        store, outs, hashes, state_db, restore=False, report_changes=True
    )
    restored = _restore(store, outs, hashes, state_db)

    assert checked == worker.OutputRestoreResult(
        available=False,
        outputs_restored=False,
        unavailable=[unrecorded, uncached],
        empty_dirs_unrecorded=[],
        to_restore=[restorable],
    )
    assert restored == worker.OutputRestoreResult(
        available=False,
        outputs_restored=False,
        unavailable=[unrecorded],
        empty_dirs_unrecorded=[],
        to_restore=[],
    )
    assert not restorable_path.exists()


def test_restore_keeps_checking_outputs_after_restoring_one(
    project: pathlib.Path, state_db: state.StateDB
) -> None:
    """A restored first output must not hide a later output that cannot be restored."""
    store = _workspace(project, {})
    restorable = _ref(_STAGE, "restorable")
    uncached = _ref(_STAGE, "uncached")
    restorable_path, restorable_hash = _cached_output(project, store, restorable, "cached")
    restorable_path.unlink()
    uncached_path = store.prepare_output(uncached)
    uncached_path.write_text("never cached")
    uncached_hash = FileHash(hash=cache.hash_file(uncached_path, None)[0])
    uncached_path.unlink()

    result = _restore(
        store,
        [restorable, uncached],
        {restorable.identity: restorable_hash, uncached.identity: uncached_hash},
        state_db,
    )

    assert result == worker.OutputRestoreResult(
        available=False,
        outputs_restored=True,
        unavailable=[uncached],
        empty_dirs_unrecorded=[],
        to_restore=[restorable],
    )
    assert restorable_path.read_text() == "cached"


def _old_lock_tree(
    project: pathlib.Path, store: store_mod.WorkspaceStore, ref: types.ArtifactRef
) -> tuple[pathlib.Path, HashInfo]:
    """A cached directory output holding an empty directory, and the hash pivot before 0.3.4
    recorded for it: its files alone, as a copy without the directory hashes."""
    directory = store.prepare_output(ref)
    (directory / "a.txt").write_text("a")
    (directory / "logs").mkdir()
    store.commit(ref, directory)
    files_only = project / "files-only"
    files_only.mkdir()
    (files_only / "a.txt").write_text("a")
    old_hash = cache.hash_directory(files_only, None)
    assert "empty_dirs" not in old_hash
    return directory, old_hash


@pytest.mark.parametrize("restore", [True, False], ids=["restore", "check"])
def test_old_lock_without_empty_dirs_makes_the_stage_run_instead_of_stripping_them(
    project: pathlib.Path, state_db: state.StateDB, restore: bool
) -> None:
    """An output holding every file its lock records plus an empty directory that a lock
    written before pivot 0.3.4 cannot record is not restored to that lock: the restore would
    delete the directory. The output is reported unusable, so the stage runs and records it."""
    store = _workspace(project, {})
    ref = _ref(_STAGE, tag=types.ArtifactTag.DIRECTORY)
    directory, old_hash = _old_lock_tree(project, store, ref)
    assert store.cache_contains(ref, old_hash), "a restore would be possible"

    result = worker.restore_outputs_to_hashes(
        store, [ref], {ref.identity: old_hash}, state_db, restore=restore, report_changes=True
    )

    assert result == worker.OutputRestoreResult(
        available=False,
        outputs_restored=False,
        unavailable=[],
        empty_dirs_unrecorded=[ref],
        to_restore=[],
    )
    assert (directory / "logs").is_dir()


def test_old_lock_still_restores_a_directory_whose_files_changed(
    project: pathlib.Path, state_db: state.StateDB
) -> None:
    """An output whose files differ from its old lock's is restored from the cache as before."""
    store = _workspace(project, {})
    ref = _ref(_STAGE, tag=types.ArtifactTag.DIRECTORY)
    directory, old_hash = _old_lock_tree(project, store, ref)
    (directory / "a.txt").unlink()
    (directory / "a.txt").write_text("edited")

    result = _restore(store, [ref], {ref.identity: old_hash}, state_db)

    assert result == worker.OutputRestoreResult(
        available=True,
        outputs_restored=True,
        unavailable=[],
        empty_dirs_unrecorded=[],
        to_restore=[ref],
    )
    assert (directory / "a.txt").read_text() == "a"


def test_unrecorded_empty_dirs_only_needs_matching_files_and_an_old_hash() -> None:
    manifest = [types.DirManifestEntry(relpath="a.txt", hash="1" * 16, size=1, isexec=False)]
    old = types.DirHash(hash="0" * 16, manifest=manifest)
    with_dirs = types.DirHash(hash="2" * 16, manifest=manifest, empty_dirs=["logs"])
    other_files = types.DirHash(
        hash="3" * 16,
        manifest=[types.DirManifestEntry(relpath="b.txt", hash="1" * 16, size=1, isexec=False)],
        empty_dirs=["logs"],
    )

    assert worker.unrecorded_empty_dirs_only(old, with_dirs)
    assert not worker.unrecorded_empty_dirs_only(with_dirs, old), "a newer lock is authoritative"
    assert not worker.unrecorded_empty_dirs_only(old, other_files), "the files differ"
    assert not worker.unrecorded_empty_dirs_only(FileHash(hash="0" * 16), with_dirs)


def test_restore_on_cache_store_reports_whether_every_output_is_stored(
    project: pathlib.Path, state_db: state.StateDB
) -> None:
    cache_store = store_mod.CacheStore(project / ".pivot" / "cache" / "files", state_dir=None)
    stored = _ref(_STAGE, "stored")
    missing = _ref(_STAGE, "missing")
    source = project / "result.txt"
    source.write_text("stored")
    cache_store.commit(stored, source)

    assert worker.restore_outputs_to_hashes(cache_store, [stored], {}, state_db) is True
    assert worker.restore_outputs_to_hashes(cache_store, [stored, missing], {}, state_db) is False


# =============================================================================
# Dependency hashing
# =============================================================================


def test_hash_dependencies_reports_unreadable_dependency(
    project: pathlib.Path, state_db: state.StateDB
) -> None:
    store = _workspace(project, {"raw": "raw.txt"})
    raw = project / "raw.txt"
    raw.write_text("secret")
    raw.chmod(0)
    try:
        hashes, missing, unreadable, entries = worker.hash_dependencies(
            {"data": _ref("raw", tag=types.ArtifactTag.INPUT)}, store, state_db
        )
    finally:
        raw.chmod(0o644)

    assert (hashes, missing, unreadable, entries) == ({}, [], ["raw"], [])


def test_hash_dependencies_records_stat_write_backs_for_each_file_it_read(
    project: pathlib.Path,
) -> None:
    """A read-only check hands back an entry for each file it read, the files of a directory
    dependency included, so the next check reads none of them; never one for a directory.
    Workers hash with a read-only StateDB, as here."""
    store = _workspace(project, {"raw": "raw.txt", "tree": "tree"})
    (project / "raw.txt").write_text("file")
    (project / "tree").mkdir()
    (project / "tree" / "a.txt").write_text("a")

    with state.StateDB(project / ".pivot", readonly=True) as readonly_db:
        hashes, missing, unreadable, entries = worker.hash_dependencies(
            {
                "data": _ref("raw", tag=types.ArtifactTag.INPUT),
                "tree": _ref("tree", tag=types.ArtifactTag.INPUT),
            },
            store,
            readonly_db,
        )

    assert (missing, unreadable) == ([], [])
    assert types.is_dir_hash(hashes["tree"])
    assert [entry[0] for entry in entries] == [
        str(project / "raw.txt"),
        str(project / "tree" / "a.txt"),
    ], "the file-hash cache learns each file read, never a directory"


# =============================================================================
# Run-cache key and generation bump
# =============================================================================


def _stage_info(
    func: types.StageFunc,
    project: pathlib.Path,
    worker_env: pathlib.Path,
    *,
    deps: dict[str, types.ArtifactRef],
    outs: list[types.ArtifactRef],
    no_commit: bool,
) -> worker.WorkerStageInfo:
    return worker.WorkerStageInfo(
        func=func,
        fingerprint=dict(_FINGERPRINT),
        deps=deps,
        outs=outs,
        store_spec=cast(
            "store_mod.StoreSpec",
            cast(
                "object",
                {
                    "kind": "workspace",
                    "cache_dir": str(worker_env / "files"),
                    "project_root": str(project),
                    "pipeline_name": "skip",
                    "input_bindings": {"raw": "raw.txt"},
                    "checkout_modes": cache.DEFAULT_CHECKOUT_MODE_ORDER.copy(),
                },
            ),
        ),
        signature=None,
        params=None,
        variant=None,
        overrides={},
        run_id="test_run",
        force=False,
        no_commit=no_commit,
        params_arg_name=None,
        project_root=project,
        state_dir=project / ".pivot",
        project_state_dir=project / ".pivot",
        collection_params={},
        incremental_inputs={},
        unsafe_fingerprint=False,
        no_fingerprint=False,
    )


def _upper(data: str) -> str:
    return data.upper()


def test_run_cache_key_marks_metric_outputs_uncacheable(
    project: pathlib.Path, worker_env: pathlib.Path
) -> None:
    """The worker's key equals the one the coordinator (engine.py:1614) and commit build."""
    data_out = _ref(_STAGE, "table")
    metric_out = _ref(_STAGE, "score", tag=types.ArtifactTag.METRIC)
    info = _stage_info(
        _upper,
        project,
        worker_env,
        deps={"data": _ref("raw", tag=types.ArtifactTag.INPUT)},
        outs=[data_out, metric_out],
        no_commit=True,
    )
    dep_hashes: dict[str, HashInfo] = {"raw": FileHash(hash="0123456789abcdef")}

    _, _, input_hash, _ = worker._check_skip_or_run(info, None, dict(_FINGERPRINT), {}, dep_hashes)

    assert input_hash == run_history.compute_input_hash(
        dict(_FINGERPRINT),
        {},
        [DepEntry(producer="raw", key=None, hash="0123456789abcdef")],
        [(f"{_STAGE}:table", True), (f"{_STAGE}:score", False)],
    )


def test_completed_run_asks_coordinator_to_bump_output_generations(
    project: pathlib.Path, worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    """Downstream generation checks depend on every committed run bumping its outputs."""
    (project / "raw.txt").write_text("hello")
    info = _stage_info(
        _upper,
        project,
        worker_env,
        deps={"data": _ref("raw", tag=types.ArtifactTag.INPUT)},
        outs=[_ref(_STAGE)],
        no_commit=False,
    )

    result = worker.execute_stage(_STAGE, info, worker_env, output_queue)

    assert result["status"] == types.StageStatus.RAN, result["reason"]
    assert "deferred_writes" in result
    deferred = result["deferred_writes"]
    assert "increment_outputs" in deferred and deferred["increment_outputs"] is True
