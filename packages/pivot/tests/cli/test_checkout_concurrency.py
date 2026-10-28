from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
import pathlib
from typing import TYPE_CHECKING

import pytest

from pivot import project
from pivot.cli import checkout as checkout_mod
from pivot.storage import cache, ownership
from pivot.types import HashInfo

if TYPE_CHECKING:
    from threading import Barrier

_N_FILES = 6
_N_DIRS = 2
_FILE_LINES = 20_000


def _helper_file_content(i: int) -> str:
    return f"file {i}\n" * _FILE_LINES


def _helper_cache_outputs(root: pathlib.Path) -> dict[str, HashInfo]:
    """Cache file and directory outputs that share one parent, then remove them."""
    cache_dir = root / "cache"
    parent = root / "data" / "shared"
    parent.mkdir(parents=True)
    hashes = dict[str, HashInfo]()
    for i in range(_N_FILES):
        path = parent / f"file{i}.txt"
        path.write_text(_helper_file_content(i))
        hashes[str(path)] = cache.save_to_cache(
            path, cache_dir, checkout_mode=cache.CheckoutMode.COPY
        )
    for i in range(_N_DIRS):
        path = parent / f"dir{i}"
        path.mkdir()
        for j in range(10):
            (path / f"f{j}.txt").write_text(f"dir {i} file {j}\n")
        hashes[str(path)] = cache.save_to_cache(
            path, cache_dir, checkout_mode=cache.CheckoutMode.COPY
        )
    for path in hashes:
        cache.remove_output(pathlib.Path(path))
    return hashes


def _helper_own_restore(path: pathlib.Path, output_hash: HashInfo) -> ownership.ForcedRestore:
    """What checkout replaces restoring a tracked path to its own place (``_restore_targets``)."""
    return ownership.ForcedRestore(
        replacement=ownership.tracked_restore(path), restored=output_hash, claims=None
    )


def _helper_checkout_all_at_barrier(
    root: pathlib.Path,
    hashes: dict[str, HashInfo],
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    behavior: checkout_mod.CheckoutBehavior,
    barrier: Barrier,
) -> list[str]:
    """Check out every output from 8 threads, like one `pivot checkout` process."""
    project._project_root_cache = root
    barrier.wait()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(
                checkout_mod._restore_path_sync,
                pathlib.Path(path),
                output_hash,
                cache_dir,
                checkout_modes,
                behavior,
                state_dir,
                target=_helper_own_restore(pathlib.Path(path), output_hash),
            )
            for path, output_hash in hashes.items()
        ]
    failures = list[str]()
    for future in futures:
        exc = future.exception()
        if exc is not None:
            failures.append(f"{type(exc).__name__}: {exc}")
        elif future.result().result == "missing":
            failures.append(f"missing: {future.result().name}")
    return failures


def _helper_checkout_crash_after_backup_rename(
    path: pathlib.Path,
    output_hash: HashInfo,
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
) -> None:
    """Simulate a restore killed immediately after its directory backup rename."""
    original_rename = pathlib.Path.rename

    def _rename_and_crash(source: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
        result = original_rename(source, target)
        if source == path and target.name.startswith(cache._BACKUP_TEMP_PREFIX):
            os._exit(73)
        return result

    pathlib.Path.rename = _rename_and_crash  # pyright: ignore[reportAttributeAccessIssue]
    _ = cache.restore_from_cache(
        path,
        output_hash,
        cache_dir,
        checkout_modes=[cache.CheckoutMode.COPY],
        state_dir=state_dir,
    )


def test_checkout_recovers_fresh_crash_orphans_under_outer_lock(tmp_path: pathlib.Path) -> None:
    """A retry of checkout removes a killed checkout's fresh target-tagged artifacts (#228)."""
    target = tmp_path / "data" / "out"
    target.mkdir(parents=True)
    (target / "result.txt").write_text("cached content")
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / ".pivot"
    output_hash = cache.save_to_cache(target, cache_dir, checkout_mode=cache.CheckoutMode.COPY)
    (target / "result.txt").write_text("old workspace content")

    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(
        target=_helper_checkout_crash_after_backup_rename,
        args=(target, output_hash, cache_dir, state_dir),
    )
    process.start()
    process.join(timeout=60)

    assert process.exitcode == 73
    assert not target.exists()

    outcome = checkout_mod._restore_path_sync(
        target,
        output_hash,
        cache_dir,
        [cache.CheckoutMode.COPY],
        checkout_mod.CheckoutBehavior.SKIP_EXISTING,
        state_dir,
        target=_helper_own_restore(target, output_hash),
    )
    assert (outcome.result, outcome.name) == ("restored", "out")
    assert (target / "result.txt").read_text() == "cached content"
    assert not [p.name for p in target.parent.iterdir() if p.name.startswith(".pivot_")]


def _helper_checkout_file_crash_before_replace(
    root: pathlib.Path,
    path: pathlib.Path,
    output_hash: HashInfo,
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
) -> None:
    """Simulate a file checkout killed after it staged its temp file, before the rename."""
    project._project_root_cache = root

    def _replace_and_crash(self: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
        os._exit(73)

    pathlib.Path.replace = _replace_and_crash  # pyright: ignore[reportAttributeAccessIssue]
    _ = checkout_mod._restore_path_sync(
        path,
        output_hash,
        cache_dir,
        [cache.CheckoutMode.COPY],
        checkout_mod.CheckoutBehavior.FORCE,
        state_dir,
        target=_helper_own_restore(path, output_hash),
    )


def test_file_checkout_removes_staging_file_left_by_killed_checkout(
    tmp_path: pathlib.Path,
) -> None:
    """A retry of a killed file checkout removes the temp file it staged (#228)."""
    data = tmp_path / "data"
    data.mkdir()
    target = data / "out.txt"
    target.write_text("cached content\n")
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / ".pivot"
    output_hash = cache.save_to_cache(target, cache_dir, checkout_mode=cache.CheckoutMode.COPY)

    process = multiprocessing.get_context("spawn").Process(
        target=_helper_checkout_file_crash_before_replace,
        args=(tmp_path, target, output_hash, cache_dir, state_dir),
    )
    process.start()
    process.join(timeout=60)

    assert process.exitcode == 73
    assert [p.name for p in data.iterdir() if p.name.startswith(".pivot_")], "nothing staged"

    outcome = checkout_mod._restore_path_sync(
        target,
        output_hash,
        cache_dir,
        [cache.CheckoutMode.COPY],
        checkout_mod.CheckoutBehavior.FORCE,
        state_dir,
        target=_helper_own_restore(target, output_hash),
    )
    assert (outcome.result, outcome.name) == ("restored", "out.txt")

    assert sorted(p.name for p in data.iterdir()) == ["out.txt"]
    assert target.read_text() == "cached content\n"


def test_file_checkout_computes_restore_token_once(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkout reuses the token from its held restore lock."""
    target = tmp_path / "data" / "out.txt"
    target.parent.mkdir(parents=True)
    target.write_text("cached content\n")
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / ".pivot"
    output_hash = cache.save_to_cache(target, cache_dir, checkout_mode=cache.CheckoutMode.COPY)
    calls = list[pathlib.Path]()
    original_token = cache._restore_target_token

    def _counted_token(path: pathlib.Path) -> str:
        calls.append(path)
        return original_token(path)

    monkeypatch.setattr(cache, "_restore_target_token", _counted_token)

    outcome = checkout_mod._restore_path_sync(
        target,
        output_hash,
        cache_dir,
        [cache.CheckoutMode.COPY],
        checkout_mod.CheckoutBehavior.FORCE,
        state_dir,
        target=_helper_own_restore(target, output_hash),
    )
    assert (outcome.result, outcome.name) == ("restored", "out.txt")
    assert calls == [target]


@pytest.mark.parametrize(
    "behavior",
    [checkout_mod.CheckoutBehavior.ERROR, checkout_mod.CheckoutBehavior.FORCE],
)
@pytest.mark.parametrize(
    "checkout_modes",
    [
        pytest.param([cache.CheckoutMode.COPY], id="copy"),
        pytest.param(cache.DEFAULT_CHECKOUT_MODE_ORDER, id="default-order"),
    ],
)
def test_concurrent_checkouts_of_same_outputs(
    tmp_path: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    behavior: checkout_mod.CheckoutBehavior,
) -> None:
    """Concurrent checkout processes of the same project all succeed (#461).

    No process may observe another's half-restored output, and --force may not
    delete an output another process is restoring.
    """
    hashes = _helper_cache_outputs(tmp_path)
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / ".pivot"
    parent = tmp_path / "data" / "shared"
    n_procs = 4
    ctx = multiprocessing.get_context("spawn")
    failures = list[str]()
    with ctx.Manager() as manager, ctx.Pool(n_procs) as pool:
        barrier = manager.Barrier(n_procs, timeout=60)
        for _ in range(5):
            results = pool.starmap(
                _helper_checkout_all_at_barrier,
                [(tmp_path, hashes, cache_dir, state_dir, checkout_modes, behavior, barrier)]
                * n_procs,
            )
            failures.extend(f for result in results for f in result)
            for i in range(_N_FILES):
                assert (parent / f"file{i}.txt").read_text() == _helper_file_content(i)
            for i in range(_N_DIRS):
                for j in range(10):
                    assert (parent / f"dir{i}" / f"f{j}.txt").read_text() == f"dir {i} file {j}\n"
            assert not [p.name for p in parent.iterdir() if p.name.startswith(".pivot_")]
            for path in hashes:
                cache.remove_output(pathlib.Path(path))

    assert failures == []
