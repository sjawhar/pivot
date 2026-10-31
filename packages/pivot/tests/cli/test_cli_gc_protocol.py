"""The naming-lock protocol, driven through real pivot processes.

`pivot gc` holds ``naming.lock`` exclusive while it deletes; every writer holds it shared for one
commit unit, from its first look at a cache object to the record that names it. These tests hold
the exclusive side themselves and run real pivot processes against it: a writer site that stops
taking the shared side lets its record land while the test holds the lock, and the test fails.
Each writer site has its own test — a stage commit, a run-cache skip, `pivot commit`, `pivot
track` and `pivot get --rev` — so a site that loses its lock is caught by the test for that site
and not by another's. The last test pins the other side of the bargain: the shared lock covers
only the commit, never a stage's execution, so a collection proceeds while a long stage runs.
"""

from __future__ import annotations

import contextlib
import fcntl
import getpass
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast

import boto3
import pytest
import xxhash
import yaml

from pivot import run_history, types
from pivot.storage import cache, gc, listing, lock, naming, state, track

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

# A stage that, when asked, marks that its body is running and then waits to be released. The
# environment variables are read at run time, so they do not change the stage's fingerprint.
_PIPELINE = """\
import os
import pathlib
import time

from pivot import Pipeline, stage


@stage
def shout(text: str) -> str:
    gate = os.environ.get("GC_PROTOCOL_GATE")
    if gate:
        gate_dir = pathlib.Path(gate)
        (gate_dir / "started").write_text("")
        if os.environ.get("GC_PROTOCOL_WAIT"):
            deadline = time.monotonic() + 120
            while not (gate_dir / "go").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
    return text.upper()


with Pipeline("shadow") as pipeline:
    shout(pipeline.input("input.txt", t=str))
"""


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update(extra)
    return env


def _run(
    root: pathlib.Path, *args: str, env: dict[str, str] | None = None, timeout: float = 180
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_PIVOT, *args],
        cwd=root,
        env=env or _env(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _start(root: pathlib.Path, *args: str, env: dict[str, str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [_PIVOT, *args],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def _project(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(exist_ok=True)
    root = path.resolve()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_run(root, "init"))
    (root / "pipeline.py").write_text(_PIPELINE)
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("hello\n")
    return root


def _locks(root: pathlib.Path) -> dict[str, str]:
    """Every stage lock in the project, by path, with its text."""
    stages = root / ".pivot" / "stages"
    return {str(p.relative_to(stages)): p.read_text() for p in sorted(stages.rglob("*.lock"))}


def _entry(cache_dir: pathlib.Path, root: pathlib.Path) -> pathlib.Path:
    """The register entry of a project whose state is in `.pivot`."""
    path = naming.project_register_path(cache_dir, root, root / ".pivot")
    assert path is not None, f"{root} has no project id"
    return path


def _register_fresh(root: pathlib.Path) -> pathlib.Path:
    """Write this project's register entry now, so the command entry takes no lock, and return
    the cache. The tests then hold the exclusive lock against the writer sites alone."""
    cache_dir = root / ".pivot" / "cache"
    naming.register_project(cache_dir, root, root / ".pivot")
    assert _entry(cache_dir, root).exists()
    return cache_dir


def _wait_for(marker: pathlib.Path, proc: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not marker.exists():
        assert proc.poll() is None, f"pivot exited before {marker.name}: {proc.communicate()[0]}"
        assert time.monotonic() < deadline, f"no {marker.name} within {timeout}s"
        time.sleep(0.05)


def _finish(proc: subprocess.Popen[str]) -> str:
    out, _ = proc.communicate(timeout=180)
    assert proc.returncode == 0, out
    return out


def _object_for(root: pathlib.Path, text: str) -> pathlib.Path:
    """Where the cache object holding ``text`` lives in the project's cache."""
    probe = root.parent / "object-probe"
    probe.write_text(text)
    key, _ = cache.hash_file(probe)
    probe.unlink()
    return cache.get_cache_path(root / ".pivot" / "cache" / "files", key)


def _publishes_outside_then_rechecks(
    proc: subprocess.Popen[str], what: str, obj: pathlib.Path, landed: Callable[[], bool]
) -> None:
    """While the test holds the naming lock: the writer's object must reach the cache (hashed
    and published before the unit) while its record waits. The test then removes the object,
    as a collection deleting under the lock would; the writer must publish it again under the
    lock before it names it (checked by the caller once the writer finishes)."""
    deadline = time.monotonic() + 120
    while not obj.exists():
        assert proc.poll() is None, f"{what} exited before publishing: {proc.communicate()[0]}"
        assert time.monotonic() < deadline, f"{what} published nothing outside its naming unit"
        time.sleep(0.05)
    time.sleep(0.5)
    assert proc.poll() is None, f"{what} did not wait for the collection"
    assert not landed(), f"{what} wrote its record during the collection"
    obj.unlink()


def _assert_waits_for_a_collection(
    root: pathlib.Path,
    cache_dir: pathlib.Path,
    args: tuple[str, ...],
    *,
    unblocked: float,
    landed: Callable[[], bool],
    obj: pathlib.Path | None = None,
) -> str:
    """Run ``pivot <args>`` while holding the naming lock exclusive: its record must not land
    until the lock is released, then must. ``unblocked`` is how long the same kind of command
    took without the lock; the test waits three times that before judging it blocked. With
    ``obj``, the object the command names: it must be published before the unit and, removed
    while the lock is held, published again inside the unit."""
    what = f"pivot {' '.join(args)}"
    assert not landed(), f"{what}: its record is there before it ran"
    assert obj is None or not obj.exists(), f"{what}: its object is cached before it ran"
    with gc.store_snapshot(cache_dir) as snapshot, gc.naming_exclusive(snapshot):
        proc = _start(root, *args, env=_env())
        try:
            if obj is not None:
                _publishes_outside_then_rechecks(proc, what, obj, landed)
            deadline = time.monotonic() + max(3 * unblocked, 5.0)
            while proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            assert proc.poll() is None, f"{what} did not wait for the collection"
            assert not landed(), f"{what} wrote its record during the collection"
        except BaseException:
            proc.kill()
            raise
    out = _finish(proc)
    assert landed(), f"{what} never wrote its record after the collection"
    assert obj is None or obj.exists(), f"{what} named an object it did not re-check"
    return out


def test_a_stage_commit_waits_while_a_collection_holds_the_naming_lock(
    tmp_path: pathlib.Path,
) -> None:
    """The normal commit site: a stage that ran publishes its outputs and writes its lock only
    once the collection releases the naming lock. ``--force`` skips the run-cache lookup, so this
    test reaches the commit site alone. The run registers its project before the stage runs, so
    the collection takes the lock while the stage body runs, and the commit is what meets it."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    cache_dir = _register_fresh(root)
    (root / "data" / "raw" / "input.txt").write_text("changed\n")
    output_object = _object_for(root, "CHANGED\n")
    assert not output_object.exists()
    before = _locks(root)
    gate = tmp_path / "gate"
    gate.mkdir()

    proc = _start(
        root, "repro", "--force", env=_env(GC_PROTOCOL_GATE=str(gate), GC_PROTOCOL_WAIT="1")
    )
    try:
        _wait_for(gate / "started", proc, timeout=120)
        with gc.store_snapshot(cache_dir) as snapshot, gc.naming_exclusive(snapshot):
            (gate / "go").write_text("")
            _publishes_outside_then_rechecks(
                proc, "a stage commit", output_object, lambda: _locks(root) != before
            )
            time.sleep(2.0)  # the commit follows the stage body within milliseconds
            assert proc.poll() is None, "a stage commit did not wait for the collection"
            assert _locks(root) == before, "a stage lock was written during the collection"
    except BaseException:
        proc.kill()
        raise

    _finish(proc)
    assert _locks(root) != before, "the commit never landed after the collection"
    assert output_object.exists(), "a stage commit named an object it did not re-check"


# Loaded into every process of a run through PYTHONPATH: pauses a worker right after its
# pre-unit run-cache lookup found a row, before the naming unit, until the test writes `go`.
_ROW_HOOK = """\
import os
import pathlib
import time

_gate = os.environ.get("GC_TEST_ROW_GATE")
if _gate:
    from pivot.executor import worker as _worker

    _original = _worker._run_cache_row

    def _held(*args, **kwargs):
        row = _original(*args, **kwargs)
        started = pathlib.Path(_gate, "started")
        if row is not None and not started.exists():
            started.write_text("")
            deadline = time.monotonic() + 120
            while not pathlib.Path(_gate, "go").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
        return row

    _worker._run_cache_row = _held
"""


def test_a_run_cache_skip_waits_while_a_collection_holds_the_naming_lock(
    tmp_path: pathlib.Path,
) -> None:
    """The run-cache skip site: a stage restored from the run cache re-links objects that are
    already cached and names them in its lock. It does so only once the collection releases the
    naming lock, since the objects it re-uses may be the ones being removed. The run registers
    before its stages, so a hook holds the worker after its lookup found a row and before the
    unit; the collection takes the lock there, and the restore is what meets it."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    (root / "data" / "raw" / "input.txt").write_text("other\n")
    _ok(_run(root, "repro"))
    (root / "data" / "raw" / "input.txt").write_text("hello\n")  # back to a cached run
    cache_dir = root / ".pivot" / "cache"
    before = _locks(root)
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(_ROW_HOOK)
    gate = tmp_path / "gate"
    gate.mkdir()

    proc = _start(root, "repro", env=_env(PYTHONPATH=str(hook), GC_TEST_ROW_GATE=str(gate)))
    try:
        _wait_for(gate / "started", proc, timeout=120)
        with gc.store_snapshot(cache_dir) as snapshot, gc.naming_exclusive(snapshot):
            (gate / "go").write_text("")
            time.sleep(3.0)  # the restore follows the lookup within milliseconds
            assert proc.poll() is None, "a run-cache skip did not wait for the collection"
            assert _locks(root) == before, "a run-cache skip wrote its lock during the collection"
    except BaseException:
        proc.kill()
        raise

    out = _finish(proc)
    # Reported cached while its lock changed: only a run-cache restore does both (an unchanged
    # skip leaves the lock as it was, and a run would be reported as ran).
    assert "0 ran, 1 cached" in out, f"the stage was not restored from the run cache: {out}"
    assert _locks(root) != before, "the run-cache skip never landed after the collection"


def test_pivot_commit_waits_while_a_collection_holds_the_naming_lock(
    tmp_path: pathlib.Path,
) -> None:
    """The `pivot commit` site: committing a `--no-commit` run's outputs caches them and writes
    the stage lock only once the collection releases the naming lock."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    (root / "data" / "raw" / "input.txt").write_text("not committed yet\n")
    started = time.monotonic()
    _ok(_run(root, "repro", "--no-commit"))
    unblocked = time.monotonic() - started
    cache_dir = _register_fresh(root)
    before = _locks(root)

    _assert_waits_for_a_collection(
        root,
        cache_dir,
        ("commit",),
        unblocked=unblocked,
        landed=lambda: _locks(root) != before,
        obj=_object_for(root, "NOT COMMITTED YET\n"),
    )


def test_pivot_track_waits_while_a_collection_holds_the_naming_lock(
    tmp_path: pathlib.Path,
) -> None:
    """The `pivot track` site: tracking a file caches it and writes its `.pvt` only once the
    collection releases the naming lock."""
    root = _project(tmp_path / "project")
    (root / "first.txt").write_text("tracked first\n")
    started = time.monotonic()
    _ok(_run(root, "track", "first.txt"))
    unblocked = time.monotonic() - started
    cache_dir = _register_fresh(root)
    (root / "notes.txt").write_text("tracked during a collection\n")

    _assert_waits_for_a_collection(
        root,
        cache_dir,
        ("track", "notes.txt"),
        unblocked=unblocked,
        landed=(root / "notes.txt.pvt").exists,
        obj=_object_for(root, "tracked during a collection\n"),
    )


def test_pivot_get_rev_waits_while_a_collection_holds_the_naming_lock(
    tmp_path: pathlib.Path,
) -> None:
    """The `pivot get --rev` site: restoring a stage's output from a revision checks its cache
    object and links it out only once the collection releases the naming lock. The output is
    kept out of git, as a cached output is, so `get` restores it from the cache (a file git
    holds is restored from git, which names no cache object and needs no lock)."""
    root = _project(tmp_path / "project")
    (root / ".gitignore").write_text("data/shadow/\n")
    _ok(_run(root, "repro"))
    git_env = _env(
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@t",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@t",
    )
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, env=git_env)
    subprocess.run(["git", "commit", "-q", "-m", "outputs"], cwd=root, check=True, env=git_env)
    output = root / "data" / "shadow" / "shout.txt"
    output.unlink()
    started = time.monotonic()
    restored = _ok(_run(root, "get", "--rev", "HEAD", "shadow/shout"))
    unblocked = time.monotonic() - started
    assert "(from cache)" in restored, restored
    output.unlink()
    cache_dir = _register_fresh(root)

    _assert_waits_for_a_collection(
        root,
        cache_dir,
        ("get", "--rev", "HEAD", "shadow/shout"),
        unblocked=unblocked,
        landed=output.exists,
    )


def test_a_collection_runs_while_a_stage_executes(tmp_path: pathlib.Path) -> None:
    """The shared lock covers a commit, never a stage's execution: `pivot gc` takes the
    exclusive lock and finishes while a stage is still running."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    gate = tmp_path / "gate"
    gate.mkdir()
    proc = _start(
        root, "repro", "--force", env=_env(GC_PROTOCOL_GATE=str(gate), GC_PROTOCOL_WAIT="1")
    )
    try:
        _wait_for(gate / "started", proc, timeout=120)
        collected = _run(root, "gc", "--yes", "--grace-hours", "24", timeout=60)
        assert collected.returncode == 0, collected.stdout + collected.stderr
        assert proc.poll() is None, "the stage finished before the collection could be checked"
    finally:
        (gate / "go").write_text("")
    _finish(proc)


# =============================================================================
# The gc side: real `pivot gc` processes against writers
# =============================================================================


def _naming_paths(cache_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """The turnstile and naming lock beside the cache's register, where `pivot gc` takes them."""
    return naming.naming_lock_paths(cache_dir)


def _probe_blocked(path: pathlib.Path) -> bool:
    """Whether another process holds ``path`` exclusive (a shared, non-blocking probe fails)."""
    if not path.exists():
        return False
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


def _wait_until(what: str, ready: Callable[[], bool], proc: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 60
    while not ready():
        assert proc.poll() is None, f"pivot gc exited before {what}: {proc.communicate()[0]}"
        assert time.monotonic() < deadline, f"pivot gc never reached {what}"
        time.sleep(0.01)


def _gc_holds_the_naming_lock(cache_dir: pathlib.Path) -> bool:
    return _probe_blocked(_naming_paths(cache_dir)[1])


def _gc_waits_for_the_naming_lock(cache_dir: pathlib.Path) -> bool:
    """`pivot gc` has taken the turnstile, so it is past its plan and waiting for the lock."""
    return _probe_blocked(_naming_paths(cache_dir)[0])


_PIN_RECORD = "path: pin.txt\nhash: 0123456789abcdef\nsize: 1\n"


@contextlib.contextmanager
def _fifo_pin(project: pathlib.Path, cache_dir: pathlib.Path) -> Iterator[threading.Semaphore]:
    """A registered peer project whose one `.pvt` is a FIFO: `pivot gc`, reading the peer's
    records, waits in `open` until the test lets that read through (``release()`` on the
    yielded semaphore lets one read through). The re-read under the exclusive lock reads this
    project's records before the peer's, so withholding that read pins gc mid re-read with the
    naming lock held: the only way a test can hold gc there, since under it gc waits for no
    other pivot process."""
    (project / ".pivot").mkdir(parents=True)
    fifo = project / "pin.pvt"
    os.mkfifo(fifo)
    naming.register_project(cache_dir, project, project / ".pivot")
    allowed = threading.Semaphore(0)
    stop = threading.Event()

    def feed() -> None:
        while not stop.is_set():
            if not allowed.acquire(timeout=0.1):
                continue
            while not stop.is_set():
                try:
                    fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                except OSError:  # ENXIO: no reader is waiting yet
                    time.sleep(0.01)
                    continue
                with os.fdopen(fd, "w") as f:
                    f.write(_PIN_RECORD)
                break

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    try:
        yield allowed
    finally:
        stop.set()
        feeder.join(timeout=5)


def _publish(object_root: pathlib.Path, scratch: pathlib.Path, text: str) -> str:
    """Put ``text`` in the cache as an object no record names, and return its key."""
    source = scratch / "publish"
    source.write_text(text)
    key, _ = cache.hash_file(source)
    cache.copy_to_cache(source, cache.get_cache_path(object_root, key))
    source.unlink()
    return key


def _name_in_lock(stages_dir: pathlib.Path, stage: str, key: str) -> None:
    """Write the stage lock a writer's commit writes, naming ``key``."""
    lock.StageLock(stage, stages_dir).write(
        types.LockData(
            code_manifest={"self:test": "abc123"},
            params={},
            dep_hashes={},
            output_hashes={types.ArtifactIdentity(stage, None): types.FileHash(hash=key)},
        )
    )


def _start_gc(root: pathlib.Path, *extra: str) -> subprocess.Popen[str]:
    return _start(root, "gc", "--yes", "--grace-hours", "0", *extra, env=_env())


@pytest.mark.parametrize("layout", ["plain", "files-symlink"])
def test_a_stage_commit_racing_pivot_gc_loses_nothing(tmp_path: pathlib.Path, layout: str) -> None:
    """The race the protocol exists for, end to end: `pivot gc` is deleting, holding the naming
    lock, and has not yet reached a cached object nobody names; a stage commits an output with
    exactly that content, a cache hit that publishes nothing. The commit waits until the
    collection is done, then publishes the object again and names it, so nothing it names is
    lost. `files-symlink` moves `<cache>/files` to another directory, as a cache whose objects
    were moved to a bigger disk is: the commit must still lock the file `pivot gc` locks."""
    root = _project(tmp_path / "project")
    # Symlink checkout, the reporter's mode: a hardlinked output would keep its object alive by
    # its link count, hiding a lost race.
    (root / ".pivot" / "config.yaml").write_text("cache:\n  checkout_mode:\n  - symlink\n")
    cache_dir = root / ".pivot" / "cache"
    if layout == "files-symlink":
        cache_dir.mkdir(parents=True, exist_ok=True)
        (tmp_path / "disk" / "objects").mkdir(parents=True)
        (cache_dir / "files").symlink_to(tmp_path / "disk" / "objects")
    _ok(_run(root, "repro"))
    object_root = cache_dir / "files"
    output_text = "VICTIM OF THE RACE\n"
    victim = _publish(object_root, tmp_path, output_text)
    _publish(object_root, tmp_path, "an orphan the collection removes\n")
    (root / "data" / "raw" / "input.txt").write_text(output_text.lower())
    _register_fresh(root)
    before = _locks(root)
    gate = tmp_path / "gate"
    gate.mkdir()

    # The run registers its project, then its stage body waits: the commit, not the run's
    # registration, is what meets the collection's lock.
    writer = _start(
        root, "repro", "--force", env=_env(GC_PROTOCOL_GATE=str(gate), GC_PROTOCOL_WAIT="1")
    )
    try:
        _wait_for(gate / "started", writer, timeout=120)
    except BaseException:
        writer.kill()
        raise
    with _fifo_pin(tmp_path / "peer", cache_dir) as pin:
        pin.release()  # the plan's read of the peer
        collector = _start_gc(root)
        try:
            _wait_until(
                "its exclusive lock", lambda: _gc_holds_the_naming_lock(cache_dir), collector
            )
            # gc holds the lock and waits on the peer's record, before it deletes anything.
            (gate / "go").write_text("")  # the stage returns; its commit meets gc's lock
            time.sleep(2.0)  # the commit follows the stage body within milliseconds
            landed = writer.poll() is not None or _locks(root) != before
        except BaseException:
            writer.kill()
            collector.kill()
            raise
        pin.release()  # the locked re-read: gc deletes the unnamed victim and finishes
        _finish(collector)

    _finish(writer)
    assert cache.get_cache_path(object_root, victim).exists(), "an object a stage names is gone"
    assert victim in "".join(_locks(root).values()), "the stage lock does not name its output"
    assert (root / "data" / "shadow" / "shout.txt").read_text() == output_text
    assert not landed, "a stage commit named an object while gc was deleting"


def test_pivot_gc_rereads_the_records_under_its_lock(tmp_path: pathlib.Path) -> None:
    """A writer already in its unit when `pivot gc` comes for the lock names an object and only
    then finishes. gc waits for it, re-reads every record under the lock, and keeps that object.
    gc's re-read is held at a peer's record for a second first, so a gc that did not wait for
    the writer, or deleted against the records it read before the lock, has already read this
    project's records and removes the object once released."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    cache_dir = _register_fresh(root)
    object_root = cache_dir / "files"
    victim = _publish(object_root, tmp_path, "named while gc waits\n")

    with _fifo_pin(tmp_path / "peer", cache_dir) as pin:
        pin.release()  # the plan's read of the peer
        with naming.naming_unit(cache_dir, project=None):  # a writer mid-commit
            collector = _start_gc(root)
            _wait_until(
                "the turnstile", lambda: _gc_waits_for_the_naming_lock(cache_dir), collector
            )
            time.sleep(1.0)  # a gc that skipped the wait is held at the peer by now
            _name_in_lock(root / ".pivot" / "stages", "late/stage", victim)
        pin.release()  # the locked re-read of the peer
        _finish(collector)

    assert cache.get_cache_path(object_root, victim).exists(), "gc removed an object a record names"


def test_pivot_gc_waits_for_no_publisher_while_it_holds_the_naming_lock(
    tmp_path: pathlib.Path,
) -> None:
    """A publisher that starts staging into a prefix after gc's plan holds that prefix's lock
    for its whole copy. gc, once it holds the naming lock, must not wait for it, or every
    writer on the cache waits behind gc for the length of that copy."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    cache_dir = _register_fresh(root)
    object_root = cache_dir / "files"
    doomed = _publish(object_root, tmp_path, "removed while a publisher stages beside it\n")
    prefix_lock = object_root / doomed[:2] / cache.PREFIX_LOCK

    with contextlib.ExitStack() as held:
        with naming.naming_unit(cache_dir, project=None):  # let gc plan, then hold the turnstile
            collector = _start_gc(root)
            _wait_until(
                "the turnstile", lambda: _gc_waits_for_the_naming_lock(cache_dir), collector
            )
            fd = cache.open_cache_lock(prefix_lock)  # a publisher begins staging there
            held.callback(os.close, fd)
            fcntl.flock(fd, fcntl.LOCK_SH)
        try:
            out, _ = collector.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            collector.kill()
            pytest.fail("pivot gc waited for a publisher while it held the naming lock")
        assert collector.returncode == 0, out


def test_pivot_gc_rereads_the_register_under_its_lock(tmp_path: pathlib.Path) -> None:
    """A project that registers while `pivot gc` waits for the naming lock is read under it: the
    objects its records name are kept, though the plan was built before it existed."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    cache_dir = _register_fresh(root)
    victim = _publish(cache_dir / "files", tmp_path, "named by a project that registers late\n")
    late = tmp_path / "late-project"
    (late / ".pivot" / "stages").mkdir(parents=True)

    with naming.naming_unit(cache_dir, project=None):  # a command registering its project
        collector = _start_gc(root)
        _wait_until("the turnstile", lambda: _gc_waits_for_the_naming_lock(cache_dir), collector)
        _name_in_lock(late / ".pivot" / "stages", "other/stage", victim)
        naming.register_project(cache_dir, late, late / ".pivot")

    _finish(collector)
    assert cache.get_cache_path(cache_dir / "files", victim).exists(), "gc used a stale register"


def test_pivot_gc_keeps_an_object_a_run_cache_row_written_before_its_lock_names(
    tmp_path: pathlib.Path,
) -> None:
    """A run that lands while `pivot gc` waits for the naming lock writes a run-cache row newer
    than any run the plan saw. The prune keeps it, so its object must be kept too: every row is
    read under the lock, not filtered by run ids read before it."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    cache_dir = _register_fresh(root)
    victim = _publish(cache_dir / "files", tmp_path, "named only by a new run-cache row\n")

    with naming.naming_unit(cache_dir, project=None):
        collector = _start_gc(root)
        _wait_until("the turnstile", lambda: _gc_waits_for_the_naming_lock(cache_dir), collector)
        with state.StateDB(root / ".pivot") as db:
            db.write_run(
                run_history.RunManifest(
                    run_id="20991231T000000-9",
                    started_at="2099-12-31T00:00:00Z",
                    ended_at="2099-12-31T00:00:01Z",
                    targeted_stages=[],
                    execution_order=[],
                    stages={},
                )
            )
            db.apply_deferred_writes(
                "late/stage",
                [],
                types.DeferredWrites(
                    run_cache_input_hash="0123456789abcdef",
                    run_cache_entry=run_history.RunCacheEntry(
                        run_id="20991231T000000-9",
                        output_hashes=[
                            run_history.output_hash_to_entry(
                                "data/late.txt", types.FileHash(hash=victim)
                            )
                        ],
                    ),
                ),
            )

    _finish(collector)
    assert cache.get_cache_path(cache_dir / "files", victim).exists(), "gc used stale run ids"


def test_pivot_gc_refuses_when_another_hosts_project_is_not_visible(tmp_path: pathlib.Path) -> None:
    """A register entry another host wrote, for a root in that host's local storage under a
    parent this host also has: its absence here proves nothing, so `pivot gc` refuses, keeps
    the entry, and removes nothing."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    cache_dir = _register_fresh(root)
    victim = _publish(cache_dir / "files", tmp_path, "named by a project on another host\n")
    elsewhere = tmp_path / "node-local"  # the parent exists here; the root does not
    elsewhere.mkdir()
    other_root = elsewhere / "project-on-another-node"
    entry = cache_dir / naming.REGISTER_DIR / "project-on-another-node.json"
    entry.write_text(
        json.dumps(
            {
                "root": str(other_root),
                "state_dir": str(other_root / ".pivot"),
                "host": "pivot-other-node",
                "pivot": "0.5.16",
                "last_write": "2026-10-28T00:00:00Z",
            }
        )
    )

    refused = _run(root, "gc", "--yes", "--grace-hours", "0")

    out = refused.stdout + refused.stderr
    assert refused.returncode != 0, out
    assert "pivot-other-node" in out, out
    assert entry.exists(), "another host's register entry was dropped"
    assert cache.get_cache_path(cache_dir / "files", victim).exists(), "an object was removed"


def test_pivot_gc_is_not_starved_by_a_stream_of_writers(tmp_path: pathlib.Path) -> None:
    """Six writers commit back to back, their units always overlapping, so the naming lock is
    never free. The turnstile still lets `pivot gc` in: it takes the turnstile, new units queue
    behind it, and the ones in flight finish. Without either half of the turnstile, gc waits
    as long as the writers keep coming."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    cache_dir = _register_fresh(root)
    stop = threading.Event()

    def writer(offset: float) -> None:
        time.sleep(offset)
        while not stop.is_set():
            with naming.naming_unit(cache_dir, project=None):
                time.sleep(0.3)

    writers = [threading.Thread(target=writer, args=(i * 0.05,), daemon=True) for i in range(6)]
    for thread in writers:
        thread.start()
    try:
        time.sleep(0.5)
        collected = _run(root, "gc", "--yes", "--grace-hours", "24", timeout=30)
    except subprocess.TimeoutExpired:
        pytest.fail("pivot gc was starved by a stream of writers")
    finally:
        stop.set()
        for thread in writers:
            thread.join(timeout=10)
    assert collected.returncode == 0, collected.stdout + collected.stderr


def test_the_first_commit_into_a_new_cache_takes_the_naming_lock(tmp_path: pathlib.Path) -> None:
    """A project's first run publishes into a cache that does not exist yet. Its commit creates
    the cache root and takes the naming lock there, like every later commit, rather than running
    unlocked because there was no directory to hold the lock files."""
    root = _project(tmp_path / "project")
    cache_dir = root / ".pivot" / "cache"
    assert not cache_dir.exists()

    _ok(_run(root, "repro"))

    turnstile, naming = _naming_paths(cache_dir)
    assert naming.exists() and turnstile.exists(), "the first commit took no naming lock"


_METRIC_ONLY_PIPELINE = """\
from typing import Annotated

from pivot.compose import Pipeline, metric, stage


@stage
def score(text: str) -> Annotated[dict[str, float], metric]:
    return {"length": float(len(text))}


with Pipeline("shadow") as pipeline:
    score(pipeline.input("input.txt", t=str))
"""


def test_a_first_commit_that_publishes_nothing_still_takes_the_naming_lock(
    tmp_path: pathlib.Path,
) -> None:
    """A metric is not cached, so a project whose only output is one publishes nothing before
    its first commit, and no run-cache lookup takes a unit in a cold run. The commit's unit must
    still create the cache root and take the lock there, not refuse or run unlocked."""
    root = _project(tmp_path / "project")
    (root / "pipeline.py").write_text(_METRIC_ONLY_PIPELINE)
    cache_dir = root / ".pivot" / "cache"
    assert not cache_dir.exists()

    _ok(_run(root, "repro"))

    turnstile, naming_lock = _naming_paths(cache_dir)
    assert naming_lock.exists() and turnstile.exists(), "the first commit took no naming lock"


def test_a_command_refuses_when_it_cannot_register(tmp_path: pathlib.Path) -> None:
    """A writer cannot name objects when its register entry cannot be written."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    cache_dir = root / ".pivot" / "cache"
    register = cache_dir / naming.REGISTER_DIR
    if register.exists():
        for entry in register.iterdir():
            entry.unlink()
        register.rmdir()
    register.write_text("not a directory\n")  # no entry can be written beneath it

    (root / "notes.txt").write_text("would name an object\n")
    refused = _run(root, "track", "notes.txt")

    out = refused.stdout + refused.stderr
    assert refused.returncode != 0, out
    assert "Cannot record this project in the cache register" in out, out
    assert str(register) in out, out


@pytest.mark.skipif(os.geteuid() == 0, reason="root can create files in a read-only directory")
def test_a_writer_refuses_when_it_cannot_open_the_naming_lock(tmp_path: pathlib.Path) -> None:
    """A writer that cannot take the naming lock (the lock files do not exist and it cannot
    create them) refuses rather than name objects unlocked, since another user's `pivot gc`
    could create the lock and delete meanwhile."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    cache_dir = root / ".pivot" / "cache"
    for path in _naming_paths(cache_dir):
        path.unlink()
    (root / "notes.txt").write_text("cannot be named unlocked\n")
    mode = cache_dir.stat().st_mode
    cache_dir.chmod(0o555)
    try:
        refused = _run(root, "track", "notes.txt")
    finally:
        cache_dir.chmod(mode)

    out = refused.stdout + refused.stderr
    assert refused.returncode != 0, out
    assert "naming lock" in out and "naming.turnstile" in out, out
    assert not (root / "notes.txt.pvt").exists(), "a writer named an object without the lock"


# =============================================================================
# Round 4b: every writer is registered, an absent root refuses, cache-root links, re-publish
# =============================================================================


def _shared_project(
    path: pathlib.Path, shared_cache: pathlib.Path, checkout: str = "symlink"
) -> pathlib.Path:
    """A project that caches into ``shared_cache`` with ``checkout`` mode (by default symlink,
    the reporter's mode)."""
    root = _project(path)
    (root / ".pivot" / "config.yaml").write_text(
        f"cache:\n  dir: {shared_cache}\n  checkout_mode:\n  - {checkout}\n"
    )
    return root


def test_the_project_that_creates_a_shared_cache_is_registered(tmp_path: pathlib.Path) -> None:
    """P13: the first command of the first project to use a shared cache creates the cache, so
    it found no cache to register in when it started. It must still be registered before it
    names anything, or another project's `pivot gc` removes its outputs."""
    shared = tmp_path / "shared-cache"
    first = _shared_project(tmp_path / "first", shared)
    other = _shared_project(tmp_path / "other", shared)
    (other / "data" / "raw" / "input.txt").write_text("another project\n")
    assert not shared.exists()
    _ok(_run(first, "repro"))  # creates the shared cache
    _ok(_run(other, "repro"))
    output = first / "data" / "shadow" / "shout.txt"
    assert output.resolve().exists()

    collected = _run(other, "gc", "--yes", "--grace-hours", "0")

    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert output.resolve().exists(), "gc removed an output of the project that made the cache"


def test_pivot_get_rev_registers_its_project(tmp_path: pathlib.Path) -> None:
    """P9: `pivot get` skips pipeline discovery, so the command-entry registration never runs
    for it. A fresh clone's `get --rev --checkout-mode symlink` links an object that only the
    clone names (the project that produced it ran elsewhere, as CI does, and is not registered
    here). Its naming unit must register the clone, or another project's gc removes the object
    and leaves the clone's link dangling."""
    shared = tmp_path / "shared-cache"
    origin = _shared_project(tmp_path / "origin", shared)
    (origin / ".gitignore").write_text("data/shadow/\n")
    git_env = _env(
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@t",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@t",
    )
    _ok(_run(origin, "repro"))
    subprocess.run(["git", "add", "-A"], cwd=origin, check=True, env=git_env)
    subprocess.run(["git", "commit", "-q", "-m", "outputs"], cwd=origin, check=True, env=git_env)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True, env=git_env)
    # The producer is not a user of this cache (it ran elsewhere). Before round 4b it was never
    # registered at all (P13), so the entry may be absent.
    if (
        origin_entry := naming.project_register_path(shared, origin, origin / ".pivot")
    ) is not None:
        origin_entry.unlink(missing_ok=True)
    other = _shared_project(tmp_path / "other", shared)
    (other / "data" / "raw" / "input.txt").write_text("an unrelated project\n")
    _ok(_run(other, "repro"))
    _ok(_run(clone, "get", "--rev", "HEAD", "shadow/shout", "--checkout-mode", "symlink"))
    restored = clone / "data" / "shadow" / "shout.txt"
    assert restored.is_symlink() and restored.resolve().exists()

    collected = _run(other, "gc", "--yes", "--grace-hours", "0")

    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert restored.resolve().exists(), "gc removed the object a fresh clone's get linked"


def test_pivot_gc_refuses_when_this_hosts_project_is_not_visible(tmp_path: pathlib.Path) -> None:
    """P11 and P12: this host's own project can be absent while its parent is present — on a
    disk that is not mounted (the empty mount point remains) or in a container's private mount
    namespace with this hostname. From here both look the same, and neither means the project
    is gone: the collection refuses and removes nothing, and the project, once visible again,
    still has its output."""
    shared = tmp_path / "shared-cache"
    here = _shared_project(tmp_path / "here", shared)
    _ok(_run(here, "repro"))
    mount_point = tmp_path / "mnt"
    mount_point.mkdir()
    away = _shared_project(mount_point / "away", shared)
    (away / "data" / "raw" / "input.txt").write_text("on another disk\n")
    _ok(_run(away, "repro"))
    _ok(_run(here, "status"))
    hidden = tmp_path / "disk-not-mounted"
    away.rename(hidden)  # the disk is not mounted: its mount point is empty

    refused = _run(here, "gc", "--yes", "--grace-hours", "0")

    hidden.rename(away)  # the disk is back
    out = refused.stdout + refused.stderr
    assert refused.returncode != 0, out
    assert "recorded by this host" in out, out
    output = away / "data" / "shadow" / "shout.txt"
    assert output.resolve().exists(), "gc removed an object a project on an unmounted disk names"


# =============================================================================
# Round 6: a registered project proves it is readable; a writer proves it registered
# =============================================================================


def test_pivot_gc_refuses_when_a_registered_project_root_is_an_empty_mountpoint(
    tmp_path: pathlib.Path,
) -> None:
    """P14: an unmounted volume leaves its project-root mount point as an empty directory.

    The cache register still names that root, but none of its state or workspace records are
    visible. The collection must refuse rather than read that absence as an empty project and
    remove the volume's symlink-checkout object. Renaming the project and leaving an empty
    directory models the unmounted mount point without requiring mount privileges.
    """
    shared = tmp_path / "shared-cache"
    here = _shared_project(tmp_path / "here", shared)
    away = _shared_project(tmp_path / "away", shared)
    (away / "data" / "raw" / "input.txt").write_text("on the mounted project volume\n")
    _ok(_run(here, "repro"))
    _ok(_run(away, "repro"))
    output = away / "data" / "shadow" / "shout.txt"
    entry = _entry(shared, away)
    assert output.is_symlink() and output.resolve().exists()
    assert entry.exists()

    hidden = tmp_path / "project-volume-unmounted"
    away.rename(hidden)
    away.mkdir()  # The mount point remains, but it has no .pivot state directory.
    try:
        refused = _run(here, "gc", "--yes", "--grace-hours", "0")
    finally:
        away.rmdir()
        hidden.rename(away)

    out = refused.stdout + refused.stderr
    assert refused.returncode != 0, out
    assert str(entry) in out, out
    assert output.resolve().exists(), "gc removed an object from the unmounted project"


@pytest.mark.skipif(os.geteuid() == 0, reason="root can create files in a read-only directory")
def test_a_writer_that_cannot_register_refuses_before_naming_objects(
    tmp_path: pathlib.Path,
) -> None:
    """M3: a shared-cache writer whose register is unwritable must not make a live link.

    A real `pivot repro` from the unregistered project must fail before its naming unit writes
    the stage lock or symlink checkout. Another project's real collection therefore cannot
    remove an output the unregistered writer was allowed to name.
    """
    shared = tmp_path / "shared-cache"
    collector = _shared_project(tmp_path / "collector", shared)
    writer = _shared_project(tmp_path / "writer", shared)
    (writer / "data" / "raw" / "input.txt").write_text("writer cannot register\n")
    _ok(_run(collector, "repro"))
    register = shared / naming.REGISTER_DIR
    mode = register.stat().st_mode
    register.chmod(0o555)
    try:
        refused = _run(writer, "repro")
    finally:
        register.chmod(mode)
    collected = _run(collector, "gc", "--yes", "--grace-hours", "0")

    out = refused.stdout + refused.stderr
    output = writer / "data" / "shadow" / "shout.txt"
    assert refused.returncode != 0, out
    assert "Cannot record this project in the cache register" in out, out
    assert not output.is_symlink(), "an unregistered writer named an object in its workspace"
    assert collected.returncode == 0, collected.stdout + collected.stderr


def test_pivot_gc_reads_the_current_state_dir_after_a_project_reconfigures(
    tmp_path: pathlib.Path,
) -> None:
    """The register's old state-dir field cannot override a project's current configuration.

    A project registers while its state is `.pivot`, then changes `core.state_dir` and writes a
    new lock there. The workspace link is removed, leaving that current lock as the only record
    that can keep the new object during another project's real collection.
    """
    shared = tmp_path / "shared-cache"
    collector = _shared_project(tmp_path / "collector", shared)
    writer = _shared_project(tmp_path / "writer", shared)
    _ok(_run(collector, "repro"))
    _ok(_run(writer, "repro"))
    entry = _entry(shared, writer)
    stale_state_dir = json.loads(entry.read_text())["state_dir"]
    (writer / ".pivot" / "config.yaml").write_text(
        (writer / ".pivot" / "config.yaml").read_text() + "core:\n  state_dir: state\n"
    )
    (writer / "data" / "raw" / "input.txt").write_text("new configured state\n")
    _ok(_run(writer, "repro"))
    # The entry's recorded state directory is stale (its nonce is current): gc must not use it.
    current = json.loads(entry.read_text())
    entry.write_text(json.dumps({**current, "state_dir": stale_state_dir}))
    output = writer / "data" / "shadow" / "shout.txt"
    assert output.is_symlink()
    current_object = output.resolve()
    assert current_object.exists()
    output.unlink()

    collected = _run(collector, "gc", "--yes", "--grace-hours", "0")

    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert current_object.exists(), "gc ignored the reconfigured project's current state directory"


def test_pivot_gc_reads_both_supported_state_directory_layouts(tmp_path: pathlib.Path) -> None:
    """A 0.5.2-upgraded project can hold live locks in `.pivot` and `core.state_dir`.

    The first real run creates the legacy lock, and a run after setting `core.state_dir` creates
    the configured one. With their workspace link gone, collection must preserve the object each
    lock names rather than inspecting only the newer directory.
    """
    shared = tmp_path / "shared-cache"
    collector = _shared_project(tmp_path / "collector", shared)
    writer = _shared_project(tmp_path / "writer", shared)
    (writer / "data" / "raw" / "input.txt").write_text("legacy state\n")
    _ok(_run(collector, "repro"))
    _ok(_run(writer, "repro"))
    output = writer / "data" / "shadow" / "shout.txt"
    assert output.is_symlink()
    legacy_object = output.resolve()
    assert legacy_object.exists()
    (writer / ".pivot" / "config.yaml").write_text(
        (writer / ".pivot" / "config.yaml").read_text() + "core:\n  state_dir: state\n"
    )
    (writer / "data" / "raw" / "input.txt").write_text("configured state\n")
    _ok(_run(writer, "repro"))
    assert output.is_symlink()
    current_object = output.resolve()
    assert current_object.exists()
    output.unlink()

    collected = _run(collector, "gc", "--yes", "--grace-hours", "0")

    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert legacy_object.exists(), "gc ignored the legacy state directory"
    assert current_object.exists(), "gc ignored the configured state directory"


@contextlib.contextmanager
def _tmpfs_mount(path: pathlib.Path) -> Iterator[Callable[[], None]]:
    """Mount a real tmpfs at ``path`` and yield a function that unmounts it."""
    if subprocess.run(["sudo", "-n", "true"], capture_output=True, check=False).returncode:
        pytest.skip("mounting a real tmpfs needs passwordless sudo")
    path.mkdir(parents=True, exist_ok=True)
    options = f"mode=0700,uid={os.getuid()},gid={os.getgid()}"
    subprocess.run(
        ["sudo", "-n", "mount", "-t", "tmpfs", "-o", options, "tmpfs", str(path)], check=True
    )
    mounted = True

    def unmount() -> None:
        nonlocal mounted
        if mounted:
            subprocess.run(["sudo", "-n", "umount", str(path)], check=True)
            mounted = False

    try:
        yield unmount
    finally:
        unmount()


def _cached_link(
    root: pathlib.Path, cache_dir: pathlib.Path, relative: str, text: str
) -> tuple[pathlib.Path, pathlib.Path]:
    """Track one file and return its `.pvt` record and the cache object only it names."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    _ok(_run(root, "track", relative))
    record = track.get_pvt_path(path)
    data = track.read_pvt_file(record)
    assert data is not None, f"{record} is not a valid .pvt"
    obj = cache.get_cache_path(cache_dir / "files", data["hash"])
    assert obj.exists(), "track did not cache the file"
    return record, obj


def test_pivot_gc_refuses_when_a_registered_project_root_mount_is_unmounted(
    tmp_path: pathlib.Path,
) -> None:
    """P14 with a real tmpfs: an unmounted project root has no project id or records."""
    shared = tmp_path / "shared-cache"
    here = _shared_project(tmp_path / "here", shared)
    _ok(_run(here, "status"))
    mount_point = tmp_path / "project-volume"
    with _tmpfs_mount(mount_point) as unmount:
        away = _shared_project(mount_point, shared)
        _, obj = _cached_link(away, shared, "data.txt", "only the mounted root names this\n")
        entry = _entry(shared, away)
        unmount()

        refused = _run(here, "gc", "--yes", "--grace-hours", "0")

    out = refused.stdout + refused.stderr
    assert refused.returncode != 0, out
    assert str(entry) in out, out
    assert obj.exists(), "gc removed an object named only on the unmounted project root"


def test_pivot_gc_refuses_when_a_registered_project_subdirectory_mount_is_unmounted(
    tmp_path: pathlib.Path,
) -> None:
    """A project data mount can hide `.pvt` records while the project root stays visible."""
    shared = tmp_path / "shared-cache"
    here = _shared_project(tmp_path / "here", shared)
    away = _shared_project(tmp_path / "away", shared)
    _ok(_run(here, "status"))
    with _tmpfs_mount(away / "mounted-data") as unmount:
        _, obj = _cached_link(
            away, shared, "mounted-data/data.txt", "named under a project mount\n"
        )
        entry = _entry(shared, away)
        unmount()

        refused = _run(here, "gc", "--yes", "--grace-hours", "0")

    out = refused.stdout + refused.stderr
    assert refused.returncode != 0, out
    assert str(entry) in out and str(away / "mounted-data") in out, out
    assert "run `pivot checkout` in the project" in out, out
    assert obj.exists(), "gc removed an object named under an unmounted project mount"


def test_pivot_gc_refuses_a_cache_mount_that_is_unmounted(tmp_path: pathlib.Path) -> None:
    """A cache ref directory on its own mount can hide the ref that keeps an object."""
    shared = tmp_path / "shared-cache"
    here = _shared_project(tmp_path / "here", shared)
    _cached_link(here, shared, "first.txt", "makes the cache\n")
    refs = shared / "files" / "refs"
    with _tmpfs_mount(refs) as unmount:
        key = _publish(shared / "files", tmp_path, "named only by a mounted cache ref\n")
        (refs / "store").symlink_to(cache.get_cache_path(shared / "files", key))
        _cached_link(here, shared, "second.txt", "records the cache mounts\n")
        unmount()

        refused = _run(here, "gc", "--yes", "--grace-hours", "0")

    out = refused.stdout + refused.stderr
    assert refused.returncode != 0, out
    assert str(refs) in out, out
    assert cache.get_cache_path(shared / "files", key).exists(), "gc removed a ref-named object"


def _rewrite_entry_host(entry: pathlib.Path, host: str) -> None:
    data = json.loads(entry.read_text())
    data["host"] = host
    entry.write_text(json.dumps(data))


def test_pivot_gc_refuses_another_hosts_root_when_this_path_is_a_different_project(
    tmp_path: pathlib.Path,
) -> None:
    """A host-local path can exist on two hosts and hold two different projects."""
    shared = tmp_path / "shared-cache"
    here = _shared_project(tmp_path / "here", shared)
    away = _shared_project(tmp_path / "same-path", shared)
    _ok(_run(here, "status"))
    _, obj = _cached_link(away, shared, "data.txt", "named only on another host\n")
    entry = _entry(shared, away)
    _rewrite_entry_host(entry, "pivot-other-node")
    away.rename(tmp_path / "project-on-other-host")
    _shared_project(away, shared)
    _cached_link(away, shared, "local.txt", "a different project at the same local path\n")

    refused = _run(here, "gc", "--yes", "--grace-hours", "0")

    out = refused.stdout + refused.stderr
    assert refused.returncode != 0, out
    assert str(entry) in out and "project id" in out, out
    assert obj.exists(), "gc read a different project at another host's registered path"


def test_pivot_gc_collects_a_shared_project_registered_by_another_host(
    tmp_path: pathlib.Path,
) -> None:
    """A project another host reads through the same filesystem (as over NFS) has one id, and
    this host reads the nonce that host wrote in it, so that host's entry stays usable."""
    shared = tmp_path / "shared-cache"
    here = _shared_project(tmp_path / "here", shared)
    away = _shared_project(tmp_path / "shared-project", shared)
    _cached_link(here, shared, "first.txt", "makes the cache\n")
    (away / "data.txt").write_text("named by the shared project\n")
    _ok(_in_view(away, "track", "data.txt", hostname="pivot-other-node"))
    kept = _tracked_object(away / "data.txt.pvt", shared)
    dead = _publish(shared / "files", tmp_path, "an orphan in a shared cache\n")

    collected = _run(here, "gc", "--yes", "--grace-hours", "0")

    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert kept.exists(), "gc removed an object the shared project names"
    assert not cache.get_cache_path(shared / "files", dead).exists(), "gc did not collect"


@pytest.mark.skipif(os.geteuid() == 0, reason="root can list a directory without read permission")
def test_pivot_gc_refuses_when_the_register_cannot_be_listed(tmp_path: pathlib.Path) -> None:
    """A register this user cannot list hides peers of an internal cache."""
    here = _project(tmp_path / "here")
    _ok(_run(here, "repro"))
    internal = here / ".pivot" / "cache"
    peer = _shared_project(tmp_path / "peer", internal)
    _, obj = _cached_link(peer, internal, "data.txt", "named by a peer of an internal cache\n")
    register = internal / naming.REGISTER_DIR
    mode = register.stat().st_mode
    register.chmod(0o300)
    try:
        refused = _run(here, "gc", "--yes", "--grace-hours", "0")
    finally:
        register.chmod(mode)

    out = refused.stdout + refused.stderr
    assert refused.returncode != 0, out
    assert str(register) in out, out
    assert obj.exists(), "gc removed a peer object while its register could not be listed"


def test_pivot_gc_keeps_everything_for_a_dangling_project_symlink(tmp_path: pathlib.Path) -> None:
    """A dangling directory link can point to records on a disk that is not mounted."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    objects = root / ".pivot" / "cache" / "files"
    key = _publish(objects, tmp_path, "named under an external disk\n")
    external = tmp_path / "external-disk" / "datasets"
    external.mkdir(parents=True)
    track.write_pvt_file(
        external / "data.txt.pvt", track.PvtData(path="data.txt", hash=key, size=29)
    )
    (root / "datasets").symlink_to(external)
    (tmp_path / "external-disk").rename(tmp_path / "external-disk-unmounted")

    collected = _run(root, "gc", "--yes", "--grace-hours", "0")

    out = collected.stdout + collected.stderr
    assert collected.returncode == 0, out
    assert "whose target is missing" in out, out
    assert cache.get_cache_path(objects, key).exists(), "gc ignored a dangling record link"


def test_pivot_gc_reads_a_symlinked_stage_lock_directory(tmp_path: pathlib.Path) -> None:
    """A symlinked directory below `stages` still holds stage locks."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    objects = root / ".pivot" / "cache" / "files"
    key = _publish(objects, tmp_path, "named by a linked stage directory\n")
    store = root / ".pivot" / "stage-store"
    _name_in_lock(store, "other/stage", key)
    (root / ".pivot" / "stages" / "other").symlink_to(store / "other")

    collected = _run(root, "gc", "--yes", "--grace-hours", "0")

    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert cache.get_cache_path(objects, key).exists(), "gc skipped a symlinked stage directory"


def test_pivot_gc_refuses_a_registered_project_with_an_unreadable_database(
    tmp_path: pathlib.Path,
) -> None:
    """A damaged state database hides run-cache rows, so collection must refuse cleanly."""
    shared = tmp_path / "shared-cache"
    here = _shared_project(tmp_path / "here", shared)
    peer = _shared_project(tmp_path / "peer", shared)
    _ok(_run(here, "status"))
    _ok(_run(peer, "repro"))
    database = peer / ".pivot" / "state.lmdb" / "data.mdb"
    database.write_bytes(b"not an lmdb file" * 4096)

    refused = _run(here, "gc", "--yes", "--grace-hours", "0")

    out = refused.stdout + refused.stderr
    assert refused.returncode != 0, out
    assert str(peer) in out and "That project may hold outputs" in out, out


def test_pivot_gc_refuses_a_project_registered_without_a_project_id(
    tmp_path: pathlib.Path,
) -> None:
    """A state directory that is a regular file cannot hold a project id. `pivot track` still
    works there, as on 0.5.2, and registers without an id; a collection refuses that entry."""
    shared = tmp_path / "shared-cache"
    here = _shared_project(tmp_path / "here", shared)
    away = _shared_project(tmp_path / "away", shared)
    (away / "state-file").write_text("not a directory\n")
    config = away / ".pivot" / "config.yaml"
    config.write_text(config.read_text() + "core:\n  state_dir: state-file\n")
    _, obj = _cached_link(away, shared, "data.txt", "tracked with an unusable state dir\n")

    refused = _run(here, "gc", "--yes", "--grace-hours", "0")

    out = refused.stdout + refused.stderr
    assert refused.returncode != 0, out
    assert str(away) in out and "records no project id" in out, out
    assert obj.exists(), "gc removed an object named by a project registered without an id"


def test_pivot_gc_keeps_everything_for_a_link_into_the_cache_root_but_not_its_objects(
    tmp_path: pathlib.Path,
) -> None:
    """P10: the cache root can be a broad scratch directory that also holds the user's data
    (`PIVOT_CACHE_DIR=/scratch`). A project's `data -> /scratch/datasets/P` leaves the project
    like any other link, and the records under it are invisible to the walk, so the collection
    keeps every object. Only a link into `<cache>/files` is the cache's own."""
    scratch = tmp_path / "scratch"
    root = _project(tmp_path / "project")
    env = _env(PIVOT_CACHE_DIR=str(scratch))
    _ok(_run(root, "repro", env=env))
    _ok(_run(root, "status", env=env))
    objects = scratch / "files"
    tracked = _publish(objects, tmp_path, "a tracked dataset under the scratch root\n")
    datasets = scratch / "datasets" / "P"
    datasets.mkdir(parents=True)
    track.write_pvt_file(
        datasets / "data.txt.pvt", track.PvtData(path="data.txt", hash=tracked, size=40)
    )
    (datasets / "data.txt").symlink_to(cache.get_cache_path(objects, tracked))
    (root / "datasets").symlink_to(datasets)

    collected = _run(root, "gc", "--yes", "--grace-hours", "0", env=env)

    out = collected.stdout + collected.stderr
    assert collected.returncode == 0, out
    assert "leave the project" in out, out
    assert cache.get_cache_path(objects, tracked).exists(), "gc removed a tracked object"


def _changes_after_publishing(
    proc: subprocess.Popen[str], obj: pathlib.Path, source: pathlib.Path, text: str
) -> None:
    """While the test holds the naming lock: once the writer has published ``obj`` (hashed
    before its unit), remove it as a collection would and change ``source``, so the writer's
    re-publish inside the unit copies bytes that no longer match the key it hashed."""
    deadline = time.monotonic() + 120
    while not obj.exists():
        assert proc.poll() is None, f"the writer exited before publishing: {proc.communicate()[0]}"
        assert time.monotonic() < deadline, "the writer published nothing outside its unit"
        time.sleep(0.05)
    time.sleep(0.5)
    obj.unlink()
    source.chmod(0o644)
    source.write_text(text)


def test_pivot_track_refuses_to_republish_a_file_that_changed_since_it_was_hashed(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot track` hashes before its unit and re-publishes inside it if a collection removed
    the object meanwhile. If the file changed in that gap, the copy would store new bytes under
    the old key, and every project naming that key would get them: it refuses instead."""
    root = _project(tmp_path / "project")
    (root / "first.txt").write_text("tracked first\n")
    _ok(_run(root, "track", "first.txt"))
    cache_dir = _register_fresh(root)
    notes = root / "notes.txt"
    notes.write_text("the bytes pivot hashed\n")
    obj = _object_for(root, "the bytes pivot hashed\n")

    with gc.store_snapshot(cache_dir) as snapshot, gc.naming_exclusive(snapshot):
        proc = _start(root, "track", "notes.txt", env=_env())
        try:
            _changes_after_publishing(proc, obj, notes, "bytes written after the hash\n")
        except BaseException:
            proc.kill()
            raise
    out, _ = proc.communicate(timeout=180)

    assert proc.returncode != 0, out
    assert "changed after pivot hashed it" in out, out
    assert not (root / "notes.txt.pvt").exists(), "a .pvt names bytes that do not match its key"
    assert not obj.exists() or cache.hash_file(obj)[0] == obj.parent.name + obj.name


def test_pivot_commit_refuses_to_republish_an_output_that_changed_since_it_was_hashed(
    tmp_path: pathlib.Path,
) -> None:
    """The same boundary at the commit sites: `pivot commit` (and a stage commit, through the
    same store) hashes before its unit and re-publishes inside it with the bytes checked."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    (root / "data" / "raw" / "input.txt").write_text("hashed before the unit\n")
    _ok(_run(root, "repro", "--no-commit"))
    cache_dir = _register_fresh(root)
    before = _locks(root)
    output = root / "data" / "shadow" / "shout.txt"
    obj = _object_for(root, "HASHED BEFORE THE UNIT\n")

    with gc.store_snapshot(cache_dir) as snapshot, gc.naming_exclusive(snapshot):
        proc = _start(root, "commit", env=_env())
        try:
            _changes_after_publishing(proc, obj, output, "REWRITTEN AFTER THE HASH\n")
        except BaseException:
            proc.kill()
            raise
    out, _ = proc.communicate(timeout=180)

    assert proc.returncode != 0, out
    assert "changed after pivot hashed it" in out, out
    assert _locks(root) == before, "a lock names bytes that do not match its key"
    assert not obj.exists() or cache.hash_file(obj)[0] == obj.parent.name + obj.name


# =============================================================================
# Round 7: every restore registers; one register entry per host and mount view
# =============================================================================


def _git(root: pathlib.Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def test_a_clone_whose_repro_only_restores_registers_before_it_links(
    tmp_path: pathlib.Path,
) -> None:
    """P16: a fresh clone whose `pivot repro` only restores a stage output from its committed
    lock links that cache object into its workspace. Its project must be registered first, or
    another project's `pivot gc` never reads the clone's lock and removes the object."""
    shared = tmp_path / "shared-cache"
    origin = _shared_project(tmp_path / "origin", shared)
    config = origin / ".pivot" / "config.yaml"
    config.write_text(config.read_text() + "core:\n  run_history_retention: 1\n")
    (origin / ".gitignore").write_text("data/shadow/\n__pycache__/\n")
    for version in ("v1", "v2"):
        (origin / "data" / "raw" / "input.txt").write_text(f"{version}\n")
        _ok(_run(origin, "repro"))
        _git(origin, "add", "-A")
        _git(origin, "commit", "-qm", version)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "checkout", "-q", "HEAD~1")
    _ok(_run(clone, "repro"))
    output = clone / "data" / "shadow" / "shout.txt"
    assert output.is_symlink() and output.read_text() == "V1\n"
    obj = output.resolve()

    collected = _run(origin, "gc", "--yes", "--grace-hours", "0")

    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert obj.exists(), "gc removed the object a clone's restore linked"


def test_a_clone_whose_repro_restores_a_tracked_file_registers_before_it_links(
    tmp_path: pathlib.Path,
) -> None:
    """P16, the tracked-file restore: `pivot repro --checkout-missing` in a fresh clone links
    the tracked file its committed `.pvt` names. The stage output is committed, so the run
    restores nothing else; the tracked restore alone must register the clone."""
    shared = tmp_path / "shared-cache"
    origin = _shared_project(tmp_path / "origin", shared)
    (origin / ".gitignore").write_text("data.txt\n__pycache__/\n")
    _ok(_run(origin, "repro"))
    for version in ("v1", "v2"):
        (origin / "data.txt").unlink(missing_ok=True)
        (origin / "data.txt").write_text(f"tracked {version}\n")
        _ok(_run(origin, "track", "--force", "data.txt"))
        _git(origin, "add", "-A")
        _git(origin, "commit", "-qm", version)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "checkout", "-q", "HEAD~1")
    _ok(_run(clone, "repro", "--checkout-missing"))
    restored = clone / "data.txt"
    assert restored.is_symlink() and restored.read_text() == "tracked v1\n"
    obj = restored.resolve()

    collected = _run(origin, "gc", "--yes", "--grace-hours", "0")

    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert obj.exists(), "gc removed the tracked object a clone's restore linked"


def _in_view(
    root: pathlib.Path, *args: str, hostname: str | None = None, setup: str = ""
) -> subprocess.CompletedProcess[str]:
    """Run `pivot ARGS` in ``root`` as this user in a new mount namespace, after root runs
    ``setup`` there: another host's (with ``hostname``) or a container's view of the project."""
    if subprocess.run(["sudo", "-n", "true"], capture_output=True, check=False).returncode:
        pytest.skip("a mount namespace needs passwordless sudo")
    user_command = shlex.join(
        [
            "sudo",
            "-n",
            "-u",
            getpass.getuser(),
            "env",
            f"HOME={os.environ['HOME']}",
            f"PATH={os.environ['PATH']}",
            _PIVOT,
            *args,
        ]
    )
    script = "\n".join(
        [
            "set -e",
            f"hostname {shlex.quote(hostname)}" if hostname else "",
            setup,
            f"cd {shlex.quote(str(root))}",
            f"exec {user_command}",
        ]
    )
    namespaces = ["-m", "-u"] if hostname else ["-m"]
    return subprocess.run(
        ["sudo", "-n", "unshare", *namespaces, "--", "bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _tracked_object(record: pathlib.Path, cache_dir: pathlib.Path) -> pathlib.Path:
    data = track.read_pvt_file(record)
    assert data is not None, f"{record} is not a valid .pvt"
    return cache.get_cache_path(cache_dir / "files", data["hash"])


def test_pivot_gc_refuses_another_hosts_view_with_a_mount_it_cannot_see(
    tmp_path: pathlib.Path,
) -> None:
    """P17a: a project shared by two hosts. node2 mounts its local disk under the project and
    tracks a file there; this host sees an empty directory at that path. node2's entry records
    the mount, which this host cannot check, so a collection here must refuse."""
    shared = tmp_path / "shared-cache"
    root = _shared_project(tmp_path / "project", shared)
    local = tmp_path / "node2-local"
    local.mkdir()
    (local / "big.csv").write_text("only on node2's local disk\n")
    (root / "data" / "node2").mkdir()
    setup = f"mount --bind {shlex.quote(str(local))} {shlex.quote(str(root / 'data' / 'node2'))}"
    _ok(_in_view(root, "track", "data/node2/big.csv", hostname="pivot-node2", setup=setup))
    obj = _tracked_object(local / "big.csv.pvt", shared)
    assert obj.exists()

    collected = _run(root, "gc", "--yes", "--grace-hours", "0")

    out = collected.stdout + collected.stderr
    assert collected.returncode != 0, out
    assert "pivot-node2" in out and str(root / "data" / "node2") in out, out
    assert obj.exists(), "gc removed an object named under another host's mount"


def test_pivot_gc_keeps_a_mount_views_evidence_after_this_host_writes(
    tmp_path: pathlib.Path,
) -> None:
    """P17b: the same, from a mount namespace that keeps this hostname (a container with host
    UTS). This host then names an object of its own in the project; its registration must not
    replace the other view's evidence, so a collection still refuses."""
    shared = tmp_path / "shared-cache"
    root = _shared_project(tmp_path / "project", shared)
    local = tmp_path / "view-local"
    local.mkdir()
    (local / "big.csv").write_text("only in the other mount view\n")
    (root / "data" / "view").mkdir()
    setup = f"mount --bind {shlex.quote(str(local))} {shlex.quote(str(root / 'data' / 'view'))}"
    _ok(_in_view(root, "track", "data/view/big.csv", setup=setup))
    obj = _tracked_object(local / "big.csv.pvt", shared)
    (root / "other.txt").write_text("this host's own file\n")
    _ok(_run(root, "track", "other.txt"))

    collected = _run(root, "gc", "--yes", "--grace-hours", "0")

    out = collected.stdout + collected.stderr
    assert collected.returncode != 0, out
    assert str(root / "data" / "view") in out, out
    assert obj.exists(), "this host's write erased the other view's mount evidence"


def test_pivot_gc_refuses_a_host_local_copy_at_the_same_path(tmp_path: pathlib.Path) -> None:
    """P15: node2 holds a `cp -a` copy of the project at the same path, project id included,
    and runs a new output there. The copy is another project: its id must not stand for this
    host's project, whose records do not name node2's new output."""
    shared = tmp_path / "shared-cache"
    work = tmp_path / "work"
    work.mkdir()
    root = _shared_project(work / "project", shared)
    _ok(_run(root, "repro"))
    node2_work = tmp_path / "node2-work"
    subprocess.run(["cp", "-a", str(work), str(node2_work)], check=True)
    (node2_work / "project" / "data" / "raw" / "input.txt").write_text("node2 run\n")
    setup = f"mount --bind {shlex.quote(str(node2_work))} {shlex.quote(str(work))}"
    _ok(_in_view(root, "repro", hostname="pivot-node2", setup=setup))
    node2_output = node2_work / "project" / "data" / "shadow" / "shout.txt"
    assert node2_output.is_symlink()
    obj = pathlib.Path(os.readlink(node2_output))
    assert obj.exists()

    collected = _run(root, "gc", "--yes", "--grace-hours", "0")

    out = collected.stdout + collected.stderr
    assert collected.returncode != 0, out
    assert "pivot-node2" in out, out
    assert obj.exists(), "gc read this host's copy for node2's project"


def test_pivot_gc_refuses_another_views_cache_mount(tmp_path: pathlib.Path) -> None:
    """The cache's own mounts, per view: another host mounts a disk at the cache's ref
    directory and publishes a ref there. This host cannot see that mount, so a collection
    refuses rather than collect the object the ref keeps."""
    shared = tmp_path / "shared-cache"
    root = _shared_project(tmp_path / "project", shared)
    _cached_link(root, shared, "first.txt", "makes the cache\n")
    key = _publish(shared / "files", tmp_path, "kept only by node2's ref\n")
    refs = shared / "files" / "refs"
    refs.mkdir()
    target = cache.get_cache_path(shared / "files", key)
    (root / "second.txt").write_text("registers node2's view\n")
    setup = "\n".join(
        [
            f"mount -t tmpfs -o uid={os.getuid()},gid={os.getgid()} tmpfs {shlex.quote(str(refs))}",
            f"ln -s {shlex.quote(str(target))} {shlex.quote(str(refs / 'store'))}",
        ]
    )
    _ok(_in_view(root, "track", "second.txt", hostname="pivot-node2", setup=setup))

    collected = _run(root, "gc", "--yes", "--grace-hours", "0")

    out = collected.stdout + collected.stderr
    assert collected.returncode != 0, out
    assert str(refs) in out, out
    assert target.exists(), "gc removed an object another view's cache mount keeps"


# =============================================================================
# Round 8: an inode number is not proof; each view proves what it last wrote
# =============================================================================


def _sudo(*args: str) -> None:
    subprocess.run(["sudo", "-n", *args], check=True, capture_output=True)


def _image_project(path: pathlib.Path, shared: pathlib.Path) -> pathlib.Path:
    """A project built in an image layer that ran `pivot repro` at build time, so its state
    directory, project id and view nonce are part of the image."""
    root = _shared_project(path, shared)
    (root / "data" / "raw" / "input.txt").write_text("image\n")
    _ok(_run(root, "repro"))
    return root


def _shared_object(shared: pathlib.Path, scratch: pathlib.Path, text: str) -> pathlib.Path:
    """Where the object holding ``text`` lives in the shared cache."""
    probe = scratch / "object-probe"
    probe.write_text(text)
    key, _ = cache.hash_file(probe)
    probe.unlink()
    return cache.get_cache_path(shared / "files", key)


def test_pivot_gc_refuses_another_containers_project_from_the_same_image(
    tmp_path: pathlib.Path,
) -> None:
    """P18c: two containers from one image (an overlay over a lower layer that holds the
    project). Their state directories show the same inode, but each container writes its own
    upper layer. c1's collection must not read its own layer as c2's project."""
    if subprocess.run(["sudo", "-n", "true"], capture_output=True, check=False).returncode:
        pytest.skip("an overlay mount needs passwordless sudo")
    shared = tmp_path / "shared-cache"
    image = tmp_path / "image"
    image.mkdir()
    _image_project(image / "project", shared)
    work = tmp_path / "work"
    work.mkdir()
    root = work / "project"

    def overlay(name: str) -> str:
        (tmp_path / name / "upper").mkdir(parents=True)
        (tmp_path / name / "work").mkdir()
        options = f"lowerdir={image},upperdir={tmp_path / name / 'upper'},workdir={tmp_path / name / 'work'}"
        return f"mount -t overlay overlay -o {shlex.quote(options)} {shlex.quote(str(work))}"

    c1, c2 = overlay("c1"), overlay("c2")
    try:
        write_c2 = f"{c2}\necho c2 > {shlex.quote(str(root / 'data' / 'raw' / 'input.txt'))}"
        _ok(_in_view(root, "repro", hostname="pivot-c2", setup=write_c2))
        c2_output = tmp_path / "c2" / "upper" / "project" / "data" / "shadow" / "shout.txt"
        obj = pathlib.Path(os.readlink(c2_output))
        assert obj.exists()

        collected = _in_view(
            root, "gc", "--yes", "--grace-hours", "0", hostname="pivot-c1", setup=c1
        )
    finally:
        # overlayfs leaves a root-owned directory in each workdir that pytest cannot remove.
        _sudo("rm", "-rf", str(tmp_path / "c1" / "work"), str(tmp_path / "c2" / "work"))

    out = collected.stdout + collected.stderr
    assert collected.returncode != 0, out
    assert "pivot-c2" in out, out
    assert obj.exists(), "c1's gc read its own layer as c2's project"


@contextlib.contextmanager
def _disk_images(tmp_path: pathlib.Path, shared: pathlib.Path) -> Iterator[pathlib.Path]:
    """An ext4 image holding a project, loop-mounted at ``tmp_path/work`` (this host's view);
    yields the mount point and unmounts it at the end."""
    if subprocess.run(["sudo", "-n", "true"], capture_output=True, check=False).returncode:
        pytest.skip("loop-mounting a disk image needs passwordless sudo")
    work = tmp_path / "work"
    work.mkdir()
    disk = tmp_path / "disk1.img"
    subprocess.run(["truncate", "-s", "128M", str(disk)], check=True)
    subprocess.run(["/usr/sbin/mkfs.ext4", "-q", "-F", str(disk)], check=True)
    _sudo("mount", "-o", "loop", str(disk), str(work))
    try:
        _sudo("chown", f"{os.getuid()}:{os.getgid()}", str(work))
        yield work
    finally:
        _sudo("umount", str(work))


def _clone_disk(tmp_path: pathlib.Path) -> str:
    """Clone disk1 (unmounted for the copy, then mounted again) and return the setup that
    mounts the clone at the same path in another view: a VM created from node1's disk."""
    work = tmp_path / "work"
    subprocess.run(["sync"], check=True)
    _sudo("umount", str(work))
    subprocess.run(
        ["cp", "--sparse=always", str(tmp_path / "disk1.img"), str(tmp_path / "disk2.img")],
        check=True,
    )
    _sudo("mount", "-o", "loop", str(tmp_path / "disk1.img"), str(work))
    return f"mount -o loop {shlex.quote(str(tmp_path / 'disk2.img'))} {shlex.quote(str(work))}"


def test_pivot_gc_refuses_a_vm_cloned_from_this_disk(tmp_path: pathlib.Path) -> None:
    """P18v: node2 is a VM created from node1's disk image: the same path, the same inode
    numbers, the same project id. node2 runs a new output; node1's collection must not read
    node1's own records as node2's project."""
    shared = tmp_path / "shared-cache"
    with _disk_images(tmp_path, shared) as work:
        root = _image_project(work / "project", shared)
        node2 = _clone_disk(tmp_path)
        write = f"{node2}\necho node2 > {shlex.quote(str(root / 'data' / 'raw' / 'input.txt'))}"
        repro = _in_view(root, "repro", hostname="pivot-node2", setup=write)
        _ok(repro)
        obj = _shared_object(shared, tmp_path, "NODE2\n")
        assert obj.exists()

        collected = _run(root, "gc", "--yes", "--grace-hours", "0")

    out = collected.stdout + collected.stderr
    assert collected.returncode != 0, out
    assert "pivot-node2" in out, out
    assert obj.exists(), "node1's gc read node1's records as node2's project"


def test_pivot_gc_refuses_a_stale_clone_after_the_original_writes_again(
    tmp_path: pathlib.Path,
) -> None:
    """P18v, clone after write: node1 registers, node2 is cloned from node1's disk, then node1
    registers again and runs a new output. node2's clone holds node1's old nonce, so node2's
    collection must refuse node1's entry instead of reading its stale copy as node1's records."""
    shared = tmp_path / "shared-cache"
    with _disk_images(tmp_path, shared) as work:
        root = _image_project(work / "project", shared)
        node2 = _clone_disk(tmp_path)
        (root / "data" / "raw" / "input.txt").write_text("node1 again\n")
        _ok(_run(root, "repro"))
        obj = _shared_object(shared, tmp_path, "NODE1 AGAIN\n")
        assert obj.exists()

        collected = _in_view(
            root, "gc", "--yes", "--grace-hours", "0", hostname="pivot-node2", setup=node2
        )

    out = collected.stdout + collected.stderr
    assert collected.returncode != 0, out
    assert obj.exists(), "node2's gc read its stale clone as node1's project"


# =============================================================================
# Round 10: a directory's members live in its listing, and the mount table is watched
# =============================================================================

_DIRECTORY_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def split(text: str) -> Annotated[dict[str, str], directory]:
    return {"a.txt": text, "b.txt": text.upper(), "sub/c.txt": "c"}


with Pipeline("shadow") as pipeline:
    split(pipeline.input("input.txt", t=str))
"""


def _directory_record(root: pathlib.Path) -> types.DirHash:
    """The directory output's record in the stage lock `pivot repro` wrote."""
    data = lock.StageLock("shadow/split", lock.get_stages_dir(root / ".pivot")).read(
        cache_dir=root / ".pivot" / "cache" / "files"
    )
    assert data is not None, "the stage wrote no lock"
    record = next(iter(data["output_hashes"].values()))
    assert types.is_dir_hash(record), "the stage's output is a directory"
    return record


def _directory_project(path: pathlib.Path) -> pathlib.Path:
    root = _project(path)
    (root / "pipeline.py").write_text(_DIRECTORY_PIPELINE)
    _ok(_run(root, "repro"))
    return root


def test_pivot_gc_keeps_the_members_a_directorys_listing_names(tmp_path: pathlib.Path) -> None:
    """#523 records a directory by its listing: the lock holds the listing's hash, its size and
    its file count, and the members are named only inside the listing. A collection must read
    it and keep every member, the listing object itself and the whole-directory tree."""
    root = _directory_project(tmp_path / "project")
    objects = root / ".pivot" / "cache" / "files"
    record = _directory_record(root)
    members = listing.read(None, record, objects)
    assert len(members["files"]) == 3, members
    member_objects = [cache.get_cache_path(objects, entry["hash"]) for entry in members["files"]]
    assert all(path.exists() for path in member_objects), member_objects
    dead = _publish(objects, tmp_path, "named by nothing\n")

    collected = _run(root, "gc", "--yes", "--grace-hours", "0")

    out = collected.stdout + collected.stderr
    assert collected.returncode == 0, out
    for path in member_objects:
        assert path.exists(), f"gc removed {path}, a member of the directory the lock records"
    assert cache.get_cache_path(objects, record["hash"]).exists(), "gc removed the listing object"
    assert not cache.get_cache_path(objects, dead).exists(), out


def test_pivot_gc_refuses_a_directory_whose_listing_is_gone(tmp_path: pathlib.Path) -> None:
    """With neither the committed listing nor the cache's copy, the files of a recorded
    directory are unknown. A collection must refuse and remove nothing, never take the members
    for dead."""
    root = _directory_project(tmp_path / "project")
    objects = root / ".pivot" / "cache" / "files"
    record = _directory_record(root)
    members = listing.read(None, record, objects)
    member_objects = [cache.get_cache_path(objects, entry["hash"]) for entry in members["files"]]
    committed = listing.output_listing_path(
        lock.get_stages_dir(root / ".pivot"), types.ArtifactIdentity("shadow/split", None)
    )
    assert committed.is_file(), committed
    committed.unlink()
    cache.clear_path(cache.get_cache_path(objects, record["hash"]))
    dead = _publish(objects, tmp_path, "named by nothing\n")

    refused = _run(root, "gc", "--yes", "--grace-hours", "0")

    out = refused.stdout + refused.stderr
    assert refused.returncode != 0, out
    assert record["hash"] in out and "no listing of it" in out, out
    flat = " ".join(out.split())
    assert "until its listing is back" in flat and "register entry" not in flat, out
    for path in member_objects:
        assert path.exists(), f"gc removed {path} with no listing to prove it dead"
    assert cache.get_cache_path(objects, dead).exists(), "a refused collection removed an object"


_MOUNT_DRIVER = """\
import pathlib
import subprocess
import sys

from pivot.storage import naming

root, state_dir, cache_dir, source, target = (pathlib.Path(a) for a in sys.argv[1:6])
naming.register_project(cache_dir, root, state_dir)
subprocess.run(
    ["sudo", "-n", "mount", "--bind", str(source), str(target)], check=True, capture_output=True
)
naming.register_project(cache_dir, root, state_dir)
print(naming.project_register_path(cache_dir, root, state_dir))
"""


def test_a_naming_unit_records_a_mount_that_appeared_since_the_last_one(
    tmp_path: pathlib.Path,
) -> None:
    """The registration memo lets a unit skip the mount-table read, so it watches the table
    instead (`_watch_mount_table`): the kernel reports every mount in this namespace. A unit
    after a mount under the project must register in full and record that mount, or a
    collection would read a register that no longer describes where this project's records
    are."""
    if subprocess.run(["sudo", "-n", "true"], capture_output=True, check=False).returncode:
        pytest.skip("a mount namespace needs passwordless sudo")
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    source = tmp_path / "late-disk"
    source.mkdir()
    target = root / "data" / "late"
    target.mkdir()
    driver = tmp_path / "driver.py"
    driver.write_text(_MOUNT_DRIVER)
    user_command = shlex.join(
        [
            "sudo",
            "-n",
            "-u",
            getpass.getuser(),
            "env",
            f"HOME={os.environ['HOME']}",
            f"PATH={os.environ['PATH']}",
            sys.executable,
            str(driver),
            str(root),
            str(root / ".pivot"),
            str(root / ".pivot" / "cache"),
            str(source),
            str(target),
        ]
    )
    done = subprocess.run(
        ["sudo", "-n", "unshare", "-m", "--", "bash", "-c", f"set -e\n{user_command}"],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    out = done.stdout + done.stderr
    assert done.returncode == 0, out
    entry = json.loads(pathlib.Path(done.stdout.strip().splitlines()[-1]).read_text())
    assert str(target.resolve()) in entry["mounts"], entry


def test_pivot_gc_reads_a_committed_listing_the_cache_no_longer_holds(
    tmp_path: pathlib.Path,
) -> None:
    """A clone has the listing files git carries and an empty cache; a cache someone emptied
    is the same. The committed listing beside the lock names the directory's files, so a
    collection reads it instead of refusing."""
    root = _directory_project(tmp_path / "project")
    objects = root / ".pivot" / "cache" / "files"
    record = _directory_record(root)
    members = listing.read(None, record, objects)
    member_objects = [cache.get_cache_path(objects, entry["hash"]) for entry in members["files"]]
    committed = listing.output_listing_path(
        lock.get_stages_dir(root / ".pivot"), types.ArtifactIdentity("shadow/split", None)
    )
    assert committed.is_file(), committed
    cache.clear_path(cache.get_cache_path(objects, record["hash"]))
    dead = _publish(objects, tmp_path, "named by nothing\n")

    collected = _run(root, "gc", "--yes", "--grace-hours", "0")

    out = collected.stdout + collected.stderr
    assert collected.returncode == 0, out
    for path in member_objects:
        assert path.exists(), f"gc removed {path}, a member the committed listing names"
    assert not cache.get_cache_path(objects, dead).exists(), out


def test_pivot_gc_keeps_the_members_of_a_tracked_directory(tmp_path: pathlib.Path) -> None:
    """`pivot track` of a directory writes a `.pvt` holding the listing's record and the
    listing beside it (`<dir>.pvt.listing`). A collection keeps every member it names."""
    root = _project(tmp_path / "project")
    tracked = root / "data" / "images"
    (tracked / "sub").mkdir(parents=True)
    (tracked / "a.txt").write_text("tracked member a\n")
    (tracked / "sub" / "b.txt").write_text("tracked member b\n")
    _ok(_run(root, "track", "data/images"))
    objects = root / ".pivot" / "cache" / "files"
    members = [_object_for(root, "tracked member a\n"), _object_for(root, "tracked member b\n")]
    assert all(path.exists() for path in members), members
    dead = _publish(objects, tmp_path, "named by nothing\n")

    collected = _run(root, "gc", "--yes", "--grace-hours", "0")

    out = collected.stdout + collected.stderr
    assert collected.returncode == 0, out
    for path in members:
        assert path.exists(), f"gc removed {path}, a member of a tracked directory"
    assert not cache.get_cache_path(objects, dead).exists(), out


_MEMBER_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def split(text: str) -> Annotated[dict[str, str], directory]:
    return {"a.txt": text, "inner.pvt": MEMBER}


with Pipeline("shadow") as pipeline:
    split(pipeline.input("input.txt", t=str))
"""


@pytest.mark.parametrize(
    "member",
    [
        pytest.param(
            "path: inner\nhash: 0123456789abcdef\nsize: 3\nnfiles: 1\n", id="names a directory"
        ),
        pytest.param("<<<<<<< not a pvt\n: [\n", id="does not parse"),
    ],
)
def test_pivot_gc_reads_a_stages_own_pvt_member_as_its_data(
    tmp_path: pathlib.Path, member: str
) -> None:
    """#540: a stage may write a `*.pvt` into its own directory output; it is the stage's data,
    not a tracked file. One that names a directory no listing describes, or one that does not
    parse, must not make the collection refuse, and the output's own objects stay."""
    root = _project(tmp_path / "project")
    (root / "pipeline.py").write_text(f"MEMBER = {member!r}\n" + _MEMBER_PIPELINE)
    _ok(_run(root, "repro"))
    objects = root / ".pivot" / "cache" / "files"
    record = _directory_record(root)
    members = listing.read(None, record, objects)
    member_objects = [cache.get_cache_path(objects, entry["hash"]) for entry in members["files"]]
    dead = _publish(objects, tmp_path, "named by nothing\n")

    collected = _run(root, "gc", "--yes", "--grace-hours", "0")

    out = collected.stdout + collected.stderr
    assert collected.returncode == 0, out
    for path in member_objects:
        assert path.exists(), f"gc removed {path}, a member of the stage's output"
    assert not cache.get_cache_path(objects, dead).exists(), out


_GIT = ["git", "-c", "user.email=gc@test", "-c", "user.name=gc", "-c", "merge.conflictstyle=merge"]


def _git(root: pathlib.Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*_GIT, *args], cwd=root, check=check, capture_output=True, text=True)


def _fill_images(images: pathlib.Path, tag: str) -> None:
    if images.exists():
        cache.clear_path(images)
    images.mkdir(parents=True)
    for name in ("a", "b"):
        (images / f"{name}-{tag}.bin").write_bytes(f"{name} image bytes for {tag}\n".encode() * 50)


def test_pivot_gc_refuses_a_tracked_directory_left_in_a_merge_conflict(
    tmp_path: pathlib.Path,
) -> None:
    """P20: a tracked directory re-tracked on two branches; `git merge` leaves conflict markers
    in its `.pvt` and `.pvt.listing`. The data is not in git, so the feature branch's files are
    in the cache alone, named only by feature's listing. A collection must refuse, naming the
    `.pvt`, and remove none of them; taking feature's side then restores its files."""
    root = _project(tmp_path / "project")
    (root / ".gitignore").write_text("/data/images/\n")
    images = root / "data" / "images"
    _fill_images(images, "main")
    _ok(_run(root, "track", "data/images"))
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "track images (main)")
    trunk = _git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    _git(root, "checkout", "-q", "-b", "feature")
    _fill_images(images, "feat")
    _ok(_run(root, "track", "--force", "data/images"))
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "re-track images (feature)")
    objects = root / ".pivot" / "cache" / "files"
    feature_listing = listing.decode((root / "data" / "images.pvt.listing").read_bytes())
    feature_objects = [
        cache.get_cache_path(objects, entry["hash"]) for entry in feature_listing["files"]
    ]
    _git(root, "checkout", "-q", trunk)
    _ok(_run(root, "checkout", "--force", "data/images"))
    _fill_images(images, "main2")
    _ok(_run(root, "track", "--force", "data/images"))
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "re-track images (main2)")
    merge = _git(root, "merge", "feature", check=False)
    conflicted = _git(root, "diff", "--name-only", "--diff-filter=U").stdout.split()
    assert merge.returncode != 0 and "data/images.pvt" in conflicted, merge.stdout + merge.stderr
    assert all(path.exists() for path in feature_objects), feature_objects

    refused = _run(root, "gc", "--yes", "--grace-hours", "0")

    out = refused.stdout + refused.stderr
    flat = " ".join(out.split())
    assert refused.returncode != 0, out
    assert str(root / "data" / "images.pvt") in flat and "does not parse" in flat, out
    assert "Fix or resolve that record" in flat and "register entry" not in flat, out
    assert "keeping every cache object" not in out, out
    for path in feature_objects:
        assert path.exists(), f"gc removed {path}, a file only feature's listing names"
    _git(root, "checkout", "--theirs", "data/images.pvt", "data/images.pvt.listing")
    _git(root, "add", "data/images.pvt", "data/images.pvt.listing")
    _ok(_run(root, "checkout", "--force", "data/images"))
    assert sorted(path.name for path in images.iterdir()) == ["a-feat.bin", "b-feat.bin"]


_TRACKED_INPUT_PIPELINE = """\
import pathlib

from pivot import Pipeline, stage


@stage
def lane(server: pathlib.Path) -> str:
    return (server / "grade.py").read_text()


with Pipeline("p") as pipeline:
    lane(pipeline.input("server", path="data/raw/server", t=pathlib.Path))
"""


def test_pivot_gc_keeps_a_tracked_inputs_files_that_pivotignore_names(
    tmp_path: pathlib.Path,
) -> None:
    """#539: a stage's lock records a directory input by a listing that leaves out what
    `.pivotignore` names (`=inputs/<path>.listing`). A tracked directory's members come from
    its whole `.pvt.listing`, never from that filtered one, so a collection keeps the tracked
    input's `.pyc`, and a checkout restores it."""
    root = _project(tmp_path / "project")
    (root / ".pivotignore").write_text("*.pyc\n__pycache__/\n")
    (root / "pipeline.py").write_text(_TRACKED_INPUT_PIPELINE)
    tracked = root / "data" / "raw" / "server"
    (tracked / "__pycache__").mkdir(parents=True)
    (tracked / "grade.py").write_text("print('grade')\n")
    pyc = tracked / "__pycache__" / "grade.cpython-312.pyc"
    pyc.write_bytes(b"\x00bytecode")
    _ok(_run(root, "track", "data/raw/server"))
    _ok(_run(root, "repro"))
    pyc_object = cache.get_cache_path(root / ".pivot" / "cache" / "files", cache.hash_file(pyc)[0])
    filtered = list((root / ".pivot" / "stages" / listing.INPUTS_DIR).glob(f"*{listing.SUFFIX}"))
    assert filtered, "the stage's lock records its directory input by a listing"
    assert all("grade.cpython-312.pyc" not in path.read_text() for path in filtered), filtered
    assert pyc_object.exists(), "track cached the .pyc"
    cache.clear_path(tracked)  # nothing but the records names the object now: no workspace link

    collected = _run(root, "gc", "--yes", "--grace-hours", "0")

    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert pyc_object.exists(), "gc read the tracked input's members from the filtered listing"
    _ok(_run(root, "checkout", "data/raw/server"))
    assert pyc.read_bytes() == b"\x00bytecode"


# =============================================================================
# Round 13: every run registers its project before any stage runs (U1)
# =============================================================================

_GATED_PIPELINE = """\
import os
import pathlib
import time

from pivot import Pipeline, stage


def _hold(name: str) -> None:
    gate = os.environ.get("GC_TEST_GATE")
    if gate and os.environ.get("GC_TEST_GATE_STAGE") == name:
        pathlib.Path(gate, "started").write_text("")
        deadline = time.monotonic() + 120
        while not pathlib.Path(gate, "go").exists() and time.monotonic() < deadline:
            time.sleep(0.02)


@stage
def up(text: str) -> str:
    _hold("up")
    return text.upper()


@stage
def down(text: str) -> str:
    _hold("down")
    return text + " down VERSION"


with Pipeline("ug") as pipeline:
    down(up(pipeline.input("input.txt", t=str)))
"""


def _registered(cache_dir: pathlib.Path, root: pathlib.Path) -> bool:
    register = naming.register_root(cache_dir) / naming.REGISTER_DIR
    return register.is_dir() and any(
        json.loads(entry.read_text()).get("root") == str(root) for entry in register.glob("*.json")
    )


def _unregister(cache_dir: pathlib.Path, root: pathlib.Path) -> None:
    """Leave ``root`` out of the register, as a project last run by a pivot release (which
    writes no entry) or one whose entry a user deleted is."""
    for entry in (naming.register_root(cache_dir) / naming.REGISTER_DIR).glob("*.json"):
        if json.loads(entry.read_text()).get("root") == str(root):
            entry.unlink()
            entry.with_suffix(".nonce").unlink(missing_ok=True)
    assert not _registered(cache_dir, root)


def _unlist(root: pathlib.Path, cache_dir: pathlib.Path) -> None:
    """Leave ``cache_dir`` out of the caches ``root``'s state directory lists, as a pivot release
    (which keeps no list) leaves it."""
    cache_id = naming.read_cache_id(cache_dir)
    for record in naming.cache_records(root / ".pivot"):
        if record.cache_id == cache_id:
            record.record.unlink()
    assert cache_id not in {record.cache_id for record in naming.cache_records(root / ".pivot")}


def _unregistered_gated_project(
    tmp_path: pathlib.Path, checkout: str = "symlink"
) -> tuple[pathlib.Path, pathlib.Path]:
    """A registered peer and a project out of the register, both on one external cache, the
    project with ``checkout`` mode; its two-stage pipeline has run once. Returns (peer, project)."""
    shared = tmp_path / "shared-cache"
    peer = _shared_project(tmp_path / "peer", shared)
    _ok(_run(peer, "repro"))
    root = _shared_project(tmp_path / "project", shared, checkout)
    # Content no record of the peer names: the peer's own stage shouts "hello", which would make
    # its lock keep the same object and hide a lost registration.
    (root / "data" / "raw" / "input.txt").write_text("only this project reads this\n")
    (root / "pipeline.py").write_text(_GATED_PIPELINE.replace("VERSION", "v1"))
    _ok(_run(root, "repro"))
    _unregister(shared, root)
    return peer, root


def _start_held(
    root: pathlib.Path, gate: pathlib.Path, stage: str, *args: str
) -> subprocess.Popen[str]:
    gate.mkdir()
    proc = _start(root, "repro", *args, env=_env(GC_TEST_GATE=str(gate), GC_TEST_GATE_STAGE=stage))
    _wait_for(gate / "started", proc, 120)
    return proc


@pytest.mark.parametrize("checkout", ["symlink", "copy"])
def test_a_project_out_of_the_register_registers_before_its_changed_stage_runs(
    tmp_path: pathlib.Path, checkout: str
) -> None:
    """U1: a project out of the register (last run by a pivot release) runs `pivot repro` with
    `down` changed and `up` skipped in place. While `down` runs, a peer's `pivot gc` reads the
    register. The project must already be in it, or gc removes `up`'s object, which only the
    project's lock names. With copy checkout no workspace link names the object either, so the
    registration alone keeps it."""
    peer, root = _unregistered_gated_project(tmp_path, checkout)
    shared = tmp_path / "shared-cache"
    up_out = root / "data" / "ug" / "up.txt"
    up_object = cache.get_cache_path(shared / "files", cache.hash_file(up_out)[0])
    assert up_object.exists(), up_object
    assert up_out.is_symlink() == (checkout == "symlink"), "the checkout mode was not applied"
    up_content = up_out.read_text()
    (root / "pipeline.py").write_text(_GATED_PIPELINE.replace("VERSION", "v2"))

    proc = _start_held(root, tmp_path / "gate", "down")
    try:
        registered = _registered(shared, root)
        collected = _run(peer, "gc", "--yes", "--grace-hours", "0")
    finally:
        (tmp_path / "gate" / "go").write_text("")
    _finish(proc)

    out = collected.stdout + collected.stderr
    assert collected.returncode == 0, out
    # The object first: in copy mode nothing but the registration can keep it.
    assert up_object.exists(), f"gc removed {up_object}, which the project's up.lock names"
    assert up_out.exists() and up_out.read_text() == up_content, "up's output is gone or dangles"
    assert registered, "the project was not in the register while its changed stage ran"


def test_a_forced_run_registers_before_its_first_stage_runs(tmp_path: pathlib.Path) -> None:
    """`--force` takes no run-cache lookup, so no stage takes a naming unit before it runs."""
    _peer, root = _unregistered_gated_project(tmp_path)
    shared = tmp_path / "shared-cache"

    proc = _start_held(root, tmp_path / "gate", "up", "--force")
    try:
        registered = _registered(shared, root)
    finally:
        (tmp_path / "gate" / "go").write_text("")
    _finish(proc)

    assert registered, "a forced run's project was not in the register while its stages ran"


def test_a_run_where_every_stage_skips_registers_its_project(tmp_path: pathlib.Path) -> None:
    """A stage skipped with its outputs in place takes no naming unit, so a run that skips every
    stage must still leave its project in the register: its locks name those outputs."""
    _peer, root = _unregistered_gated_project(tmp_path)
    shared = tmp_path / "shared-cache"

    out = _ok(_run(root, "repro"))

    assert "0 ran" in out, out
    assert _registered(shared, root), "a run that skipped every stage left its project unregistered"


# =============================================================================
# Round 14: the library path registers where it restores (L1) and refuses root= (L2)
# =============================================================================

_LIBRARY_RUN = """\
import pathlib
import sys

from pivot import config, project
from pivot.executor import core
from pivot.storage import naming


def _watch_first_restore(configured_cache: pathlib.Path, report: pathlib.Path) -> None:
    # At the first tracked-file restore, record whether the project is already in the register
    # of the cache that restore links from: the order, not only the end state.
    restore = core._restore_tracked_file

    def watched(*args, **kwargs):
        if not report.exists():
            entry = naming.project_register_path(
                configured_cache, project.get_project_root(), config.locate_state_dir()
            )
            report.write_text("registered" if entry is not None and entry.exists() else "absent")
        return restore(*args, **kwargs)

    core._restore_tracked_file = watched


if __name__ == "__main__":  # pivot starts processes with spawn, which re-imports this module
    _watch_first_restore(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
    core.run(cache_dir=pathlib.Path(sys.argv[1]), checkout_missing=True)
"""


def test_a_library_run_with_another_cache_registers_where_it_restores_tracked_files(
    tmp_path: pathlib.Path,
) -> None:
    """L1: `executor.core.run(cache_dir=C2, checkout_missing=True)` registers in C2, the engine's
    cache, but restores a missing tracked file from the configured cache C1. The project must be
    registered in C1 before that restore links C1's object, or a peer's `pivot gc` on C1
    removes it and the restored link dangles. A hook at the restore checks the order."""
    shared = tmp_path / "shared-cache"
    peer = _shared_project(tmp_path / "peer", shared)
    _ok(_run(peer, "repro"))
    root = _shared_project(tmp_path / "project", shared)
    tracked = root / "notes.txt"
    content = "tracked notes, in the configured cache only\n"
    tracked.write_text(content)
    _ok(_run(root, "track", "notes.txt"))
    tracked_object = cache.get_cache_path(shared / "files", cache.hash_file(tracked)[0])
    assert tracked_object.exists(), tracked_object
    _unregister(shared, root)
    # As a project last run by a pivot release, which keeps no list of caches: else the engine's
    # unit in C2 would register it in C1 through that list, before the restore's own registration.
    _unlist(root, shared)
    tracked.unlink()
    other_cache = tmp_path / "other-cache"
    other_cache.mkdir()
    driver = tmp_path / "library_run.py"
    driver.write_text(_LIBRARY_RUN)
    report = tmp_path / "registered-at-restore"

    done = subprocess.run(
        [sys.executable, str(driver), str(other_cache), str(shared), str(report)],
        cwd=root,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert done.returncode == 0, done.stdout + done.stderr
    assert tracked.read_text() == content, "the tracked file was not restored from C1"
    assert report.read_text() == "registered", "the restore linked C1's object before registering"
    registered = _registered(shared, root)
    collected = _run(peer, "gc", "--yes", "--grace-hours", "0")
    assert registered, "the project was not registered in the cache its restore linked from"
    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert tracked_object.exists(), "gc removed the object the project's .pvt names"


_ROOTED_MODULE = """\
import os
import pathlib

from pivot import Pipeline, stage


@stage
def made() -> str:
    return "made elsewhere"


ext = Pipeline("ext", root=pathlib.Path(os.environ["GC_TEST_ELSEWHERE"]))
with ext:
    made()
top = Pipeline("top", root=pathlib.Path(os.environ["GC_TEST_ROOT"]))
top.include(ext)
"""

_ROOTED_DRIVER = """\
import sys

from pivot import exceptions
from pivot.executor import core

import rooted_pipeline

if __name__ == "__main__":  # pivot starts processes with spawn, which re-imports this module
    try:
        core.run(pipeline=rooted_pipeline.top)
    except exceptions.PipelineConfigError as error:
        print(f"REFUSED: {error}")
        sys.exit(3)
"""


def test_a_library_run_refuses_an_included_pipeline_rooted_elsewhere(
    tmp_path: pathlib.Path,
) -> None:
    """L2: the CLI's discovery refuses a pipeline whose `root=` is not the project root. Given
    one through `executor.core.run(pipeline=...)`, the library ran it and wrote the included
    stage's lock under that other root, where no `pivot gc` reads it. It must refuse the same."""
    root = _project(tmp_path / "project")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (root / "rooted_pipeline.py").write_text(_ROOTED_MODULE)
    driver = root / "rooted_driver.py"
    driver.write_text(_ROOTED_DRIVER)

    done = subprocess.run(
        [sys.executable, str(driver)],
        cwd=root,
        # PYTHONPATH: the driver and every worker import the stage module by name
        env=_env(PYTHONPATH=str(root), GC_TEST_ELSEWHERE=str(elsewhere), GC_TEST_ROOT=str(root)),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    out = done.stdout + done.stderr
    assert done.returncode == 3 and "REFUSED" in out and "root=" in out, out
    assert not list(elsewhere.rglob("*.lock")), "a stage lock was written outside the project"


# =============================================================================
# Round 15: a project that uses two caches is checked in both (N1)
# =============================================================================


def _use_cache(
    root: pathlib.Path, cache_dir: pathlib.Path, *, retention: int | None = None
) -> None:
    """Point the project at ``cache_dir`` with copy checkout, so no workspace link keeps an
    object: only the project's records and its registration can. ``retention`` keeps that many
    runs of history (and their run-cache rows)."""
    core = "" if retention is None else f"core:\n  run_history_retention: {retention}\n"
    (root / ".pivot" / "config.yaml").write_text(
        f"cache:\n  dir: {cache_dir}\n  checkout_mode:\n  - copy\n{core}"
    )


class _TwoCaches(NamedTuple):
    peer: pathlib.Path
    root: pathlib.Path
    c1: pathlib.Path
    c2: pathlib.Path
    output_in_c2: pathlib.Path


def _two_cache_project(tmp_path: pathlib.Path, order: str) -> _TwoCaches:
    """A peer registered in C2 only, and a project that ran on C2 and on C1, in ``order``
    (`c2-then-c1`: its last registration is in C1; `c1-then-c2`: in C2). The project's lock
    names the object of its output in C2."""
    c1, c2 = tmp_path / "cache-one", tmp_path / "cache-two"
    c1.mkdir()
    peer = _shared_project(tmp_path / "peer", c2)
    _ok(_run(peer, "repro"))
    root = _project(tmp_path / "project")
    (root / "data" / "raw" / "input.txt").write_text("only this project reads this\n")
    _use_cache(root, c2)
    _ok(_run(root, "repro"))  # its output's object goes to C2
    output = root / "data" / "shadow" / "shout.txt"
    output_in_c2 = cache.get_cache_path(c2 / "files", cache.hash_file(output)[0])
    assert output_in_c2.exists(), output_in_c2
    _use_cache(root, c1)
    _ok(_run(root, "repro"))  # registers in C1 before its stage skips
    if order == "c1-then-c2":
        _use_cache(root, c2)
        _ok(_run(root, "repro"))
    assert _registered(c1, root) and _registered(c2, root), "the project is not in both registers"
    return _TwoCaches(peer, root, c1, c2, output_in_c2)


@pytest.mark.parametrize("order", ["c2-then-c1", "c1-then-c2"])
def test_a_copy_of_a_project_that_moved_on_through_another_cache_refuses(
    tmp_path: pathlib.Path, order: str
) -> None:
    """N1: a project uses C2 and C1; its disk is cloned; the original then names objects
    through C1 only. That step rotates the project's nonce in its entry in C2 too, so the
    clone's state directory holds an older nonce for C2 than C2's entry, and a collection of C2
    run where the clone is visible refuses, naming the project. A nonce mismatch never
    suggests `pivot forget-cache`: no list a copy edits can make an entry current."""
    setup = _two_cache_project(tmp_path, order)
    clone = tmp_path / "clone-of-the-disk"
    shutil.copytree(setup.root, clone, symlinks=True)
    # The original goes on, through C1.
    _use_cache(setup.root, setup.c1)
    (setup.root / "data" / "raw" / "input.txt").write_text("the original moved on\n")
    _ok(_run(setup.root, "repro"))
    # The clone, at the original's path, is what this host now sees.
    setup.root.rename(tmp_path / "original-elsewhere")
    clone.rename(setup.root)
    dead = _publish(setup.c2 / "files", tmp_path, "named by nothing\n")

    refused = _run(setup.peer, "gc", "--yes", "--grace-hours", "0")

    out = refused.stdout + refused.stderr
    flat = " ".join(out.split())
    assert refused.returncode != 0, out
    assert str(setup.root) in flat and "older or different nonce" in flat, out
    assert "forget-cache" not in flat, out
    assert cache.get_cache_path(setup.c2 / "files", dead).exists(), "a refusal removed an object"
    assert setup.output_in_c2.exists(), out


def test_a_copy_refuses_through_another_cache_the_original_reached_when_it_could_not_reach_this_one(
    tmp_path: pathlib.Path,
) -> None:
    """The cross-cache check at collection time (`gc._other_caches_reason`): a project uses C2
    and C1, and a copy of it is taken. The original's next step, through C1, cannot reach C2 (it
    is out of the original's reach then), so C2's entry keeps the nonce the copy holds, and only
    C1's entry moves on. A collection of C2 where the copy is visible accepts C2's own entry, but
    reads the copy's older nonce for C1 against C1's entry, and refuses; it never suggests
    `pivot forget-cache`."""
    setup = _two_cache_project(tmp_path, "c2-then-c1")
    clone = tmp_path / "clone-of-the-disk"
    shutil.copytree(setup.root, clone, symlinks=True)
    away = tmp_path / "cache-two-out-of-reach"
    setup.c2.rename(away)
    _use_cache(setup.root, setup.c1)
    (setup.root / "data" / "raw" / "input.txt").write_text("the original moved on\n")
    ran = _ok(_run(setup.root, "repro"))
    assert "cannot reach it" in ran, ran
    away.rename(setup.c2)
    setup.root.rename(tmp_path / "original-elsewhere")
    clone.rename(setup.root)
    dead = _publish(setup.c2 / "files", tmp_path, "named by nothing\n")

    refused = _run(setup.peer, "gc", "--yes", "--grace-hours", "0")

    out = refused.stdout + refused.stderr
    flat = " ".join(out.split())
    assert refused.returncode != 0, out
    assert "cache-one" in flat and "different nonce" in flat, out
    assert "forget-cache" not in flat, out
    assert cache.get_cache_path(setup.c2 / "files", dead).exists(), "a refusal removed an object"
    assert setup.output_in_c2.exists(), out


@pytest.mark.parametrize("order", ["c2-then-c1", "c1-then-c2"])
def test_a_project_that_uses_two_caches_is_collected_when_both_entries_are_current(
    tmp_path: pathlib.Path, order: str
) -> None:
    """The control: no copy, the project's last step in C1 rotated C1's nonce in its own state
    directory too. A collection of C2 reads both entries, accepts the project, keeps its output
    and removes what no record names."""
    setup = _two_cache_project(tmp_path, order)
    _use_cache(setup.root, setup.c1)
    (setup.root / "data" / "raw" / "input.txt").write_text("the original moved on\n")
    _ok(_run(setup.root, "repro"))
    dead = _publish(setup.c2 / "files", tmp_path, "named by nothing\n")

    collected = _run(setup.peer, "gc", "--yes", "--grace-hours", "0")

    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert not cache.get_cache_path(setup.c2 / "files", dead).exists(), collected.stdout
    assert setup.output_in_c2.exists(), "gc removed an object the project's run-cache row names"


class _MovedOnCopy(NamedTuple):
    peer: pathlib.Path
    root: pathlib.Path
    c1: pathlib.Path
    c2: pathlib.Path
    named_again: pathlib.Path


def _copy_then_move_on(tmp_path: pathlib.Path, mode: str) -> _MovedOnCopy:
    """N1 as the reviewer's `p19_caches.sh` drives it, with a directory for the disk. P (copy
    checkout, one run of history) runs v1 then v2 in C2, so v1's output X is in C2 and named by
    nothing. Before the disk is copied, ``mode`` decides P's use of C1: none (`plain`), a
    `pivot checkout` in C1 (`c1first`), the same with C1 on the copied disk (`c1local`), or from
    another mount namespace (`c1view`, which needs passwordless sudo). The original then runs v1
    again in C1 (from another mount namespace for `c1view`), so its lock names X again. Last the
    copy is put at the disk's path, and the original moved aside, so this host sees the copy
    where P was."""
    disk = tmp_path / "disk"
    disk.mkdir()
    c1 = disk / "cache-one" if mode == "c1local" else tmp_path / "cache-one"
    c2 = tmp_path / "cache-two"
    c1.mkdir()
    peer = _shared_project(tmp_path / "peer", c2)
    _ok(_run(peer, "repro"))
    root = _project(disk / "project")
    raw = root / "data" / "raw" / "input.txt"
    _use_cache(root, c2, retention=1)
    raw.write_text("v1 only this project reads\n")
    _ok(_run(root, "repro"))
    named_again = _shared_object(c2, tmp_path, "V1 ONLY THIS PROJECT READS\n")
    assert named_again.exists(), named_again
    raw.write_text("v2 only this project reads\n")
    _ok(_run(root, "repro"))
    _use_cache(root, c1, retention=1)
    if mode == "c1view":
        _ok(_in_view(root, "checkout"))
    elif mode != "plain":
        _ok(_run(root, "checkout"))
    copied = tmp_path / "copied-disk"
    shutil.copytree(disk, copied, symlinks=True)
    raw.write_text("v1 only this project reads\n")
    _ok(_in_view(root, "repro") if mode == "c1view" else _run(root, "repro"))
    key = named_again.parent.name + named_again.name
    assert key in (root / ".pivot" / "stages" / "shadow" / "shout.lock").read_text()
    disk.rename(tmp_path / "original-disk")
    copied.rename(disk)
    return _MovedOnCopy(peer, root, c1, c2, named_again)


@pytest.mark.parametrize("mode", ["plain", "c1local", "c1view"])
def test_a_copy_refuses_once_the_original_names_through_a_cache_it_did_not_list(
    tmp_path: pathlib.Path, mode: str
) -> None:
    """N1, the topologies a collection-time check of the copy's own list cannot see: the
    original first uses C1 after the copy (`plain`), C1 was copied with the disk (`c1local`), or
    the original uses C1 from another mount namespace, another view (`c1view`). In each, the
    original's step in C1 rotates its nonce in C2 (`c1view`: registers that view in C2, a cache
    another view listed), so a collection of C2 where the copy is visible refuses, keeps X, and
    never suggests `pivot forget-cache`."""
    setup = _copy_then_move_on(tmp_path, mode)
    dead = _publish(setup.c2 / "files", tmp_path, "named by nothing\n")

    refused = _run(setup.peer, "gc", "--yes", "--grace-hours", "0")

    out = refused.stdout + refused.stderr
    flat = " ".join(out.split())
    assert refused.returncode != 0, out
    assert str(setup.root) in flat and "nonce" in flat, out
    assert "forget-cache" not in flat, out
    assert setup.named_again.exists(), f"gc removed X, which the original's lock names: {out}"
    assert cache.get_cache_path(setup.c2 / "files", dead).exists(), "a refusal removed an object"


def test_forget_cache_in_a_copy_never_makes_a_collection_accept_it(tmp_path: pathlib.Path) -> None:
    """N1 `forget`: C1 is out of the copy's reach. The collection of C2 refuses for the nonce
    the original rotated there, without suggesting `pivot forget-cache`; a copy's user who runs
    it anyway removes C1 from the copy's list, and the next collection refuses all the same:
    no list in the copy changes C2's entry."""
    setup = _copy_then_move_on(tmp_path, "c1first")
    c1_id = naming.read_cache_id(setup.c1)
    setup.c1.rename(tmp_path / "out-of-the-copys-reach")

    first = _run(setup.peer, "gc", "--yes", "--grace-hours", "0")

    first_out = first.stdout + first.stderr
    assert first.returncode != 0, first_out
    assert "forget-cache" not in first_out, first_out
    _ok(_run(setup.root, "forget-cache", c1_id))
    assert c1_id not in {record.cache_id for record in naming.cache_records(setup.root / ".pivot")}

    second = _run(setup.peer, "gc", "--yes", "--grace-hours", "0")

    second_out = second.stdout + second.stderr
    assert second.returncode != 0, second_out
    assert setup.named_again.exists(), f"gc removed X after the copy forgot C1: {second_out}"


@pytest.mark.parametrize("unreachable", ["moved away", "another cache at its path"])
def test_a_naming_step_warns_once_about_a_listed_cache_it_cannot_reach(
    tmp_path: pathlib.Path, unreachable: str
) -> None:
    """A listed cache a naming step cannot reach does not fail the step. The command warns once,
    naming the cache and `pivot forget-cache` (safe only in the project itself), and the view's
    record marks the cache unconfirmed. A collection of the step's own cache still checks every
    listed cache and refuses while that one cannot be reached, naming `pivot forget-cache`.
    `pivot forget-cache` removes it from this view's list and changes nothing else: the next
    command does not warn, the next collection proceeds, and the removed cache keeps its files."""
    setup = _two_cache_project(tmp_path, "c2-then-c1")
    c1_id = naming.read_cache_id(setup.c1)
    c1_files_before = sorted(p.relative_to(setup.c1) for p in setup.c1.rglob("*"))
    away = tmp_path / "unmounted-disk"
    setup.c1.rename(away)
    if unreachable == "another cache at its path":
        (setup.c1 / naming.REGISTER_DIR).mkdir(parents=True)
        (setup.c1 / naming.REGISTER_DIR / naming.CACHE_ID_FILE).write_text("f" * 32 + "\n")
    _use_cache(setup.root, setup.c2)
    (setup.root / "data" / "raw" / "input.txt").write_text("a run with C1 out of reach\n")

    ran = _ok(_run(setup.root, "repro"))

    flat = " ".join(ran.split())
    assert flat.count("cannot reach it") == 1, ran
    assert f"cache {setup.c1} (id {c1_id})" in flat, ran
    assert f"pivot forget-cache {c1_id}` here, in the project itself: never in a copy" in flat, ran
    why = (
        f"another cache is at that path now (id {'f' * 32})"
        if unreachable == "another cache at its path"
        else "it cannot be read"
    )
    assert why in flat, ran
    marked = [r for r in naming.cache_records(setup.root / ".pivot") if r.cache_id == c1_id]
    assert marked and all(record.unconfirmed_since for record in marked), marked
    dead = _publish(setup.c2 / "files", tmp_path, "named by nothing\n")
    refused = _run(setup.peer, "gc", "--yes", "--grace-hours", "0")
    refusal = " ".join((refused.stdout + refused.stderr).split())
    assert refused.returncode != 0, refusal
    where = (
        f"another cache is at that path now (id {'f' * 32})"
        if unreachable == "another cache at its path"
        else "that cache cannot be read"
    )
    assert f"(id {c1_id}), where {where}" in refusal, refusal
    assert f"pivot forget-cache {c1_id}` in the project itself, never in a copy" in refusal, refusal
    assert cache.get_cache_path(setup.c2 / "files", dead).exists(), "a refusal removed an object"

    unknown = _run(setup.root, "forget-cache", str(tmp_path / "never-a-cache"))
    assert unknown.returncode != 0, unknown.stdout + unknown.stderr
    assert "cache-two" in unknown.stdout + unknown.stderr, "the refusal lists no cache"
    forgotten = _ok(_run(setup.root, "forget-cache", str(setup.c1)))
    assert c1_id in forgotten and "Nothing in that cache changed" in forgotten, forgotten
    listed = {record.cache_id for record in naming.cache_records(setup.root / ".pivot")}
    assert listed == {naming.read_cache_id(setup.c2)}, listed
    again = _ok(_run(setup.root, "checkout"))
    assert "cannot reach it" not in again, again
    collected = _run(setup.peer, "gc", "--yes", "--grace-hours", "0")
    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert not cache.get_cache_path(setup.c2 / "files", dead).exists(), collected.stdout
    if unreachable == "moved away":
        away.rename(setup.c1)
        assert sorted(p.relative_to(setup.c1) for p in setup.c1.rglob("*")) == c1_files_before, (
            "forget-cache changed the cache it removed from the list"
        )


def test_a_naming_step_registers_again_in_a_listed_cache_that_lost_the_projects_entry(
    tmp_path: pathlib.Path,
) -> None:
    """A naming step in C2 brings the project's entry in every listed cache up to date: in C1,
    whose entry for the project was deleted, it registers in full again, with a nonce pair that
    matches."""
    setup = _two_cache_project(tmp_path, "c2-then-c1")
    _unregister(setup.c1, setup.root)
    _use_cache(setup.root, setup.c2)

    _ok(_run(setup.root, "checkout"))

    assert _registered(setup.c1, setup.root), "the step in C2 left C1 without the project's entry"
    entry = _entry(setup.c1, setup.root)
    state_nonce, entry_nonce = naming.read_nonce_pair(
        setup.root / ".pivot",
        naming.view_key_of(*naming.this_view()),
        naming.read_cache_id(setup.c1),
        naming.entry_nonce_path(entry),
    )
    assert state_nonce == entry_nonce


def test_forget_cache_refuses_a_cache_this_view_reaches_and_leaves_other_views_records(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot forget-cache` is only for a cache this view cannot reach: while C1 holds the
    project's current entry it refuses and removes nothing. Once C1 is out of reach it removes
    this view's record of C1 alone; a record another view of the project (another host or mount
    namespace; written here by hand) holds for C1 stays, for that view to remove."""
    setup = _two_cache_project(tmp_path, "c2-then-c1")
    c1_id = naming.read_cache_id(setup.c1)
    gc_views = setup.root / ".pivot" / "state.lmdb" / naming.VIEW_NONCE_DIR
    own = sorted(gc_views.glob(f"*-{c1_id}.cache"))
    assert len(own) == 1, own

    refused = _run(setup.root, "forget-cache", c1_id)

    flat = " ".join((refused.stdout + refused.stderr).split())
    assert refused.returncode != 0, flat
    assert "forget-cache is only for a cache this view can no longer reach" in flat, flat
    assert own[0].exists(), "a refusal removed the record"

    other_view = gc_views / f"{'0' * 16}-{c1_id}.cache"
    other_view.write_text(json.dumps({"cache": str(setup.c1)}))
    setup.c1.rename(tmp_path / "out-of-reach")
    forgotten = _ok(_run(setup.root, "forget-cache", c1_id))

    assert "this view of the project" in " ".join(forgotten.split()), forgotten
    assert not own[0].exists(), "this view's record of C1 is still listed"
    assert other_view.exists(), "forget-cache removed another view's record"


def test_a_collection_checks_the_caches_every_view_of_a_project_lists(
    tmp_path: pathlib.Path,
) -> None:
    """A collection checks every cache any view of a project lists, not only the entry's view's:
    a record another view (a container's mount namespace; written here by hand) holds for C1,
    where that view has no entry, refuses the collection of C2, naming the record to delete if
    that view is gone, and never suggests `pivot forget-cache`, which removes only this view's
    records. Without the record, the collection proceeds."""
    setup = _two_cache_project(tmp_path, "c2-then-c1")
    c1_id = naming.read_cache_id(setup.c1)
    record = (
        setup.root / ".pivot" / "state.lmdb" / naming.VIEW_NONCE_DIR / f"{'0' * 16}-{c1_id}.cache"
    )
    record.write_text(json.dumps({"cache": str(setup.c1)}))
    dead = _publish(setup.c2 / "files", tmp_path, "named by nothing\n")

    refused = _run(setup.peer, "gc", "--yes", "--grace-hours", "0")

    flat = " ".join((refused.stdout + refused.stderr).split())
    assert refused.returncode != 0, flat
    assert "Another view of the project" in flat and f"delete {record}" in flat, flat
    assert "forget-cache" not in flat, flat
    assert cache.get_cache_path(setup.c2 / "files", dead).exists(), "a refusal removed an object"
    record.unlink()
    collected = _run(setup.peer, "gc", "--yes", "--grace-hours", "0")
    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert not cache.get_cache_path(setup.c2 / "files", dead).exists(), collected.stdout


def test_a_copy_of_a_cache_whose_objects_are_shared_refuses(tmp_path: pathlib.Path) -> None:
    """N1 with a `files/` link: C1 is on the copied disk, but its object store is a link to
    shared storage, and the project also uses C2. C1's register lives beside the shared store,
    not on the disk (`naming.store_layout`), so the original's step in C1 rotates the nonce of
    the very entry the copy's collection of its C1 reads (and the nonce in C2 too). The copy's
    `pivot gc` refuses and keeps X, which the original's lock names, in the shared store."""
    disk = tmp_path / "disk"
    disk.mkdir()
    shared = tmp_path / "shared-objects"
    shared.mkdir()
    c1, c2 = disk / "cache-one", tmp_path / "cache-two"
    c1.mkdir()
    (c1 / "files").symlink_to(shared)
    root = _project(disk / "project")
    raw = root / "data" / "raw" / "input.txt"
    _use_cache(root, c2, retention=1)
    _ok(_run(root, "repro"))  # C2 is listed
    _use_cache(root, c1, retention=1)
    raw.write_text("v1 only this project reads\n")
    _ok(_run(root, "repro"))
    named_again = _shared_object(c1, tmp_path, "V1 ONLY THIS PROJECT READS\n")
    assert named_again.exists() and named_again.resolve().is_relative_to(shared), named_again
    raw.write_text("v2 only this project reads\n")
    _ok(_run(root, "repro"))
    copied = tmp_path / "copied-disk"
    shutil.copytree(disk, copied, symlinks=True)
    raw.write_text("v1 only this project reads\n")
    _ok(_run(root, "repro"))  # the original names X again, through C1
    disk.rename(tmp_path / "original-disk")
    copied.rename(disk)

    refused = _run(root, "gc", "--yes", "--grace-hours", "0")

    flat = " ".join((refused.stdout + refused.stderr).split())
    assert refused.returncode != 0, flat
    assert "older or different nonce" in flat, flat
    assert "forget-cache" not in flat, flat
    assert named_again.exists(), f"the copy's gc removed X from the shared store: {flat}"


# =============================================================================
# Round 15: the objects a run read by hash are live (#544)
# =============================================================================

_READ_BY_HASH_PIPELINE = """\
import pivot
from pivot import Pipeline, stage


@stage
def plan(pin: str) -> dict:
    return {"rows": [pivot.read_by_hash(h).read_text().strip() for h in pin.split()]}


with Pipeline("p") as pipeline:
    plan(pipeline.input("counted.pin", t=str))
"""


@pytest.mark.parametrize("commit", ["run", "no-commit then commit"])
def test_pivot_gc_keeps_an_object_a_stage_read_by_hash_after_its_pvt_is_gone(
    tmp_path: pathlib.Path, commit: str
) -> None:
    """#544: a stage reads an object by the hash its pin names; the lock, the run-cache row and
    (under `--no-commit`) the `nocommit:` record list it under `reads`. Once the `.pvt` that
    brought it into the cache is gone, those records alone name it, and a collection keeps it."""
    root = _project(tmp_path / "project")
    (root / "pipeline.py").write_text(_READ_BY_HASH_PIPELINE)
    snapshot = root / "store" / "snapshot.txt"
    snapshot.parent.mkdir()
    snapshot.write_text("a row only read by hash\n")
    _ok(_run(root, "track", "store/snapshot.txt"))
    content_hash = cache.hash_file(snapshot)[0]
    read_object = cache.get_cache_path(root / ".pivot" / "cache" / "files", content_hash)
    (root / "data" / "raw" / "counted.pin").write_text(f"{content_hash}\n")
    if commit == "run":
        _ok(_run(root, "repro"))
    else:
        _ok(_run(root, "repro", "--no-commit"))
    snapshot.unlink()
    snapshot.with_name("snapshot.txt.pvt").unlink()
    dead = _publish(root / ".pivot" / "cache" / "files", tmp_path, "named by nothing\n")

    collected = _run(root, "gc", "--yes", "--grace-hours", "0")

    out = collected.stdout + collected.stderr
    assert collected.returncode == 0, out
    assert read_object.exists(), f"gc removed {read_object}, which the stage read by hash"
    assert not cache.get_cache_path(root / ".pivot" / "cache" / "files", dead).exists(), out
    if commit != "run":
        _ok(_run(root, "commit"))
        lock = yaml.safe_load((root / ".pivot" / "stages" / "p" / "plan.lock").read_text())
        assert lock["reads"] == [content_hash], lock


# =============================================================================
# Round 16: an object a running stage read by hash, and a nonce pair half rotated
# =============================================================================

# `use` reads the object its pin names (a cache hit takes no naming unit), then, when asked,
# marks that it read it and waits to be released.
_READ_THEN_HOLD_PIPELINE = """\
import os
import pathlib
import time

import pivot
from pivot import Pipeline, stage


@stage
def use(pin: str) -> str:
    data = pivot.read_by_hash(pin.strip()).read_text()
    gate = os.environ.get("GC_TEST_GATE")
    if gate:
        pathlib.Path(gate, "started").write_text("")
        deadline = time.monotonic() + 120
        while not pathlib.Path(gate, "go").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
    return "used " + data


with Pipeline("rd") as pipeline:
    use(pipeline.input("pin.txt", t=str))
"""


class _S3Client(Protocol):
    """The boto3 S3 client calls these tests make; boto3 ships no stubs."""

    def create_bucket(self, **kwargs: object) -> object: ...

    def put_object(self, **kwargs: object) -> object: ...


def _remote_holding(root: pathlib.Path, endpoint: str, data: bytes) -> dict[str, str]:
    """Point the project at a fresh bucket on the moto server at ``endpoint`` as its one remote,
    holding ``data`` as an object another project pushed; return the env that reaches it."""
    env = _env(
        AWS_ENDPOINT_URL=endpoint,
        AWS_ACCESS_KEY_ID="x",
        AWS_SECRET_ACCESS_KEY="x",
        AWS_DEFAULT_REGION="us-east-1",
    )
    bucket = f"gc-reads-{uuid.uuid4().hex[:8]}"
    s3 = cast(
        "_S3Client",
        boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name="us-east-1",
            aws_access_key_id="x",
            aws_secret_access_key="x",
        ),
    )
    s3.create_bucket(Bucket=bucket)
    key = xxhash.xxh64(data).hexdigest()
    s3.put_object(Bucket=bucket, Key=f"cache/files/{key[:2]}/{key[2:]}", Body=data)
    _ok(_run(root, "config", "set", "remotes.origin", f"s3://{bucket}/cache", env=env))
    return env


@pytest.mark.parametrize("commit", ["run", "no-commit then commit"])
@pytest.mark.parametrize("remote", ["none", "holds the object"])
def test_a_commit_never_names_an_object_a_collection_removed_while_the_stage_ran(
    tmp_path: pathlib.Path, moto_services: dict[str, str], commit: str, remote: str
) -> None:
    """R1: a stage reads an object by hash on a cache hit, which takes no naming unit and names
    nothing until the lock does. A peer's `pivot gc` removes it while the stage still runs. The
    commit (the stage's own, or `pivot commit` of a `--no-commit` run) must not write a lock
    whose `reads` names an object the cache no longer holds: it fetches the object again from
    the remote before its naming unit, and inside the unit refuses, failing the stage and
    committing nothing, when the object is still missing."""
    shared = tmp_path / "shared-cache"
    peer = _shared_project(tmp_path / "peer", shared)
    _ok(_run(peer, "repro"))
    root = _shared_project(tmp_path / "project", shared, "copy")
    (root / "pipeline.py").write_text(_READ_THEN_HOLD_PIPELINE)
    snapshot = b"an old snapshot, read only by hash\n"
    key = _publish(shared / "files", tmp_path, snapshot.decode())
    read_object = cache.get_cache_path(shared / "files", key)
    (root / "data" / "raw" / "pin.txt").write_text(f"{key}\n")
    env = _env() if remote == "none" else _remote_holding(root, moto_services["s3"], snapshot)
    gate = tmp_path / "gate"
    gate.mkdir()
    args = ("repro",) if commit == "run" else ("repro", "--no-commit")
    proc = _start(root, *args, env={**env, "GC_TEST_GATE": str(gate)})
    _wait_for(gate / "started", proc, 120)

    collected = _run(peer, "gc", "--yes", "--grace-hours", "0")
    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert not read_object.exists(), "the peer's gc kept the object only the running stage read"
    (gate / "go").write_text("")
    ran, _ = proc.communicate(timeout=180)
    if commit == "run":
        committed = subprocess.CompletedProcess(args, proc.returncode, ran, "")
    else:
        assert proc.returncode == 0, ran
        committed = subprocess.run(
            [_PIVOT, "commit"], cwd=root, env=env, capture_output=True, text=True, timeout=180
        )

    out = committed.stdout + committed.stderr
    stage_lock = root / ".pivot" / "stages" / "rd" / "use.lock"
    if remote == "none":
        assert committed.returncode != 0, out
        assert "no longer in the cache" in " ".join(out.split()), out
        assert not stage_lock.exists(), f"a lock names {key}, which the cache no longer holds"
    else:
        assert committed.returncode == 0, out
        assert yaml.safe_load(stage_lock.read_text())["reads"] == [key]
        assert read_object.exists(), f"the lock names {key}, which the commit did not fetch again"


def _flock_waiter_on(path: pathlib.Path) -> bool:
    """Whether a process waits for an flock on ``path`` (a `->` line of `/proc/locks`)."""
    inode = path.stat().st_ino
    for line in pathlib.Path("/proc/locks").read_text().splitlines():
        fields = line.split()
        if "->" in fields and "FLOCK" in fields and fields[-3].endswith(f":{inode}"):
            return True
    return False


def test_a_collection_waits_for_a_nonce_rotation_another_cache_has_half_written(
    tmp_path: pathlib.Path,
) -> None:
    """A naming step in C1 rotates the project's nonce pair for C2 without C2's naming lock: the
    state directory's nonce first, then the entry's, both under the state nonce file's exclusive
    flock. A collection of C2 that read the pair between the two writes would find them unequal
    and refuse a live project; it reads under the shared flock, so it waits for the rotation to
    finish and then collects."""
    setup = _two_cache_project(tmp_path, "c2-then-c1")
    entry = _entry(setup.c2, setup.root)
    view_key = naming.view_key_of(*naming.this_view())
    state_nonce = (
        setup.root
        / ".pivot"
        / "state.lmdb"
        / naming.VIEW_NONCE_DIR
        / f"{view_key}-{naming.read_cache_id(setup.c2)}"
    )
    fresh = f"{os.urandom(16).hex()}\n".encode()
    dead = _publish(setup.c2 / "files", tmp_path, "named by nothing\n")
    fd = os.open(state_nonce, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        assert os.pwrite(fd, fresh, 0) == len(fresh)  # the state side written, the entry's not
        proc = _start_gc(setup.peer)
        _wait_until("it waits for the rotation", lambda: _flock_waiter_on(state_nonce), proc)
        naming.entry_nonce_path(entry).write_bytes(fresh)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    out = _finish(proc)

    assert not cache.get_cache_path(setup.c2 / "files", dead).exists(), out
    assert setup.output_in_c2.exists(), out


# =============================================================================
# Round 17: the register follows the object store; a read-only view lists nothing it cannot
# register in; a copy given its own identity
# =============================================================================


def _store_objects(store: pathlib.Path) -> set[str]:
    """The keys of every object in the object store ``store``."""
    return {
        prefix.name + obj.name
        for prefix in store.iterdir()
        if prefix.is_dir() and len(prefix.name) == 2
        for obj in prefix.iterdir()
        if not obj.name.startswith(".")
    }


def test_a_disk_copy_of_a_cache_root_whose_objects_are_shared_refuses(
    tmp_path: pathlib.Path,
) -> None:
    """SF: one cache root on the cloned disk, its `files/` a link to shared storage; P (copy
    checkout, one run of history) uses only it. The disk is copied after v1, and the original
    then runs v2, naming Y. The register and the naming lock follow the object store
    (`naming.store_layout`), into the directory that holds it, which both disks reach: the
    original's v2 step rotates its nonce where the copy's collection reads it, so the copy's
    `pivot gc` of its own cache root refuses instead of removing Y."""
    store = tmp_path / "shared" / "objects"
    store.mkdir(parents=True)
    with _disk_images(tmp_path, store) as work:
        cache_root = work / "cache"
        cache_root.mkdir()
        (cache_root / "files").symlink_to(store)
        root = _project(work / "project")
        _use_cache(root, cache_root, retention=1)
        raw = root / "data" / "raw" / "input.txt"
        raw.write_text("v1 only this project reads\n")
        _ok(_run(root, "repro"))
        node2 = _clone_disk(tmp_path)
        raw.write_text("v2 only this project reads\n")
        _ok(_run(root, "repro"))
        y = _shared_object(cache_root, tmp_path, "V2 ONLY THIS PROJECT READS\n")
        key = y.parent.name + y.name
        assert key in (root / ".pivot" / "stages" / "shadow" / "shout.lock").read_text()
        register = naming.register_root(cache_root)

        collected = _in_view(
            root, "gc", "--yes", "--grace-hours", "0", hostname="pivot-node2", setup=node2
        )

    out = collected.stdout + collected.stderr
    assert collected.returncode != 0, out
    assert "nonce" in " ".join(out.split()), out
    assert cache.get_cache_path(store, key).exists(), f"the copy's gc removed Y: {out}"
    assert register == store.resolve().parent, register


@pytest.mark.parametrize(
    "layout", ["two links", "two links, a tree under C2", "a link to a plain root's store"]
)
def test_cache_roots_that_use_one_object_store_share_its_register(
    tmp_path: pathlib.Path, layout: str
) -> None:
    """SF without a copy: two cache roots use one object store, A's through C1 and B's through
    C2, which links C1's store (or both link a third directory). Each root kept a register of
    its own, so A's collection of C1 read only A and removed B's objects. Both now keep one
    register and one lock beside the store, and A's collection reads B too. It keeps B's
    directory output, whose symlink checkout links a tree, and removes only what no record
    names. A stage's commit puts the tree beside the store (`gc.StoreSnapshot.trees`); a command given
    C2's unresolved `files/` (a `kind: cache` store, a restore) puts it under C2 itself, which
    the test makes by hand: a link there still names the tree's key (`gc._tree_of_store`)."""
    if layout.startswith("two links"):
        store = tmp_path / "shared" / "objects"
        store.mkdir(parents=True)
        c1, c2 = tmp_path / "cache-one", tmp_path / "cache-two"
        for cache_root in (c1, c2):
            cache_root.mkdir()
            (cache_root / "files").symlink_to(store)
    else:
        c1, c2 = tmp_path / "plain-root", tmp_path / "linking-root"
        store = c1 / "files"
        store.mkdir(parents=True)
        c2.mkdir()
        (c2 / "files").symlink_to(store)
    a = _shared_project(tmp_path / "a", c1, "copy")
    _ok(_run(a, "repro"))
    b = _shared_project(tmp_path / "b", c2, "symlink")
    (b / "pipeline.py").write_text(_DIRECTORY_PIPELINE)
    (b / "data" / "raw" / "input.txt").write_text("only b reads this\n")
    _ok(_run(b, "repro"))
    split = b / "data" / "shadow" / "split"
    tree = split.resolve()
    assert tree.is_relative_to(store.resolve().parent / "trees"), tree
    if layout == "two links, a tree under C2":
        moved = c2 / "trees" / tree.parent.name / tree.name
        moved.parent.mkdir(parents=True)
        shutil.copytree(tree, moved, symlinks=True)
        split.unlink()
        split.symlink_to(moved)
    named = _store_objects(store)
    dead = _publish(store, tmp_path, "named by nothing\n")

    collected = _run(a, "gc", "--yes", "--grace-hours", "0")

    out = collected.stdout + collected.stderr
    assert collected.returncode == 0, out
    assert _store_objects(store) - {dead} == named, f"gc of C1 removed B's objects: {out}"
    assert not cache.get_cache_path(store, dead).exists(), out
    assert (split / "a.txt").read_text(), "B's output no longer resolves"
    assert split.resolve().is_dir(), "gc removed B's tree"
    assert _registered(c1, b), "B is not in the register a collection of C1 reads"
    resolved = {naming.register_root(c).resolve() for c in (c1, c2)}
    assert resolved == {store.resolve().parent}, resolved
    locks = {tuple(p.resolve() for p in naming.naming_lock_paths(c)) for c in (c1, c2)}
    assert len(locks) == 1, locks


def test_one_collection_at_a_time_runs_on_one_object_store(tmp_path: pathlib.Path) -> None:
    """`gc.lock` lives beside the register too, so two collections through two cache roots of
    one object store never run together: while one holds the store's `gc.lock` (taken here
    through C1), a collection through C2 refuses and removes nothing."""
    store = tmp_path / "shared" / "objects"
    store.mkdir(parents=True)
    c1, c2 = tmp_path / "cache-one", tmp_path / "cache-two"
    for cache_root in (c1, c2):
        cache_root.mkdir()
        (cache_root / "files").symlink_to(store)
    root = _shared_project(tmp_path / "project", c2, "copy")
    _ok(_run(root, "repro"))
    dead = _publish(store, tmp_path, "named by nothing\n")
    fd = cache.open_cache_lock(naming.register_root(c1) / gc.GC_LOCK)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        refused = _run(root, "gc", "--yes", "--grace-hours", "0")
    finally:
        os.close(fd)

    out = refused.stdout + refused.stderr
    assert refused.returncode != 0, out
    assert "another pivot gc holds" in " ".join(out.split()), out
    assert cache.get_cache_path(store, dead).exists(), "a refusal removed an object"


# Loaded through PYTHONPATH into `pivot gc`: pauses it when it calls `GC_TEST_PAUSE_AT`
# (`execute_plan`: under the exclusive naming lock, after its re-read of the register and the
# records; `naming_exclusive`: after the plan, before the exclusive lock) until the test writes
# `go`.
_PLAN_HOOK = """\
import os
import pathlib
import time

_gate = os.environ.get("GC_TEST_PLAN_GATE")
if _gate:
    from pivot.storage import gc as _gc

    _point = os.environ["GC_TEST_PAUSE_AT"]
    _original = getattr(_gc, _point)

    def _held(*args, **kwargs):
        pathlib.Path(_gate, "paused").write_text("")
        deadline = time.monotonic() + 120
        while not pathlib.Path(_gate, "go").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        return _original(*args, **kwargs)

    setattr(_gc, _point, _held)
"""


@pytest.mark.parametrize("window", ["under the exclusive lock", "before the exclusive lock"])
@pytest.mark.parametrize("change", ["re-pointed link", "replaced store directory"])
def test_a_collection_refuses_when_its_store_moves_while_it_runs(
    tmp_path: pathlib.Path, change: str, window: str
) -> None:
    """RP: root R links store S1 (project P), root R2 links S2 (project Q); P and Q produce the
    same object K, so K is in both stores. P moves on, so K is named by nothing in S1, while Q's
    lock names K in S2. P's `pivot gc` pauses, either under its exclusive lock at `execute_plan`
    or after its plan and before that lock, and then R's `files/` is re-pointed to S2, or S1's
    directory is renamed and S2's put at its path. A collection resolved the layout at each step,
    so it unlinked K through the new link, from S2 (and, in the second window, took S2's lock
    while it removed trees in S1). It now resolves the layout once, before `gc.lock`
    (`gc.store_snapshot`), uses that for every lock, read and unlink, and checks it before
    deleting: it refuses, nothing is removed from either store, and Q's K stays."""
    s1, s2 = tmp_path / "d1" / "store", tmp_path / "d2" / "store"
    r, r2 = tmp_path / "r", tmp_path / "r2"
    for directory in (s1, s2, r, r2):
        directory.mkdir(parents=True)
    (r / "files").symlink_to(s1)
    (r2 / "files").symlink_to(s2)
    p = _shared_project(tmp_path / "P", r, "copy")
    _use_cache(p, r, retention=1)
    q = _shared_project(tmp_path / "Q", r2, "copy")
    for root in (p, q):
        (root / "data" / "raw" / "input.txt").write_text("the same text in both stores\n")
        _ok(_run(root, "repro"))
    key = cache.hash_file(q / "data" / "shadow" / "shout.txt")[0]
    (p / "data" / "raw" / "input.txt").write_text("P moved on\n")
    _ok(_run(p, "repro"))
    assert cache.get_cache_path(s1, key).exists() and cache.get_cache_path(s2, key).exists()
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(_PLAN_HOOK)
    gate = tmp_path / "gate"
    gate.mkdir()
    proc = _start(
        p,
        "gc",
        "--yes",
        "--grace-hours",
        "0",
        env=_env(
            PYTHONPATH=str(hook),
            GC_TEST_PLAN_GATE=str(gate),
            GC_TEST_PAUSE_AT="execute_plan"
            if window == "under the exclusive lock"
            else "naming_exclusive",
        ),
    )
    try:
        _wait_for(gate / "paused", proc, timeout=120)
        if change == "re-pointed link":
            relinked = r / "files.relinked"
            relinked.symlink_to(s2)
            relinked.replace(r / "files")
            q_object = cache.get_cache_path(s2, key)
            p_object = cache.get_cache_path(s1, key)
        else:
            s1.rename(s1.with_name("store-before"))
            s2.rename(s1)
            q_object = cache.get_cache_path(s1, key)
            p_object = cache.get_cache_path(s1.with_name("store-before"), key)
        (gate / "go").write_text("")
        out, _ = proc.communicate(timeout=180)
    except BaseException:
        proc.kill()
        raise

    flat = " ".join(out.split())
    assert proc.returncode != 0, flat
    assert "Refusing to go on collecting" in flat, flat
    assert q_object.exists(), f"the collection removed Q's object from the other store: {flat}"
    assert p_object.exists(), f"the collection removed an object after its store moved: {flat}"


# Loaded through PYTHONPATH into `pivot gc`: pauses it right after its first snapshot check in
# `execute_plan` (the check before a prefix's removals) until the test writes `go`.
_CHECK_HOOK = """\
import os
import pathlib
import time

_gate = os.environ.get("GC_TEST_CHECK_GATE")
if _gate:
    from pivot.storage import gc as _gc

    _original = _gc.StoreSnapshot.check
    _paused = []

    def _held(self):
        _original(self)
        if not _paused:
            _paused.append(True)
            pathlib.Path(_gate, "paused").write_text("")
            deadline = time.monotonic() + 120
            while not pathlib.Path(_gate, "go").exists() and time.monotonic() < deadline:
                time.sleep(0.02)

    _gc.StoreSnapshot.check = _held
"""


@pytest.mark.parametrize(
    "replaced", ["store directory", "prefix directory", "prefix link", "prefix bind mount"]
)
def test_a_directory_replaced_after_the_check_never_receives_a_removal(
    tmp_path: pathlib.Path, replaced: str
) -> None:
    """RP, the windows after the plan: P's `pivot gc` through R (`files/` → S1) passes its
    snapshot check before a prefix's removals and pauses. Then S1's store directory is renamed
    and S2's (whose project Q names K) put at its path; or S1's prefix directory of K is renamed
    aside and S2's moved to its name; or it is renamed aside and replaced by a link to S2's; or
    S2's prefix directory is bind-mounted over it. Removals through paths reached Q's K. Now the
    removals open the prefix again from the held store, without following a link, and require
    the device and inode the plan recorded: after a store swap the prefix is found in the held
    store and P's dead K goes from there; a replaced or covered prefix is refused, naming it,
    and nothing is removed there. Q's K stays in every case."""
    if (
        replaced == "prefix bind mount"
        and subprocess.run(["sudo", "-n", "true"], capture_output=True, check=False).returncode
    ):
        pytest.skip("a bind mount needs passwordless sudo")
    s1, s2 = tmp_path / "d1" / "store", tmp_path / "d2" / "store"
    r, r2 = tmp_path / "r", tmp_path / "r2"
    for directory in (s1, s2, r, r2):
        directory.mkdir(parents=True)
    (r / "files").symlink_to(s1)
    (r2 / "files").symlink_to(s2)
    p = _shared_project(tmp_path / "P", r, "copy")
    _use_cache(p, r, retention=1)
    q = _shared_project(tmp_path / "Q", r2, "copy")
    for root in (p, q):
        (root / "data" / "raw" / "input.txt").write_text("the same text in both stores\n")
        _ok(_run(root, "repro"))
    key = cache.hash_file(q / "data" / "shadow" / "shout.txt")[0]
    (p / "data" / "raw" / "input.txt").write_text("P moved on\n")
    _ok(_run(p, "repro"))
    assert cache.get_cache_path(s1, key).exists() and cache.get_cache_path(s2, key).exists()
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(_CHECK_HOOK)
    gate = tmp_path / "gate"
    gate.mkdir()
    proc = _start(
        p,
        "gc",
        "--yes",
        "--grace-hours",
        "0",
        env=_env(PYTHONPATH=str(hook), GC_TEST_CHECK_GATE=str(gate)),
    )
    prefix = key[:2]
    aside = s1.parent / f"{prefix}-before"
    try:
        _wait_for(gate / "paused", proc, timeout=120)
        if replaced == "store directory":
            s1.rename(s1.with_name("store-before"))
            s2.rename(s1)
            p_object = cache.get_cache_path(s1.with_name("store-before"), key)
            q_object = cache.get_cache_path(s1, key)
        elif replaced == "prefix directory":
            (s1 / prefix).rename(aside)
            (s2 / prefix).rename(s1 / prefix)
            p_object = aside / key[2:]
            q_object = cache.get_cache_path(s1, key)
        elif replaced == "prefix link":
            (s1 / prefix).rename(aside)
            (s1 / prefix).symlink_to(s2 / prefix)
            p_object = aside / key[2:]
            q_object = cache.get_cache_path(s2, key)
        else:
            subprocess.run(
                ["sudo", "-n", "mount", "--bind", str(s2 / prefix), str(s1 / prefix)], check=True
            )
            p_object = cache.get_cache_path(s1, key)  # Q's K shows here while the mount holds
            q_object = cache.get_cache_path(s2, key)
        assert q_object.exists() and p_object.exists()
        (gate / "go").write_text("")
        out, _ = proc.communicate(timeout=180)
    except BaseException:
        proc.kill()
        raise
    finally:
        if replaced == "prefix bind mount":
            subprocess.run(["sudo", "-n", "umount", str(s1 / prefix)], check=False)

    flat = " ".join(out.split())
    assert q_object.exists(), f"the collection removed Q's object through a replaced path: {flat}"
    if replaced == "store directory":
        assert proc.returncode == 0, flat
        assert not p_object.exists(), f"P's dead object was not removed from the held store: {flat}"
        return
    assert proc.returncode != 0, flat
    listed = s1.resolve() / prefix
    assert f"the prefix directory {listed} is no longer the directory the plan listed" in flat
    assert "so nothing was removed there" in flat, flat
    if replaced == "prefix bind mount":
        p_object = cache.get_cache_path(s1, key)  # the mount is gone: P's own K shows again
    assert p_object.exists(), f"a refusal removed P's object: {flat}"


def test_a_prefix_removed_and_made_again_after_the_plan_follows_no_link(
    tmp_path: pathlib.Path,
) -> None:
    """After the plan, S1's prefix directory of K is removed entirely, with P's K in it, and a
    new directory is made at its path, by a run of `mkdir`s that may hand it the removed
    directory's inode (no descriptor holds the old one); the new directory holds a plain copy of
    Q's object K. When the inode was reused, the removals open a directory with the device and
    inode the plan recorded and re-check each condemned name there: the copy, made after the
    collection's cutoff, is kept (the grace re-check), and the collection exits 0, having
    reached nothing outside that directory. When the inode was not reused, the collection
    refuses, naming the prefix. Q's K in S2 stays either way."""
    s1, s2 = tmp_path / "d1" / "store", tmp_path / "d2" / "store"
    r, r2 = tmp_path / "r", tmp_path / "r2"
    for directory in (s1, s2, r, r2):
        directory.mkdir(parents=True)
    (r / "files").symlink_to(s1)
    (r2 / "files").symlink_to(s2)
    p = _shared_project(tmp_path / "P", r, "copy")
    _use_cache(p, r, retention=1)
    q = _shared_project(tmp_path / "Q", r2, "copy")
    for root in (p, q):
        (root / "data" / "raw" / "input.txt").write_text("the same text in both stores\n")
        _ok(_run(root, "repro"))
    key = cache.hash_file(q / "data" / "shadow" / "shout.txt")[0]
    (p / "data" / "raw" / "input.txt").write_text("P moved on\n")
    _ok(_run(p, "repro"))
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(_CHECK_HOOK)
    gate = tmp_path / "gate"
    gate.mkdir()
    proc = _start(
        p,
        "gc",
        "--yes",
        "--grace-hours",
        "0",
        env=_env(PYTHONPATH=str(hook), GC_TEST_CHECK_GATE=str(gate)),
    )
    prefix = s1 / key[:2]
    q_object = cache.get_cache_path(s2, key)
    try:
        _wait_for(gate / "paused", proc, timeout=120)
        old_inode = prefix.stat().st_ino
        cache.clear_path(prefix)
        reused = False
        for attempt in range(4000):
            candidate = s1 / f"made-{attempt}"
            candidate.mkdir()
            if candidate.stat().st_ino == old_inode:
                candidate.rename(prefix)
                reused = True
                break
        if not reused:
            prefix.mkdir()
        copy = prefix / key[2:]
        shutil.copyfile(q_object, copy)
        (gate / "go").write_text("")
        out, _ = proc.communicate(timeout=180)
    except BaseException:
        proc.kill()
        raise

    flat = " ".join(out.split())
    assert q_object.exists(), f"the collection reached Q's object in S2: {flat}"
    if reused:
        assert proc.returncode == 0, flat
        assert copy.exists(), f"the copy made after the collection's cutoff was removed: {flat}"
    else:
        assert proc.returncode != 0, flat
        assert "so nothing was removed there" in flat, flat
        assert copy.exists(), f"a refusal removed the new directory's copy: {flat}"


def test_a_tree_store_that_appears_during_a_collection_is_never_read(
    tmp_path: pathlib.Path,
) -> None:
    """RP (c): R's `files/` links store S1 (project P, copy checkout), so its tree store
    (beside S1) does not exist when P's `pivot gc` starts. The collection pauses before its plan,
    and the tree store appears as a link to another cache's `trees/`, where project Q's
    directory output (symlink checkout) lives. A store absent at the start stays absent for the
    whole collection: it lists nothing there, and Q's tree stays."""
    s1 = tmp_path / "d1" / "store"
    r, qc = tmp_path / "r", tmp_path / "q-cache"
    for directory in (s1, r):
        directory.mkdir(parents=True)
    (r / "files").symlink_to(s1)
    p = _shared_project(tmp_path / "P", r, "copy")
    _ok(_run(p, "repro"))
    q = _shared_project(tmp_path / "Q", qc, "symlink")
    (q / "pipeline.py").write_text(_DIRECTORY_PIPELINE)
    (q / "data" / "raw" / "input.txt").write_text("only Q has this tree\n")
    _ok(_run(q, "repro"))
    q_tree = (q / "data" / "shadow" / "split").resolve()
    assert q_tree.is_relative_to(qc.resolve() / "trees"), q_tree
    assert not (s1.parent / "trees").exists() and not (r / "trees").exists()
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(_PLAN_HOOK)
    gate = tmp_path / "gate"
    gate.mkdir()
    proc = _start(
        p,
        "gc",
        "--yes",
        "--grace-hours",
        "0",
        env=_env(PYTHONPATH=str(hook), GC_TEST_PLAN_GATE=str(gate), GC_TEST_PAUSE_AT="build_plan"),
    )
    try:
        _wait_for(gate / "paused", proc, timeout=120)
        (s1.parent / "trees").symlink_to(qc / "trees")
        (gate / "go").write_text("")
        out, _ = proc.communicate(timeout=180)
    except BaseException:
        proc.kill()
        raise

    flat = " ".join(out.split())
    assert proc.returncode == 0, flat
    assert q_tree.is_dir(), f"the collection removed Q's tree through a store that appeared: {flat}"
    assert (q / "data" / "shadow" / "split" / "a.txt").read_text(), "Q's output no longer resolves"


def _read_only_view(cache_dir: pathlib.Path) -> str:
    """A mount-namespace setup in which ``cache_dir`` is read-only (`_in_view`)."""
    quoted = shlex.quote(str(cache_dir))
    return f"mount --bind {quoted} {quoted}\nmount -o remount,ro,bind {quoted}"


def _files_under(target: pathlib.Path) -> list[pathlib.Path]:
    """``target`` itself when it is a file, else every file under it (following a link to a
    tree, as a reader would)."""
    if not target.is_dir():
        return [target] if target.exists() or target.is_symlink() else []
    return sorted(path for path in target.rglob("*") if not path.is_dir())


@pytest.mark.parametrize("kind", ["file", "directory"])
@pytest.mark.parametrize("checkout", ["symlink", "hardlink", "copy"])
@pytest.mark.parametrize("view", ["read-only", "writable"])
def test_a_view_that_sees_the_cache_read_only_never_links_from_it(
    tmp_path: pathlib.Path, view: str, checkout: str, kind: str
) -> None:
    """B3: P tracks X (a file, or a directory) in the shared cache H, then tracks other bytes
    there, so nothing of P's names X. Q (never run on the host) holds P's old `.pvt` and runs
    `pivot checkout`. Where H is read-only Q cannot register in H, so a link would name X where
    the host's `pivot gc` cannot see it: in symlink and hardlink checkout the command refuses,
    naming `cache.checkout_mode: [copy]` and a read-write mount, and links nothing; in copy
    checkout it restores copies, Q's own data, which still read after P's gc. The control: where
    H is writable Q registers, links (or copies) in any mode, and after P's gc X is still in H
    and Q's files read."""
    shared = tmp_path / "H"
    p = _shared_project(tmp_path / "P", shared, "symlink")
    q = _shared_project(tmp_path / "Q", shared, checkout)
    if kind == "file":
        name = "data/notes.bin"
        (p / name).write_text("notes v1\n")
    else:
        name = "images"
        _fill_images(p / name, "v1")
    _ok(_run(p, "track", name))
    record = p / f"{name}.pvt"
    listing_file = listing.tracked_listing_path(record)
    (q / f"{name}.pvt").parent.mkdir(parents=True, exist_ok=True)
    (q / f"{name}.pvt").write_text(record.read_text())
    if listing_file.exists():
        listing.tracked_listing_path(q / f"{name}.pvt").write_bytes(listing_file.read_bytes())
    v1 = {path.relative_to(p): path.read_bytes() for path in _files_under(p / name)}
    keys = {cache.hash_file(p / rel)[0] for rel in v1}
    if kind == "file":
        cache.clear_path(p / name)
        (p / name).write_text("notes v2\n")
    else:
        _fill_images(p / name, "v2")
    _ok(_run(p, "track", "--force", name))
    assert all(cache.get_cache_path(shared / "files", key).exists() for key in keys)

    restored = (
        _in_view(q, "checkout", setup=_read_only_view(shared))
        if view == "read-only"
        else _run(q, "checkout")
    )

    flat = " ".join((restored.stdout + restored.stderr).split())
    target = q / name
    linked = view == "read-only" and checkout != "copy"
    if linked:
        assert restored.returncode != 0, flat
        assert "Refusing to link" in flat and "`cache.checkout_mode: [copy]`" in flat, flat
        assert "mount the cache read-write" in flat, flat
        assert not target.is_symlink(), f"Q linked X from a cache it is not registered in: {flat}"
        for path in _files_under(target):
            assert not path.is_symlink() and path.stat().st_nlink == 1, f"{path} is a link: {flat}"
    else:
        assert restored.returncode == 0, flat
        assert {path.relative_to(q): path.read_bytes() for path in _files_under(target)} == v1
    collected = _run(p, "gc", "--yes", "--grace-hours", "0")
    assert collected.returncode == 0, collected.stdout + collected.stderr
    if view == "writable":
        assert all(cache.get_cache_path(shared / "files", key).exists() for key in keys), (
            f"P's gc removed X, which the registered Q names: {collected.stdout}"
        )
    if not linked:
        assert {path.relative_to(q): path.read_bytes() for path in _files_under(target)} == v1, (
            "Q's files no longer read after P's gc"
        )


def test_a_directory_commit_from_a_read_only_view_never_links_the_tree(
    tmp_path: pathlib.Path,
) -> None:
    """B3': P (symlink checkout, one run of history) commits the directory output of v1, tree
    T1, then v2, so nothing of P's names T1 once a collection prunes the v1 row. Q, a project
    with the same pipeline at v1 that never ran, runs `pivot repro` where the cache H is
    read-only: its commit found T1 in the cache and linked its output to it with Q unregistered,
    and P's `pivot gc` then removed T1 under Q's link. The commit now refuses before it removes
    the output, which stays as the stage wrote it, and Q's files still read after P's gc."""
    shared = tmp_path / "H"
    p = _shared_project(tmp_path / "P", shared, "symlink")
    q = _shared_project(tmp_path / "Q", shared, "symlink")
    for root in (p, q):
        (root / ".pivot" / "config.yaml").write_text(
            f"cache:\n  dir: {shared}\n  checkout_mode:\n  - symlink\n"
            + "core:\n  run_history_retention: 1\n"
        )
        (root / "pipeline.py").write_text(_DIRECTORY_PIPELINE)
        (root / "data" / "raw" / "input.txt").write_text("v1 both projects\n")
    _ok(_run(p, "repro"))
    t1 = (p / "data" / "shadow" / "split").resolve()
    assert t1.is_relative_to(shared.resolve() / "trees"), t1
    (p / "data" / "raw" / "input.txt").write_text("v2 only P\n")
    _ok(_run(p, "repro"))

    ran = _in_view(q, "repro", setup=_read_only_view(shared))

    flat = " ".join((ran.stdout + ran.stderr).split())
    split = q / "data" / "shadow" / "split"
    assert ran.returncode != 0, flat
    assert "Refusing to link" in flat and "mount the cache read-write" in flat, flat
    assert not split.is_symlink(), f"Q's output was linked to T1 while Q is unregistered: {flat}"
    assert (split / "a.txt").read_text() == "v1 both projects\n", flat
    collected = _run(p, "gc", "--yes", "--grace-hours", "0")
    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert (split / "a.txt").read_text() == "v1 both projects\n", "Q's output no longer reads"


def test_a_directory_track_from_a_read_only_view_links_nothing(tmp_path: pathlib.Path) -> None:
    """B3' audit, `pivot track`: track never links the workspace to the cache, in any checkout
    mode (`cli/track.py` publishes the objects and writes the listing and `.pvt`; the user's
    directory stays). P tracks `images` (symlink checkout), then tracks other bytes there, so
    nothing of P's names the first objects. Q holds the first bytes in its own `images` and
    tracks them where the cache H is read-only: the objects are already there, so track
    succeeds, unregistered, and leaves Q's directory a plain directory. P's gc then removes the
    first objects, and Q's files still read: they were never a link."""
    shared = tmp_path / "H"
    p = _shared_project(tmp_path / "P", shared, "symlink")
    q = _shared_project(tmp_path / "Q", shared, "symlink")
    _fill_images(p / "images", "first")
    _ok(_run(p, "track", "images"))
    first = track.read_pvt_file(p / "images.pvt")
    assert first is not None
    _fill_images(p / "images", "second")
    _ok(_run(p, "track", "--force", "images"))
    _fill_images(q / "images", "first")

    tracked = _in_view(q, "track", "images", setup=_read_only_view(shared))

    flat = " ".join((tracked.stdout + tracked.stderr).split())
    images = q / "images"
    assert tracked.returncode == 0, flat
    assert not images.is_symlink(), f"track linked Q's directory into the cache: {flat}"
    q_record = track.read_pvt_file(q / "images.pvt")
    assert q_record is not None and q_record["hash"] == first["hash"], flat
    collected = _run(p, "gc", "--yes", "--grace-hours", "0")
    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert sorted(path.name for path in images.iterdir()) == ["a-first.bin", "b-first.bin"]
    assert (images / "a-first.bin").read_bytes().startswith(b"a image bytes for first")


def test_a_collection_of_many_prefixes_fits_a_small_descriptor_limit(
    tmp_path: pathlib.Path,
) -> None:
    """A collection opens one prefix directory at a time, so it needs the same few descriptors
    however many prefixes hold dead objects. With `ulimit -n 64` (below macOS's default of 256)
    and dead objects in well over 64 prefixes, `pivot gc` removes every one of them; holding a
    descriptor per condemned prefix failed with "Too many open files"."""
    shared = tmp_path / "shared-cache"
    root = _shared_project(tmp_path / "project", shared, "copy")
    _ok(_run(root, "repro"))
    dead = [_publish(shared / "files", tmp_path, f"dead object {i}\n") for i in range(300)]
    prefixes = {key[:2] for key in dead}
    assert len(prefixes) > 100, len(prefixes)

    collected = subprocess.run(
        [
            "bash",
            "-c",
            'ulimit -n 64 && exec "$0" "$@"',
            _PIVOT,
            "gc",
            "--yes",
            "--grace-hours",
            "0",
        ],
        cwd=root,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )

    out = collected.stdout + collected.stderr
    assert collected.returncode == 0, out
    assert "Too many open files" not in out, out
    left = [key for key in dead if cache.get_cache_path(shared / "files", key).exists()]
    assert not left, f"{len(left)} dead objects stayed: {out}"


def test_a_collection_lists_the_trees_under_its_own_cache_root(tmp_path: pathlib.Path) -> None:
    """A linked layout has two tree stores: beside the object store, where a stage's commit puts
    a tree, and under the cache root itself, where a command given the unresolved `files/` puts
    one. A collection lists both (`gc.StoreSnapshot.trees`): a dead tree under the collecting
    root's own `trees/` is removed, and the live one stays."""
    store = tmp_path / "shared" / "objects"
    store.mkdir(parents=True)
    c1 = tmp_path / "cache-one"
    c1.mkdir()
    (c1 / "files").symlink_to(store)
    b = _project(tmp_path / "b")
    (b / ".pivot" / "config.yaml").write_text(
        f"cache:\n  dir: {c1}\n  checkout_mode:\n  - symlink\ncore:\n  run_history_retention: 1\n"
    )
    (b / "pipeline.py").write_text(_DIRECTORY_PIPELINE)
    raw = b / "data" / "raw" / "input.txt"
    raw.write_text("first\n")
    _ok(_run(b, "repro"))
    split = b / "data" / "shadow" / "split"
    first = split.resolve()
    assert first.is_relative_to(store.parent.resolve() / "trees"), first
    raw.write_text("second\n")
    _ok(_run(b, "repro"))
    second = split.resolve()
    assert second != first, second
    moved = c1 / "trees" / first.parent.name / first.name
    moved.parent.mkdir(parents=True)
    shutil.copytree(first, moved, symlinks=True)
    cache.clear_path(first)

    collected = _run(b, "gc", "--yes", "--grace-hours", "0")

    out = collected.stdout + collected.stderr
    assert collected.returncode == 0, out
    assert not moved.exists(), f"gc did not list the trees under its own cache root: {out}"
    assert second.is_dir(), f"gc removed the live tree: {out}"


def test_a_cache_whose_files_link_to_a_mount_of_its_own_refuses_writers_and_collection(
    tmp_path: pathlib.Path,
) -> None:
    """A `files/` link to the root of a mount of its own: the directory that holds the store is
    on another filesystem, which other hosts or mount namespaces need not reach, so the register
    cannot follow the store. A writer refuses before it names anything (and registers nowhere),
    and a collection refuses, both naming the layout and the supported ones, rather than write
    or collect on a register that may not show every project of the store."""
    mount_point = tmp_path / "objects-mount"
    with _tmpfs_mount(mount_point):
        cache_root = tmp_path / "cache"
        cache_root.mkdir()
        (cache_root / "files").symlink_to(mount_point)
        root = _shared_project(tmp_path / "project", cache_root, "copy")

        written = _run(root, "repro")

        wrote = " ".join((written.stdout + written.stderr).split())
        assert written.returncode != 0, wrote
        assert (
            f"Refusing to name objects in {cache_root}: its files/ links to {mount_point}, "
            + "which is a mount of its own"
        ) in wrote, wrote
        assert "mount the filesystem at the cache root" in wrote, wrote
        assert not (root / ".pivot" / "stages").exists() or not list(
            (root / ".pivot" / "stages").rglob("*.lock")
        ), "a writer recorded a lock through the unshared store"
        assert not _registered(cache_root, root)
        dead = _publish(mount_point, tmp_path, "named by nothing\n")

        refused = _run(root, "gc", "--yes", "--grace-hours", "0")

        flat = " ".join((refused.stdout + refused.stderr).split())
        assert refused.returncode != 0, flat
        assert f"its files/ links to {mount_point}, which is a mount of its own" in flat, flat
        assert cache.get_cache_path(mount_point, dead).exists(), "a refusal removed an object"


@pytest.mark.parametrize("mount", ["another device", "bind mount"])
def test_a_cache_whose_files_is_a_mount_point_refuses_writers_and_collection(
    tmp_path: pathlib.Path, mount: str
) -> None:
    """`files/` itself a mount point (`another device`: a tmpfs; `bind mount`: a directory on
    the same device, which only the mount table shows, in a mount namespace): the register stays
    at the cache root, on a filesystem other hosts that mount the store need not see, so it
    cannot show every project of the store. A writer refuses before it names anything, and a
    collection refuses, both naming the layout and the supported ones; nothing is removed."""
    cache_root = tmp_path / "cache"
    files = cache_root / "files"
    if mount == "another device":
        files.mkdir(parents=True)
        with _tmpfs_mount(files):
            root = _shared_project(tmp_path / "project", cache_root, "copy")
            written = _run(root, "repro")
            dead = _publish(files, tmp_path, "named by nothing\n")
            refused = _run(root, "gc", "--yes", "--grace-hours", "0")
            kept = cache.get_cache_path(files, dead).exists()
    else:
        root = _shared_project(tmp_path / "project", cache_root, "copy")
        _ok(_run(root, "repro"))
        store = tmp_path / "store-on-the-same-disk"
        shutil.copytree(files, store)
        dead = _publish(store, tmp_path, "named by nothing\n")
        (root / "data" / "raw" / "input.txt").write_text("a run in the view\n")
        bind = f"mount --bind {shlex.quote(str(store))} {shlex.quote(str(files))}"
        written = _in_view(root, "repro", setup=bind)
        refused = _in_view(root, "gc", "--yes", "--grace-hours", "0", setup=bind)
        kept = cache.get_cache_path(store, dead).exists()

    wrote = " ".join((written.stdout + written.stderr).split())
    assert written.returncode != 0, wrote
    assert (
        f"Refusing to name objects in {cache_root}: its files/ ({files.resolve()}) is a mount "
        + "of its own"
    ) in wrote, wrote
    flat = " ".join((refused.stdout + refused.stderr).split())
    assert refused.returncode != 0, flat
    assert f"its files/ ({files.resolve()}) is a mount of its own" in flat, flat
    assert "mount the filesystem at the cache root" in flat, flat
    assert kept, f"a refusal removed an object: {flat}"


@pytest.mark.parametrize("bound", ["files/", "the whole cache root"])
def test_a_view_that_binds_another_roots_files_cannot_write_into_it(
    tmp_path: pathlib.Path, bound: str
) -> None:
    """S8: a plain cache root H (project P, on the host) whose `files/` another mount namespace
    binds at its own cache root R's `files/` (project Q there, as a container given only the
    objects). Q's steps would register at R, where P's collection of H never looks, and that
    collection removed Q's object. Now Q's writer refuses through the bound `files/`, so no view
    writes into a store whose register is elsewhere. The control binds the whole cache root H at
    R, a supported layout: Q's run works, registers in H's register, and P's collection keeps
    Q's object."""
    h, r = tmp_path / "H", tmp_path / "R"
    p = _shared_project(tmp_path / "P", h, "copy")
    _ok(_run(p, "repro"))
    q = _shared_project(tmp_path / "Q", r, "copy")
    (q / "data" / "raw" / "input.txt").write_text("only Q, in its view\n")
    if bound == "files/":
        (r / "files").mkdir(parents=True)
        bind = f"mount --bind {shlex.quote(str(h / 'files'))} {shlex.quote(str(r / 'files'))}"
    else:
        r.mkdir()
        bind = f"mount --bind {shlex.quote(str(h))} {shlex.quote(str(r))}"

    ran = _in_view(q, "repro", setup=bind)

    out = " ".join((ran.stdout + ran.stderr).split())
    q_output = q / "data" / "shadow" / "shout.txt"
    if bound == "files/":
        assert ran.returncode != 0, out
        assert f"Refusing to name objects in {r}: its files/ ({r / 'files'}) is a mount" in out
        assert "mount the filesystem at the cache root" in out, out
        assert not (q / ".pivot" / "stages" / "shadow" / "shout.lock").exists(), out
        return
    assert ran.returncode == 0, out
    q_object = cache.get_cache_path(h / "files", cache.hash_file(q_output)[0])
    assert q_object.exists(), "Q's object is not in H's store"
    assert _registered(h, q), "Q's view did not register in H's register"
    collected = _run(p, "gc", "--yes", "--grace-hours", "0")
    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert q_object.exists(), f"P's gc of H removed Q's object: {collected.stdout}"


def test_a_view_that_cannot_write_a_listed_cache_leaves_it_unlisted(tmp_path: pathlib.Path) -> None:
    """A mount namespace in which C2 is read-only takes a naming step in C1. Confirming C2 there
    cannot write C2's register, so that view never holds an entry in C2; it must not list C2
    either, or every collection of C1 would refuse for an entry no command of that view can
    write. The host's collection of C1 proceeds; the host's own record of C2 stays."""
    setup = _two_cache_project(tmp_path, "c2-then-c1")
    c2_id = naming.read_cache_id(setup.c2)
    state_dir = setup.root / ".pivot"
    host_view = naming.view_key_of(*naming.this_view())
    read_only = (
        f"mount --bind {shlex.quote(str(setup.c2))} {shlex.quote(str(setup.c2))}\n"
        + f"mount -o remount,ro,bind {shlex.quote(str(setup.c2))}"
    )

    _ok(_in_view(setup.root, "checkout", setup=read_only))

    dead = _publish(setup.c1 / "files", tmp_path, "named by nothing\n")
    collected = _run(setup.root, "gc", "--yes", "--grace-hours", "0")
    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert not cache.get_cache_path(setup.c1 / "files", dead).exists(), collected.stdout
    c2_records = [r for r in naming.cache_records(state_dir) if r.cache_id == c2_id]
    assert [r.view_key for r in c2_records] == [host_view], c2_records
    assert len(naming.cache_records(state_dir)) == 3, "the read-only view did not list C1"


def test_a_copy_given_its_own_project_id_never_collects_the_originals_objects(
    tmp_path: pathlib.Path,
) -> None:
    """CW's documented practice: a copy of a project (here a copy of its directory, at the
    original's path on this host) whose project id file is deleted before its first pivot
    command records a new id. When the copy then reruns and collects the shared cache, the
    original's entry no longer names the project at that path ("a different project id"), so
    the collection refuses instead of reading the copy as the original, and the original's
    object, which only its own records name, stays."""
    shared = tmp_path / "shared-cache"
    peer = _shared_project(tmp_path / "peer", shared)
    _ok(_run(peer, "repro"))
    root = _shared_project(tmp_path / "project", shared, "copy")
    _use_cache(root, shared, retention=1)
    (root / "data" / "raw" / "input.txt").write_text("the original's input\n")
    _ok(_run(root, "repro"))
    original_object = _shared_object(shared, tmp_path, "THE ORIGINAL'S INPUT\n")
    assert original_object.exists()
    copy = tmp_path / "copy-of-the-disk"
    shutil.copytree(root, copy, symlinks=True)
    root.rename(tmp_path / "original-elsewhere")
    copy.rename(root)
    (root / ".pivot" / "state.lmdb" / naming.PROJECT_ID_FILE).unlink()
    # The copy writes first: its run, with one run of history, leaves the original's output
    # named by no record of the copy's.
    (root / "data" / "raw" / "input.txt").write_text("the copy's own input\n")
    _ok(_run(root, "repro"))

    refused = _run(root, "gc", "--yes", "--grace-hours", "0")

    flat = " ".join((refused.stdout + refused.stderr).split())
    assert refused.returncode != 0, flat
    assert "holds a different project id" in flat, flat
    assert original_object.exists(), "the copy's gc removed the original's object"


# `cut` runs once per file of `plan`'s directory output (`pipeline.map`, #545): its instances are
# `mp/cut@<key>`, each with its own lock.
_MAPPED_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, loaders, stage


@stage
def plan(
    plan: dict[str, int],
) -> Annotated[dict[str, dict[str, int]], directory, loaders.YAML[dict[str, int]]()]:
    return {f"{name}.yaml": {"value": value} for name, value in plan.items()}


@stage
def cut(entry: dict[str, int]) -> dict[str, int]:
    return {"cut": entry["value"] * 7}


with Pipeline("mp") as pipeline:
    cut(pipeline.map(plan(pipeline.input("plan.yaml", t=dict[str, int]))))
"""


def test_the_locks_of_a_mapped_stages_instances_keep_their_objects_once_unexpanded(
    tmp_path: pathlib.Path,
) -> None:
    """A mapped stage's instances (`mp/cut@a`, `mp/cut@b`) each record their output in a lock of
    their own. Once `b` leaves the plan, the next run no longer expands `mp/cut@b`, but its lock
    stays, and a collection reads every lock on disk, expanded or not: a peer's `pivot gc`
    (with one run of history, so no run-cache row keeps them) keeps both instances' objects and
    removes only what no record names."""
    shared = tmp_path / "shared-cache"
    peer = _shared_project(tmp_path / "peer", shared)
    _ok(_run(peer, "repro"))
    root = _shared_project(tmp_path / "project", shared, "copy")
    _use_cache(root, shared, retention=1)
    (root / "pipeline.py").write_text(_MAPPED_PIPELINE)
    plan = root / "data" / "raw" / "plan.yaml"
    plan.write_text("a: 1\nb: 2\n")
    _ok(_run(root, "repro"))
    stages = root / ".pivot" / "stages" / "mp"
    locks = {path.name: path for path in stages.glob("cut@*.lock")}
    assert set(locks) == {"cut@a.lock", "cut@b.lock"}, sorted(p.name for p in stages.iterdir())
    objects = {
        name: cache.get_cache_path(
            shared / "files", yaml.safe_load(path.read_text())["outs"][0]["hash"]
        )
        for name, path in locks.items()
    }
    assert all(path.exists() for path in objects.values()), objects
    b_lock = locks["cut@b.lock"].read_text()

    plan.write_text("a: 1\n")
    second = _ok(_run(root, "repro"))
    assert "cut@b" not in second, f"the run still expanded mp/cut@b: {second}"
    assert locks["cut@b.lock"].read_text() == b_lock, "the unexpanded instance's lock changed"
    dead = _publish(shared / "files", tmp_path, "named by nothing\n")

    collected = _run(peer, "gc", "--yes", "--grace-hours", "0")

    out = collected.stdout + collected.stderr
    assert collected.returncode == 0, out
    assert not cache.get_cache_path(shared / "files", dead).exists(), out
    assert objects["cut@a.lock"].exists(), f"gc removed mp/cut@a's output: {out}"
    assert objects["cut@b.lock"].exists(), f"gc removed the unexpanded mp/cut@b's output: {out}"
