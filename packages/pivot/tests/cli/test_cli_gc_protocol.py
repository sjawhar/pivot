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
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

import pytest

from pivot import run_history, types
from pivot.storage import cache, gc, lock, state, track

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
    path.mkdir()
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


def _register_fresh(root: pathlib.Path) -> pathlib.Path:
    """Write this project's register entry now, so the command entry takes no lock, and return
    the cache. The tests then hold the exclusive lock against the writer sites alone."""
    cache_dir = root / ".pivot" / "cache"
    gc.register_project(cache_dir, root, root / ".pivot")
    assert gc.register_entry_is_fresh(cache_dir, root)
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
    with gc.naming_exclusive(cache_dir):
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
    test reaches the commit site alone."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    cache_dir = _register_fresh(root)
    (root / "data" / "raw" / "input.txt").write_text("changed\n")
    output_object = _object_for(root, "CHANGED\n")
    assert not output_object.exists()
    before = _locks(root)
    gate = tmp_path / "gate"
    gate.mkdir()

    with gc.naming_exclusive(cache_dir):
        proc = _start(root, "repro", "--force", env=_env(GC_PROTOCOL_GATE=str(gate)))
        try:
            _wait_for(gate / "started", proc, timeout=120)
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


def test_a_run_cache_skip_waits_while_a_collection_holds_the_naming_lock(
    tmp_path: pathlib.Path,
) -> None:
    """The run-cache skip site: a stage restored from the run cache re-links objects that are
    already cached and names them in its lock. It does so only once the collection releases the
    naming lock, since the objects it re-uses may be the ones being removed."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    (root / "data" / "raw" / "input.txt").write_text("other\n")
    started = time.monotonic()
    _ok(_run(root, "repro"))
    unblocked = time.monotonic() - started
    (root / "data" / "raw" / "input.txt").write_text("hello\n")  # back to a cached run
    cache_dir = _register_fresh(root)
    before = _locks(root)

    with gc.naming_exclusive(cache_dir):
        proc = _start(root, "repro", env=_env())
        try:
            # An unblocked run of this size took `unblocked` seconds; give it three times that.
            deadline = time.monotonic() + max(3 * unblocked, 5.0)
            while proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
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
    """The turnstile and naming lock in the cache root, where `pivot gc` takes them."""
    return cache_dir / gc.NAMING_TURNSTILE, cache_dir / gc.NAMING_LOCK


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
    gc.register_project(cache_dir, project, project / ".pivot")
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

    with _fifo_pin(tmp_path / "peer", cache_dir) as pin:
        pin.release()  # the plan's read of the peer
        collector = _start_gc(root)
        _wait_until("its exclusive lock", lambda: _gc_holds_the_naming_lock(cache_dir), collector)
        # gc holds the lock and waits on the peer's record, before it deletes anything.
        writer = _start(root, "repro", "--force", env=_env(GC_PROTOCOL_GATE=str(gate)))
        try:
            _wait_for(gate / "started", writer, timeout=120)
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
        with gc.naming_unit(cache_dir, project=None):  # a writer mid-commit
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
        with gc.naming_unit(cache_dir, project=None):  # let gc plan, then hold it at the turnstile
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

    with gc.naming_unit(cache_dir, project=None):  # a command registering its project
        collector = _start_gc(root)
        _wait_until("the turnstile", lambda: _gc_waits_for_the_naming_lock(cache_dir), collector)
        _name_in_lock(late / ".pivot" / "stages", "other/stage", victim)
        gc.register_project(cache_dir, late, late / ".pivot")

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

    with gc.naming_unit(cache_dir, project=None):
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
    entry = gc.register_path(cache_dir, other_root)
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
            with gc.naming_unit(cache_dir, project=None):
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


def test_a_project_that_cannot_register_says_so_once(tmp_path: pathlib.Path) -> None:
    """A command that cannot write its project's register entry still runs, but warns once,
    naming the path and the reason: a project missing from the register is invisible to a
    `pivot gc` run from another project that shares the cache."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    cache_dir = root / ".pivot" / "cache"
    register = cache_dir / gc.REGISTER_DIR
    if register.exists():
        for entry in register.iterdir():
            entry.unlink()
        register.rmdir()
    register.write_text("not a directory\n")  # no entry can be written beneath it

    status = _run(root, "status")

    out = status.stdout + status.stderr
    assert status.returncode == 0, out
    assert out.count("Cannot record this project in the cache register") == 1, out
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


def _shared_project(path: pathlib.Path, shared_cache: pathlib.Path) -> pathlib.Path:
    """A project that caches into ``shared_cache`` with symlink checkout (the reporter's mode)."""
    root = _project(path)
    (root / ".pivot" / "config.yaml").write_text(
        f"cache:\n  dir: {shared_cache}\n  checkout_mode:\n  - symlink\n"
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
    gc.register_path(shared, origin).unlink(missing_ok=True)
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

    with gc.naming_exclusive(cache_dir):
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

    with gc.naming_exclusive(cache_dir):
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
