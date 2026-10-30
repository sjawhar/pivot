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
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

import pytest

from pivot import run_history, types
from pivot.storage import cache, gc, listing, lock, state, track

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
    path = gc.project_register_path(cache_dir, root, root / ".pivot")
    assert path is not None, f"{root} has no project id"
    return path


def _register_fresh(root: pathlib.Path) -> pathlib.Path:
    """Write this project's register entry now, so the command entry takes no lock, and return
    the cache. The tests then hold the exclusive lock against the writer sites alone."""
    cache_dir = root / ".pivot" / "cache"
    gc.register_project(cache_dir, root, root / ".pivot")
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
    entry = cache_dir / gc.REGISTER_DIR / "project-on-another-node.json"
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


def test_a_command_refuses_when_it_cannot_register(tmp_path: pathlib.Path) -> None:
    """A writer cannot name objects when its register entry cannot be written."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    cache_dir = root / ".pivot" / "cache"
    register = cache_dir / gc.REGISTER_DIR
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
    if (origin_entry := gc.project_register_path(shared, origin, origin / ".pivot")) is not None:
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
    register = shared / gc.REGISTER_DIR
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
    register = internal / gc.REGISTER_DIR
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
    for path in member_objects:
        assert path.exists(), f"gc removed {path} with no listing to prove it dead"
    assert cache.get_cache_path(objects, dead).exists(), "a refused collection removed an object"


_MOUNT_DRIVER = """\
import pathlib
import subprocess
import sys

from pivot.storage import gc

root, state_dir, cache_dir, source, target = (pathlib.Path(a) for a in sys.argv[1:6])
gc.register_project(cache_dir, root, state_dir)
subprocess.run(
    ["sudo", "-n", "mount", "--bind", str(source), str(target)], check=True, capture_output=True
)
gc.register_project(cache_dir, root, state_dir)
print(gc.project_register_path(cache_dir, root, state_dir))
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
    return {
        "a.txt": text,
        "inner.pvt": "path: inner\\nhash: 0123456789abcdef\\nsize: 3\\nnfiles: 1\\n",
    }


with Pipeline("shadow") as pipeline:
    split(pipeline.input("input.txt", t=str))
"""


def test_pivot_gc_reads_a_stages_own_pvt_member_as_its_data(tmp_path: pathlib.Path) -> None:
    """#540: a stage may write a `*.pvt` into its own directory output; it is the stage's data,
    not a tracked file. One that names a directory no listing describes must not make the
    collection refuse, and the output's own objects stay."""
    root = _project(tmp_path / "project")
    (root / "pipeline.py").write_text(_MEMBER_PIPELINE)
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
