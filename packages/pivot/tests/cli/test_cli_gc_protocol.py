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
import itertools
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
from pivot.storage import cache, gc, lock, state

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


def _assert_waits_for_a_collection(
    root: pathlib.Path,
    cache_dir: pathlib.Path,
    args: tuple[str, ...],
    *,
    unblocked: float,
    landed: Callable[[], bool],
) -> str:
    """Run ``pivot <args>`` while holding the naming lock exclusive: its record must not land
    until the lock is released, then must. ``unblocked`` is how long the same kind of command
    took without the lock; the test waits three times that before judging it blocked."""
    what = f"pivot {' '.join(args)}"
    assert not landed(), f"{what}: its record is there before it ran"
    with gc.naming_exclusive(cache_dir):
        proc = _start(root, *args, env=_env())
        try:
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
    before = _locks(root)
    gate = tmp_path / "gate"
    gate.mkdir()

    with gc.naming_exclusive(cache_dir):
        proc = _start(root, "repro", "--force", env=_env(GC_PROTOCOL_GATE=str(gate)))
        try:
            _wait_for(gate / "started", proc, timeout=120)
            time.sleep(2.0)  # the commit follows the stage body within milliseconds
            assert proc.poll() is None, "a stage commit did not wait for the collection"
            assert _locks(root) == before, "a stage lock was written during the collection"
        except BaseException:
            proc.kill()
            raise

    _finish(proc)
    assert _locks(root) != before, "the commit never landed after the collection"


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
        root, cache_dir, ("commit",), unblocked=unblocked, landed=lambda: _locks(root) != before
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


@contextlib.contextmanager
def _holding_prefix(object_root: pathlib.Path, prefix: str) -> Iterator[None]:
    """Hold a prefix directory's lock, so a collection that reaches it waits there."""
    fd = cache.open_cache_lock(object_root / prefix / cache.PREFIX_LOCK)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _content_in_prefix(
    scratch: pathlib.Path, prefix: str, make: Callable[[int], str]
) -> tuple[str, str]:
    """The first text ``make`` yields whose cache key starts with ``prefix``, and that key."""
    probe = scratch / "probe"
    for index in itertools.count():
        text = make(index)
        probe.write_text(text)
        key, _ = cache.hash_file(probe)
        if key.startswith(prefix):
            probe.unlink()
            return text, key
    raise AssertionError("unreachable")


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
    output_text, victim = _content_in_prefix(tmp_path, "ff", lambda i: f"V{i}\n")
    _publish(object_root, tmp_path, output_text)
    _publish(object_root, tmp_path, "an orphan the collection removes first\n")
    (root / "data" / "raw" / "input.txt").write_text(output_text.lower())
    _register_fresh(root)
    before = _locks(root)
    gate = tmp_path / "gate"
    gate.mkdir()

    with contextlib.ExitStack() as held:
        # Let gc plan (it lists every prefix under that prefix's lock), then pin it at `ff`.
        with gc.naming_unit(cache_dir):
            collector = _start_gc(root)
            _wait_until(
                "the turnstile", lambda: _gc_waits_for_the_naming_lock(cache_dir), collector
            )
            held.enter_context(_holding_prefix(object_root, "ff"))
        _wait_until("its exclusive lock", lambda: _gc_holds_the_naming_lock(cache_dir), collector)
        # gc is deleting and will wait at `ff`, where the victim is, while it holds the lock.
        writer = _start(root, "repro", "--force", env=_env(GC_PROTOCOL_GATE=str(gate)))
        try:
            _wait_for(gate / "started", writer, timeout=120)
            time.sleep(2.0)  # the commit follows the stage body within milliseconds
            landed = writer.poll() is not None or _locks(root) != before
        except BaseException:
            writer.kill()
            collector.kill()
            raise

    _finish(collector)
    _finish(writer)
    assert cache.get_cache_path(object_root, victim).exists(), "an object a stage names is gone"
    assert victim in "".join(_locks(root).values()), "the stage lock does not name its output"
    assert (root / "data" / "shadow" / "shout.txt").read_text() == output_text
    assert not landed, "a stage commit named an object while gc was deleting"


def test_pivot_gc_rereads_the_records_under_its_lock(tmp_path: pathlib.Path) -> None:
    """A writer already in its unit when `pivot gc` comes for the lock names an object and only
    then finishes. gc waits for it, re-reads every record under the lock, and keeps that object.
    gc is pinned at the victim's prefix for a second first, so a gc that did not wait for the
    writer, or deleted against the records it read before the lock, has already read them and
    would remove the object once released."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    cache_dir = _register_fresh(root)
    object_root = cache_dir / "files"
    text, _ = _content_in_prefix(tmp_path, "ff", lambda i: f"named while gc waits {i}\n")
    victim = _publish(object_root, tmp_path, text)

    with contextlib.ExitStack() as held:
        with gc.naming_unit(cache_dir):  # a writer mid-commit
            collector = _start_gc(root)
            _wait_until(
                "the turnstile", lambda: _gc_waits_for_the_naming_lock(cache_dir), collector
            )
            held.enter_context(_holding_prefix(object_root, "ff"))
            time.sleep(1.0)  # a gc that skipped the wait has read its records by now
            _name_in_lock(root / ".pivot" / "stages", "late/stage", victim)
        time.sleep(0.5)

    _finish(collector)
    assert cache.get_cache_path(object_root, victim).exists(), "gc removed an object a record names"


def test_pivot_gc_rereads_the_register_under_its_lock(tmp_path: pathlib.Path) -> None:
    """A project that registers while `pivot gc` waits for the naming lock is read under it: the
    objects its records name are kept, though the plan was built before it existed."""
    root = _project(tmp_path / "project")
    _ok(_run(root, "repro"))
    cache_dir = _register_fresh(root)
    victim = _publish(cache_dir / "files", tmp_path, "named by a project that registers late\n")
    late = tmp_path / "late-project"
    (late / ".pivot" / "stages").mkdir(parents=True)

    with gc.naming_unit(cache_dir):  # a command registering its project
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

    with gc.naming_unit(cache_dir):
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
            with gc.naming_unit(cache_dir):
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
