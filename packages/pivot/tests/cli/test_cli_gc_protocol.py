"""The naming-lock protocol, driven through real pivot processes.

`pivot gc` holds ``naming.lock`` exclusive while it deletes; every writer holds it shared for one
commit unit, from its first look at a cache object to the record that names it. These tests hold
the exclusive side themselves and run real ``pivot repro`` processes against it: a writer site
that stops taking the shared side lets its commit land while the test holds the lock, and the
test fails. Each writer site has its own test, so a site that loses its lock is caught by the
test for that site and not by another's. The last test pins the other side of the bargain: the
shared lock covers only the commit, never a stage's execution, so a collection proceeds while a
long stage runs.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

from pivot.storage import gc

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
