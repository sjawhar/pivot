"""A fingerprint-cache entry is written to the state directory of the run that produced it,
whenever and wherever the flush that writes it runs, and whenever the value is edited.

A flush runs where the producer no longer does. A TUI session's engine thread keeps
fingerprinting a reload after the quit has returned from the command, so process exit flushes
what that thread queued; a library caller flushes at process exit as well, after whatever its
program did next. Neither reads `core.state_dir` again, so a value edited in between takes
nothing: the session's entries stay in `s1`, the library's in `libstate`, and the edited
directory is never created.

A run reads the value once, before its first stage, and holds it: the CLI invocation holds it
for a command, and an engine run with no invocation around it holds it for the run
(`state_dir.run_invocation`). So a stage that edits `core.state_dir` moves no entry of its own
run either, including the stages the engine fingerprints only once that stage has finished.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
import typing

from pivot.storage import state
from tests import helpers

if typing.TYPE_CHECKING:
    import pathlib

_ONE_STAGE_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""

# The reload's fingerprinting must still be running when the quit reaches the close callbacks,
# so the reloaded pipeline has many stages and says when its import starts.
_RELOAD_STAGES = 200

_LIBRARY_CALLER = """\
import os
import pathlib
import sys


def main() -> None:
    os.chdir(sys.argv[1])

    from pivot import discovery, executor

    executor.run(pipeline=discovery.discover_pipeline())
    pathlib.Path(".pivot/config.yaml").write_text("core:\\n  state_dir: otherstate\\n")


if __name__ == "__main__":
    main()
"""

_CHAIN_PIPELINE = """\
import pathlib

from pivot import Pipeline, stage


@stage
def first(value: str) -> str:
    pathlib.Path(".pivot/config.yaml").write_text("core:\\n  state_dir: otherstate\\n")
    return value.upper()


@stage
def second(value: str) -> str:
    return value + "-second"


with Pipeline("lib") as pipeline:
    second(first(pipeline.input("input.txt", t=str)))
"""

_LONGER_CHAIN_PIPELINE = """\
import pathlib

from pivot import Pipeline, stage


def helper_a(value: str) -> str:
    return value.upper()


def helper_b(value: str) -> str:
    return value + "b"


def helper_c(value: str) -> str:
    return value + "c"


@stage
def a(value: str) -> str:
    pathlib.Path(".pivot/config.yaml").write_text("core:\\n  state_dir: otherstate\\n")
    return helper_a(value)


@stage
def b(value: str) -> str:
    return helper_b(value)


@stage
def c(value: str) -> str:
    return helper_c(value)


with Pipeline("lib") as pipeline:
    c(b(a(pipeline.input("input.txt", t=str))))
"""

_RUN_CALLER = """\
import os
import sys


def main() -> None:
    os.chdir(sys.argv[1])

    from pivot import discovery, executor

    results = executor.run(pipeline=discovery.discover_pipeline())
    assert results, results
    assert all(r["status"] == "ran" for r in results.values()), results


if __name__ == "__main__":
    main()
"""

_TWO_RUNS_CALLER = """\
import os
import pathlib
import sys


def main() -> None:
    os.chdir(sys.argv[1])

    from pivot import discovery, executor

    pipeline = discovery.discover_pipeline()
    executor.run(pipeline=pipeline)
    pathlib.Path(".pivot/config.yaml").write_text("core:\\n  state_dir: otherstate\\n")
    executor.run(pipeline=pipeline, force=True)


if __name__ == "__main__":
    main()
"""

_TWO_INDEPENDENT_STAGES = """\
from pivot import Pipeline, stage


def helper_first(value: str) -> str:
    return value.upper()


def helper_second(value: str) -> str:
    return value + "-second"


@stage
def first(value: str) -> str:
    return helper_first(value)


@stage
def second(value: str) -> str:
    return helper_second(value)


with Pipeline("lib") as pipeline:
    raw = pipeline.input("input.txt", t=str)
    first(raw)
    second(raw)
"""

_DIRECT_FINGERPRINT_CALLER = """\
import os
import pathlib
import sys


def main() -> None:
    os.chdir(sys.argv[1])

    from pivot import discovery, fingerprint

    pipeline = discovery.discover_pipeline()
    pipeline.ensure_fingerprint("lib/first")
    pathlib.Path(".pivot/config.yaml").write_text("core:\\n  state_dir: otherstate\\n")
    pipeline.ensure_fingerprint("lib/second")
    fingerprint.flush_and_close()


if __name__ == "__main__":
    main()
"""


def _reload_pipeline(count: int) -> str:
    """A pipeline of ``count`` independent stages that reports the start of its import."""
    lines = [
        "import pathlib",
        "",
        "from pivot import Pipeline, stage",
        "",
        "pathlib.Path('reload_started').touch()",
    ]
    for index in range(count):
        lines += [
            "",
            "",
            f"def helper_{index}(value: str) -> str:",
            f"    return value.upper() + '{index}'",
            "",
            "",
            "@stage",
            f"def stage_{index}(value: str) -> str:",
            f"    return helper_{index}(value)",
        ]
    lines += [
        "",
        "",
        'with Pipeline("test") as pipeline:',
        '    raw = pipeline.input("input.txt", t=str)',
    ]
    lines += [f"    stage_{index}(raw)" for index in range(count)]
    return "\n".join(lines) + "\n"


def _project(tmp_path: pathlib.Path, state_dir: str) -> tuple[pathlib.Path, pathlib.Path]:
    home = tmp_path / "home"
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {state_dir}\n")
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("first\n")
    (root / "pipeline.py").write_text(_ONE_STAGE_PIPELINE)
    return home, root


def _cached_keys(state_dir: pathlib.Path, prefix: bytes) -> int:
    """How many fingerprint-cache entries of one kind the state database holds."""
    with state.StateDB(state_dir, readonly=True) as db:
        return len(list(db.iter_prefix(prefix)))


def _run_library_caller(
    tmp_path: pathlib.Path, home: pathlib.Path, root: pathlib.Path, source: str
) -> subprocess.CompletedProcess[str]:
    """Run ``source`` as a program that imports pivot, never ``pivot.cli``, in ``root``."""
    caller = tmp_path / "caller.py"
    caller.write_text(source)
    return subprocess.run(
        [sys.executable, str(caller), str(root)],
        cwd=root,
        env=helpers.cli_env(home),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=300,
        check=False,
    )


def _wait_for(
    proc: subprocess.Popen[bytes],
    log: pathlib.Path,
    done: typing.Callable[[], bool],
    what: str,
) -> None:
    """Wait up to 60 seconds for ``done`` while the pivot process keeps running."""
    deadline = time.monotonic() + 60
    while not done():
        assert proc.poll() is None, f"pivot exited before {what}:\n{log.read_text()}"
        assert time.monotonic() < deadline, f"no {what} after 60 s:\n{log.read_text()}"
        time.sleep(0.05)


def test_a_tui_quit_while_the_engine_fingerprints_writes_its_caches_to_its_own_state_dir(
    tmp_path: pathlib.Path,
) -> None:
    home, root = _project(tmp_path, "s1")
    log = tmp_path / "pivot.log"
    started = root / "reload_started"
    with log.open("wb") as output:
        proc = subprocess.Popen(
            [helpers.PIVOT, "repro", "--watch", "--tui", "--debounce", "50"],
            cwd=root,
            env=helpers.cli_env(home, TERM="xterm-256color"),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    try:
        _wait_for(proc, log, (root / "data" / "test" / "process.txt").exists, "startup run")
        # The reload re-fingerprints every stage: the engine thread is still doing that when
        # the quit below returns from the command.
        (root / "pipeline.py").write_text(_reload_pipeline(_RELOAD_STAGES))
        _wait_for(proc, log, started.exists, "reload")
        (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: s2\n")
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    output_text = log.read_text(errors="replace")
    assert proc.returncode == 0, output_text
    assert not (root / "s2").exists(), sorted(p.name for p in (root / "s2").iterdir())
    assert _cached_keys(root / "s1", b"sm:") > 0, output_text
    assert _cached_keys(root / "s1", b"fp:") > 0, output_text


def test_a_library_caller_that_edits_the_value_before_exit_keeps_its_caches(
    tmp_path: pathlib.Path,
) -> None:
    home, root = _project(tmp_path, "libstate")

    ran = _run_library_caller(tmp_path, home, root, _LIBRARY_CALLER)

    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert not (root / "otherstate").exists(), sorted(
        p.name for p in (root / "otherstate").iterdir()
    )
    assert _cached_keys(root / "libstate", b"sm:") > 0, ran.stdout + ran.stderr
    assert _cached_keys(root / "libstate", b"fp:") > 0, ran.stdout + ran.stderr


def test_a_library_run_whose_stage_edits_the_value_keeps_later_entries_in_its_dir(
    tmp_path: pathlib.Path,
) -> None:
    """`first -> second`: the engine fingerprints `second` only once `first` has finished, so
    the edited value is on disk when `second`'s entries are produced. They belong to the run,
    which chose `libstate` before `first` ran."""
    home, root = _project(tmp_path, "libstate")
    (root / "pipeline.py").write_text(_CHAIN_PIPELINE)

    ran = _run_library_caller(tmp_path, home, root, _RUN_CALLER)

    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert not (root / "otherstate").exists(), sorted(
        p.relative_to(root).as_posix() for p in (root / "otherstate").rglob("*")
    )
    # One stage manifest for each of the two stages, and the run's manifest, in libstate.
    assert _cached_keys(root / "libstate", b"sm:") == 2, ran.stdout + ran.stderr
    assert _cached_keys(root / "libstate", b"run:") == 1, ran.stdout + ran.stderr


def test_a_library_run_of_a_longer_chain_keeps_every_entry_in_its_own_state_dir(
    tmp_path: pathlib.Path,
) -> None:
    """`a -> b -> c`, where `a` edits the value: the stages the engine fingerprints after it
    file their AST hashes and manifests in the run's directory, as 0.5.2 and main do."""
    home, root = _project(tmp_path, "libstate")
    (root / "pipeline.py").write_text(_LONGER_CHAIN_PIPELINE)

    ran = _run_library_caller(tmp_path, home, root, _RUN_CALLER)

    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert not (root / "otherstate").exists(), sorted(
        p.relative_to(root).as_posix() for p in (root / "otherstate").rglob("*")
    )
    counts = {
        prefix: _cached_keys(root / "libstate", prefix) for prefix in (b"fp:", b"sm:", b"run:")
    }
    # Each stage files its own function and its helper, its manifest, and the run files one
    # manifest of its own.
    assert counts == {b"fp:": 6, b"sm:": 3, b"run:": 1}, (counts, ran.stdout + ran.stderr)


def test_a_library_caller_sees_a_value_edited_between_two_runs(tmp_path: pathlib.Path) -> None:
    """Each engine run starts its own record, so the second run reads the edited value."""
    home, root = _project(tmp_path, "libstate")

    ran = _run_library_caller(tmp_path, home, root, _TWO_RUNS_CALLER)

    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert _cached_keys(root / "libstate", b"run:") == 1, ran.stdout + ran.stderr
    assert _cached_keys(root / "otherstate", b"run:") == 1, ran.stdout + ran.stderr


def test_direct_fingerprinting_sees_a_value_edited_between_two_stages(
    tmp_path: pathlib.Path,
) -> None:
    """Outside a run, the resolution memory answers only for the bytes it parsed."""
    home, root = _project(tmp_path, "libstate")
    (root / "pipeline.py").write_text(_TWO_INDEPENDENT_STAGES)

    ran = _run_library_caller(tmp_path, home, root, _DIRECT_FINGERPRINT_CALLER)

    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert _cached_keys(root / "libstate", b"sm:") == 1, ran.stdout + ran.stderr
    assert _cached_keys(root / "otherstate", b"sm:") == 1, ran.stdout + ran.stderr
    assert _cached_keys(root / "libstate", b"fp:") == 2, ran.stdout + ran.stderr
    assert _cached_keys(root / "otherstate", b"fp:") == 2, ran.stdout + ran.stderr
