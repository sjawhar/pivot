"""A fingerprint-cache entry is written to the state directory of the run that produced it,
whenever and wherever the flush that writes it runs.

A flush runs where the producer no longer does. A TUI session's engine thread keeps
fingerprinting a reload after the quit has returned from the command, so process exit flushes
what that thread queued; a library caller flushes at process exit as well, after whatever its
program did next. Neither reads `core.state_dir` again, so a value edited in between takes
nothing: the session's entries stay in `s1`, the library's in `libstate`, and the edited
directory is never created.
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
    caller = tmp_path / "caller.py"
    caller.write_text(_LIBRARY_CALLER)

    ran = subprocess.run(
        [sys.executable, str(caller), str(root)],
        cwd=root,
        env=helpers.cli_env(home),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=300,
        check=False,
    )

    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert not (root / "otherstate").exists(), sorted(
        p.name for p in (root / "otherstate").iterdir()
    )
    assert _cached_keys(root / "libstate", b"sm:") > 0, ran.stdout + ran.stderr
    assert _cached_keys(root / "libstate", b"fp:") > 0, ran.stdout + ran.stderr
