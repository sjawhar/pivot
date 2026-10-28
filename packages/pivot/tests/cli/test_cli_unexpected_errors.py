"""Errors no pivot command expected, through the installed `pivot` binary.

With a cache object's two-character directory at mode 000, `pivot verify` and
`pivot checkout --only-missing` fail with the operating system's message and the path it names,
on one line with no exception class, errno or traceback. `pivot checkout` with cache objects
missing exits 1 after its own report, with no `Error: Exit(1)` line. A stage whose worker process
dies is logged as one error line; only `--verbose` adds the traceback.
"""

from __future__ import annotations

import errno
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def big() -> Annotated[dict[str, str], directory]:
    return {f"f{index}.txt": f"{index}\\n" for index in range(3)}


@stage
def use(files: dict[str, str]) -> str:
    return str(len(files))


with Pipeline("p") as pipeline:
    use(big())
"""

# An exception class name, an OSError's "[Errno N]", a traceback, or click's Exit printed as an
# error: none of these may reach the user.
_PYTHON_TEXT = re.compile(r"Traceback|\[Errno|\b[A-Z]\w+(?:Error|Exception|Group)\b|\bExit\(")

_EACCES = os.strerror(errno.EACCES)

_DYING_PIPELINE = """\
import os

from pivot import Pipeline, stage


@stage
def die() -> str:
    os._exit(3)


with Pipeline("p") as pipeline:
    die()
"""

# loky's message for a worker that exited with code 3: the reason `pivot repro --json` reports
# for the stage, unchanged by how the error is logged.
_WORKER_DIED = (
    "A worker process managed by the executor was unexpectedly terminated. This could be caused "
    + "by a segmentation fault while calling the function or by an excessive memory usage "
    + "causing the Operating System to kill the worker.\n\nThe exit codes of the workers are "
    + "{EXIT(3)}\nDetailed tracebacks of the workers should have been printed to stderr in the "
    + "executor process if faulthandler was not disabled."
)

_TRACEBACK = "Traceback (most recent call last)"


def _pivot(cwd: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_PIVOT, *args],
        cwd=cwd,
        env={k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def _helper_project(tmp_path: pathlib.Path) -> pathlib.Path:
    """A project whose stage output ``data/p/big`` holds three files, from ``pivot repro``."""
    root = tmp_path / "proj"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init"))
    (root / "pipeline.py").write_text(_PIPELINE)
    _ok(_pivot(root, "repro"))
    return root


def _helper_failed(proc: subprocess.CompletedProcess[str]) -> list[str]:
    """The stderr lines of a command that exited 1 and printed no Python exception text."""
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, out
    assert _PYTHON_TEXT.search(out) is None, out
    return proc.stderr.splitlines()


@pytest.mark.skipif(os.geteuid() == 0, reason="root can search a directory at mode 000")
def test_an_unsearchable_cache_directory_is_named_in_one_line(tmp_path: pathlib.Path) -> None:
    """A symlink checkout whose first member's cache object lies in a directory at mode 000:
    `verify` and `checkout --only-missing` each print one error line, the OS's message and the
    path it could not reach, where they printed `PermissionError(13, 'Permission denied')` and
    `[Errno 13] Permission denied: '<path>'`."""
    root = _helper_project(tmp_path)
    output = root / "data" / "p" / "big"
    shutil.rmtree(output)
    _ok(_pivot(root, "checkout", "--checkout-mode", "symlink", "data/p/big"))
    cache_object = output / "f0.txt"
    cache_object = cache_object.parent / os.readlink(cache_object)
    prefix_dir = cache_object.parent
    # Another member whose object is elsewhere, so --only-missing has something to restore
    missing = next(
        member
        for member in sorted(output.iterdir())
        if (member.parent / os.readlink(member)).parent != prefix_dir
    )
    missing.unlink()

    prefix_dir.chmod(0)
    try:
        verify = _pivot(root, "verify")
        checkout = _pivot(
            root, "checkout", "--checkout-mode", "symlink", "--only-missing", "data/p/big"
        )
    finally:
        prefix_dir.chmod(0o755)

    verify_errors = _helper_failed(verify)
    assert f"Failed to get explanation for p/big: {_EACCES}: {cache_object}" in verify_errors
    assert verify_errors[-1] == f"Error: {_EACCES}: {prefix_dir}"
    assert _helper_failed(checkout)[-1] == f"Error: {_EACCES}: {cache_object}"
    assert not missing.is_symlink(), "nothing is restored"


def test_checkout_with_missing_cache_objects_exits_1_after_its_report(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot checkout` reports the files it cannot restore and exits 1; the exit is not printed
    as an error of its own (`Error: Exit(1)`)."""
    root = _helper_project(tmp_path)
    shutil.rmtree(root / ".pivot" / "cache" / "files")
    shutil.rmtree(root / "data" / "p" / "big")

    proc = _pivot(root, "checkout", "data/p/big")

    errors = _helper_failed(proc)
    report = proc.stdout.splitlines()
    assert "Missing 3 file(s) from:" in report
    assert report[-1] == "Run 'pivot pull' to fetch from remote storage."
    assert not any(line.startswith("Error:") for line in errors), proc.stderr


def test_a_dead_worker_is_one_error_line_and_only_verbose_shows_its_traceback(
    tmp_path: pathlib.Path,
) -> None:
    """A stage whose worker process exits: `pivot repro` logs `Stage <name> failed: <reason>` and
    no traceback, `pivot --verbose repro` adds the traceback, and `pivot repro --json` reports
    loky's message as the stage's reason, exactly as before."""
    root = tmp_path / "proj"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init"))
    (root / "pipeline.py").write_text(_DYING_PIPELINE)

    default = _pivot(root, "repro")
    verbose = _pivot(root, "--verbose", "repro")
    as_json = _pivot(root, "repro", "--json")

    error_line = f"Stage p/die failed: Unexpected TerminatedWorkerError: {_WORKER_DIED}"
    for proc in (default, verbose):
        out = proc.stdout + proc.stderr
        assert proc.returncode == 1, out
        # Without whitespace, so a line wrapped by the console still matches
        assert "".join(error_line.split()) in "".join(out.split()), out
    assert _TRACEBACK not in default.stdout + default.stderr
    assert _TRACEBACK in verbose.stdout + verbose.stderr
    events = [json.loads(line) for line in as_json.stdout.splitlines() if line.startswith("{")]
    assert [
        (event["stage"], event["status"], event["reason"])
        for event in events
        if event["type"] == "stage_complete"
    ] == [("p/die", "failed", _WORKER_DIED)], as_json.stdout + as_json.stderr
