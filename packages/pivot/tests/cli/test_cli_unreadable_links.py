"""A symlink inside a directory artifact that pivot can lstat but not read (readlink fails, as on a
faulty FUSE or NFS mount), through the installed `pivot` binary: each command that meets it
exits 1 with one error that names the link and says what to do, prints no exception class or
traceback, and changes nothing.

A sitecustomize hook on PYTHONPATH makes ``os.readlink`` raise EIO for the links named in
``PIVOT_TEST_UNREADABLE_LINKS``, in the pivot process and in every worker process it starts.
``os.lstat`` is not patched, so each link is still a symlink.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys

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

_UNREADABLE_LINKS_HOOK = """\
import errno
import os

_LINKS = frozenset(os.environ["PIVOT_TEST_UNREADABLE_LINKS"].split(os.pathsep))
_readlink = os.readlink


def _readlink_failing_for_the_links(path, *, dir_fd=None):
    if dir_fd is None and os.path.abspath(path) in _LINKS:
        raise OSError(errno.EIO, os.strerror(errno.EIO), os.fspath(path))
    return _readlink(path, dir_fd=dir_fd)


os.readlink = _readlink_failing_for_the_links
"""

# An exception class name (OSError, ExceptionGroup, TaskGroup, UnreadableLinkError), a
# traceback, or click's Exit printed as an error: none of these may reach the user.
_EXCEPTION_TEXT = re.compile(r"Traceback|\b[A-Z]\w+(?:Error|Exception|Group)\b|\bExit\(")


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update(extra)
    return env


def _pivot(
    cwd: pathlib.Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_PIVOT, *args],
        cwd=cwd,
        env=env or _env(),
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
    """A project whose stage output ``data/p/big`` and tracked directory ``raw`` are committed
    to git and then checked out file by file as symlinks into the cache."""
    root = tmp_path / "proj"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init"))
    (root / "pipeline.py").write_text(_PIPELINE)
    (root / "raw").mkdir()
    for index in range(2):
        (root / "raw" / f"t{index}.txt").write_text(f"{index}\n")
    _ok(_pivot(root, "repro"))
    _ok(_pivot(root, "track", "raw"))
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=root,
        check=True,
    )
    for target in ("data/p/big", "raw"):
        shutil.rmtree(root / target)
        _ok(_pivot(root, "checkout", "--checkout-mode", "symlink", target))
        assert all(member.is_symlink() for member in (root / target).iterdir())
    return root


def _helper_unreadable(tmp_path: pathlib.Path, link: pathlib.Path) -> dict[str, str]:
    """An environment in which ``link`` can be lstat'd but not read."""
    hook = tmp_path / "hook"
    hook.mkdir(exist_ok=True)
    (hook / "sitecustomize.py").write_text(_UNREADABLE_LINKS_HOOK)
    env = _env(PIVOT_TEST_UNREADABLE_LINKS=str(link))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(hook), env.get("PYTHONPATH")]))
    return env


def _helper_refused(proc: subprocess.CompletedProcess[str], link: pathlib.Path) -> str:
    """The command exited 1 with one error naming ``link`` and what to do, and nothing else."""
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, out
    assert "Error in sitecustomize" not in out, out
    # Without whitespace, so a line wrapped by the console still matches.
    flat = "".join(out.split())
    assert "".join(f"Cannot read symlink '{link}':".split()) in flat, out
    assert "readable,orremovethelink." in flat, out
    assert _EXCEPTION_TEXT.search(out) is None, out
    return out


def test_repro_fails_the_stage_and_keeps_an_unreadable_user_symlink(
    tmp_path: pathlib.Path,
) -> None:
    """A user symlink inside a cached stage output that cannot be read: the output check cannot
    tell whether the output matches its lock, so the stage fails naming the link. The output is
    not restored from the cache over it, which would remove the link."""
    root = _helper_project(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("the user's file\n")
    user_link = root / "data" / "p" / "big" / "u.txt"
    user_link.symlink_to(outside)
    lock = root / ".pivot" / "stages" / "p" / "big.lock"
    lock_before = lock.read_bytes()

    env = _helper_unreadable(tmp_path, user_link)
    out = _helper_refused(_pivot(root, "repro", env=env), user_link)

    assert "Failed stages" in out, out
    assert os.readlink(user_link) == os.fspath(outside), "the link is kept"
    assert outside.read_text() == "the user's file\n"
    assert lock.read_bytes() == lock_before


def test_commit_refuses_an_unreadable_checkout_link(tmp_path: pathlib.Path) -> None:
    """``pivot commit`` of the stage whose output holds the link, and of the stage that reads
    that output: each stage is skipped naming the link, and no lock changes."""
    root = _helper_project(tmp_path)
    link = root / "data" / "p" / "big" / "f0.txt"
    locks = {name: root / ".pivot" / "stages" / "p" / f"{name}.lock" for name in ("big", "use")}
    before = {name: path.read_bytes() for name, path in locks.items()}
    env = _helper_unreadable(tmp_path, link)

    output = _helper_refused(_pivot(root, "commit", "p/big", env=env), link)
    downstream = _helper_refused(_pivot(root, "commit", env=env), link)

    assert "Failed 1 stage(s)" in output, output
    assert "unreadable deps" in downstream, downstream
    assert {name: path.read_bytes() for name, path in locks.items()} == before


def test_checkout_refuses_an_unreadable_checkout_link(tmp_path: pathlib.Path) -> None:
    """``pivot checkout`` over the output and ``pivot checkout --only-missing`` with another
    member deleted: both refuse naming the link, and nothing is written."""
    root = _helper_project(tmp_path)
    link = root / "data" / "p" / "big" / "f0.txt"
    env = _helper_unreadable(tmp_path, link)

    _helper_refused(
        _pivot(root, "checkout", "--checkout-mode", "symlink", "data/p/big", env=env), link
    )
    deleted = root / "data" / "p" / "big" / "f1.txt"
    deleted.unlink()
    _helper_refused(
        _pivot(
            root, "checkout", "--checkout-mode", "symlink", "--only-missing", "data/p/big", env=env
        ),
        link,
    )

    assert not deleted.is_symlink(), "the missing member is not restored"
    assert link.is_symlink()


def test_diff_refuses_an_unreadable_checkout_link(tmp_path: pathlib.Path) -> None:
    """``pivot diff`` refuses naming the link, rather than report the member as removed."""
    root = _helper_project(tmp_path)
    link = root / "data" / "p" / "big" / "f0.txt"

    _helper_refused(
        _pivot(root, "diff", "data/p/big", "--json", env=_helper_unreadable(tmp_path, link)), link
    )


def test_tracked_directory_commands_refuse_an_unreadable_link(tmp_path: pathlib.Path) -> None:
    """A tracked directory with the link: ``pivot track --force`` does not record the directory
    without it, ``pivot status --tracked-only`` does not call it modified, and ``pivot repro``
    stops before any stage runs. Each refuses naming the link."""
    root = _helper_project(tmp_path)
    link = root / "raw" / "t0.txt"
    pvt = root / "raw.pvt"
    pvt_before = pvt.read_bytes()
    env = _helper_unreadable(tmp_path, link)

    _helper_refused(_pivot(root, "track", "--force", "raw", env=env), link)
    _helper_refused(_pivot(root, "status", "--tracked-only", env=env), link)
    _helper_refused(_pivot(root, "repro", env=env), link)

    assert pvt.read_bytes() == pvt_before
    assert link.is_symlink()
