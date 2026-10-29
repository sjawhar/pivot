"""Two teammates share a git repository, each with their own cache. B has run the stage; A runs
it with new params and commits the lock and the listing beside it; B runs ``git pull`` but not
``pivot pull``. The stage writes a sidecar-shaped ``fake.pvt`` in its directory output. B's
``status`` does not call that file a user's, B's ``repro`` reruns the stage over it, and B's
target-less ``checkout --force`` restores no path it names: the stage's own members are read
from the listing committed beside the lock, which B's cache lacks.

Runs the installed ``pivot`` console script.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, StageParams, directory, stage


class TreeParams(StageParams):
    n: int = 1


@stage
def tree(params: TreeParams) -> Annotated[dict[str, str], directory]:
    return {
        "value.txt": str(params.n),
        "fake.pvt": f"path: value.txt\\nhash: abcdef{params.n:010x}\\nsize: 1\\n",
    }


with Pipeline("b") as pipeline:
    tree(params=TreeParams())
"""

_OWNERSHIP_TEXT = ("outside every stage output", "Refusing", "refuses to replace")


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update(
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.com",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.com",
    )
    return env


def _git(cwd: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, env=_env(), check=True, capture_output=True, text=True
    ).stdout


def _pivot(cwd: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_PIVOT, *args], cwd=cwd, env=_env(), capture_output=True, text=True, timeout=300
    )


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def _no_ownership_text(out: str) -> None:
    flat = " ".join(out.split())
    for text in _OWNERSHIP_TEXT:
        assert text not in flat, out


def _b_after_as_rerun_arrived_through_git(tmp_path: pathlib.Path) -> pathlib.Path:
    """Two clones of one repository. A and B each run the stage at n=1 with their own cache; A
    then reruns it at n=2 and commits the lock and the listing beside it; B runs ``git pull``
    and not ``pivot pull``, so B's cache holds n=1 alone. Returns B's root."""
    a = tmp_path / "a"
    a.mkdir()
    _git(a, "init", "-q", "-b", "main")
    _ok(_pivot(a, "init"))
    (a / "pipeline.py").write_text(_PIPELINE)
    (a / ".gitignore").write_text("/data/\n__pycache__/\n")
    _ok(_pivot(a, "repro"))
    _git(a, "add", "-A")
    _git(a, "commit", "-qm", "n=1")

    b = tmp_path / "b"
    _git(tmp_path, "clone", "-q", str(a), str(b))
    _ok(_pivot(b, "repro"))  # B runs the stage itself: its own cache, n=1
    assert "abcdef0000000001" in (b / "data" / "b" / "tree" / "fake.pvt").read_text()

    (a / "params.yaml").write_text("b/tree:\n  n: 2\n")
    _ok(_pivot(a, "repro"))
    _git(a, "add", "-A")
    _git(a, "commit", "-qm", "n=2")

    _git(b, "checkout", "--", ".")  # B's own rerun may have rewritten its lock: discard
    _git(b, "pull", "-q", "--no-rebase", "origin", "main")  # no `pivot pull`
    return b


def test_after_git_pull_the_stages_own_pvt_stays_the_stages(tmp_path: pathlib.Path) -> None:
    b = _b_after_as_rerun_arrived_through_git(tmp_path)
    fake = b / "data" / "b" / "tree" / "fake.pvt"

    status = _pivot(b, "status")
    rerun = _pivot(b, "repro")

    assert status.returncode == 0, status.stdout + status.stderr
    _no_ownership_text(status.stdout + status.stderr)
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    _no_ownership_text(rerun.stdout + rerun.stderr)
    assert (b / "data" / "b" / "tree" / "value.txt").read_text() == "2"
    assert "abcdef0000000002" in fake.read_text()


def test_after_git_pull_a_targetless_forced_checkout_restores_no_path_the_member_names(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot checkout --force` with no target restores every tracked file as well, so it
    discovers the project's ``*.pvt`` files. ``fake.pvt`` is the stage's own member by the
    listing A committed beside the lock, which B's cache has no copy of, so it drives no
    restore. B is still missing A's objects, so the stage's own output cannot be restored: the
    checkout fails over that output's objects, names none of the member's, and leaves the
    workspace as it was (BET-412)."""
    b = _b_after_as_rerun_arrived_through_git(tmp_path)
    tree = b / "data" / "b" / "tree"
    written = (tree / "fake.pvt").read_text()

    forced = _pivot(b, "checkout", "--force")

    out = forced.stdout + forced.stderr
    assert forced.returncode == 1, out
    assert "cache object missing" in out, out
    _no_ownership_text(out)
    assert "abcdef" not in out, out  # no restore of the path the stage's own member names
    assert (tree / "value.txt").read_text() == "1", out
    assert (tree / "fake.pvt").read_text() == written
