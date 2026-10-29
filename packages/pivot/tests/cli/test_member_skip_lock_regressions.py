"""PR540 reviewer, round 7 (sjawhar/pivot#540 at 31f438d2).

Regression locks for the round-6 findings the head fixes without a test; each passes at the head
and fails at the round-6 head ca47c682:

- M1: another stage's committed listing that does not match its lock (a lock reverted without
  its listing) is read without a warning: the stage is not the command's.
- M2: a deleted stage's lock records `data/b/tree`, and `data/b` is now a symlink to
  `data/b_real`. The `.pvt` walk does not enter `data/b`, so the files under `data/b_real/tree`
  are tracked files to it; `track --force` must not refuse them as the deleted stage's.
- M3: with `core.state_dir` outside the project, the refusal names the lock by its absolute
  path and prints no "outside base" warning.
- M4: `track`'s refusal of a file a deleted stage left names the remedy, and the remedy works:
  remove the lock, then `pivot track --force <path>`; the file is then tracked.

Red tests; each passes on main cea092a6 and fails at the head:

- A: a lock of a stage the command does not run holds a YAML scalar the safe loader cannot
  build (`when: 2024-13-01`); commands that never run that stage must not fail.
- C: the deleted stage's member is a symlink to its cache object, as pivot links each file of an
  output when the cache lies on another filesystem; `track` must refuse the file it left, or the
  file must then be tracked.

`_pivot` keeps `PYTHONPATH`, so a mutant run that points it at a mutated `src` reaches the CLI.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

from pivot.storage import cache

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

_ROOT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def alive() -> str:
    return "alive"


with Pipeline("{name}") as pipeline:
    alive()
"""

_PARAMS_MEMBER_PIPELINE = """\
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

_SIDECAR_MEMBER_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {{"value.txt": "1", "value.txt.pvt": {sidecar!r}}}


with Pipeline("b") as pipeline:
    tree()
"""


def _pivot(cwd: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    return subprocess.run(
        [_PIVOT, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=300
    )


def _ok(proc: subprocess.CompletedProcess[str]) -> subprocess.CompletedProcess[str]:
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc


def _tracked(root: pathlib.Path) -> list[str]:
    status = _ok(_pivot(root, "status", "--tracked-only", "--json"))
    return sorted(info["path"] for info in json.loads(status.stdout)["tracked_files"])


def _init(root: pathlib.Path) -> None:
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init"))
    (root / "pipeline.py").write_text(_ROOT_PIPELINE.format(name="a"))


def test_another_stages_stale_committed_listing_is_read_quietly(tmp_path: pathlib.Path) -> None:
    """b ran at n=1, then at n=2; the listing beside b/tree's lock is then put back to n=1's, so
    it does not hash to the lock's record and the cache's copy decides. From the root (pipeline
    `a` loaded, one tracked file), `--quiet status --tracked-only` exits 0 with nothing on stderr,
    and b's member is still skipped."""
    root = tmp_path / "project"
    _init(root)
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "kept.txt").write_text("kept")
    _ok(_pivot(root, "track", "data/raw/kept.txt"))
    (root / "other").mkdir()
    (root / "other" / "pipeline.py").write_text(_PARAMS_MEMBER_PIPELINE)
    _ok(_pivot(root / "other", "repro"))
    listing_file = root / ".pivot" / "stages" / "b" / "tree=.listing"
    first = listing_file.read_bytes()
    (root / "params.yaml").write_text("b/tree:\n  n: 2\n")
    _ok(_pivot(root / "other", "repro"))
    assert listing_file.read_bytes() != first
    listing_file.unlink()
    listing_file.write_bytes(first)
    _ok(_pivot(root, "repro"))

    quiet = _pivot(root, "--quiet", "status", "--tracked-only")

    assert quiet.returncode == 0, quiet.stdout + quiet.stderr
    assert quiet.stderr == "", quiet.stderr
    assert _tracked(root) == ["data/raw/kept.txt"]


def _deleted_stage_left_a_sidecar(
    tmp_path: pathlib.Path, state_dir: pathlib.Path | None = None
) -> pathlib.Path:
    """b's stage wrote `value.txt` and a `value.txt.pvt` with the bytes `pivot track` writes for
    it, then the stage was deleted; its lock stays (in `state_dir`'s stages directory when given,
    set as `core.state_dir` before anything runs, else in `.pivot`)."""
    scratch = tmp_path / "scratch"
    _init(scratch)
    (scratch / "value.txt").write_text("1")
    _ok(_pivot(scratch, "track", "value.txt"))
    sidecar = (scratch / "value.txt.pvt").read_text()
    root = tmp_path / "project"
    _init(root)
    if state_dir is not None:
        (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {state_dir}\n")
    (root / "other").mkdir()
    (root / "other" / "pipeline.py").write_text(_SIDECAR_MEMBER_PIPELINE.format(sidecar=sidecar))
    for cwd in (root / "other", root):
        _ok(_pivot(cwd, "repro"))
    assert (root / "data" / "b" / "tree" / "value.txt.pvt").read_text() == sidecar
    (root / "other" / "pipeline.py").write_text(_ROOT_PIPELINE.format(name="b"))
    _ok(_pivot(root / "other", "repro"))
    stages = (root / ".pivot" if state_dir is None else state_dir) / "stages"
    assert (stages / "b" / "tree.lock").is_file()
    return root


def test_the_refusal_names_a_lock_outside_the_project_without_a_warning(
    tmp_path: pathlib.Path,
) -> None:
    """`core.state_dir` is an absolute directory outside the project, so b/tree's lock lies
    there. `track`'s refusal names that lock by its absolute path, and nothing else is printed
    about it (no "outside base" warning from turning it into a project-relative path)."""
    state = tmp_path / "state"
    root = _deleted_stage_left_a_sidecar(tmp_path, state)

    refused = _pivot(root, "track", "data/b/tree/value.txt")

    out = " ".join((refused.stdout + refused.stderr).split())
    assert refused.returncode != 0, out
    assert f"'{state / 'stages' / 'b' / 'tree.lock'}'" in out, out
    assert "outside base" not in out, out


def test_a_file_under_a_symlinked_parent_of_a_deleted_stages_output_stays_trackable(
    tmp_path: pathlib.Path,
) -> None:
    """b/tree's lock records `data/b/tree`; `data/b` is moved to `data/b_real` and replaced by a
    symlink to it. The walk does not enter `data/b`, so `data/b_real/tree/value.txt.pvt` is a
    tracked file to every command; `track --force` re-tracks it (a plain `track` stops earlier,
    at "already tracked") rather than refusing it as the deleted stage's, and the file stays
    listed and restorable."""
    root = _deleted_stage_left_a_sidecar(tmp_path)
    data = root / "data"
    shutil.move(data / "b", data / "b_real")
    (data / "b").symlink_to("b_real")
    assert _tracked(root) == ["data/b_real/tree/value.txt"]

    tracked = _pivot(root, "track", "--force", "data/b_real/tree/value.txt")

    out = " ".join((tracked.stdout + tracked.stderr).split())
    assert tracked.returncode == 0, out
    assert "no longer exists in any pipeline" not in out, out
    assert _tracked(root) == ["data/b_real/tree/value.txt"], out
    (data / "b_real" / "tree" / "value.txt").unlink()
    _ok(_pivot(root, "checkout", "--only-missing"))
    assert (data / "b_real" / "tree" / "value.txt").read_text() == "1"


def test_the_remedy_track_names_for_a_file_a_deleted_stage_left_works(
    tmp_path: pathlib.Path,
) -> None:
    """`track` refuses `data/b/tree/value.txt`, naming the lock and `pivot track --force` with
    the path. After the lock is removed, that command exits 0 and the file is tracked: listed by
    `status`, and restored by `checkout --only-missing` once deleted."""
    root = _deleted_stage_left_a_sidecar(tmp_path)
    path = "data/b/tree/value.txt"

    refused = _pivot(root, "track", path)

    out = " ".join((refused.stdout + refused.stderr).split())
    assert refused.returncode != 0, out
    assert "'.pivot/stages/b/tree.lock'" in out, out
    assert f"pivot track --force {path}" in out, out
    (root / ".pivot" / "stages" / "b" / "tree.lock").unlink()
    _ok(_pivot(root, "track", "--force", path))
    assert _tracked(root) == [path]
    (root / path).unlink()
    _ok(_pivot(root, "checkout", "--only-missing"))
    assert (root / path).read_text() == "1"


def test_commands_run_beside_a_foreign_lock_holding_a_date_yaml_cannot_build(
    tmp_path: pathlib.Path,
) -> None:
    """Pipeline `a` at the root, run once, with one tracked file in data/raw, since deleted.
    `.pivot/stages/x.lock` holds `when: 2024-13-01`, which PyYAML's safe loader turns into a
    `ValueError`. `status`, `checkout --only-missing`, `repro` and `verify` exit 0, as on main,
    and the tracked file is restored."""
    root = tmp_path / "project"
    _init(root)
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "kept.txt").write_text("kept")
    _ok(_pivot(root, "track", "data/raw/kept.txt"))
    _ok(_pivot(root, "repro"))
    (raw / "kept.txt").unlink()
    (root / ".pivot" / "stages" / "x.lock").write_text("when: 2024-13-01\n")

    results = {
        " ".join(args): _pivot(root, *args)
        for args in (("status",), ("checkout", "--only-missing"), ("repro",), ("verify",))
    }

    for command, proc in results.items():
        assert proc.returncode == 0, f"{command}: {proc.stdout}{proc.stderr}"
    assert (raw / "kept.txt").read_text() == "kept"


def test_track_of_a_file_a_deleted_stage_left_as_a_cache_symlink_is_tracked_or_refused(
    tmp_path: pathlib.Path,
) -> None:
    """As `test_track_refuses_a_file_a_deleted_stage_left`, with the member `value.txt.pvt` a
    symlink to its cache object (pivot's own link when a hard link fails). `track` must refuse,
    or the file must then be tracked: listed by `status`, and restored by
    `checkout --only-missing` once deleted."""
    root = _deleted_stage_left_a_sidecar(tmp_path)
    member = root / "data" / "b" / "tree" / "value.txt.pvt"
    digest = cache.hash_file(member)[0]
    cached = root / ".pivot" / "cache" / "files" / digest[:2] / digest[2:]
    assert cached.is_file()
    member.unlink()
    member.symlink_to(cached)
    assert member.is_symlink()

    tracked = _pivot(root, "track", "data/b/tree/value.txt")

    if tracked.returncode != 0:
        return  # refused: the user is told, and nothing claims to track the file
    out = tracked.stdout + tracked.stderr
    assert _tracked(root) == ["data/b/tree/value.txt"], out
    (root / "data" / "b" / "tree" / "value.txt").unlink()
    _ok(_pivot(root, "checkout", "--only-missing"))
    assert (root / "data" / "b" / "tree" / "value.txt").read_text() == "1"
