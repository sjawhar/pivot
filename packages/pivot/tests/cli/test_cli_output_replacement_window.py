"""A user's tracked file that lands in a stage output while the installed `pivot` binary replaces
that output, and a stage write or restore killed while its old directory output is renamed aside.

A sitecustomize hook on PYTHONPATH runs in the pivot process and in every worker it starts:

- ``incremental``: right after the first ownership check of an incremental output's restore
  (``worker._prepare_incremental_inputs``), the user's tracked ``data/raw/<name>`` is renamed over
  the output, as ``mv`` would, with its sidecar beside it;
- ``commit``: right after a stage commit's ownership check (``WorkspaceStore.commit``), the
  user's tracked ``late.txt`` and its sidecar are copied into the directory output;
- ``kill``: the process is killed the moment it renames the output named
  ``PIVOT_TEST_LATE_OUTPUT`` to a backup name.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from pivot.storage import cache

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

_HOOK = """\
import os

_MODE = os.environ.get("PIVOT_TEST_LATE_MODE")
if _MODE:
    import pathlib
    import shutil
    import sys

    _ROOT = pathlib.Path(os.environ["PIVOT_TEST_LATE_ROOT"])
    _MARKER = _ROOT / ".late-landed"

    def _called_from(function, module_file):
        frame = sys._getframe(2)
        while frame is not None:
            code = frame.f_code
            if code.co_name == function and code.co_filename.endswith(module_file):
                return True
            frame = frame.f_back
        return False

    if _MODE == "kill":
        _rename = pathlib.Path.rename

        def _rename_then_die(self, target):
            moved = _rename(self, target)
            if (
                self.name == os.environ["PIVOT_TEST_LATE_OUTPUT"]
                and pathlib.Path(target).name.startswith(".pivot_backup_")
                and not _MARKER.exists()
            ):
                _MARKER.write_text(str(target))
                os._exit(73)
            return moved

        pathlib.Path.rename = _rename_then_die
    else:
        from pivot.storage import ownership

        _refuse = ownership.refuse_output_replacements

        def _refuse_then_land(*args, **kwargs):
            _refuse(*args, **kwargs)
            if _MARKER.exists():
                return
            output = pathlib.Path(args[1][0]["path"])
            raw = _ROOT / "data" / "raw"
            if _MODE == "incremental" and _called_from("_prepare_incremental_inputs", "worker.py"):
                staged = output.with_name(".late-staged")
                shutil.copy2(raw / output.name, staged)
                os.replace(staged, output)
                shutil.copy2(raw / (output.name + ".pvt"), output.with_name(output.name + ".pvt"))
            elif (
                _MODE == "commit"
                and _called_from("commit", "store.py")
                and output.is_dir()
                and not output.is_symlink()
            ):
                shutil.copy2(raw / "late.txt", output / "late.txt")
                shutil.copy2(raw / "late.txt.pvt", output / "late.txt.pvt")
            else:
                return
            _MARKER.write_text(str(output))

        ownership.refuse_output_replacements = _refuse_then_land
"""

_INCREMENTAL_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, incremental, stage


@stage
def grow(row: str, previous: str) -> Annotated[str, incremental]:
    return previous + row


with Pipeline("inc") as pipeline:
    grow(pipeline.input("row.txt"), previous=pipeline.previous())
"""

_TREE_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {f"f{index}.txt": f"tree {index}\\n" for index in range(5)}


with Pipeline("c") as pipeline:
    tree()
"""

_KILL_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def note() -> str:
    return "note\\n"


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {f"f{index}.txt": f"tree {index}\\n" for index in range(5)}


with Pipeline("k") as pipeline:
    note()
    tree()
"""


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


def _helper_project(tmp_path: pathlib.Path, pipeline: str) -> pathlib.Path:
    root = tmp_path / "proj"
    (root / "data" / "raw").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init"))
    (root / "pipeline.py").write_text(pipeline)
    return root


def _helper_hooked(
    tmp_path: pathlib.Path, root: pathlib.Path, mode: str, **extra: str
) -> dict[str, str]:
    """An environment in which the hook acts in ``mode`` on the project at ``root``."""
    hook = tmp_path / "hook"
    hook.mkdir(exist_ok=True)
    (hook / "sitecustomize.py").write_text(_HOOK)
    env = _env(PIVOT_TEST_LATE_MODE=mode, PIVOT_TEST_LATE_ROOT=str(root), **extra)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(hook), env.get("PYTHONPATH")]))
    return env


def _helper_refused(proc: subprocess.CompletedProcess[str], sidecar: str) -> str:
    """The command failed, refusing to replace an output, naming ``sidecar``."""
    out = proc.stdout + proc.stderr
    assert "Error in sitecustomize" not in out, out
    assert proc.returncode == 1, out
    # Without whitespace, so a line wrapped by the console still matches.
    flat = "".join(out.split())
    assert "Refusingtoreplaceastageoutputthatholdsuser-ownedfiles" in flat, out
    assert sidecar in flat, out
    return out


def _helper_backups(directory: pathlib.Path) -> list[str]:
    return sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.name.startswith(cache._BACKUP_TEMP_PREFIX)
    )


def test_repro_keeps_a_tracked_file_moved_over_an_incremental_output_before_its_restore(
    tmp_path: pathlib.Path,
) -> None:
    """An incremental stage restores its previous output before it runs. A user's tracked file
    moved over that output, with its sidecar beside it, right after the restore's first check:
    the restore checks the entry it replaces again, refuses naming the sidecar, and keeps the
    user's bytes. Pivot used to remove the output between those checks, deleting them."""
    root = _helper_project(tmp_path, _INCREMENTAL_PIPELINE)
    (root / "data" / "raw" / "row.txt").write_text("one\n")
    # The bytes a user who tracked a file named like the output has, and its sidecar
    (root / "data" / "raw" / "grow.txt").write_text("user-owned incremental bytes\n")
    _ok(_pivot(root, "repro"))
    _ok(_pivot(root, "track", "data/raw/grow.txt"))
    (root / "data" / "raw" / "row.txt").write_text("two\n")
    output = root / "data" / "inc" / "grow.txt"

    proc = _pivot(root, "repro", env=_helper_hooked(tmp_path, root, "incremental"))

    assert (root / ".late-landed").read_text() == str(output), proc.stdout + proc.stderr
    _helper_refused(proc, "data/inc/grow.txt.pvt")
    assert output.is_file(), "the user's file was deleted"
    assert output.read_text() == "user-owned incremental bytes\n"
    assert output.with_name("grow.txt.pvt").is_file()
    assert _helper_backups(output.parent) == []


def test_symlink_commit_keeps_a_tracked_file_copied_into_the_output_after_its_check(
    tmp_path: pathlib.Path,
) -> None:
    """With symlink checkout first, a stage's commit replaces its directory output with a link to
    the cache tree. A user's tracked file and sidecar copied into the output right after the
    commit's check: the commit checks the output again, renamed aside, refuses naming the sidecar,
    and puts the output back with both files. Pivot used to delete them, and exit 0."""
    root = _helper_project(tmp_path, _TREE_PIPELINE)
    (root / "data" / "raw" / "late.txt").write_text("late user bytes\n")
    _ok(_pivot(root, "config", "set", "cache.checkout_mode", "symlink,copy"))
    _ok(_pivot(root, "track", "data/raw/late.txt"))
    output = root / "data" / "c" / "tree"

    proc = _pivot(root, "repro", env=_helper_hooked(tmp_path, root, "commit"))

    assert (root / ".late-landed").read_text() == str(output), proc.stdout + proc.stderr
    _helper_refused(proc, "data/c/tree/late.txt.pvt")
    assert output.is_dir()
    assert not output.is_symlink()
    assert (output / "late.txt").read_text() == "late user bytes\n", "the user's file was deleted"
    assert (output / "late.txt.pvt").is_file()
    assert sorted(path.name for path in output.iterdir()) == [
        *(f"f{index}.txt" for index in range(5)),
        "late.txt",
        "late.txt.pvt",
    ]
    assert _helper_backups(output.parent) == []


@pytest.mark.parametrize(
    "then",
    [
        pytest.param(("repro", "k/tree"), id="then-restore"),
        pytest.param(("repro", "--force", "k/tree"), id="then-write"),
    ],
)
def test_the_backup_a_killed_stage_write_leaves_goes_with_the_next_replacement(
    tmp_path: pathlib.Path, then: tuple[str, ...]
) -> None:
    """A stage write killed right after it renamed its old directory output aside leaves that
    output beside the path as a backup. The next command that writes or restores the output (a
    restore from the cache, a stage write) checks the backup under the path's restore lock and
    removes it. A rerun used to keep it until a restore of the output."""
    root = _helper_project(tmp_path, _KILL_PIPELINE)
    _ok(_pivot(root, "repro"))
    output = root / "data" / "k" / "tree"

    killed = _pivot(
        root,
        "repro",
        "--force",
        "k/tree",
        env=_helper_hooked(tmp_path, root, "kill", PIVOT_TEST_LATE_OUTPUT="tree"),
    )

    out = killed.stdout + killed.stderr
    assert "Error in sitecustomize" not in out, out
    assert killed.returncode != 0, out
    assert (root / ".late-landed").exists(), out
    assert not output.exists()
    assert len(_helper_backups(output.parent)) == 1

    _ok(_pivot(root, *then))

    assert _helper_backups(output.parent) == []
    assert sorted(path.name for path in output.iterdir()) == [f"f{index}.txt" for index in range(5)]


def test_a_file_outputs_rerun_renames_nothing_aside(tmp_path: pathlib.Path) -> None:
    """A stage's file output is checked in place and unlinked, never renamed aside (on overlayfs a
    rename copies a lower-layer file up whole first): the hook that kills the process at a rename
    to a backup name never fires, and the rerun succeeds."""
    root = _helper_project(tmp_path, _KILL_PIPELINE)
    _ok(_pivot(root, "repro"))
    output = root / "data" / "k" / "note.txt"

    _ok(
        _pivot(
            root,
            "repro",
            "--force",
            "k/note",
            env=_helper_hooked(tmp_path, root, "kill", PIVOT_TEST_LATE_OUTPUT="note.txt"),
        )
    )

    assert not (root / ".late-landed").exists()
    assert output.read_text() == "note\n"
    assert _helper_backups(output.parent) == []


def test_forced_checkout_keeps_a_killed_restores_backup_that_holds_a_user_file(
    tmp_path: pathlib.Path,
) -> None:
    """A forced checkout killed right after it renamed a directory output aside leaves it as a
    backup, and a user's tracked file is then copied into that backup. The next forced checkout
    of the output checks the backup, keeps it with a warning naming the file, and restores the
    output, as every other restore does. It used to fail, naming the file, and restore nothing."""
    root = _helper_project(tmp_path, _KILL_PIPELINE)
    (root / "data" / "raw" / "late.txt").write_text("late user bytes\n")
    _ok(_pivot(root, "repro"))
    _ok(_pivot(root, "track", "data/raw/late.txt"))
    output = root / "data" / "k" / "tree"

    killed = _pivot(
        root,
        "checkout",
        "--force",
        "data/k/tree",
        env=_helper_hooked(tmp_path, root, "kill", PIVOT_TEST_LATE_OUTPUT="tree"),
    )
    out = killed.stdout + killed.stderr
    assert "Error in sitecustomize" not in out, out
    assert killed.returncode != 0, out
    assert not output.exists()
    (backup_name,) = _helper_backups(output.parent)
    backup = output.parent / backup_name
    shutil.copy2(root / "data" / "raw" / "late.txt", backup / "late.txt")
    shutil.copy2(root / "data" / "raw" / "late.txt.pvt", backup / "late.txt.pvt")

    flat = "".join(_ok(_pivot(root, "checkout", "--force", "data/k/tree")).split())

    assert "Keepinginterruptedrestorebackup" in flat, flat
    assert "data/k/tree/late.txt.pvt" in flat, flat
    assert sorted(path.name for path in output.iterdir()) == [f"f{index}.txt" for index in range(5)]
    assert _helper_backups(output.parent) == [backup_name]
    assert (backup / "late.txt").read_text() == "late user bytes\n"
    assert (backup / "late.txt.pvt").is_file()
