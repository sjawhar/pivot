"""Every command that leaves a stage output on disk records each file's hash in the StateDB.

`pivot status --explain` and `status.get_pipeline_explanations` open the StateDB read-only and
look each file up by its (mtime_ns, size, device, inode). A file with no matching entry is read
again on every call (Dispatch BET-237), so each command that writes or checks an output records
it.
The tests drive the installed `pivot` binary, as a user does.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import uuid
from typing import TYPE_CHECKING

import boto3
import pytest
import xxhash

from pivot.storage import cache, state

if TYPE_CHECKING:
    from collections.abc import Callable

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")
_OUTPUT = pathlib.Path("data", "syn", "make_tree.dat")
_PIPELINE = """\
import pathlib
import tempfile

from pivot import Pipeline, stage


@stage
def make_tree() -> pathlib.Path:
    out = pathlib.Path(tempfile.mkdtemp()) / "tree"
    for i in range(12):
        (out / f"d{i % 3}").mkdir(parents=True, exist_ok=True)
        (out / f"d{i % 3}" / f"f{i:02d}.txt").write_text(f"file {i}\\n")
    return out


@stage
def count(tree: pathlib.Path) -> str:
    return str(sum(1 for _ in tree.rglob("*.txt")))


with Pipeline("syn") as pipeline:
    count(make_tree())
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


def _helper_commit(root: pathlib.Path, message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", message],
        cwd=root,
        check=True,
    )


def _helper_project(root: pathlib.Path, env: dict[str, str] | None = None) -> pathlib.Path:
    """A project whose two stages ran once and whose locks are committed."""
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init", env=env))
    (root / "pipeline.py").write_text(_PIPELINE)
    (root / ".gitignore").write_text("/data/\n__pycache__/\n")
    _ok(_pivot(root, "repro", env=env))
    _helper_commit(root, "run")
    return root


def _helper_unrecorded(root: pathlib.Path) -> list[str]:
    """Files of the output with no StateDB entry giving their content's hash at their stat."""
    output = root / _OUTPUT
    files = sorted(path for path in output.rglob("*") if path.is_file())
    assert len(files) == 12, f"The output holds every file: {files}"
    with state.StateDB(root / ".pivot", readonly=True) as state_db:
        found = state_db.get_many([(path, path.stat()) for path in files])
    return [
        str(path.relative_to(output))
        for path in files
        if found[path] != xxhash.xxh64(path.read_bytes()).hexdigest()
    ]


def _helper_delete_output(root: pathlib.Path) -> None:
    cache.remove_output(root / _OUTPUT)


def _helper_delete_one_subdirectorys_files(root: pathlib.Path) -> None:
    for path in sorted((root / _OUTPUT / "d0").iterdir()):
        path.unlink()


@pytest.mark.parametrize(
    ("delete", "command"),
    [
        pytest.param(_helper_delete_output, ("checkout",), id="checkout-default"),
        pytest.param(
            _helper_delete_output, ("checkout", "--checkout-mode", "copy"), id="checkout-copy"
        ),
        pytest.param(
            _helper_delete_output,
            ("checkout", "--checkout-mode", "symlink"),
            id="checkout-symlink",
        ),
        pytest.param(
            _helper_delete_one_subdirectorys_files,
            ("checkout", "--only-missing", "--checkout-mode", "copy"),
            id="checkout-only-missing-copy",
        ),
        pytest.param(
            _helper_delete_output,
            ("get", "--rev", "HEAD", "--checkout-mode", "copy", "syn/make_tree"),
            id="get-copy",
        ),
    ],
)
def test_restoring_an_output_records_every_file_hash(
    tmp_path: pathlib.Path,
    delete: Callable[[pathlib.Path], None],
    command: tuple[str, ...],
) -> None:
    """After a restore, every file of the output has a StateDB entry for its content, so the
    next explanation reads none of them."""
    root = _helper_project(tmp_path / "project")
    delete(root)

    _ok(_pivot(root, *command))

    assert _helper_unrecorded(root) == [], "Every restored file's hash is recorded"
    assert "0 will run" in _ok(_pivot(root, "status", "--explain"))


def test_repro_records_the_hashes_its_skip_check_reads(tmp_path: pathlib.Path) -> None:
    """A repro that finds both stages current, with no file hash recorded, records every file
    it read to decide, so the next explanation reads none of them."""
    root = _helper_project(tmp_path / "project")
    cache.remove_output(root / ".pivot" / "state.lmdb")

    out = _ok(_pivot(root, "repro"))

    assert "0 ran" in out, out
    assert _helper_unrecorded(root) == [], "Every file the skip check read is recorded"


def _helper_s3_env(moto_services: dict[str, str]) -> dict[str, str]:
    return _env(
        AWS_ENDPOINT_URL=moto_services["s3"],
        AWS_ACCESS_KEY_ID="x",
        AWS_SECRET_ACCESS_KEY="x",
        AWS_DEFAULT_REGION="us-east-1",
    )


def _helper_push_to_new_bucket(
    root: pathlib.Path, moto_services: dict[str, str], env: dict[str, str]
) -> None:
    """Point the project's default remote at a new moto bucket and push its cache there."""
    bucket = f"hash-entries-{uuid.uuid4().hex[:8]}"
    s3 = boto3.client(
        "s3",
        endpoint_url=moto_services["s3"],
        region_name="us-east-1",
        aws_access_key_id="x",
        aws_secret_access_key="x",
    )
    s3.create_bucket(Bucket=bucket)  # pyright: ignore[reportAttributeAccessIssue] - boto3 ships no stubs
    _ok(_pivot(root, "config", "set", "remotes.origin", f"s3://{bucket}/cache", env=env))
    _ok(_pivot(root, "config", "set", "default_remote", "origin", env=env))
    _ok(_pivot(root, "push", env=env))


def test_pull_into_a_fresh_clone_records_every_file_hash(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """`pivot pull` into a clone that has never run the pipeline records each file it writes."""
    env = _helper_s3_env(moto_services)
    src = _helper_project(tmp_path / "src", env)
    _helper_push_to_new_bucket(src, moto_services, env)
    _helper_commit(src, "remote")
    clone = (tmp_path / "clone").resolve()
    subprocess.run(["git", "clone", "-q", str(src), str(clone)], check=True)

    _ok(_pivot(clone, "pull", env=env))

    assert _helper_unrecorded(clone) == [], "Every pulled file's hash is recorded"
    assert "0 will run" in _ok(_pivot(clone, "status", "--explain", env=env))


def test_checkout_of_an_existing_output_records_what_it_read(tmp_path: pathlib.Path) -> None:
    """Checkout reads an output already on disk to tell whether it matches its lock, and
    records what it read, so the next explanation reads none of it."""
    root = _helper_project(tmp_path / "project")
    cache.remove_output(root / ".pivot" / "state.lmdb")

    out = _ok(_pivot(root, "checkout"))

    assert "Skipped" in out, out
    assert _helper_unrecorded(root) == [], "Every file checkout compared is recorded"


def test_checkout_records_in_the_stage_state_directory_explain_reads(
    tmp_path: pathlib.Path,
) -> None:
    """With ``core.state_dir`` set elsewhere, a stage still keeps its state in ``.pivot``,
    and explanations look its outputs up there, so that is where checkout records them."""
    root = _helper_project(tmp_path / "project")
    _ok(_pivot(root, "config", "set", "core.state_dir", ".state"))
    _helper_delete_output(root)

    _ok(_pivot(root, "checkout", "--checkout-mode", "copy"))

    assert _helper_unrecorded(root) == [], "Every restored file's hash is in the stage StateDB"


# One stage reading one raw input and writing one file: nothing else reads its output, so only
# the stage's own skip check can record it.
_INPUT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def shout(raw: str) -> str:
    return raw.upper()


with Pipeline("inp") as pipeline:
    shout(pipeline.input("words.txt", t=str))
"""
_INPUT = pathlib.Path("data", "raw", "words.txt")
_SHOUT = pathlib.Path("data", "inp", "shout.txt")


def _helper_input_project(root: pathlib.Path, env: dict[str, str] | None = None) -> pathlib.Path:
    root.mkdir(parents=True)
    root = root.resolve()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init", env=env))
    (root / "pipeline.py").write_text(_INPUT_PIPELINE)
    (root / ".gitignore").write_text("/data/\n__pycache__/\n")
    (root / _INPUT).parent.mkdir(parents=True)
    (root / _INPUT).write_text("hello\n")
    _ok(_pivot(root, "repro", env=env))
    _helper_commit(root, "run")
    return root


def _helper_recorded(root: pathlib.Path, path: pathlib.Path) -> bool:
    """Whether the project's StateDB holds ``path``'s content hash at its current stat."""
    with state.StateDB(root / ".pivot", readonly=True) as state_db:
        found = state_db.get(path, path.stat())
    return found == xxhash.xxh64(path.read_bytes()).hexdigest()


def test_coordinator_skip_check_records_the_dependency_and_output_it_read(
    tmp_path: pathlib.Path,
) -> None:
    """With no file hash recorded, repro's skip check reads the stage's input to compare it
    with the lock, then its output to check it is intact; it records both."""
    root = _helper_input_project(tmp_path / "project")
    cache.remove_output(root / ".pivot" / "state.lmdb")

    out = _ok(_pivot(root, "repro"))

    assert "0 ran" in out, out
    assert _helper_recorded(root, root / _INPUT), "The dependency read to decide is recorded"
    assert _helper_recorded(root, root / _SHOUT), "The output checked is recorded"


def test_coordinator_restore_records_the_output_it_restored(tmp_path: pathlib.Path) -> None:
    """A skip check that finds the output gone restores it from the cache under a write lock,
    and records the restored file."""
    root = _helper_input_project(tmp_path / "project")
    cache.remove_output(root / ".pivot" / "state.lmdb")
    (root / _SHOUT).unlink()

    out = _ok(_pivot(root, "repro"))

    assert "0 ran" in out, out
    assert (root / _SHOUT).read_text() == "HELLO\n"
    assert _helper_recorded(root, root / _SHOUT), "The restored output is recorded"


def test_get_from_git_records_the_file_it_wrote(tmp_path: pathlib.Path) -> None:
    """`pivot get` of a file git holds writes it from the revision and records it."""
    root = _helper_input_project(tmp_path / "project")
    notes = root / "notes.txt"
    notes.write_text("committed notes\n")
    _helper_commit(root, "notes")
    notes.unlink()

    out = _ok(_pivot(root, "get", "--rev", "HEAD", "notes.txt"))

    assert "(from git)" in out, out
    assert _helper_recorded(root, notes), "The file written from git is recorded"


def test_track_and_repro_checkout_missing_record_the_tracked_file(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot track` records the hash it read, and repro's restore of a missing tracked file
    (``--checkout-missing``) records the file it restored."""
    root = _helper_input_project(tmp_path / "project")
    tracked = root / "notes.txt"
    tracked.write_text("tracked notes\n")

    _ok(_pivot(root, "track", "notes.txt"))
    assert _helper_recorded(root, tracked), "track records the hash it read"

    tracked.unlink()
    cache.remove_output(root / ".pivot" / "state.lmdb")
    _ok(_pivot(root, "repro", "--checkout-missing"))

    assert tracked.read_text() == "tracked notes\n"
    assert _helper_recorded(root, tracked), "The restored tracked file is recorded"
