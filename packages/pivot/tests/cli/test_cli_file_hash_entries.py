"""Every command that leaves a stage output on disk records each file's hash in the StateDB.

`pivot status --explain` and `status.get_pipeline_explanations` open the StateDB read-only and
look each file up by its (mtime_ns, size, device, inode). A file with no matching entry is read
again on every call (Dispatch BET-237), so each command that writes or checks an output records
it.
The tests drive the installed `pivot` binary, as a user does.
"""

from __future__ import annotations

import errno
import os
import pathlib
import subprocess
import sys
import uuid
from typing import TYPE_CHECKING

import boto3
import lmdb
import pytest
import xxhash

from pivot import cli, project, types
from pivot.executor import core as executor_core
from pivot.storage import cache, listing, state

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import click.testing

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


def _helper_unrecorded(root: pathlib.Path, state_dir: pathlib.Path | None = None) -> list[str]:
    """Files of the output with no entry in ``state_dir``'s StateDB (``.pivot``'s when None)
    giving their content's hash at their stat."""
    output = root / _OUTPUT
    files = sorted(path for path in output.rglob("*") if path.is_file())
    assert len(files) == 12, f"The output holds every file: {files}"
    with state.StateDB(state_dir or root / ".pivot", readonly=True) as state_db:
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


def _helper_commit_a_tree_then_delete_output(root: pathlib.Path) -> None:
    """Rerun in symlink mode, which commits the output as a marked whole-directory cache tree,
    then delete it, so a symlink checkout takes the tree fast path, not a file-by-file restore."""
    _ok(_pivot(root, "config", "set", "cache.checkout_mode", "symlink"))
    _ok(_pivot(root, "repro", "--force"))
    cache.remove_output(root / _OUTPUT)


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
            _helper_commit_a_tree_then_delete_output,
            ("checkout", "--checkout-mode", "symlink"),
            id="checkout-symlink-tree",
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
    next explanation reads none of them. The run's entries are dropped first: a hardlink or
    symlink restore leaves files the run's entries still match, so only this command's own
    recording can make the check pass."""
    root = _helper_project(tmp_path / "project")
    delete(root)
    cache.remove_output(root / ".pivot" / "state.lmdb")

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
    """With ``core.state_dir`` set, a stage keeps its state in that directory, and explanations
    look its outputs up there, so that is where checkout records them, not in ``.pivot``."""
    root = _helper_project(tmp_path / "project")
    _ok(_pivot(root, "config", "set", "core.state_dir", ".state"))
    _helper_delete_output(root)

    _ok(_pivot(root, "checkout", "--checkout-mode", "copy"))

    assert _helper_unrecorded(root, root / ".state") == [], (
        "Every restored file's hash is in the stage StateDB"
    )


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


def test_track_of_a_directory_records_every_file_it_read(tmp_path: pathlib.Path) -> None:
    """`pivot track` of a directory records the hash of each file it read, so the next status
    reads none of them."""
    root = _helper_input_project(tmp_path / "project")
    tracked = root / "data" / "tracked"
    (tracked / "sub").mkdir(parents=True)
    for index in range(3):
        (tracked / f"t{index}.txt").write_text(f"tracked {index}\n")
    (tracked / "sub" / "deep.txt").write_text("deep\n")

    _ok(_pivot(root, "track", "data/tracked"))

    files = sorted(path for path in tracked.rglob("*") if path.is_file())
    assert len(files) == 4, f"The tracked directory holds every file: {files}"
    assert [path for path in files if not _helper_recorded(root, path)] == [], (
        "track records the hash of every file of the directory it read"
    )


def test_known_file_hashes_skips_a_member_it_cannot_stat(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed directory output whose one member a forced rerun left as a checkout link that
    ``lstat`` reads but ``stat`` cannot (EIO, ESTALE, EACCES on the link's ``readlink`` or its
    target) gets no known-hash entry for that member, so the lookup never fails the command.
    The other members keep their entries: the result is exactly the list the lookup returns when
    that member is not in the listing at all. At 87c648fc the unguarded stat raised the member's
    ``OSError`` straight out of ``known_file_hashes``, failing the stage."""
    out = tmp_path / "out"
    (out / "sub").mkdir(parents=True)
    (out / "good.txt").write_bytes(b"good")
    bad = out / "sub" / "bad.txt"
    bad.write_bytes(b"bad")
    good_entry = types.DirManifestEntry(
        relpath="good.txt", hash=xxhash.xxh64(b"good").hexdigest(), size=4, isexec=False
    )
    bad_entry = types.DirManifestEntry(
        relpath="sub/bad.txt", hash=xxhash.xxh64(b"bad").hexdigest(), size=3, isexec=False
    )
    record = types.DirHash(hash="0" * 16, size=7, nfiles=2)
    full = listing.DirListing(files=[good_entry, bad_entry], empty_dirs=[])
    without_bad = listing.DirListing(files=[good_entry], empty_dirs=[])

    real_stat = os.stat

    def stat_eio_on_bad(path: int | str | os.PathLike[str], *args: object, **kwargs: object):
        if not isinstance(path, int) and os.fspath(path) == str(bad):
            raise OSError(errno.EIO, "Input/output error")
        return real_stat(path, *args, **kwargs)  # type: ignore[arg-type] - pass-through

    monkeypatch.setattr(os, "stat", stat_eio_on_bad)

    skipped = cache.known_file_hashes(out, record, full)
    expected = cache.known_file_hashes(out, record, without_bad)

    assert [path for path, _, _ in expected] == [out / "good.txt"]
    assert [(str(path), file_hash) for path, _, file_hash in skipped] == [
        (str(path), file_hash) for path, _, file_hash in expected
    ]
    assert str(bad) not in {str(path) for path, _, _ in skipped}


@pytest.mark.parametrize(
    "command", [("checkout", "--checkout-mode", "symlink"), ("repro",)], ids=["checkout", "repro"]
)
def test_a_corrupt_lock_listing_uses_the_canonical_cache_copy(
    tmp_path: pathlib.Path, command: tuple[str, ...]
) -> None:
    """A malformed committed listing cannot make checkout or repro fail when its cache copy is
    available. The cache copy is the canonical listing that the lock's directory hash names."""
    root = _helper_project(tmp_path / "project")
    listing_path = next(
        path
        for path in (root / ".pivot" / "stages").rglob("*.listing")
        if b"f00.txt" in path.read_bytes()
    )
    listing_path.write_bytes(b"this is not a pivot listing\n")
    cache.remove_output(root / _OUTPUT)

    _ok(_pivot(root, *command))

    assert len([path for path in (root / _OUTPUT).rglob("*") if path.is_file()]) == 12


def test_checkout_restores_with_a_damaged_state_database(tmp_path: pathlib.Path) -> None:
    """Round 7-1: a damaged `state.lmdb` raises `lmdb.Error` on a writable filesystem. Status
    and verify may report that damage as on main, but main's checkout never opened the database,
    so checkout must restore without the hash cache rather than fail after it has a cache object
    for every file."""
    root = tmp_path / "project"
    files = _helper_tracked_files(root, 3)
    for path in files[1:]:
        path.unlink()
    (root / ".pivot" / "state.lmdb" / "data.mdb").write_bytes(b"\x00not an lmdb file" * 8192)

    proc = _pivot(root, "checkout", "--checkout-mode", "copy")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert [path.read_text() for path in files] == [f"tracked {i}\n" for i in range(3)]


def test_checkout_restores_with_a_state_database_it_cannot_write(tmp_path: pathlib.Path) -> None:
    """`.pivot/state.lmdb` readable but not writable, on a writable filesystem: LMDB will not open
    it even read-only, and `state.open_readonly` raises (the mount is writable). Main's checkout
    never opened the database, so checkout restores without the hash cache."""
    root = tmp_path / "project"
    files = _helper_tracked_files(root, 3)
    for path in files[1:]:
        path.unlink()
    lmdb_dir = root / ".pivot" / "state.lmdb"
    modes = {path: path.stat().st_mode for path in [lmdb_dir, *lmdb_dir.iterdir()]}
    for path, mode in modes.items():
        path.chmod(mode & ~0o222)
    try:
        proc = _pivot(root, "checkout", "--checkout-mode", "copy")
    finally:
        for path, mode in modes.items():
            path.chmod(mode)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert [path.read_text() for path in files] == [f"tracked {i}\n" for i in range(3)]


def _helper_tracked_files(root: pathlib.Path, count: int) -> list[pathlib.Path]:
    """A project with ``count`` tracked single files, committed, and no pipeline."""
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init"))
    files = [root / f"t{index}.txt" for index in range(count)]
    for index, path in enumerate(files):
        path.write_text(f"tracked {index}\n")
    _ok(_pivot(root, "track", *(path.name for path in files)))
    _helper_commit(root, "track")
    return files


def _helper_invoke(
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    root: pathlib.Path,
    *args: str,
) -> click.testing.Result:
    """Run ``pivot <args>`` in this process in ``root``, so a test can patch the StateDB."""
    monkeypatch.chdir(root)
    monkeypatch.setattr(project, "_project_root_cache", None)
    return runner.invoke(cli.cli, list(args))


def _helper_restore_args(files: list[pathlib.Path], command: str) -> tuple[str, ...]:
    if command == "checkout":
        return ("checkout", "--checkout-mode", "copy")
    return ("get", "--rev", "HEAD", "--checkout-mode", "copy", *(path.name for path in files))


def _helper_state_on_a_read_only_mount(monkeypatch: pytest.MonkeyPatch, root: pathlib.Path) -> None:
    """Make ``root``'s `.pivot/state.lmdb` behave as on a read-only mount, in this process: LMDB
    refuses to open it with a plain `lmdb.Error`, as it does for an existing database there, and
    `statvfs` of `state.lmdb` itself (not its parent) reports `ST_RDONLY`."""
    lmdb_path = root / ".pivot" / "state.lmdb"
    real_statvfs = os.statvfs

    class _StatVfs:
        f_flag: int

        def __init__(self, flag: int) -> None:
            self.f_flag = flag

    def lmdb_error(*_: object, **__: object) -> object:
        raise lmdb.Error("mdb_env_open: Read-only file system")

    def statvfs(path: str | os.PathLike[str]) -> object:
        if pathlib.Path(path) == lmdb_path:
            return _StatVfs(os.ST_RDONLY)
        return real_statvfs(path)

    monkeypatch.setattr(state, "_acquire_shared_env", lmdb_error)
    monkeypatch.setattr(os, "statvfs", statvfs)


def test_checkout_restores_when_its_state_database_is_on_a_read_only_filesystem(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, runner: click.testing.CliRunner
) -> None:
    """Round 7-1: LMDB reports an existing database on a read-only mount as plain `lmdb.Error`.
    `state.open_readonly` checks the mount of `state.lmdb` itself, not its parent, and yields no
    cache, so checkout compares and restores files as main does."""
    root = tmp_path / "project"
    files = _helper_tracked_files(root, 3)
    for path in files[1:]:
        path.unlink()
    _helper_state_on_a_read_only_mount(monkeypatch, root)

    result = _helper_invoke(runner, monkeypatch, root, *_helper_restore_args(files, "checkout"))

    assert result.exit_code == 0, result.output
    assert [path.read_text() for path in files] == [f"tracked {i}\n" for i in range(3)]


def test_status_reports_tracked_files_with_its_state_database_on_a_read_only_filesystem(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, runner: click.testing.CliRunner
) -> None:
    """The tracked-file status opens the hash cache through `state.open_readonly`, so a state
    database on a read-only mount leaves it reading the files instead of failing."""
    root = tmp_path / "project"
    _helper_tracked_files(root, 3)
    _helper_state_on_a_read_only_mount(monkeypatch, root)

    result = _helper_invoke(runner, monkeypatch, root, "status")

    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("command", [("status", "--explain"), ("verify",)])
def test_explanations_run_with_their_state_database_on_a_read_only_filesystem(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: click.testing.CliRunner,
    command: tuple[str, ...],
) -> None:
    """Review item 1: the explanation opens the hash cache through `state.open_readonly`, so a
    state database on a read-only mount leaves every stage explanation reading the files."""
    root = _helper_project(tmp_path / "project")
    _helper_state_on_a_read_only_mount(monkeypatch, root)

    result = _helper_invoke(runner, monkeypatch, root, *command)

    assert "Failed to get explanation" not in result.output
    assert result.exit_code == 0, result.output


def test_the_tracked_file_check_runs_with_its_state_database_on_a_read_only_filesystem(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repro's pre-flight (`executor.core.verify_tracked_files`) opens the hash cache through
    `state.open_readonly`, so a state database on a read-only mount leaves it reading the tracked
    files instead of failing."""
    root = tmp_path / "project"
    _helper_tracked_files(root, 3)
    monkeypatch.chdir(root)
    monkeypatch.setattr(project, "_project_root_cache", None)
    _helper_state_on_a_read_only_mount(monkeypatch, root)

    executor_core.verify_tracked_files(root)


@pytest.mark.parametrize("command", ["checkout", "get"])
def test_restoring_many_targets_saves_their_hashes_once_per_state_directory(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: click.testing.CliRunner,
    command: str,
) -> None:
    """Round 5, B3: checkout and get committed one StateDB write transaction and fdatasync per
    target (500 targets: 500 writes where main did none; +52% on a 2,000-file checkout). They
    collect every target's entries and save each state directory's once."""
    root = tmp_path / "project"
    files = _helper_tracked_files(root, 5)
    for path in files:
        path.unlink()
    saves = list[int]()
    real_save_file_hash_entries = state.StateDB.save_file_hash_entries

    def counting_save_file_hash_entries(
        self: state.StateDB,
        entries: Iterable[types.FileHashEntry],
        *,
        timeout: float | None = None,
    ) -> None:
        entry_list = list(entries)
        saves.append(len(entry_list))
        real_save_file_hash_entries(self, entry_list, timeout=timeout)

    monkeypatch.setattr(state.StateDB, "save_file_hash_entries", counting_save_file_hash_entries)

    result = _helper_invoke(runner, monkeypatch, root, *_helper_restore_args(files, command))

    assert result.exit_code == 0, result.output
    assert saves == [len(files)], "one write for the one state directory, holding every file"
    assert all(_helper_recorded(root, path) for path in files)


@pytest.mark.parametrize("command", ["checkout", "get", "track", "status"])
def test_a_failed_hash_save_leaves_the_command_result_as_without_the_cache(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: click.testing.CliRunner,
    command: str,
) -> None:
    """The hash cache is best-effort, so a StateDB write that fails (here a full map) is logged
    and dropped; the command's exit code and the workspace are what they would be with no cache.
    `status` (round 6, B4) saves the hashes of the tracked files it read the same way: their
    entries are dropped first, so it has some to save."""
    root = tmp_path / "project"
    files = _helper_tracked_files(root, 3)
    if command == "track":
        extra = root / "extra.txt"
        extra.write_text("extra\n")
        args: tuple[str, ...] = ("track", extra.name)
    elif command == "status":
        cache.remove_output(root / ".pivot" / "state.lmdb")
        args = ("status",)
    else:
        for path in files:
            path.unlink()
        args = _helper_restore_args(files, command)

    def full_save_many(*_: object, **__: object) -> None:
        raise state.DatabaseFullError("state database is full")

    monkeypatch.setattr(state.StateDB, "save_many", full_save_many)

    result = _helper_invoke(runner, monkeypatch, root, *args)

    assert result.exit_code == 0, result.output
    if command == "track":
        assert (root / "extra.txt.pvt").is_file()
    else:
        assert [path.read_text() for path in files] == [f"tracked {i}\n" for i in range(3)]
