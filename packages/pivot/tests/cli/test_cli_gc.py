"""`pivot gc` removes every cache object no record names, and only those."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import threading
import time
from typing import TYPE_CHECKING, TypedDict

import pytest

from pivot import cli, project, run_history, types
from pivot.storage import cache, gc, lock, state, track

if TYPE_CHECKING:
    import pathlib

    import click.testing

_STAGE = "demo/train"


class Workspace(TypedDict):
    """A project, its cache and the directories a test writes records into."""

    root: pathlib.Path
    cache_dir: pathlib.Path
    object_root: pathlib.Path
    state_dir: pathlib.Path
    stages_dir: pathlib.Path
    scratch: pathlib.Path


def _helper_workspace(root: pathlib.Path, cache_dir: pathlib.Path) -> Workspace:
    """Create a project at ``root`` that caches into ``cache_dir``."""
    state_dir = root / ".pivot"
    stages_dir = state_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    object_root = cache_dir / gc.OBJECTS_DIR
    object_root.mkdir(parents=True, exist_ok=True)
    scratch = root.parent / f"scratch-{root.name}"
    scratch.mkdir(exist_ok=True)
    return Workspace(
        root=root,
        cache_dir=cache_dir,
        object_root=object_root,
        state_dir=state_dir,
        stages_dir=stages_dir,
        scratch=scratch,
    )


@pytest.fixture
def workspace(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    """A project whose cache lives inside it, so no register is needed."""
    root = tmp_path / "project"
    root.mkdir()
    space = _helper_workspace(root, root / ".pivot" / "cache")
    monkeypatch.chdir(root)
    monkeypatch.setattr(project, "_project_root_cache", None)
    return space


@pytest.fixture
def shared(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    """A project whose cache lives outside it, so the register decides."""
    root = tmp_path / "first"
    root.mkdir()
    space = _helper_workspace(root, tmp_path / "shared-cache")
    monkeypatch.setenv("PIVOT_CACHE_DIR", str(space["cache_dir"]))
    monkeypatch.chdir(root)
    monkeypatch.setattr(project, "_project_root_cache", None)
    return space


def _helper_publish(space: Workspace, content: str) -> str:
    """Publish one file object the way a stage does, and return its cache key."""
    source = space["scratch"] / f"source-{content}"
    source.write_text(content)
    key, _ = cache.hash_file(source)
    cache.copy_to_cache(source, cache.get_cache_path(space["object_root"], key))
    source.unlink()
    return key


def _helper_publish_tree(space: Workspace, name: str) -> types.DirHash:
    """Publish one directory object, its member blobs included, naming nothing."""
    source = space["scratch"] / name
    source.mkdir()
    (source / "a.txt").write_text(f"{name}-a")
    (source / "b.txt").write_text(f"{name}-b")
    output_hash = cache.save_to_cache(
        source, space["object_root"], checkout_mode=cache.CheckoutMode.SYMLINK
    )
    assert types.is_dir_hash(output_hash), "a directory hashes to a tree"
    cache.remove_output(source)
    return output_hash


def _helper_write_lock(
    space: Workspace,
    stage_name: str = _STAGE,
    outs: dict[str | None, types.HashInfo] | None = None,
    deps: dict[str, types.HashInfo] | None = None,
) -> None:
    """Write the lock file a stage writes after it runs."""
    lock.StageLock(stage_name, space["stages_dir"]).write(
        types.LockData(
            code_manifest={"self:test": "abc123"},
            params={},
            dep_hashes={
                types.ArtifactIdentity(producer, None): hash_info
                for producer, hash_info in (deps or {}).items()
            },
            output_hashes={
                types.ArtifactIdentity(stage_name, key): hash_info
                for key, hash_info in (outs or {}).items()
            },
        )
    )


def _helper_write_run_cache_row(space: Workspace, hash_info: types.HashInfo) -> None:
    """Write the run-cache row a committed stage writes."""
    with state.StateDB(space["state_dir"]) as db:
        db.apply_deferred_writes(
            _STAGE,
            [],
            types.DeferredWrites(
                run_cache_input_hash="0123456789abcdef",
                run_cache_entry=run_history.RunCacheEntry(
                    run_id="__committed__",
                    output_hashes=[
                        run_history.output_hash_to_entry("data/demo/train.txt", hash_info)
                    ],
                ),
            ),
        )


def _helper_run(runner: click.testing.CliRunner, *args: str) -> click.testing.Result:
    return runner.invoke(cli.cli, ["gc", *args])


def _helper_json(runner: click.testing.CliRunner, *args: str) -> gc.GcResult:
    """Run a collection and parse the one document it writes to stdout."""
    result = _helper_run(runner, "--json", *args)
    assert result.exit_code == 0, result.output
    return json.loads(result.output[result.output.index("{") :])


def _helper_keys(space: Workspace) -> set[str]:
    """Every cache key still in the object store."""
    keys = set[str]()
    for prefix in space["object_root"].iterdir():
        if len(prefix.name) != 2:
            continue
        keys.update(
            prefix.name + entry.name
            for entry in prefix.iterdir()
            if cache.is_valid_hash(prefix.name + entry.name)
        )
    return keys


# =============================================================================
# Every class of record keeps the objects it names
# =============================================================================


def test_gc_keeps_an_object_a_lock_names(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """An object a stage lock's outs name survives; an object nothing names does not."""
    named = _helper_publish(workspace, "named")
    _helper_publish(workspace, "unnamed")
    _helper_write_lock(workspace, outs={None: types.FileHash(hash=named)})

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert result.exit_code == 0, result.output
    assert _helper_keys(workspace) == {named}, result.output


def test_gc_keeps_an_object_only_a_lock_dep_names(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """A consumer's lock outlives the producer that made its input, so deps are roots."""
    dep = _helper_publish(workspace, "dependency")
    _helper_publish(workspace, "unnamed")
    _helper_write_lock(workspace, deps={"upstream/clean": types.FileHash(hash=dep)})

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert result.exit_code == 0, result.output
    assert _helper_keys(workspace) == {dep}, result.output


def test_gc_keeps_an_object_only_a_pvt_names(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """`pivot track` writes a .pvt and no lock; its object is live."""
    tracked = _helper_publish(workspace, "tracked")
    _helper_publish(workspace, "unnamed")
    track.write_pvt_file(
        workspace["root"] / "data.txt.pvt",
        track.PvtData(path="data.txt", hash=tracked, size=len("tracked")),
    )

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert result.exit_code == 0, result.output
    assert _helper_keys(workspace) == {tracked}, result.output


def test_gc_keeps_an_object_only_a_run_cache_row_names(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """A run-cache row restores outputs the current lock no longer names."""
    cached = _helper_publish(workspace, "cached")
    _helper_publish(workspace, "unnamed")
    _helper_write_run_cache_row(workspace, types.FileHash(hash=cached))

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert result.exit_code == 0, result.output
    assert _helper_keys(workspace) == {cached}, result.output


def test_gc_keeps_an_object_only_a_workspace_symlink_names(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """`pivot repro --no-commit` and a deleted stage leave outputs with no lock."""
    linked = _helper_publish(workspace, "linked")
    _helper_publish(workspace, "unnamed")
    out = workspace["root"] / "data" / "demo" / "out.txt"
    out.parent.mkdir(parents=True)
    out.symlink_to(cache.get_cache_path(workspace["object_root"], linked))

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert result.exit_code == 0, result.output
    assert _helper_keys(workspace) == {linked}, result.output
    assert out.resolve().exists(), "the checked-out output must not dangle"


def test_gc_keeps_an_object_only_a_cache_ref_names(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """A `kind: cache` store keeps its refs inside the object root."""
    referenced = _helper_publish(workspace, "referenced")
    _helper_publish(workspace, "unnamed")
    ref = workspace["object_root"] / "refs" / "demo" / "train"
    ref.parent.mkdir(parents=True)
    ref.symlink_to(cache.get_cache_path(workspace["object_root"], referenced).resolve())

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert result.exit_code == 0, result.output
    assert _helper_keys(workspace) == {referenced}, result.output


def test_gc_keeps_a_trees_members_while_the_tree_lives(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """A tree holds copies, so its members must be kept beside it or a repair fails."""
    tree = _helper_publish_tree(workspace, "kept-tree")
    _helper_publish(workspace, "unnamed")
    _helper_write_lock(workspace, outs={None: tree})

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert result.exit_code == 0, result.output
    members = {entry["hash"] for entry in tree["manifest"]}
    assert _helper_keys(workspace) == {tree["hash"], *members}, result.output


def test_gc_removes_a_tree_with_its_marker_and_leaves_its_lock_file(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """A tree no record names goes with its `.verified` marker; lock files never go."""
    tree = _helper_publish_tree(workspace, "dead-tree")
    tree_path = cache.get_cache_path(workspace["object_root"], tree["hash"])
    marker = tree_path.with_name(tree_path.name + cache.TREE_MARKER_SUFFIXES[0])
    tree_lock = tree_path.with_name(tree_path.name + ".lock")
    tree_lock.touch()
    assert marker.exists(), "a published tree is marked"

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert result.exit_code == 0, result.output
    assert not tree_path.exists(), result.output
    assert not marker.exists(), "the marker outlived its tree"
    assert tree_lock.exists(), "a lock file was deleted"


def test_gc_keeps_an_object_published_inside_the_grace(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """An object published while a stage runs has no record yet, and is kept."""
    fresh = _helper_publish(workspace, "fresh")

    result = _helper_run(runner, "--yes")

    assert result.exit_code == 0, result.output
    assert _helper_keys(workspace) == {fresh}, result.output
    assert "1 published in the last 24.0h" in result.output, result.output


def test_gc_keeps_an_object_published_from_an_old_source_file(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """A cached file carries its source's mtime, so an object's age is its ctime."""
    source = workspace["scratch"] / "old-source"
    source.write_text("published from an old file")
    a_month_ago = time.time() - 30 * 24 * 3600
    os.utime(source, (a_month_ago, a_month_ago))
    key, _ = cache.hash_file(source)
    cache.copy_to_cache(source, cache.get_cache_path(workspace["object_root"], key))
    cached = cache.get_cache_path(workspace["object_root"], key)
    assert cached.stat().st_mtime < time.time() - 3600, "the copy carried the source's mtime"

    result = _helper_run(runner, "--yes")

    assert result.exit_code == 0, result.output
    assert _helper_keys(workspace) == {key}, result.output
    assert "1 published in the last 24.0h" in result.output, result.output


def test_gc_keeps_an_object_hardlinked_outside_the_cache(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """Some directory entry outside the cache holds that inode: a fact, not a guess."""
    held = _helper_publish(workspace, "held")
    os.link(cache.get_cache_path(workspace["object_root"], held), workspace["scratch"] / "held")

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert result.exit_code == 0, result.output
    assert _helper_keys(workspace) == {held}, result.output
    assert "1 still hardlinked elsewhere" in result.output, result.output


def test_gc_leaves_the_prefix_lock_and_temporary_entries(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """A collection lists by "this name is a key", so it never takes a publisher's entry."""
    _helper_publish(workspace, "dead")
    prefix = next(p for p in workspace["object_root"].iterdir() if len(p.name) == 2)
    temp = prefix / ".pivot_tmp_someone"
    temp.write_text("a publisher's staged file")

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert result.exit_code == 0, result.output
    assert _helper_keys(workspace) == set(), result.output
    assert temp.exists(), "a temporary entry a publisher owns was removed"
    assert (prefix / cache.PREFIX_LOCK).exists(), "the prefix lock was removed"


# =============================================================================
# A shared cache
# =============================================================================


def test_gc_refuses_an_external_cache_that_records_no_project(
    shared: Workspace, runner: click.testing.CliRunner
) -> None:
    """Another project's locks may name these objects, and nothing says who they are."""
    dead = _helper_publish(shared, "dead")

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert result.exit_code != 0, result.output
    assert f"Refusing to collect {shared['cache_dir']}: it is outside this project" in (
        result.output
    ), result.output
    assert "the cache records no project that uses it" in result.output, result.output
    assert "pivot gc --projects /path/to/other-project --projects /path/to/another" in (
        result.output
    ), result.output
    assert "If this project is the only user, run: pivot gc --projects ." in result.output
    assert "Nothing was removed." in result.output
    assert _helper_keys(shared) == {dead}, "a refusal removes nothing"


def test_gc_after_projects_needs_no_flag_next_time(
    shared: Workspace, runner: click.testing.CliRunner
) -> None:
    """--projects registers what it names, so the refusal is a one-time cost per cache."""
    named = _helper_publish(shared, "named")
    _helper_publish(shared, "dead")
    _helper_write_lock(shared, outs={None: types.FileHash(hash=named)})

    first = _helper_run(runner, "--yes", "--grace-hours", "0", "--projects", ".")
    assert first.exit_code == 0, first.output
    assert _helper_keys(shared) == {named}, first.output

    _helper_publish(shared, "dead again")
    second = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert second.exit_code == 0, second.output
    assert _helper_keys(shared) == {named}, second.output


def test_gc_keeps_what_a_second_projects_locks_name(
    shared: Workspace, runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """One worktree cannot read another's locks; the register is what says they exist."""
    mine = _helper_publish(shared, "mine")
    theirs = _helper_publish(shared, "theirs")
    _helper_publish(shared, "dead")
    _helper_write_lock(shared, outs={None: types.FileHash(hash=mine)})

    other_root = tmp_path / "second"
    other_root.mkdir()
    other = _helper_workspace(other_root, shared["cache_dir"])
    _helper_write_lock(other, stage_name="other/build", outs={None: types.FileHash(hash=theirs)})

    result = _helper_run(runner, "--yes", "--grace-hours", "0", "--projects", str(other_root))

    assert result.exit_code == 0, result.output
    assert _helper_keys(shared) == {mine, theirs}, result.output
    assert "in 2 projects" in result.output, result.output


def test_gc_drops_a_registered_project_whose_root_is_gone(
    shared: Workspace, runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """An entry whose directory is gone took its locks and its links with it."""
    named = _helper_publish(shared, "named")
    _helper_write_lock(shared, outs={None: types.FileHash(hash=named)})
    retired = tmp_path / "retired"
    retired.mkdir()
    gc.register_project(shared["cache_dir"], retired, retired / ".pivot")
    retired.rmdir()

    result = _helper_run(runner, "--yes", "--grace-hours", "0", "--projects", ".")

    assert result.exit_code == 0, result.output
    assert "Dropped 1 projects from the cache register; their roots are gone: " in (
        result.output
    ), result.output
    assert str(retired) in result.output, result.output


def test_gc_refuses_when_a_registered_project_cannot_be_read(
    shared: Workspace, runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A project whose records cannot be read may hold outputs this run would remove."""
    dead = _helper_publish(shared, "dead")
    other_root = tmp_path / "second"
    other_root.mkdir()
    other = _helper_workspace(other_root, shared["cache_dir"])
    gc.register_project(shared["cache_dir"], other_root, other["state_dir"])
    gc.register_project(shared["cache_dir"], shared["root"], shared["state_dir"])
    # A lock file no reader can open: the collection learns nothing about what it names.
    (other["stages_dir"] / "other" / "build.lock").mkdir(parents=True)

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert result.exit_code != 0, result.output
    assert f"Refusing to collect {shared['cache_dir']}: it records 2 projects and 1 cannot " in (
        result.output
    ), result.output
    assert f"  {other_root} (host " in result.output, result.output
    assert "Is a directory" in result.output, result.output
    assert "That project may hold outputs this run would remove." in result.output
    assert "an entry whose directory is gone is dropped" in result.output
    assert _helper_keys(shared) == {dead}, "a refusal removes nothing"


def test_gc_refuses_when_a_register_entry_cannot_be_read(
    shared: Workspace, runner: click.testing.CliRunner
) -> None:
    """A register entry pivot cannot parse hides a project that may own these objects."""
    dead = _helper_publish(shared, "dead")
    entry = gc.register_path(shared["cache_dir"], shared["root"])
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("{not json")

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert result.exit_code != 0, result.output
    assert "it records 1 projects and 1 cannot be read:" in result.output, result.output
    assert str(entry) in result.output, result.output
    assert _helper_keys(shared) == {dead}, "a refusal removes nothing"


def test_gc_needs_no_register_when_the_cache_is_inside_the_project(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """A cache under the project root has one user by construction."""
    _helper_publish(workspace, "dead")

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert result.exit_code == 0, result.output
    assert _helper_keys(workspace) == set(), result.output


# =============================================================================
# Dry run, confirmation, disclosure and output
# =============================================================================


def test_gc_dry_run_removes_nothing_and_counts_what_a_real_run_removes(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """The plan a user reads is the collection they then get."""
    named = _helper_publish(workspace, "named")
    dead = _helper_publish(workspace, "dead")
    other_dead = _helper_publish(workspace, "other dead")
    _helper_write_lock(workspace, outs={None: types.FileHash(hash=named)})

    planned = _helper_json(runner, "--dry-run", "--grace-hours", "0")
    assert _helper_keys(workspace) == {named, dead, other_dead}, "a dry run removes nothing"

    done = _helper_json(runner, "--yes", "--grace-hours", "0")

    assert planned["dry_run"] is True
    assert done["dry_run"] is False
    assert planned["removed"] == done["removed"], (planned, done)
    assert planned["kept"] == done["kept"], (planned, done)
    assert planned["scanned"] == done["scanned"], (planned, done)
    assert _helper_keys(workspace) == {named}


def test_gc_dry_run_exits_zero_and_says_nothing_was_removed(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    dead = _helper_publish(workspace, "dead")

    result = _helper_run(runner, "--dry-run", "--grace-hours", "0")

    assert result.exit_code == 0, result.output
    assert "Would remove 1 objects" in result.output, result.output
    assert "Nothing was removed (--dry-run)." in result.output, result.output
    assert _helper_keys(workspace) == {dead}


def test_gc_remove_linked_discloses_the_objects_it_takes(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """Before removing an object another directory entry holds, the plan says how many."""
    held = _helper_publish(workspace, "held")
    os.link(cache.get_cache_path(workspace["object_root"], held), workspace["scratch"] / "held")

    plan = _helper_run(runner, "--dry-run", "--grace-hours", "0", "--remove-linked")
    assert plan.exit_code == 0, plan.output
    assert "--remove-linked: 1 of the 1 objects to remove" in plan.output, plan.output
    assert "are still hardlinked outside this cache" in plan.output, plan.output
    assert _helper_keys(workspace) == {held}

    done = _helper_run(runner, "--yes", "--grace-hours", "0", "--remove-linked")

    assert done.exit_code == 0, done.output
    assert "--remove-linked: 1 of the 1 objects to remove" in done.output, done.output
    assert _helper_keys(workspace) == set()


def test_gc_refuses_to_remove_without_a_terminal_and_without_yes(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """Cron and CI cannot delete by accident."""
    dead = _helper_publish(workspace, "dead")

    result = _helper_run(runner, "--grace-hours", "0")

    assert result.exit_code != 0, result.output
    assert "stdin is not a terminal" in result.output, result.output
    assert _helper_keys(workspace) == {dead}


def test_gc_json_document_has_the_documented_shape(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    named = _helper_publish(workspace, "named")
    _helper_publish(workspace, "dead")
    _helper_write_lock(workspace, outs={None: types.FileHash(hash=named)})

    document = _helper_json(runner, "--yes", "--grace-hours", "0")

    assert set(document) == {"cache", "scanned", "kept", "removed", "projects", "dry_run"}
    assert document["cache"] == str(workspace["cache_dir"])
    assert document["projects"] == [str(workspace["root"])]
    assert document["scanned"]["objects"] == 2
    assert document["removed"]["objects"] == 1
    assert document["kept"]["objects"] == 1
    assert set(document["kept"]["by"]) == {
        "locks",
        "run_cache",
        "links",
        "grace",
        "hardlinked",
    }, document
    assert document["kept"]["by"]["locks"] == 1


def test_gc_summary_counts_every_class_of_record(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """The summary says what kept each object."""
    named = _helper_publish(workspace, "named")
    cached = _helper_publish(workspace, "cached")
    linked = _helper_publish(workspace, "linked")
    _helper_write_lock(workspace, outs={None: types.FileHash(hash=named)})
    _helper_write_run_cache_row(workspace, types.FileHash(hash=cached))
    out = workspace["root"] / "out.txt"
    out.symlink_to(cache.get_cache_path(workspace["object_root"], linked))

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert result.exit_code == 0, result.output
    assert "Scanned 3 objects" in result.output, result.output
    assert "2 named by 1 locks and 1 run-cache entries in 1 projects" in result.output
    assert "1 by 1 workspace links" in result.output, result.output
    assert "Removed 0 objects" in result.output, result.output


# =============================================================================
# Concurrency
# =============================================================================


def test_gc_refuses_while_another_collection_holds_the_cache(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """One collection at a time, so two never plan against each other's deletions."""
    dead = _helper_publish(workspace, "dead")
    fd = cache.open_cache_lock(workspace["cache_dir"] / gc.GC_LOCK)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        result = _helper_run(runner, "--yes", "--grace-hours", "0")
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    assert result.exit_code != 0, result.output
    assert "another pivot gc holds" in result.output, result.output
    assert _helper_keys(workspace) == {dead}


def _helper_refuse_flock(_fd: int, _operation: int) -> None:
    """What flock() does on Lustre without `flock` and on NFS with no lock manager."""
    raise OSError(errno.ENOSYS, os.strerror(errno.ENOSYS))


def test_gc_refuses_grace_zero_where_the_filesystem_refuses_flock(
    workspace: Workspace, runner: click.testing.CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where the locks exclude nothing, the grace is the only guard left, so it must stay."""
    dead = _helper_publish(workspace, "dead")
    monkeypatch.setattr(fcntl, "flock", _helper_refuse_flock)

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert result.exit_code != 0, result.output
    assert "this filesystem refuses flock()" in result.output, result.output
    assert "Run with a grace period." in result.output, result.output
    assert _helper_keys(workspace) == {dead}, "a refusal removes nothing"


def test_gc_warns_but_collects_with_a_grace_where_flock_is_refused(
    workspace: Workspace, runner: click.testing.CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a grace period the collection may go on, and says what the locks are worth."""
    _helper_publish(workspace, "dead")
    monkeypatch.setattr(fcntl, "flock", _helper_refuse_flock)
    # 0.0001h is 0.36s: waiting longer puts the object outside the grace whatever the load.
    time.sleep(0.6)

    result = _helper_run(runner, "--yes", "--grace-hours", "0.0001")

    assert result.exit_code == 0, result.output
    assert "refuses flock(), so pivot's cache locks exclude nothing here" in result.output
    assert _helper_keys(workspace) == set(), result.output


def test_gc_waits_for_a_prefix_a_publisher_holds(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """The prefix lock a staging publisher holds keeps the collection out of that prefix."""
    dead = _helper_publish(workspace, "dead")
    dead_path = cache.get_cache_path(workspace["object_root"], dead)
    fd = cache.open_cache_lock(dead_path.parent / cache.PREFIX_LOCK)
    fcntl.flock(fd, fcntl.LOCK_EX)
    outcome = list[int]()

    def _collect() -> None:
        outcome.append(_helper_run(runner, "--yes", "--grace-hours", "0").exit_code)

    collector = threading.Thread(target=_collect)
    collector.start()
    try:
        time.sleep(1.0)
        assert dead_path.exists(), "the collection deleted inside a prefix a publisher holds"
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        collector.join(timeout=60)

    assert not collector.is_alive(), "the collection never finished"
    assert outcome == [0], outcome
    assert not dead_path.exists(), "the collection did not resume once the lock was free"


# =============================================================================
# Run history
# =============================================================================


def test_gc_prunes_run_history_to_the_retention(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """The state-database warning sends users here, so `gc` applies the retention policy."""
    (workspace["state_dir"] / "config.yaml").write_text("core:\n  run_history_retention: 1\n")
    with state.StateDB(workspace["state_dir"]) as db:
        for index in range(3):
            db.write_run(
                run_history.RunManifest(
                    run_id=f"20260101T00000{index}-{index}",
                    started_at="2026-01-01T00:00:00Z",
                    ended_at="2026-01-01T00:00:01Z",
                    targeted_stages=[],
                    execution_order=[],
                    stages={},
                )
            )

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert result.exit_code == 0, result.output
    assert "Pruned 2 runs beyond the retention of 1" in result.output, result.output
    with state.StateDB(workspace["state_dir"]) as db:
        assert len(db.list_runs()) == 1, result.output


# =============================================================================
# Concurrency, round 2: the records re-read that keeps an object a run names
# =============================================================================


def test_gc_keeps_an_object_a_lock_named_after_the_plan(
    workspace: Workspace, runner: click.testing.CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deletions re-read the records after the scan, so a lock a concurrent run writes
    between the plan and the delete keeps its object even when the grace does not cover it.
    Guard two of B1; a mutant that drops the re-read removes this object and fails here."""
    key = _helper_publish(workspace, "named after the plan")
    time.sleep(1.1)  # older than the one-second grace, so only the re-read can save it
    real_execute = gc.execute_plan

    def write_lock_then_execute(plan: gc.Plan, **kwargs: object) -> gc.ObjectCount:
        _helper_write_lock(workspace, outs={None: types.FileHash(hash=key)})
        return real_execute(plan, **kwargs)  # pyright: ignore[reportArgumentType] - the same call

    monkeypatch.setattr(gc, "execute_plan", write_lock_then_execute)

    result = _helper_run(runner, "--yes", "--grace-hours", str(1.0 / 3600.0))

    assert result.exit_code == 0, result.output
    assert key in _helper_keys(workspace), "the deletions did not re-read the records"


def test_gc_keeps_a_republished_object_hardlinked_after_the_plan(
    workspace: Workspace, runner: click.testing.CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hardlink-mode control that already passed: a hardlink checkout made between the plan
    and the delete shares the object's inode, so the per-object re-check keeps it by nlink even
    with no record and no grace. A mutant that drops the execute-time re-check fails here."""
    key = _helper_publish(workspace, "hardlinked after the plan")
    path = cache.get_cache_path(workspace["object_root"], key)
    time.sleep(1.1)
    real_execute = gc.execute_plan

    def hardlink_then_execute(plan: gc.Plan, **kwargs: object) -> gc.ObjectCount:
        os.link(path, workspace["root"] / "data_hardlink")
        return real_execute(plan, **kwargs)  # pyright: ignore[reportArgumentType] - the same call

    monkeypatch.setattr(gc, "execute_plan", hardlink_then_execute)

    result = _helper_run(runner, "--yes", "--grace-hours", str(1.0 / 3600.0))

    assert result.exit_code == 0, result.output
    assert key in _helper_keys(workspace), "a hardlinked object was removed"


# =============================================================================
# BLK-2: a symlinked directory inside the project must be walked
# =============================================================================


def test_gc_keeps_an_object_a_pvt_under_a_symlinked_directory_names(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """A project whose data directory is a symlink to another directory inside the project: the
    walk must follow it and see the .pvt beneath it. The mutant that skips symlinked dirs misses
    this record and removes the object."""
    tracked = _helper_publish(workspace, "under-symlink")
    real_dir = workspace["root"] / "data_real"
    real_dir.mkdir()
    track.write_pvt_file(
        real_dir / "tracked.txt.pvt",
        track.PvtData(path="tracked.txt", hash=tracked, size=len("under-symlink")),
    )
    link_dir = workspace["root"] / "data"
    link_dir.symlink_to(real_dir)

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert result.exit_code == 0, result.output
    assert tracked in _helper_keys(workspace), (
        "an object named by a .pvt under a symlinked directory was removed"
    )


# =============================================================================
# BLK-3: a registered peer of an internal cache must be read
# =============================================================================


def test_gc_keeps_an_object_a_registered_peer_of_an_internal_cache_names(
    workspace: Workspace, runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A cache inside the project root can have a registered peer (another project pointing its
    PIVOT_CACHE_DIR at this one's internal cache). The collection must read the register and keep
    every object the peer's locks name."""
    mine = _helper_publish(workspace, "mine")
    theirs = _helper_publish(workspace, "theirs")
    _helper_write_lock(workspace, outs={None: types.FileHash(hash=mine)})
    peer = tmp_path / "peer"
    peer_stages = peer / ".pivot" / "stages"
    peer_stages.mkdir(parents=True)
    lock.StageLock("peer/stage", peer_stages).write(
        types.LockData(
            code_manifest={"self:peer": "abc123"},
            params={},
            dep_hashes={},
            output_hashes={
                types.ArtifactIdentity("peer/stage", None): types.FileHash(hash=theirs)
            },
        )
    )
    gc.register_project(workspace["cache_dir"], peer, peer / ".pivot")

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert result.exit_code == 0, result.output
    assert theirs in _helper_keys(workspace), (
        "an object named by a registered peer of an internal cache was removed"
    )
