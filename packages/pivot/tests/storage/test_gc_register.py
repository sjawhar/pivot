import fcntl
import json
import logging
import os
import pathlib
import shutil
import subprocess
import threading

import pytest

from pivot import exceptions
from pivot import state_dir as state_dir_mod
from pivot.storage import cache, gc, naming


def _project(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """A project root, its `.pivot` state directory and an existing cache directory."""
    root = tmp_path / "project"
    (root / ".pivot").mkdir(parents=True)
    root = root.resolve()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return root, root / ".pivot", cache_dir


def _unit(cache_dir: pathlib.Path, root: pathlib.Path, state_dir: pathlib.Path) -> None:
    with naming.naming_unit(cache_dir, project=(root, state_dir)):
        pass


def _read_register(cache_dir: pathlib.Path) -> tuple[list[gc.ProjectScope], list[tuple[str, str]]]:
    """What a collection of ``cache_dir`` reads from its register."""
    with gc.store_snapshot(cache_dir) as snapshot:
        return gc.read_register(snapshot)


def test_a_link_refusal_is_per_cache_and_lifts_once_a_registration_succeeds(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A naming step that cannot register in a read-only cache makes the process refuse links
    into that cache (`cache.refuse_links_from`). The refusal is per cache: a link into another
    cache is not refused. A long-lived process that later gains write access lifts it at its
    next registration there (`naming.register_project`), so it is never stuck refusing."""
    monkeypatch.setattr(cache, "_UNREGISTERED_ROOTS", dict[str, str]())
    root, state_dir, cache_dir = _project(tmp_path)
    other = tmp_path / "other-cache"
    other.mkdir()
    target = cache_dir / "files" / "ab" / "cdef0123456789"
    link = tmp_path / "link"
    cache.refuse_links_from(cache_dir, (cache_dir, naming.register_root(cache_dir)))

    with pytest.raises(exceptions.CacheError, match="Refusing to link"):
        cache.refuse_unregistered_link(target, link)
    cache.refuse_unregistered_link(other / "files" / "ab" / "cdef0123456789", link)

    naming.register_project(cache_dir, root, state_dir)

    cache.refuse_unregistered_link(target, link)


def test_a_naming_unit_reregisters_after_the_mount_table_changes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A long-lived writer (the engine, `--watch`) runs many naming units in one process. When
    a disk is mounted under the project between two of them, the second records the mount, so
    a collection can refuse once it is unmounted. Here the table is a regular file, which no
    kernel watch covers (`_watch_mount_table` refuses it), so every unit registers in full;
    `test_a_naming_unit_records_a_mount_that_appeared_since_the_last_one` covers the watch. The
    process's one watch is opened anew, as in a process that starts with this table: one an
    earlier test opened on `/proc/self/mounts` would see no change to the regular file."""
    root, state_dir, cache_dir = _project(tmp_path)
    table = tmp_path / "mounts"
    table.write_bytes(b"/dev/root / ext4 rw 0 0\n")
    monkeypatch.setattr(naming, "_MOUNTS_TABLE", table)
    monkeypatch.setattr(naming, "_SHARED_MOUNT_WATCH", naming._SharedMountWatch())
    _unit(cache_dir, root, state_dir)
    entry = naming.project_register_path(cache_dir, root, state_dir)
    assert entry is not None
    assert json.loads(entry.read_text())["mounts"] == []

    data = root / "data"
    data.mkdir()
    table.write_bytes(table.read_bytes() + f"tmpfs {data} tmpfs rw 0 0\n".encode())
    _unit(cache_dir, root, state_dir)

    assert json.loads(entry.read_text())["mounts"] == [str(data)]


def test_a_naming_unit_reregisters_after_the_project_id_is_replaced(
    tmp_path: pathlib.Path,
) -> None:
    """A state database reset between two naming units gives the project a new id; the second
    unit writes the entry for the new id, so a collection does not refuse the live project."""
    root, state_dir, cache_dir = _project(tmp_path)
    _unit(cache_dir, root, state_dir)
    first = naming.project_register_path(cache_dir, root, state_dir)
    assert first is not None and first.exists()

    (state_dir / "state.lmdb" / naming.PROJECT_ID_FILE).unlink()
    _unit(cache_dir, root, state_dir)

    second = naming.project_register_path(cache_dir, root, state_dir)
    assert second is not None and second != first
    assert second.exists(), "the unit after the reset did not register the new project id"


def test_a_naming_unit_reregisters_after_another_writer_replaces_the_entry(
    tmp_path: pathlib.Path,
) -> None:
    """Another host that shares the project can rewrite its entry. The next unit here writes
    this host's evidence back instead of trusting what it verified before."""
    root, state_dir, cache_dir = _project(tmp_path)
    _unit(cache_dir, root, state_dir)
    entry = naming.project_register_path(cache_dir, root, state_dir)
    assert entry is not None
    replaced = json.loads(entry.read_text())
    replaced["host"] = "pivot-other-node"
    entry.write_text(json.dumps(replaced))

    _unit(cache_dir, root, state_dir)

    assert json.loads(entry.read_text())["host"] != "pivot-other-node"


def test_a_naming_unit_reopens_a_view_nonce_file_that_was_replaced(tmp_path: pathlib.Path) -> None:
    """The view's nonce file is replaced between two units (its state database was reset, or
    another tool rewrote it). The next unit must write its nonce to the file now at that path,
    not to the one it opened before, so the entry and the file a collection reads agree."""
    root, state_dir, cache_dir = _project(tmp_path)
    _unit(cache_dir, root, state_dir)
    entry = naming.project_register_path(cache_dir, root, state_dir)
    assert entry is not None
    nonces = state_dir / "state.lmdb" / naming.VIEW_NONCE_DIR
    (nonce_file,) = (p for p in nonces.iterdir() if p.suffix != naming.CACHE_RECORD_SUFFIX)
    nonce_file.unlink()
    nonce_file.write_text("0" * 32 + "\n")

    _unit(cache_dir, root, state_dir)

    register_nonce = entry.with_suffix(".nonce").read_text().strip()
    assert register_nonce == nonce_file.read_text().strip()


def test_a_naming_unit_reregisters_after_its_entry_is_deleted(tmp_path: pathlib.Path) -> None:
    """A user deletes the register entry (as a refusal tells them to, for a project they took
    for gone). The next unit of a process that still uses the project must write it again, or
    the project's records would be invisible to every collection."""
    root, state_dir, cache_dir = _project(tmp_path)
    _unit(cache_dir, root, state_dir)
    entry = naming.project_register_path(cache_dir, root, state_dir)
    assert entry is not None
    entry.unlink()

    _unit(cache_dir, root, state_dir)

    assert entry.exists(), "a naming unit ran without its project in the register"


def test_a_second_process_keeps_the_entry_another_just_wrote(tmp_path: pathlib.Path) -> None:
    """Every process of one view writes the same entry but its timestamp, and a worker pool is
    a dozen of them. A process whose first unit finds that entry keeps it, so the register
    takes one write per view instead of one per process."""
    root, state_dir, cache_dir = _project(tmp_path)
    _unit(cache_dir, root, state_dir)
    entry = naming.project_register_path(cache_dir, root, state_dir)
    assert entry is not None
    before = entry.stat()

    naming._VERIFIED_REGISTRATIONS.clear()  # as a second process of this view would start
    _unit(cache_dir, root, state_dir)

    after = entry.stat()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino), (
        "a second process replaced an entry that already said what it would write"
    )


def test_a_registration_refreshes_an_entry_whose_last_write_is_old(tmp_path: pathlib.Path) -> None:
    """``last_write`` tells a reader whether a registered project is still in use, so an entry
    that says what this registration would say is still rewritten once it is stale."""
    root, state_dir, cache_dir = _project(tmp_path)
    _unit(cache_dir, root, state_dir)
    entry = naming.project_register_path(cache_dir, root, state_dir)
    assert entry is not None
    document = json.loads(entry.read_text())
    document["last_write"] = "2020-01-01T00:00:00Z"
    entry.write_text(json.dumps(document, sort_keys=True) + "\n")

    naming._VERIFIED_REGISTRATIONS.clear()  # as a second process of this view would start
    _unit(cache_dir, root, state_dir)

    assert json.loads(entry.read_text())["last_write"] != "2020-01-01T00:00:00Z", (
        "a stale entry was kept, so the register no longer shows the project in use"
    )


def test_a_registration_truncates_a_nonce_file_that_held_more(tmp_path: pathlib.Path) -> None:
    """A nonce file is rewritten in place. One that held longer bytes (another tool wrote
    there) must be cut to the nonce, or every reader refuses it and a collection refuses the
    project for ever."""
    root, state_dir, cache_dir = _project(tmp_path)
    _unit(cache_dir, root, state_dir)
    entry = naming.project_register_path(cache_dir, root, state_dir)
    assert entry is not None
    nonces = state_dir / "state.lmdb" / naming.VIEW_NONCE_DIR
    (nonce_file,) = (p for p in nonces.iterdir() if p.suffix != naming.CACHE_RECORD_SUFFIX)
    nonce_file.write_text("0" * 64 + "\n")

    naming._VERIFIED_REGISTRATIONS.clear()  # as a second process of this view would start
    _unit(cache_dir, root, state_dir)

    nonce = nonce_file.read_text()
    assert len(nonce) == 33 and nonce == entry.with_suffix(".nonce").read_text(), nonce


def _swap(a: pathlib.Path, b: pathlib.Path) -> None:
    """Exchange two directories' contents by renaming, so ``a``'s path shows ``b``'s copy."""
    spare = a.with_name(a.name + ".swap")
    a.rename(spare)
    b.rename(a)
    spare.rename(b)


def test_a_clone_taken_while_a_writer_runs_holds_a_nonce_its_next_unit_replaces(
    tmp_path: pathlib.Path,
) -> None:
    """P19: a long-running writer registers; its state directory is cloned (a VM snapshot or
    `docker commit`); the same process then names something new. The clone's collection, which
    sees its copy at the same path, must refuse the entry rather than read the stale copy."""
    root, state_dir, cache_dir = _project(tmp_path)
    _unit(cache_dir, root, state_dir)
    clone = tmp_path / "clone-state"
    shutil.copytree(state_dir, clone, symlinks=True)

    _unit(cache_dir, root, state_dir)  # the same process names a new object

    _swap(state_dir, clone)  # the clone's view: its copy at the same path
    try:
        live, unreadable = _read_register(cache_dir)
    finally:
        _swap(state_dir, clone)
    assert live == [], "the clone's collection accepted the running writer's entry"
    assert len(unreadable) == 1, unreadable


def test_two_views_that_share_a_view_key_refuse_each_others_entry(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two cloned VMs that keep the hostname, machine id and initial mount namespace share a view
    key, so the last writer's entry replaces the other's. The view that did not write last holds
    another nonce, so its collection refuses. (The view that wrote last accepts the entry and
    reads its own records; the other view's objects named since the clone are not in them.)"""
    monkeypatch.setattr(naming, "this_view", lambda: ("cloned-host", "same-machine-id", 1))
    root, state_dir, cache_dir = _project(tmp_path)
    _unit(cache_dir, root, state_dir)
    clone = tmp_path / "clone-state"
    shutil.copytree(state_dir, clone, symlinks=True)
    _swap(state_dir, clone)
    try:
        _unit(cache_dir, root, state_dir)  # the clone registers last, under the same key
    finally:
        _swap(state_dir, clone)

    live, unreadable = _read_register(cache_dir)

    assert live == [], "the view that did not write last accepted the other's entry"
    assert len(unreadable) == 1, unreadable


def test_a_unit_registers_its_view_in_a_cache_only_another_view_listed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N1 `c1view` without a mount namespace: view 1 of the project uses C2; its state directory
    is copied; view 2 then names objects through C1 only. View 2's step registers it in C2 too,
    a cache only view 1 listed, writing a nonce the copy never held, so a collection of C2 that
    reads the copy refuses. View 1's own pair is never touched by view 2."""
    root, state_dir, c2 = _project(tmp_path)
    c1 = tmp_path / "cache-one"
    c1.mkdir()
    monkeypatch.setattr(naming, "this_view", lambda: ("host", "machine", 1))
    _unit(c2, root, state_dir)
    view_one = naming.project_register_path(c2, root, state_dir)
    assert view_one is not None
    view_one_nonce = naming.entry_nonce_path(view_one).read_text()
    copy = tmp_path / "copy-state"
    shutil.copytree(state_dir, copy, symlinks=True)
    naming._VERIFIED_REGISTRATIONS.clear()  # view 2 is another process
    monkeypatch.setattr(naming, "this_view", lambda: ("host", "machine", 2))

    _unit(c1, root, state_dir)

    view_two = naming.project_register_path(c2, root, state_dir)
    assert view_two is not None and view_two.exists(), "view 2's step left C2 without its entry"
    assert naming.entry_nonce_path(view_one).read_text() == view_one_nonce
    _swap(state_dir, copy)
    try:
        _live, unreadable = _read_register(c2)
    finally:
        _swap(state_dir, copy)
    assert any(str(view_two) in reason for _label, reason in unreadable), unreadable


def test_a_unit_warns_once_per_command_about_a_listed_cache_it_cannot_reach(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A listed cache a naming step cannot reach draws one warning per command, however many
    units the command takes, and none from a worker's stage (a record that reports nothing and
    is no library entry point), whose command's own step warns."""
    root, state_dir, cache_dir = _project(tmp_path)
    other = tmp_path / "other-cache"
    other.mkdir()
    _unit(other, root, state_dir)
    other.rename(tmp_path / "gone")

    def warnings_of(*, report: bool) -> list[str]:
        caplog.clear()
        token = state_dir_mod.begin_invocation(report=report)
        try:
            with caplog.at_level(logging.WARNING, logger=naming.__name__):
                _unit(cache_dir, root, state_dir)
                _unit(cache_dir, root, state_dir)
        finally:
            state_dir_mod.end_invocation(token)
        return [r.getMessage() for r in caplog.records if "cannot reach it" in r.getMessage()]

    assert len(warnings_of(report=True)) == 1
    assert warnings_of(report=False) == []


def test_a_unit_registers_again_when_files_is_relinked(tmp_path: pathlib.Path) -> None:
    """The registration memo lets a long-lived writer's units skip the reads while the evidence
    it checked is unchanged. Re-pointing the cache's `files/` to another object store moves the
    register beside that store (`naming.store_layout`), so the next unit registers there in
    full instead of rotating the nonce of an entry no collection of the store reads; so does a
    second re-pointing, which the process's memo of the linked layout must notice."""
    root, state_dir, cache_dir = _project(tmp_path)
    _unit(cache_dir, root, state_dir)
    before = naming.project_register_path(cache_dir, root, state_dir)
    assert before is not None and before.exists()
    store = tmp_path / "shared" / "objects"
    store.mkdir(parents=True)
    (cache_dir / "files").symlink_to(store)

    _unit(cache_dir, root, state_dir)

    after = naming.project_register_path(cache_dir, root, state_dir)
    assert after is not None and after.parent.parent == store.resolve().parent
    assert after.exists(), "the unit after the relink left the store's register without an entry"
    other = tmp_path / "other-shared" / "objects"
    other.mkdir(parents=True)
    relinked = cache_dir / "files.relinked"
    relinked.symlink_to(other)
    relinked.replace(cache_dir / "files")

    _unit(cache_dir, root, state_dir)

    again = naming.project_register_path(cache_dir, root, state_dir)
    assert again is not None and again.parent.parent == other.resolve().parent
    assert again.exists(), "the unit after the second relink wrote no entry beside that store"


def test_registrations_across_many_caches_hold_a_bounded_number_of_descriptors(
    tmp_path: pathlib.Path,
) -> None:
    """The registration memo keeps both nonce files of each registration open, so a long-lived
    process (a watch or serve session, a library caller) that registers in many caches must not
    hold more descriptors with each new one. It keeps the most recently used registrations, and
    every registration shares one mount-table watch: 100 registrations across 100 caches leave
    the open-descriptor count where 20 left it. The memo of each cache root's store layout is
    bounded too."""
    root, state_dir, _cache_dir = _project(tmp_path)
    caches = [tmp_path / f"cache-{i}" for i in range(100)]
    for cache_dir in caches:
        (cache_dir / "files").mkdir(parents=True)

    def open_descriptors() -> int:
        return len(os.listdir("/proc/self/fd"))

    before = open_descriptors()
    for cache_dir in caches[:20]:
        naming.register_project(cache_dir, root, state_dir)
    after_twenty = open_descriptors()
    for cache_dir in caches[20:]:
        naming.register_project(cache_dir, root, state_dir)
    after_hundred = open_descriptors()

    assert after_hundred == after_twenty, (before, after_twenty, after_hundred)
    assert after_hundred - before <= 2 * naming._MAX_VERIFIED_REGISTRATIONS + 1
    assert len(naming._VERIFIED_REGISTRATIONS) <= naming._MAX_VERIFIED_REGISTRATIONS
    assert len(naming._STORE_LAYOUTS) <= naming._MAX_STORE_LAYOUTS
    # An evicted registration is done in full again, and is then remembered.
    naming.register_project(caches[0], root, state_dir)
    assert (caches[0], root, state_dir) in naming._VERIFIED_REGISTRATIONS


def test_a_writer_through_a_linked_root_waits_for_the_lock_a_collection_takes(
    tmp_path: pathlib.Path,
) -> None:
    """A naming unit resolves the cache's layout once, for its lock and its registration. Through
    a cache root whose `files/` links another root's store, it takes the naming lock beside that
    store, the one a collection through either root holds (`naming.naming_lock_paths`): while
    that lock is held exclusively, the unit waits."""
    root, state_dir, plain = _project(tmp_path)
    (plain / "files").mkdir()
    linking = tmp_path / "linking-root"
    linking.mkdir()
    (linking / "files").symlink_to(plain / "files")
    _unit(plain, root, state_dir)
    _ts_path, nm_path = naming.naming_lock_paths(plain)
    entered = threading.Event()

    def write() -> None:
        _unit(linking, root, state_dir)
        entered.set()

    fd = os.open(nm_path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        assert not entered.wait(1.0), "the unit through the linking root took another lock"
    finally:
        os.close(fd)
    assert entered.wait(30.0), "the unit never took the lock once it was free"
    writer.join(30.0)


def test_a_mount_that_leaves_files_the_same_directory_still_changes_its_layout(
    tmp_path: pathlib.Path,
) -> None:
    """A process keeps each cache root's store layout while `files/` is the same entry and the
    mount table has not changed (`naming._STORE_LAYOUTS`). `files/` bind-mounted onto itself
    keeps its device and inode, and only the mount table shows it; the mount-change count must
    then make the next step see an unshared store, so a writer refuses it."""
    if subprocess.run(["sudo", "-n", "true"], capture_output=True, check=False).returncode:
        pytest.skip("bind-mounting needs passwordless sudo")
    root, state_dir, cache_dir = _project(tmp_path)
    files = cache_dir / "files"
    files.mkdir()
    _unit(cache_dir, root, state_dir)
    assert not naming.store_layout(cache_dir).unshared
    subprocess.run(["sudo", "-n", "mount", "--bind", str(files), str(files)], check=True)
    try:
        assert naming.store_layout(cache_dir).unshared, "the layout memo kept a stale answer"
        with pytest.raises(exceptions.CacheError, match="is a mount of its own"):
            _unit(cache_dir, root, state_dir)
    finally:
        subprocess.run(["sudo", "-n", "umount", str(files)], check=True)
    assert not naming.store_layout(cache_dir).unshared


def test_a_files_link_whose_inodes_were_reused_changes_its_layout(tmp_path: pathlib.Path) -> None:
    """The layout memo keys `files/` and the directory it links to by inode as well as ctime. The
    store directory is removed, a directory under another parent is made until it gets the same
    inode (ext4 hands a just-freed inode out again), and `files/` is replaced by a link to it
    that reuses the old link's inode too. Device, inode and mode all match the memo's; the ctimes
    do not, so the next resolution puts the register beside the new store, not the old one."""
    cache_dir = tmp_path / "cache"
    store = tmp_path / "a" / "store"
    other = tmp_path / "b"
    for directory in (cache_dir, store, other):
        directory.mkdir(parents=True)
    files = cache_dir / "files"
    files.symlink_to(store)
    assert naming.store_layout(cache_dir).register_root == store.resolve().parent
    store_inode, link_inode = os.stat(store).st_ino, os.lstat(files).st_ino
    store.rmdir()
    reused: pathlib.Path | None = None
    for attempt in range(4000):
        candidate = other / f"s{attempt}"
        candidate.mkdir()
        if os.stat(candidate).st_ino == store_inode:
            reused = candidate
            break
    files.unlink()
    files.symlink_to(reused if reused is not None else other / "s0")
    if reused is None or os.lstat(files).st_ino != link_inode:
        pytest.skip("this filesystem did not hand the freed inodes out again")

    layout = naming.store_layout(cache_dir)

    assert layout.register_root == reused.resolve().parent, "the layout memo kept the old store"
    assert layout.linked_store == reused.resolve()


def test_a_naming_step_reuses_the_layout_it_resolved_until_files_changes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The layout memo keeps a naming step from reading the mount table: the first step resolves
    the cache's layout (`naming._mount_root`, counted here), the next step reuses it while nothing
    changed, and a `files/` replaced since (another directory at its path) is resolved again."""
    root, state_dir, cache_dir = _project(tmp_path)
    files = cache_dir / "files"
    files.mkdir()
    monkeypatch.setattr(naming, "_STORE_LAYOUTS", dict[pathlib.Path, naming._KnownLayout]())
    monkeypatch.setattr(naming, "_mount_generation", lambda: 0)
    resolved = list[pathlib.Path]()
    mount_root = naming._mount_root

    def counted(path: pathlib.Path) -> bool:
        resolved.append(path)
        return mount_root(path)

    monkeypatch.setattr(naming, "_mount_root", counted)
    _unit(cache_dir, root, state_dir)
    first = len(resolved)
    assert first >= 1, "the first step did not resolve the layout"

    _unit(cache_dir, root, state_dir)
    _unit(cache_dir, root, state_dir)

    assert len(resolved) == first, f"later steps resolved the layout again: {resolved}"
    files.rmdir()
    files.mkdir()
    _unit(cache_dir, root, state_dir)
    assert len(resolved) > first, "a replaced files/ was not resolved again"
