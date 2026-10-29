import json
import pathlib
import shutil

import pytest

from pivot.storage import gc


def _project(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """A project root, its `.pivot` state directory and an existing cache directory."""
    root = tmp_path / "project"
    (root / ".pivot").mkdir(parents=True)
    root = root.resolve()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return root, root / ".pivot", cache_dir


def _unit(cache_dir: pathlib.Path, root: pathlib.Path, state_dir: pathlib.Path) -> None:
    with gc.naming_unit(cache_dir, project=(root, state_dir)):
        pass


def test_a_naming_unit_reregisters_after_the_mount_table_changes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A long-lived writer (the engine, `--watch`) runs many naming units in one process. When
    a disk is mounted under the project between two of them, the second records the mount, so
    a collection can refuse once it is unmounted."""
    root, state_dir, cache_dir = _project(tmp_path)
    table = tmp_path / "mounts"
    table.write_bytes(b"/dev/root / ext4 rw 0 0\n")
    monkeypatch.setattr(gc, "_MOUNTS_TABLE", table)
    _unit(cache_dir, root, state_dir)
    entry = gc.project_register_path(cache_dir, root, state_dir)
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
    first = gc.project_register_path(cache_dir, root, state_dir)
    assert first is not None and first.exists()

    (state_dir / "state.lmdb" / gc.PROJECT_ID_FILE).unlink()
    _unit(cache_dir, root, state_dir)

    second = gc.project_register_path(cache_dir, root, state_dir)
    assert second is not None and second != first
    assert second.exists(), "the unit after the reset did not register the new project id"


def test_a_naming_unit_reregisters_after_another_writer_replaces_the_entry(
    tmp_path: pathlib.Path,
) -> None:
    """Another host that shares the project can rewrite its entry. The next unit here writes
    this host's evidence back instead of trusting what it verified before."""
    root, state_dir, cache_dir = _project(tmp_path)
    _unit(cache_dir, root, state_dir)
    entry = gc.project_register_path(cache_dir, root, state_dir)
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
    entry = gc.project_register_path(cache_dir, root, state_dir)
    assert entry is not None
    nonces = state_dir / "state.lmdb" / gc.VIEW_NONCE_DIR
    (nonce_file,) = nonces.iterdir()
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
    entry = gc.project_register_path(cache_dir, root, state_dir)
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
    entry = gc.project_register_path(cache_dir, root, state_dir)
    assert entry is not None
    before = entry.stat()

    gc._VERIFIED_REGISTRATIONS.clear()  # pyright: ignore[reportPrivateUsage] - another process
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
    entry = gc.project_register_path(cache_dir, root, state_dir)
    assert entry is not None
    document = json.loads(entry.read_text())
    document["last_write"] = "2020-01-01T00:00:00Z"
    entry.write_text(json.dumps(document, sort_keys=True) + "\n")

    gc._VERIFIED_REGISTRATIONS.clear()  # pyright: ignore[reportPrivateUsage] - another process
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
    entry = gc.project_register_path(cache_dir, root, state_dir)
    assert entry is not None
    nonces = state_dir / "state.lmdb" / gc.VIEW_NONCE_DIR
    (nonce_file,) = nonces.iterdir()
    nonce_file.write_text("0" * 64 + "\n")

    gc._VERIFIED_REGISTRATIONS.clear()  # pyright: ignore[reportPrivateUsage] - another process
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
        live, unreadable = gc.read_register(cache_dir)
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
    monkeypatch.setattr(gc, "_this_view", lambda: ("cloned-host", "same-machine-id", 1))
    root, state_dir, cache_dir = _project(tmp_path)
    _unit(cache_dir, root, state_dir)
    clone = tmp_path / "clone-state"
    shutil.copytree(state_dir, clone, symlinks=True)
    _swap(state_dir, clone)
    try:
        _unit(cache_dir, root, state_dir)  # the clone registers last, under the same key
    finally:
        _swap(state_dir, clone)

    live, unreadable = gc.read_register(cache_dir)

    assert live == [], "the view that did not write last accepted the other's entry"
    assert len(unreadable) == 1, unreadable
