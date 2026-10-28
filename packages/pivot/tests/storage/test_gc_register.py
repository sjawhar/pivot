import json
import pathlib

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
