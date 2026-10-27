from __future__ import annotations

import os
import shutil
import stat
from typing import TYPE_CHECKING

from tests import conftest

if TYPE_CHECKING:
    import pathlib


def test_restore_owner_access_makes_locked_directories_deletable_and_changes_nothing_else(
    tmp_path: pathlib.Path,
) -> None:
    """A read-only cache tree that holds only a subdirectory, and a directory a test made
    unreadable, can be deleted afterwards; a read-only directory a symlink reaches outside the
    root and a file hardlinked from outside keep their modes."""
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    locked = outside / "locked"
    locked.mkdir(parents=True)
    locked.chmod(0o555)
    shared = outside / "object"
    shared.write_text("cached\n")
    shared.chmod(0o444)
    tree = root / "cache" / "ab" / "tree"
    (tree / "sub").mkdir(parents=True)
    os.link(shared, tree / "sub" / "object")
    tree.chmod(0o555)
    unreadable = root / "unreadable"
    (unreadable / "deep").mkdir(parents=True)
    unreadable.chmod(0)
    (root / "link").symlink_to(outside, target_is_directory=True)

    conftest.restore_owner_access(root)

    shutil.rmtree(root)
    assert stat.S_IMODE(locked.stat().st_mode) == 0o555, "Should not follow a symlink out"
    assert stat.S_IMODE(shared.stat().st_mode) == 0o444, "Should not chmod a file"
