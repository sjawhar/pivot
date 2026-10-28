"""`config list --global` and `--local` print every stored row of a config file whose mapping
keys YAML does not read as strings, and exit 0.

A scoped listing shows the file as it stands, so Pivot renders each segment of a key path as
text: YAML reads `1`, `true` and `null` as an integer, a boolean and None, and the loader keeps
them, so the row for `{1: x}` under `core.state_dir` reads `core.state_dir.1 = x`. The path keeps
the stored segments, so a global row still carries `(global, ignored)` when it comes from the
global `core.state_dir` entry, whatever the type of its keys. The string-keyed mapping is the
control. Pivot 0.5.2, main 825fa474 and 71a5fa21 all exit 0 here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests import helpers

if TYPE_CHECKING:
    import pathlib

_CASES = {
    "state_dir-int-key": ("core:\n  state_dir: {1: x}\n", "core.state_dir.1 = x"),
    "state_dir-bool-key": ("core:\n  state_dir: {true: x}\n", "core.state_dir.True = x"),
    "state_dir-null-key": ("core:\n  state_dir: {null: x}\n", "core.state_dir.None = x"),
    "other-int-key": ("extra:\n  2: x\n", "extra.2 = x"),
    "string-key-control": ("core:\n  state_dir: {x: 1}\n", "core.state_dir.x = 1"),
}


def _project(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    home = tmp_path / "home"
    (home / ".config" / "pivot").mkdir(parents=True)
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    return home, root


@pytest.mark.parametrize("case", list(_CASES))
def test_config_list_global_shows_non_string_mapping_keys(
    tmp_path: pathlib.Path, case: str
) -> None:
    text, row_start = _CASES[case]
    home, root = _project(tmp_path)
    (home / ".config" / "pivot" / "config.yaml").write_text(text)

    listed = helpers.run_pivot(root, home, "config", "list", "--global")

    assert listed.returncode == 0, listed.stdout + listed.stderr
    rows = [line for line in listed.stdout.splitlines() if line.startswith(row_start)]
    assert len(rows) == 1, listed.stdout + listed.stderr
    if row_start.startswith("core.state_dir"):
        assert rows[0].endswith("(global, ignored)"), listed.stdout
    else:
        assert rows[0].endswith("(global)"), listed.stdout


@pytest.mark.parametrize("case", list(_CASES))
def test_config_list_local_shows_non_string_mapping_keys(tmp_path: pathlib.Path, case: str) -> None:
    text, row_start = _CASES[case]
    home, root = _project(tmp_path)
    (root / ".pivot" / "config.yaml").write_text(text)

    listed = helpers.run_pivot(root, home, "config", "list", "--local")

    assert listed.returncode == 0, listed.stdout + listed.stderr
    rows = [line for line in listed.stdout.splitlines() if line.startswith(row_start)]
    assert len(rows) == 1, listed.stdout + listed.stderr
