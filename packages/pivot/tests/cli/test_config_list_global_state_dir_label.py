"""Inside a project, `config list --global` labels every row of the global `core.state_dir`
`(global, ignored)`, as `config get --global core.state_dir` does.

Only the project's own `.pivot/config.yaml` sets the state directory, so the global value has no
effect, whatever its type. A mapping flattens to one row per nested key (`core.state_dir.x`), and
the label comes from where each row came from in the file, not from the row's key text. The
number and string values are controls. The merged `config list` shows the ignored global value
in one row with the same label, and `--json` keeps 0.5.2's shape, with no source labels.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tests import helpers

if TYPE_CHECKING:
    import pathlib


@pytest.mark.parametrize(
    "value", ["{x: 1}", "5", "altstate"], ids=["mapping", "integer-control", "string-control"]
)
def test_config_list_global_labels_every_global_state_dir_row_ignored(
    tmp_path: pathlib.Path, value: str
) -> None:
    home = tmp_path / "home"
    (home / ".config" / "pivot").mkdir(parents=True)
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (home / ".config" / "pivot" / "config.yaml").write_text(f"core:\n  state_dir: {value}\n")

    shown = helpers.run_pivot(root, home, "config", "get", "--global", "core.state_dir")
    listed = helpers.run_pivot(root, home, "config", "list", "--global")

    # `config get --global` already labels the value ignored: same project, same global file.
    assert shown.returncode == 0, shown.stdout + shown.stderr
    assert shown.stdout.rstrip().endswith("(global, ignored)"), shown.stdout
    assert listed.returncode == 0, listed.stdout + listed.stderr
    rows = [line for line in listed.stdout.splitlines() if line.startswith("core.state_dir")]
    assert rows, listed.stdout
    assert all(row.endswith("(global, ignored)") for row in rows), listed.stdout


@pytest.mark.parametrize(
    "value", ["{x: 1}", "[a, b]", "5", "globalstate"], ids=["mapping", "list", "number", "string"]
)
def test_merged_config_list_labels_a_global_state_dir_of_any_type_ignored(
    tmp_path: pathlib.Path, value: str
) -> None:
    home = tmp_path / "home"
    (home / ".config" / "pivot").mkdir(parents=True)
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (home / ".config" / "pivot" / "config.yaml").write_text(f"core:\n  state_dir: {value}\n")

    listed = helpers.run_pivot(root, home, "config", "list")
    as_json = helpers.run_pivot(root, home, "config", "list", "--json")

    assert listed.returncode == 0, listed.stdout + listed.stderr
    rows = [line for line in listed.stdout.splitlines() if line.startswith("core.state_dir")]
    assert len(rows) == 1, listed.stdout
    assert "(global, ignored)" in rows[0], listed.stdout
    assert as_json.returncode == 0, as_json.stdout + as_json.stderr
    assert "ignored" not in as_json.stdout, as_json.stdout
    assert isinstance(json.loads(as_json.stdout), dict)
