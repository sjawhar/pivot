"""R7 defect G2: a readable local config whose `core.state_dir` is not a path stops listing.

With a hand-edited `core.state_dir` of `''`, `5`, `[a, b]` or `{x: 1}`, the file itself is valid YAML.
0.5.2 runs `list`, `list --json`, `dag`, `plots show` and `init --force` (exit 0) and lists the stage
in shell completion, because none of them read `core.state_dir`. At ce74ad41 each exits 1 with
`core.state_dir in <file> must be a path`, and TAB completion returns nothing:
`project.state_dir_if_readable` (the 06:42Z warn-first path) catches only a config that cannot be
parsed; `_resolve_configured_state_dir` raises for an unusable value and nothing catches it in
`compose._stage_state_dir`, `discovery.glob_all_pipelines` or `cli/init.py`.

The rollout rule the 06:42Z ruling applied to an unreadable file (warn, keep listing, exit 0) fits
this case the same way: listing never needs the state directory. `status` is the control: it fails
with one `Error:` line on the head, as `repro` and `status` fail on 0.5.2 for the non-string values.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""

_VALUES = {"empty": "''", "integer": "5", "list": "[a, b]", "mapping": "{x: 1}"}


def _run(
    cwd: pathlib.Path, home: pathlib.Path, *args: str, extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HOME": str(home), **(extra or {})}
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False, env=env)


def _project(tmp_path: pathlib.Path, literal: str) -> tuple[pathlib.Path, pathlib.Path]:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    assert _run(root, home, "git", "init", "-q").returncode == 0
    assert _run(root, home, _PIVOT, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    assert _run(root, home, _PIVOT, "repro").returncode == 0
    (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {literal}\n")
    return home, root


@pytest.mark.parametrize("value", list(_VALUES), ids=list(_VALUES))
@pytest.mark.parametrize(
    "args",
    [("list",), ("dag",), ("plots", "show"), ("init", "--force")],
    ids=lambda args: "-".join(args),
)
def test_commands_that_never_read_state_still_run(
    tmp_path: pathlib.Path, value: str, args: tuple[str, ...]
) -> None:
    """Like 0.5.2, these exit 0 without a traceback."""
    home, root = _project(tmp_path, _VALUES[value])

    result = _run(root, home, _PIVOT, *args)

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output
    if args in {("list",), ("dag",)}:
        assert "test/process" in result.stdout, output


@pytest.mark.parametrize("value", list(_VALUES), ids=list(_VALUES))
def test_list_json_still_parses(tmp_path: pathlib.Path, value: str) -> None:
    """Like 0.5.2, `list --json` exits 0 and stdout is the stage list."""
    home, root = _project(tmp_path, _VALUES[value])

    result = _run(root, home, _PIVOT, "list", "--json")

    assert result.returncode == 0, result.stdout + result.stderr
    assert [stage["name"] for stage in json.loads(result.stdout)["stages"]] == ["test/process"]


@pytest.mark.parametrize("value", list(_VALUES), ids=list(_VALUES))
def test_tab_completion_still_lists_stages(tmp_path: pathlib.Path, value: str) -> None:
    """Like 0.5.2, `pivot run <TAB>` offers the stage."""
    home, root = _project(tmp_path, _VALUES[value])

    result = _run(
        root,
        home,
        _PIVOT,
        extra={"_PIVOT_COMPLETE": "bash_complete", "COMP_WORDS": "pivot run ", "COMP_CWORD": "2"},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "plain,test/process" in result.stdout.splitlines(), result.stdout + result.stderr


@pytest.mark.parametrize("value", list(_VALUES), ids=list(_VALUES))
def test_status_still_refuses_in_one_line(tmp_path: pathlib.Path, value: str) -> None:
    """A state command keeps the 0.5.2 exit code without a traceback."""
    home, root = _project(tmp_path, _VALUES[value])

    result = _run(root, home, _PIVOT, "status")

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    expected = 0 if value == "empty" else 1
    assert result.returncode == expected, output
    if expected != 0:
        assert any(line.startswith("Error") for line in result.stderr.splitlines()), output
