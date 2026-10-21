"""R7 defect G3: a local config that is not UTF-8 stops listing, outside the 06:42Z warn-first path.

`.pivot/config.yaml` holding one non-UTF-8 byte (here a Latin-1 `é` in a comment) cannot be read by
the config loader: ruamel raises `UnicodeDecodeError`, which is not a `ConfigError`. 0.5.2 runs
`list`, `list --json`, `list --all`, `dag`, `plots show` and `init --force` (exit 0) and lists the
stage in shell completion, because none of them read the config. At ce74ad41 they exit 1 with
`Failed to load pipeline.py: 'utf-8' codec can't decode byte 0xe9 ...` (TAB completion returns
nothing): `project.state_dir_if_readable` catches only `ConfigError`. The global-config path
(`warn_if_global_state_dir_ignored`) already catches `UnicodeDecodeError`, and with a non-UTF-8
global file every command matches 0.5.2.

The 06:42Z ruling covers an unreadable local config: `list` and `dag` warn once on stderr and exit 0.
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

_UNKNOWN = "The state directory is unknown"


def _run(
    cwd: pathlib.Path, home: pathlib.Path, *args: str, extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HOME": str(home), **(extra or {})}
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, errors="replace", check=False, env=env
    )


def _project(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
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
    (root / ".pivot" / "config.yaml").write_bytes(b"core:\n  max_workers: 2\n# caf\xe9\n")
    return home, root


@pytest.mark.parametrize(
    "args",
    [("list",), ("list", "--all"), ("dag",), ("plots", "show"), ("init", "--force")],
    ids=lambda args: "-".join(args),
)
def test_commands_that_never_read_state_still_run(
    tmp_path: pathlib.Path, args: tuple[str, ...]
) -> None:
    """Like 0.5.2, these exit 0 without a traceback; `list` and `dag` still show the stage."""
    home, root = _project(tmp_path)

    result = _run(root, home, _PIVOT, *args)

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output
    if args[0] in {"list", "dag"}:
        assert "test/process" in result.stdout, output


@pytest.mark.parametrize("args", [("list",), ("dag",)], ids=lambda args: "-".join(args))
def test_list_and_dag_warn_once_on_stderr(tmp_path: pathlib.Path, args: tuple[str, ...]) -> None:
    """The 06:42Z warn-first rule: one warning on stderr naming the unknown state directory."""
    home, root = _project(tmp_path)

    result = _run(root, home, _PIVOT, *args)

    assert result.stderr.count(_UNKNOWN) == 1, result.stderr
    assert _UNKNOWN not in result.stdout, result.stdout


def test_list_json_still_parses(tmp_path: pathlib.Path) -> None:
    """Like 0.5.2, `list --json` exits 0 and stdout is the stage list."""
    home, root = _project(tmp_path)

    result = _run(root, home, _PIVOT, "list", "--json")

    assert result.returncode == 0, result.stdout + result.stderr
    assert [stage["name"] for stage in json.loads(result.stdout)["stages"]] == ["test/process"]


def test_tab_completion_still_lists_stages(tmp_path: pathlib.Path) -> None:
    """Like 0.5.2, `pivot run <TAB>` offers the stage."""
    home, root = _project(tmp_path)

    result = _run(
        root,
        home,
        _PIVOT,
        extra={"_PIVOT_COMPLETE": "bash_complete", "COMP_WORDS": "pivot run ", "COMP_CWORD": "2"},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "plain,test/process" in result.stdout.splitlines(), result.stdout + result.stderr


def test_history_still_fails(tmp_path: pathlib.Path) -> None:
    """Control, passes on 0.5.2 and at ce74ad41: a state command exits 1 with one `Error:` line."""
    home, root = _project(tmp_path)

    result = _run(root, home, _PIVOT, "history")

    assert result.returncode == 1, result.stdout + result.stderr
    assert any(line.startswith("Error") for line in result.stderr.splitlines()), result.stderr
