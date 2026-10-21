"""R5 defect 4: with a global core.state_dir, commands that never load the merged config print
no warning, so `history`/`show` silently read an empty .pivot/state.lmdb after an upgrade."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import typing

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


def _run(cwd: pathlib.Path, home: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HOME": str(home)}
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False, env=env)


def _project(tmp_path: pathlib.Path, value: str) -> tuple[pathlib.Path, pathlib.Path, str]:
    home = tmp_path / "home"
    global_config = home / ".config" / "pivot" / "config.yaml"
    global_config.parent.mkdir(parents=True)
    global_config.write_text(f"core:\n  state_dir: {value}\n")
    root = tmp_path / "project"
    root.mkdir()
    assert _run(root, home, "git", "init", "-q").returncode == 0
    assert _run(root, home, _PIVOT, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    repro = _run(root, home, _PIVOT, "repro")
    assert repro.returncode == 0, repro.stdout + repro.stderr
    warning = (
        f"core.state_dir in {global_config} is ignored; set it in {root / '.pivot' / 'config.yaml'}"
    )
    return home, root, warning


def _case_id(value: object) -> str | None:
    """Name a case by its command words; the boolean column takes pytest's default id."""
    if isinstance(value, tuple):
        return "-".join(str(word) for word in typing.cast("tuple[object, ...]", value))
    return None


@pytest.mark.parametrize("spelling", ["relative", "absolute"])
@pytest.mark.parametrize(
    ("args", "uses_state"),
    [
        # Control, warns at 392187f8: status loads the merged config.
        (("status",), True),
        # Read the state database whose location changed for global-value users.
        (("history",), True),
        (("show",), True),
        (("fingerprint", "reset"), True),
        # Other project commands that print nothing about the ignored value.
        (("init", "--force"), True),
        (("list",), True),
        (("dag",), True),
        (("plots", "show"), True),
        # PR503_ROUND10 L2 ruling: only commands that use the state directory warn. These
        # read remotes, imports or another key, never state, so they stay quiet.
        (("remote", "list"), False),
        (("update", "--dry-run"), False),
        (("config", "set", "core.max_workers", "2"), False),
        (("config", "unset", "core.max_workers"), False),
    ],
    ids=_case_id,
)
def test_every_command_warns_once_about_an_ignored_global_state_dir(
    tmp_path: pathlib.Path, spelling: str, args: tuple[str, ...], uses_state: bool
) -> None:
    """Every command that uses the state directory prints the one-line warning exactly once."""
    value = "globalstate" if spelling == "relative" else str(tmp_path / "shared-state")
    home, root, warning = _project(tmp_path, value)

    result = _run(root, home, _PIVOT, *args)

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.stderr.splitlines().count(warning) == (1 if uses_state else 0), result.stderr
