"""State-directory warnings are counted per CLI invocation, not per process.

Embedders drive the CLI in one process (`pivot.cli.main`, `tests/integration/inproc_serve.py`).
Each invocation that uses the state directory prints its warning once, and `-q` silences it, no
matter what an earlier invocation in the same process did.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from pivot import cli, project

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Sequence

_IGNORED = "is ignored; set it in"


@pytest.fixture
def project_with_global_state_dir(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> pathlib.Path:
    home = tmp_path / "home"
    (home / ".config" / "pivot").mkdir(parents=True)
    (home / ".config" / "pivot" / "config.yaml").write_text("core:\n  state_dir: global-state\n")
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".pivot").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(root)
    project._project_root_cache = None
    return root


def _warnings_per_invocation(
    capsys: pytest.CaptureFixture[str], steps: Sequence[Sequence[str]]
) -> list[int]:
    """Run each step in this process and count the ignored-global notices it writes to stderr.

    The CLI's log handler writes to the current ``sys.stderr`` (``cli._StderrHandler``), which
    capsys replaces, so each step's notices land in its own capture.
    """
    counts = list[int]()
    for step in steps:
        _ = capsys.readouterr()
        cli.cli.main(list(step), standalone_mode=False)
        counts.append(capsys.readouterr().err.count(_IGNORED))
    return counts


@pytest.mark.usefixtures("project_with_global_state_dir")
def test_each_invocation_warns_once(capsys: pytest.CaptureFixture[str]) -> None:
    steps = [("config", "get", "core.state_dir"), ("config", "get", "core.state_dir")]

    assert _warnings_per_invocation(capsys, steps) == [1, 1]


@pytest.mark.usefixtures("project_with_global_state_dir")
def test_quiet_silences_a_later_invocation(capsys: pytest.CaptureFixture[str]) -> None:
    steps = [("config", "get", "core.max_workers"), ("-q", "config", "get", "core.state_dir")]

    assert _warnings_per_invocation(capsys, steps) == [0, 0]
