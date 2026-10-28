"""Each in-process CLI invocation that resolves the state directory reports an ignored global
`core.state_dir` once, whatever earlier invocations in the same process did.

The resolved state directory lives as long as one invocation, as the warning record does, so a
repeated command, or a command after a `-q` one, resolves the state directory again and warns
again; `-q` silences only its own invocation. `test_state_dir_warnings_per_invocation.py` covers
`config get core.state_dir`, which warns without resolving the state directory.
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

_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


@pytest.fixture
def project_with_global_state_dir(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> pathlib.Path:
    """A project with a pipeline, under a HOME whose global config sets core.state_dir."""
    home = tmp_path / "home"
    (home / ".config" / "pivot").mkdir(parents=True)
    (home / ".config" / "pivot" / "config.yaml").write_text("core:\n  state_dir: global-state\n")
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".pivot").mkdir()
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(root)
    project._project_root_cache = None
    return root


def _notices_per_invocation(
    capsys: pytest.CaptureFixture[str], steps: Sequence[Sequence[str]]
) -> list[int]:
    """Run each step in this process; count the ignored-global notices each writes to stderr."""
    counts = list[int]()
    for step in steps:
        _ = capsys.readouterr()
        cli.cli.main(list(step), standalone_mode=False)
        counts.append(capsys.readouterr().err.count(_IGNORED))
    return counts


@pytest.mark.usefixtures("project_with_global_state_dir")
@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        pytest.param((("list",), ("list",)), [1, 1], id="list-then-list"),
        pytest.param((("status",), ("status",)), [1, 1], id="status-then-status"),
        pytest.param((("list",), ("status",)), [1, 1], id="list-then-status"),
        pytest.param((("-q", "list"), ("list",)), [0, 1], id="quiet-list-then-list"),
        pytest.param((("list",), ("-q", "list")), [1, 0], id="list-then-quiet-list"),
    ],
)
def test_each_invocation_that_uses_state_warns_once(
    capsys: pytest.CaptureFixture[str], steps: Sequence[Sequence[str]], expected: list[int]
) -> None:
    assert _notices_per_invocation(capsys, steps) == expected
