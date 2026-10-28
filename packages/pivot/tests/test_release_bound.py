"""The release, not a human, writes pivot-tui's requirement on pivot.

``scripts/set_tui_pivot_bound.py`` runs from ``[tool.semantic_release] build_command``, between
semantic-release's version bump and its release commit, so a published pivot-tui vX requires the
pivot released beside it. A hand-written bound is wrong as soon as another pull request releases
first: ``pivot>=0.5.12`` was true when it was written and false once #522 took 0.5.12 without
``pivot.display``, which gives a user ``ImportError: cannot import name 'display' from 'pivot'``.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import tomllib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "set_tui_pivot_bound.py"
_TUI_PYPROJECT = _REPO_ROOT / "packages" / "pivot-tui" / "pyproject.toml"
_PIVOT_REQUIREMENT = re.compile(r"^pivot(?![\w.\-])")


def _requirement(pyproject: pathlib.Path) -> str:
    dependencies = tomllib.loads(pyproject.read_text("utf-8"))["project"]["dependencies"]
    return next(r for r in dependencies if _PIVOT_REQUIREMENT.match(r))


def _run_release_step(pyproject: pathlib.Path, version: str) -> subprocess.CompletedProcess[str]:
    """Run the script as the release runs it: the version comes from ``NEW_VERSION``."""
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--pyproject", str(pyproject)],
        env={**os.environ, "NEW_VERSION": version},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "previous",
    ['"pivot"', '"pivot>=0.5.11"', '"pivot>=0.5.12"'],
    ids=["no_bound", "released_bound", "future_bound"],
)
def test_the_release_writes_the_version_being_released(
    tmp_path: pathlib.Path, previous: str
) -> None:
    seeded = _TUI_PYPROJECT.read_text("utf-8").replace(
        f'"{_requirement(_TUI_PYPROJECT)}"', previous, 1
    )
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(seeded, encoding="utf-8")

    result = _run_release_step(pyproject, "9.9.9")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _requirement(pyproject) == "pivot>=9.9.9"
    written = pyproject.read_text("utf-8")
    # Only that one requirement changes: the rest of the file is byte-identical.
    assert written.replace('"pivot>=9.9.9"', previous, 1) == seeded


def test_the_release_refuses_a_version_it_was_not_given(tmp_path: pathlib.Path) -> None:
    """``NEW_VERSION`` unset must fail the release, never write an empty or stale bound."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_TUI_PYPROJECT.read_text("utf-8"), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--pyproject", str(pyproject)],
        env={key: value for key, value in os.environ.items() if key != "NEW_VERSION"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0, result.stdout
    assert "NEW_VERSION" in result.stderr
    assert pyproject.read_text("utf-8") == _TUI_PYPROJECT.read_text("utf-8")


def test_the_release_build_command_writes_and_stages_the_bound() -> None:
    """The bound reaches the release commit only if the build command writes it, stages it, and
    does both before the wheels are built from that file."""
    build_command = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text("utf-8"))["tool"][
        "semantic_release"
    ]["build_command"]

    steps = [step.strip() for step in build_command.split("&&")]
    write = next(i for i, step in enumerate(steps) if "scripts/set_tui_pivot_bound.py" in step)
    stage = next(i for i, step in enumerate(steps) if "packages/pivot-tui/pyproject.toml" in step)
    build = next(i for i, step in enumerate(steps) if "uv build" in step)

    assert steps[stage].startswith("git add"), steps[stage]
    assert write < stage < build, build_command
