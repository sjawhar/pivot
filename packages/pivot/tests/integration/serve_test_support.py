"""Shared setup and observations for serve integration tests."""

from __future__ import annotations

import pathlib

_NOOP_PIPELINE = """\
import pathlib

from pivot.compose import Pipeline, stage


pipeline = Pipeline("test", root=pathlib.Path(__file__).parent)


@stage
def noop() -> None:
    pass


with pipeline:
    noop()
"""


def create_noop_project(project: pathlib.Path) -> None:
    """Create a project with a pipeline containing one no-op stage."""
    (project / ".pivot").mkdir(parents=True)
    (project / ".git").mkdir()
    (project / "pipeline.py").write_text(_NOOP_PIPELINE)


def process_signal_disposition(pid: int, signum: int) -> str:
    """Return whether the kernel reports ``signum`` ignored, caught, or at its default."""
    masks = {
        key: int(value, 16)
        for key, _, value in (
            line.partition(":")
            for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines()
        )
        if key in ("SigIgn", "SigCgt")
    }
    bit = 1 << (signum - 1)
    return "ignored" if masks["SigIgn"] & bit else "caught" if masks["SigCgt"] & bit else "default"
