"""Watch-path wiring tests for ``pivot repro``."""

from __future__ import annotations

import pathlib

import pytest

from pivot.cli import helpers as cli_helpers
from pivot.cli import repro
from pivot.compose import Pipeline, stage
from pivot.engine import sources as engine_sources
from pivot.types import OnError


@stage
def _helper_watch_output() -> dict[str, int]:
    return {"value": 1}


def test_watch_paths_include_project_root(mock_discovery: Pipeline) -> None:
    """Initial watch wiring covers pipeline.py and imported project modules."""
    with mock_discovery:
        _helper_watch_output()

    paths = repro._watch_paths_from_registry(cli_helpers.get_all_stages())

    assert paths == [mock_discovery.root]


class _SourceRecorder:
    added: list[object]

    def __init__(self) -> None:
        self.added = list[object]()

    def add_source(self, source: object) -> None:
        self.added.append(source)


@pytest.mark.parametrize("force", [False, True], ids=["unforced", "forced"])
def test_watch_sources_request_startup_run(force: bool) -> None:
    """Watch mode runs the pipeline once at startup; skip detection keeps cached stages cached."""
    recorder = _SourceRecorder()

    repro._configure_watch_sources(
        recorder,  # pyright: ignore[reportArgumentType] - records add_source calls only
        [pathlib.Path("/project")],
        300,
        force=force,
        stages=["demo/clean"],
        resource_capacity={},
        no_commit=False,
        on_error=OnError.FAIL,
    )

    initial_runs = [s for s in recorder.added if isinstance(s, engine_sources.OneShotSource)]
    assert len(initial_runs) == 1, "watch mode must request one startup run"
    request = initial_runs[0]._event
    assert request["force"] is force
    assert request["stages"] == ["demo/clean"]
