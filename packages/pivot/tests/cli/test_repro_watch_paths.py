"""Watch-path wiring tests for ``pivot repro``."""

from __future__ import annotations

from pivot.cli import helpers as cli_helpers
from pivot.cli import repro
from pivot.compose import Pipeline, stage


@stage
def _helper_watch_output() -> dict[str, int]:
    return {"value": 1}


def test_watch_paths_include_project_root(mock_discovery: Pipeline) -> None:
    """Initial watch wiring covers pipeline.py and imported project modules."""
    with mock_discovery:
        _helper_watch_output()

    paths = repro._watch_paths_from_registry(cli_helpers.get_all_stages())

    assert paths == [mock_discovery.root]
