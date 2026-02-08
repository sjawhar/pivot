from __future__ import annotations

import pytest

from pivot.cli import helpers as cli_helpers


@pytest.fixture(autouse=True)
def _mock_get_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock cli_helpers.get_stage to return all-cached outputs.

    Sync functions filter non-cached outputs via registry; tests without a
    pipeline can return empty outs to allow hashes through.
    """

    def _get_stage(name: str) -> dict[str, object]:
        return {"outs": []}

    monkeypatch.setattr(cli_helpers, "get_stage", _get_stage)
