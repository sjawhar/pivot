from __future__ import annotations

from typing import TYPE_CHECKING

from pivot.cli import completion

if TYPE_CHECKING:
    import pathlib

    import pytest


def test_get_stages_fast_ignores_pivot_yaml_without_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    (tmp_path / ".pivot").mkdir()
    (tmp_path / "pivot.yaml").write_text("stages:\n  train:\n    python: train.py\n")

    monkeypatch.chdir(tmp_path)

    assert completion._get_stages_fast() is None
