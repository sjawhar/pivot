from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pivot import project

if TYPE_CHECKING:
    import pathlib


@pytest.fixture
def pipeline_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Set up a temporary pipeline directory with project markers.

    Creates both `.pivot` directory and `pivot.yaml` for compatibility with
    all reactive tests. The global autouse fixtures (clean_registry, reset_pivot_state)
    handle clearing registry and project root cache.
    """
    (tmp_path / ".pivot").mkdir()
    (tmp_path / "pivot.yaml").write_text("version: 1\n")
    monkeypatch.chdir(tmp_path)
    # Explicitly reset project root cache since we just created project markers
    monkeypatch.setattr(project, "_project_root_cache", None)
    return tmp_path
