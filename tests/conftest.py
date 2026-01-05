from __future__ import annotations

import logging
import pathlib
import tempfile
from typing import TYPE_CHECKING

import pytest

from pivot import console, project
from pivot.registry import REGISTRY

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytest_mock import MockerFixture


@pytest.fixture
def tmp_pipeline_dir() -> Generator[pathlib.Path]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield pathlib.Path(tmpdir)


@pytest.fixture
def sample_data_file(tmp_pipeline_dir: pathlib.Path) -> pathlib.Path:
    data_file = tmp_pipeline_dir / "data.csv"
    data_file.write_text("id,value\n1,10\n2,20\n3,30\n")
    return data_file


@pytest.fixture(autouse=True)
def clean_registry(mocker: MockerFixture) -> Generator[None]:
    mocker.patch.dict(REGISTRY._stages, clear=True)
    yield


_PIVOT_LOGGERS = ("pivot", "pivot.project", "pivot.executor", "pivot.registry", "")


def _reset_pivot_globals() -> None:
    """Reset console singleton, project root cache, and logging handlers."""
    console._console = None
    project._project_root_cache = None
    for name in _PIVOT_LOGGERS:
        logging.getLogger(name).handlers.clear()


@pytest.fixture(autouse=True)
def reset_pivot_state() -> Generator[None]:
    """Reset global pivot state between tests.

    CliRunner can leave console singleton pointing to closed streams,
    and project root cache pointing to old directories.
    """
    _reset_pivot_globals()
    yield
    _reset_pivot_globals()
