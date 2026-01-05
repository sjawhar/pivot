from __future__ import annotations

import pathlib
import tempfile
from typing import TYPE_CHECKING

import pytest

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
