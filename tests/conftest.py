from __future__ import annotations

import logging
import pathlib
import subprocess
import tempfile
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from pivot import config, console, project
from pivot.registry import REGISTRY

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytest_mock import MockerFixture

# Type alias for git_repo fixture: (repo_path, commit_fn)
GitRepo = tuple[pathlib.Path, Callable[[str], str]]


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


@pytest.fixture(autouse=True)
def reset_pivot_state(mocker: MockerFixture) -> Generator[None]:
    """Reset global pivot state between tests.

    CliRunner can leave console singleton pointing to closed streams,
    and project root cache pointing to old directories.
    """
    mocker.patch.object(config, "_config_cache", None)
    mocker.patch.object(console, "_console", None)
    mocker.patch.object(project, "_project_root_cache", None)
    for name in _PIVOT_LOGGERS:
        logging.getLogger(name).handlers.clear()
    yield


@pytest.fixture
def set_project_root(tmp_path: pathlib.Path, mocker: MockerFixture) -> Generator[pathlib.Path]:
    """Set project root to tmp_path for tests that register stages with temp paths."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    yield tmp_path


@pytest.fixture
def git_repo(tmp_path: pathlib.Path) -> GitRepo:
    """Create a git repo in tmp_path, return (path, commit_fn)."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True
    )

    def commit(message: str) -> str:
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", message], cwd=tmp_path, check=True, capture_output=True
        )
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    return tmp_path, commit
