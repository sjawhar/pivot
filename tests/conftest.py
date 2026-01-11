from __future__ import annotations

import logging
import pathlib
import subprocess
import tempfile
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from pivot import project
from pivot.executor import core as executor_core
from pivot.registry import REGISTRY
from pivot.tui import console

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


@pytest.fixture(scope="session", autouse=True)
def cleanup_worker_pool() -> Generator[None]:
    """Kill loky worker pool at end of test session to prevent orphaned workers."""
    yield
    executor_core._cleanup_worker_pool()


# Type alias for make_valid_lock_content fixture
ValidLockContentFactory = Callable[..., dict[str, object]]


@pytest.fixture
def make_valid_lock_content() -> ValidLockContentFactory:
    """Factory fixture for creating valid lock file data with all required fields."""

    def _factory(
        code_manifest: dict[str, str] | None = None,
        params: dict[str, object] | None = None,
        deps: list[dict[str, object]] | None = None,
        outs: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return {
            "code_manifest": code_manifest or {},
            "params": params or {},
            "deps": deps or [],
            "outs": outs or [],
            "dep_generations": {},
        }

    return _factory
