from __future__ import annotations

import contextlib
import importlib
import linecache
import logging
import multiprocessing as mp
import pathlib
import subprocess
import sys
import tempfile
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, cast

import click.testing
import pytest

from pivot import project
from pivot.config import io as config_io
from pivot.executor import core as executor_core
from pivot.registry import REGISTRY
from pivot.tui import console

# Add tests directory to sys.path so helpers.py can be imported
_tests_dir = pathlib.Path(__file__).parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from pivot.types import OutputMessage, TuiQueue
    from pivot.watch.engine import WatchEngine

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
    mocker.patch.object(REGISTRY, "_cached_dag", None)
    yield


_PIVOT_LOGGERS = ("pivot", "pivot.project", "pivot.executor", "pivot.registry", "")


@pytest.fixture(autouse=True)
def reset_pivot_state(mocker: MockerFixture) -> Generator[None]:
    """Reset global pivot state between tests.

    CliRunner can leave console singleton pointing to closed streams,
    project root cache pointing to old directories, and merged config cached.
    """
    mocker.patch.object(console, "_console", None)
    mocker.patch.object(project, "_project_root_cache", None)
    config_io.clear_config_cache()
    for name in _PIVOT_LOGGERS:
        logging.getLogger(name).handlers.clear()
    yield


@pytest.fixture
def set_project_root(tmp_path: pathlib.Path, mocker: MockerFixture) -> Generator[pathlib.Path]:
    """Set project root to tmp_path for tests that register stages with temp paths."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    yield tmp_path


def init_git_repo(path: pathlib.Path, monkeypatch: pytest.MonkeyPatch | None = None) -> None:
    """Initialize a git repo with user config at the given path.

    This is a helper function for tests that need git in a path they control.
    For tests that just need a standard git repo, use the `git_repo` fixture.

    Args:
        path: Directory to initialize as a git repository.
        monkeypatch: Optional MonkeyPatch to set GIT_CONFIG_GLOBAL, avoiding 2 subprocess calls.
    """
    # Resolve to absolute path to avoid issues with cwd changes
    abs_path = path.resolve()
    if monkeypatch is not None:
        # Use GIT_CONFIG_GLOBAL to avoid per-repo config subprocess calls
        # Place config in parent directory to avoid it being included in git commits
        config_file = abs_path.parent / f".gitconfig_test_{abs_path.name}"
        config_file.write_text("[user]\n\temail = test@test.com\n\tname = Test\n")
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config_file))
        subprocess.run(["git", "init"], cwd=abs_path, check=True, capture_output=True)
    else:
        # Fallback: configure per-repo (3 subprocess calls)
        subprocess.run(["git", "init"], cwd=abs_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=abs_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"], cwd=abs_path, check=True, capture_output=True
        )


@contextlib.contextmanager
def stage_module_isolation(path: pathlib.Path) -> Generator[None]:
    """Add path to sys.path and ensure 'stages' module is freshly imported.

    Cleans up sys.path and removes cached 'stages' module on exit to prevent
    test pollution. Use this in fixtures that need to import a stages.py file
    from a test-specific directory.

    Example:
        @pytest.fixture
        def my_pipeline(tmp_path: Path, mocker: MockerFixture) -> Generator[Path]:
            # ... setup ...
            mocker.patch.object(project, "_project_root_cache", tmp_path)
            with stage_module_isolation(tmp_path):
                yield tmp_path
    """
    if "stages" in sys.modules:
        del sys.modules["stages"]
    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        # Guard against ValueError if path was removed by other code
        if path_str in sys.path:
            sys.path.remove(path_str)
        if "stages" in sys.modules:
            del sys.modules["stages"]


@pytest.fixture
def git_repo(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> GitRepo:
    """Create a git repo in tmp_path, return (path, commit_fn).

    Uses GIT_CONFIG_GLOBAL to configure user settings with a single subprocess call
    instead of 3, improving test performance.
    """
    init_git_repo(tmp_path, monkeypatch)

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
def clear_source_caches() -> Generator[None]:
    """Clear linecache and importlib caches at session start.

    This prevents stale bytecode/source cache from causing fingerprinting tests
    to fail when source files have changed. inspect.getsource() relies on
    linecache, which can return outdated content if not cleared.
    """
    linecache.clearcache()
    importlib.invalidate_caches()
    yield


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


@pytest.fixture
def pipeline_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Set up a temporary pipeline directory with .pivot marker.

    Creates `.pivot` directory but NOT pivot.yaml, allowing tests to register
    stages programmatically without auto-discovery. Tests that need pivot.yaml
    (e.g., those using CLI commands that trigger auto-discovery) should define
    a local fixture override that creates pivot.yaml.
    """
    (tmp_path / ".pivot").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project, "_project_root_cache", None)
    return tmp_path


@pytest.fixture
def runner() -> click.testing.CliRunner:
    """Create a CLI runner for testing."""
    return click.testing.CliRunner()


@pytest.fixture
def output_queue() -> Generator[mp.Queue[OutputMessage]]:
    """Create a multiprocessing queue for worker output using spawn context.

    Uses spawn context to match production behavior and avoid Python 3.13+
    deprecation warnings about fork() in multi-threaded contexts.
    """
    spawn_ctx = mp.get_context("spawn")
    manager = spawn_ctx.Manager()
    # Manager().Queue() returns Queue[Any] - cast through object for type safety
    queue = cast("mp.Queue[OutputMessage]", cast("object", manager.Queue()))
    yield queue
    manager.shutdown()


# =============================================================================
# TUI Test Mocks
# =============================================================================


class MockWatchEngine:
    """Mock watch engine for TUI testing."""

    def __init__(self) -> None:
        self._keep_going: bool = False

    def run(
        self,
        tui_queue: TuiQueue | None = None,
        output_queue: mp.Queue[OutputMessage] | None = None,
    ) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def toggle_keep_going(self) -> bool:
        self._keep_going = not self._keep_going
        return self._keep_going

    @property
    def keep_going(self) -> bool:
        return self._keep_going


@pytest.fixture
def mock_watch_engine() -> WatchEngine:
    """Provide a mock watch engine for TUI testing."""
    return MockWatchEngine()  # pyright: ignore[reportReturnType]


@pytest.fixture
def worker_env(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Set up worker execution environment with cache and stages directories."""
    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "files").mkdir()
    (tmp_path / ".pivot" / "stages").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".pivot" / "pending" / "stages").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return cache_dir
