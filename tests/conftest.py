from __future__ import annotations

import contextlib
import importlib
import linecache
import logging
import multiprocessing as mp
import os
import pathlib
import subprocess
import sys
import tempfile
from collections.abc import AsyncGenerator, Callable, Generator
from typing import TYPE_CHECKING

import click.testing
import pytest

from pivot import project
from pivot.cli import console
from pivot.config import io as config_io
from pivot.executor import core as executor_core
from pivot.pipeline import pipeline as pipeline_mod
from pivot.registry import StageRegistry

# Add tests directory to sys.path so helpers.py can be imported
_tests_dir = pathlib.Path(__file__).parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from pivot.engine.engine import Engine
    from pivot.types import OutputMessage

# Type alias for git_repo fixture: (repo_path, commit_fn)
GitRepo = tuple[pathlib.Path, Callable[[str], str]]

_MEMORY_PER_WORKER_BYTES = 2 * 1024**3
_MAX_WORKERS_CAP = 16


@pytest.hookimpl(tryfirst=True, optionalhook=True)
def pytest_xdist_auto_num_workers(config: pytest.Config) -> int:
    """Cap ``-n auto`` based on cgroup memory, not CPU count.

    Each xdist worker spawns loky workers + a multiprocessing Manager,
    so ~2 GB per worker is needed. Falls back to min(cpu_count, 8)
    outside containers, capped at 16 workers.
    """
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count() or 1
    limit_bytes = _get_cgroup_memory_limit_bytes()
    if limit_bytes is not None:
        workers = min(limit_bytes // _MEMORY_PER_WORKER_BYTES, cpu_count)
    else:
        workers = min(cpu_count, 8)
    return max(1, min(workers, _MAX_WORKERS_CAP))


def _get_cgroup_memory_limit_bytes() -> int | None:
    try:
        v2 = pathlib.Path("/sys/fs/cgroup/memory.max")
        if v2.exists():
            text = v2.read_text().strip()
            if text != "max":
                return int(text)

        v1 = pathlib.Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if v1.exists():
            limit = int(v1.read_text().strip())
            if limit < (1 << 60):
                return limit
    except (OSError, ValueError):
        pass

    return None


@pytest.fixture
def tmp_pipeline_dir() -> Generator[pathlib.Path]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield pathlib.Path(tmpdir)


@pytest.fixture
def sample_data_file(tmp_pipeline_dir: pathlib.Path) -> pathlib.Path:
    data_file = tmp_pipeline_dir / "data.csv"
    data_file.write_text("id,value\n1,10\n2,20\n3,30\n")
    return data_file


@pytest.fixture
def test_pipeline(tmp_path: pathlib.Path) -> Generator[pipeline_mod.Pipeline]:
    """Provide a fresh Pipeline for tests.

    Also sets up the module-level test pipeline in helpers.py so that
    register_test_stage() works without explicit pipeline parameter.

    Note: This fixture does NOT mock the project root. Tests that register
    stages with path annotations should either:
    1. Use mock_discovery fixture (which mocks project root)
    2. Or explicitly mock project._project_root_cache themselves
    """
    import helpers

    pipeline = pipeline_mod.Pipeline("test", root=tmp_path)
    helpers.set_test_pipeline(pipeline)
    yield pipeline
    helpers.set_test_pipeline(None)


@pytest.fixture
def mock_discovery(
    test_pipeline: pipeline_mod.Pipeline,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> pipeline_mod.Pipeline:
    """Mock discover_pipeline to return the test_pipeline.

    Use this fixture for CLI tests that need stages to be discovered
    without creating actual pivot.yaml or pipeline.py files.

    This fixture:
    - Creates .pivot and .git directories in the pipeline root
    - Changes cwd to the pipeline root
    - Sets project._project_root_cache to the pipeline root
    - Mocks discover_pipeline to return test_pipeline
    - Mocks get_pipeline_from_context for cli_helpers

    Tests using this fixture should NOT use isolated_filesystem() since
    this fixture already sets up the environment correctly.

    Note: The test_pipeline fixture is automatically used, so stages
    can be registered via register_test_stage().
    """
    from pivot import discovery
    from pivot.cli import decorators as cli_decorators

    # Set up filesystem environment
    (test_pipeline.root / ".pivot").mkdir(exist_ok=True)
    (test_pipeline.root / ".git").mkdir(exist_ok=True)
    monkeypatch.chdir(test_pipeline.root)

    # Mock discovery
    mocker.patch.object(discovery, "discover_pipeline", return_value=test_pipeline)
    mocker.patch.object(project, "_project_root_cache", test_pipeline.root)
    mocker.patch.object(cli_decorators, "get_pipeline_from_context", return_value=test_pipeline)
    return test_pipeline


@pytest.fixture
def test_registry() -> StageRegistry:
    """Provide a fresh StageRegistry for tests that need direct registry access."""
    return StageRegistry()


@pytest.fixture
def clean_registry() -> None:
    """No-op fixture for backwards compatibility.

    Previously cleared the global REGISTRY between tests.
    Now that REGISTRY is removed, this is kept for tests that still use
    @pytest.mark.usefixtures("clean_registry") - they can be gradually updated.
    """
    pass


_PIVOT_LOGGERS = ("pivot", "pivot.project", "pivot.executor", "pivot.registry", "")


@pytest.fixture(scope="session")
def _default_project_root(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Session-scoped default project root with .pivot directory.

    Provides a fallback project root for tests that don't explicitly set one.
    This prevents ProjectNotInitializedError when code calls get_project_root().
    """
    root = tmp_path_factory.mktemp("default_project")
    (root / ".pivot").mkdir()
    (root / ".git").mkdir()
    return root


@pytest.fixture(autouse=True)
def reset_pivot_state(
    mocker: MockerFixture, _default_project_root: pathlib.Path
) -> Generator[None]:
    """Reset global pivot state between tests.

    CliRunner can leave console singleton pointing to closed streams,
    project root cache pointing to old directories, and merged config cached.

    Sets project root to a default directory with .pivot to prevent
    ProjectNotInitializedError. Tests that need a specific project root
    should use set_project_root, mock_discovery, or pipeline_dir fixtures.

    Tests using isolated_filesystem() that create their own .pivot should
    reset the cache: project._project_root_cache = None
    """
    mocker.patch.object(console, "_console", None)
    # Direct assignment (not mocker.patch) so tests can override by setting to None
    project._project_root_cache = _default_project_root
    config_io.clear_config_cache()
    for name in _PIVOT_LOGGERS:
        logging.getLogger(name).handlers.clear()
    yield
    project._project_root_cache = None


@pytest.fixture
def set_project_root(tmp_path: pathlib.Path, mocker: MockerFixture) -> Generator[pathlib.Path]:
    """Set project root to tmp_path for tests that register stages with temp paths.

    Creates .pivot directory so project root discovery works.
    Also creates .git for fingerprinting and other git-dependent operations.
    """
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)
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
def prewarm_worker_pool() -> Generator[None]:
    """Pre-warm loky worker pool before tests run.

    This eliminates cold-start overhead on the first test that uses workers.
    """
    executor_core.prepare_workers(stage_count=2, parallel=True, max_workers=2)
    yield


@pytest.fixture(scope="session", autouse=True)
def cleanup_worker_pool() -> Generator[None]:
    """Kill loky worker pool at end of test session to prevent orphaned workers."""
    yield
    executor_core._cleanup_worker_pool()


# Type alias for make_valid_lock_content fixture
# Returns dict matching StorageLockData structure (code_manifest, params, deps, outs, dep_generations)
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
def stages_dir(pipeline_dir: pathlib.Path) -> pathlib.Path:
    """Return the stages directory, creating it if needed."""
    dir_path = pipeline_dir / ".pivot" / "stages"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


@pytest.fixture
def runner() -> click.testing.CliRunner:
    """Create a CLI runner for testing."""
    return click.testing.CliRunner()


@contextlib.contextmanager
def isolated_pivot_dir(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> Generator[pathlib.Path]:
    """Context manager for isolated filesystem with .pivot directory.

    Use this instead of runner.isolated_filesystem() for tests that need
    real pipeline discovery (not using mock_discovery fixture).

    Creates .pivot and .git directories and resets the project root cache
    so find_project_root() discovers the isolated directory.

    Example:
        def test_something(runner, tmp_path):
            with isolated_pivot_dir(runner, tmp_path) as cwd:
                Path("pipeline.py").write_text("...")
                result = runner.invoke(cli, ["status"])
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        cwd = pathlib.Path.cwd()
        (cwd / ".pivot").mkdir()
        (cwd / ".git").mkdir()
        project._project_root_cache = None
        yield cwd


@pytest.fixture
def output_queue() -> Generator[mp.Queue[OutputMessage]]:
    """Create a multiprocessing queue for worker output using spawn context.

    Uses spawn context to match production behavior and avoid Python 3.13+
    deprecation warnings about fork() in multi-threaded contexts.
    """
    spawn_ctx = mp.get_context("spawn")
    q: mp.Queue[OutputMessage] = spawn_ctx.Queue()
    try:
        yield q
    finally:
        q.close()
        with contextlib.suppress(OSError, ValueError):
            q.join_thread()


@pytest.fixture
async def test_engine(test_pipeline: pipeline_mod.Pipeline) -> AsyncGenerator[Engine]:
    """Provide a context-managed Engine instance.

    The engine is properly closed after each test to ensure sinks are cleaned up.
    """
    from pivot.engine.engine import Engine

    async with Engine(pipeline=test_pipeline) as eng:
        yield eng


@pytest.fixture
def worker_env(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Set up worker execution environment with cache and stages directories."""
    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "files").mkdir(exist_ok=True)
    (tmp_path / ".pivot" / "stages").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    return cache_dir


# =============================================================================
# RPC Test Utilities
# =============================================================================


def send_rpc(
    sock_path: pathlib.Path, method: str, params: dict[str, object] | None = None
) -> dict[str, object]:
    """Send JSON-RPC request via Unix socket and return response.

    This is a synchronous helper for integration tests that need to communicate
    with the engine's RPC server via agent.sock.

    Args:
        sock_path: Path to the Unix socket (typically .pivot/agent.sock).
        method: JSON-RPC method name (e.g., "status", "run", "events_since").
        params: Optional parameters for the method.

    Returns:
        Parsed JSON response as a dict.
    """
    import json
    import socket

    request: dict[str, object] = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params:
        request["params"] = params

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(5.0)
        sock.connect(str(sock_path))
        sock.sendall(json.dumps(request).encode() + b"\n")
        response = sock.recv(4096).decode()
    return json.loads(response)
