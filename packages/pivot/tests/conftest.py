# pyright: reportMissingImports=false, reportImplicitRelativeImport=false
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import linecache
import logging
import multiprocessing as mp
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import AsyncGenerator, Callable, Generator
from typing import TYPE_CHECKING

import click.testing
import pytest
from moto import server as moto_server

from pivot import discovery, project
from pivot.compose import Pipeline
from pivot.config import io as config_io
from pivot.executor import core as executor_core
from pivot.registry import PipelineLike, StageRegistry

_tests_dir = pathlib.Path(__file__).parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))
_tests_root = _tests_dir.parent
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))
pythonpath = os.environ.get("PYTHONPATH")
if pythonpath:
    os.environ["PYTHONPATH"] = os.pathsep.join([str(_tests_root), pythonpath])
else:
    os.environ["PYTHONPATH"] = str(_tests_root)


if TYPE_CHECKING:
    from pytest_mock import MockerFixture
    from types_aiobotocore_s3 import S3Client

    from pivot.engine.engine import Engine
    from pivot.engine.types import OutputEvent
    from pivot.types import OutputMessage

# Type alias for git_repo fixture: (repo_path, commit_fn)
GitRepo = tuple[pathlib.Path, Callable[[str], str]]

# Each xdist worker spawns loky workers + Manager; ~2 GB per worker needed
_MEMORY_PER_WORKER_BYTES = 2 * 1024**3
# Hard cap on worker count to prevent resource exhaustion
_MAX_WORKERS_CAP = 16


def pytest_configure() -> None:
    """Give every process the suite starts SIGINT at its default disposition.

    A background job of a non-interactive shell (``cmd &`` in a script) starts with SIGINT
    ignored, and an ignored signal stays ignored across exec, through ``uv run`` too. The tests
    that press Ctrl-C on pivot would then wait out their timeouts. A handled signal resets to the
    default on exec, so installing Python's own handler here gives each child the default.
    """
    if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
        signal.signal(signal.SIGINT, signal.default_int_handler)


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
        # Fallback: cap at 8 workers outside containers (conservative default)
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
            # Threshold: 2^60 bytes (~1 EB) — skip unrealistic limits
            if limit < (1 << 60):
                return limit
    except (OSError, ValueError):
        # Ignore if cgroup files are unreadable (non-Linux, CI, or permission issues)
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
def test_pipeline(tmp_path: pathlib.Path) -> Generator[PipelineLike]:
    """Provide a fresh Pipeline for tests.

    Note: This fixture does NOT mock the project root. Tests that register
    stages with path annotations should either:
    1. Use mock_discovery fixture (which mocks project root)
    2. Or explicitly mock project._project_root_cache themselves
    """
    pipeline = Pipeline("test", root=tmp_path)
    yield pipeline


@pytest.fixture
def mock_discovery(
    test_pipeline: PipelineLike,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> PipelineLike:
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

    Note: The test_pipeline fixture is automatically used.
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
def reset_pivot_state(_default_project_root: pathlib.Path) -> Generator[None]:
    """Reset global pivot state between tests.

    CliRunner can leave the project root cache pointing to old directories and merged
    config cached.

    Sets project root to a default directory with .pivot to prevent
    ProjectNotInitializedError. Tests that need a specific project root
    should use set_project_root, mock_discovery, or pipeline_dir fixtures.

    Tests using isolated_filesystem() that create their own .pivot should
    reset the cache: project._project_root_cache = None
    """
    # Direct assignment (not mocker.patch) so tests can override by setting to None
    project._project_root_cache = _default_project_root
    config_io.clear_config_cache()
    for name in _PIVOT_LOGGERS:
        logging.getLogger(name).handlers.clear()
    yield
    project._project_root_cache = None


@pytest.fixture(autouse=True)
def restore_import_machinery(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Give the next test the import machinery this test started with.

    One pytest-xdist worker runs many tests in one process. A test that installs Pivot's user-code
    loader (``discovery.install_user_code_loader``, as a worker's initializer does) or runs a
    discovery changes process-wide import state: ``sys.meta_path``, ``sys.path_hooks``,
    ``sys.path_importer_cache``, ``importlib.util.spec_from_file_location``, and the import
    session a discovery leaves for fingerprinting and workers, with its records of loaded source.
    Left behind, they change what a later test imports and fingerprints. The session is ended
    before ``monkeypatch`` undoes the test's ``sys.path`` and ``sys.modules`` changes, which this
    fixture depends on for that order.
    """
    meta_path = list(sys.meta_path)
    path_hooks = list(sys.path_hooks)
    importer_cache = dict(sys.path_importer_cache)
    spec_from_file_location = importlib.util.spec_from_file_location
    session = discovery._import_session
    yield
    if discovery._import_session is not None and discovery._import_session is not session:
        discovery._close_import_session(discovery._import_session)
    sys.meta_path[:] = meta_path
    sys.path_hooks[:] = path_hooks
    sys.path_importer_cache.clear()
    sys.path_importer_cache.update(importer_cache)
    importlib.util.spec_from_file_location = spec_from_file_location


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


_VENV_LIBRARY = "pivot_test_native_ext"


@pytest.fixture
def venv_inside_root_project(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[pathlib.Path]:
    """A project whose virtualenv sits inside its root, as `uv venv` in the project makes.

    `pipeline.py` imports a library from `<root>/.venv/.../site-packages` that, like numpy's C
    extension, fails when imported a second time in one process.
    """
    site_packages = tmp_path / ".venv" / "lib" / "python3.13" / "site-packages"
    library = site_packages / _VENV_LIBRARY
    library.mkdir(parents=True)
    (library / "__init__.py").write_text(
        "import pathlib\n"
        + "_marker = pathlib.Path(__file__).with_name('imported')\n"
        + "if _marker.exists():\n"
        + "    raise ImportError('cannot load module more than once per process')\n"
        + "_marker.touch()\n"
    )
    (tmp_path / "pipeline.py").write_text(
        "from typing import TypedDict\n\n"
        + f"import {_VENV_LIBRARY}\n"
        + "from pivot.compose import Pipeline, stage\n\n\n"
        + "class TrainOutputs(TypedDict):\n    model: str\n    score: str\n\n\n"
        + "@stage\n"
        + "def train() -> TrainOutputs:\n"
        + '    return {"model": "m", "score": "1"}\n\n\n'
        + 'with Pipeline("venv") as pipeline:\n'
        + "    train()\n"
    )
    (tmp_path / ".git").mkdir()
    (tmp_path / ".pivot").mkdir()
    monkeypatch.syspath_prepend(str(site_packages))
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    sys.modules.pop(_VENV_LIBRARY, None)


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
def start_resource_trackers() -> Generator[None]:
    """Start process trackers before any in-process Textual capture begins."""
    executor_core.start_resource_trackers()
    yield


@pytest.fixture(scope="session", autouse=True)
def cleanup_worker_pool() -> Generator[None]:
    """Kill loky worker pool at end of test session to prevent orphaned workers."""
    yield
    executor_core._cleanup_worker_pool()


# Type alias for make_valid_lock_content fixture
# Returns dict matching StorageLockData structure (code_manifest, params, deps, outs)
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
async def test_engine(test_pipeline: PipelineLike) -> AsyncGenerator[Engine]:
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


class AsyncEventCaptureSink:
    """Async sink that captures all events for inspection."""

    def __init__(self) -> None:
        self.events: list[OutputEvent] = []

    async def handle(self, event: OutputEvent) -> None:
        self.events.append(event)

    async def close(self) -> None:
        pass


@pytest.fixture
def minimal_pipeline(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, test_pipeline: PipelineLike
) -> PipelineLike:
    """Set up a minimal pipeline for testing."""
    from pivot import config

    monkeypatch.setattr(config, "get_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(config, "get_state_dir", lambda: tmp_path / "state")
    monkeypatch.chdir(tmp_path)

    (tmp_path / ".pivot").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)

    return test_pipeline


# =============================================================================
# tests.helpers default pipeline
# =============================================================================


@pytest.fixture(autouse=True)
def bind_test_pipeline_helper(request: pytest.FixtureRequest) -> Generator[None]:
    """Point tests.helpers' default pipeline at the test_pipeline fixture.

    Active only for tests that request test_pipeline (directly or through
    mock_discovery, minimal_pipeline, test_engine, ...), so register_test_stage()
    registers into that pipeline without an explicit pipeline argument.
    """
    if "test_pipeline" not in request.fixturenames:
        yield
        return
    from tests import helpers

    helpers.set_test_pipeline(request.getfixturevalue("test_pipeline"))
    yield
    helpers.set_test_pipeline(None)


# =============================================================================
# S3 (moto)
# =============================================================================


@pytest.fixture(scope="session")
def moto_services() -> Generator[dict[str, str]]:
    """Serve moto in-process, replacing pytest-aioboto3's moto_services.

    The plugin's fixture runs ``moto_server`` as a subprocess with stdout/stderr
    piped and never read; after ~70 requests in one pytest process the pipe
    buffer fills with request logs, moto_server blocks writing, and every
    later S3 call hangs. ThreadedMotoServer has no pipe to fill.
    """
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    yield {"s3": f"http://{host}:{port}"}
    server.stop()


@pytest.fixture
async def moto_s3_bucket(
    moto_patch_session: object, aioboto3_s3_client: S3Client
) -> AsyncGenerator[str]:
    """Create a test bucket in moto with unique name for xdist support.

    Uses pytest-aioboto3's moto_patch_session to mock S3 and aioboto3_s3_client
    to create the bucket. Bucket name includes a unique suffix to prevent
    conflicts when running tests in parallel with pytest-xdist.

    Yields:
        str: The bucket name (e.g., "test-bucket-a1b2c3d4").
    """
    bucket_name = f"test-bucket-{uuid.uuid4().hex[:8]}"
    await aioboto3_s3_client.create_bucket(Bucket=bucket_name)
    yield bucket_name


# =============================================================================
# GCS (fake-gcs-server)
# =============================================================================

_FAKE_GCS_IMAGE = "fsouza/fake-gcs-server:1.56.1"
_FAKE_GCS_LABEL = "owner=pivot-tests"
_FAKE_GCS_START_TIMEOUT = 60


def _free_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


def _fake_gcs_ready(endpoint: str) -> bool:
    try:
        with urllib.request.urlopen(f"{endpoint}/storage/v1/b", timeout=1) as resp:
            return resp.status == 200
    except (urllib.error.URLError, ConnectionError):
        return False


@pytest.fixture(scope="session")
def fake_gcs_server() -> Generator[str]:
    """Serve fake-gcs-server from Docker for this test process and yield its endpoint URL.

    One container per xdist worker, removed when the worker's session ends. Without Docker the
    GCS tests are skipped locally and fail in CI (``CI`` set), so CI never loses them silently.
    """
    if shutil.which("docker") is None:
        message = "fake-gcs-server needs Docker, which is not installed"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)

    port = _free_local_port()
    endpoint = f"http://127.0.0.1:{port}"
    run = subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--label",
            _FAKE_GCS_LABEL,
            "--publish",
            f"127.0.0.1:{port}:{port}",
            _FAKE_GCS_IMAGE,
            "-scheme",
            "http",
            "-port",
            str(port),
            "-backend",
            "memory",
            "-external-url",
            endpoint,
            "-log-level",
            "error",
        ],
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        pytest.fail(f"docker run {_FAKE_GCS_IMAGE} failed: {run.stderr.strip()}")
    container = run.stdout.strip()
    try:
        deadline = time.monotonic() + _FAKE_GCS_START_TIMEOUT
        while not _fake_gcs_ready(endpoint):
            if time.monotonic() > deadline:
                logs = subprocess.run(
                    ["docker", "logs", container], capture_output=True, text=True
                ).stderr
                pytest.fail(f"fake-gcs-server did not answer at {endpoint}: {logs}")
            time.sleep(0.1)
        yield endpoint
    finally:
        subprocess.run(["docker", "rm", "--force", container], capture_output=True)


@pytest.fixture
def gcs_bucket(fake_gcs_server: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """Create a uniquely named fake-gcs-server bucket and point gcloud-aio-storage at it."""
    monkeypatch.setenv("STORAGE_EMULATOR_HOST", fake_gcs_server)
    name = f"pivot-test-{uuid.uuid4().hex[:12]}"
    request = urllib.request.Request(
        f"{fake_gcs_server}/storage/v1/b?project=pivot-test",
        data=json.dumps({"name": name}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as resp:
        assert resp.status == 200, resp.read()
    return name
