"""Test helpers shared by test modules: pipelines built with the compositional API or written as
pipeline.py text, tracked files with no data or removed once tracked, a cache's whole-directory
trees and a cache register on a read-only mount, git projects that the installed `pivot` runs in,
the output of `pivot repro --jsonl` and its restore warnings, the dry run and the run of `pivot
repro --all --keep-going`, sockets, a call that must not wait on a FIFO or a lock, an entry of
each kind where a path belongs, and venvs that CI-only tests install with uv."""

# pyright: reportMissingImports=false, reportImplicitRelativeImport=false

from __future__ import annotations

import errno
import functools
import inspect
import json
import os
import pathlib
import re
import socket
import stat
import subprocess
import sys
import textwrap
import threading
import typing
from typing import TYPE_CHECKING, Any

import anyio
import pytest

from pivot import cli, compose, discovery, exceptions, project, stage_def
from pivot import names as stage_names
from pivot.storage import cache, naming, state, store, track
from pivot.storage import mounts as mounts_module

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from multiprocessing.synchronize import Event

    import click.testing

    from pivot.registry import PipelineLike
    from pivot.storage import listing

DepValue = compose.ArtifactHandle | str | list[compose.ArtifactHandle]

# The `pivot` command installed beside the interpreter running the tests. CI runs the suite
# without uv on PATH, so tests start this path, never `uv run pivot`.
PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

# Pipeline p's stage p/plan writes the directory output data/p/plan/, holding a.txt and b.txt
PRODUCER_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def plan() -> Annotated[dict[str, str], directory]:
    return {"a.txt": "a", "b.txt": "b"}


with Pipeline("p") as pipeline:
    plan()
"""

# PRODUCER_PIPELINE with p/cut, mapped over p/plan's files: instance p/cut@<key> writes the file
# data/p/cut@<key>.txt
MAPPED_FILE_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def plan() -> Annotated[dict[str, str], directory]:
    return {"a.txt": "a", "b.txt": "b"}


@stage
def cut(entry: str) -> str:
    return entry


with Pipeline("p") as pipeline:
    cut(pipeline.map(plan()))
"""


def track_without_data(data_path: pathlib.Path) -> pathlib.Path:
    """Write the ``.pvt`` of ``data_path`` alone, as ``pivot track`` names it, with no data on disk
    and no object in any cache: a tracked file that is missing and that ``--checkout-missing``
    cannot restore. Makes the directory that holds the ``.pvt``; returns the ``.pvt``'s path."""
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pvt_path = track.get_pvt_path(data_path)
    track.write_pvt_file(
        pvt_path, track.PvtData(path=data_path.name, hash="0123456789abcdef", size=10)
    )
    return pvt_path


def track_then_remove(runner: click.testing.CliRunner, root: pathlib.Path, name: str) -> None:
    """Track ``root / name`` with the real ``pivot track`` (its objects go to the cache), then
    remove it from the workspace."""
    assert runner.invoke(cli.cli, ["track", name]).exit_code == 0
    cache.remove_output(root / name)


def cache_whole_tree(root: pathlib.Path, tracked: pathlib.Path) -> listing.DirListing:
    """Put in the cache of the project at ``root`` the whole-directory tree of tracked directory
    ``tracked``, complete and marked, as a SYMLINK commit of a stage output with the same files
    leaves it: the files its ``.pvt`` record lists, built from their cache objects, which hash to
    that record. Returns the record's listing."""
    files = root / ".pivot" / "cache" / "files"
    record = track.read_pvt_file(track.get_pvt_path(tracked))
    assert record is not None
    members = track.tracked_members(tracked, record, files)
    twin = root / "twin"
    twin.mkdir()
    for entry in members["files"]:
        member = twin / entry["relpath"]
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_bytes(cache.get_cache_path(files, entry["hash"]).read_bytes())
        member.chmod(0o755 if entry["isexec"] else 0o644)
    for relpath in members["empty_dirs"]:
        (twin / relpath).mkdir(parents=True)
    saved = cache.save_to_cache(
        twin, files, checkout_modes=[cache.CheckoutMode.SYMLINK, cache.CheckoutMode.COPY]
    )
    assert saved == track.pvt_to_hash_info(record), "the tree is the tracked directory's"
    twin.unlink()
    return members


def cache_whole_tree_lacking(root: pathlib.Path, tracked: pathlib.Path, relpath: str) -> str:
    """``cache_whole_tree`` of tracked directory ``tracked`` in the project at ``root``, then the
    cache object of its member ``relpath`` removed. Returns that object's hash."""
    members = cache_whole_tree(root, tracked)
    member_hash = next(entry["hash"] for entry in members["files"] if entry["relpath"] == relpath)
    cache.get_cache_path(root / ".pivot" / "cache" / "files", member_hash).unlink()
    return member_hash


def register_on_a_read_only_mount(
    monkeypatch: pytest.MonkeyPatch, cache_root: pathlib.Path
) -> None:
    """Make the register of the cache at ``cache_root`` behave, in this process, as on a
    filesystem mounted read-only, which this user cannot make: opening a naming lock file there
    fails with EROFS, as it does on such a mount (``naming.naming_unit`` then refuses this
    process's links into the cache), and ``os.statvfs`` of the register reports ``ST_RDONLY``
    beside its real figures. The links a registration refuses go to a table of this test's
    own."""
    register = os.path.realpath(naming.register_root(cache_root))
    real_open_cache_lock = cache.open_cache_lock
    real_statvfs = os.statvfs

    def open_cache_lock(path: pathlib.Path, *, dir_fd: int | None = None) -> int:
        if dir_fd is None and os.path.realpath(path.parent) == register:
            raise OSError(errno.EROFS, os.strerror(errno.EROFS), str(path))
        return real_open_cache_lock(path, dir_fd=dir_fd)

    def statvfs(path: str | os.PathLike[str]) -> os.statvfs_result:
        found = real_statvfs(path)
        if os.path.realpath(path) != register:
            return found
        return os.statvfs_result((*found[:8], found.f_flag | os.ST_RDONLY, *found[9:]))

    monkeypatch.setattr(cache, "_UNREGISTERED_ROOTS", dict[str, str]())
    monkeypatch.setattr(cache, "open_cache_lock", open_cache_lock)
    monkeypatch.setattr(os, "statvfs", statvfs)


# The line `TrackedFileMissingError` ends with when a file it lists overlaps a stage output
OVERLAP_STOPS = (
    "Pivot does not run a stage whose output overlaps a file above. "
    + "That stage replaces its output when it runs."
)


# `pivot repro --all --keep-going --dry-run --jsonl`: its one JSON document on stdout
DRY_RUN_KEEP_GOING = ["repro", "--all", "--keep-going", "--dry-run", "--jsonl"]

# A row of a DRY_RUN_KEEP_GOING document: a stage the run would skip as unchanged
UNCHANGED_ROW = {"would_run": False, "reason": "unchanged"}


def ok(proc: subprocess.CompletedProcess[str]) -> str:
    """The output of ``proc``, a command that must exit 0: its stdout, then its stderr."""
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def git_project(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """An initialized pivot project in a git repository under ``tmp_path``, and the home its
    commands run with (``run_pivot``)."""
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    for args in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "t@example.com"),
        ("git", "config", "user.name", "t"),
    ):
        ok(run_cli(root, home, *args))
    ok(run_pivot(root, home, "init"))
    return root, home


def dry_run_keep_going(
    root: pathlib.Path, home: pathlib.Path, *args: str
) -> tuple[dict[str, Any], str]:
    """``DRY_RUN_KEEP_GOING`` with ``args`` in project ``root``, which must exit 0: its
    document, parsed from its stdout alone, and its stderr."""
    proc = run_pivot(root, home, *DRY_RUN_KEEP_GOING, *args)
    ok(proc)
    return json.loads(proc.stdout), proc.stderr


def run_keep_going(
    root: pathlib.Path, home: pathlib.Path, *args: str
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, Any]]]:
    """The run ``dry_run_keep_going`` with ``args`` predicts, `pivot repro --all --keep-going
    --jsonl` in project ``root``, and its JSONL events (``jsonl_events``)."""
    proc = run_pivot(root, home, "repro", "--all", "--keep-going", "--jsonl", *args)
    return proc, jsonl_events(proc.stdout)


def jsonl_events(stdout: str) -> list[dict[str, Any]]:
    """The events a `--jsonl` run's ``stdout`` holds, one JSON object a line."""
    return [json.loads(line) for line in stdout.splitlines()]


def stage_statuses(stdout: str) -> dict[str, object]:
    """The status of each stage a `--jsonl` run's ``stdout`` reports (``completed_stages``), by
    stage."""
    return {stage: event["status"] for stage, event in completed_stages(stdout).items()}


def refused_mapped_project(tmp_path: pathlib.Path, code: str) -> tuple[pathlib.Path, pathlib.Path]:
    """A ``git_project`` whose p/plan (``PRODUCER_PIPELINE``) is current, and where
    data/p/cut@a.txt, the output p/cut@a would write, is a missing tracked file; then ``code``
    declares p/cut. The file is tracked before p/cut is declared, since `pivot track` refuses a
    path an output the pipeline declares holds."""
    root, home = git_project(tmp_path)
    (root / "pipeline.py").write_text(PRODUCER_PIPELINE)
    ok(run_pivot(root, home, "repro"))
    tracked = root / "data" / "p" / "cut@a.txt"
    tracked.write_text("tracked bytes")
    ok(run_pivot(root, home, "track", "data/p/cut@a.txt"))
    tracked.unlink()
    (root / "pipeline.py").write_text(code)
    return root, home


def uncached_incremental_project(
    tmp_path: pathlib.Path, code: str
) -> tuple[pathlib.Path, pathlib.Path]:
    """A ``git_project`` where every stage of ``code``, whose p/cut instances keep an
    incremental output, ran; then p/cut@a's lock is gone, so its output data/p/cut@a.txt is on
    disk but not in the cache."""
    root, home = git_project(tmp_path)
    (root / "pipeline.py").write_text(code)
    ok(run_pivot(root, home, "repro"))
    (root / ".pivot" / "stages" / "p" / "cut@a.lock").unlink()
    return root, home


def one_stage_pipeline_code(name: str, stage_name: str) -> str:
    """Return a complete compositional pipeline.py with one value-returning stage."""
    return f'''\
from pivot.compose import Pipeline, stage


@stage
def {stage_name}() -> str:
    return "{stage_name} output"


with Pipeline("{name}") as pipeline:
    {stage_name}()
'''


def write_pipeline(root: pathlib.Path, name: str, code: str) -> None:
    """Write pipeline.py ``code`` into the new directory ``root / name``."""
    (root / name).mkdir()
    (root / name / "pipeline.py").write_text(code)


def skipping_lines(output: str) -> list[str]:
    """Each `Skipping:` line of ``output``: a stage a run leaves out, with why."""
    return [line for line in output.splitlines() if "Skipping:" in line]


def restore_warnings(stderr: str) -> list[str]:
    """Each `Cannot restore` warning line of ``stderr``: a tracked path a restore left missing."""
    return [line for line in stderr.splitlines() if line.startswith("Cannot restore ")]


def call_without_waiting[T](
    call: Callable[[], T], release: Callable[[], None], timeout: float = 120
) -> T:
    """``call()``, run in a thread, and what it returns or raises. When it is still running
    after ``timeout`` seconds, as an open of a FIFO for reading waits for a writer or a lock
    waits for its holder, ``release()`` frees what it waits on (``free_fifo`` for a FIFO), the
    thread has ``timeout`` seconds more to end, and the test fails instead of hanging."""
    outcome = list[T | BaseException]()

    def _run() -> None:
        try:
            outcome.append(call())
        except BaseException as error:  # re-raised in the test's thread below
            outcome.append(error)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout)
    waited = thread.is_alive()
    if waited:
        release()
        thread.join(timeout)
    assert not waited, f"the call was still running after {timeout} s"
    (result,) = outcome
    if isinstance(result, BaseException):
        raise result
    return result


def free_fifo(path: pathlib.Path) -> None:
    """Free each open for reading that waits on the FIFO ``path`` leads to (symlinks followed),
    and put an empty regular file in its place, so that no later open waits either (a stage's
    worker hashes an input, then opens it again to load it). Nothing where ``path`` leads to no
    FIFO: a call that waits elsewhere fails its test all the same."""
    fifo = pathlib.Path(os.path.realpath(path))
    try:
        if not stat.S_ISFIFO(os.lstat(fifo).st_mode):
            return
    except FileNotFoundError:
        return
    try:
        # A writer frees each open for reading that waits
        writer = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
    except OSError as e:
        if e.errno != errno.ENXIO:  # ENXIO: no open for reading waits
            raise
        writer = None
    try:
        fifo.unlink()
        fifo.touch()
    finally:
        if writer is not None:
            os.close(writer)


def occupy(path: pathlib.Path, kind: str) -> None:
    """Put an entry of ``kind`` at ``path``, where none is, and check with ``os.lstat`` that it
    is there, of its type (for "nothing", that no entry is), so no test passes on an entry made
    elsewhere: "nothing"; a "regular_file"; a "directory" (mode 0o700); a "socket"; a "fifo"
    (mode 0o600); a symlink to the file target.txt ("symlink_to_file"), the directory somedir
    ("symlink_to_directory") or the FIFO pipe ("symlink_to_fifo"), each made beside it; a
    symlink to nothing ("dangling_symlink"); or a symlink to itself ("symlink_loop"). The modes
    are fixed, so a test may name them (``stat.filemode``).

    A socket is bound by ``os.path.relpath(path)``, which works from any working directory at or
    above the directory holding ``path``: an address holds at most 108 bytes, which an absolute
    path under a test's temporary directory may pass. Its file stays once the socket is closed.
    """
    match kind:
        case "nothing":
            assert not os.path.lexists(path), path
            return
        case "regular_file":
            path.write_text("data")
            made = stat.S_IFREG
        case "directory":
            path.mkdir(mode=0o700)
            made = stat.S_IFDIR
        case "socket":
            with socket.socket(socket.AF_UNIX) as bound:
                bound.bind(os.path.relpath(path))
            made = stat.S_IFSOCK
        case "fifo":
            os.mkfifo(path, 0o600)
            made = stat.S_IFIFO
        case "symlink_to_file":
            (path.parent / "target.txt").write_text("target")
            path.symlink_to("target.txt")
            made = stat.S_IFLNK
        case "symlink_to_directory":
            (path.parent / "somedir").mkdir()
            path.symlink_to("somedir")
            made = stat.S_IFLNK
        case "symlink_to_fifo":
            os.mkfifo(path.parent / "pipe", 0o600)
            path.symlink_to("pipe")
            made = stat.S_IFLNK
        case "dangling_symlink":
            path.symlink_to("nowhere")
            made = stat.S_IFLNK
        case "symlink_loop":
            path.symlink_to(path.name)
            made = stat.S_IFLNK
        case _:
            raise ValueError(f"no entry kind {kind!r}")
    assert stat.S_IFMT(os.lstat(path).st_mode) == made, f"{path} is no {kind}"


def completed_stages(stdout: str) -> dict[str, dict[str, object]]:
    """The `stage_complete` event of each stage a `--jsonl` run reports, by stage."""
    events = jsonl_events(stdout)
    return {event["stage"]: event for event in events if event["type"] == "stage_complete"}


def cli_env(home: pathlib.Path, **extra: str) -> dict[str, str]:
    """The environment an installed-CLI test starts a command in.

    This process's environment without any ``PIVOT*`` variable (a developer's
    ``PIVOT_CACHE_DIR`` or ``PIVOT_UNSAFE_FINGERPRINTING`` never reaches the command) and
    without ``VIRTUAL_ENV``, with ``HOME`` set to ``home`` so the developer's global config
    does not apply, then ``extra``.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PIVOT") and key != "VIRTUAL_ENV"
    }
    env["HOME"] = str(home)
    env.update(extra)
    return env


def run_cli(
    cwd: pathlib.Path, home: pathlib.Path, *argv: str, extra_env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` (``pivot`` or a setup command such as ``git``) in ``cli_env(home)``.

    Output is captured as text, with undecodable bytes replaced; the exit code is not checked.
    """
    return subprocess.run(
        argv,
        cwd=cwd,
        env=cli_env(home, **(extra_env or {})),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=300,
        check=False,
    )


def run_pivot(
    cwd: pathlib.Path, home: pathlib.Path, *args: str, extra_env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the installed ``pivot`` with ``args`` in ``cwd``, as ``run_cli`` runs a command."""
    return run_cli(cwd, home, PIVOT, *args, extra_env=extra_env)


async def wait_for_socket(socket_path: pathlib.Path, timeout: float = 5.0) -> None:
    """Wait for Unix socket to be created and connectable.

    Args:
        socket_path: Path to the Unix socket file.
        timeout: Maximum time to wait in seconds.

    Raises:
        TimeoutError: If socket is not created within timeout.
    """
    deadline = anyio.current_time() + timeout
    while anyio.current_time() < deadline:
        if socket_path.exists():
            try:
                mode = socket_path.stat().st_mode
                if stat.S_ISSOCK(mode):
                    connection = await anyio.connect_unix(str(socket_path))
                    await connection.aclose()
                    return
            except (OSError, anyio.BrokenResourceError, anyio.ClosedResourceError):
                pass
        await anyio.sleep(0.05)
    msg = f"Socket {socket_path} not created and connectable within {timeout}s"
    raise TimeoutError(msg)


def clean_env() -> dict[str, str]:
    """Return ``os.environ`` without the workspace's ``VIRTUAL_ENV*`` and ``PYTHONPATH*``."""
    return {k: v for k, v in os.environ.items() if not k.startswith(("VIRTUAL_ENV", "PYTHONPATH"))}


def uv_executable() -> str:
    """The uv executable ``PIVOT_TEST_UV`` names. CI runs the suite without uv on PATH and sets
    it; a test that needs uv runs only in CI (``CI`` set), and fails here if it is unset."""
    uv = os.environ.get("PIVOT_TEST_UV", "")
    if not (pathlib.Path(uv).is_file() and os.access(uv, os.X_OK)):
        pytest.fail(f"PIVOT_TEST_UV must name the uv executable; it is {uv!r}", pytrace=False)
    return uv


def venv_pivot(venv: pathlib.Path, *specs: str) -> str:
    """Make a Python 3.13 venv at ``venv`` with ``uv_executable()``, install ``specs`` into it
    (a wheel path, ``pivot==<version>``, other requirements), and return its ``pivot``."""
    uv = uv_executable()
    commands = (
        [uv, "venv", "--python", "3.13", str(venv)],
        [uv, "pip", "install", "--python", str(venv / "bin" / "python"), *specs],
    )
    for command in commands:
        ran = subprocess.run(
            command,
            cwd=venv.parent,
            env=clean_env(),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        assert ran.returncode == 0, ran.stdout + ran.stderr
    return str(venv / "bin" / "pivot")


@functools.cache
def git_version() -> tuple[int, int]:
    """The major and minor version of the ``git`` on PATH, which the git-oracle tests run."""
    out = subprocess.run(["git", "--version"], capture_output=True, text=True, check=True).stdout
    found = re.search(r"(\d+)\.(\d+)", out)
    assert found is not None, out
    return int(found[1]), int(found[2])


def git_differs_before_2_52(line: bytes) -> bool:
    """Whether a git older than 2.52 matches the ignore-file line ``line`` otherwise than 2.52
    and Pivot do: a pattern holding a ``/`` whose first wildcard is a ``**`` right after a byte
    that is not ``/``. git 2.52 gives wildmatch the literal prefix's last byte, so that ``**``
    is a ``*``; before it, git dropped the whole prefix and the ``**`` spanned directories (git
    commit 1940a02dc1, "match_pathname(): give fnmatch one char of prefix context")."""
    pattern = line.removesuffix(b"\r").partition(b"\0")[0].rstrip(b" ")
    if not pattern or pattern.startswith(b"#"):
        return False
    pattern = pattern.removeprefix(b"!").removesuffix(b"/")
    if b"/" not in pattern:
        return False
    pattern = pattern.removeprefix(b"/")
    first = next((i for i, byte in enumerate(pattern) if byte in b"*?[\\"), len(pattern))
    return first > 0 and pattern[first : first + 2] == b"**" and pattern[first - 1] != ord("/")


def incremental_stop(output: str, stage: str) -> str | None:
    """How a released pivot's ``repro`` left ``stage``, an incremental stage, without running it.

    ``"refused"`` when it printed its cache-missing refusal for the stage; ``"blocked"`` when it
    marked the stage blocked behind another stage's failure; ``None`` otherwise (the stage ran,
    or the output does not name it). Which of two independent incremental stages refuses and
    which is blocked depends on which one a worker reaches first, so a test asserts that each
    stopped and that one refused, never the order.
    """
    flat = " ".join(output.split())
    if f"Cache missing for incremental output '{stage}'" in flat:
        return "refused"
    if re.search(rf"(?<!\S){re.escape(stage)} blocked(?!\S)", flat):
        return "blocked"
    return None


# Default pipeline for register_test_stage(). The conftest fixture
# bind_test_pipeline_helper sets it to the test_pipeline fixture's Pipeline for
# every test that uses test_pipeline (directly or via mock_discovery etc.).
_test_pipeline: PipelineLike | None = None


def get_test_pipeline() -> PipelineLike:
    """Get the current test pipeline.

    Raises RuntimeError if no test pipeline is set.
    """
    if _test_pipeline is None:
        raise RuntimeError(
            "No test pipeline available. Use the 'test_pipeline' fixture or call "
            + "set_test_pipeline() before registering stages."
        )
    return _test_pipeline


def set_test_pipeline(pipeline: PipelineLike | None) -> None:
    """Set the module-level test pipeline."""
    global _test_pipeline
    _test_pipeline = pipeline


def _original_func(func: Callable[..., Any]) -> Callable[..., Any]:
    """Return the undecorated function behind an @stage wrapper."""
    return inspect.unwrap(func)


def _param_types(func: Callable[..., Any]) -> dict[str, object]:
    """Resolve parameter type hints, unwrapping Annotated[T, ...] to T."""
    hints = typing.get_type_hints(_original_func(func), include_extras=True)
    result = dict[str, object]()
    for param_name, hint in hints.items():
        if param_name == "return":
            continue
        if typing.get_origin(hint) is typing.Annotated:
            hint = typing.get_args(hint)[0]
        result[param_name] = hint
    return result


def _params_class(hint: object) -> type[stage_def.StageParams] | None:
    if isinstance(hint, type) and issubclass(hint, stage_def.StageParams):
        return hint
    return None


def _annotation_params_class(
    func: Callable[..., Any], param: inspect.Parameter
) -> type[stage_def.StageParams] | None:
    """Resolve a parameter's StageParams class without evaluating other annotations.

    String annotations are looked up by name in the function's module globals; a
    name defined only in the generated file (``extra_code``) resolves to None.
    """
    annotation = param.annotation
    if isinstance(annotation, str):
        annotation = getattr(func, "__globals__", {}).get(annotation.strip())
    return _params_class(annotation)


def register_test_stage(
    func: Callable[..., Any],
    name: str | None = None,
    params: type[stage_def.StageParams] | stage_def.StageParams | None = None,
    resources: dict[str, int | str] | None = None,
    *,
    deps: Mapping[str, DepValue] | None = None,
    pipeline: PipelineLike | None = None,
) -> compose.ArtifactHandle:
    """Register a stage for testing and return its output handle.

    The stage is recorded with the compositional API (``with Pipeline(...)`` and
    an ``@stage`` call) and merged into the target pipeline, so it is registered
    as ``<pipeline name>/<name>``. Outputs come from the function's return type;
    pivot chooses their paths (see ``get_output_path``).

    Args:
        func: The stage function, plain or already decorated with ``@stage``.
        name: Stage name (defaults to the function name).
        params: StageParams class or instance. ``None`` instantiates the
            function's StageParams parameter type with its defaults.
        resources: Scheduler resources written into the registered stage info.
        deps: Value for each dependency parameter: an ArtifactHandle (another
            stage's output, e.g. a previous ``register_test_stage`` return value,
            or ``pipeline.input(...)``), a list of handles, or a ``str`` naming an
            input file relative to the project root.
        pipeline: Pipeline to register with. Defaults to the test pipeline set
            via ``set_test_pipeline`` (the ``test_pipeline`` fixture binds it).

    Raises:
        ValidationError: If a stage with the same name is already registered.
        TypeError: If ``params`` is given but the function has no StageParams parameter.
    """
    target = pipeline or get_test_pipeline()
    stage_func = func if hasattr(func, "__wrapped__") else compose.stage(func)
    param_types = _param_types(func)
    params_arg = next(
        (n for n, hint in param_types.items() if _params_class(hint) is not None), None
    )
    if params is not None and params_arg is None:
        raise TypeError(
            f"register_test_stage: {_original_func(func).__name__} has no StageParams "
            + "parameter, but params was given"
        )

    with compose.Pipeline(target.name, root=target.root) as builder:
        kwargs = dict[str, object]()
        for param_name, value in (deps or {}).items():
            if isinstance(value, str):
                hint = param_types.get(param_name)
                input_type = hint if isinstance(hint, type) else None
                if input_type is None:
                    kwargs[param_name] = builder.input(value, path=value)
                else:
                    kwargs[param_name] = builder.input(value, t=input_type, path=value)
            else:
                kwargs[param_name] = value
        if params_arg is not None:
            if isinstance(params, stage_def.StageParams):
                kwargs[params_arg] = params
            elif params is not None:
                kwargs[params_arg] = params()
            else:
                params_cls = _params_class(param_types[params_arg])
                assert params_cls is not None
                kwargs[params_arg] = params_cls()
        handle = typing.cast("compose.ArtifactHandle", stage_func(**kwargs))
        node = handle._source
        if name is not None and isinstance(node, compose._StageNode):
            node.name = name

    stage_name = name or _original_func(func).__name__
    qualified_name = f"{target.name}/{stage_name}"
    if qualified_name in target.list_stages():
        raise exceptions.ValidationError(f"Stage '{qualified_name}' already registered")
    target.include(builder)
    if resources:
        target.get_stage(qualified_name)["resources"] = dict(resources)
    return handle


def get_output_path(
    stage_name: str, key: str | None = None, *, pipeline: PipelineLike | None = None
) -> pathlib.Path:
    """Return the workspace path pivot writes a registered stage's output to.

    Args:
        stage_name: Stage name, bare (``"train"``) or qualified (``"test/train"``).
        key: Output key for TypedDict-returning stages; ``None`` for single outputs.
        pipeline: Pipeline holding the stage. Defaults to the test pipeline.
    """
    target = pipeline or get_test_pipeline()
    qualified_name = stage_names.resolve_stage_name(stage_name, dict.fromkeys(target.list_stages()))
    outs = target.get_stage(qualified_name)["outs"]
    ref = next((out for out in outs if out.identity.key == key), None)
    if ref is None:
        available = [out.identity.key for out in outs]
        raise KeyError(f"Stage '{qualified_name}' has no output {key!r}. Available: {available}")
    workspace = store.WorkspaceStore(
        project_root=project.get_project_root(),
        pipeline_name=target.name,
        input_bindings=target.input_bindings,
    )
    return workspace.resolve_display_path(ref)


def create_pipeline_py(
    stages: Sequence[Callable[..., Any]],
    *,
    path: pathlib.Path | None = None,
    extra_imports: str = "",
    extra_code: str = "",
    names: Mapping[str, str] | None = None,
    args: Mapping[str, Mapping[str, str]] | None = None,
) -> pathlib.Path:
    """Create a pipeline.py file that defines the given stages in a ``Pipeline("test")``.

    This is useful for CLI tests that use runner.isolated_filesystem()
    and need discover_pipeline() to find stages.

    The file holds each stage function's source, wraps it in ``@stage`` if needed,
    and calls the stages in order inside ``with Pipeline("test") as pipeline:``,
    binding each call's handle to ``<stage name>_out`` so later stages can use it.

    Args:
        stages: Stage functions to include, in call order. Their source code
            is copied into the generated file.
        path: Directory to create pipeline.py in. Defaults to current directory.
        extra_imports: Additional import statements to include at the top.
        extra_code: Additional code to include (e.g., StageParams classes).
        names: Mapping of function names to stage names; the copied function
            is renamed, e.g. {"_helper_process": "process"} defines ``process``.
        args: Mapping of stage name to {parameter: Python expression} for the
            stage call, e.g. {"process": {"data": "pipeline.input('in.csv', path='in.csv')"}}
            or {"train": {"data": "process_out"}}. A StageParams parameter
            without an expression gets ``ParamsType()``; a parameter with a default
            may be omitted; any other parameter must have an expression.

    Returns:
        Path to the created pipeline.py file.

    Raises:
        ValueError: If a dependency parameter has no expression in ``args``.
    """
    target_dir = path or pathlib.Path.cwd()
    pipeline_path = target_dir / "pipeline.py"
    names = names or {}
    args = args or {}

    lines = [
        "from __future__ import annotations",
        "",
        "import pathlib",
        "from typing import Annotated, TypedDict",
        "",
        "from pivot import compose, loaders, stage_def",
        "from pivot.compose import Pipeline, stage",
    ]
    if extra_imports:
        lines.append(extra_imports)
    lines.append("")
    if extra_code:
        lines.append(extra_code)
        lines.append("")

    calls = list[str]()
    for func in stages:
        original = _original_func(func)
        func_name = original.__name__
        stage_name = names.get(func_name, func_name)
        source = textwrap.dedent(inspect.getsource(original))
        if stage_name != func_name:
            source = re.sub(
                rf"^(\s*(?:async\s+)?def\s+){re.escape(func_name)}\(",
                rf"\g<1>{stage_name}(",
                source,
                count=1,
                flags=re.MULTILINE,
            )
        lines.append(source)
        if not hasattr(func, "__wrapped__"):
            lines.append(f"{stage_name} = stage({stage_name})")
        lines.append("")

        stage_args = args.get(stage_name, {})
        call_args = list[str]()
        for param_name, param in inspect.signature(original).parameters.items():
            if param_name in stage_args:
                call_args.append(f"{param_name}={stage_args[param_name]}")
                continue
            params_cls = _annotation_params_class(original, param)
            if params_cls is not None:
                call_args.append(f"{param_name}={params_cls.__name__}()")
                continue
            if param.default is not inspect.Parameter.empty:
                continue
            raise ValueError(
                f"create_pipeline_py: stage {stage_name!r} parameter {param_name!r} "
                + "needs an expression in args"
            )
        calls.append(f"    {stage_name}_out = {stage_name}({', '.join(call_args)})")

    lines.append('with Pipeline("test") as pipeline:')
    lines.extend(calls or ["    pass"])
    lines.append("")

    pipeline_path.write_text("\n".join(lines))
    return pipeline_path


def discover_project(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    files: Mapping[str, str],
    pipeline: str,
) -> PipelineLike:
    """Write a project at `tmp_path/project`: `files` (each path relative to its root, with its
    text), `pipeline` as its `pipeline.py`, and `.git` and `.pivot` directories. Make it the
    project root and return the pipeline discovery finds there."""
    root = tmp_path / "project"
    for rel_path, text in {**files, "pipeline.py": pipeline}.items():
        (root / rel_path).parent.mkdir(parents=True, exist_ok=True)
        (root / rel_path).write_text(text)
    (root / ".git").mkdir(exist_ok=True)
    (root / ".pivot").mkdir(exist_ok=True)
    monkeypatch.setattr(project, "_project_root_cache", root)
    loaded = discovery.discover_pipeline(project_root=root)
    assert loaded is not None, "the project must define a pipeline"
    return loaded


def mount_table(
    monkeypatch: pytest.MonkeyPatch,
    path: pathlib.Path,
    mount_entries: Sequence[tuple[int, str, str] | tuple[int, str, str, str]],
) -> None:
    """Point the shared mount parser at mountinfo entries.

    Each entry is ``(device, mount point, filesystem type[, super options])``; options default
    to ``rw``. Tests use ``tmpfs,inode32`` and ``tmpfs,inode64`` to exercise tmpfs's inode rule.
    """
    lines = list[str]()
    for mount_id, entry in enumerate(mount_entries, 90):
        device, point, fstype, *rest = entry
        super_options = rest[0] if rest else "rw"
        escaped = point.replace("\\", "\\134").replace(" ", "\\040")
        lines.append(
            f"{mount_id} 1 {os.major(device)}:{os.minor(device)} / {escaped} "
            + f"rw,relatime shared:{mount_id} - {fstype} /dev/fake{mount_id} {super_options}\n"
        )
    path.write_text("".join(lines))
    monkeypatch.setattr(mounts_module, "_MOUNTINFO", path)


def hold_write_lock(state_dir: str, held: Event, release: Event) -> None:
    """Hold the state database's write lock in ``state_dir`` until ``release`` is set; the
    target of a spawned process, so it lives in this importable module."""
    db = state.StateDB(pathlib.Path(state_dir))
    with db, db._write_transaction(timeout=10.0):
        held.set()
        release.wait(120)
