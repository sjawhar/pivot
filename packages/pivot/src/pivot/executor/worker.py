"""Worker process execution for pipeline stages.

Functions that execute in separate processes via ProcessPoolExecutor.
Must be module-level and picklable.
"""

from __future__ import annotations

import _thread
import builtins
import codecs
import collections
import collections.abc as collections_abc
import contextlib
import ctypes
import dataclasses
import errno
import functools
import importlib
import importlib._bootstrap as importlib_bootstrap
import importlib.util
import inspect
import logging
import os
import pathlib
import queue
import random
import signal
import sys
import threading
import traceback
import warnings
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    NoReturn,
    NotRequired,
    TypedDict,
    cast,
    overload,
    override,
)

import psutil
import pydantic

from pivot import (
    config,
    exceptions,
    fingerprint,
    loaded_source,
    loaders,
    metrics,
    parameters,
    project,
    reads,
    run_history,
    skip,
    stage_def,
    stage_io,
    types,
)
from pivot import state_dir as state_dir_mod
from pivot.storage import artifact_lock, cache, listing, lock, naming, ownership, state
from pivot.storage import store as store_mod
from pivot.types import (
    DeferredWrites,
    FileHash,
    FileHashEntry,
    HashInfo,
    LockData,
    LogMessage,
    NoCommitRun,
    OutputMessage,
    OutputMessageKind,
    StageResult,
    StageStatus,
    StateChange,
    is_dir_hash,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence
    from multiprocessing import Queue
    from types import ModuleType, TracebackType


logger = logging.getLogger(__name__)


@dataclasses.dataclass(slots=True)
class _StageImports:
    """The user modules one stage call imported, under the covered thread rule.

    A covered thread is one inside a covered module's first import, or a thread a covered thread
    started during this stage (`_on_covered_thread`). What a covered thread imports belongs to
    the covered import, as a module-level import of it would have before the stage ran.
    """

    # The user modules the stage's manifest covers (`fingerprint.covered_user_modules`).
    covered: frozenset[str]
    # User modules the stage requested through `__import__` or `importlib.import_module` on a
    # thread that is not a covered thread, already resident or not. The same request on a covered
    # thread is not recorded, already resident or not, so a warm worker and a cold one agree.
    imported: set[str] = dataclasses.field(default_factory=set[str])
    # `sys.modules` keys added while the window below was open, however they were registered
    # (`exec_module`, `EntryPoint.load()`, `importlib.__import__`). The snapshot cannot tell
    # which thread registered a key, so this part of the rule is by time.
    added_in_window: set[str] = dataclasses.field(default_factory=set[str])
    # User source modules that became resident during the stage outside the window, by any
    # route: the `sys.modules` snapshot's count, set when the stage returns. A covered import
    # that requests such a module later does not remove it.
    new_outside_window: set[str] = dataclasses.field(default_factory=set[str])
    # The window is open while any thread is inside a covered module's first import or any
    # covered thread of this stage is running: `window_depth` counts those holds, and
    # `window_resident_before` holds the `sys.modules` keys at the moment it opened. Guarded by
    # `_stage_imports_lock`.
    window_depth: int = 0
    window_resident_before: frozenset[str] = frozenset()

    def uncovered(self) -> set[str]:
        """The user modules the stage imported that its manifest does not cover."""
        return (self.imported | self.new_outside_window) - self.covered


class _CoveredImportState(threading.local):
    """This thread's part in the covered thread rule: how deep it is inside a covered module's
    first import and for which stage, and the stage it is a covered thread of when a covered
    thread started it (`_run_as_covered_thread`)."""

    depth: int = 0
    importing_for: _StageImports | None = None
    covered_for: _StageImports | None = None
    # A Pivot-owned thread start is not user code and must not inherit the starting user's
    # coverage. `_start_pivot_thread` sets this only around its `thread.start()` call.
    starting_pivot_thread: bool = False


_stage_imports: _StageImports | None = None
_stage_imports_lock = threading.Lock()
_covered_import_state = _CoveredImportState()
_original_import = builtins.__import__
_original_import_module = importlib.import_module
_original_reload = importlib.reload
_original_start_new_thread = _thread.start_new_thread
_original_start_joinable_thread = _thread.start_joinable_thread
_replaced_thread_starts: dict[tuple[ModuleType, str], object] = {}
_hooks_installed = False


def _on_covered_thread(tracked: _StageImports) -> bool:
    """Whether this thread is a covered thread of the stage `tracked` records: inside a covered
    module's first import for that stage, or a thread a covered thread started during it."""
    state = _covered_import_state
    return (state.depth > 0 and state.importing_for is tracked) or state.covered_for is tracked


def _open_window(tracked: _StageImports) -> None:
    """Hold the stage's window for unhooked registrations open (`_StageImports.window_depth`)."""
    with _stage_imports_lock:
        if tracked.window_depth == 0 and _stage_imports is tracked:
            tracked.window_resident_before = frozenset(sys.modules)
        tracked.window_depth += 1


def _close_window(tracked: _StageImports) -> None:
    """Release one hold on the window. When the last is released, every `sys.modules` key added
    while it was open is recorded (`_StageImports.added_in_window`)."""
    with _stage_imports_lock:
        tracked.window_depth -= 1
        if tracked.window_depth == 0:
            if _stage_imports is tracked:
                tracked.added_in_window.update(set(sys.modules) - tracked.window_resident_before)
            tracked.window_resident_before = frozenset()


def _is_user_source_module(module: object) -> bool:
    """Whether `module` is user code loaded from a source file.

    A module created without a file (CPython's `pyexpat.errors`, Cython's `_cython_3_3_0`) has
    no source to fingerprint, so the stage-import check ignores it.
    """
    if not isinstance(module, type(sys)):
        return False
    return isinstance(vars(module).get("__file__"), str) and fingerprint.is_user_code(module)


def _package_chain(name: str) -> list[str]:
    """`name` and its packages: `a.b.c` gives `a`, `a.b`, `a.b.c`."""
    parts = name.split(".")
    return [".".join(parts[:index]) for index in range(1, len(parts) + 1)]


def _record_stage_import(names: Sequence[str]) -> None:
    """Record each of `names` that is resident user source for the running stage, whether or not
    it was resident before. On a covered thread of that stage nothing is recorded."""
    tracked = _stage_imports
    if tracked is None or _on_covered_thread(tracked):
        return
    found = [name for name in names if _is_user_source_module(sys.modules.get(name))]
    if not found:
        return
    with _stage_imports_lock:
        if _stage_imports is tracked:
            tracked.imported.update(found)


def _imports_only_covered(tracked: _StageImports, name: str, fromlist: Sequence[str]) -> bool:
    """Whether importing `name` (with `fromlist`, for a `from` import) first-imports at least
    one module, and every module it first-imports by name is one the stage's manifest covers.

    When `name` is resident, a `from` name its module does not hold yet is a submodule the
    import will load. When it is not, the `from` names cannot be told from attributes before the
    import runs; a submodule among them is recorded after the import (`_tracked_import`).
    """
    pending = [prefix for prefix in _package_chain(name) if prefix not in sys.modules]
    parent = sys.modules.get(name)
    if parent is not None:
        namespace: dict[str, object] = getattr(parent, "__dict__", {})
        pending.extend(
            f"{name}.{item}"
            for item in fromlist
            if f"{name}.{item}" not in sys.modules and item not in namespace
        )
    return bool(pending) and all(module in tracked.covered for module in pending)


@contextlib.contextmanager
def _covered_import(
    tracked: _StageImports, name: str | None, fromlist: Sequence[str]
) -> Generator[None]:
    """Make this thread a covered thread while it first-imports a covered module, and hold the
    stage's window for unhooked registrations open meanwhile (`_open_window`)."""
    state = _covered_import_state
    if state.depth == 0 and (name is None or not _imports_only_covered(tracked, name, fromlist)):
        yield
        return
    if state.depth == 0:
        state.importing_for = tracked
        _open_window(tracked)
    state.depth += 1
    try:
        yield
    finally:
        state.depth -= 1
        if state.depth == 0:
            state.importing_for = None
            _close_window(tracked)


def _run_as_covered_thread(tracked: _StageImports, function: Callable[[], object]) -> None:
    """Run a thread's body as a covered thread of the stage `tracked` records, then release the
    hold on that stage's window its start took."""
    _covered_import_state.covered_for = tracked
    try:
        function()
    finally:
        _covered_import_state.covered_for = None
        _close_window(tracked)


def _tracked_start_new_thread(
    function: Callable[..., object],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    /,
) -> int:
    """`_thread.start_new_thread` and its synonym `start_new`. A thread a covered thread of the
    running stage starts is a covered thread of that stage until its function returns, and holds
    the stage's window open until then."""
    call_kwargs = {} if kwargs is None else kwargs
    tracked = _stage_imports
    if (
        tracked is None
        or _covered_import_state.starting_pivot_thread
        or not callable(function)
        or not _on_covered_thread(tracked)
    ):
        return _original_start_new_thread(function, args, call_kwargs)

    def covered_function(*call_args: object, **named: object) -> None:
        _run_as_covered_thread(tracked, lambda: function(*call_args, **named))

    _open_window(tracked)
    try:
        return _original_start_new_thread(covered_function, args, call_kwargs)
    except BaseException:
        _close_window(tracked)
        raise


def _tracked_start_joinable_thread(
    function: Callable[[], object],
    handle: _thread._ThreadHandle | None = None,  # pyright: ignore[reportPrivateUsage] - `start_joinable_thread`'s own signature
    daemon: bool = True,
) -> _thread._ThreadHandle:  # pyright: ignore[reportPrivateUsage] - `start_joinable_thread`'s own signature
    """`_thread.start_joinable_thread`, under the rule `_tracked_start_new_thread` applies.

    `threading.Thread.start` starts every `threading.Thread` through it, so `threading.Timer`
    and `concurrent.futures.ThreadPoolExecutor` workers follow the rule too.
    """
    tracked = _stage_imports
    if (
        tracked is None
        or _covered_import_state.starting_pivot_thread
        or not callable(function)
        or not _on_covered_thread(tracked)
    ):
        return _original_start_joinable_thread(function, handle=handle, daemon=daemon)
    _open_window(tracked)
    try:
        return _original_start_joinable_thread(
            lambda: _run_as_covered_thread(tracked, function), handle=handle, daemon=daemon
        )
    except BaseException:
        _close_window(tracked)
        raise


# The thread starts each map to their replacement while hooks are installed: every `_thread`
# start function, plus `threading`'s `start_joinable_thread` binding that `Thread.start` calls.
_THREAD_START_HOOKS: dict[tuple[ModuleType, str], Callable[..., object]] = {
    (_thread, "start_new_thread"): _tracked_start_new_thread,
    (_thread, "start_new"): _tracked_start_new_thread,
    (_thread, "start_joinable_thread"): _tracked_start_joinable_thread,
    (threading, "_start_joinable_thread"): _tracked_start_joinable_thread,
}


def _import_package(globals: Mapping[str, object] | None) -> str | None:
    """The package CPython will use for a relative ``__import__``.

    CPython 3.13's ``importlib._bootstrap._calc___package__`` first reads ``__package__``, then
    ``__spec__.parent``, then falls back to ``__name__`` and ``__path__`` (CPython
    ``Lib/importlib/_bootstrap.py:1442-1459``). The hook must resolve the same module as the
    original import before it can record it.
    """
    if globals is None:
        return None
    calc_package = cast(
        "Callable[[Mapping[str, object]], str]",
        vars(importlib_bootstrap)["_calc___package__"],
    )
    return calc_package(globals)


def _tracked_import(
    name: str,
    globals: Mapping[str, object] | None = None,  # noqa: A002 - CPython's `__import__` signature
    locals: Mapping[str, object] | None = None,  # noqa: A002 - CPython's `__import__` signature
    fromlist: Sequence[str] | None = (),
    level: int = 0,
) -> object:
    tracked = _stage_imports
    if tracked is None:
        result = _original_import(name, globals, locals, fromlist or (), level)
        if not level:
            # Unpickling a stage imports the modules it refers to before the stage starts. Record
            # each one the recorder saw compile; `record_unseen_modules` judges the rest.
            loaded_source.record_modules(_package_chain(name))
        return result
    package = _import_package(globals) if level else None
    absolute: str | None = name
    if level:
        try:
            absolute = importlib.util.resolve_name(
                "." * level + name, package if isinstance(package, str) else None
            )
        except (ImportError, ValueError):
            absolute = None
    items = [item for item in fromlist or () if item != "*"]
    with _covered_import(tracked, absolute, items):
        result = _original_import(name, globals, locals, fromlist or (), level)
    if absolute is not None:
        names = _package_chain(absolute) + [f"{absolute}.{item}" for item in items]
        loaded_source.record_modules(names)
        _record_stage_import(names)
    return result


def _tracked_import_module(name: str, package: str | None = None) -> object:
    tracked = _stage_imports
    if tracked is None:
        result = _original_import_module(name, package)
        if not name.startswith("."):
            loaded_source.record_modules(_package_chain(name))
        return result
    try:
        absolute = importlib.util.resolve_name(name, package) if name.startswith(".") else name
    except (ImportError, ValueError):
        absolute = None
    with _covered_import(tracked, absolute, []):
        result = _original_import_module(name, package)
    if absolute is not None:
        loaded_source.record_modules(_package_chain(absolute))
        _record_stage_import(_package_chain(absolute))
    return result


def _tracked_reload(module: ModuleType) -> ModuleType:
    """Reload `module`; the discovery recorder wraps the reload's loader and refreshes its record."""
    result = _original_reload(module)
    tracked = _stage_imports
    if tracked is not None:
        _record_stage_import([result.__name__])
    return result


def install_import_hooks() -> None:
    """Route `builtins.__import__` and `importlib.import_module` through the stage-import check,
    and every thread start in `_THREAD_START_HOOKS` through the covered thread rule.
    The worker initializer (`executor.core._init_worker`) calls this; nothing else in Pivot's
    own process installs it. The hooks record nothing while no stage is running.
    """
    global _original_import, _original_import_module, _original_reload, _hooks_installed
    global _original_start_new_thread, _original_start_joinable_thread
    if _hooks_installed:
        return
    _original_import = builtins.__import__
    _original_import_module = importlib.import_module
    _original_reload = importlib.reload
    _original_start_new_thread = _thread.start_new_thread
    _original_start_joinable_thread = _thread.start_joinable_thread
    builtins.__import__ = _tracked_import
    importlib.import_module = _tracked_import_module
    importlib.reload = _tracked_reload
    for (module, name), hook in _THREAD_START_HOOKS.items():
        _replaced_thread_starts[module, name] = vars(module)[name]
        vars(module)[name] = hook
    _hooks_installed = True


def _uninstall_import_hooks() -> None:
    global _hooks_installed
    builtins.__import__ = _original_import
    importlib.import_module = _original_import_module
    importlib.reload = _original_reload
    for (module, name), original in _replaced_thread_starts.items():
        vars(module)[name] = original
    _replaced_thread_starts.clear()
    _hooks_installed = False


def _start_pivot_thread(thread: threading.Thread) -> None:
    """Start a thread Pivot owns without letting it inherit user-code coverage.

    A Pivot thread can start while a covered import is running (the pipe reader is the critical
    case). It remains outside the covered thread rule, so later user loads still count. All
    Pivot-owned starts in a worker route through this helper.
    """
    covered_state = _covered_import_state
    was_starting_pivot_thread = covered_state.starting_pivot_thread
    covered_state.starting_pivot_thread = True
    try:
        thread.start()
    finally:
        covered_state.starting_pivot_thread = was_starting_pivot_thread


@contextlib.contextmanager
def _track_stage_imports(covered: frozenset[str]) -> Generator[_StageImports]:
    """Record the user modules one stage call imports.

    A stage executed in a process without the worker initializer (`execute_stage` called
    directly) gets the hooks for this call only. When the stage returns, every user source module
    that became resident during it outside the window is recorded
    (`_StageImports.new_outside_window`). A covered thread still running then holds the window
    open, so the keys added since it opened count as added in it; that thread is not a covered
    thread of a later stage.
    """
    global _stage_imports
    installed_here = not _hooks_installed
    if installed_here:
        install_import_hooks()
    tracked = _StageImports(covered=covered)
    with _stage_imports_lock:
        assert _stage_imports is None, "Worker cannot execute two stages concurrently"
        resident_before = frozenset(sys.modules)
        _stage_imports = tracked
    try:
        yield tracked
    finally:
        with _stage_imports_lock:
            resident_after = set(sys.modules)
            if tracked.window_depth:
                tracked.added_in_window.update(resident_after - tracked.window_resident_before)
            tracked.window_resident_before = frozenset()
            _stage_imports = None
        tracked.new_outside_window.update(
            name
            for name in resident_after - resident_before - tracked.added_in_window
            if _is_user_source_module(sys.modules.get(name))
        )
        if installed_here:
            _uninstall_import_hooks()


def _untrusted_modules(
    stage_info: WorkerStageInfo,
    fingerprint_manifest: dict[str, str],
    stage_imports: _StageImports,
    project_root: pathlib.Path,
) -> tuple[list[str], list[str]]:
    """The user modules this stage used whose code it cannot vouch for: (edited, unreadable).

    What it used: the modules its manifest hashes, the module that defines the stage, and the
    modules it imported while it ran. A `@no_fingerprint` stage's code is its declared files, so
    for it the modules loaded from those files count instead of its imports.

    A worker first judges every user module it loaded that no recorder saw compile, as discovery
    does (`loaded_source.record_unseen_modules`): a module a stage's module loaded with a loader
    it called directly (`SourceFileLoader(...).load_module()`, `spec_from_loader`) holds the file
    as this worker read it, which may be after an edit the coordinator's record predates.
    """
    loaded_source.record_user_modules()
    if loaded_source.is_worker():
        loaded_source.record_unseen_modules()
    used = set[str]()
    manifest = fingerprint_manifest
    if stage_info["no_fingerprint"]:
        files = set[str]()
        for key in fingerprint_manifest:
            if not key.startswith("file:"):
                continue
            candidate = pathlib.Path(key.removeprefix("file:"))
            path = candidate if candidate.is_absolute() else project_root / candidate
            if path.is_relative_to(project_root):
                files.add(os.path.realpath(path))
        used.update(loaded_source.modules_from_files(files))
        manifest = {}
    else:
        stage_func = inspect.unwrap(stage_info["func"])
        stage_file = inspect.getsourcefile(stage_func)
        if (
            loaded_source.coordinator_records(stage_func.__module__)
            or isinstance(stage_file, str)
            and pathlib.Path(stage_file).resolve().is_relative_to(project_root)
        ):
            used.add(stage_func.__module__)
        used.update(stage_imports.imported)
        used.update(stage_imports.new_outside_window)
        used.update(stage_imports.added_in_window)
    return loaded_source.check_stage(
        manifest,
        used,
        stage_info["loaded_sources_token"],
        stage_info["loaded_sources"] if "loaded_sources" in stage_info else None,
    )


# How long a stopped worker's running stage may take to unwind before the worker exits.
TERMINATION_GRACE_S = 10.0


class WorkerTerminated(KeyboardInterrupt):  # noqa: N818 - names the event, as KeyboardInterrupt does
    """Raised in a worker's running stage when its run is cancelled or its Pivot process dies.

    It is a KeyboardInterrupt, as Ctrl-C raises at a terminal: a stage's ``except Exception``
    does not stop it, and its ``except KeyboardInterrupt`` cleanup, ``finally`` blocks and
    context managers run. A stage that catches it and carries on has its result discarded.
    """


@dataclasses.dataclass(slots=True)
class _StopState:
    """Process-wide stop state of a worker, set by its signal handlers and its watchdog.

    Signals and the parent watchdog act on the whole process, so this state is too.
    """

    stage_running: bool = False
    # The running stage's run was cancelled: its result must not be committed.
    stop_requested: bool = False
    # WorkerTerminated was raised into the running stage; never raise it again mid-cleanup.
    interrupted: bool = False
    # The worker exits once the running stage has unwound (SIGTERM or parent death).
    exit_requested: bool = False
    exit_timer_started: bool = False
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


_stop = _StopState()

# tgkill's system call number for each process ABI, keyed by CPython's platform triplet
# (`sys.implementation._multiarch`; CPython's Misc/platform_triplet.c names them). Each number is
# the line of Linux v7.0 cited beside it, under https://github.com/torvalds/linux/blob/v7.0/.
_TGKILL_SYSCALLS = {
    f"{arch}-linux-{libc}": number
    for arch, libcs, number in (
        ("x86_64", ("gnu", "musl"), 234),  # arch/x86/entry/syscalls/syscall_64.tbl#L246
        ("i386", ("gnu", "musl"), 270),  # arch/x86/entry/syscalls/syscall_32.tbl#L285
        # scripts/syscall.tbl#L167 (arch/arm64/tools/syscall_64.tbl is a symbolic link to it)
        ("aarch64", ("gnu", "musl"), 131),
        # arch/arm/tools/syscall.tbl#L286 (EABI, the only ABI these triplets name)
        ("arm", ("gnueabihf", "gnueabi", "musleabihf", "musleabi"), 268),
        ("powerpc64le", ("gnu", "musl"), 250),  # arch/powerpc/kernel/syscalls/syscall.tbl#L328
        ("s390x", ("gnu", "musl"), 241),  # arch/s390/kernel/syscalls/syscall.tbl#L198
        # scripts/syscall.tbl#L167, the table riscv and loongarch use (scripts/Makefile.asm-headers#L20;
        # neither arch/riscv nor arch/loongarch kernel/Makefile.syscalls sets another)
        ("riscv64", ("gnu", "musl"), 131),
        ("loongarch64", ("gnu", "gnuf32", "gnusf", "musl", "muslf32", "muslsf"), 131),
    )
    for libc in libcs
}


@functools.cache
def _tgkill() -> Callable[[int, int, int], int] | None:
    """``tgkill(tgid, tid, sig)`` through the C library's ``syscall()``, which glibc and musl
    both export, with tgkill's number for this process's ABI. None where that number is not
    known (not Linux, or an ABI not in ``_TGKILL_SYSCALLS``), logged once: there a stopped
    worker gets ``os.kill``, Pivot's behaviour before tgkill, and the race ``terminate``
    describes remains."""
    abi: object = getattr(sys.implementation, "_multiarch", None)
    number = _TGKILL_SYSCALLS.get(abi) if isinstance(abi, str) else None
    if number is None:
        logger.debug(
            "No tgkill system call number for %s (machine %s): stopped workers get os.kill",
            abi,
            os.uname().machine,
        )
        return None
    syscall = ctypes.CDLL(None, use_errno=True).syscall
    syscall.restype = ctypes.c_long
    return lambda tgid, tid, sig: syscall(
        ctypes.c_long(number), ctypes.c_long(tgid), ctypes.c_long(tid), ctypes.c_long(sig)
    )


def terminate(pid: int) -> None:
    """Send worker ``pid`` SIGTERM (``_stop_worker``) at its main thread, where its stage runs.

    A signal sent to a process (``os.kill``) can reach any of its threads. Linux gives it to
    the main thread only while that thread is on a CPU or has no signal pending. Once Ctrl-C
    has woken the main thread and load keeps it waiting for a CPU, Linux gives SIGTERM to
    another thread, such as one of the OpenBLAS threads NumPy starts on import, and that thread
    takes the Ctrl-C too. CPython's handler there only marks the signals for the main thread,
    which finds no signal when it runs and goes back to sleep: the stage never stops, and Pivot
    kills the worker after ``TERMINATION_GRACE_S`` with the stage's cleanup not run. tgkill
    sends the signal to one thread, which alone can take it.

    Where ``_tgkill`` knows no system call number, ``os.kill`` sends it. On macOS that is
    enough: XNU gives a signal sent to a process to the first of its threads that does not block
    it ("We attempt to deliver a proc-wide signal to the first thread in the task",
    ``get_signalthread``, https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/kern/kern_sig.c#L1935-L1974),
    and adds each new thread at the end of that list (bsd/kern/kern_fork.c#L1368), so the main
    thread comes first. Read from the source, not run on macOS.

    A tgkill that fails for any reason but a process that is gone (ESRCH, raised as
    ProcessLookupError) falls back to ``os.kill``, with a warning naming the worker, so a stop
    is never left unsent.

    A stage that keeps SIGTERM blocked on the main thread holds this signal pending until
    ``WorkerPool`` SIGKILLs the worker after the grace; with ``os.kill`` another thread could
    take it and the worker exit sooner. Its cleanup runs either way, from Ctrl-C's SIGINT.
    """
    tgkill = _tgkill()
    if tgkill is None:
        os.kill(pid, signal.SIGTERM)
        return
    # A process's main thread has the process's id as its thread id. tgkill to a process whose
    # main thread has exited while others run reaches no thread; a loky worker never gets there,
    # because it leaves its main thread only through os._exit, which ends the whole process.
    if tgkill(pid, pid, signal.SIGTERM) == 0:
        return
    error = ctypes.get_errno()
    if error == errno.ESRCH:
        raise ProcessLookupError(error, os.strerror(error))
    logger.warning(
        "tgkill could not send SIGTERM to worker %s (%s); sending it to the process",
        pid,
        os.strerror(error),
    )
    os.kill(pid, signal.SIGTERM)


def kill_descendants(pid: int) -> None:
    """SIGKILL every live descendant of process ``pid``. A process that detached from that
    tree (double-forked, or orphaned when its parent exited) is not one."""
    try:
        descendants = psutil.Process(pid).children(recursive=True)
    except psutil.Error:
        return
    for descendant in descendants:
        with contextlib.suppress(psutil.Error):
            descendant.kill()


def _exit_worker(code: int) -> NoReturn:
    """Exit this worker, first killing the processes still descended from it, which would
    otherwise be left to init (a stage's ``Popen`` with no cleanup, or one whose cleanup was
    cut off)."""
    kill_descendants(os.getpid())
    os._exit(code)


def _exit_after_grace() -> None:
    threading.Event().wait(TERMINATION_GRACE_S)
    _exit_worker(1)


def _start_exit_timer() -> None:
    """Exit this worker ``TERMINATION_GRACE_S`` from now, whatever its running stage does."""
    with _stop.lock:
        if _stop.exit_timer_started:
            return
        _stop.exit_timer_started = True
    _start_pivot_thread(
        threading.Thread(target=_exit_after_grace, daemon=True, name="pivot-stop-grace")
    )


def _interrupt_stage(_signum: int, _frame: object) -> None:
    """SIGINT handler of a worker: Ctrl-C at a terminal, which reaches Pivot and its workers at
    once. Interrupt the running stage once, so its cleanup runs, and keep the worker up until
    Pivot stops it: loky takes a worker that exits on its own for a broken pool and SIGKILLs
    every other worker, cutting their cleanup off."""
    if not _stop.stage_running or _stop.interrupted:
        return
    _stop.stop_requested = True
    _stop.interrupted = True
    raise WorkerTerminated


def _stop_worker(_signum: int, _frame: object) -> None:
    """SIGTERM handler of a worker: interrupt the running stage once, exit once it has unwound,
    and exit regardless after ``TERMINATION_GRACE_S``. An idle worker exits at once."""
    _stop.stop_requested = True
    _stop.exit_requested = True
    _start_exit_timer()
    if not _stop.stage_running:
        _exit_worker(0)
    if _stop.interrupted:
        return
    _stop.interrupted = True
    raise WorkerTerminated


def _raise_if_stopped() -> None:
    """Refuse to commit the result of a stage whose run was cancelled, which can only be here
    because the stage caught ``WorkerTerminated`` and carried on."""
    if _stop.stop_requested:
        raise WorkerTerminated


def unwind_when_stopped(parent_pid: int) -> None:
    """Loky worker initializer: stop this worker when it is told to or its parent dies.

    SIGTERM (``WorkerPool.hard_cancel``, sent at the main thread by ``terminate``) or SIGINT
    (Ctrl-C at a terminal) raises ``WorkerTerminated`` in the running stage, so its cleanup runs
    (stopping the processes or containers it started); after SIGTERM the worker then exits.
    When the Pivot process that started the worker dies, as when it is SIGKILLed, a watchdog
    thread sends this worker's main thread SIGTERM (a signal to that thread interrupts a
    blocking call there) and exits the worker after ``TERMINATION_GRACE_S`` itself, in case the
    stage replaced the SIGTERM handler or is blocked where no signal reaches it. Every exit
    first kills the processes still descended from the worker.

    A SIGINT the worker inherited as ignored (pivot run as a background job of a
    non-interactive shell) stays ignored, as it does for pivot itself.
    """
    signal.signal(signal.SIGTERM, _stop_worker)
    if signal.getsignal(signal.SIGINT) is not signal.SIG_IGN:
        signal.signal(signal.SIGINT, _interrupt_stage)
    main_thread = threading.main_thread().ident
    assert main_thread is not None

    def _watch_parent() -> None:
        poll = threading.Event()
        while os.getppid() == parent_pid:
            poll.wait(1)
        _stop.exit_requested = True  # first: a stage starting now keeps stop_requested set
        _stop.stop_requested = True
        _start_exit_timer()
        signal.pthread_kill(main_thread, signal.SIGTERM)

    _start_pivot_thread(
        threading.Thread(target=_watch_parent, daemon=True, name="pivot-parent-watch")
    )


class _QueueLoggingHandler(logging.Handler):
    """Logging handler that sends records to the output queue.

    Installed per-task in execute_stage() to capture log messages from worker processes.
    This ensures logging (e.g., stale lock warnings) appears in TUI Logs panel instead
    of corrupting the display by writing to inherited stderr.
    """

    _stage_name: str
    _queue: Queue[OutputMessage]

    def __init__(
        self, stage_name: str, output_queue: Queue[OutputMessage], level: int = logging.INFO
    ) -> None:
        super().__init__(level=level)
        self._stage_name = stage_name
        self._queue = output_queue
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    @override
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with contextlib.suppress(queue.Full, ValueError, OSError):
                self._queue.put(
                    LogMessage(
                        kind=OutputMessageKind.LOG, stage=self._stage_name, line=msg, is_stderr=True
                    ),
                    block=False,
                )
        except Exception:
            pass  # Never raise from emit - could cause recursion

    @override
    def handleError(self, record: logging.LogRecord) -> None:
        pass  # Suppress default stderr writing to prevent TUI corruption


@contextlib.contextmanager
def _queue_logging(stage_name: str, output_queue: Queue[OutputMessage]) -> Generator[None]:
    """Context manager to capture logging to the output queue.

    Removes stream handlers that write to stdout/stderr (which would corrupt TUI)
    and installs a handler that sends log records to the output queue instead.
    Original handlers are restored on exit.

    Note: Handler manipulation is not fully atomic, but this is acceptable because:
    1. Worker processes are single-threaded for stage execution
    2. Individual addHandler/removeHandler calls are internally synchronized
    3. Worst case is a brief window where a log goes to the wrong handler
    """
    handler = _QueueLoggingHandler(stage_name, output_queue)
    root_logger = logging.getLogger()

    # Remove existing stderr/stdout handlers to prevent TUI corruption
    removed_handlers = list[logging.Handler]()
    for h in root_logger.handlers[:]:
        if isinstance(h, logging.StreamHandler):
            stream_handler = cast("logging.StreamHandler[Any]", h)
            if hasattr(stream_handler, "stream") and stream_handler.stream in (
                sys.stderr,
                sys.stdout,
            ):
                root_logger.removeHandler(stream_handler)
                removed_handlers.append(stream_handler)

    root_logger.addHandler(handler)
    try:
        yield
    finally:
        root_logger.removeHandler(handler)
        for h in removed_handlers:
            root_logger.addHandler(h)


class WorkerStageInfo(TypedDict):
    """Stage info subset passed to worker processes."""

    func: Callable[..., Any]
    fingerprint: dict[str, str]
    deps: dict[str, types.ArtifactRef]
    outs: list[types.ArtifactRef]
    store_spec: store_mod.StoreSpec
    signature: inspect.Signature | None
    params: stage_def.StageParams | None
    variant: str | None
    overrides: parameters.ParamsOverrides
    run_id: str
    force: bool
    no_commit: bool
    params_arg_name: str | None
    project_root: pathlib.Path
    state_dir: pathlib.Path
    state_dir_resolution: state_dir_mod.StateDirResolution
    collection_params: dict[str, str]
    incremental_inputs: dict[str, types.ArtifactIdentity]
    # The fingerprint downgraded a refusal to a warning: the run cache is neither read nor written.
    unsafe_fingerprint: bool
    # The stage is `@no_fingerprint`: its declared files are its code, so imports are not checked.
    no_fingerprint: bool
    # `loaded_source.token()` of the user code the coordinator loaded when it sent the stage. The
    # worker got those digests once, when it started (`core._init_worker`); `loaded_sources`
    # carries them again only when the coordinator loaded code after that.
    loaded_sources_token: str
    loaded_sources: NotRequired[dict[str, str]]


def _make_result(
    status: Literal[StageStatus.RAN, StageStatus.CACHED, StageStatus.FAILED],
    reason: str,
    ring_buffer: _OutputRingBuffer,
    input_hash: str | None = None,
    file_hash_entries: list[FileHashEntry] | None = None,
) -> StageResult:
    """Build StageResult with collected metrics for cross-process transfer."""
    result = StageResult(
        status=status,
        reason=reason,
        input_hash=input_hash,
        output_lines=ring_buffer.snapshot(),
        metrics=metrics.get_entries(),
    )
    if file_hash_entries:
        result["deferred_writes"] = DeferredWrites(file_hash_entries=file_hash_entries)
    return result


def _execute_stage(
    stage_name: str,
    stage_info: WorkerStageInfo,
    cache_dir: pathlib.Path,
    output_queue: Queue[OutputMessage],
) -> StageResult:
    """Worker function executed in separate process. Must be module-level for pickling.

    Flag interactions:
    - --force: Always run stage, even if skip detection says it's unchanged
    """
    # Clear metrics at start - each stage collects its own metrics
    metrics.clear()
    # ``cache_dir`` is the cache root the engine resolved from config (the directory holding
    # ``files/``), as spelled there: the naming lock lives in it, the same file `pivot gc`
    # locks. The store's object root is resolved, so a ``files`` symlink would move its parent.
    naming_root = cache_dir
    ring_buffer = _OutputRingBuffer()
    state_dir = stage_info["state_dir"]
    project_root = stage_info["project_root"]
    project_state_dir = state_dir_mod.locate_state_dir(project_root)
    store = store_mod.store_from_spec(stage_info["store_spec"])

    def register_before_restore() -> None:
        """Register the project in the cache the store restores from, before a skipped stage
        links that cache's objects outside a publishing unit. `execute_stage` is exported, so a
        caller without the engine's registration restores here; the store spec names the
        cache's object root as spelled in config (the store itself resolves it)."""
        naming.register_use(
            pathlib.Path(stage_info["store_spec"]["cache_dir"]).parent,
            project_root,
            project_state_dir,
        )

    # Set project root cache explicitly - workers in reusable pool may have
    # stale cache from previous execution in different project/test.
    project._project_root_cache = project_root  # pyright: ignore[reportPrivateUsage]

    # Ensure worker has correct cwd for this stage (workers in reusable pool
    # may have stale cwd from previous execution in different project).
    # _queue_logging captures log messages to the output queue (for TUI display).
    with contextlib.chdir(project_root), _queue_logging(stage_name, output_queue):
        no_commit = stage_info["no_commit"]

        production_lock = lock.StageLock(stage_name, lock.get_stages_dir(stage_info["state_dir"]))
        current_fingerprint = stage_info["fingerprint"]
        stage_outs = stage_info["outs"]

        params_instance = stage_info["params"]
        overrides = stage_info["overrides"]
        try:
            current_params = parameters.get_effective_params(params_instance, stage_name, overrides)
            if params_instance is not None:
                params_instance = parameters.apply_overrides(params_instance, stage_name, overrides)
        except pydantic.ValidationError as e:
            return _make_result(
                StageStatus.FAILED,
                f"Invalid params override in params.yaml: {e.error_count()} validation error(s)",
                ring_buffer,
            )

        # Acquire artifact locks (READ on deps, WRITE on outs)
        deps_info = stage_info["deps"]
        outs_info = stage_info["outs"]
        lock_requests = artifact_lock.expand_lock_requests(deps_info, outs_info, project_root)
        lock_service = artifact_lock.LocalFlockLockService(stage_info["state_dir"] / "locks")

        def _on_lock_status(_key: str, _mode: artifact_lock.LockMode, _elapsed: float) -> None:
            with contextlib.suppress(queue.Full):
                output_queue.put_nowait(
                    StateChange(
                        kind=OutputMessageKind.STATE, stage=stage_name, state="waiting_on_lock"
                    )
                )

        lock_handle = lock_service.acquire_many(lock_requests, on_status=_on_lock_status)

        input_hash: str | None = None
        try:
            with lock_handle:
                with contextlib.suppress(queue.Full):
                    output_queue.put_nowait(
                        StateChange(kind=OutputMessageKind.STATE, stage=stage_name, state="running")
                    )
                with stage_io.PIVOT_IO:
                    lock_data = production_lock.read(
                        cache_dir=store_cache_dir(store), warn_old_format=False
                    )

                with _open_state_db(state_dir) as state_db:
                    # The *.pvt files the stage claims in its outputs: its lock's, and its
                    # nocommit: record's until a lock is written (a restore or a later run
                    # replaces them, as it does the lock's).
                    claims = stage_claims(
                        store,
                        stage_outs,
                        lock_data,
                        state_db.get_no_commit_record(stage_name),
                        project_root,
                        functools.partial(
                            ownership.record_members, project_root, state_dir, stage_name
                        ),
                        stage_lock=production_lock,
                    )
                    outputs_unusable: str | None = None
                    if lock_data is not None and not stage_info["force"]:
                        out_paths = [types.identity_key(out.identity) for out in stage_outs]
                        if can_skip_via_generation(
                            stage_name=stage_name,
                            fingerprint=stage_info["fingerprint"],
                            deps=stage_info["deps"],
                            outs_paths=out_paths,
                            current_params=current_params,
                            lock_data=lock_data,
                            state_db=state_db,
                            store=store,
                        ):
                            input_hash = run_history.stage_input_hash(
                                current_fingerprint,
                                current_params,
                                stage_info["deps"].values(),
                                lock_data["dep_hashes"],
                                stage_outs,
                            )
                            restored_entries = list[FileHashEntry]()
                            restoration = restore_outputs_to_hashes(
                                store,
                                stage_outs,
                                lock_data["output_hashes"],
                                state_db,
                                state_dir=project_state_dir,
                                listing_lock=None
                                if "older_format" in lock_data
                                else production_lock,
                                report_changes=True,
                                file_hash_entries=restored_entries,
                                claims=claims,
                                register=register_before_restore,
                            )
                            if restoration.available:
                                return _make_result(
                                    StageStatus.CACHED,
                                    "unchanged (generation)",
                                    ring_buffer,
                                    input_hash=input_hash,
                                    file_hash_entries=restored_entries,
                                )
                            outputs_unusable = _outputs_unusable_reason(restoration)

                    hashed_dirs = dict[str, listing.HashedDirectory]()
                    dep_hashes, missing, unreadable, file_hash_entries = hash_dependencies(
                        stage_info["deps"], store, state_db, hashed_dirs=hashed_dirs
                    )

                    if missing:
                        return _make_result(
                            StageStatus.FAILED,
                            f"missing deps: {', '.join(missing)}",
                            ring_buffer,
                        )

                    if unreadable:
                        return _make_result(
                            StageStatus.FAILED,
                            f"unreadable deps: {', '.join(unreadable)}",
                            ring_buffer,
                        )

                    skip_reason, run_reason, input_hash, run_cache_allowed = _check_skip_or_run(
                        stage_info,
                        lock_data,
                        current_fingerprint,
                        current_params,
                        dep_hashes,
                        unavailable_reads=None
                        if stage_info["force"]
                        else reads.lock_unavailable(lock_data, store_cache_dir(store)),
                    )

                    if outputs_unusable is not None and skip_reason is not None:
                        skip_reason = None
                        run_reason = outputs_unusable

                    # Override skip decision if force flag is set
                    if stage_info["force"] and skip_reason is not None:
                        skip_reason = None
                        run_reason = "forced"

                    if skip_reason is not None and lock_data is not None:
                        restored_entries = list[FileHashEntry]()
                        restoration = restore_outputs_to_hashes(
                            store,
                            stage_outs,
                            lock_data["output_hashes"],
                            state_db,
                            state_dir=project_state_dir,
                            listing_lock=None if "older_format" in lock_data else production_lock,
                            report_changes=True,
                            file_hash_entries=restored_entries,
                            claims=claims,
                            register=register_before_restore,
                        )
                        if restoration.available:
                            # The dependencies read to decide, and the outputs read or restored
                            return _make_result(
                                StageStatus.CACHED,
                                skip_reason,
                                ring_buffer,
                                input_hash=input_hash,
                                file_hash_entries=file_hash_entries + restored_entries,
                            )
                        run_reason = _outputs_unusable_reason(restoration)

                    # Check run cache for previously executed configuration (skip if forcing)
                    run_cache_wanted = (
                        bool(run_reason) and not stage_info["force"] and run_cache_allowed
                    )
                    legacy_input_hash = (
                        older_pivot_input_hash(
                            current_fingerprint,
                            current_params,
                            stage_info["deps"].values(),
                            _to_identity_hashes(dep_hashes),
                            stage_outs,
                            hashed_dirs,
                            store_cache_dir(store),
                        )
                        if run_cache_wanted
                        else None
                    )
                    if (
                        run_cache_wanted
                        and _run_cache_row(
                            state_db, stage_name, input_hash, legacy_input_hash, stage_outs
                        )
                        is not None
                    ):
                        # Naming-lock unit (shared): the run-cache restore checks whether
                        # objects exist and creates workspace links, then the lock file names
                        # them. gc cannot delete while this unit is open. With no row to
                        # restore, nothing is named and no unit is taken; the engine registered
                        # the project before any stage ran (`Engine._orchestrate_execution`).
                        with naming.naming_unit(
                            naming_root, project=(project_root, project_state_dir)
                        ):
                            run_cache_skip = _try_skip_via_run_cache_with_store(
                                stage_name,
                                input_hash,
                                stage_outs,
                                store,
                                state_db,
                                state_dir=project_state_dir,
                                legacy_input_hash=legacy_input_hash,
                                claims=claims,
                            )
                            if run_cache_skip is not None:
                                if no_commit:
                                    # Nothing is committed; the coordinator saves only the file
                                    # hashes, the dependencies read and the outputs restored.
                                    return _make_result(
                                        StageStatus.CACHED,
                                        "unchanged (run cache)",
                                        ring_buffer,
                                        input_hash=input_hash,
                                        file_hash_entries=file_hash_entries
                                        + run_cache_skip["file_hash_entries"],
                                    )
                                new_lock_data = LockData(
                                    code_manifest=current_fingerprint,
                                    params=current_params,
                                    dep_hashes=_to_identity_hashes(
                                        dict(sorted(dep_hashes.items()))
                                    ),
                                    output_hashes=_to_identity_hashes(
                                        dict(sorted(run_cache_skip["output_hashes"].items()))
                                    ),
                                    output_paths=lock_output_paths(stage_outs, store, project_root),
                                )
                                if run_cache_skip["reads"]:
                                    new_lock_data["reads"] = run_cache_skip["reads"]
                                _raise_if_stopped()
                                deferred = _commit_lock_and_build_deferred(
                                    stage_info,
                                    new_lock_data,
                                    input_hash,
                                    run_cache_skip["output_hashes"],
                                    production_lock,
                                    state_db,
                                    store=store,
                                    hashed_dirs=hashed_dirs,
                                    file_hash_entries=file_hash_entries
                                    + run_cache_skip["file_hash_entries"],
                                    increment_outputs=run_cache_skip["outputs_restored"],
                                )
                                return StageResult(
                                    status=StageStatus.CACHED,
                                    reason="unchanged (run cache)",
                                    input_hash=input_hash,
                                    output_lines=ring_buffer.snapshot(),
                                    metrics=metrics.get_entries(),
                                    deferred_writes=deferred,
                                )

                covered_modules = frozenset(
                    fingerprint.covered_user_modules(
                        current_fingerprint,
                        {types.identity_key(out.identity) for out in stage_outs},
                    )
                )
                with (
                    _track_stage_imports(covered_modules) as stage_imports,
                    reads.recording(store_cache_dir(store)) as recorded,
                ):
                    written = _run_stage_function_with_store(
                        stage_info["func"],
                        stage_name,
                        output_queue,
                        ring_buffer,
                        params_instance,
                        stage_info["deps"],
                        stage_outs,
                        store,
                        stage_info["params_arg_name"],
                        stage_info["collection_params"],
                        stage_info["incremental_inputs"],
                        None if lock_data is None else lock_data["output_hashes"],
                        state_dir,
                        project_state_dir,
                        project_root,
                        claims,
                    )
                output_paths = written["paths"]
                read_hashes = recorded.hashes()
                fetched = recorded.fetched()
                # A @no_fingerprint stage declares its code as files (`code_deps`); that
                # declaration is its contract, so its imports are not checked.
                uncovered = (
                    set[str]() if stage_info["no_fingerprint"] else stage_imports.uncovered()
                )
                lock_manifest = current_fingerprint
                # Under --no-commit no lock is written now: `pivot commit` writes this run's
                # manifest, marks included, when it records these same outputs.
                commit_note = (
                    " `pivot commit` keeps that when it records these outputs." if no_commit else ""
                )
                if uncovered:
                    modules = ", ".join(sorted(uncovered))
                    logger.warning(
                        "Stage '%s' imported user modules its fingerprint does not cover: %s. "
                        + "Its result will not be reused; the next run reruns it.%s Fix: import "
                        + "them statically and read them by name in the stage's own code "
                        + "(`import pkg.mod`, then `pkg.mod.NAME`, or pass `pkg.mod` itself), or "
                        + "move the values into StageParams.",
                        stage_name,
                        modules,
                        commit_note,
                    )
                    lock_manifest = {
                        **current_fingerprint,
                        f"{skip.UNCOVERED_IMPORT_KEY_PREFIX}{modules}": "",
                    }
                edited, unreadable = _untrusted_modules(
                    stage_info, current_fingerprint, stage_imports, project_root
                )
                if edited:
                    edited_modules = ", ".join(edited)
                    logger.warning(
                        "Stage '%s' imported user modules that changed after pivot loaded them: "
                        + "%s. Its fingerprint records the code pivot loaded, so its result will "
                        + "not be reused; the next run reruns it.%s A watch or server session "
                        + "loads a changed module outside the project root when pivot starts "
                        + "again.",
                        stage_name,
                        edited_modules,
                        commit_note,
                    )
                    lock_manifest = {
                        **lock_manifest,
                        f"{skip.EDITED_SOURCE_KEY_PREFIX}{edited_modules}": "",
                    }
                if unreadable:
                    unreadable_modules = ", ".join(unreadable)
                    logger.warning(
                        "Stage '%s' uses user modules whose source pivot cannot read: %s. Pivot "
                        + "cannot tell which code ran, so its result is never reused; every run "
                        + "reruns it while pivot cannot read them.%s The first stage in each "
                        + "process that uses such a module logs why pivot cannot trust it, and "
                        + "what to do.",
                        stage_name,
                        unreadable_modules,
                        commit_note,
                    )
                    lock_manifest = {
                        **lock_manifest,
                        f"{skip.UNREADABLE_SOURCE_KEY_PREFIX}{unreadable_modules}": "",
                    }
                output_hashes = dict[str, HashInfo]()
                if no_commit:
                    read = list[tuple[pathlib.Path, os.stat_result, str]]()
                    output_hashes = _hash_output_paths(
                        output_paths, cache_dir=store_cache_dir(store), pending=read
                    )
                    file_hash_entries.extend(cache.deferred_file_hash_entries(read))
                # Under --no-commit, write no lock, cache object, generation or run-cache
                # entry. Leave the record of what ran, which `pivot commit` writes if the outputs
                # are still these, and hand file-hash cache entries to the coordinator for its
                # best-effort save. The same record already lists the `*.pvt` members this run
                # wrote: each was recorded before it was written, and the record stays until a
                # lock is written.
                if no_commit:
                    no_commit_record = NoCommitRun(
                        code_manifest=lock_manifest,
                        output_hashes={
                            key: hash_info["hash"]
                            for key, hash_info in sorted(output_hashes.items())
                        },
                        output_paths={
                            types.identity_key(identity): path
                            for identity, path in lock_output_paths(
                                stage_outs, store, project_root
                            ).items()
                        },
                    )
                    if read_hashes:
                        no_commit_record["reads"] = read_hashes
                    no_commit_writes = DeferredWrites(
                        file_hash_entries=file_hash_entries, no_commit_record=no_commit_record
                    )
                    if fetched:
                        no_commit_writes["remote_hashes"] = fetched
                    return StageResult(
                        status=StageStatus.RAN,
                        reason=run_reason,
                        input_hash=input_hash,
                        output_lines=ring_buffer.snapshot(),
                        metrics=metrics.get_entries(),
                        deferred_writes=no_commit_writes,
                    )
                else:
                    # Hash and publish before the naming unit, so a large output's reading and
                    # copying holds no lock a waiting collection would turn into a stall for
                    # every other commit on the cache.
                    with stage_io.PIVOT_IO:
                        prepared = _prepare_outputs_with_store(
                            output_paths,
                            store,
                            project_state_dir,
                            hashed_dirs,
                            claims,
                            written["pvt_members"],
                        )
                    # A read by hash names nothing until the lock does, so a collection may have
                    # removed an object the stage read on a cache hit: fetch each such object
                    # again before the unit (no download inside one), and refuse inside it any
                    # that is still not in the cache.
                    unfetched = reads.fetch_missing(read_hashes, recorded)
                    fetched = recorded.fetched()
                    # Naming-lock unit (shared), the naming step only: re-check each object (one
                    # a collection removed since is published again), link the outputs, write the
                    # lock that names them. gc cannot delete while this unit is open.
                    with naming.naming_unit(naming_root, project=(project_root, project_state_dir)):
                        reads.require_cached(read_hashes, recorded.cache_dir, unfetched)
                        with stage_io.PIVOT_IO:
                            output_hashes = _commit_outputs_with_store(
                                output_paths,
                                store,
                                project_state_dir,
                                prepared,
                                hashed_dirs,
                                claims,
                                written["pvt_members"],
                            )
                        for ref, path in output_paths.items():
                            output_hash = output_hashes[types.identity_key(ref.identity)]
                            file_hash_entries.extend(
                                cache.deferred_file_hash_entries(
                                    cache.known_file_hashes(
                                        path,
                                        output_hash,
                                        hashed_dirs[output_hash["hash"]]["listing"]
                                        if is_dir_hash(output_hash)
                                        else None,
                                    )
                                )
                            )

                        # Build lock data
                        new_lock_data = LockData(
                            code_manifest=lock_manifest,
                            params=current_params,
                            dep_hashes=_to_identity_hashes(dict(sorted(dep_hashes.items()))),
                            output_hashes=_to_identity_hashes(dict(sorted(output_hashes.items()))),
                            output_paths=lock_output_paths(stage_outs, store, project_root),
                        )
                        if read_hashes:
                            new_lock_data["reads"] = read_hashes

                        # Single StateDB open for post-execution work. _open_state_db is #530's
                        # one-line unexpected-error boundary; keep it inside the naming unit.
                        with _open_state_db(state_dir) as state_db:
                            _raise_if_stopped()
                            deferred = _commit_lock_and_build_deferred(
                                stage_info,
                                new_lock_data,
                                input_hash,
                                output_hashes,
                                production_lock,
                                state_db,
                                store=store,
                                hashed_dirs=hashed_dirs,
                                reusable=not uncovered and not edited and not unreadable,
                                file_hash_entries=file_hash_entries,
                            )
                            if fetched:
                                deferred["remote_hashes"] = fetched
                            return StageResult(
                                status=StageStatus.RAN,
                                reason=run_reason,
                                input_hash=input_hash,
                                output_lines=ring_buffer.snapshot(),
                                metrics=metrics.get_entries(),
                                deferred_writes=deferred,
                            )
        except (
            exceptions.OutputMissingError,
            exceptions.CacheError,
            exceptions.CacheRestoreError,
            exceptions.CorruptListingError,
            exceptions.StageDefinitionError,
            exceptions.InvalidPathError,
            exceptions.OutputOwnershipError,
            exceptions.MemberRecordError,
        ) as exc:
            return _make_result(
                StageStatus.FAILED, exc.format_user_message(), ring_buffer, input_hash=input_hash
            )
        except SystemExit as e:
            return _make_result(
                StageStatus.FAILED,
                f"Stage called sys.exit({e.code})",
                ring_buffer,
                input_hash=input_hash,
            )
        except KeyboardInterrupt:
            return _make_result(
                StageStatus.FAILED,
                "KeyboardInterrupt",
                ring_buffer,
                input_hash=input_hash,
            )
        except Exception as exc:
            # The record keeps the traceback. The console prints one line only for a failure of
            # pivot's own I/O on state it owns (`stage_io`): its cache, its state database, the
            # stage's lock file, and preparing and creating the stage's output. Any other
            # failure, out of the stage function, a loader or writer, or anything unmarked,
            # prints its traceback as before; --verbose prints it in both cases.
            result = _make_result(
                StageStatus.FAILED,
                traceback.format_exc(),
                ring_buffer,
                input_hash=input_hash,
            )
            if stage_io.raised_by_pivot(exc):
                result["display_reason"] = exceptions.describe_error(exc)
            return result


def _open_state_db(state_dir: pathlib.Path) -> state.StateDB:
    """Open the stage's state database read-only; a failure to open it is pivot's own I/O."""
    with stage_io.PIVOT_IO:
        return state.StateDB(state_dir, readonly=True)


def user_code_files() -> list[str]:
    """Return source paths for user modules this process has loaded.

    A path that was deleted after import is still reported: the coordinator must restart
    instead of letting the worker keep executing the cached module.
    """
    files = set[str]()
    for module in tuple(sys.modules.values()):
        if not fingerprint.is_user_code(module):
            continue
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            continue
        try:
            files.add(str(pathlib.Path(module_file).resolve(strict=False)))
        except OSError:
            files.add(str(pathlib.Path(module_file).absolute()))
    return sorted(files)


def _begin_stage() -> None:
    """Mark a stage running, clearing a Ctrl-C that interrupted the previous one. The running
    flag is set last, so a stop signal before it is handled as one to an idle worker."""
    _stop.interrupted = False
    if not _stop.exit_requested:
        _stop.stop_requested = False
    _stop.stage_running = True


def execute_stage(
    stage_name: str,
    stage_info: WorkerStageInfo,
    cache_dir: pathlib.Path,
    output_queue: Queue[OutputMessage],
) -> StageResult:
    """Execute a stage and report every user-code module resident in this worker.

    A stage interrupted by a stop signal (``unwind_when_stopped``) fails without committing
    anything. After SIGTERM or the death of its Pivot process, the worker exits once the stage
    has unwound instead of returning.
    """
    invocation = state_dir_mod.begin_invocation(
        report=False,
        state_dirs={stage_info["project_root"]: stage_info["state_dir_resolution"]},
    )
    try:
        _begin_stage()
        result = _execute_stage(stage_name, stage_info, cache_dir, output_queue)
    except WorkerTerminated:
        # Raised outside the stage's own error handling, as while its logging is set up.
        result = StageResult(
            status=StageStatus.FAILED, reason="KeyboardInterrupt", input_hash=None, output_lines=[]
        )
    finally:
        _stop.stage_running = False
        # Every way out of the stage drops its recorded failure, whose frames hold the stage's
        # locals and loaded inputs, so a warm worker does not keep them until its next stage.
        stage_io.clear()
        state_dir_mod.end_invocation(invocation)
    if _stop.exit_requested:
        _exit_worker(1)
    result["user_code_files"] = user_code_files()
    return result


def _to_identity_hashes(hashes: dict[str, HashInfo]) -> dict[types.ArtifactIdentity, HashInfo]:
    return {types.identity_from_key(key): info for key, info in hashes.items()}


class OutputRestoreResult(NamedTuple):
    """Whether output restoration succeeded and wrote workspace bytes.

    ``unavailable`` lists outputs that neither match their locked hash nor can be restored
    from the cache. ``empty_dirs_unrecorded`` lists outputs that hold every file their locked
    listing records plus empty directories it records none of (``unrecorded_empty_dirs_only``):
    restoring such an output would delete the directories, so the stage runs instead.
    ``to_restore`` lists outputs that differ from their locked hash and that the cache holds:
    a restore writes them from the cache, a check only names them, so a check whose result is
    available with nothing to restore writes nothing when restored. A check
    (``restore=False``) lists every such output; a restore stops at the first unusable one,
    since the stage must run anyway.
    """

    available: bool
    outputs_restored: bool
    unavailable: list[types.ArtifactRef]
    empty_dirs_unrecorded: list[types.ArtifactRef]
    to_restore: list[types.ArtifactRef]


def unrecorded_empty_dirs_only(expected: listing.DirListing, actual: listing.DirListing) -> bool:
    """Whether a directory output holding ``actual`` is ``expected``'s files plus empty
    directories, where ``expected`` records none.

    Pivot before 0.3.4 listed a directory's files alone, and a lock pivot 0.3.4 or later wrote
    for a directory with no empty directories looks the same; pivot reads such a lock as it is
    (``StageLock.read``), its listing recording no empty directories. Restoring an output to it
    deletes the empty directories on disk, and the stages that read the output then hash the
    stripped tree to the old value and stay cached, though a fresh build of a stage that makes
    the directories would hold them. Running the stage instead costs one run where the
    directories were only added by hand; that run rebuilds the output without them.
    """
    return (
        not expected["empty_dirs"]
        and bool(actual["empty_dirs"])
        and actual["files"] == expected["files"]
    )


def empty_dirs_unrecorded_reason(refs: list[types.ArtifactRef]) -> str:
    """Say why a stage whose lock is current must run: ``refs`` are its outputs whose lock does
    not record the empty directories they hold."""
    keys = ", ".join(types.identity_key(ref.identity) for ref in refs)
    return f"Output holds empty directories its lock does not record: {keys}"


def _outputs_unusable_reason(restoration: OutputRestoreResult) -> str:
    """The reason a stage whose lock is current runs because its outputs could not be used."""
    if restoration.empty_dirs_unrecorded:
        return empty_dirs_unrecorded_reason(restoration.empty_dirs_unrecorded)
    return "outputs missing from cache"


def stage_claims(
    store: store_mod.Store,
    outs: list[types.ArtifactRef],
    lock_data: LockData | None,
    no_commit_record: types.NoCommitRecord | None,
    project_root: pathlib.Path,
    recorder: ownership.MemberRecorder | None = None,
    *,
    stage_lock: lock.StageLock,
) -> ownership.StageClaims | None:
    """The stage's claims on the ``*.pvt`` files in its workspace outputs
    (``ownership.StageClaims``), from its lock (``stage_lock``, which read ``lock_data``; a
    directory's members from the listing committed beside it) and its ``nocommit:`` record; None
    for a store that writes outside the workspace."""
    if not isinstance(store, store_mod.WorkspaceStore):
        return None
    locked = lock_data["output_hashes"] if lock_data is not None else {}
    paths = {out.identity: store.resolve_display_path(out) for out in outs}
    return ownership.stage_claims(
        project_root,
        {path: locked[identity] for identity, path in paths.items() if identity in locked},
        no_commit_record,
        cache_dir=store.cache_dir,
        recorder=recorder,
        listing_readers=ownership.lock_listing_readers(
            stage_lock, lock_data, paths, store.cache_dir
        ),
    )


@overload
def restore_outputs_to_hashes(
    store: store_mod.Store,
    outs: list[types.ArtifactRef],
    output_hashes: dict[types.ArtifactIdentity, HashInfo],
    state_db: state.StateDB | None,
    *,
    state_dir: pathlib.Path | None = None,
    listing_lock: lock.StageLock | None = None,
    restore: bool = True,
    report_changes: Literal[False] = False,
    file_hash_entries: list[FileHashEntry] | None = None,
    claims: ownership.StageClaims | None = None,
    register: Callable[[], None] | None = None,
) -> bool: ...


@overload
def restore_outputs_to_hashes(
    store: store_mod.Store,
    outs: list[types.ArtifactRef],
    output_hashes: dict[types.ArtifactIdentity, HashInfo],
    state_db: state.StateDB | None,
    *,
    state_dir: pathlib.Path | None = None,
    listing_lock: lock.StageLock | None = None,
    restore: bool = True,
    report_changes: Literal[True],
    file_hash_entries: list[FileHashEntry] | None = None,
    claims: ownership.StageClaims | None = None,
    register: Callable[[], None] | None = None,
) -> OutputRestoreResult: ...


def restore_outputs_to_hashes(
    store: store_mod.Store,
    outs: list[types.ArtifactRef],
    output_hashes: dict[types.ArtifactIdentity, HashInfo],
    state_db: state.StateDB | None,
    *,
    state_dir: pathlib.Path | None = None,
    listing_lock: lock.StageLock | None = None,
    restore: bool = True,
    report_changes: bool = False,
    file_hash_entries: list[FileHashEntry] | None = None,
    claims: ownership.StageClaims | None = None,
    register: Callable[[], None] | None = None,
) -> bool | OutputRestoreResult:
    """Verify outputs match recorded hashes and restore cacheable mismatches.

    ``listing_lock`` supplies a directory output's committed listing before the cache object,
    so an evicted listing object does not make a still-restorable output unavailable.
    ``file_hash_entries``, when given, receives a StateDB file-hash entry for every file the
    check had to read and every file restored from the cache, so the next check finds each
    without reading it. ``register``, when given, runs once before the first restore: a restore
    links cache objects into the workspace outside a publishing unit, so the project must be in
    the register of the cache it restores from first. A caller inside a naming unit passes none
    (that unit registered). ``claims`` (the stage's) decide which ``*.pvt`` files in an output
    are the stage's own, and record the ones a restore writes that the lock does not list
    (``store.WorkspaceStore.restore_from_cache``).
    """
    unavailable = list[types.ArtifactRef]()
    empty_dirs_unrecorded = list[types.ArtifactRef]()
    to_restore = list[types.ArtifactRef]()
    outputs_restored = False
    if not isinstance(store, store_mod.WorkspaceStore):
        unavailable = [ref for ref in outs if not store.exists(ref)]
    else:
        for ref in outs:
            expected_hash = output_hashes.get(ref.identity)
            if expected_hash is not None:
                output_path = store.resolve_display_path(ref)
                read = list[tuple[pathlib.Path, os.stat_result, str]]()
                actual_hash: HashInfo | None
                actual_listing: listing.DirListing | None
                try:
                    actual_hash, actual_listing = hash_output(
                        output_path,
                        state_db,
                        cache_dir=store.cache_dir,
                        pending=None if file_hash_entries is None else read,
                    )
                except OSError:
                    # An output the check cannot read differs from its lock. A symlink it cannot
                    # classify raises UnreadableLinkError instead, so a caller refuses rather
                    # than restoring over a link that may be a user's.
                    actual_hash, actual_listing = None, None
                if file_hash_entries is not None:
                    file_hash_entries.extend(cache.deferred_file_hash_entries(read))
                if actual_hash == expected_hash:
                    continue
                members = None
                if is_dir_hash(expected_hash):
                    try:
                        members = (
                            listing.read(
                                listing_lock.output_listing_path(ref.identity),
                                expected_hash,
                                store.cache_dir,
                            )
                            if listing_lock is not None
                            else store.cached_listing(expected_hash)
                        )
                    except exceptions.ListingMissingError:
                        pass
                    except exceptions.CorruptListingError as e:
                        # The output cannot be restored from this listing: the stage runs,
                        # which rewrites it.
                        logger.warning("%s. %s", e, e.REMEDY)
                if (
                    members is not None
                    and actual_listing is not None
                    and unrecorded_empty_dirs_only(members, actual_listing)
                ):
                    empty_dirs_unrecorded.append(ref)
                    if restore:
                        break
                    continue
                if (members is not None or not is_dir_hash(expected_hash)) and (
                    store.cache_contains(ref, expected_hash, members)
                ):
                    to_restore.append(ref)
                    if not restore:
                        continue
                    restored = list[tuple[pathlib.Path, os.stat_result, str]]()
                    if register is not None:
                        register()
                        register = None
                    if store.restore_from_cache(
                        ref,
                        expected_hash,
                        state_dir=state_dir,
                        members=members,
                        recorded=restored,
                        claims=claims,
                    ):
                        outputs_restored = True
                        if file_hash_entries is not None:
                            file_hash_entries.extend(cache.deferred_file_hash_entries(restored))
                        continue
            unavailable.append(ref)
            if restore:
                break
    available = not unavailable and not empty_dirs_unrecorded
    if report_changes:
        return OutputRestoreResult(
            available=available,
            outputs_restored=outputs_restored,
            unavailable=unavailable,
            empty_dirs_unrecorded=empty_dirs_unrecorded,
            to_restore=to_restore,
        )
    return available


def _check_skip_or_run(
    stage_info: WorkerStageInfo,
    lock_data: LockData | None,
    current_fingerprint: dict[str, str],
    current_params: dict[str, Any],
    dep_hashes: dict[str, HashInfo],
    *,
    unavailable_reads: types.UnavailableReads | None = None,
) -> tuple[str | None, str, str, bool]:
    """Determine if stage can skip or must run. ``unavailable_reads`` is
    ``reads.lock_unavailable`` of ``lock_data``."""
    out_identities = [out.identity for out in stage_info["outs"]]
    dep_hashes_by_identity = _to_identity_hashes(dep_hashes)
    input_hash = run_history.stage_input_hash(
        current_fingerprint,
        current_params,
        stage_info["deps"].values(),
        dep_hashes_by_identity,
        stage_info["outs"],
    )
    decision = skip.check_stage(
        lock_data=lock_data,
        fingerprint=current_fingerprint,
        params=current_params,
        dep_hashes=dep_hashes_by_identity,
        out_paths=out_identities,
        run_cache_allowed=not stage_info["incremental_inputs"]
        and not stage_info["unsafe_fingerprint"],
        unavailable_reads=unavailable_reads,
    )
    if decision["changed"]:
        return None, decision["reason"], input_hash, decision["run_cache_allowed"]
    return "unchanged", "", input_hash, decision["run_cache_allowed"]


def store_cache_dir(store: store_mod.Store) -> pathlib.Path | None:
    """The cache object root whose symlinks count as directory content for ``store``."""
    return store.cache_dir if isinstance(store, store_mod.WorkspaceStore) else None


def hash_output(
    path: pathlib.Path,
    state_db: state.StateDB | None = None,
    *,
    cache_dir: pathlib.Path | None = None,
    pending: list[tuple[pathlib.Path, os.stat_result, str]] | None = None,
) -> tuple[HashInfo, listing.DirListing | None]:
    """Compute output hash without saving to cache, with a directory's listing (None for a
    file).

    ``cache_dir`` is the cache object root whose symlinks count as directory content.
    ``pending`` receives the file-hash entry of each file read, as ``cache.hash_file`` says.
    """
    if path.is_dir():
        hashed = cache.hash_directory(path, state_db, cache_dir=cache_dir, pending=pending)
        return hashed["record"], hashed["listing"]
    file_hash, _ = cache.hash_file(path, state_db, pending=pending)
    return FileHash(hash=file_hash), None


def lock_output_paths(
    outs: list[types.ArtifactRef],
    store: store_mod.Store,
    project_root: pathlib.Path,
) -> dict[types.ArtifactIdentity, str]:
    """Project-relative, POSIX output paths for a production workspace lock."""
    if not isinstance(store, store_mod.WorkspaceStore):
        return {}
    root = project_root.resolve()
    return {
        ref.identity: store.resolve_display_path(ref).relative_to(root).as_posix() for ref in outs
    }


def _set_deterministic_seeds() -> None:
    """Set random seeds for reproducible stage execution.

    Called before each stage to ensure determinism. Users can override
    by calling random.seed() or np.random.seed() in their stage code.
    """
    random.seed(0)
    try:
        import numpy as np

        np.random.seed(0)
    except ImportError:
        pass  # NumPy is optional; stdlib random is still seeded


def _execute_with_joblib_protection(func: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    """Execute stage with joblib threading backend to avoid nested multiprocessing issues.

    By default, configures joblib to use threading backend, which works well for
    GIL-releasing code (NumPy, pandas) and avoids resource_tracker race conditions.

    Users can override via PIVOT_NESTED_PARALLELISM=processes env var, or by wrapping
    their Parallel calls in their own parallel_config() context.
    """
    try:
        from joblib import parallel_config
    except ImportError:
        parallel_config = None
    if parallel_config is None:
        # Called outside the except block so a stage error is not chained onto the ImportError.
        logger.debug("joblib not installed - nested parallelism protection disabled")
        return func(**kwargs)

    # Environment variable allows users to opt into multiprocessing
    env_override = os.environ.get("PIVOT_NESTED_PARALLELISM")
    if env_override == "processes":
        # Disable memmapping to prevent resource_tracker race conditions.
        # This avoids KeyError tracebacks when Pivot's loky pool and joblib's
        # nested loky pool have concurrent cleanup.
        logger.debug("Nested parallelism: processes mode (memmapping disabled)")
        with parallel_config(backend="loky", max_nbytes=None):
            return func(**kwargs)

    # Default: threading backend (safe for NumPy/pandas workloads)
    logger.debug("Nested parallelism: threading mode")
    with parallel_config(backend="threading"):
        return func(**kwargs)


def _load_dep_from_path(ref: types.ArtifactRef, path: pathlib.Path) -> Any:
    """Load an artifact after the caller has resolved its workspace/cache path."""
    loader = cast("Any", ref.format)
    if ref.tag is types.ArtifactTag.DIRECTORY and ref.directory is None:
        values = dict[str, Any]()
        for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
            rel_key = file_path.relative_to(path).as_posix()
            values[rel_key] = loader.load(file_path)
        return values
    return loader.load(path)


def _resolve_output_values(
    result: Any, outs: list[types.ArtifactRef], stage_name: str
) -> dict[types.ArtifactRef, Any]:
    if not outs:
        if result is not None:
            logger.warning(
                "Stage '%s' returned value but has no Out annotation - discarding", stage_name
            )
        return {}

    if result is None:
        raise RuntimeError(f"Stage '{stage_name}' has output annotations but returned None")

    if len(outs) == 1 and outs[0].identity.key is None:
        return {outs[0]: result}

    if not isinstance(result, collections_abc.Mapping):
        raise RuntimeError(
            f"Stage '{stage_name}' returned {type(result).__name__} but expected a mapping"
        )
    # Cast needed: isinstance narrows Any to Mapping[Unknown, Unknown]; output keys are strings
    result_map = cast("collections_abc.Mapping[str, Any]", result)

    value_map = dict[types.ArtifactRef, Any]()
    missing = list[str]()
    for out in outs:
        key = out.identity.key
        if key is None:
            raise RuntimeError(f"Stage '{stage_name}' has multiple outputs but missing output key")
        if key not in result_map:
            missing.append(key)
            continue
        value_map[out] = result_map[key]

    if missing:
        raise KeyError(
            f"Missing return output keys: {sorted(missing)}. Return value keys: {sorted(result_map.keys())}"
        )

    extra = set(result_map.keys()) - {
        out.identity.key for out in outs if out.identity.key is not None
    }
    if extra:
        logger.warning("Extra keys in return value not declared as outputs: %s", sorted(extra))

    return value_map


def _paths_overlap(first: pathlib.Path, second: pathlib.Path) -> bool:
    """Return whether either path contains the other."""
    return first.is_relative_to(second) or second.is_relative_to(first)


def _reject_path_inside_project(
    ref: types.ArtifactRef,
    value: object,
    project_root: pathlib.Path,
    dependency_paths: collections_abc.Collection[pathlib.Path] = (),
) -> None:
    """A PathOnly output is moved to the output path; never move a file pivot does not own."""
    if not isinstance(value, pathlib.Path):
        raise TypeError(
            f"Stage '{ref.identity.producer}' returned {type(value).__name__}; "
            + "a PathOnly output must return pathlib.Path"
        )
    source = value if value.is_absolute() else project_root / value
    lexical_root = project_root.absolute()
    lexical_source = source.absolute()
    resolved_root = project_root.resolve()
    resolved_source = source.resolve()
    if lexical_source.is_relative_to(lexical_root) or resolved_source.is_relative_to(resolved_root):
        raise exceptions.StageDefinitionError(
            f"Stage '{ref.identity.producer}' returned {value}, which is inside the project "
            + f"({project_root}). Pivot moves a returned path to the stage's output path, so it "
            + "must be a file or directory the stage created outside the project, e.g. under "
            + "tempfile.mkdtemp()."
        )
    for dependency_path in dependency_paths:
        lexical_dependency = dependency_path.absolute()
        resolved_dependency = dependency_path.resolve()
        if any(
            _paths_overlap(source_path, dependency_variant)
            for source_path in (lexical_source, resolved_source)
            for dependency_variant in (lexical_dependency, resolved_dependency)
        ):
            raise exceptions.StageDefinitionError(
                f"Stage '{ref.identity.producer}' returned {value}, which overlaps dependency "
                + f"{dependency_path}. Pivot cannot move a path the stage received as input."
            )


def _check_output_value(
    ref: types.ArtifactRef,
    value: Any,
    project_root: pathlib.Path,
    dependency_paths: collections_abc.Collection[pathlib.Path],
) -> Any:
    """Refuse an output value that cannot be written, before anything of the stage is written.

    Returns what ``_write_output_with_store`` writes: the value, or for a directory output its
    checked ``(path inside the directory, value)`` pairs.
    """
    if ref.tag is types.ArtifactTag.DIRECTORY:
        label = "the return value" if ref.identity.key is None else f"output {ref.identity.key!r}"
        members = stage_def.directory_out_members(value, ref.identity.producer, label)
        if isinstance(ref.format, loaders.PathOnly):
            for _, item_value in members:
                _reject_path_inside_project(ref, item_value, project_root, dependency_paths)
        return members
    if isinstance(ref.format, loaders.PathOnly):
        _reject_path_inside_project(ref, value, project_root, dependency_paths)
    return value


class WrittenOutputs(TypedDict):
    """A stage's written outputs, and in each the ``*.pvt`` files it wrote, with their content."""

    paths: dict[types.ArtifactRef, pathlib.Path]
    pvt_members: dict[types.ArtifactRef, ownership.StageMembers]


def _output_entries(
    ref: types.ArtifactRef, value: Any, output_path: pathlib.Path
) -> list[tuple[pathlib.Path, Any]]:
    """Each path writing an output's checked value fills, with what it writes there: a directory
    output's keys, else the output path itself."""
    if ref.tag is types.ArtifactTag.DIRECTORY:
        return [
            (output_path / relpath, item_value)
            for relpath, item_value in cast("list[tuple[str, Any]]", value)
        ]
    return [(output_path, value)]


def _brought_pvt_files(
    ref: types.ArtifactRef, path: pathlib.Path, value: Any
) -> list[pathlib.Path]:
    """The ``*.pvt`` files writing ``value`` at ``path`` puts there: ``path`` itself when it is a
    ``*.pvt`` name, and each one inside a directory a ``PathOnly`` value moves there."""
    if path.name.endswith(".pvt"):
        return [path]
    if (
        not isinstance(ref.format, loaders.PathOnly)
        or not isinstance(value, pathlib.Path)
        or value.is_symlink()
        or not value.is_dir()
    ):
        return []
    return [
        path / pathlib.Path(dirpath, name).relative_to(value)
        for dirpath, _dirnames, filenames in os.walk(value)
        for name in filenames
        if name.endswith(".pvt")
    ]


def _write_outputs_with_store(
    output_values: dict[types.ArtifactRef, Any],
    store: store_mod.Store,
    project_root: pathlib.Path,
    dependency_paths: collections_abc.Collection[pathlib.Path] = (),
    claims: ownership.StageClaims | None = None,
    state_dir: pathlib.Path | None = None,
) -> WrittenOutputs:
    """Write a stage's outputs once every one of them has passed its checks.

    A refused value (a directory output's key, a PathOnly path inside the project) fails the
    stage before any output is prepared, so the previous run's outputs stay as they were.
    ``claims`` (the stage's, for a workspace store) decide which ``*.pvt`` files already in an
    output are the stage's, and record each one this run writes before it lands there.
    ``state_dir`` holds the restore locks under which each old output is removed.
    """
    checked = {
        ref: _check_output_value(ref, value, project_root, dependency_paths)
        for ref, value in output_values.items()
    }
    returned = dict[types.ArtifactRef, ownership.StageMembers]()
    if isinstance(store, store_mod.WorkspaceStore):
        # A `*.pvt` this run writes at a path is the stage's there if it does not parse as a
        # sidecar; any other is the stage's only if its lock or nocommit: record lists it
        for ref, value in checked.items():
            output_path = ownership.normalized(store.resolve_display_path(ref))
            brought = [
                pvt
                for path, item_value in _output_entries(ref, value, output_path)
                for pvt in _brought_pvt_files(ref, path, item_value)
            ]
            if brought:
                returned[ref] = {pvt: frozenset[str]() for pvt in brought}
        # Check all workspace outputs first, so an ownership refusal changes none of the stage's
        # previous outputs. Each output is prepared immediately before its writer runs, preserving
        # sibling outputs if that writer raises. prepare_output checks the output it clears, so a
        # single output needs no check before it: nothing happens between the two.
        if len(checked) > 1:
            store.refuse_output_replacements(checked, claims, returned)
    paths = dict[types.ArtifactRef, pathlib.Path]()
    written = dict[types.ArtifactRef, ownership.StageMembers]()
    for ref, value in checked.items():
        paths[ref], members = _write_output_with_store(
            ref, value, store, claims, returned.get(ref), state_dir
        )
        if members:
            written[ref] = members
    return WrittenOutputs(paths=paths, pvt_members=written)


def _write_output_with_store(
    ref: types.ArtifactRef,
    value: Any,
    store: store_mod.Store,
    claims: ownership.StageClaims | None = None,
    returned: ownership.StageMembers | None = None,
    state_dir: pathlib.Path | None = None,
) -> tuple[pathlib.Path, ownership.StageMembers]:
    """Write one output; return its path and the ``*.pvt`` files written in it, with content."""
    with stage_io.PIVOT_IO:
        output_path = store.prepare_output(ref, claims, returned, state_dir=state_dir)
    if isinstance(store, store_mod.WorkspaceStore):
        store.refuse_output_replacement_at_write(ref, claims, returned)
    writer = cast("Any", ref.format)

    output_label = (
        f"its {type(writer).__name__} output"
        if ref.identity.key is None
        else f"output {ref.identity.key!r}"
    )

    def save_output(output_value: Any, destination: pathlib.Path) -> None:
        with (
            warnings.catch_warnings(record=True) as caught,
            stage_io.writing_output(destination),
        ):
            warnings.simplefilter("always", loaders.LossyOutputWarning)
            try:
                writer.save(output_value, destination)
            except TypeError as e:
                if isinstance(writer, loaders.YAML):
                    raise TypeError(
                        f"Stage '{ref.identity.producer}' cannot write {output_label}: {e}"
                    ) from e
                raise
        for caught_warning in caught:
            if issubclass(caught_warning.category, loaders.LossyOutputWarning):
                logger.warning(
                    "Stage '%s' wrote %s %s: %s",
                    ref.identity.producer,
                    output_label,
                    types.LOSSY_OUTPUT_PHRASE,
                    caught_warning.message,
                )
            else:
                warnings.warn_explicit(
                    caught_warning.message,
                    caught_warning.category,
                    caught_warning.filename,
                    caught_warning.lineno,
                    source=caught_warning.source,
                )

    # A directory output was just created empty and no key lies inside another key's path, so
    # every directory on the way to a file is one created here, never a returned path
    recorder = claims["recorder"] if claims is not None else None
    recorded_entries = list[tuple[pathlib.Path, Any]]()
    for path, item_value in _output_entries(ref, value, output_path):
        if recorder is not None and _brought_pvt_files(ref, path, item_value):
            recorded_entries.append((path, item_value))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        save_output(item_value, path)
    if recorder is None or not recorded_entries:
        return output_path, {}
    return output_path, _save_recorded(output_path, recorded_entries, save_output, recorder)


def _save_recorded(
    output_path: pathlib.Path,
    entries: list[tuple[pathlib.Path, Any]],
    save_output: Callable[[Any, pathlib.Path], None],
    recorder: ownership.MemberRecorder,
) -> ownership.StageMembers:
    """Write entries that bring ``*.pvt`` files into the output at ``output_path``: each under a
    name beside the output (``_staging_path``), then record the ``*.pvt`` files they hold as the
    stage's, then move each into place. A crash leaves at worst a record without a file in the
    output, and a staged copy beside it that no check of the output walks. Returns those files,
    each with its recorded content."""
    staged = dict[pathlib.Path, pathlib.Path]()
    try:
        for path, item_value in entries:
            path.parent.mkdir(parents=True, exist_ok=True)
            staged[path] = _staging_path(output_path, staged.values())
            save_output(item_value, staged[path])
        recorded = {
            pvt: cache.hash_file(source)[0]
            for path, staged_path in staged.items()
            for pvt, source in _staged_pvt_files(path, staged_path)
        }
        recorder(recorded)
        for path, staged_path in staged.items():
            staged_path.replace(path)
        staged.clear()
    finally:
        for staged_path in staged.values():
            cache.remove_output(staged_path)
    return {ownership.normalized(pvt): frozenset([content]) for pvt, content in recorded.items()}


def _staged_pvt_files(
    path: pathlib.Path, staged_path: pathlib.Path
) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """Each ``*.pvt`` file ``staged_path`` holds, as (its path once moved to ``path``, where it
    is now)."""
    if staged_path.is_dir() and not staged_path.is_symlink():
        found = list[tuple[pathlib.Path, pathlib.Path]]()
        for dirpath, _dirnames, filenames in os.walk(staged_path):
            for name in filenames:
                source = pathlib.Path(dirpath, name)
                if name.endswith(".pvt") and source.is_file():
                    found.append((path / source.relative_to(staged_path), source))
        return found
    if path.name.endswith(".pvt") and staged_path.is_file():
        return [(path, staged_path)]
    return []


def _staging_path(
    output_path: pathlib.Path, taken: collections_abc.Collection[pathlib.Path]
) -> pathlib.Path:
    """An unused name beside ``output_path``, not in ``taken`` and not a ``*.pvt`` name.

    Beside the output, never inside it: what a killed run leaves there is no file of the output,
    so it never blocks the next write or restore of it. The rename into place stays atomic: the
    name is in the output's parent directory, on the output's filesystem.
    """
    candidate = output_path.with_name(f".{output_path.name}.pivot-staged")
    count = 0
    while candidate in taken or candidate.exists() or candidate.is_symlink():
        count += 1
        candidate = output_path.with_name(f".{output_path.name}.pivot-staged-{count}")
    return candidate


def _reconstruct_list_kwargs(
    kwargs: dict[str, Any],
    collection_params: dict[str, str],
) -> dict[str, Any]:
    lists = dict[str, list[tuple[int, Any]]]()
    result = dict[str, Any]()
    for name, value in kwargs.items():
        bracket = name.rfind("[")
        if bracket != -1 and name.endswith("]"):
            base_name = name[:bracket]
            index = int(name[bracket + 1 : -1])
            if base_name not in lists:
                lists[base_name] = list[tuple[int, Any]]()
            lists[base_name].append((index, value))
        else:
            result[name] = value
    for base_name, indexed in lists.items():
        indexed.sort(key=lambda x: x[0])
        values = [v for _, v in indexed]
        result[base_name] = tuple(values) if collection_params.get(base_name) == "tuple" else values
    for param_name, ctype in collection_params.items():
        if param_name not in result:
            result[param_name] = () if ctype == "tuple" else []
    return result


def _prepare_incremental_inputs(
    stage_name: str,
    outs: list[types.ArtifactRef],
    incremental_inputs: dict[str, types.ArtifactIdentity],
    prior_outputs: Mapping[types.ArtifactIdentity, HashInfo] | None,
    store: store_mod.Store,
    stage_state_dir: pathlib.Path,
    project_state_dir: pathlib.Path,
    claims: ownership.StageClaims | None = None,
) -> dict[str, Any]:
    """Restore previous incremental outputs and load them for the stage call.

    ``prior_outputs`` are the output hashes the stage's last run recorded, or None when it has
    none.
    """
    if not incremental_inputs:
        return {}
    if not isinstance(store, store_mod.WorkspaceStore):
        raise RuntimeError("Incremental outputs require a workspace store.")

    outs_by_identity = {out.identity: out for out in outs}
    values = dict[str, Any]()
    for param_name, identity in incremental_inputs.items():
        ref = outs_by_identity[identity]
        loader = cast("loaders.Loader[Any, Any]", ref.format)
        if prior_outputs is None:
            values[param_name] = loader.empty()
            continue

        output_hash = prior_outputs.get(identity)
        stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(stage_state_dir))
        if output_hash is None:
            raise exceptions.CacheRestoreError(
                f"Cache missing for incremental output '{types.identity_key(identity)}'. "
                + f"Run 'pivot pull' to restore it, or remove '{stage_lock.path}' and rerun "
                + "to reset this incremental output."
            )
        output_path = store.resolve_display_path(ref)
        # The restore checks the output, then replaces it with a copy (a directory renamed aside
        # and checked there, a file checked in place just before the rename): nothing goes first
        if not store.restore_from_cache(
            ref,
            output_hash,
            state_dir=project_state_dir,
            checkout_modes=[config.CheckoutMode.COPY],
            claims=claims,
        ):
            raise exceptions.CacheRestoreError(
                f"Cache missing for incremental output '{types.identity_key(identity)}'. "
                + f"Run 'pivot pull' to restore it, or remove '{stage_lock.path}' and rerun "
                + "to reset this incremental output."
            )
        values[param_name] = _load_dep_from_path(ref, output_path)
    return values


def _run_stage_function_with_store(
    func: Callable[..., Any],
    stage_name: str,
    output_queue: Queue[OutputMessage],
    ring_buffer: _OutputRingBuffer,
    params: stage_def.StageParams | None,
    deps: dict[str, types.ArtifactRef],
    outs: list[types.ArtifactRef],
    store: store_mod.Store,
    params_arg_name: str | None,
    collection_params: dict[str, str],
    incremental_inputs: dict[str, types.ArtifactIdentity],
    prior_outputs: Mapping[types.ArtifactIdentity, HashInfo] | None,
    state_dir: pathlib.Path,
    project_state_dir: pathlib.Path,
    project_root: pathlib.Path,
    claims: ownership.StageClaims | None = None,
) -> WrittenOutputs:
    with (
        _QueueWriter(stage_name, output_queue, is_stderr=False, ring_buffer=ring_buffer),
        _QueueWriter(stage_name, output_queue, is_stderr=True, ring_buffer=ring_buffer),
    ):
        kwargs = dict[str, Any]()

        if params is not None:
            if params_arg_name is None:
                raise RuntimeError(
                    f"Stage '{stage_name}' has params but params_arg_name is None - this indicates a bug in registration"
                )
            kwargs[params_arg_name] = params

        dependency_paths = list[pathlib.Path]()
        for name, ref in deps.items():
            dependency_path = store.checkout(ref)
            dependency_paths.append(dependency_path)
            kwargs[name] = _load_dep_from_path(ref, dependency_path)
        kwargs.update(
            _prepare_incremental_inputs(
                stage_name,
                outs,
                incremental_inputs,
                prior_outputs,
                store,
                state_dir,
                project_state_dir,
                claims,
            )
        )

        kwargs = _reconstruct_list_kwargs(kwargs, collection_params)

        _set_deterministic_seeds()

        result = _execute_with_joblib_protection(func, kwargs)
        _raise_if_stopped()

        output_values = _resolve_output_values(result, outs, stage_name)
        return _write_outputs_with_store(
            output_values, store, project_root, dependency_paths, claims, project_state_dir
        )


def _hash_output_paths(
    outputs_by_ref: dict[types.ArtifactRef, pathlib.Path],
    state_db: state.StateDB | None = None,
    *,
    cache_dir: pathlib.Path | None = None,
    pending: list[tuple[pathlib.Path, os.stat_result, str]] | None = None,
) -> dict[str, HashInfo]:
    output_hashes = dict[str, HashInfo]()
    for ref, path in outputs_by_ref.items():
        if not path.exists():
            raise exceptions.OutputMissingError(f"Stage did not produce output: {path}")
        if path.is_dir():
            output_hashes[types.identity_key(ref.identity)] = cache.hash_directory(
                path, state_db, cache_dir=cache_dir, warn_left_out_symlinks=True, pending=pending
            )["record"]
        else:
            file_hash, _ = cache.hash_file(path, state_db, pending=pending)
            output_hashes[types.identity_key(ref.identity)] = FileHash(hash=file_hash)
    return output_hashes


def _prepare_outputs_with_store(
    outputs_by_ref: dict[types.ArtifactRef, pathlib.Path],
    store: store_mod.Store,
    project_state_dir: pathlib.Path,
    hashed_dirs: dict[str, listing.HashedDirectory],
    claims: ownership.StageClaims | None = None,
    written: dict[types.ArtifactRef, ownership.StageMembers] | None = None,
) -> dict[types.ArtifactRef, types.FileHash | listing.HashedDirectory]:
    """Hash and publish each output before the naming-lock unit (a workspace store only),
    refusing first an output whose link would replace a user's ``*.pvt`` file.
    ``hashed_dirs`` receives each hashed directory output, by hash."""
    prepared = dict[types.ArtifactRef, types.FileHash | listing.HashedDirectory]()
    for ref, path in outputs_by_ref.items():
        if not path.exists():
            raise exceptions.OutputMissingError(f"Stage did not produce output: {path}")
        if isinstance(store, store_mod.WorkspaceStore):
            prepared[ref] = store.prepare_commit(
                ref,
                path,
                state_dir=project_state_dir,
                hashed_dirs=hashed_dirs,
                claims=claims,
                written=written.get(ref) if written is not None else None,
            )
    return prepared


def _commit_outputs_with_store(
    outputs_by_ref: dict[types.ArtifactRef, pathlib.Path],
    store: store_mod.Store,
    project_state_dir: pathlib.Path,
    prepared: dict[types.ArtifactRef, types.FileHash | listing.HashedDirectory],
    hashed_dirs: dict[str, listing.HashedDirectory],
    claims: ownership.StageClaims | None = None,
    written: dict[types.ArtifactRef, ownership.StageMembers] | None = None,
) -> dict[str, HashInfo]:
    """Inside the naming-lock unit: finish each prepared output, or commit it whole."""
    output_hashes = dict[str, HashInfo]()
    for ref, path in outputs_by_ref.items():
        if not path.exists():
            raise exceptions.OutputMissingError(f"Stage did not produce output: {path}")
        if isinstance(store, store_mod.WorkspaceStore):
            output_hash = store.finish_commit(
                ref,
                path,
                prepared[ref],
                state_dir=project_state_dir,
                claims=claims,
                written=written.get(ref) if written is not None else None,
            )
        else:
            output_hash = store.commit(ref, path, hashed_dirs=hashed_dirs)
        output_hashes[types.identity_key(ref.identity)] = output_hash
    return output_hashes


class _OutputRingBuffer:
    """Bounded ring buffer for captured output lines."""

    _lines: collections.deque[tuple[str, bool]]
    _dropped_count: int
    _lock: threading.Lock

    def __init__(self, max_lines: int = 1000) -> None:
        self._lines = collections.deque(maxlen=max_lines)
        self._dropped_count = 0
        self._lock = threading.Lock()

    def append(self, line: str, is_stderr: bool) -> None:
        with self._lock:
            if len(self._lines) == self._lines.maxlen:
                self._dropped_count += 1
            self._lines.append((line, is_stderr))

    def snapshot(self) -> list[tuple[str, bool]]:
        with self._lock:
            lines = list(self._lines)
            if self._dropped_count > 0:
                lines.insert(0, (f"[{self._dropped_count} earlier lines truncated]", False))
            return lines


class _QueueWriter:
    """Context manager for capturing stdout/stderr to a queue.

    Handles stream redirection, output capture, and automatic flushing.
    Implements minimal file-like interface needed by print() and common libraries.
    Thread-safe: multiple threads can write concurrently (needed when nested
    joblib uses threading backend).
    """

    _stage_name: str
    _queue: Queue[OutputMessage]
    _is_stderr: bool
    _ring_buffer: _OutputRingBuffer
    _buffer: str
    _redirect: contextlib.AbstractContextManager[object]
    _lock: threading.Lock
    _read_fd: int | None
    _write_fd: int | None
    _reader_thread: threading.Thread | None

    def __init__(
        self,
        stage_name: str,
        output_queue: Queue[OutputMessage],
        *,
        is_stderr: bool,
        ring_buffer: _OutputRingBuffer,
    ) -> None:
        self._stage_name = stage_name
        self._queue = output_queue
        self._is_stderr = is_stderr
        self._ring_buffer = ring_buffer
        self._buffer = ""
        self._lock = threading.Lock()
        self._read_fd = None
        self._write_fd = None
        self._reader_thread = None
        # Create redirect context manager (not yet entered)
        # _QueueWriter implements write/flush but not full IO[str] interface
        io_target = cast("IO[str]", cast("object", self))
        if is_stderr:
            self._redirect = contextlib.redirect_stderr(io_target)
        else:
            self._redirect = contextlib.redirect_stdout(io_target)

    def _pipe_reader(self) -> None:
        assert self._read_fd is not None  # Guaranteed by _ensure_pipe() before thread starts
        # One incremental decoder per pipe: a multi-byte character split across two reads
        # is held until its remaining bytes arrive instead of becoming U+FFFD.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                data = os.read(self._read_fd, 8192)
                if not data:
                    break
                self.write(decoder.decode(data))
        except OSError:
            pass
        # Flush a truncated trailing character so it is reported rather than dropped.
        self.write(decoder.decode(b"", final=True))

    def _ensure_pipe(self) -> None:
        if self._read_fd is not None:
            return
        self._read_fd, self._write_fd = os.pipe()
        self._reader_thread = threading.Thread(
            target=self._pipe_reader, daemon=True, name=f"pipe-reader-{self._stage_name}"
        )
        _start_pivot_thread(self._reader_thread)

    def __enter__(self) -> _QueueWriter:
        self._redirect.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._redirect.__exit__(exc_type, exc_val, exc_tb)
        if self._write_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._write_fd)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5.0)
        if self._read_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._read_fd)
        self.flush()

    def _send_line(self, line: str) -> None:
        """Save line locally and send to queue for real-time display."""
        self._ring_buffer.append(line, self._is_stderr)
        # Queue failure only affects real-time display; output is already saved locally
        with contextlib.suppress(queue.Full, ValueError, OSError):
            self._queue.put(
                LogMessage(
                    kind=OutputMessageKind.LOG,
                    stage=self._stage_name,
                    line=line,
                    is_stderr=self._is_stderr,
                ),
                block=False,
            )

    def write(self, s: str) -> int:
        with self._lock:
            self._buffer += s
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line:
                    self._send_line(line)
        return len(s)

    def flush(self) -> None:
        with self._lock:
            if self._buffer:
                self._send_line(self._buffer)
                self._buffer = ""

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        self._ensure_pipe()
        if self._write_fd is None:
            raise OSError("QueueWriter pipe not available")
        return self._write_fd


def hash_dependencies(
    deps: dict[str, types.ArtifactRef],
    store: store_mod.Store,
    state_db: state.StateDB | None = None,
    *,
    hashed_dirs: dict[str, listing.HashedDirectory] | None = None,
) -> tuple[dict[str, HashInfo], list[str], list[str], list[FileHashEntry]]:
    """Hash all dependencies and collect file-hash cache write-backs.

    Every dependency hash is ``store.hash_artifact``'s, so a directory input leaves out the
    files ``.pivotignore`` names wherever a stage's staleness is decided.

    With a read-only (or no) ``state_db``, the write-backs are the StateDB file-hash entries of
    every file read, a file of a directory dependency included, for the coordinator to save; a
    writable ``state_db`` saves them itself. ``hashed_dirs`` receives each directory
    dependency, by hash, so the lock writer has its listing.
    """
    _t = metrics.start()
    hashes = dict[str, HashInfo]()
    missing = list[str]()
    unreadable = list[str]()
    read = list[tuple[pathlib.Path, os.stat_result, str]]()
    pending = read if state_db is None or state_db.readonly else None
    for ref in deps.values():
        key = types.identity_key(ref.identity)
        try:
            hashes[key] = store.hash_artifact(
                ref, state_db, pending=pending, hashed_dirs=hashed_dirs
            )
        except FileNotFoundError:
            missing.append(key)
        except IsADirectoryError:
            # One file of a directory output names a directory the producer wrote there
            unreadable.append(f"{key} (a directory, not a file)")
        except exceptions.UnreadableLinkError as exc:
            unreadable.append(f"{key} ({exc})")
        except OSError:
            unreadable.append(key)
    metrics.end("worker.hash_dependencies", _t)
    return hashes, missing, unreadable, cache.deferred_file_hash_entries(read)


# -----------------------------------------------------------------------------
# Generation tracking for O(1) skip detection
# -----------------------------------------------------------------------------


def can_skip_via_generation(
    stage_name: str,
    fingerprint: dict[str, str],
    deps: dict[str, types.ArtifactRef],
    outs_paths: list[str],
    current_params: dict[str, Any],
    lock_data: LockData,
    state_db: state.StateDB,
    store: store_mod.Store,
) -> bool:
    """Check generation counters and cached dependency hashes without rehashing. A lock that
    records objects its run read by hash that are now unavailable (``reads.lock_unavailable``)
    is left to the lock comparison, which names them."""
    if lock_data["code_manifest"] != fingerprint:
        return False
    if not skip.params_equal(lock_data["params"], current_params):
        return False

    normalized_outs = sorted(outs_paths)
    locked_out_paths = sorted(
        types.identity_key(identity) for identity in lock_data["output_hashes"]
    )
    if normalized_outs != locked_out_paths:
        return False
    if reads.lock_unavailable(lock_data, store_cache_dir(store)) is not None:
        return False
    if not deps:
        return True

    recorded_gens = state_db.get_dep_generations(stage_name)
    if not recorded_gens:
        return False
    refs = list(deps.values())
    gen_keys = [_generation_key(ref) for ref in refs]
    current_gens = state_db.get_many_generations(gen_keys)
    stats = list[tuple[pathlib.Path, os.stat_result]]()
    for gen_key, ref in zip(gen_keys, refs, strict=True):
        expected_hash = lock_data["dep_hashes"].get(ref.identity)
        if expected_hash is None or is_dir_hash(expected_hash):
            return False
        current_gen = current_gens.get(gen_key)
        if current_gen is None or current_gen != recorded_gens.get(gen_key):
            return False
        try:
            path = store.checkout(ref)
            stats.append((path, path.stat()))
        except OSError:
            return False
    cached_hashes = state_db.get_many(stats)
    for (path, _), ref in zip(stats, refs, strict=True):
        if cached_hashes[path] != lock_data["dep_hashes"][ref.identity]["hash"]:
            return False
    return True


def _generation_key(ref: types.ArtifactRef) -> str:
    """The generation counter a dependency follows: its producer's output, for a directory file.

    A file of a directory output changes only when its directory's generation does; when the
    generation moved, the file's hash decides.
    """
    return types.identity_key(types.source_identity(ref))


def compute_dep_generation_map(
    deps: dict[str, types.ArtifactRef],
    state_db: state.StateDB,
) -> dict[str, int]:
    """Compute dependency identity -> generation map for recording."""
    gen_keys = [_generation_key(ref) for ref in deps.values()]
    current_gens = state_db.get_many_generations(gen_keys)

    gen_record = dict[str, int]()
    for gen_key in gen_keys:
        gen = current_gens.get(gen_key)
        if gen is not None:
            gen_record[gen_key] = gen
    return gen_record


def input_listing_paths(
    deps: dict[str, types.ArtifactRef],
    store: store_mod.Store,
    dep_hashes: Mapping[types.ArtifactIdentity, HashInfo],
) -> dict[types.ArtifactIdentity, str]:
    """Each pipeline input dependency's project-relative path, which names the listing file of
    one that is a directory. Empty when ``dep_hashes``, the lock's, records no directory: the
    lock writer reads a path only for a directory."""
    if not isinstance(store, store_mod.WorkspaceStore) or not any(
        is_dir_hash(hash_info) for hash_info in dep_hashes.values()
    ):
        return {}
    return {
        ref.identity: store.input_relpath(ref)
        for ref in deps.values()
        if ref.tag is types.ArtifactTag.INPUT
    }


def older_pivot_input_hash(
    code_manifest: dict[str, str],
    params: dict[str, object],
    deps: collections_abc.Iterable[types.ArtifactRef],
    dep_hashes: Mapping[types.ArtifactIdentity, HashInfo],
    outs: collections_abc.Iterable[types.ArtifactRef],
    hashed_dirs: Mapping[str, listing.HashedDirectory],
    cache_dir: pathlib.Path | None,
) -> str | None:
    """The run-cache key an older pivot files the same run under: unmarked
    (``run_history.compute_input_hash``), with each directory dependency hashed as that pivot
    hashes it (``listing.legacy_digest`` of its members: this run's listing in ``hashed_dirs``,
    else the cache's object). None when a directory's members cannot be read here; that run's
    entry is then not found, and the stage reruns once."""
    legacy = dict[types.ArtifactIdentity, HashInfo]()
    for identity, hash_info in dep_hashes.items():
        if not is_dir_hash(hash_info):
            legacy[identity] = hash_info
            continue
        hashed = hashed_dirs.get(hash_info["hash"])
        if hashed is not None:
            members = hashed["listing"]
        else:
            try:
                members = listing.read(None, hash_info, cache_dir)
            except (exceptions.ListingMissingError, exceptions.CorruptListingError):
                return None
        legacy[identity] = FileHash(hash=listing.legacy_digest(members))
    return run_history.stage_input_hash(code_manifest, params, deps, legacy, outs, marked=False)


def _commit_lock_and_build_deferred(
    stage_info: WorkerStageInfo,
    lock_data: LockData,
    input_hash: str,
    output_hashes: dict[str, HashInfo],
    production_lock: lock.StageLock,
    state_db: state.StateDB,
    *,
    store: store_mod.Store,
    hashed_dirs: Mapping[str, listing.HashedDirectory],
    reusable: bool = True,
    file_hash_entries: list[FileHashEntry] | None = None,
    increment_outputs: bool = True,
) -> DeferredWrites:
    """Commit lock file and build deferred writes for StateDB.

    Only called in the commit (non --no-commit) path. Writes the production lock file with the
    listing of each directory it records (``hashed_dirs`` holds the ones this run hashed; the
    rest come from the cache) and returns DeferredWrites for the coordinator to apply.
    """
    with stage_io.PIVOT_IO:
        production_lock.write(
            lock_data,
            hashed_dirs=hashed_dirs,
            input_paths=input_listing_paths(stage_info["deps"], store, lock_data["dep_hashes"]),
            cache_dir=store_cache_dir(store),
            outs=stage_info["outs"],
        )
    return _build_deferred_writes(
        stage_info,
        input_hash,
        output_hashes,
        state_db,
        reads=lock_data["reads"] if "reads" in lock_data else [],
        reusable=reusable,
        legacy_input_hash=None
        if reusable
        else older_pivot_input_hash(
            stage_info["fingerprint"],
            lock_data["params"],
            stage_info["deps"].values(),
            lock_data["dep_hashes"],
            stage_info["outs"],
            hashed_dirs,
            store_cache_dir(store),
        ),
        file_hash_entries=file_hash_entries,
        increment_outputs=increment_outputs,
    )


def _build_deferred_writes(
    stage_info: WorkerStageInfo,
    input_hash: str,
    output_hashes: dict[str, HashInfo],
    state_db: state.StateDB,
    *,
    reads: list[str],
    reusable: bool = True,
    legacy_input_hash: str | None = None,
    file_hash_entries: list[FileHashEntry] | None = None,
    increment_outputs: bool = True,
) -> DeferredWrites:
    """Build deferred writes for coordinator to apply.

    A result that is not ``reusable`` (its run imported user modules its fingerprint does not
    cover, or modules changed after the coordinator loaded them) files no run-cache entry and
    deletes any entry at its input hash and at ``legacy_input_hash``, the unmarked key an older
    pivot files under; its lock carries a key no fingerprint has and `skip.check_stage` never
    consults the run cache for it, so no tier skips it. A filed entry carries ``reads``, the
    objects the run read by hash, for the lock a restore from it writes.
    """
    # A lock written now supersedes the stage's whole nocommit: record: what a --no-commit run
    # left for `pivot commit`, and the `*.pvt` members it lists.
    result = DeferredWrites(clear_no_commit_record=True)

    if increment_outputs:
        result["increment_outputs"] = True

    # Dependency generations (read current values)
    gen_record = compute_dep_generation_map(stage_info["deps"], state_db)
    if gen_record:
        result["dep_generations"] = gen_record

    # Incremental outputs continue from the latest lock state and never use the run cache.
    # An unsafe fingerprint may match a run whose captured values differed; a non-reusable
    # result ran code its fingerprint does not describe. Never file either. A non-reusable run
    # also removes any entry an earlier run filed at its key, which the check never saw.
    if reusable and not stage_info["unsafe_fingerprint"]:
        outs = stage_info["outs"]
        cached_paths = (
            set[str]()
            if any(out.tag is types.ArtifactTag.INCREMENTAL for out in outs)
            else {
                types.identity_key(out.identity)
                for out in outs
                if out.tag is not types.ArtifactTag.METRIC
            }
        )
        output_entries = [
            run_history.output_hash_to_entry(path, oh)
            for path, oh in output_hashes.items()
            if path in cached_paths
        ]
        if output_entries:
            result["run_cache_input_hash"] = input_hash
            result["run_cache_entry"] = run_history.RunCacheEntry(
                run_id=stage_info["run_id"],
                output_hashes=output_entries,
            )
            if reads:
                result["run_cache_entry"]["reads"] = reads
    if not reusable:
        result["run_cache_delete_input_hashes"] = [
            input_hash,
            *([] if legacy_input_hash is None else [legacy_input_hash]),
        ]

    if file_hash_entries:
        result["file_hash_entries"] = file_hash_entries

    return result


# -----------------------------------------------------------------------------
# Run cache for skip detection (like DVC's run cache)
# -----------------------------------------------------------------------------


class RunCacheSkipResult(TypedDict):
    """Result from successful run cache skip."""

    output_hashes: dict[str, HashInfo]
    outputs_restored: bool
    file_hash_entries: list[FileHashEntry]
    reads: list[str]  # the objects the restored run read by hash, for the lock it writes


def _run_cache_output_hash(
    output: run_history.OutputHashEntry, cache_dir: pathlib.Path | None
) -> HashInfo | None:
    """A run-cache output entry's hash. An entry an older pivot filed for a directory lists its
    files inline: it becomes the record of its listing, which is put in ``cache_dir`` for the
    restore to read, as ``StageLock.read`` does for an older pivot's lock. None when such an
    entry is malformed or there is no cache."""
    raw = cast("Mapping[str, object]", output)
    if "manifest" not in raw:
        return run_history.entry_to_output_hash(output)
    members = listing.from_inline(raw["manifest"], raw.get("empty_dirs", []))
    if members is None or cache_dir is None:
        return None
    data = listing.encode(members)
    record = listing.record_of(members, data)
    listing.store_object(cache_dir, record["hash"], data)
    return record


def _run_cache_row(
    state_db: state.StateDB,
    stage_name: str,
    input_hash: str,
    legacy_input_hash: str | None,
    stage_outs: list[types.ArtifactRef],
) -> run_history.RunCacheEntry | None:
    """The run-cache row that could restore this stage's outputs, or None: the one at
    ``input_hash``, else the one an older pivot filed at ``legacy_input_hash``. A stage with a
    metric output is never restored from the run cache.

    Read twice per lookup: before the naming unit (so a stage with no row, as in every stage of
    a cold run, takes no unit for it) and again inside it by ``_try_skip_via_run_cache_with_store``
    (so a row a prune removed in between restores nothing).
    """
    if any(out.tag is types.ArtifactTag.METRIC for out in stage_outs):
        return None
    entry = state_db.lookup_run_cache(stage_name, input_hash)
    if entry is None and legacy_input_hash is not None:
        entry = state_db.lookup_run_cache(stage_name, legacy_input_hash)
    return entry


def _try_skip_via_run_cache_with_store(
    stage_name: str,
    input_hash: str,
    stage_outs: list[types.ArtifactRef],
    store: store_mod.Store,
    state_db: state.StateDB,
    *,
    state_dir: pathlib.Path,
    legacy_input_hash: str | None = None,
    claims: ownership.StageClaims | None = None,
) -> RunCacheSkipResult | None:
    """Restore all cached outputs from a matching run-cache entry: the one at ``input_hash``,
    else the one an older pivot filed at ``legacy_input_hash`` (its unmarked key,
    ``run_history.compute_input_hash``), read in memory (``_run_cache_output_hash``).

    No lock records the entry's outputs until one is written (never, under ``--no-commit``), so
    each restore records the ``*.pvt`` members it writes as the stage's (``claims``) first. An
    entry whose run read objects by hash that are now unavailable (``reads.unavailable``) is not
    used: the lock written from it would record objects the stage cannot read again.
    """
    entry = _run_cache_row(state_db, stage_name, input_hash, legacy_input_hash, stage_outs)
    if entry is None:
        return None
    entry_reads = entry["reads"] if "reads" in entry else []
    if entry_reads and reads.unavailable(entry_reads, store_cache_dir(store)) is not None:
        return None
    output_hashes = dict[str, HashInfo]()
    for output in entry["output_hashes"]:
        output_hash = _run_cache_output_hash(output, store_cache_dir(store))
        if output_hash is None:
            return None
        output_hashes[output["path"]] = output_hash
    expected_keys = {types.identity_key(ref.identity) for ref in stage_outs}
    if output_hashes.keys() != expected_keys:
        return None
    identity_hashes = _to_identity_hashes(output_hashes)

    restored_entries = list[FileHashEntry]()
    restoration = restore_outputs_to_hashes(
        store,
        stage_outs,
        identity_hashes,
        state_db,
        state_dir=state_dir,
        report_changes=True,
        file_hash_entries=restored_entries,
        claims=claims,
    )
    if not restoration.available:
        return None
    return RunCacheSkipResult(
        output_hashes=output_hashes,
        outputs_restored=restoration.outputs_restored,
        file_hash_entries=restored_entries,
        reads=entry_reads,
    )
