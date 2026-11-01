"""The source each user module was compiled from, recorded when this process loaded it.

A stage's fingerprint must describe the code its worker runs. The worker receives a project
module's functions by value: the code this process compiled when discovery imported the module.
`inspect.getsource` reads a file's current lines through `linecache`, which re-reads a file whose
size or mtime changed, so a file edited after the import (by an upstream stage, or in an editor
during a run) would be hashed as edited while the worker runs the version that was loaded.
Fingerprinting therefore reads source and source digests from what this module recorded, and a
worker compares the modules a stage used against what the coordinator loaded (`check_stage`).

The contract is about the module, not the loader. `discovery._ProjectSourceLoader` records the
bytes it compiles; every other user-code module an import finds (an editable install's finder, a
zip on `sys.path`, `importlib.util.spec_from_file_location`, a sourceless `.pyc`) is recorded when
its loader executes it (`discovery._RecordingLoader`), and a module the recorder loaded without
recording it when it appears (`record_modules`, `record_user_modules`). The recorder trusts a
loader's source only where it can show that source runs: for CPython's `SourceFileLoader`, which
`_ProjectSourceLoader` replaces, and a zip archive's `.py` member, Pivot compiles and runs the
bytes it records itself, and a loader that ran Pivot's own inside it keeps the record Pivot's
loader made. Every other user module is recorded as unreadable, with a reason that
names its loader: Pivot cannot tell which code it holds, so a stage that uses it is never
trusted. A user module no recorder saw compile, because it was imported before the recorder was
installed (by `sitecustomize`, a `.pth` import line or the program that runs Pivot), by a loader
called directly (`SourceFileLoader(...).load_module()`) or by a loader with only `load_module`,
which the recorder does not wrap, is recorded from its file only where Pivot can show the file
holds the code it was compiled from, and as unreadable otherwise (`record_unseen_modules`, at the
end of discovery, when a worker starts and after each stage a worker runs).

Import machinery and `linecache` are process-global, so this record is too. A module loaded again
(a watch reload) replaces its entry, and `prune`, at the end of an import session, drops the
entries, and the `linecache` lines, of modules no longer loaded and of modules no recorder saw
compile. A worker gets the coordinator's records once, when it starts
(`digests_for_workers`, `set_coordinator`); a stage carries their `token`, and all of them only
when the coordinator's record changed after the workers started (`digests_for_stage`).
"""

from __future__ import annotations

import _imp
import ast
import importlib.machinery
import importlib.util
import inspect
import io
import json
import linecache
import logging
import marshal
import multiprocessing.process
import os
import subprocess
import sys
import tempfile
import time
import tokenize
import types
import warnings
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import cloudpickle
import xxhash

# `fingerprint` imports this module; the cycle resolves through `sys.modules`. A local import
# would run inside the worker's import hook, which calls `record_modules`, and recurse.
from pivot import fingerprint, import_origin

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping

logger = logging.getLogger(__name__)


class _Loaded(NamedTuple):
    """One user module's source when this process loaded it.

    `data` and `digest` are None for an unreadable module: Pivot cannot show which code it holds.
    `reason` says why Pivot cannot trust an unreadable module's code, and what to do.
    """

    filename: str  # the name the compiled code carries (`co_filename`, the module's `__file__`)
    path: str  # `filename` resolved
    module: str
    data: bytes | None
    digest: str | None
    reason: str | None = None


# Every recorded module with source, by the name its code carries and by its resolved path.
_by_file = dict[str, _Loaded]()
# Every recorded user module, unreadable ones included, by module name.
_by_module = dict[str, _Loaded]()
# The module object each name was recorded for, so a module is read once per object.
_recorded_objects = dict[str, types.ModuleType]()
# The `linecache` entry made from each recorded module's text, by the name its code carries.
_line_entries = dict[str, tuple[int, None, list[str], str]]()
# `token()` of the current record, computed when first asked for after a change.
_token: str | None = None
# In a worker: the coordinator's `token()` and `digests()` when the worker started.
_coordinator: tuple[str, dict[str, str]] | None = None
# The `token()` of the records last given to starting workers (`digests_for_workers`).
_sent_token: str | None = None
# The modules `record_unseen_modules` recorded; `prune` forgets them when the session ends.
_unseen = set[str]()
# The unreadable modules whose reason this process has logged (`check_stage`).
_warned = set[str]()
# What a worker imports for each module name (`worker_origin`), by name, `sys.path` and the
# working directory (a relative entry names a directory under it); `prune` forgets them when the
# session ends.
_worker_origins = dict[tuple[str, tuple[object, ...], str], import_origin.Origin | None]()
# What a new interpreter started as a worker starts reported for each module name (`_confirm`),
# by name, the worker's `sys.path` and the working directory; `prune` forgets them.
_confirmed = dict[tuple[str, tuple[str, ...], str], import_origin.Origin | None]()
# The `sys.path` and working directory for which this session already confirmed every suspected
# module the program held (`_held_suspects`); `prune` forgets them.
_confirmed_held = set[tuple[tuple[object, ...], str]]()
# The modules `sys.modules` held when the latest import session started, by name: the ones the
# program that runs Pivot loaded (`note_held_modules`). `prune` keeps them, so a pipeline
# fingerprinted after its session ended (a later discovery found nothing) still compares against
# what the program held; the next session's start replaces them. Replaced whole, never changed
# in place, so a fingerprint on another thread never reads it half made.
_held: Mapping[str, types.ModuleType] = {}
# Whether this process was forked without exec from one that imported this module: it holds
# modules its parent loaded before this process started (`_note_fork`).
_forked = False

# Manifest key kinds whose entry hashes code or a value of a user module. `ext:` records a
# function by name only (its body is not hashed), and `module:` only marks an import as covered.
_CODE_KINDS = frozenset({"func", "class", "const", "default", "schema", "partial", "loader"})

# How long before the process start a file's change time must be for the file to count as
# unchanged since. A file system that keeps whole seconds (ext4 with 128-byte inodes) stamps a
# change up to 1 s before it happened, the kernel's coarse clock up to one tick before, and a
# forward step of the wall clock between an edit and the import of this module moves the start
# later by the step. The three share this one margin: 2 s covers a step of up to 2 s where the
# file system stamps exact times, and of about 1 s where it keeps whole seconds. A file system
# that keeps change times in 2-second steps (FAT, Linux's `vfat`) spends all of it: on a FAT32 loop
# mount every change time is an even second, so an edit reads up to just under 2 s early (1.7 s
# measured), and the margin covers no clock step there (a stated limit). Linux's FAT driver also
# sets an inode's change time to its modification time when it reads the inode from disk
# (`fs/fat/inode.c`, `fat_fill_inode`), so there a modification time put back puts the change
# time back once the inode is read again; only `_bytecode_reason` can then catch the edit (a
# stated limit). The cost: a module edited less than 2 s before a process started is unreadable
# in that process.
_CHANGE_TIME_SLACK_NS = 2_000_000_000

# On Linux, the wall clock less the boot clock when this module was imported: the wall-clock time
# of boot as the wall clock read then. `_process_start_ns` takes the smaller of it and the value
# now, so a forward step of the wall clock after the import cannot move the start later.
_WALL_AT_BOOT_ON_IMPORT_NS = (
    time.time_ns() - time.clock_gettime_ns(time.CLOCK_BOOTTIME) if sys.platform == "linux" else 0
)


def _note_fork() -> None:
    global _forked
    _forked = True


os.register_at_fork(after_in_child=_note_fork)


def record(filename: str, module: str, data: bytes) -> None:
    """Record that `module` was compiled from `data`, the content of `filename` at load time."""
    path = os.path.realpath(filename)
    _store(_Loaded(filename, path, module, data, _hash(data)))
    if (loaded := sys.modules.get(module)) is not None:
        _recorded_objects[module] = loaded


def record_loaded_module(
    module: types.ModuleType, data: bytes | None, reason: str | None = None
) -> None:
    """Record a user module at the loader execution that produced this module object: from
    `data`, or as unreadable, for `reason`, when `data` is None.

    A source read afterward is not a snapshot: the file can already have changed. The discovery
    and worker import recorders call this wrapper around `loader.exec_module`.
    """
    if not fingerprint.is_user_code(module):
        return
    name = module.__name__
    filename = _module_file(module)
    if filename is None:
        return
    _recorded_objects[name] = module
    path = os.path.realpath(filename)
    if data is None:
        _store(_Loaded(filename, path, name, None, None, reason))
    else:
        _store(_Loaded(filename, path, name, data, _hash(data)))


# The reason `record_modules` gives a module the recorder loaded but has not recorded: one still
# executing, whose body imported a submodule. Its recorder records it when the execution ends.
_NO_SOURCE_RECORDED = "pivot's recorder loaded it but recorded no source for the code it ran"


def record_marker(name: str) -> object:
    """What `recorded_since` compares against: module `name`'s record now."""
    return _by_module.get(name)


def recorded_since(name: str, marker: object) -> bool:
    """Whether a recorder recorded module `name` since `record_marker(name)` returned `marker`:
    readable where Pivot's loader compiled it, or unreadable for a loader the recorder does not
    trust. A record made before, or the one `record_modules` makes for a module still executing
    (`_NO_SOURCE_RECORDED`), is not."""
    loaded = _by_module.get(name)
    return loaded is not marker and loaded is not None and loaded.reason != _NO_SOURCE_RECORDED


def record_modules(names: Iterable[str]) -> None:
    """Bring the record up to date with `sys.modules` for `names`. A module object that took the
    place of a recorded module from the same file (a module that puts a ModuleType subclass
    under its own name while it loads) keeps that record. A module loaded through the recorder
    that its recorder has not recorded yet, one still executing whose body imported a submodule,
    is unreadable (`_NO_SOURCE_RECORDED`) until the recorder records it as the execution ends. A
    module any other loader loaded is left to `record_unseen_modules`.
    """
    for name in names:
        module = sys.modules.get(name)
        if not isinstance(module, types.ModuleType) or _recorded_objects.get(name) is module:
            continue
        filename = _module_file(module) if fingerprint.is_user_code(module) else None
        if filename is None:
            _recorded_objects[name] = module
            continue
        existing = _by_module.get(name)
        if existing is not None and existing.path == os.path.realpath(filename):
            _recorded_objects[name] = module
            continue
        # The recorder wrapper is known by its type name, to avoid a circular import that the
        # import hook would loop on. A module another loader loaded stays unread here, so that
        # `record_unseen_modules` still judges it.
        loader = getattr(vars(module).get("__spec__"), "loader", None)
        if type(loader).__name__ != "_RecordingLoader":
            continue
        _recorded_objects[name] = module
        _store(_Loaded(filename, os.path.realpath(filename), name, None, None, _NO_SOURCE_RECORDED))


def record_user_modules() -> None:
    """Record every loaded user-code module not recorded yet (`record_modules`)."""
    record_modules(list(sys.modules))


def record_unseen_modules() -> None:
    """Record every loaded user module that no recorder saw compile: one imported before Pivot
    installed its recorder, by `sitecustomize`, a `.pth` import line or the program that runs
    Pivot, or by a loader the recorder does not wrap (a loader called directly, as
    `SourceFileLoader(...).load_module()` or `spec_from_loader` with `exec_module`).

    Its record holds its file's bytes where they are shown to be the bytes it was compiled from
    (`_verified_source`). Otherwise it is unreadable, with the reason, so a stage that uses it is
    never reused, and `check_stage` logs the reason once per process. Discovery calls this once
    it has imported the project, and a worker when it starts and after each stage it runs
    (`worker._untrusted_modules`), so a worker judges every user module it loaded, whatever
    loaded it; `prune` forgets these records when the import session ends, so the next discovery
    judges the modules again.
    """
    started = _process_start_ns()
    for name in list(sys.modules):
        module = sys.modules.get(name)
        if not isinstance(module, types.ModuleType) or _recorded_objects.get(name) is module:
            continue
        filename = _module_file(module) if fingerprint.is_user_code(module) else None
        _recorded_objects[name] = module
        if filename is None:
            continue
        data, why = _verified_source(module, filename, started)
        _store(
            _Loaded(
                filename,
                os.path.realpath(filename),
                name,
                data,
                None if data is None else _hash(data),
                None if why is None else _unseen_reason(why),
            )
        )
        _unseen.add(name)


def _unseen_reason(why: str) -> str:
    """Why Pivot cannot trust a module no recorder saw compile, for `why`, and what to do."""
    return (
        "pivot did not see it compile (it was imported before pivot installed its recorder, by "
        + "sitecustomize, a .pth file or the program that runs pivot, or by a loader the "
        + "recorder does not wrap), and cannot show that its file holds the code it runs, "
        + f"because {why}. Fix: import it from the pipeline or from a stage instead, with no "
        + "import hook that loads it with its own loader"
    )


def prune() -> None:
    """Forget every module that is no longer the one `sys.modules` holds under its name, as after
    an import session removed the modules it imported, and every module `record_unseen_modules`
    recorded, with the `linecache` lines pinned for them."""
    global _token
    for name, loaded in list(_by_module.items()):
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", None)
        if (
            name not in _unseen
            and isinstance(module_file, str)
            and os.path.realpath(module_file) == loaded.path
        ):
            continue
        del _by_module[name]
        _recorded_objects.pop(name, None)
        for key in (loaded.filename, loaded.path):
            if _by_file.get(key) is loaded:
                del _by_file[key]
        entry = _line_entries.pop(loaded.filename, None)
        if entry is not None and linecache.cache.get(loaded.filename) is entry:
            del linecache.cache[loaded.filename]
        _token = None
    _unseen.clear()
    _worker_origins.clear()
    _confirmed.clear()
    _confirmed_held.clear()


def note_held_modules(modules: Iterable[tuple[str, object]]) -> None:
    """Keep the modules `sys.modules` holds as an import session starts (`check_worker_imports`)."""
    global _held
    _held = {name: module for name, module in modules if isinstance(module, types.ModuleType)}


def digest(file: str) -> str | None:
    """The xxhash64 digest of the content `file` had when this process loaded a module from it,
    or None if it recorded no source for it. `file` is a module's `__file__` or its resolved
    path."""
    loaded = _by_file.get(file)
    return None if loaded is None else loaded.digest


def digests() -> dict[str, str]:
    """Every recorded user module by name: its resolved path and digest, the digest empty for an
    unreadable module, followed by the reason Pivot recorded for an unreadable one."""
    return {name: _record_text(loaded) for name, loaded in _by_module.items()}


def token() -> str:
    """A digest of `digests()`: equal tokens mean equal records."""
    global _token
    if _token is None:
        _token = _token_of(digests())
    return _token


def digests_for_workers() -> dict[str, str]:
    """`digests()`, for the initializer arguments of a pool's starting workers."""
    global _sent_token
    _sent_token = token()
    return digests()


def digests_for_stage() -> dict[str, str] | None:
    """What a stage must carry besides `token()`: None while the record is the one starting
    workers got last (`digests_for_workers`), else all of `digests()`."""
    return None if _sent_token == token() else digests()


def set_coordinator(coordinator_digests: Mapping[str, str]) -> None:
    """In a worker, keep the coordinator's `digests()` for `check_stage`."""
    global _coordinator
    _coordinator = (_token_of(coordinator_digests), dict(coordinator_digests))


def is_worker() -> bool:
    """Whether this process is a worker: it started with the coordinator's records
    (`set_coordinator`). In any other process a stage is checked against this process's own
    records, so there is no other process's record to differ from."""
    return _coordinator is not None


def coordinator_records(module_name: str) -> bool:
    """Whether this worker started with a coordinator record for `module_name`."""
    return _coordinator is not None and module_name in _coordinator[1]


def getsource(obj: Callable[..., Any] | types.CodeType) -> str:
    """`inspect.getsource(obj)`, read from the text `obj`'s file had when it was loaded.

    `inspect` reads source through `linecache`. The loaded text goes in as an entry without an
    mtime, which `linecache.checkcache` leaves alone (as it does for source a module loader
    supplied), and goes back in whenever something has cleared or replaced it. The file is found
    as `inspect.getsource` finds it, after following `__wrapped__` (a `functools.cache` helper);
    an object it cannot find a file for gets `inspect.getsource`'s own answer.
    """
    try:
        filename = inspect.getsourcefile(
            obj if isinstance(obj, types.CodeType) else inspect.unwrap(obj)
        )
    except (TypeError, ValueError):
        filename = None
    loaded = None if filename is None else _by_file.get(filename)
    if filename is not None and loaded is not None and loaded.data is not None:
        entry = _line_entries.get(filename)
        if entry is None:
            lines = _source_lines(loaded.data)
            if lines is not None:
                entry = (len(loaded.data), None, lines, filename)
                _line_entries[filename] = entry
        if entry is not None and linecache.cache.get(filename) is not entry:
            linecache.cache[filename] = entry
    return inspect.getsource(obj)


def modules_from_files(paths: set[str]) -> set[str]:
    """The names of the modules recorded, by this process or the coordinator's records it holds,
    as loaded from one of the resolved `paths`."""
    names = {name for name, loaded in _by_module.items() if loaded.path in paths}
    if _coordinator is not None:
        names.update(
            name for name, text in _coordinator[1].items() if _record_fields(text)[0] in paths
        )
    return names


def check_stage(
    manifest: Mapping[str, str],
    outputs: Collection[str],
    imported: Iterable[str],
    coordinator_token: str,
    coordinator_digests: Mapping[str, str] | None,
) -> tuple[list[str], list[str]]:
    """The user modules one stage used that it cannot trust: (edited, unreadable).

    A stage used a module when its manifest hashes that module's code or values (a `func:`,
    `class:`, `const:`, `default:`, `schema:`, `partial:` or `loader:` entry, its loaders'
    included, which the stage's output keys `outputs` place; not an `ext:` by-name entry), or
    when it imported the module while it ran (`imported`). A module another stage on this worker
    used is not this stage's concern.

    Edited: this process holds a record for the module (path and content) that differs from
    the coordinator's, so the stage ran code its fingerprint does not describe. A different file
    for the same name (a new package shadowing a module) differs too. Unreadable: either side has
    no source it can trust for the module, so no one can tell which code ran; the reason is logged
    once per process.

    The coordinator's records are the ones the worker started with, or `coordinator_digests` when
    the stage carries them. When neither matches `coordinator_token`, nothing can be compared, so
    every used module this process recorded counts as edited. A process that never got the
    coordinator's records is the coordinator itself, which has no other process's record to
    compare against: only unreadable modules count there.
    """
    known = set(_by_module)
    expected: dict[str, str] | None = None
    records_stale = False
    if _coordinator is not None:
        started_with, start_records = _coordinator
        if coordinator_digests is not None:
            expected = dict(coordinator_digests)
        elif started_with == coordinator_token:
            expected = start_records
        else:
            expected = {}
            records_stale = True
        known.update(expected)
    used = _manifest_modules(manifest, outputs, known)
    used.update(name for name in imported if name in known)
    edited = list[str]()
    unreadable = list[str]()
    for name in sorted(used):
        own = _by_module.get(name)
        if expected is None:
            if own is not None and own.digest is None:
                unreadable.append(name)
                _warn_unreadable(name, own.reason)
            continue
        wanted = expected.get(name)
        # A direct executor can receive a preloaded test or library function without discovery
        # ever sending a module record to its workers. It has no coordinator snapshot to compare;
        # do not turn that legacy programmatic path into a permanent rerun. A normal discovery
        # records every user module, unreadable included, so `wanted` is present there.
        if wanted is None:
            if records_stale and own is not None:
                edited.append(name)
            continue
        if own is not None and own.digest is None:
            unreadable.append(name)
            _warn_unreadable(name, own.reason)
        elif _record_is_unreadable(wanted):
            unreadable.append(name)
            _warn_unreadable(name, _record_fields(wanted)[2])
        elif own is not None and _record_text(own) != wanted:
            edited.append(name)
    return edited, unreadable


def check_worker_imports(manifest: Mapping[str, str], outputs: Collection[str]) -> None:
    """Refuse the stage when a module its manifest names (`fingerprint.by_name_modules`, which the
    stage's output keys `outputs` place), which the program that runs Pivot held before the
    import session started, is not the module its worker would import (`fingerprint.refuse`:
    inside `fingerprint.stage_fingerprint_scope`, the refusal is reported with the stage's other
    refusals).

    A worker imports by name each module that discovery did not register by value, with the
    `sys.path` loky gives it (`worker_path`), where an import session puts the project root first.
    A program that runs Pivot in its own process (a notebook, a script, a test) may hold a module
    of that name loaded from another file, which `import` in `pipeline.py` returns from
    `sys.modules`: the stage would be fingerprinted from that module while its worker runs the
    other one. That is a soundness risk, so it is an error.

    The check is a cheap prediction (`worker_origin`), then, only for a module the prediction
    places elsewhere, a confirmation in a new interpreter started as a worker starts (`_confirm`),
    which sees what `site`, a `sitecustomize` or a `.pth` file imports at startup. It refuses only
    when that interpreter also places the module elsewhere. The first confirmation of a session
    (for one `sys.path` and working directory) also confirms every other held module the
    prediction places elsewhere (`_held_suspects`), so the stages of a session start one
    interpreter, not one each. A module nothing would find, or whose finder raised, is left
    alone: the worker's import fails loudly. A module the session imported itself resolved on the
    worker's `sys.path`; one replaced in `sys.modules` during the session is not the program's any
    more. The program's `__main__` is never compared: a worker gets its functions by value.
    """
    held_modules = _held
    by_value = cloudpickle.list_registry_pickle_by_value()
    suspects = dict[str, import_origin.Origin]()
    for name in sorted(fingerprint.by_name_modules(manifest, outputs)):
        if (held := _suspect_origin(name, held_modules, by_value)) is not None:
            suspects[name] = held
    if not suspects:
        return
    to_confirm = suspects
    session_key = (tuple(sys.path), os.getcwd())
    if session_key not in _confirmed_held:
        _confirmed_held.add(session_key)
        to_confirm = {**_held_suspects(held_modules, by_value), **suspects}
    confirmed, failure = _confirm(sorted(to_confirm))
    problems = list[str]()
    for name, held in suspects.items():
        if failure is None:
            worker = confirmed[name]
            if worker is None or worker[0] == "unknown" or worker == held:
                continue
        else:
            worker = worker_origin(name)
            assert worker is not None
        problems.append(
            f"module '{name}' is {import_origin.describe(held)} in this process, but a worker "
            + f"would import {import_origin.describe(worker)}"
        )
    if not problems:
        return
    checked = (
        "Pivot started an interpreter as it starts a worker to confirm this"
        if failure is None
        else f"Pivot could not start an interpreter to confirm this ({failure})"
    )
    fingerprint.refuse(
        "; ".join(problems)
        + f". The stage would run code its fingerprint does not describe ({checked}). Fix: "
        + "rename one of the two modules; or, when this program imported its own module of that "
        + "name before it ran Pivot, run Pivot in a new process (the `pivot` command)."
    )


def _suspect_origin(
    name: str, held_modules: Mapping[str, types.ModuleType], by_value: set[str]
) -> import_origin.Origin | None:
    """The origin of held module `name` when the prediction places the module a worker imports
    elsewhere; None when the module is not compared, or a worker would import the same one."""
    module = sys.modules.get(name)
    if (
        name == "__main__"
        or module is None
        or held_modules.get(name) is not module
        or _pickled_by_value(name, by_value)
    ):
        return None
    held = import_origin.of_module(module)
    return None if worker_origin(name) in (None, held) else held


def _held_suspects(
    held_modules: Mapping[str, types.ModuleType], by_value: set[str]
) -> dict[str, import_origin.Origin]:
    """Every held module the prediction places elsewhere (`_suspect_origin`), by name: what one
    confirmation checks for every stage of a session. A submodule of a top-level module the
    prediction places where it is resolves in the same directories, so only the submodules of a
    suspect top-level module, or of one the program does not hold, are predicted. This only
    prepares the confirmations the stages' own checks read, so a module whose origin cannot be
    read is left to them."""
    by_top = dict[str, list[str]]()
    for name in held_modules:
        by_top.setdefault(name.partition(".")[0], []).append(name)
    suspects = dict[str, import_origin.Origin]()
    for top, names in by_top.items():
        try:
            if top in held_modules and _suspect_origin(top, held_modules, by_value) is None:
                continue
            for name in names:
                if (held := _suspect_origin(name, held_modules, by_value)) is not None:
                    suspects[name] = held
        except Exception:  # noqa: BLE001 - a module's own attributes can raise; see above
            logger.debug("Cannot predict held module %s", top, exc_info=True)
    return suspects


def _pickled_by_value(name: str, registry: set[str]) -> bool:
    """Whether cloudpickle sends module `name` by value: it or one of its packages is
    registered (as `cloudpickle.cloudpickle._is_registered_pickle_by_value` decides)."""
    parts = name.split(".")
    return any(".".join(parts[:index]) in registry for index in range(1, len(parts) + 1))


def worker_path() -> list[str]:
    """The `sys.path` a new worker searches: this process's, `""` replaced by the directory this
    process started in (`loky.backend.spawn.get_preparation_data`), a relative entry joined to
    the working directory, which loky gives the worker (a `FileFinder` makes its entry absolute
    when it is created), and without the entries that are not `str`, which `PathFinder` skips."""
    # Typeshed does not declare `ORIGINAL_DIR`; loky reads the same attribute.
    start_dir = cast("str", vars(multiprocessing.process)["ORIGINAL_DIR"])
    cwd = os.getcwd()
    # Typed `list[str]`, but a program can put any object in `sys.path` (a `pathlib.Path`).
    entries = cast("Iterable[object]", sys.path)
    return [os.path.join(cwd, entry or start_dir) for entry in entries if isinstance(entry, str)]


def worker_origin(name: str) -> import_origin.Origin | None:
    """Where a worker would import module `name` from, predicted in this process without
    importing it (`import_origin.find_spec` on `worker_path()`): None when nothing would find it,
    `("unknown", why)` when a finder raised. A worker started earlier from a warm pool keeps the
    `sys.path` and working directory it started with."""
    key = (name, tuple(sys.path), os.getcwd())
    if key in _worker_origins:
        return _worker_origins[key]
    try:
        spec = import_origin.find_spec(name, worker_path())
    except Exception as exc:  # noqa: BLE001 - a finder's error leaves the origin unknown
        origin: import_origin.Origin | None = ("unknown", f"{type(exc).__name__}: {exc}")
    else:
        origin = None if spec is None else import_origin.of_spec(spec)
    _worker_origins[key] = origin
    return origin


# A new interpreter started as a loky worker starts (`python -m loky.backend.popen_loky_posix`,
# with this process's environment and working directory): it imports what that module imports
# before `prepare()` gives it the worker's `sys.path`, then reports from `import_origin`, loaded
# by its file so that nothing of pivot's is imported. The report goes to the file descriptor in
# `sys.argv[2]`, since `site`, a `sitecustomize` or a `.pth` file may write to stdout.
_CONFIRM_PROGRAM = """\
import sys
import argparse, loky.backend.popen_loky_posix
import importlib.util
spec = importlib.util.spec_from_file_location("_pivot_import_origin", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.report_worker_origins(int(sys.argv[2]))
"""
_CONFIRM_TIMEOUT_S = 120


def _confirm(
    names: list[str],
) -> tuple[dict[str, import_origin.Origin | None], str | None]:
    """What an interpreter started as a worker starts reports for each of `names`
    (`import_origin.report_worker_origins`), and None; or no results and why it could not run."""
    path = worker_path()
    cwd = os.getcwd()
    results = dict[str, import_origin.Origin | None]()
    missing = list[str]()
    for name in names:
        key = (name, tuple(path), cwd)
        if key in _confirmed:
            results[name] = _confirmed[key]
        else:
            missing.append(name)
    if not missing:
        return results, None
    payload = "\0".join([str(len(missing)), *missing, *path])
    with tempfile.TemporaryFile() as report:
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _CONFIRM_PROGRAM,
                    import_origin.__file__,
                    str(report.fileno()),
                ],
                input=os.fsencode(payload),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                pass_fds=(report.fileno(),),
                timeout=_CONFIRM_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {}, f"{type(exc).__name__}: {exc}"
        if proc.returncode != 0:
            lines = proc.stderr.decode(errors="replace").strip().splitlines()
            return {}, f"exit {proc.returncode}: {lines[-1] if lines else 'no output'}"
        report.seek(0)
        text = report.read()
    try:
        reported = cast("dict[str, import_origin.Origin | None]", ast.literal_eval(text.decode()))
    except (ValueError, SyntaxError, UnicodeDecodeError) as exc:
        return {}, f"unreadable report: {exc}"
    for name in missing:
        results[name] = _confirmed[(name, tuple(path), cwd)] = reported.get(name)
    return results, None


def _warn_unreadable(name: str, reason: str | None) -> None:
    """Log, once per process, why Pivot cannot trust an unreadable module's code."""
    if not reason or name in _warned:
        return
    _warned.add(name)
    logger.warning(
        "Pivot cannot trust the code of user module '%s', so a stage that uses it is never "
        + "reused: %s.",
        name,
        reason,
    )


def _manifest_modules(
    manifest: Mapping[str, str], outputs: Collection[str], known: set[str]
) -> set[str]:
    """The names in `known` whose code or values a manifest entry hashes: an entry whose kind is
    one of `_CODE_KINDS`, with the module named first in what follows the kind (up to a `:`).
    Each entry is read as its walk keyed it (`fingerprint.manifest_entries`, which the stage's
    output keys `outputs` place), so no dependency, output, field or stage named like a kind is
    read as one."""
    used = set[str]()
    for entry in fingerprint.manifest_entries(manifest, outputs):
        kind, colon, rest = entry.partition(":")
        if not colon or kind not in _CODE_KINDS:
            continue
        dotted = rest.partition(":")[0].split(".")
        for end in range(1, len(dotted) + 1):
            prefix = ".".join(dotted[:end])
            if prefix in known:
                used.add(prefix)
    return used


def _store(loaded: _Loaded) -> None:
    global _token
    _unseen.discard(loaded.module)
    previous = _by_module.get(loaded.module)
    if previous is not None:
        for key in (previous.filename, previous.path):
            if _by_file.get(key) is previous:
                del _by_file[key]
        if previous.filename != loaded.filename:
            entry = _line_entries.pop(previous.filename, None)
            if entry is not None and linecache.cache.get(previous.filename) is entry:
                del linecache.cache[previous.filename]
    _by_module[loaded.module] = loaded
    if loaded.data is not None:
        _by_file[loaded.filename] = loaded
        _by_file[loaded.path] = loaded
    _line_entries.pop(loaded.filename, None)
    _token = None


def _hash(data: bytes) -> str:
    return xxhash.xxh64(data).hexdigest()


def _record_text(loaded: _Loaded) -> str:
    text = f"{loaded.path}\0{loaded.digest or ''}"
    return text if loaded.reason is None else f"{text}\0{loaded.reason}"


def _record_fields(text: str) -> tuple[str, str, str]:
    """A record's resolved path, digest (empty for an unreadable module) and reason (empty unless
    the module is unreadable and Pivot recorded why)."""
    path, _, rest = text.partition("\0")
    digest, _, reason = rest.partition("\0")
    return path, digest, reason


def _record_is_unreadable(text: str) -> bool:
    return not _record_fields(text)[1]


def _module_file(module: types.ModuleType) -> str | None:
    """The file a module's code came from: its `__file__`, else its spec's origin."""
    filename = vars(module).get("__file__")
    if not isinstance(filename, str):
        filename = getattr(vars(module).get("__spec__"), "origin", None)
    return filename if isinstance(filename, str) else None


def _process_start_ns() -> int | None:
    """When this process started, in wall-clock nanoseconds, or None if the system does not say,
    or this process was forked from one that imported this module: its modules were loaded before
    it started.

    Linux gives the start in clock ticks since boot. Added to the wall-clock time of boot, the
    smaller of its value when this module was imported and now, it lands within one tick before
    the true start, unless the wall clock stepped forward between the start and the import of
    this module: then it is late by the step. psutil reads it elsewhere.
    """
    if _forked:
        return None
    if sys.platform != "linux":
        import psutil

        try:
            return int(psutil.Process().create_time() * 1_000_000_000)
        except psutil.Error:
            return None
    try:
        with open("/proc/self/stat") as stat:
            fields = stat.read().rpartition(")")[2].split()
        boot_ns = int(fields[19]) * 1_000_000_000 // os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None
    wall_at_boot = time.time_ns() - time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    return boot_ns + min(_WALL_AT_BOOT_ON_IMPORT_NS, wall_at_boot)


def _verified_source(
    module: types.ModuleType, filename: str, started: int | None
) -> tuple[bytes | None, str | None]:
    """`filename`'s bytes if they are the bytes `module` was compiled from, else None and why
    Pivot cannot show that.

    No recorder saw the module compile, so its file is read now. Those bytes are its code when
    CPython's source file loader imported it (a plain `sys.path` import), the file's change time
    is at least `_CHANGE_TIME_SLACK_NS` older than this process's start, so every import in this
    process read these bytes, and the bytecode that loader may have run instead is the code of
    these bytes (`_bytecode_reason`). A forward step of the wall clock between an edit and the
    import of this module spends that margin, less what the file system's timestamp granularity
    already takes of it, and a larger step makes the edit look older than the start.
    """
    spec = vars(module).get("__spec__")
    loader = getattr(spec, "loader", None) or vars(module).get("__loader__")
    if type(loader) is not importlib.machinery.SourceFileLoader:
        name = f"{type(loader).__module__}.{type(loader).__qualname__}"
        return None, f"its loader, {name}, is not CPython's source file loader"
    if started is None:
        return None, "pivot cannot tell when this process loaded it"
    since = started - _CHANGE_TIME_SLACK_NS
    try:
        with open(filename, "rb") as file:
            data = file.read()
            source = os.fstat(file.fileno())
        link = os.lstat(filename)
    except OSError as error:
        return None, f"its file cannot be read ({error.strerror})"
    if max(source.st_ctime_ns, link.st_ctime_ns) >= since or len(data) != source.st_size:
        return None, "its file changed after this process started"
    reason = _bytecode_reason(filename, data, source, since)
    return (data, None) if reason is None else (None, reason)


def _bytecode_reason(filename: str, data: bytes, source: os.stat_result, since: int) -> str | None:
    """Why the bytecode CPython's source file loader may have run for `filename` might not be the
    code of `data`, the file's bytes since before `since`; None when it is, or was not run.

    The loader runs the cached file when it accepts it for the source (`_bytecode_accepted`) and
    compiles the source otherwise. A cached file that has not changed since `since` is the one
    the import saw, so it must be rejected for this source or be the code of `data`. A cached
    file written since then replaced the one the import saw. CPython's import writes mtime-checked
    bytecode only after rejecting the file it found for the source, which this process saw too,
    so it then compiled the source as well; such a file must also be the code of `data`. Any other
    replacement could have followed an import that ran stale bytecode. With no cached file, and
    none removed from its directory since `since`, the import compiled the source.
    """
    try:
        cached = importlib.util.cache_from_source(filename)
    except NotImplementedError:  # this interpreter caches no bytecode
        return None
    try:
        with open(cached, "rb") as file:
            bytecode = file.read()
            changed = os.fstat(file.fileno()).st_ctime_ns >= since
    except FileNotFoundError:
        try:
            removed = os.stat(os.path.dirname(cached)).st_ctime_ns >= since
        except FileNotFoundError:
            return None
        except OSError as error:
            return f"its bytecode directory cannot be read ({error.strerror})"
        return "its bytecode directory changed after this process started" if removed else None
    except OSError as error:
        return f"its bytecode file cannot be read ({error.strerror})"
    accepted = _bytecode_accepted(bytecode, data, source)
    if not changed and not accepted:
        return None
    if accepted and (not changed or bytecode[4:8] == bytes(4)):
        # Compiling here must not repeat the module's SyntaxWarnings: its import showed them.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                compiled = compile(data, filename, "exec", dont_inherit=True)
                if marshal.loads(bytecode[16:]) == compiled:
                    return None
            except (EOFError, TypeError, ValueError, SyntaxError):
                pass
    if changed:
        return "its bytecode file changed after this process started"
    return "its cached bytecode is not the code of its file"


def _bytecode_accepted(bytecode: bytes, data: bytes, source: os.stat_result) -> bool:
    """Whether CPython's source file loader runs `bytecode` for a source file with this content
    and status rather than compiling it, as `SourceLoader.get_code` decides (PEP 552)."""
    if len(bytecode) < 16 or bytecode[:4] != importlib.util.MAGIC_NUMBER:
        return False
    flags = int.from_bytes(bytecode[4:8], "little")
    if flags == 0:
        return bytecode[8:16] == (int(source.st_mtime) & 0xFFFFFFFF).to_bytes(4, "little") + (
            source.st_size & 0xFFFFFFFF
        ).to_bytes(4, "little")
    if flags & ~0b11:
        return False
    if _imp.check_hash_based_pycs == "never" or (
        _imp.check_hash_based_pycs == "default" and not flags & 0b10
    ):
        return True
    return bytecode[8:16] == importlib.util.source_hash(data)


def _token_of(records: Mapping[str, str]) -> str:
    return _hash(json.dumps(sorted(records.items())).encode())


def _source_lines(data: bytes) -> list[str] | None:
    """Split source as `linecache.updatecache` splits a file (`tokenize.open`, universal
    newlines, a final newline added), so the text `inspect` returns is the one it would read
    from a file with this content. None if the text cannot be decoded."""
    buffer = io.BytesIO(data)
    try:
        encoding, _ = tokenize.detect_encoding(buffer.readline)
        buffer.seek(0)
        lines = io.TextIOWrapper(buffer, encoding, line_buffering=True).readlines()
    except (SyntaxError, UnicodeDecodeError, LookupError):
        return None
    if not lines:
        return ["\n"]
    if not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines
