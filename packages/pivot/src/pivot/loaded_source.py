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
zip on `sys.path`, `importlib.util.spec_from_file_location`, a sourceless `.pyc`) is recorded from
its loader when it appears (`record_modules`, `record_user_modules`). A user module whose loader
gives no source is recorded as unreadable: Pivot cannot tell which code it holds, so a stage that
uses it is never trusted. A user module no recorder saw compile, because it was imported before
the recorder was installed (by `sitecustomize`, a `.pth` import line or the program that runs
Pivot) or by a loader called directly (`SourceFileLoader(...).load_module()`), is recorded from
its file only where Pivot can show the file holds the code it was compiled from, and as
unreadable otherwise (`record_unseen_modules`, at the end of discovery, when a worker starts and
after each stage a worker runs).

Import machinery and `linecache` are process-global, so this record is too. A module loaded again
(a watch reload) replaces its entry, and `prune`, at the end of an import session, drops the
entries, and the `linecache` lines, of modules no longer loaded and of modules no recorder saw
compile. A worker gets the coordinator's records once, when it starts
(`digests_for_workers`, `set_coordinator`); a stage carries their `token`, and all of them only
when the coordinator's record changed after the workers started (`digests_for_stage`).
"""

from __future__ import annotations

import _imp
import importlib.machinery
import importlib.util
import inspect
import io
import json
import linecache
import logging
import marshal
import os
import sys
import time
import tokenize
import types
import warnings
from typing import TYPE_CHECKING, Any, NamedTuple

import xxhash

# `fingerprint` imports this module; the cycle resolves through `sys.modules`. A local import
# would run inside the worker's import hook, which calls `record_modules`, and recurse.
from pivot import fingerprint

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

logger = logging.getLogger(__name__)


class _Loaded(NamedTuple):
    """One user module's source when this process loaded it.

    `data` and `digest` are None for a module whose loader gives no source: it is unreadable.
    `reason` says why a module no recorder saw compile is unreadable (`record_unseen_modules`).
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
# file system stamps exact times, and of about 1 s where it keeps whole seconds. The cost: a
# module edited less than 2 s before a process started is unreadable in that process.
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


def record_loaded_module(module: types.ModuleType, data: bytes | None) -> None:
    """Record a user module at the loader execution that produced this module object.

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
    _store(_Loaded(filename, path, name, data, None if data is None else _hash(data)))


def record_modules(names: Iterable[str]) -> None:
    """Bring the record up to date with `sys.modules` for `names`. A module object that took the
    place of a recorded module from the same file (a module that puts a ModuleType subclass
    under its own name while it loads) keeps that record. A module loaded through the recorder
    whose source the recorder did not capture (the unpickling path, ``importlib.reload``) is
    unreadable. A module any other loader loaded is left to `record_unseen_modules`.
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
        _store(_Loaded(filename, os.path.realpath(filename), name, None, None))


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
        data, reason = _verified_source(module, filename, started)
        _store(
            _Loaded(
                filename,
                os.path.realpath(filename),
                name,
                data,
                None if data is None else _hash(data),
                reason,
            )
        )
        _unseen.add(name)


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


def digest(file: str) -> str | None:
    """The xxhash64 digest of the content `file` had when this process loaded a module from it,
    or None if it recorded no source for it. `file` is a module's `__file__` or its resolved
    path."""
    loaded = _by_file.get(file)
    return None if loaded is None else loaded.digest


def digests() -> dict[str, str]:
    """Every recorded user module by name: its resolved path and digest, the digest empty for an
    unreadable module, followed by the reason for one no recorder saw compile."""
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
    imported: Iterable[str],
    coordinator_token: str,
    coordinator_digests: Mapping[str, str] | None,
) -> tuple[list[str], list[str]]:
    """The user modules one stage used that it cannot trust: (edited, unreadable).

    A stage used a module when its manifest hashes that module's code or values (a `func:`,
    `class:`, `const:`, `default:`, `schema:`, `partial:` or `loader:` entry, not an `ext:`
    by-name entry), or when it imported the module while it ran (`imported`). A module another
    stage on this worker used is not this stage's concern.

    Edited: this process holds a record for the module (path and content) that differs from
    the coordinator's, so the stage ran code its fingerprint does not describe. A different file
    for the same name (a new package shadowing a module) differs too. Unreadable: either side has
    no source for the module, so no one can tell which code ran. For a module no recorder saw
    compile (`record_unseen_modules`), the reason is logged once per process.

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
    used = _manifest_modules(manifest, known)
    used.update(name for name in imported if name in known)
    edited = list[str]()
    unreadable = list[str]()
    for name in sorted(used):
        own = _by_module.get(name)
        if expected is None:
            if own is not None and own.digest is None:
                unreadable.append(name)
                _warn_unseen(name, own.reason)
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
            _warn_unseen(name, own.reason)
        elif _record_is_unreadable(wanted):
            unreadable.append(name)
            _warn_unseen(name, _record_fields(wanted)[2])
        elif own is not None and _record_text(own) != wanted:
            edited.append(name)
    return edited, unreadable


def _warn_unseen(name: str, reason: str | None) -> None:
    """Log, once per process, why a module no recorder saw compile cannot be trusted."""
    if not reason or name in _warned:
        return
    _warned.add(name)
    logger.warning(
        "Pivot did not see user module '%s' compile: it was imported before pivot installed its "
        + "recorder (by sitecustomize, a .pth file or the program that runs pivot), or by a "
        + "loader the recorder does not wrap. Pivot cannot show that its file holds the code it "
        + "runs, because %s, so a stage that uses it is never reused. Fix: import it from the "
        + "pipeline or from a stage instead.",
        name,
        reason,
    )


def _manifest_modules(manifest: Mapping[str, str], known: set[str]) -> set[str]:
    """The names in `known` whose code or values a manifest entry hashes (`_CODE_KINDS`)."""
    used = set[str]()
    for key in manifest:
        parts = key.split(":")
        for index, kind in enumerate(parts[:-1]):
            if kind not in _CODE_KINDS:
                continue
            dotted = parts[index + 1].split(".")
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


def loader_source(loader: object, filename: str, module_name: str) -> bytes | None:
    """The source a module's loader gives before it executes: raw `.py` bytes (`get_data`), else
    decoded loader source (`get_source`). None means the module is unreadable."""
    if filename.endswith(".py"):
        get_data = getattr(loader, "get_data", None)
        if callable(get_data):
            try:
                data = get_data(filename)
            except (ImportError, OSError):
                data = None
            if isinstance(data, bytes):
                return data
    get_source = getattr(loader, "get_source", None)
    if not callable(get_source):
        return None
    try:
        source = get_source(module_name)
    except (ImportError, OSError, SyntaxError, UnicodeDecodeError):
        return None
    return source.encode() if isinstance(source, str) else None


def _hash(data: bytes) -> str:
    return xxhash.xxh64(data).hexdigest()


def _record_text(loaded: _Loaded) -> str:
    text = f"{loaded.path}\0{loaded.digest or ''}"
    return text if loaded.reason is None else f"{text}\0{loaded.reason}"


def _record_fields(text: str) -> tuple[str, str, str]:
    """A record's resolved path, digest (empty for an unreadable module) and reason (empty unless
    no recorder saw the module compile)."""
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
        return None, f"its loader ({type(loader).__qualname__}) is not CPython's source file loader"
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
