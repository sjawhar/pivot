"""The source each user module was compiled from, recorded when this process loaded it.

A stage's fingerprint must describe the code its worker runs. The worker receives a project
module's functions by value: the code this process compiled when discovery imported the module.
`inspect.getsource` reads a file's current lines through `linecache`, which re-reads a file whose
size or mtime changed, so a file edited after the import (by an upstream stage, or in an editor
during a run) would be hashed as edited while the worker runs the version that was loaded.
Fingerprinting therefore reads source and source digests from what this module recorded, and a
worker compares the modules a stage used against what the coordinator loaded (`check_stage`).

The contract is about the module, not the loader. `discovery._ProjectSourceLoader` records the
bytes it compiles; every other user-code module (an editable install's finder, a zip on
`sys.path`, `importlib.util.spec_from_file_location`, a sourceless `.pyc`) is recorded from its
loader when it appears (`record_modules`, `record_user_modules`). A user module whose loader gives
no source is recorded as unreadable: Pivot cannot tell which code it holds, so a stage that uses
it is never trusted.

Import machinery and `linecache` are process-global, so this record is too. A module loaded again
(a watch reload) replaces its entry, and `prune` drops the entries, and the `linecache` lines, of
modules no longer loaded. A worker gets the coordinator's records once, when it starts
(`digests_for_workers`, `set_coordinator`); a stage carries their `token`, and all of them only
when the coordinator's record changed after the workers started (`digests_for_stage`).
"""

from __future__ import annotations

import inspect
import io
import json
import linecache
import os
import sys
import tokenize
import types
from typing import TYPE_CHECKING, Any, NamedTuple

import xxhash

# `fingerprint` imports this module; the cycle resolves through `sys.modules`. A local import
# would run inside the worker's import hook, which calls `record_modules`, and recurse.
from pivot import fingerprint

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping


class _Loaded(NamedTuple):
    """One user module's source when this process loaded it.

    `data` and `digest` are None for a module whose loader gives no source: it is unreadable.
    """

    filename: str  # the name the compiled code carries (`co_filename`, the module's `__file__`)
    path: str  # `filename` resolved
    module: str
    data: bytes | None
    digest: str | None


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

# Manifest key kinds whose entry hashes code or a value of a user module. `ext:` records a
# function by name only (its body is not hashed), and `module:` only marks an import as covered.
_CODE_KINDS = frozenset({"func", "class", "const", "default", "schema", "partial", "loader"})


def record(filename: str, module: str, data: bytes) -> None:
    """Record that `module` was compiled from `data`, the content of `filename` at load time."""
    path = os.path.realpath(filename)
    _store(_Loaded(filename, path, module, data, xxhash.xxh64(data).hexdigest()))
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
    filename = vars(module).get("__file__")
    spec = vars(module).get("__spec__")
    if not isinstance(filename, str):
        filename = getattr(spec, "origin", None)
    if not isinstance(filename, str):
        return
    _recorded_objects[name] = module
    path = os.path.realpath(filename)
    _store(_Loaded(filename, path, name, data, None if data is None else _hash(data)))


def record_modules(names: Iterable[str], *, refresh: bool = False) -> None:
    """Record user modules that appear to have been loaded through the recorder but whose source
    the recorder did not capture (the unpickling path, ``importlib.reload``). Modules the
    recorder already captured are recorded with source. Modules loaded before the recorder was
    installed are left unrecorded: fingerprinting reads their file directly, as on main.
    """
    for name in names:
        module = sys.modules.get(name)
        if not isinstance(module, types.ModuleType) or (
            not refresh and _recorded_objects.get(name) is module
        ):
            continue
        _recorded_objects[name] = module
        if not fingerprint.is_user_code(module):
            continue
        filename = vars(module).get("__file__")
        spec = vars(module).get("__spec__")
        if not isinstance(filename, str):
            filename = getattr(spec, "origin", None)
        if not isinstance(filename, str):
            continue
        existing = _by_module.get(name)
        if existing is not None and existing.path == os.path.realpath(filename) and not refresh:
            continue
        # Only record a module whose loader is the recorder wrapper (its type name, to avoid a
        # circular import that the import hook would loop on). A module whose loader is anything
        # else was loaded before the recorder was installed and is not our concern.
        loader = getattr(spec, "loader", None)
        if not refresh and type(loader).__name__ != "_RecordingLoader":
            continue
        _store(_Loaded(filename, os.path.realpath(filename), name, None, None))


def record_user_modules() -> None:
    """Record every loaded user-code module not recorded yet (`record_modules`)."""
    record_modules(list(sys.modules))


def prune() -> None:
    """Forget every module that is no longer the one `sys.modules` holds under its name, as after
    an import session removed the modules it imported, with the `linecache` lines pinned for it."""
    global _token
    for name, loaded in list(_by_module.items()):
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", None)
        if isinstance(module_file, str) and os.path.realpath(module_file) == loaded.path:
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


def digest(file: str) -> str | None:
    """The xxhash64 digest of the content `file` had when this process loaded a module from it,
    or None if it recorded no source for it. `file` is a module's `__file__` or its resolved
    path."""
    loaded = _by_file.get(file)
    return None if loaded is None else loaded.digest


def digests() -> dict[str, str]:
    """Every recorded user module by name: its resolved path and digest, the digest empty for an
    unreadable module."""
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
            name for name, text in _coordinator[1].items() if text.rpartition("\0")[0] in paths
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
    no source for the module, so no one can tell which code ran.

    The coordinator's records are the ones the worker started with, or `coordinator_digests` when
    the stage carries them. When neither matches `coordinator_token`, nothing can be compared, so
    every used module this process recorded counts as edited. A process that never got the
    coordinator's records is the coordinator itself, which ran what it fingerprinted: only
    unreadable modules count there.
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
        wanted = None if expected is None else expected.get(name)
        if _coordinator is None:
            if own is not None and own.digest is None:
                unreadable.append(name)
            continue
        # A direct executor can receive a preloaded test or library function without discovery
        # ever sending a module record to its workers. It has no coordinator snapshot to compare;
        # do not turn that legacy programmatic path into a permanent rerun. A normal discovery
        # records every user module, unreadable included, so `wanted` is present there.
        if wanted is None:
            if records_stale and own is not None:
                edited.append(name)
            continue
        if (own is not None and own.digest is None) or _record_is_unreadable(wanted):
            unreadable.append(name)
        elif own is not None and _record_text(own) != wanted:
            edited.append(name)
    return edited, unreadable


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
    return f"{loaded.path}\0{loaded.digest or ''}"


def _record_is_unreadable(text: str) -> bool:
    return not text.rpartition("\0")[2]


def _token_of(records: Mapping[str, str]) -> str:
    return xxhash.xxh64(json.dumps(sorted(records.items())).encode()).hexdigest()


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
