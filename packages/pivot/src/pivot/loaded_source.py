"""The source text each user module was compiled from, recorded when this process loaded it.

A stage's fingerprint must describe the code its worker runs. The worker receives a project
module's functions by value: the code this process compiled when discovery imported the module.
`inspect.getsource` reads a file's current lines through `linecache`, which re-reads a file whose
size or mtime changed, so a file edited after the import (by an upstream stage, or in an editor
during a run) would be hashed as edited while the worker runs the version that was loaded.
Fingerprinting therefore reads source and source digests from what `record` kept, and a worker
compares the modules it imported itself against what the coordinator loaded (`edited_modules`).

Import machinery and `linecache` are process-global, so this record is too. A file loaded again
(a watch reload) replaces its entry, and `prune` drops the entries of modules no longer loaded.
A worker gets the coordinator's digests once, when it starts (`digests_for_workers`,
`set_coordinator`); a stage carries their `token`, and all of them only when the coordinator's
record changed after the workers started (`digests_for_stage`), as when a user module imports
another from a thread it started.
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

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class _Loaded(NamedTuple):
    filename: str  # the name the compiled code carries (`co_filename`, the module's `__file__`)
    path: str  # `filename` resolved
    module: str
    data: bytes
    digest: str


# Every loaded file, by the name its code carries and by its resolved path.
_loaded = dict[str, _Loaded]()
# The `linecache` entry made from each loaded file's text, by the name its code carries.
_line_entries = dict[str, tuple[int, None, list[str], str]]()
# `token()` of the current record, computed when first asked for after a change.
_token: str | None = None
# In a worker: the coordinator's `token()` and `digests()` when the worker started.
_coordinator: tuple[str, dict[str, str]] | None = None
# The `token()` of the digests last given to starting workers (`digests_for_workers`).
_sent_token: str | None = None


def record(filename: str, module: str, data: bytes) -> None:
    """Record that `module` was compiled from `data`, the content of `filename` at load time."""
    global _token
    path = os.path.realpath(filename)
    loaded = _Loaded(filename, path, module, data, xxhash.xxh64(data).hexdigest())
    _loaded[filename] = loaded
    _loaded[path] = loaded
    _line_entries.pop(filename, None)
    _token = None


def prune() -> None:
    """Forget every file whose module is no longer the one `sys.modules` holds under its name,
    as after an import session removed the modules it imported."""
    global _token
    for key, loaded in list(_loaded.items()):
        module = sys.modules.get(loaded.module)
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or os.path.realpath(module_file) != loaded.path:
            del _loaded[key]
            _line_entries.pop(key, None)
            _token = None


def digest(file: str) -> str | None:
    """The xxhash64 digest of the content `file` had when this process loaded it, or None if
    this process loaded no module from it. `file` is a module's `__file__` or its resolved path."""
    loaded = _loaded.get(file)
    return None if loaded is None else loaded.digest


def digests() -> dict[str, str]:
    """The digest of every file this process loaded, by resolved path."""
    return {path: loaded.digest for path, loaded in _loaded.items() if path == loaded.path}


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
    """In a worker, keep the coordinator's `digests()` for `edited_modules`."""
    global _coordinator
    _coordinator = (_token_of(coordinator_digests), dict(coordinator_digests))


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
    if filename is not None and (loaded := _loaded.get(filename)) is not None:
        entry = _line_entries.get(filename)
        if entry is None:
            lines = _source_lines(loaded.data)
            if lines is not None:
                entry = (len(loaded.data), None, lines, filename)
                _line_entries[filename] = entry
        if entry is not None and linecache.cache.get(filename) is not entry:
            linecache.cache[filename] = entry
    return inspect.getsource(obj)


def edited_modules(
    coordinator_token: str, coordinator_digests: Mapping[str, str] | None
) -> list[str]:
    """In a worker, the modules it loaded itself from a file whose content differs from what the
    coordinator loaded from it. The stage carries the coordinator's `token()` when it sent the
    stage, and its `digests_for_stage()`.

    A process that never got the coordinator's digests is the coordinator, which ran what it
    loaded: none. A stage that carries no digests is compared with those the worker started
    with; if those are not the ones the stage was sent with (the token differs), there is
    nothing to compare against, so every module the worker loaded itself counts as changed.
    """
    if _coordinator is None:
        return []
    started_with, loaded_elsewhere = _coordinator
    own = [loaded for path, loaded in _loaded.items() if path == loaded.path]
    if coordinator_digests is not None:
        loaded_elsewhere = coordinator_digests
    elif started_with != coordinator_token:
        return sorted({loaded.module for loaded in own})
    return sorted(
        {
            loaded.module
            for loaded in own
            if (other := loaded_elsewhere.get(loaded.path)) is not None and other != loaded.digest
        }
    )


def _token_of(file_digests: Mapping[str, str]) -> str:
    return xxhash.xxh64(json.dumps(sorted(file_digests.items())).encode()).hexdigest()


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
