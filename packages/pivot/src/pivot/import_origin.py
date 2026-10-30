"""Where a module comes from, as a Pivot worker imports it by name.

A leaf: it imports nothing from pivot and only modules a loky worker has loaded before its
`sys.path` is set, so `loaded_source` can run it in a new interpreter started as a worker starts
(`report_worker_origins`) without changing what that interpreter holds.

An origin is a tuple: `("file", <resolved path>)`, `("namespace", <resolved directories>...)`,
`("built-in",)`, `("frozen",)`, `("no file",)`, or `("unknown", <why>)` when a finder raised.
"""

from __future__ import annotations

import importlib.machinery
import os
import sys
import types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

type Origin = tuple[str, ...]

NO_FILE: Origin = ("no file",)

_BUILTIN_OR_FROZEN = ("built-in", "frozen")


def of_module(module: types.ModuleType) -> Origin:
    """The origin of a loaded module: its spec's location, else its `__file__`, else the
    directories of a namespace package."""
    spec = getattr(module, "__spec__", None)
    if isinstance(spec, importlib.machinery.ModuleSpec):
        if not spec.has_location and spec.origin in _BUILTIN_OR_FROZEN:
            return (spec.origin,)
        if spec.has_location and isinstance(spec.origin, str):
            return ("file", os.path.realpath(spec.origin))
    filename = getattr(module, "__file__", None)
    if isinstance(filename, str):
        return ("file", os.path.realpath(filename))
    path = getattr(module, "__path__", None)
    if path is not None:
        return _namespace(path)
    return NO_FILE


def of_spec(spec: importlib.machinery.ModuleSpec) -> Origin:
    """The origin of a module an import would load from `spec`."""
    if not spec.has_location and spec.origin in _BUILTIN_OR_FROZEN:
        return (spec.origin,)
    if spec.has_location and isinstance(spec.origin, str):
        return ("file", os.path.realpath(spec.origin))
    if spec.submodule_search_locations is not None:
        return _namespace(spec.submodule_search_locations)
    return NO_FILE


def describe(origin: Origin) -> str:
    """An origin in an error message."""
    match origin:
        case ("file", path):
            return path
        case ("namespace", *directories):
            return f"a namespace package in {', '.join(directories) or 'no directory'}"
        case ("built-in",):
            return "a built-in module"
        case ("frozen",):
            return "a frozen module"
        case ("unknown", why):
            return f"unknown (a finder raised {why})"
        case _:
            return "a module with no file"


def find_spec(name: str, path: Sequence[str]) -> importlib.machinery.ModuleSpec | None:
    """The spec an import of `name` finds, with `path` as `sys.path`, without importing it or
    its packages: built-in and frozen modules first (no file shadows them), then each finder of
    `sys.meta_path` in order, `PathFinder` searching `path`. A submodule is searched in its
    package's spec's directories, as an import searches the package's `__path__`."""
    search: Sequence[str] | None = None
    spec: importlib.machinery.ModuleSpec | None = None
    parts = name.split(".")
    for index in range(len(parts)):
        spec = _find_one(".".join(parts[: index + 1]), search, path)
        if spec is None:
            return None
        search = spec.submodule_search_locations
        if search is None and index + 1 < len(parts):
            return None
    return spec


def _find_one(
    fullname: str, search: Sequence[str] | None, path: Sequence[str]
) -> importlib.machinery.ModuleSpec | None:
    if search is None:
        if fullname in sys.builtin_module_names:
            return importlib.machinery.BuiltinImporter.find_spec(fullname)
        frozen = importlib.machinery.FrozenImporter.find_spec(fullname)
        if frozen is not None:
            return frozen
    for finder in tuple(sys.meta_path):
        if isinstance(finder, type) and issubclass(
            finder, (importlib.machinery.BuiltinImporter, importlib.machinery.FrozenImporter)
        ):
            continue
        # Discovery's recorder only asks the finders after it (`discovery._RecordingFinder`).
        if type(finder).__module__ == "pivot.discovery":
            continue
        if finder is importlib.machinery.PathFinder:
            spec = importlib.machinery.PathFinder.find_spec(
                fullname, list(path if search is None else search)
            )
        else:
            finder_find_spec = getattr(finder, "find_spec", None)
            if finder_find_spec is None:
                continue
            spec = finder_find_spec(fullname, None if search is None else list(search))
        if spec is not None:
            return spec
    return None


def _namespace(directories: Iterable[str]) -> Origin:
    return ("namespace", *sorted(os.path.realpath(directory) for directory in directories))


def report_worker_origins() -> None:
    """Write to stdout the origin of each module named on stdin, in an interpreter started as a
    Pivot worker starts, once given the worker's `sys.path` (also on stdin) as loky's `prepare()`
    gives it: the loaded module's origin when startup already imported it (by `site`, a
    `sitecustomize` or a `.pth` import line, or loky's own imports), else the origin of the spec
    an import would find (`find_spec`), or None when nothing would. Nothing is imported.

    Input: NUL-separated fields, encoded with `os.fsencode`: the number of names, the names, then
    the entries of the worker's `sys.path`. Output: `repr` of a dict from name to origin.
    """
    fields = os.fsdecode(sys.stdin.buffer.read()).split("\0")
    count = int(fields[0])
    names, path = fields[1 : 1 + count], fields[1 + count :]
    sys.path = path
    results = dict[str, Origin | None]()
    for name in names:
        module = sys.modules.get(name)
        if isinstance(module, types.ModuleType):
            results[name] = of_module(module)
            continue
        try:
            spec = find_spec(name, path)
        except Exception as exc:  # noqa: BLE001 - a finder's error is reported, not raised
            results[name] = ("unknown", f"{type(exc).__name__}: {exc}")
        else:
            results[name] = None if spec is None else of_spec(spec)
    sys.stdout.write(repr(results))
