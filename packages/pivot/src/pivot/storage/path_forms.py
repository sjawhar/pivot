"""The two forms of a path, and the stored paths a path is, lies inside or holds in them.

A path's forms are its absolute normalized path and its resolved path (``os.path.realpath``,
which resolves a missing path's existing ancestors), so a path that reaches another through a
symlink meets it in its resolved form. A ``pygtrie.Trie`` keyed by ``Path.parts`` decides
containment, never a string prefix: ``data-backup/x`` is not inside ``data``. These checks of
a path against stored paths by both forms ask here: the missing tracked files a stage reads and
the paths a ``--checkout-missing`` restore creates (``engine.graph.get_runnable_stages``), the
restorable paths a dependency's restore changes (``engine.graph.tracked_overlap``), and the
stage outputs a missing tracked path overlaps (``ownership.tracked_output_refusal``). Each
caller composes its own order of forms and questions from these primitives, and says which.
"""

from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING

import pygtrie

if TYPE_CHECKING:
    from collections.abc import Iterable


def resolved(path: pathlib.Path) -> pathlib.Path:
    """The resolved form of ``path``: ``os.path.realpath``, which resolves a missing path's
    existing ancestors."""
    return pathlib.Path(os.path.realpath(path))


def forms(path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Absolute normalized ``path`` and its resolved form (``resolved``), in that order."""
    return path, resolved(path)


def parts_trie[V](entries: Iterable[tuple[pathlib.Path, V]]) -> pygtrie.Trie[V]:
    """A trie of ``entries``, each path keyed by its ``Path.parts`` to its value; of entries
    with the same parts, the last one's value stays."""
    return pygtrie.Trie((path.parts, value) for path, value in entries)


def forms_trie(paths: Iterable[str]) -> pygtrie.Trie[str]:
    """A trie of the absolute normalized ``paths``, each keyed by the parts of both its forms
    (``forms``) to the path itself; where a form of one path is a form of an earlier one, the
    later path stays."""
    return parts_trie((form, path) for path in paths for form in forms(pathlib.Path(path)))


def enclosing[V](trie: pygtrie.Trie[V], *paths: pathlib.Path) -> V | None:
    """The value of the outermost stored path of ``trie`` that is or holds a path of ``paths``:
    of ``paths``, in their order, the first that a stored path is or holds decides. None for
    none."""
    for path in paths:
        step = trie.shortest_prefix(path.parts)
        if step is not None and step.is_set:
            return trie[step.key]
    return None


def held[V: (str, pathlib.Path)](trie: pygtrie.Trie[V], *paths: pathlib.Path) -> list[V]:
    """For each path of ``paths`` that holds a stored path of ``trie``, the values of the stored
    paths it is or holds; each value once, sorted in its type's order: a ``str`` by character, a
    ``pathlib.Path`` by its parts (``a/x`` before ``a-1`` as paths, after it as strings)."""
    return sorted(_held(trie, paths))


def overlapping[V: (str, pathlib.Path)](trie: pygtrie.Trie[V], *paths: pathlib.Path) -> V | None:
    """``enclosing`` of ``paths``, else the first value ``held`` gives for them, else None:
    "is or lies inside" is asked of every path of ``paths`` before "holds" is."""
    found = enclosing(trie, *paths)
    if found is not None:
        return found
    inside = _held(trie, paths)
    return min(inside) if inside else None


def _held[V](trie: pygtrie.Trie[V], paths: Iterable[pathlib.Path]) -> set[V]:
    """For each path of ``paths`` that holds a stored path of ``trie``, the values of the stored
    paths it is or holds."""
    return {
        value
        for path in paths
        if trie.has_subtrie(path.parts)
        for value in trie.values(prefix=path.parts)
    }
