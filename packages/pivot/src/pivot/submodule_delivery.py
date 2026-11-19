"""How a stage's worker gets each submodule the stage reads through the library package that holds
it (`fingerprint.read_submodules`): sent with the stage, or imported by name in the worker.

Code that reads `company.local` through the package `company` finds there only what something
bound. A plain run's `import company.local` binds it; in a worker, code that cloudpickle sends by
value runs none of its module's imports, and cloudpickle builds a module it sends by value as a new
module object, which no package holds. The coordinator decides each such submodule from what
loky's pickler, which pickles the worker's call, carries of it by value (`submodule_delivery`); the
worker binds the sent ones on their packages and imports the others by name
(`executor.worker._deliver_submodules`). Why, and the stated limits:
`docs/architecture/fingerprinting.md`, "Imports a Stage Makes While It Runs".
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, NamedTuple, final, override

import cloudpickle
import loky.backend.reduction

from pivot import fingerprint, loaded_source

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


class SubmoduleDelivery(NamedTuple):
    """How a stage's worker gets the submodules the stage reads through the library packages that
    hold them (`submodule_delivery`), each in the order of their names, a package before its
    submodules: the modules sent with the stage, which the worker binds on their packages
    (`sent`), and the names of those it imports by name (`imported`), before the stage runs."""

    sent: tuple[types.ModuleType, ...]
    imported: tuple[str, ...]


def submodule_delivery(value: object, names: Sequence[str]) -> SubmoduleDelivery:
    """How a stage's worker gets each of the submodules `names` (`fingerprint.read_submodules`, a
    package before its submodules) that its package in this process holds bound
    (`_bound_submodules`; the others reach no worker), from what loky's pickler carries of it by
    value when it pickles `value`, the part of the worker's call that holds the stage
    (`_carried`). Nothing is pickled when no such submodule is bound.

    - Its module object carried: the module is sent. The call pickles it anyway, so sending it
      adds nothing to what loky pickles, and sent, it is the module the worker binds on its
      package, so the values the call carries of it are its own (pickle's memo).
    - Only a function or class of it carried: the module is sent where pickling it alone
      succeeds, as above, and imported by name otherwise, as pivot 0.10.4 does; the function or
      class the stage holds and the module the worker imports are then two copies.
    - Neither: the module is imported by name, as a plain run imports it.

    A module sent for a function or class of it joins what the call carries, so what pickling it
    alone carries of the others is decided by the same rules (the namespace of a module whose
    object the call pickles is pickled with the call already), and each submodule is pickled
    alone at most once. When pickling `value` raises, every submodule is imported by name: the
    worker's call, which holds `value`, then fails to pickle with the same error, which fails the
    stage before a worker gets it."""
    candidates = _bound_submodules(names)
    if not candidates:
        return SubmoduleDelivery(sent=(), imported=())
    carried = _carried(value, candidates)
    if carried is None:
        return SubmoduleDelivery(sent=(), imported=tuple(candidates))
    sent = set(carried.modules)
    pending = [name for name in candidates if name in carried.members]
    tried = set[str]()
    while pending:
        name = pending.pop()
        if name in sent or name in tried:
            continue
        tried.add(name)
        alone = _carried(candidates[name], candidates)
        if alone is not None:
            sent.add(name)
            sent.update(alone.modules)
            pending.extend(other for other in candidates if other in alone.members)
    return SubmoduleDelivery(
        sent=tuple(module for name, module in candidates.items() if name in sent),
        imported=tuple(name for name in candidates if name not in sent),
    )


def _bound_submodules(names: Iterable[str]) -> dict[str, types.ModuleType]:
    """Each of the submodules `names` that its package in this process holds bound
    (`fingerprint.bound_submodule`), by name, in the order of `names`: not one an import session
    took out of `sys.modules` since, nor one no import bound."""
    bound = dict[str, types.ModuleType]()
    for name in names:
        package, _, attribute = name.rpartition(".")
        parent = sys.modules.get(package)
        if isinstance(parent, types.ModuleType):
            module = fingerprint.bound_submodule(parent, attribute)
            if module is not None:
                bound[name] = module
    return bound


class _Carried(NamedTuple):
    """What loky's pickler carries by value of the candidate submodules when it pickles one value
    (`_Recorder`), each by name: those whose module object it pickles (`modules`), and those of
    which it pickles a function or class (`members`)."""

    modules: frozenset[str]
    members: frozenset[str]


def _carried(value: object, candidates: Mapping[str, types.ModuleType]) -> _Carried | None:
    """What loky's pickler carries by value of the submodules `candidates` (each by name) when it
    pickles `value` (`_Recorder`), or None when that pickling raises."""
    recorder = _Recorder(candidates)
    try:
        recorder.dump(value)
    except Exception:  # noqa: BLE001 - `submodule_delivery` decides what a failure means
        return None
    return _Carried(modules=frozenset(recorder.modules), members=frozenset(recorder.members))


class _Discarded:
    """A binary file that keeps nothing written to it."""

    def write(self, data: bytes) -> int:
        return len(data)


# The pickler loky's call queue pickles a worker's call with, as `loky.backend.reduction` sets it
# when it is imported (pivot sets no other): cloudpickle's, with loky's own reducers, which reduce a
# bound method as `getattr(instance, name)`.
_LokyPickler = loky.backend.reduction.get_loky_pickler()


@final
class _Recorder(_LokyPickler):
    """loky's pickler (`_LokyPickler`), writing nothing (`_Discarded`), that records which of the
    submodules `candidates` (each by name) it pickles something of by value, as cloudpickle
    decides it for each object it pickles: in `modules`, each whose module object it pickles by
    value, which it does where the module or one of its packages is registered so
    (`loaded_source.pickled_by_value`; that is what cloudpickle's private
    `_should_pickle_by_reference` decides for a module that `sys.modules` holds, as it holds each
    candidate, `fingerprint.bound_submodule`); in `members`, each that the `__module__` of a
    function or class names that cloudpickle's `reducer_override` reduces, which it does for one
    it pickles by value. No private name of cloudpickle is called."""

    def __init__(self, candidates: Mapping[str, types.ModuleType]) -> None:
        super().__init__(_Discarded())
        self._names = frozenset(candidates)
        # Each module object by its identity, so an object is matched without reading it.
        self._modules = {id(module): name for name, module in candidates.items()}
        self._registry = cloudpickle.list_registry_pickle_by_value()
        self.modules = set[str]()
        self.members = set[str]()

    @override
    def reducer_override(self, obj: object) -> object:
        reduced = super().reducer_override(obj)
        if reduced is not NotImplemented:
            module = getattr(obj, "__module__", None)
            if isinstance(module, str) and module in self._names:
                self.members.add(module)
        elif (name := self._modules.get(id(obj))) is not None and loaded_source.pickled_by_value(
            name, self._registry
        ):
            self.modules.add(name)
        return reduced
