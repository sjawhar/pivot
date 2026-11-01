import ast
import atexit
import contextlib
import contextvars
import dataclasses
import enum
import functools
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import inspect
import json
import linecache
import locale
import logging
import marshal
import os
import pathlib
import re
import sys
import textwrap
import types
import typing
import weakref
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, cast

import xxhash

from pivot import ast_utils, exceptions, loaded_source, metrics

if TYPE_CHECKING:
    import pydantic.json_schema

    from pivot import loaders
    from pivot.storage.state import StateDB

_logger = logging.getLogger(__name__)

_PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
# Bump when the hashing scheme changes. Version 3 strips absolute source paths from
# marshalled code objects. Version 4 qualifies manifest keys by definition module. Version 5
# invalidates manifests that might have been persisted after unsafe capture warnings. Version 6
# also invalidates entries written before discovery imported pipeline.py as a real module.
# Version 7 invalidates manifests whose source maps omitted imported project modules.
# Version 8 adds default argument values, which earlier manifests did not record.
# Version 9 keys manifests by the stage function's identity as well as the stage name.
# Version 10 drops a bare @stage decorator from code hashes, as the called form already was.
# Version 11 walks the function behind a functools.cache or lru_cache helper.
# Version 12 hashes set members and dict entries in sorted order, not hash-seed order.
# Version 13 hashes a function value by its name and code instead of its address-bearing repr.
# Version 14 walks user functions and classes bound in functools.partial arguments.
# Version 15 hashes an object with the default repr by its class and attributes, not its address.
# Version 16 invalidates manifests hashed before StageParams forbade extra fields, which adds
# additionalProperties: false to a params schema; no project source edit would invalidate them.
# Version 17 records the live values a walk read, which a cache hit reads again.
# Version 18 keys manifests by the function's full identity (closure values and defaults
# included) and keys closure values under the binding the closure was reached through.
# Version 19 records each library function or class a stage reads by name (`ext:` entries).
# Version 20 hashes a secret in a default, loader config or partial by its keyed digest.
# Version 21 hashes a dataclass instance field by field, not by its generated repr.
# Version 22 tracks class attributes a stage reads (`Config.SCALE`) and the schema of a
# pydantic class read as a global.
# Version 23 tracks only class attributes a user class body wrote, not library-defined ones.
# Version 24 records `module.NAME` reads with the keys a global gets (`const:`, `func:`,
# `ext:`), not `mod:`, and follows `settings.Config.SCALE` and `pkg.sub.helper`.
# Version 25 records call-site artifact wiring (parameter bindings, collection kind/order,
# and project-relative input paths) in the code manifest.
# Version 26 records tuples and frozensets of scalars and compiled regexes by value.
# Version 27 keys cached function hashes and manifest source files by each file's content, not
# its mtime, size and inode, which an edit can keep.
# Version 28 treats pivot's own packages as library code outside site-packages too, so a stage
# calling into pivot_tui installed editable from its checkout no longer walks its source, and
# decides a namespace package by the rule that decides a regular module (`is_library_path`).
# Version 29 decides a namespace package by its site-packages, dist-packages and pivot-package
# directories, not by the interpreter's prefix, so a manifest that left out the user's part of a
# namespace shared with a checkout under the prefix is not reused.
# Version 30 follows every callable user class member, dataclass defaults, and body imports of
# user modules discovery loaded, and records the user modules each manifest covers.
# Version 31 hashes a loaded module's functions from the source it was compiled from, under that
# source's digest. Earlier entries hashed the file as `inspect` re-read it, which an edit after the
# digest was taken could change.
# Version 32 records each module a worker imports by name (`byname:` entries) and, in a cached
# manifest, where each such module resolves (`o`), which a hit re-checks.
# Version 33 fingerprints a pivot loader that a partial's arguments or a dataclass field default
# holds as a top-level loader is fingerprinted (`_held_loader_entries`, `held:` entries), and
# hashes a class with no source by what defines it, not by its address (`_sourceless_class_hash`).
_CACHE_SCHEMA_VERSION = 33
_REPR_SIZE_LIMIT = 10_000

_SCALAR_TYPES = (bool, int, float, str, bytes, type(None))

_SITE_PACKAGE_PATHS = ("site-packages", "dist-packages")

# Type alias for AST hash cache entry tuple.
# Format: (rel_path, source_digest, qualname, py_version, schema_version, hash_hex)
type AstHashEntry = tuple[str, str, str, str, int, str]


class _PydanticModelProtocol(Protocol):
    model_fields: Mapping[str, Any]

    @classmethod
    def model_json_schema(
        cls, *, schema_generator: "type[pydantic.json_schema.GenerateJsonSchema]"
    ) -> dict[str, Any]: ...


def _init_stdlib_paths() -> tuple[pathlib.Path, ...]:
    """Build resolved stdlib paths for symlink-safe comparison.

    Resolves sys.prefix and sys.base_prefix to handle environments where Python
    is installed via symlinks (e.g., Homebrew). Deduplicates when not in a venv.
    """
    paths = list[pathlib.Path]()
    for prefix in (sys.prefix, sys.base_prefix):
        try:
            resolved = pathlib.Path(prefix).resolve()
        except OSError:
            resolved = pathlib.Path(prefix)
        if resolved not in paths:
            paths.append(resolved)
    return tuple(paths)


_STDLIB_PATHS = _init_stdlib_paths()


def _pivot_tui_dirs() -> tuple[pathlib.Path, ...]:
    """Resolve pivot_tui's directories from what the installed pivot-tui distribution records:
    the site directory it is installed in, which counts on its own only for a directory not named
    site-packages (``pip install --target``), and each directory a path line of its ``.pth`` file
    adds to ``sys.path`` (an editable install). Never from looking the name up on ``sys.path``,
    where a user's own module or package named ``pivot_tui`` may come first."""
    try:
        tui = importlib.metadata.distribution("pivot-tui")
    except importlib.metadata.PackageNotFoundError:
        return ()
    site_dir = pathlib.Path(str(tui.locate_file("")))
    roots = [site_dir]
    for file in tui.files or ():
        if file.suffix != ".pth":
            continue
        # Read as site.addpackage reads it: a file it cannot read adds no path, the text is UTF-8
        # (a byte order mark dropped) or else in the locale's encoding, and comment, blank and
        # import lines add no path.
        try:
            content = file.read_binary()
        except OSError:
            continue
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode(locale.getencoding())
        lines = (line.rstrip() for line in text.splitlines())
        roots.extend(
            site_dir / line
            for line in lines
            if line and not line.startswith(("#", "import ", "import\t"))
        )
    return tuple((root / "pivot_tui").resolve() for root in roots if (root / "pivot_tui").is_dir())


# Pivot's own packages: pivot's directory wherever it is installed, and pivot_tui's. An editable
# install, as pivot's workspace makes, puts them in a source checkout outside site-packages.
_PIVOT_PACKAGE_DIRS = (pathlib.Path(__file__).resolve().parent, *_pivot_tui_dirs())

# Cache for hash_function_ast results using weak references.
# This avoids repeated AST parsing for the same function during fingerprinting
# while ensuring stale entries are automatically cleaned up when functions are GC'd.
# Note: WeakKeyDictionary is not thread-safe. Fingerprinting runs single-threaded
# per process (multiprocessing uses separate memory spaces), so this is safe.
_hash_function_ast_cache: weakref.WeakKeyDictionary[Callable[..., Any], str] = (
    weakref.WeakKeyDictionary()
)

# Cache for getclosurevars results. This is expensive (~0.5ms per call) and the same
# function may be visited multiple times during recursive fingerprinting.
_getclosurevars_cache: weakref.WeakKeyDictionary[Callable[..., Any], inspect.ClosureVars] = (
    weakref.WeakKeyDictionary()
)

# Cache for is_user_code results. Called 10K+ times for 125 stages, mostly for the same
# objects. Path resolution is expensive (~0.05ms per call).
_is_user_code_cache: weakref.WeakKeyDictionary[object, bool] = weakref.WeakKeyDictionary()

# Cache for get_type_hints results. Called ~10x per stage during fingerprinting.
_get_type_hints_cache: weakref.WeakKeyDictionary[Callable[..., Any], dict[str, Any]] = (
    weakref.WeakKeyDictionary()
)

# Cache for _schema_hash results, keyed by pydantic model class.
_schema_hash_cache: weakref.WeakKeyDictionary[type, str] = weakref.WeakKeyDictionary()

# Module-level state for persistent AST hash caching.
# Fingerprinting happens during discovery (single-threaded in coordinator process).
# Workers (via ProcessPoolExecutor) have their own memory space and don't share this
# state - any pending writes in workers are lost, which is fine since workers use
# readonly StateDB and fingerprinting should complete during discovery.
_state_db: "StateDB | None" = None
_state_db_init_attempted: bool = False
# The state directory `_state_db` belongs to: the producing directory of its last lookup.
_state_db_dir: pathlib.Path | None = None

# Pending cache writes, each filed under the state directory of the run that produced it
# (`_producing_state_dir`). A flush writes every entry to the directory it is filed under and
# resolves none, because a flush runs where its producer no longer does: the end of a CLI
# invocation, a TUI quit during fingerprinting, process exit for a library caller. A
# `core.state_dir` edited in between never takes a previous run's entries.
_pending_ast_writes = dict[pathlib.Path, list[AstHashEntry]]()
# Manifest cache values: (key_bytes, value_bytes) tuples.
_pending_manifest_writes = dict[pathlib.Path, list[tuple[bytes, bytes]]]()


# Source files visited during a single stage's fingerprinting.
# Maps rel_path -> content digest (`_source_digest`). Set by _collecting_sources().
_active_source_map: dict[str, str] | None = None

# Stage name used for error reporting during fingerprinting.
_current_stage_name: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_stage_name", default=None
)

# Unsafe capture errors collected while recursively walking one stage's dependencies.
_capture_errors: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "_capture_errors", default=None
)

# Unsafe captures are allowed only with an explicit warning and must never populate the
# persistent manifest cache.
_unsafe_capture_warning: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_unsafe_capture_warning", default=False
)

# The stage `stage_fingerprint_scope` is fingerprinting, told when a refusal is downgraded.
_stage_scope: contextvars.ContextVar["StageFingerprintScope | None"] = contextvars.ContextVar(
    "_stage_scope", default=None
)

# Live values read while computing a cacheable manifest: (path, token) per read. A path is the
# steps from the stage function to the object being visited, then the step that reads the value.
_recorded_reads: contextvars.ContextVar[list[tuple[tuple[str, ...], str]] | None] = (
    contextvars.ContextVar("_recorded_reads", default=None)
)

# Steps from the stage function to the object the walk is visiting.
_walk_path: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "_walk_path", default=()
)

# Manifest reference of the function being walked. Its closure cells and defaults are keyed
# under it, so two closures from one factory (`h2 = make(2)`, `h5 = make(5)`) keep apart.
_owner_ref: contextvars.ContextVar[str | None] = contextvars.ContextVar("_owner_ref", default=None)


class _HeldLoader(typing.NamedTuple):
    """A pivot loader a serialized value holds (`_serialize_collecting`)."""

    loader: Any
    text: str  # its canonical form as JSON
    classes: tuple[int, ...]  # the ids of its class and of the loaders' classes inside it
    path: tuple[list[str], ...]  # its position: a cell per step, filled once its container sorts


@dataclasses.dataclass(frozen=True, slots=True)
class _LoaderForm:
    """The canonical form of one pivot loader, kept for one stage (`_form_memo`)."""

    loader: Any  # held so its id is not reused while the memo lives
    canonical: Any
    text: str
    classes: tuple[int, ...]


# The pivot loaders `_canonicalize_value_for_hash` meets while it serializes a value whose held
# loaders are fingerprinted (`_serialize_collecting`); None outside one, and inside a held
# loader's own form, whose loaders are its own fingerprint's.
_held_loaders: contextvars.ContextVar[list[_HeldLoader] | None] = contextvars.ContextVar(
    "_held_loaders", default=None
)

# The position of the value being canonicalized inside the value `_serialize_collecting`
# serializes: the part's name, then a cell per container step. An ordered container's step is
# the item's index; a set's or a dict's is the item's rank in the sorted canonical form, written
# once the container sorts, so it does not follow the hash seed (equal forms share a rank).
_held_path: contextvars.ContextVar[tuple[list[str], ...]] = contextvars.ContextVar(
    "_held_path", default=()
)

# The ids of the classes of the pivot loaders met inside the loader being canonicalized, so two
# loaders with one form but different classes (two classes one factory made) keep apart in the
# walk memo (`_held_memo`); None outside a loader's form.
_form_classes: contextvars.ContextVar[list[int] | None] = contextvars.ContextVar(
    "_form_classes", default=None
)

# Whether the canonical form of the loader being canonicalized holds a `["cycle", N]`. Only a
# form with none is kept (`_form_memo`): a loader whose form holds no cycle holds nothing that
# holds it back, so its form is the same on any path it is met on; None outside a loader's form.
_form_cycle: contextvars.ContextVar[list[bool] | None] = contextvars.ContextVar(
    "_form_cycle", default=None
)

# The canonical forms of the pivot loaders met while one stage, or one top-level
# `get_loader_fingerprint` call, is fingerprinted, by the loader's id: a chain of held loaders is
# canonicalized once, not once per level (`_loader_memos`).
_form_memo: contextvars.ContextVar[dict[int, _LoaderForm] | None] = contextvars.ContextVar(
    "_form_memo", default=None
)

# The entries a loader's class decides (`_loader_class_entries`), by the class's id, with the
# class held so the id is not reused, for one stage or one top-level call (`_loader_memos`): many
# loaders of one class, held or not, walk the class once. Per stage, as `_held_memo`.
_loader_class_memo: contextvars.ContextVar[dict[int, tuple[type, dict[str, str]]] | None] = (
    contextvars.ContextVar("_loader_class_memo", default=None)
)

# The ids of the loaders `get_loader_fingerprint` is fingerprinting, outermost first: a loader that
# one of their walks reaches again (a cycle) is not fingerprinted again (`_held_loader_entries`).
_loader_stack: contextvars.ContextVar[tuple[int, ...]] = contextvars.ContextVar(
    "_loader_stack", default=()
)

# The manifests of the held loaders fingerprinted for one stage (`stage_fingerprint_scope`), by
# their canonical form and the ids of the classes in it, each with the loader it was walked for
# (held so those classes live), so a loader held many times is walked once. Kept per stage, not
# per pass: a held loader's walk reports its refusals with the stage that holds it.
_held_memo: contextvars.ContextVar[
    dict[tuple[str, tuple[int, ...]], tuple[Any, dict[str, str]]] | None
] = contextvars.ContextVar("_held_memo", default=None)


@contextlib.contextmanager
def _loader_memos() -> Iterator[None]:
    """Keep the loader memos (`_form_memo`, `_loader_class_memo`, `_held_memo`) for the block,
    unless an enclosing stage scope or call keeps them already."""
    form = _form_memo.set({}) if _form_memo.get() is None else None
    classes = _loader_class_memo.set({}) if _loader_class_memo.get() is None else None
    held = _held_memo.set({}) if _held_memo.get() is None else None
    try:
        yield
    finally:
        if held is not None:
            _held_memo.reset(held)
        if classes is not None:
            _loader_class_memo.reset(classes)
        if form is not None:
            _form_memo.reset(form)


# Per-pass memo, keyed by kind and object id, for function identities (`function_identity`),
# type-hint classes (`_hint_classes`), user-class members (`_user_methods`) and mark decisions
# (`_mark_decision`), and by file name for source file digests (`_source_digest`).
# Set only inside `fingerprint_pass()`; a reload, a mutated cell or an edit is seen by the next
# pass.
_pass_memo: contextvars.ContextVar[dict[tuple[str, int | str], Any] | None] = (
    contextvars.ContextVar("_pass_memo", default=None)
)


type _TypeAttributeProbe = tuple[bool, bool, Any, bool, bool]
# Per real type: the type (held so its id is not reused) and its facts per attribute name.
type _TypeAttributeMemo = dict[int, tuple[type[Any], dict[str, _TypeAttributeProbe]]]
# Per real type whose route its MRO alone decides: the type and the route (see `_type_memo`).
type _TypeDumpRoutes = dict[int, tuple[type[Any], _DumpRoute | None]]

# The real-MRO facts of the types met while the memo lives, and the canonical form's
# `model_dump` route for those whose MRO alone decides it: `ASK_KEPT` for a type whose MRO both
# defines `__getattr__` and keeps `model_dump` (as a pydantic model), None for one with no
# class-side hook for it (no `__getattr__`, no class-level `model_dump`, no `__getattribute__`
# or `__class__` but `object`'s), whose instances main's C `hasattr` then asks, reading only the
# real instance dict. They are separate from `_pass_memo` so the hot probe does not allocate a
# `(kind, id)` tuple or call its generic wrapper for each value. Keyed by id with the type held,
# never hashed: a metaclass `__hash__` or `__eq__` never runs, and an id is not reused while the
# memo lives. It lives for one `fingerprint_pass()`, or else for one stage's fingerprint
# (`_type_facts_scope`).
_type_memo: contextvars.ContextVar[tuple[_TypeAttributeMemo, _TypeDumpRoutes] | None] = (
    contextvars.ContextVar("_type_memo", default=None)
)


@contextlib.contextmanager
def _type_facts_scope() -> Iterator[None]:
    """Hold the real-MRO facts of the types met inside, unless an enclosing scope already does.

    `fingerprint_pass()` holds them for a pass over many stages; a stage's fingerprint, a cache
    hit's re-read included, holds them for itself, so a value repeated across a stage's bound
    arguments pays the walk once. The facts are not kept past the scope, so a class changed
    later is read again.
    """
    if _type_memo.get() is not None:
        yield
        return
    token = _type_memo.set(({}, {}))
    try:
        yield
    finally:
        _type_memo.reset(token)


@contextlib.contextmanager
def fingerprint_pass() -> Iterator[None]:
    """Share function identities, type-hint classes, user-class members, mark decisions, type
    MRO facts and source digests across one pass over many stages.

    A cache hit re-reads every recorded value, and stages of one pipeline read the same
    helpers, model classes and source files: inside the pass each is named or read once. The
    memo holds strong references, so an id is never reused within the pass, and it ends with
    the pass.
    """
    if _pass_memo.get() is not None:
        yield
        return
    token = _pass_memo.set({})
    # The pass holds the type facts itself (as `_type_facts_scope` would), without a second
    # context manager per pass.
    type_token = _type_memo.set(({}, {}))
    try:
        yield
    finally:
        _type_memo.reset(type_token)
        _pass_memo.reset(token)


def _memoized[T](kind: str, obj: object, compute: Callable[[], T]) -> T:
    """Return `compute()`, reusing the value for `obj` within one `fingerprint_pass()`."""
    memo = _pass_memo.get()
    if memo is None:
        return compute()
    key = (kind, id(obj))
    if key not in memo:
        memo[key] = (obj, compute())
    return cast("T", memo[key][1])


class FunctionIdentity(TypedDict):
    """What makes two function objects the same function (`function_identity`)."""

    module: str
    qualname: str
    file: str | None
    code: str | None
    cells: list[tuple[str, Any]]
    defaults: list[tuple[str, Any]]


def _producing_state_dir() -> pathlib.Path | None:
    """The state directory a cache entry produced now belongs to, or None when it has none.

    The rule for every cache here: a pending entry belongs to the state directory of the call
    that produced it, and a flush writes it there and never resolves a directory again.

    Read where the entry is produced, never where it is written. A CLI invocation resolves its
    state directory once and every producer of that invocation, the engine thread of a TUI
    session included, reads that one. A library caller has no invocation, so each library entry
    point that produces entries holds a record of its own (``state_dir.entry_invocation``):
    an engine run, discovery, a pipeline's load, each ``Pipeline.ensure_fingerprint`` and each
    ``get_stage_fingerprint_cached`` call. A pipeline's fingerprints and runs start from the
    resolution it made when it loaded, an unusable value included, so a ``core.state_dir``
    edited afterwards, by a stage or by the caller, moves none of its entries, and the record
    answers every entry without reading the configuration. Only ``get_stage_fingerprint``
    called by hand, outside every entry point, reads and resolves the configuration again for
    each lookup: correct, and slow. An unreadable or unusable value (other than ``''``, which
    falls back to ``.pivot``) leaves the entry unfiled, and the caches are best effort, so it
    is dropped; for a path that cannot become a directory the entry goes where the record
    decided once (``state_dir.locate_state_dir``): ``.pivot``, or the path, whose flush fails
    and drops the entry while it still cannot be opened. Nothing is created here: the StateDB
    that reads or writes the entries creates the directory (``state_dir.make_state_dir``).
    """
    from pivot import project, state_dir

    try:
        return state_dir.locate_state_dir(project.get_project_root())
    except Exception:
        # ConfigError (unusable value), ProjectNotInitializedError, UnicodeDecodeError, etc.
        return None


def _close_state_db() -> None:
    """Close the readonly StateDB; the next lookup opens it again."""
    global _state_db, _state_db_init_attempted, _state_db_dir
    if _state_db is not None:
        _state_db.close()
        _state_db = None
    _state_db_init_attempted = False
    _state_db_dir = None


def flush_and_close() -> None:
    """Write the pending cache entries, then close the readonly StateDB.

    The CLI calls this as each invocation ends, process exit calls it for a library caller,
    and either can run after the run that produced the entries. Each entry goes to the state
    directory it was filed under, so neither resolves one here. Entries that fail to write
    are dropped: the caches are best effort. The flush opens its own writable StateDB.
    """
    flush_ast_hash_cache()
    flush_manifest_cache()
    _pending_ast_writes.clear()
    _pending_manifest_writes.clear()
    _close_state_db()


atexit.register(flush_and_close)


def _get_state_db() -> "StateDB | None":
    """Get the readonly StateDB of the state directory entries are produced for now.

    Lookups read the directory where a miss files its new entry (``_producing_state_dir``).
    When that directory changes, as for a pipeline loaded after an edit of ``core.state_dir``,
    the handle is closed and the new directory's opened, so no handle outlives the resolution
    it was opened for. One that fails to open is not tried again until the directory changes
    or the caches flush (graceful degradation).

    Not thread-safe, but fingerprinting runs single-threaded per process
    (see comment at line 33). Workers use separate memory spaces.
    """
    global _state_db, _state_db_init_attempted, _state_db_dir
    state_dir = _producing_state_dir()
    if state_dir != _state_db_dir:
        _close_state_db()
        _state_db_dir = state_dir
    if _state_db is not None:
        return _state_db
    if _state_db_init_attempted or state_dir is None:
        return None  # Already tried and failed, or no directory to read
    _state_db_init_attempted = True
    try:
        from pivot.storage import state

        _state_db = state.StateDB(state_dir, readonly=True)
        return _state_db
    except Exception:
        # OSError (filesystem), ImportError (module), lmdb.Error, etc.
        return None


def _make_manifest_cache_key(stage_name: str, func: Callable[..., Any]) -> bytes:
    """Build StateDB key for the manifest of `func` registered as `stage_name`.

    The key names the function by its identity (`function_identity`) as well as the stage,
    because two pipeline files can register the same stage name with different functions, and
    one factory can build stages that differ only in closure values. A cached manifest for one
    never answers for the other.
    """
    identity = function_identity(func)
    digest = xxhash.xxh64(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    names = f"{identity['module']}\x00{identity['qualname']}\x00{digest}"
    return f"sm:{stage_name}\x00{names}\x00{_PYTHON_VERSION}\x00{_CACHE_SCHEMA_VERSION}".encode()


def function_identity(func: Callable[..., Any]) -> FunctionIdentity:
    """Name one function for every check that asks whether two function objects are the same.

    Module, qualname, source file (relative to the project root) and code hash say where the
    function is defined and what it does. Closure cell values and default values say what it
    was built with: `make_stage(2)` and `make_stage(5)` share a definition and differ only in
    a cell. Values take their canonical form (`_canonicalize_value_for_hash`); an
    ArtifactHandle default is a wired input and is named only as one. Library code is named by
    module and qualname only.

    Used by the stage-manifest cache key, the canonical form of a function value (loader
    config, partial arguments), and compose's check that one stage name is one stage. Within
    one `fingerprint_pass()` each function is named once.
    """
    return _memoized("identity", func, lambda: _function_identity(func, ()))


def _function_identity(func: Callable[..., Any], active: tuple[int, ...]) -> FunctionIdentity:
    """`function_identity`, with `active` holding the functions already being named."""
    file = _project_file(func)
    identity = FunctionIdentity(
        module=getattr(func, "__module__", None) or "<unknown>",
        qualname=_get_qualname_for_cache(func),
        file=file,
        code=None,
        cells=[],
        defaults=[],
    )
    if not is_user_code(func):
        # An install path outside the project differs from machine to machine, and a function
        # the walk records by name is named by its module and qualified name only.
        if file is not None and (pathlib.PurePath(file).is_absolute() or _recorded_by_name(func)):
            identity["file"] = None
        return identity
    if id(func) in active:
        return identity
    active = (*active, id(func))
    identity["code"] = hash_function_ast(func)
    code = getattr(func, "__code__", None)
    closure: tuple[types.CellType, ...] = getattr(func, "__closure__", None) or ()
    if isinstance(code, types.CodeType):
        identity["cells"] = [
            (name, _canonical_binding(_cell_value(cell), active))
            for name, cell in zip(code.co_freevars, closure, strict=True)
        ]
    identity["defaults"] = [
        (parameter, _canonical_binding(value, active))
        for parameter, value in _function_defaults(func)
    ]
    return identity


_EMPTY_CELL = object()


def _cell_value(cell: types.CellType) -> Any:
    """Return a closure cell's value, or `_EMPTY_CELL` for a name not yet bound."""
    try:
        return cell.cell_contents
    except ValueError:
        return _EMPTY_CELL


def _canonical_binding(value: Any, active: tuple[int, ...]) -> Any:
    """Canonical form of a closure cell or default value in a function identity."""
    from pivot import compose

    if value is _EMPTY_CELL:
        return ["empty"]
    if _is_artifact_handles(value, compose.ArtifactHandle):
        return ["handle"]
    return _canonicalize_value_for_hash(value, active)


def _project_file(func: Callable[..., Any]) -> str | None:
    """Return the file defining `func`, relative to the project root when it is inside it."""
    from pivot import project

    code = getattr(func, "__code__", None)
    if not isinstance(code, types.CodeType):
        return None
    try:
        root: pathlib.Path | None = project.get_project_root()
    except exceptions.ProjectNotInitializedError:
        root = None
    return _portable_path(code.co_filename, root)


@functools.lru_cache(maxsize=4096)
def _portable_path(filename: str, root: pathlib.Path | None) -> str:
    """Return `filename` relative to `root` when it is inside it, else unchanged."""
    try:
        resolved = pathlib.Path(filename).resolve()
    except OSError:
        return filename
    if root is not None and resolved.is_relative_to(root):
        return str(resolved.relative_to(root))
    return filename


def _source_digest(file: str) -> str:
    """Digest of source file ``file``'s content, which the function-hash and manifest caches
    record for it.

    For a file this process loaded a module from, that is the content it was compiled from
    (``loaded_source``), which ``loaded_source.getsource`` also reads: a stage's worker runs that
    code, whatever the file holds now. Any other file is read now.

    An edit can keep a file's mtime, size and inode (two writes inside one kernel timestamp
    tick, or a tool that restores the mtime), so the caches compare content. Within one
    `fingerprint_pass()` each file is read once. The read also drops the lines `linecache`
    holds for ``file``, which `inspect.getsource` checks only by that mtime and size, so a
    function hash computed from them is of the content it is cached under.

    Raises OSError when the file cannot be read.
    """
    if (loaded := loaded_source.digest(file)) is not None:
        return loaded
    memo = _pass_memo.get()
    key = ("source", file)
    if memo is not None and key in memo:
        return cast("str", memo[key])
    with open(file, "rb") as source:
        digest = xxhash.xxh64(source.read()).hexdigest()
    linecache.cache.pop(file, None)
    if memo is not None:
        memo[key] = digest
    return digest


def _get_func_source_info(func: Callable[..., Any]) -> tuple[str, str] | None:
    """Get (rel_path, content digest) for function source file.

    Returns None if source info unavailable (builtins, exec'd code, outside project).
    """
    try:
        file = inspect.getsourcefile(func)
        if file is None:
            return None
        path = pathlib.Path(file).resolve()
        if not path.is_file():
            return None  # exec'd code ("<string>") or a deleted file
        from pivot import project

        project_root = project.get_project_root()
        rel_path = str(path.relative_to(project_root))
        return (rel_path, _source_digest(file))
    except (TypeError, OSError, ValueError):
        # TypeError: builtins, exec'd
        # OSError: file unreadable
        # ValueError: path outside project root
        return None


@contextlib.contextmanager
def _collecting_sources() -> Iterator[dict[str, str]]:
    """Scope a source map for the duration of a fingerprint walk.

    Saves and restores any previously active source map, so nested calls
    (if they ever occur) don't clobber the outer collector.
    """
    global _active_source_map
    previous = _active_source_map
    source_map = dict[str, str]()
    _active_source_map = source_map
    try:
        yield source_map
    finally:
        _active_source_map = previous


# Memo of the last sys.modules scan: (loaded modules, project root, project module files).
# Every read compares it against the current sys.modules values by identity, so it is never
# served stale. A set captured once per pass and passed explicitly would miss modules imported
# after capture (fingerprints are computed lazily from several entry points).
_project_modules_snapshot: (
    tuple[list[object], pathlib.Path, list[tuple[str, pathlib.Path]]] | None
) = None


def _project_module_files(project_root: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    """Return (relative path, path) for every loaded project module.

    Classifying the ~2,000 loaded modules is reused until a module is imported, removed or
    replaced, so a fingerprint pass scans sys.modules once rather than once per stage.
    """
    global _project_modules_snapshot
    loaded = list[object](sys.modules.values())
    snapshot = _project_modules_snapshot
    if (
        snapshot is not None
        and snapshot[1] == project_root
        and len(snapshot[0]) == len(loaded)
        and all(previous is current for previous, current in zip(snapshot[0], loaded, strict=True))
    ):
        return snapshot[2]
    files = list[tuple[str, pathlib.Path]]()
    for module in loaded:
        # Some libraries install non-module objects in sys.modules.
        if not isinstance(module, types.ModuleType):
            continue
        file = module.__dict__.get("__file__")
        if not isinstance(file, str) or not is_user_code(module):
            continue
        try:
            path = pathlib.Path(file).resolve()
            rel_path = str(path.relative_to(project_root))
        except (OSError, ValueError):
            continue
        files.append((rel_path, path))
    _project_modules_snapshot = (loaded, project_root, files)
    return files


def _record_project_modules(source_map: dict[str, str]) -> None:
    """Add the file of every loaded project module to a manifest's source map.

    A manifest records values bound in module namespaces, and import-time code in any project
    module can change them without touching the walked functions' files: `from settings import
    FACTOR`, a module rebinding an imported helper, or one module assigning another's
    attribute. A cached manifest is therefore valid only while none of those files changed.
    """
    from pivot import project

    for rel_path, path in _project_module_files(project.get_project_root()):
        if rel_path in source_map:
            continue
        try:
            source_map[rel_path] = _source_digest(str(path))
        except OSError:
            continue


def _try_manifest_cache_hit(stage_name: str, func: Callable[..., Any]) -> dict[str, str] | None:
    """Try to load a cached manifest; returns None on miss.

    A hit needs every recorded source file unchanged, every live value the walk read to
    give the same token again (`_recorded_reads_hold`), and every module the manifest names by
    name to resolve where it did (`_by_name_origins`). Values computed at import from the
    environment or a data file change without touching any source file, and a project file can
    take over a name a library held (vendoring `helpers.py` where `helpers` was installed)
    without changing any file the walk read.
    """
    db = _get_state_db()
    if db is None:
        return None

    key = _make_manifest_cache_key(stage_name, func)
    try:
        raw = db.get_raw(key)
    except Exception:
        return None
    if raw is None:
        return None

    try:
        data: object = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    if not isinstance(data, dict):
        return None
    # Cast to dict[str, Any] after isinstance validation - json.loads returns dict[str, Any] for objects
    typed_data = cast("dict[str, Any]", data)
    manifest_raw: object = typed_data.get("m")
    sources_raw: object = typed_data.get("s")
    reads_raw: object = typed_data.get("r")
    origins_raw: object = typed_data.get("o")
    if (
        not isinstance(manifest_raw, dict)
        or not isinstance(sources_raw, dict)
        or not isinstance(reads_raw, list)
        or not isinstance(origins_raw, dict)
    ):
        return None

    # Narrow types after validation: manifest is {str: str}, sources is {str: digest}
    manifest = cast("dict[str, str]", manifest_raw)
    sources = cast("dict[str, object]", sources_raw)

    # If no source files were tracked, we have nothing to validate against.
    # This happens for stages with only builtins/exec'd code. Force recompute
    # rather than returning a stale manifest that can never be invalidated.
    if not sources:
        return None

    # Check every source file's content
    from pivot import project

    project_root = project.get_project_root()
    for rel_path, cached_digest in sources.items():
        if not isinstance(cached_digest, str):
            return None  # Corrupted cache entry
        # Guard against path traversal from corrupted/malicious cache entries
        if rel_path.startswith("/") or ".." in pathlib.Path(rel_path).parts:
            return None
        try:
            digest = _source_digest(str(project_root / rel_path))
        except OSError:
            return None  # File deleted or inaccessible
        if digest != cached_digest:
            return None  # File changed

    if not _recorded_reads_hold(func, cast("list[object]", reads_raw)):
        return None
    if _by_name_origins(manifest) != origins_raw:
        return None
    return manifest


def _by_name_origins(manifest: Mapping[str, str]) -> dict[str, list[str] | None]:
    """Where a worker would import each module `manifest`, the walk's, names by name
    (`by_name_modules`; the walk records no loader's entries), predicted without importing it
    (`loaded_source.worker_origin`), in the JSON form a cached manifest records."""
    origins = dict[str, list[str] | None]()
    for name in sorted(by_name_modules(manifest, ())):
        origin = loaded_source.worker_origin(name)
        origins[name] = None if origin is None else list(origin)
    return origins


class _ValueKind(enum.Enum):
    """How the walk treats a value read through a global, a cell, a default or `module.NAME`."""

    PARTIAL = "partial"
    REFUSED = "refused"
    CLASS_ABOVE_MARK = "class_above_mark"
    USER = "user"
    EXTERNAL = "external"
    MODULE = "module"
    LOGGER = "logger"
    SCALAR = "scalar"


def _immutable_value_text(value: Any) -> str | None:
    """Return the manifest text for a safely trackable immutable value, if any.

    Scalars keep their existing repr. Exact tuples and frozensets are tracked only when every
    member is an exact scalar type, one level deep. Compiled regexes record their full pattern
    and flags because their repr truncates long patterns.
    """
    if isinstance(value, _SCALAR_TYPES):
        return repr(value)
    if type(value) in (tuple, frozenset) and all(
        type(item) in _SCALAR_TYPES for item in cast("tuple[Any, ...] | frozenset[Any]", value)
    ):
        return _serialize_value_for_hash(value)
    if isinstance(value, re.Pattern):
        return _serialize_value_for_hash(value)
    return None


def _value_kind(value: Any) -> _ValueKind:
    """Classify a value a walked function reads; the walk and a cache hit share this.

    A value on a `__wrapped__` chain that holds a `@fingerprint_by_name` mark takes the mark
    decision (`_mark_decision`) first: a user class instance above the mark is
    CLASS_ABOVE_MARK, and an object the walk records by name is EXTERNAL, also when it is a
    marked instance of a class-based decorator.
    """
    mark = _mark_decision(value)
    if mark is _MarkDecision.REFUSED:
        return _ValueKind.CLASS_ABOVE_MARK
    # functools.partial and partialmethod fail is_user_code() (module is functools/stdlib).
    if isinstance(value, (functools.partial, functools.partialmethod)):
        return _ValueKind.PARTIAL
    if mark is _MarkDecision.BY_NAME and not _carries_instance(value):
        return _ValueKind.EXTERNAL
    if _is_callable_instance(value) or _is_user_class_instance(value) or _carries_instance(value):
        return _ValueKind.REFUSED
    if callable(value):
        return _ValueKind.USER if is_user_code(value) else _ValueKind.EXTERNAL
    if isinstance(value, types.ModuleType):
        return _ValueKind.MODULE
    if isinstance(value, logging.Logger):
        return _ValueKind.LOGGER
    if _immutable_value_text(value) is not None:
        return _ValueKind.SCALAR
    return _ValueKind.REFUSED


def _value_token(value: Any) -> str:
    """Summarise the walk's own entries for a read value; what it reads has its own steps.

    A user function or class is named by its identity, closure values and defaults included,
    so a binding that now holds another closure from the same factory reads as changed even
    where the walk did not descend into it again.
    """
    kind = _value_kind(value)
    if kind is _ValueKind.PARTIAL:
        return f"partial:{_serialize_value_for_hash(value)}"
    if kind is _ValueKind.USER:
        prefix = "class" if isinstance(value, type) else "function"
        identity = json.dumps(function_identity(value), sort_keys=True)
        return f"{prefix}:{xxhash.xxh64(identity.encode()).hexdigest()}"
    if kind is _ValueKind.MODULE:
        return f"module:{value.__name__}"
    if kind is _ValueKind.SCALAR:
        return f"{type(value).__name__}:{_immutable_value_text(value)}"
    if kind is _ValueKind.EXTERNAL:
        return f"external:{_library_name(value)}"
    return kind.value


def _default_token(value: Any) -> str:
    """Like `_value_token`, for a default argument (see `_process_default_values`)."""
    from pivot import compose, stage_def

    if isinstance(value, stage_def.StageParams):
        return f"params:{_serialize_value_for_hash(value)}"
    if _is_artifact_handles(value, compose.ArtifactHandle):
        return "handles"
    return _value_token(value)


# A walk step names one move from a visited object to a value it reads: `g:NAME` a global
# (also recorded when absent), `c:NAME` a closure cell, `d:PARAM` a default, `a:ATTR` a module
# attribute, `ca:ATTR` a class attribute (`Config.SCALE`), `n:NAME` and `np:DOTTED` names in a
# class's module, `m:NAME` a callable user class member, `fd:NAME` a dataclass field default,
# `bi:MODULE` a loaded module a function body imports, `u` the function behind a C-level
# wrapper, `pf`, `pa:INDEX` and `pk:KEY` a partial's parts, `i:INDEX`, `k:REPR` and `s:INDEX`
# container members, `hc:INDEX` a class in type hints and `fc:INDEX` a class in a model's field
# annotations. `hints` and `schema` read the visited object itself, and so do `hl` and `hd`: the
# fingerprints of the pivot loaders a partial's arguments (`hl`) or a dataclass field default
# (`hd`) holds (`_held_loaders_token`).
_UNREAD = object()


def _read_step(owner: Any, step: str) -> Any:
    """Return the value `step` reaches from `owner`, or `_UNREAD` when nothing is there."""
    kind, _, arg = step.partition(":")
    try:
        match kind:
            case "g":
                return owner.__globals__[arg]
            case "c":
                return owner.__closure__[owner.__code__.co_freevars.index(arg)].cell_contents
            case "d":
                return dict(_function_defaults(owner))[arg]
            case "a" | "ca":
                return getattr(owner, arg)
            case "n":
                return vars(sys.modules[owner.__module__])[arg]
            case "np":
                namespace = vars(sys.modules[owner.__module__])
                return _resolve_dotted_path(tuple(arg.split(".")), namespace)
            case "m":
                return dict(_user_methods(owner)).get(arg, _UNREAD)
            case "fd":
                return _dataclass_field_value(owner, arg)
            case "bi":
                module = _body_import_module(arg)
                return _UNREAD if module is None else module
            case "u":
                return unwrap(owner, stop=_stop_at_mark)
            case "pf":
                return owner.func
            case "pa":
                return owner.args[int(arg)]
            case "pk":
                return owner.keywords[arg]
            case "i":
                return owner[int(arg)]
            case "k":
                return next(item for key, item in owner.items() if repr(key) == arg)
            case "s":
                return sorted(owner, key=_serialize_value_for_hash)[int(arg)]
            case "hc":
                return _hint_classes(owner)[int(arg)]
            case "fc":
                return _model_field_classes(owner)[int(arg)]
            case _:
                return _UNREAD
    except (
        AttributeError,
        KeyError,
        IndexError,
        ValueError,
        TypeError,
        StopIteration,
        ImportError,
    ):
        return _UNREAD


def _read_token(owner: Any, step: str) -> str:
    """Return the token for what `step` reads from `owner` now.

    Equal tokens mean the walk makes the same decision there and adds the same entries for
    the value itself.
    """
    if step == "hints":
        return "|".join(_value_token(cls) for cls in _hint_classes(owner))
    if step == "schema":
        if not hasattr(owner, "model_json_schema"):
            return "absent"
        field_classes = "|".join(_value_token(cls) for cls in _model_field_classes(owner))
        return f"{_schema_hash(owner)}|{field_classes}"
    if step.startswith("fd:"):
        return _dataclass_field_token(owner, step.partition(":")[2])
    if step in ("hl", "hd"):
        return _held_loaders_token(owner, step)
    value = _read_step(owner, step)
    if value is _UNREAD:
        return "absent"
    if step.startswith("d:"):
        return _default_token(value)
    if step.startswith("ca:") and _is_user_enum_member(value):
        return f"enum:{_enum_member_text(value)}"
    return _value_token(value)


def _record_read(owner: Any, step: str) -> None:
    """Record what `owner`'s `step` reads, while a cacheable manifest is being computed."""
    reads = _recorded_reads.get()
    if reads is not None:
        reads.append(((*_walk_path.get(), step), _read_token(owner, step)))


@contextlib.contextmanager
def _walking(step: str) -> Iterator[None]:
    """Visit the value that `step` reads from the object being visited."""
    token = _walk_path.set((*_walk_path.get(), step))
    try:
        yield
    finally:
        _walk_path.reset(token)


def _recorded_reads_hold(func: Callable[..., Any], reads: list[object]) -> bool:
    """Return whether every read recorded from `func` gives its recorded token again."""
    owners: dict[tuple[str, ...], Any] = {(): func}
    for entry in reads:
        if not isinstance(entry, list) or len(cast("list[object]", entry)) != 2:
            return False
        path, token = cast("list[object]", entry)
        if not isinstance(path, list) or not path or not isinstance(token, str):
            return False
        steps = cast("list[object]", path)
        if not all(isinstance(step, str) for step in steps):
            return False
        names = cast("list[str]", steps)
        owner = _walk_to(tuple(names[:-1]), owners)
        if owner is _UNREAD or _read_token(owner, names[-1]) != token:
            return False
    return True


def _walk_to(path: tuple[str, ...], owners: dict[tuple[str, ...], Any]) -> Any:
    """Follow `path` from the stage function, reusing the objects already reached."""
    if path not in owners:
        parent = _walk_to(path[:-1], owners)
        owners[path] = parent if parent is _UNREAD else _read_step(parent, path[-1])
    return owners[path]


def flush_ast_hash_cache() -> None:
    """Write each pending AST hash to the state directory it is filed under, best-effort.

    A write that cannot happen at once (``state.save_best_effort``: another process holds the
    write lock past its short wait, or the database cannot be opened or written) keeps its
    entries pending under the same directory, so a later flush retries them instead of holding
    a read command behind a pivot write.
    """
    global _pending_ast_writes
    if not _pending_ast_writes:
        return

    pending = _pending_ast_writes
    _pending_ast_writes = dict[pathlib.Path, list[AstHashEntry]]()

    for state_dir, entries in pending.items():
        saved = False
        try:
            from pivot.storage import state

            saved = state.save_best_effort(
                state_dir,
                lambda db, entries=entries: db.save_ast_hash_many(entries),
                what=f"{len(entries)} AST hashes",
            )
        except Exception:
            _logger.debug(
                "Failed to flush AST hash cache (%d entries)", len(entries), exc_info=True
            )
        if saved:
            metrics.count("fingerprint.ast_hash_cache.flush")
        else:
            _pending_ast_writes.setdefault(state_dir, list[AstHashEntry]()).extend(entries)


def flush_manifest_cache() -> None:
    """Write each pending manifest to the state directory it is filed under, best-effort.

    A skipped write keeps its entries pending under the same directory, as
    ``flush_ast_hash_cache`` does.
    """
    global _pending_manifest_writes
    if not _pending_manifest_writes:
        return

    pending = _pending_manifest_writes
    _pending_manifest_writes = dict[pathlib.Path, list[tuple[bytes, bytes]]]()

    for state_dir, entries in pending.items():
        saved = False
        try:
            from pivot.storage import state

            saved = state.save_best_effort(
                state_dir,
                lambda db, entries=entries: db.put_raw_many(entries),
                what=f"{len(entries)} manifest entries",
            )
        except Exception:
            _logger.debug(
                "Failed to flush manifest cache (%d entries)", len(entries), exc_info=True
            )
        if saved:
            metrics.count("fingerprint.manifest_cache.flush")
        else:
            _pending_manifest_writes.setdefault(state_dir, list[tuple[bytes, bytes]]()).extend(
                entries
            )


def _normalize_changed_paths(
    paths: Sequence[str | os.PathLike[str] | pathlib.Path],
) -> set[str]:
    """Normalize changed paths to project-relative strings."""
    from pivot import project

    project_root = project.get_project_root()
    normalized = set[str]()
    for path in paths:
        candidate = pathlib.Path(path)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        with contextlib.suppress(OSError):
            candidate = candidate.resolve()
        try:
            rel_path = candidate.relative_to(project_root)
        except ValueError:
            continue
        normalized.add(str(rel_path))
    return normalized


def _manifest_references_paths(raw: bytes, changed_paths: set[str]) -> bool:
    """Check if a cached manifest references any changed path."""
    try:
        data: object = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    typed_data = cast("dict[str, Any]", data)
    sources_raw: object = typed_data.get("s")
    if not isinstance(sources_raw, dict):
        return False
    sources = cast("dict[str, object]", sources_raw)
    return any(source_path in changed_paths for source_path in sources)


def invalidate_manifests_for_paths(
    paths: Sequence[str | os.PathLike[str] | pathlib.Path],
) -> None:
    """Invalidate cached manifests that reference changed source files.

    A reload runs this where the edited pipeline will run, so the stored manifests of that
    state directory are the ones scanned. A pending manifest that reads a changed file is
    dropped wherever it is filed, because it must reach no state directory.
    """
    if not paths:
        return
    changed_paths = _normalize_changed_paths(paths)
    if not changed_paths:
        return

    state_dir = _producing_state_dir()
    if state_dir is None:
        return

    from pivot.storage import state

    keys_to_delete = set[bytes]()
    try:
        with state.StateDB(state_dir, readonly=True) as db:
            for key, raw in db.iter_prefix(b"sm:"):
                if _manifest_references_paths(raw, changed_paths):
                    keys_to_delete.add(key)
    except Exception:
        _logger.debug("Failed to scan manifest cache for invalidation", exc_info=True)

    for pending_dir, pending in _pending_manifest_writes.items():
        stale = {key for key, raw in pending if _manifest_references_paths(raw, changed_paths)}
        if not stale:
            continue
        _pending_manifest_writes[pending_dir] = [
            (key, value) for key, value in pending if key not in stale
        ]
        if pending_dir == state_dir:
            keys_to_delete.update(stale)

    if not keys_to_delete:
        return

    try:
        with state.StateDB(state_dir, readonly=False) as db:
            db.delete_raw_many(list(keys_to_delete))
    except Exception:
        _logger.debug(
            "Failed to invalidate manifest cache entries (%d)",
            len(keys_to_delete),
            exc_info=True,
        )


def get_stage_fingerprint(
    func: Callable[..., Any], visited: set[int] | None = None
) -> dict[str, str]:
    """Generate fingerprint manifest capturing all code dependencies.

    Returns dict with keys:
    - 'self:<name>': Function itself (hash)
    - 'func:<module>.<qualname>': Referenced helper functions (hash, transitive)
    - 'class:<module>.<qualname>': Referenced class definitions (hash, transitive)
    - 'const:<module>.<name>': Primitive constants, read as globals or as `module.NAME`
    - 'ext:<module>.<name>': Library functions and classes, named by definition
    """
    if visited is None:
        visited = set()

    _t = metrics.start()
    try:
        with _collected_capture_errors(), _type_facts_scope():
            return _get_stage_fingerprint_impl(func, visited)
    finally:
        metrics.end("fingerprint.get_stage_fingerprint", _t)


@contextlib.contextmanager
def _collected_capture_errors() -> Iterator[None]:
    """Collect unsafe-capture errors and raise them together, unless a caller already does.

    Each walked function reports what it reads, so one variable read by a loader's `load()`
    and `save()`, or by one loader on two inputs, produces the same message several times.
    Each distinct message is listed once, in first-seen order.
    """
    if _capture_errors.get() is not None:
        yield
        return
    errors = list[str]()
    token = _capture_errors.set(errors)
    try:
        yield
    finally:
        _capture_errors.reset(token)
    if errors:
        raise exceptions.StageDefinitionError("\n\n".join(dict.fromkeys(errors)))


def refuse(message: str) -> None:
    """Refuse the stage being fingerprinted: inside `stage_fingerprint_scope` the refusal is
    reported with the stage's other refusals when the scope ends, else it is raised now. Unsafe
    fingerprinting does not downgrade it (`loaded_source.check_worker_imports`)."""
    errors = _capture_errors.get()
    if errors is None:
        raise exceptions.StageDefinitionError(message)
    errors.append(message)


class StageFingerprintScope(TypedDict):
    """What `stage_fingerprint_scope` learned about one stage's fingerprint."""

    unsafe: bool  # a refusal in it was downgraded to a warning by unsafe fingerprinting


@contextlib.contextmanager
def stage_fingerprint_scope(stage_name: str) -> Iterator[StageFingerprintScope]:
    """Name the stage being fingerprinted, raise all of its unsafe-capture errors together, and
    report whether unsafe fingerprinting downgraded any of them to a warning (`unsafe`).

    Wraps a stage's manifest and its input and output loader fingerprints, so a refusal found
    in loader config is reported with the stage's own refusals. A pivot loader the stage's
    values hold is canonicalized once (`_form_memo`) and walked once (`_held_memo`) in it, and a
    loader class's own entries are computed once (`_loader_class_memo`).
    """
    scope = StageFingerprintScope(unsafe=False)
    token = _current_stage_name.set(stage_name)
    scope_token = _stage_scope.set(scope)
    memo_token = _held_memo.set({})
    form_token = _form_memo.set({})
    class_token = _loader_class_memo.set({})
    try:
        with _collected_capture_errors(), _type_facts_scope():
            yield scope
    finally:
        _loader_class_memo.reset(class_token)
        _form_memo.reset(form_token)
        _held_memo.reset(memo_token)
        _stage_scope.reset(scope_token)
        _current_stage_name.reset(token)


def get_stage_fingerprint_cached(stage_name: str, func: Callable[..., Any]) -> dict[str, str]:
    """Like get_stage_fingerprint, but with manifest-level caching.

    A hit skips the walk; `_try_manifest_cache_hit` says what it checks. On a miss the walk
    records every live value it reads, and the result is queued for `flush_and_close`. The call
    holds one state-directory record (``state_dir.entry_invocation``), so one made by hand,
    outside every entry point, reads ``core.state_dir`` once, not at each cache lookup.
    """
    from pivot import state_dir as state_dir_mod

    _t = metrics.start()
    with state_dir_mod.entry_invocation(), _type_facts_scope():
        # Try cache hit
        cached = _try_manifest_cache_hit(stage_name, func)
        if cached is not None:
            metrics.count("fingerprint.manifest_cache.hit")
            metrics.end("fingerprint.get_stage_fingerprint_cached", _t)
            return cached

        metrics.count("fingerprint.manifest_cache.miss")
        errors = _capture_errors.get()
        errors_before = len(errors) if errors is not None else 0
        unsafe_token = _unsafe_capture_warning.set(False)
        reads = list[tuple[tuple[str, ...], str]]()
        reads_token = _recorded_reads.set(reads)
        path_token = _walk_path.set(())
        try:
            # Compute with source tracking
            token = _current_stage_name.set(stage_name)
            try:
                with _collecting_sources() as source_map:
                    manifest = get_stage_fingerprint(func)
            finally:
                _current_stage_name.reset(token)

            # Inside a stage_fingerprint_scope, refusals are raised later; never cache them.
            refused = errors is not None and len(errors) > errors_before
            state_dir = None if _unsafe_capture_warning.get() or refused else _producing_state_dir()
            if state_dir is not None:
                _record_project_modules(source_map)
                key = _make_manifest_cache_key(stage_name, func)
                value = json.dumps(
                    {
                        "m": manifest,
                        "s": source_map,
                        "r": [[list(path), read_token] for path, read_token in reads],
                        "o": _by_name_origins(manifest),
                    },
                    separators=(",", ":"),
                ).encode()
                _pending_manifest_writes.setdefault(state_dir, list[tuple[bytes, bytes]]()).append(
                    (key, value)
                )

            return manifest
        finally:
            _walk_path.reset(path_token)
            _recorded_reads.reset(reads_token)
            _unsafe_capture_warning.reset(unsafe_token)
            metrics.end("fingerprint.get_stage_fingerprint_cached", _t)


def _collect_nested_code_globals(code: types.CodeType) -> set[str]:
    """Collect global names referenced by nested code objects in co_consts.

    Recursively walks co_consts to find nested code objects (from nested
    functions, lambdas, comprehensions, etc.) and returns the set of global
    names they reference via co_names.

    Notes:
        co_names is a superset of actual LOAD_GLOBAL names — it also includes
        attribute access names. The caller filters via func.__globals__ and
        _process_closure_values, making false positives benign (over-invalidation
        is safe; under-invalidation is catastrophic).

        On Python 3.12+ (PEP 709), list/dict/set comprehensions are inlined and
        no longer produce code objects in co_consts. Their globals appear in the
        parent function's co_names and are handled by getclosurevars() instead.
        Generator expressions still produce code objects and are walked here.
    """
    nested_globals = set[str]()
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            nested_globals.update(const.co_names)
            nested_globals.update(_collect_nested_code_globals(const))
    return nested_globals


def _source_tree(func: Callable[..., Any]) -> ast.Module | None:
    """Parse `func`'s own source, without following a functools.wraps chain.

    None for a wrapper without code of its own (functools.cache): it has no source of its own.
    """
    code = getattr(func, "__code__", None)
    if hasattr(func, "__wrapped__"):
        if not isinstance(code, types.CodeType):
            return None
        target: object = code
    else:
        target = func
    try:
        return ast.parse(textwrap.dedent(loaded_source.getsource(target)))
    except (OSError, TypeError, SyntaxError):
        return None


def _wrapped_source_tree(func: Callable[..., Any]) -> ast.Module | None:
    """Parse the source `inspect.getsource(func)` returns, stopped above a mark
    (`_source_object`): for a functools.wraps wrapper, the function at the end of its
    `__wrapped__` chain, or the user decorator above a marked function on that chain."""
    target = _source_object(func)
    if target is None:
        return None
    try:
        return ast.parse(textwrap.dedent(loaded_source.getsource(target)))
    except (OSError, TypeError, SyntaxError):
        return None


def _check_dynamic_name_access(
    func: Callable[..., Any], tree: ast.Module, *, dunder_import: bool = False
) -> None:
    """Refuse `getattr` with a computed name, `globals()`/`locals()`, and `x.import_module(...)`.

    With `dunder_import`, also refuse `__import__(...)` and `x.__import__(...)`. The
    `@fingerprint_by_name` decorator asks for this: the walk does not read a marked body, so
    the worker's stage-import check would count such an import of a user module on every run.
    Other dynamic imports are left to the worker's stage-import check, which makes a stage that
    loads uncovered user code at run time not reusable.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            if node.func.id == "getattr" and len(node.args) >= 2:
                if not isinstance(node.args[1], ast.Constant):
                    message = (
                        f"'{getattr(func, '__qualname__', '?')}' uses getattr() "
                        + "with non-literal attribute. Use direct attribute access."
                    )
                    raise exceptions.StageDefinitionError(message)
            elif node.func.id in ("globals", "locals"):
                message = (
                    f"'{getattr(func, '__qualname__', '?')}' uses {node.func.id}() "
                    + "which bypasses fingerprint tracking."
                )
                raise exceptions.StageDefinitionError(message)
            dynamic = dunder_import and node.func.id == "__import__"
        elif isinstance(node.func, ast.Attribute):
            dynamic = node.func.attr == "import_module" or (
                dunder_import and node.func.attr == "__import__"
            )
        else:
            dynamic = False
        if dynamic:
            message = (
                f"'{getattr(func, '__qualname__', '?')}' uses dynamic imports. "
                + "Use static imports instead."
            )
            raise exceptions.StageDefinitionError(message)


def _get_stage_fingerprint_impl(func: Callable[..., Any], visited: set[int]) -> dict[str, str]:
    """Internal implementation of get_stage_fingerprint."""
    manifest = dict[str, str]()

    func_id = id(func)
    if func_id in visited:
        return manifest
    visited.add(func_id)

    func_name = getattr(func, "__name__", "<lambda>")
    manifest[f"self:{func_name}"] = hash_function_ast(func)

    tree: ast.Module | None = None
    if is_user_code(func):
        tree = _source_tree(func)
        # The refusals read the source `inspect.getsource` returns, which follows a
        # functools.wraps chain to the wrapped function, as on main, and stops at the user
        # decorator above a mark; the body-import walk below reads the wrapper's own source.
        check_tree = _wrapped_source_tree(func) if hasattr(func, "__wrapped__") else tree
        if check_tree is not None:
            _check_dynamic_name_access(func, check_tree)

    _t_closure = metrics.start()
    try:
        # Use cached result if available (getclosurevars is expensive at ~0.5ms)
        if (closure_vars := _getclosurevars_cache.get(func)) is None:
            closure_vars = inspect.getclosurevars(func)
            _getclosurevars_cache[func] = closure_vars
    except (TypeError, AttributeError):
        metrics.end("fingerprint.getclosurevars", _t_closure)
        if isinstance(func, type):
            _process_class_body_dependencies(func, manifest, visited)
            _process_user_methods(func, manifest, visited)
            return manifest
        # A C-level wrapper such as functools.cache or lru_cache is not a Python function, so
        # getclosurevars rejects it. The function it wraps reads the module values, so walk
        # that function's globals, nonlocals and defaults instead. The unwrap stops at a user
        # decorator above a mark, which the walk enters, or at the mark, which it does not.
        unwrapped = unwrap(func, stop=_stop_at_mark)
        if unwrapped is not func and inspect.isfunction(unwrapped):
            _record_read(func, "u")
            if not fingerprinted_by_name(unwrapped):
                with _walking("u"):
                    for key, value in _get_stage_fingerprint_impl(unwrapped, visited).items():
                        if not key.startswith("self:"):
                            manifest[key] = value
        return manifest
    metrics.end("fingerprint.getclosurevars", _t_closure)

    _process_closure_values(closure_vars.globals, func, manifest, visited, prefix="g")
    _process_closure_values(closure_vars.nonlocals, func, manifest, visited, prefix="c")
    # A builtin name is a global lookup that found nothing; a module may bind it at import.
    for name in closure_vars.builtins:
        if not name.startswith("__"):
            _record_read(func, f"g:{name}")

    # Discover globals referenced by nested functions/lambdas
    # (getclosurevars only sees the outer function's own bytecode)
    code_obj = getattr(func, "__code__", None)
    func_globals = getattr(func, "__globals__", None)
    if code_obj is not None and func_globals is not None:
        already_processed = set(closure_vars.globals.keys())
        nested_global_names = _collect_nested_code_globals(code_obj)
        new_names = nested_global_names - already_processed
        if new_names:
            nested_globals_dict = dict[str, Any]()
            for name in sorted(new_names):
                if name in func_globals:
                    nested_globals_dict[name] = func_globals[name]
                elif not name.startswith("__"):
                    _record_read(func, f"g:{name}")
            if nested_globals_dict:
                _process_closure_values(nested_globals_dict, func, manifest, visited, prefix="g")

    _process_default_values(func, manifest, visited)
    _process_type_hint_dependencies(func, manifest, visited)
    if tree is not None:
        _process_body_imports(func, tree, manifest, visited)

    return manifest


def _process_user_methods(cls: type, manifest: dict[str, str], visited: set[int]) -> None:
    """Walk each callable user class member as a stage helper or refuse opaque state."""
    class_ref = _definition_ref(cls)
    stage_name = _current_stage_name.get() or cls.__qualname__
    for label, member in _user_methods(cls):
        step = f"m:{label}"
        _record_read(cls, step)
        with _walking(step):
            binding_ref = f"{class_ref}.{label}"
            _process_captured_value(
                binding_ref,
                binding_ref,
                member,
                cls,
                manifest,
                visited,
                stage_name,
                chain=None,
            )
            if isinstance(member, type) and hasattr(member, "model_fields"):
                _hash_pydantic_schema(
                    cast("type[_PydanticModelProtocol]", member), manifest, visited
                )
    _process_dataclass_defaults(cls, manifest, visited)


def _dataclass_field_value(cls: type, name: str) -> Any:
    """Return one dataclass field's default or factory, or `_UNREAD` when absent."""
    fields = cast("Mapping[str, dataclasses.Field[Any]]", vars(cls).get("__dataclass_fields__", {}))
    field = fields.get(name)
    if field is None:
        return _UNREAD
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory
    return _UNREAD


def _dataclass_field_token(cls: type, name: str) -> str:
    """Return a stable token for a dataclass field's live default or factory."""
    fields = cast("Mapping[str, dataclasses.Field[Any]]", vars(cls).get("__dataclass_fields__", {}))
    field = fields.get(name)
    if field is None:
        return "absent"
    if field.default_factory is not dataclasses.MISSING:
        return _value_token(field.default_factory)
    if field.default is dataclasses.MISSING:
        return "absent"
    serialized = _serialize_value_for_hash(field.default)
    return f"field:{xxhash.xxh64(serialized.encode()).hexdigest()}"


def _process_dataclass_defaults(cls: type, manifest: dict[str, str], visited: set[int]) -> None:
    """Track each own dataclass field's live default or factory."""
    if not dataclasses.is_dataclass(cls):
        return
    fields = cast("Mapping[str, dataclasses.Field[Any]]", vars(cls).get("__dataclass_fields__", {}))
    annotations = cast("Mapping[str, Any]", vars(cls).get("__annotations__", {}))
    class_ref = _definition_ref(cls)
    stage_name = _current_stage_name.get() or cls.__qualname__
    for name in annotations:
        field = fields.get(name)
        if field is None:
            continue
        step = f"fd:{name}"
        _record_read(cls, step)
        binding_ref = f"{class_ref}.{name}"
        if field.default is not dataclasses.MISSING:
            (serialized,), held = _serialize_collecting([("default", field.default)])
            manifest[f"default:{binding_ref}"] = xxhash.xxh64(serialized.encode()).hexdigest()
            if held:
                with _walking(step):
                    _merge_held_loaders(manifest, held, read="hd")
            continue
        if field.default_factory is dataclasses.MISSING:
            continue
        with _walking(step):
            _process_captured_value(
                binding_ref,
                binding_ref,
                field.default_factory,
                cls,
                manifest,
                visited,
                stage_name,
                chain=None,
            )


def _class_member_parts(name: str, raw: Any) -> list[tuple[str, Any]]:
    """Return the concrete callable values one class-dict member can execute."""
    if isinstance(raw, property):
        return [(name, raw.fget), (f"{name}.fset", raw.fset), (f"{name}.fdel", raw.fdel)]
    if isinstance(raw, functools.cached_property):
        return [(name, cast("functools.cached_property[Any]", raw).func)]
    if isinstance(raw, (classmethod, staticmethod)):
        return [(name, cast("Any", raw.__func__))]
    if isinstance(raw, functools.partialmethod):
        return [
            (
                name,
                functools.partial(
                    cast("Callable[..., Any]", raw.func), *raw.args, **(raw.keywords or {})
                ),
            )
        ]
    if isinstance(raw, functools.singledispatchmethod):
        dispatcher = cast("Any", raw.dispatcher)
        members = [(name, raw.func)]
        for registered_type, function in cast("Mapping[type, Any]", dispatcher.registry).items():
            if registered_type is object:
                continue
            label = f"{name}.register:{registered_type.__module__}.{registered_type.__qualname__}"
            members.append((label, function))
        return members
    return [(name, raw)]


@functools.lru_cache(maxsize=4096)
def _is_user_source_file(filename: str) -> bool:
    """Whether a code object's filename is a real source file outside library directories."""
    try:
        path = pathlib.Path(filename).resolve()
    except OSError:
        return False
    return path.is_file() and not is_library_path(path)


def _written_in_user_source(function: types.FunctionType) -> bool:
    """Whether a function's own code object was compiled from a user source file."""
    return _is_user_source_file(function.__code__.co_filename)


def _user_member_value(value: Any) -> Any:
    """Return a class member that needs tracking, or `_UNREAD` when library machinery owns it.

    A member on a chain that holds a mark takes the mark decision (`_mark_decision`) before
    any unwrap. A member recorded by name, such as a marked `@contextmanager` or
    `@functools.cache` helper held as a class attribute, is returned, so its name is in the
    manifest. A member with a user class instance above the mark is returned too, so the walk
    refuses it rather than dropping it. Each unwrap below stops at a user decorator above a
    mark, or at the mark (`_stop_at_mark`), and a marked object it reaches is returned.
    """
    decision = _mark_decision(value)
    if decision is _MarkDecision.BY_NAME or decision is _MarkDecision.REFUSED:
        return value
    if isinstance(value, functools.partial):
        return cast("Any", value)
    if inspect.isfunction(value):
        if _written_in_user_source(value):
            return value
        unwrapped = unwrap(value, stop=_stop_at_mark)
        if fingerprinted_by_name(unwrapped):
            return unwrapped
        if unwrapped is not value and inspect.isfunction(unwrapped):
            return unwrapped if _written_in_user_source(unwrapped) else _UNREAD
        return _UNREAD
    if isinstance(value, type):
        return value if is_user_code(value) else _UNREAD
    if _is_callable_instance(value) or _carries_instance(value):
        return value
    if callable(value):
        unwrapped = unwrap(value, stop=_stop_at_mark)
        if fingerprinted_by_name(unwrapped):
            return unwrapped
        if unwrapped is not value and inspect.isfunction(unwrapped):
            return unwrapped if _written_in_user_source(unwrapped) else _UNREAD
    if _has_attribute(value, "__get__") and _is_user_class_instance(value):
        return value
    return _UNREAD


def _user_methods(cls: type) -> list[tuple[str, Any]]:
    """Trackable or refused callable members user classes expose, labelled for cache replay."""

    def collect() -> list[tuple[str, Any]]:
        found = list[tuple[str, Any]]()
        mro = cls.__mro__
        user = [is_user_code(klass) for klass in mro]
        end = max((index + 1 for index, is_user in enumerate(user) if is_user), default=0)
        seen = set[str]()
        for klass, is_user in zip(mro[:end], user[:end], strict=True):
            if not is_user:
                continue
            for name, raw in vars(klass).items():
                if name in seen:
                    continue
                seen.add(name)
                for label, member in _class_member_parts(name, raw):
                    if member is None:
                        continue
                    value = _user_member_value(member)
                    if value is not _UNREAD:
                        found.append((label, value))
        return found

    return _memoized("members", cls, collect)


class _BodyImport(typing.NamedTuple):
    """One name an import statement in a function body binds."""

    bound: str  # the name the statement binds in the body
    module: str  # the absolute name of the module the statement imports
    target: str  # the module `bound` names (`import`), or `module.attribute` (`from`)
    attribute: str | None  # the attribute of `module` a `from` import binds; None for `import`


def _body_imports(func: Callable[..., Any], tree: ast.Module) -> list[_BodyImport]:
    """Return imports in `func`'s own body, resolving relative names without executing them."""
    globals_dict = getattr(func, "__globals__", {})
    package = globals_dict.get("__package__")
    found = list[_BodyImport]()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is not None:
                    found.append(_BodyImport(alias.asname, alias.name, alias.name, None))
                else:
                    top = alias.name.partition(".")[0]
                    found.append(_BodyImport(top, alias.name, top, None))
        elif isinstance(node, ast.ImportFrom):
            try:
                module = importlib.util.resolve_name(
                    "." * node.level + (node.module or ""), package
                )
            except (ImportError, ValueError):
                continue
            for alias in node.names:
                bound = alias.asname or alias.name
                found.append(_BodyImport(bound, module, f"{module}.{alias.name}", alias.name))
    return found


def _body_import_module(name: str) -> types.ModuleType | None:
    """Return an already-loaded user module without executing an import."""
    module = sys.modules.get(name)
    return module if isinstance(module, types.ModuleType) and is_user_code(module) else None


def _module_spec_without_import(name: str) -> importlib.machinery.ModuleSpec | None:
    """Resolve `name` through finders without running package initializers.

    A top-level name the filesystem finder does not locate is asked of every `sys.meta_path`
    finder (`importlib.util.find_spec`): an editable install's finder (setuptools' PEP 660)
    locates its package this way. `find_spec` runs no package initializer for a top-level name,
    but a finder may import modules of its own (setuptools' distutils shim imports
    `setuptools._distutils`). A finder that raises is treated as not finding the name, so the
    lookup never refuses a stage over an error in a finder.
    """
    path: Sequence[str] | None = None
    spec: importlib.machinery.ModuleSpec | None = None
    parts = name.split(".")
    for index, part in enumerate(parts):
        current = ".".join(parts[: index + 1])
        loaded = sys.modules.get(current)
        if isinstance(loaded, types.ModuleType):
            spec = getattr(loaded, "__spec__", None)
            if not isinstance(spec, importlib.machinery.ModuleSpec):
                return None
        else:
            try:
                spec = importlib.machinery.PathFinder.find_spec(part, path)
                if spec is None and index == 0:
                    spec = importlib.util.find_spec(part)
            except Exception:  # noqa: BLE001 - any finder error means "not found"
                spec = None
            if spec is None:
                return None
        path = spec.submodule_search_locations
        if index + 1 < len(parts) and path is None:
            return None
    return spec


def _mark_module_covered(module_name: str, manifest: dict[str, str]) -> None:
    """Record that this manifest covers user module `module_name` and its user packages.

    The worker compares the user modules a stage imports with these entries.
    """
    parts = module_name.split(".")
    for index in range(1, len(parts) + 1):
        name = ".".join(parts[:index])
        module = sys.modules.get(name)
        if isinstance(module, types.ModuleType) and is_user_code(module):
            manifest.setdefault(f"module:{name}", "")


def _loader_entry(key: str, outputs: Collection[str]) -> str | None:
    """The key of a loader's own manifest (`get_loader_fingerprint`) that `key` holds in a stage
    manifest (`merge_loader_fingerprint`), or None: `dep:<dependency>:KEY`, a dependency named by
    its parameter (`name` or `name[i]`, no `:`), or `out:<output>:KEY`, `<output>` one of the
    stage's `outputs` (`types.identity_key`: its stage, or its stage and field, neither with a
    `:`). Those names end the prefix, not the key after it, so no parameter, field or stage named
    like a kind of entry (`module`, `byname`, `func`, ...) is read as one."""
    if key.startswith("dep:"):
        _dependency, colon, entry = key.removeprefix("dep:").partition(":")
        return entry if colon else None
    if not key.startswith("out:"):
        return None
    stage, colon, entry = key.removeprefix("out:").partition(":")
    if not colon:
        return None
    if stage in outputs:
        return entry
    field, colon, entry = entry.partition(":")
    return entry if colon and f"{stage}:{field}" in outputs else None


def _held_entry(key: str) -> str | None:
    """The key of a held loader's own manifest that `key` holds (`_merge_held_loaders`), after
    each `held:<digest>:` prefix: the digest is hexadecimal, so it holds no `:`. A key without
    the prefix is returned as it is, and a malformed one (`held:<digest>` alone) is None."""
    while key.startswith("held:"):
        _digest, colon, key = key.removeprefix("held:").partition(":")
        if not colon:
            return None
    return key


def manifest_entries(manifest: Mapping[str, str], outputs: Collection[str]) -> Iterator[str]:
    """Each entry of a stage manifest as the walk that made it keyed it: a stage's own key as it
    is, and a loader's without its `dep:`/`out:` prefix (`_loader_entry`, which the stage's
    output keys `outputs` place) and without the `held:<digest>:` prefix of each loader that
    holds it (`_held_entry`). An entry's kind is its first segment, so no dependency, output,
    field or stage name is read as a kind."""
    for key in manifest:
        entry = _loader_entry(key, outputs) if key.startswith(("dep:", "out:")) else key
        if entry is not None and (entry := _held_entry(entry)) is not None:
            yield entry


def covered_user_modules(manifest: Mapping[str, str], outputs: Collection[str]) -> set[str]:
    """Names of the user modules a stage manifest covers: its `module:NAME` entries, the stage's
    own and its loaders' (`manifest_entries`, which the stage's output keys `outputs` place)."""
    return {
        entry.removeprefix("module:")
        for entry in manifest_entries(manifest, outputs)
        if entry.startswith("module:")
    }


def _mark_by_name(module_name: object, manifest: dict[str, str]) -> None:
    """Record that a stage's worker imports module `module_name` by name to run code the walk
    does not hash: a library module read as a global or imported in a function body, or the
    module that defines a callable recorded by name (`ext:`). `loaded_source.check_worker_imports`
    compares each with the module the worker would import, and the manifest cache re-checks
    where each resolves (`_try_manifest_cache_hit`). A lock does not record these entries
    (`without_by_name`)."""
    if isinstance(module_name, str) and module_name:
        manifest.setdefault(f"byname:{module_name}", "")


def _by_name_entry(key: str) -> str | None:
    """The module a `byname:` entry names. Only a key that starts with `byname:` is one: a
    loader's `byname:` entries join the stage's own without a prefix (`merge_loader_fingerprint`),
    and each other key starts with a fixed kind (`self:`, `dep:`, `out:`, `wire:`, ...), so no
    dependency, output, stage or module name spells one."""
    if key.startswith("byname:"):
        return key.removeprefix("byname:")
    return None


def by_name_modules(manifest: Mapping[str, str], outputs: Collection[str]) -> set[str]:
    """Names of the modules a stage's worker imports by name, as far as its manifest says: the
    user modules it covers (`covered_user_modules`, which the stage's output keys `outputs` place)
    and the `byname:` entries, its loaders' included."""
    names = covered_user_modules(manifest, outputs)
    names.update(name for key in manifest if (name := _by_name_entry(key)) is not None)
    return names


def without_by_name(manifest: Mapping[str, str]) -> dict[str, str]:
    """`manifest` as a lock records it: without the `byname:` entries, which the worker-import
    check and the manifest cache read. They name only modules whose code no entry hashes, so a
    lock written before they existed stays current and an upgrade reruns nothing. Every `dep:`
    and `out:` entry stays, a dependency or output named `byname` included."""
    return {key: value for key, value in manifest.items() if _by_name_entry(key) is None}


def _is_library_spec(spec: importlib.machinery.ModuleSpec) -> bool:
    """Whether a top-level module's spec locates library code (`is_library_path`), or no file."""
    if spec.has_location and spec.origin is not None:
        return is_library_path(pathlib.Path(spec.origin).resolve())
    locations = list(spec.submodule_search_locations or ())
    if not locations:
        return True
    return all(is_library_path(pathlib.Path(location).resolve()) for location in locations)


def _unloaded_body_import_error(
    func: Callable[..., Any], module_name: str
) -> exceptions.StageDefinitionError:
    qualname = getattr(func, "__qualname__", "?")
    return exceptions.StageDefinitionError(
        f"'{qualname}' imports user module '{module_name}' in its body, but pipeline discovery "
        + "did not load it, so Pivot cannot fingerprint it without running it. "
        + "Fix: import it at module level (or in pipeline.py) so discovery loads it, "
        + "or move the value into the stage's StageParams."
    )


def _is_user_module_name(name: str) -> bool:
    """Whether `name` locates a user module, found without importing it or its packages."""
    spec = _module_spec_without_import(name)
    return spec is not None and not _is_library_spec(spec)


def _process_body_imports(
    func: Callable[..., Any], tree: ast.Module, manifest: dict[str, str], visited: set[int]
) -> None:
    """Walk body imports of user modules discovery loaded; refuse unloaded user modules.

    Pivot never imports a module to fingerprint it: importing a module discovery did not load
    can run any code, so reading its source cannot decide its fingerprint.
    """
    imports = _body_imports(func, tree)
    if not imports:
        return
    stage_name = _current_stage_name.get() or getattr(func, "__name__", "<unknown>")
    owner = _walk_owner(func)
    for body_import in imports:
        module = _body_import_module(body_import.module)
        if module is None:
            if _is_user_module_name(body_import.module):
                raise _unloaded_body_import_error(func, body_import.module)
            _mark_by_name(body_import.module, manifest)
            continue
        _mark_module_covered(body_import.module, manifest)
        imported = f"bi:{body_import.module}"
        _record_read(func, imported)
        attribute = body_import.attribute
        if attribute is not None and not _names_submodule(module, attribute):
            with _walking(imported):
                step = f"a:{attribute}"
                _record_read(module, step)
                with _walking(step):
                    binding_ref = f"{module.__name__}.{attribute}"
                    _process_captured_value(
                        body_import.bound,
                        binding_ref,
                        getattr(module, attribute),
                        func,
                        manifest,
                        visited,
                        stage_name,
                        chain=body_import.bound,
                    )
            continue
        target = _body_import_module(body_import.target)
        if target is None:
            if _is_user_module_name(body_import.target):
                raise _unloaded_body_import_error(func, body_import.target)
            _mark_by_name(body_import.target, manifest)
            continue
        step = f"bi:{body_import.target}"
        if step != imported:
            _record_read(func, step)
        with _walking(step):
            _process_captured_value(
                body_import.bound,
                f"{owner}.{body_import.bound}",
                target,
                func,
                manifest,
                visited,
                stage_name,
                chain=body_import.bound,
            )


def _names_submodule(module: types.ModuleType, attribute: str) -> bool:
    """Whether `from module import attribute` binds a submodule: `module` lacks the attribute,
    so the import loads `module.attribute`, or the attribute is that submodule."""
    value = vars(module).get(attribute, _UNREAD)
    return value is _UNREAD or (
        isinstance(value, types.ModuleType) and value.__name__ == f"{module.__name__}.{attribute}"
    )


def _process_class_body_dependencies(
    cls: type, manifest: dict[str, str], visited: set[int]
) -> None:
    try:
        source = loaded_source.getsource(cls)
        tree = ast.parse(textwrap.dedent(source))
    except (OSError, TypeError, SyntaxError):
        return
    class_def = next((node for node in tree.body if isinstance(node, ast.ClassDef)), None)
    if class_def is None:
        return

    module = sys.modules.get(cls.__module__)
    if module is None:
        return
    ns = vars(module)

    names = set[str]()
    dotted_refs = list[tuple[str, ...]]()

    for base_node in class_def.bases:
        _collect_annotation_names(base_node, names, dotted_refs)

    for stmt in class_def.body:
        if isinstance(stmt, ast.AnnAssign):
            _collect_annotation_names(stmt.annotation, names, dotted_refs)

    for name in sorted(names):
        _record_read(cls, f"n:{name}")
        if name not in ns:
            continue
        value = ns[name]
        if isinstance(value, type) and is_user_code(value):
            with _walking(f"n:{name}"):
                key = f"class:{_unique_ref(value)}"
                if key not in manifest:
                    _add_callable_to_manifest(key, value, manifest, visited)
                if hasattr(value, "model_fields"):
                    _hash_pydantic_schema(
                        cast("type[_PydanticModelProtocol]", value), manifest, visited
                    )

    for parts in dotted_refs:
        step = f"np:{'.'.join(parts)}"
        _record_read(cls, step)
        resolved = _resolve_dotted_path(parts, ns)
        if resolved is None or not isinstance(resolved, type) or not is_user_code(resolved):
            continue
        with _walking(step):
            key = f"class:{_unique_ref(resolved)}"
            if key not in manifest:
                _add_callable_to_manifest(key, resolved, manifest, visited)
            if hasattr(resolved, "model_fields"):
                _hash_pydantic_schema(
                    cast("type[_PydanticModelProtocol]", resolved), manifest, visited
                )


def _collect_annotation_names(
    node: ast.AST, names: set[str], dotted_refs: list[tuple[str, ...]]
) -> None:
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, ast.Subscript):
        _collect_annotation_names(node.value, names, dotted_refs)
        _collect_annotation_names(node.slice, names, dotted_refs)
    elif isinstance(node, ast.BinOp):
        _collect_annotation_names(node.left, names, dotted_refs)
        _collect_annotation_names(node.right, names, dotted_refs)
    elif isinstance(node, ast.Tuple):
        for elt in node.elts:
            _collect_annotation_names(elt, names, dotted_refs)
    elif isinstance(node, ast.Attribute):
        parts = _collect_dotted_path(node)
        if len(parts) >= 2:
            dotted_refs.append(parts)
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        # Handle dotted string annotations like "pkg.Type" by treating as dotted ref
        if "." in node.value:
            parts = tuple(node.value.split("."))
            if len(parts) >= 2:
                dotted_refs.append(parts)
        else:
            names.add(node.value)


def _collect_dotted_path(node: ast.Attribute) -> tuple[str, ...]:
    parts = [node.attr]
    current = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return tuple(reversed(parts))
    return ()


def _resolve_dotted_path(parts: tuple[str, ...], ns: dict[str, Any]) -> Any:
    if parts[0] not in ns:
        return None
    obj = ns[parts[0]]
    for part in parts[1:]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _method_function(method: Callable[..., Any]) -> Callable[..., Any]:
    """Return the underlying function behind a bound method."""
    return cast("Callable[..., Any]", getattr(method, "__func__", method))


def get_loader_fingerprint(loader: "loaders.Writer[Any] | loaders.Reader[Any]") -> dict[str, str]:
    """Generate fingerprint manifest for a loader instance.

    Handles Reader, Writer, and Loader (which inherits from both) instances.
    Only fingerprints methods that exist on the handler type:
    - Reader: 'loader:<module>.<qualname>:load' + empty() if overridden
    - Writer: 'loader:<module>.<qualname>:save'
    - Loader: All of the above
    - Always: 'loader:<module>.<qualname>:config' for dataclass field values

    A user loader class's `load`, `save` and `empty`, the user members of the class that their
    code reaches (`_reachable_loader_members`: a helper `load` calls as `self.decorate()`, a
    property it reads, transitively), and the functions and classes bound in its config fields,
    are walked like stage helpers, so the module values they read are tracked or refused. A
    member no such code reads is not walked. Methods inherited from `pivot.loaders` are
    framework code and are only hashed. A pivot loader the config holds (`Outer(Inner())`, also
    inside a container, a partial or a dataclass in a field) gets this fingerprint too, under
    `held:<digest>:` (`_merge_held_loaders`).

    The loader is marked as being fingerprinted for the whole call (`_loader_stack`), so a
    loader its own walk reaches again (its config holds it, or a dataclass default its code
    reads does) is not walked again. The loader memos are kept for the call (`_loader_memos`),
    unless a stage's scope keeps them already.
    """
    stack_token = _loader_stack.set((*_loader_stack.get(), id(cast("object", loader))))
    try:
        with _loader_memos():
            return _loader_manifest(loader)
    finally:
        _loader_stack.reset(stack_token)


def _loader_manifest(loader: "loaders.Writer[Any] | loaders.Reader[Any]") -> dict[str, str]:
    """The manifest `get_loader_fingerprint` gives `loader`: its class's entries
    (`_loader_class_entries`), then the walk of the values its config fields bind, its config
    hash, and the loaders its config holds."""
    loader_type: type[Any] = type(cast("object", loader))
    class_name = _definition_ref(loader_type)
    manifest = dict(_loader_class_entries(loader_type))
    visited = set[int]()
    loader_any = cast("Any", loader)
    fields = dataclasses.fields(loader_any)
    for field in fields:
        value = getattr(loader_any, field.name)
        _process_bound_values(f"{class_name}.{field.name}", value, manifest, visited)
    texts, held = _serialize_collecting(
        [(field.name, getattr(loader_any, field.name)) for field in fields]
    )
    # The text `loader_config_text` gives.
    config_str = ",".join(f"{field.name}={text}" for field, text in zip(fields, texts, strict=True))
    if config_str:
        manifest[f"loader:{class_name}:config"] = xxhash.xxh64(config_str.encode()).hexdigest()
    if held:
        _merge_held_loaders(manifest, held, read=None)

    return manifest


def _loader_class_entries(loader_type: type[Any]) -> dict[str, str]:
    """The entries of a loader's manifest that its class decides: the walks of its user-defined
    `load`, `save` and `empty` and of the members they reach, and the hashes of its `load`,
    `save` and overridden `empty`. Kept for the stage (`_loader_class_memo`), so many loaders of
    one class walk it once."""
    memo = _loader_class_memo.get()
    cached = None if memo is None else memo.get(id(loader_type))
    if cached is not None:
        return cached[1]
    manifest = dict[str, str]()
    class_name = _definition_ref(loader_type)
    visited = set[int]()

    # Import here to avoid circular import
    from pivot import loaders as loaders_module

    for method_name in _LOADER_METHODS:
        method = _user_defined_method(loader_type, method_name)
        if method is not None:
            _process_callable_dependency(f"{class_name}.{method_name}", method, manifest, visited)
    stage_name = _current_stage_name.get() or loader_type.__qualname__
    for binding_ref, member in _reachable_loader_members(loader_type):
        _process_captured_value(
            binding_ref, binding_ref, member, loader_type, manifest, visited, stage_name, chain=None
        )

    # Use each method definition rather than a temporary bound method so the
    # weak AST hash cache retains the entry across loader fingerprint calls.
    methods = cast("Any", loader_type)
    if issubclass(loader_type, loaders_module.Writer):
        manifest[f"loader:{class_name}:save"] = hash_function_ast(_method_function(methods.save))

    # Fingerprint load() if this is a Reader
    if issubclass(loader_type, loaders_module.Reader):
        manifest[f"loader:{class_name}:load"] = hash_function_ast(_method_function(methods.load))

        # Only fingerprint empty() if it's overridden in a Loader subclass
        if issubclass(loader_type, loaders_module.Loader):
            for cls in loader_type.__mro__:
                if cls is loaders_module.Loader:
                    # Reached Loader base class without finding override
                    break
                if "empty" in cls.__dict__:
                    # Found override in a subclass
                    manifest[f"loader:{class_name}:empty"] = hash_function_ast(
                        _method_function(methods.empty)
                    )
                    break
    if memo is not None:
        memo[id(methods)] = (methods, manifest)
    return manifest


def merge_loader_fingerprint(
    manifest: dict[str, str], prefix: str, loader: "loaders.Writer[Any] | loaders.Reader[Any]"
) -> None:
    """Add `loader`'s manifest (`get_loader_fingerprint`) to a stage's `manifest`: each entry
    under `prefix` (`dep:<parameter>:`, `out:<output>:`), but the `byname:` entries as they are
    (`_merge_prefixed`)."""
    _merge_prefixed(manifest, prefix, get_loader_fingerprint(loader))


def _merge_prefixed(manifest: dict[str, str], prefix: str, entries: Mapping[str, str]) -> None:
    """Add a loader's manifest `entries` to `manifest`, each under `prefix`, but the `byname:`
    entries as they are. Those name modules the loader's code makes a worker import, not the
    dependency, output or holder, so a `byname:` entry is only ever a top-level key
    (`_by_name_entry`), which a lock leaves out, and a dependency or output named `byname` keeps
    every entry of its own."""
    for key, value in entries.items():
        if _by_name_entry(key) is None:
            manifest[f"{prefix}{key}"] = value
        else:
            manifest.setdefault(key, value)


def _partial_parts(partial_obj: Any) -> list[tuple[str, Any]]:
    """A partial's arguments and keywords, each named, as `_serialize_collecting` takes them."""
    return [("args", partial_obj.args), ("kwargs", partial_obj.keywords)]


def _serialize_collecting(
    parts: Sequence[tuple[str, Any]],
) -> tuple[list[str], list[_HeldLoader]]:
    """Serialize each part (`_serialize_value_for_hash`), and collect each pivot loader its
    canonical form meets (`_canonical_loader`) with its position: the part's name, then a step
    per container (`_held_path`). The serialization that records the value already visits
    them, so no new traversal is made."""
    held = list[_HeldLoader]()
    texts = list[str]()
    held_token = _held_loaders.set(held)
    try:
        for name, value in parts:
            path_token = _held_path.set(([name],))
            try:
                texts.append(_serialize_value_for_hash(value))
            finally:
                _held_path.reset(path_token)
    finally:
        _held_loaders.reset(held_token)
    return texts, held


def _held_position(item: _HeldLoader) -> str:
    """Where a held loader sits in the value that holds it (`_held_path`), as text."""
    return "/".join(cell[0] for cell in item.path)


def _is_pivot_loader(value: Any) -> bool:
    """Whether `value` is an instance of a pivot loader (`loaders.Reader` or `loaders.Writer`),
    decided by identity along its class's MRO, so no class's `__hash__` or `__eq__` runs."""
    from pivot import loaders as loaders_module

    return any(
        cls is loaders_module.Reader or cls is loaders_module.Writer for cls in type(value).__mro__
    )


def _held_loader_entries(held: list[_HeldLoader]) -> dict[str, dict[str, str]]:
    """The manifest of each pivot loader in `held`, as `get_loader_fingerprint` gives a
    top-level loader of its class, keyed by the digest of its position in the value that holds
    it (`_held_path`), so a swap of two loaders or an edit to one of two equal-looking ones is a
    change. Two loaders share a position only as equal forms in a set; each of those is keyed by
    its position and its manifest.

    The loaders are walked in the order of their positions and forms, which follows no hash
    seed: one being fingerprinted already (`_loader_stack`: a cycle) is left out, since its
    entries are in the manifest that holds it, so the order decides which of two loaders in a
    cycle holds the other's entries. Loaders that share a position and a form but not their
    classes are each walked with the memo as it stood before them, so their order does not
    matter either. A loader held many times is walked once per stage (`_held_memo`, by its form
    and the classes in it: two classes one factory made keep apart). A held loader's walk
    records no read of its own; its holder's caller records one read for all of them
    (`_merge_held_loaders`)."""
    with _loader_memos():
        return _held_loaders_walked(held)


def _held_loaders_walked(held: list[_HeldLoader]) -> dict[str, dict[str, str]]:
    """`_held_loader_entries`, with the loader memos kept (`_loader_memos`)."""
    stack = _loader_stack.get()
    memo = _held_memo.get()
    ordered = sorted(
        (item for item in held if id(item.loader) not in stack),
        key=lambda item: (_held_position(item), item.text),
    )
    walked = dict[tuple[str, tuple[int, ...]], dict[str, str]]()
    by_position = dict[str, list[dict[str, str]]]()
    index = 0
    while index < len(ordered):
        position, text = _held_position(ordered[index]), ordered[index].text
        end = index
        while end < len(ordered) and (
            _held_position(ordered[end]),
            ordered[end].text,
        ) == (position, text):
            end += 1
        group = ordered[index:end]
        apart = len({item.classes for item in group}) > 1
        before = None if memo is None or not apart else dict(memo)
        for item in group:
            key = (item.text, item.classes)
            entries = walked.get(key)
            if entries is None:
                entries = _walk_held_loader(item.loader, key, memo, before)
                walked[key] = entries
            found = by_position.setdefault(position, [])
            if entries not in found:
                found.append(entries)
        index = end
    keyed = dict[str, dict[str, str]]()
    for position, found in by_position.items():
        for entries in found:
            content = position if len(found) == 1 else f"{position}\0{_held_token({'': entries})}"
            keyed[xxhash.xxh64(content.encode()).hexdigest()] = entries
    return dict(sorted(keyed.items()))


def _walk_held_loader(
    loader: Any,
    key: tuple[str, tuple[int, ...]],
    memo: dict[tuple[str, tuple[int, ...]], tuple[Any, dict[str, str]]] | None,
    before: dict[tuple[str, tuple[int, ...]], tuple[Any, dict[str, str]]] | None,
) -> dict[str, str]:
    """`get_loader_fingerprint(loader)` with no read recorded, kept in the stage's `memo` under
    `key`; with `before`, walked with a copy of it as the memo instead (`_held_loader_entries`)."""
    cached = None if memo is None else memo.get(key)
    if cached is not None:
        return cached[1]
    reads_token = _recorded_reads.set(None)
    memo_token = None if before is None else _held_memo.set(dict(before))
    try:
        entries = get_loader_fingerprint(loader)
    finally:
        if memo_token is not None:
            _held_memo.reset(memo_token)
        _recorded_reads.reset(reads_token)
    if memo is not None:
        memo[key] = (loader, entries)
    return entries


def _held_token(entries: Mapping[str, Mapping[str, str]]) -> str:
    """The read token for the held loaders' manifests `entries` (`_held_loader_entries`)."""
    return xxhash.xxh64(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def _merge_held_loaders(
    manifest: dict[str, str], held: list[_HeldLoader], *, read: str | None
) -> None:
    """Add to `manifest` the fingerprint of each pivot loader in `held`, under
    `held:<digest>:` (`_held_loader_entries`; the digest is hexadecimal, so the prefix holds no
    `:` and no user name: `_held_entry`). With `read` (`hl` for a partial's arguments, `hd` for
    a dataclass field default), inside a cacheable stage walk, record one read of the value
    being visited, whose token is those fingerprints: a cache hit fingerprints its held loaders
    again, as each run fingerprints a top-level loader."""
    entries = _held_loader_entries(held)
    for digest, loader_entries in entries.items():
        _merge_prefixed(manifest, f"held:{digest}:", loader_entries)
    reads = _recorded_reads.get()
    if read is not None and entries and reads is not None:
        reads.append(((*_walk_path.get(), read), _held_token(entries)))


def _held_loaders_token(value: Any, step: str) -> str:
    """The read token of `value` (`_merge_held_loaders`): for `hl` a partial, its arguments and
    keywords; for `hd` a dataclass field default. The fingerprints of the loaders it holds."""
    parts = _partial_parts(value) if step == "hl" else [("default", value)]
    _texts, held = _serialize_collecting(parts)
    return _held_token(_held_loader_entries(held))


def loader_config_text(loader: "loaders.Writer[Any] | loaders.Reader[Any]") -> str:
    """Return a loader's dataclass field values as `name=<canonical JSON>`, comma-joined.

    This is the text hashed as `loader:<module>.<qualname>:config`. It does not depend on
    PYTHONHASHSEED or object identity: a function in a field is its definition and code hash,
    so a loader re-created by importing its pipeline file again gives the same text.
    """
    # Cast to Any for dataclass introspection - loader is a dataclass but generic params are unknown
    loader_any = cast("Any", loader)
    return ",".join(
        f"{field.name}={_serialize_value_for_hash(getattr(loader_any, field.name))}"
        for field in dataclasses.fields(loader_any)
    )


def _user_defined_method(cls: type, name: str) -> Callable[..., Any] | None:
    """Return the function behind `cls`'s nearest `name` attribute when user code defines it."""
    for klass in cls.__mro__:
        if name in klass.__dict__:
            member = klass.__dict__[name]
            function = getattr(member, "__func__", member)
            if inspect.isfunction(function) and is_user_code(function):
                return function
            return None
    return None


_LOADER_METHODS = ("load", "save", "empty")


def _reachable_loader_members(loader_type: type) -> list[tuple[str, Any]]:
    """The user members of a loader class that its user-defined `load`, `save` and `empty`
    reach, each with its manifest reference (`<module>.<defining class>.<label>`).

    Code reaches a member through the method's first parameter (`self`, or `cls`) in each
    static form (`_member_reads`): `self.<name>`, `getattr(self, "<name>")`,
    `type(self).<name>` and `self.__class__.<name>`, `super().<name>`, and the dunder methods a
    builtin call on `self` runs (`str(self)` runs `__str__`). The functions such names hold reach
    further: a helper method, a property's getter and setter, a static or class method, and
    the function a `functools.wraps` wrapper wraps. A name is looked up on the class's MRO as
    Python looks it up, from after the defining class for `super()`. A class constant read so
    (`self.SUFFIX`) is a value the walk records or refuses as it does a global. A dataclass
    field is config (`loader_config_text`), not a member; a name a library class answers is
    framework code. A member no reaching code reads is not walked, so an unused helper adds no
    entry and no refusal.

    When reaching code uses `self` in a way no static form names (`self` passed to another
    function, `vars(self)`, an operator, `type(self)(...)`), any member may run, so every user
    member of the class is walked (`_all_loader_members`).
    """

    def collect() -> list[tuple[str, Any]]:
        mro = loader_type.__mro__
        fields = _dataclass_field_names(loader_type)
        found = list[tuple[str, Any]]()
        seen = set[tuple[int, str]]()
        work = list[tuple[Any, type, bool]]()  # a function, its class, whether it takes self
        for name in _LOADER_METHODS:
            resolved = _loader_member(mro, name, 0, fields)
            if resolved is None:
                continue
            owner, raw = resolved
            seen.add((id(owner), name))
            function = getattr(raw, "__func__", raw)
            if inspect.isfunction(function):
                work.extend(_member_functions(function, owner, raw))
        while work:
            function, defined_in, takes_self = work.pop()
            tree = _source_tree(function)
            if tree is None:
                continue
            reads, dynamic = _member_reads(tree, takes_self)
            if dynamic:
                return _all_loader_members(loader_type, fields)
            for via_super, name in reads:
                start = mro.index(defined_in) + 1 if via_super else 0
                resolved = _loader_member(mro, name, start, fields)
                if resolved is None:
                    continue
                owner, raw = resolved
                if (id(owner), name) in seen:
                    continue
                seen.add((id(owner), name))
                owner_ref = _definition_ref(owner)
                for label, value in _member_values(name, raw):
                    found.append((f"{owner_ref}.{label}", value))
                    if inspect.isfunction(value):
                        work.extend(_member_functions(value, owner, raw))
        return found

    return _memoized("loader-members", loader_type, collect)


def _dataclass_field_names(cls: type) -> set[str]:
    """The names of a dataclass's fields; empty for any other class."""
    if not dataclasses.is_dataclass(cls):
        return set[str]()
    return {field.name for field in dataclasses.fields(cls)}


def _member_values(name: str, raw: Any) -> list[tuple[str, Any]]:
    """The values of one class-dict member the walk records: each callable part it can run
    that user code owns (`_class_member_parts`, `_user_member_value`), or the member itself
    when it is a plain value (a class constant), which the walk records or refuses as it does
    a global (`_process_captured_value`)."""
    values = list[tuple[str, Any]]()
    for label, member in _class_member_parts(name, raw):
        if member is None:
            continue
        value = _user_member_value(member)
        if value is not _UNREAD:
            values.append((label, value))
        elif not callable(member) and not _has_attribute(type(member), "__get__"):
            values.append((label, member))
    return values


def _member_functions(function: Any, owner: type, raw: Any) -> list[tuple[Any, type, bool]]:
    """The functions whose bodies the reachable walk reads for a member function, each with
    the class that holds the member and whether it takes the instance first: each function a
    `functools.wraps` chain wraps (`__wrapped__`) that user source defines, and each wrapper
    on the chain but one that only forwards `*args` to what it wraps (it names no member, and
    receives the instance unnamed, which `_member_reads` would read as a dynamic use)."""
    takes_self = not isinstance(raw, staticmethod)
    functions = list[tuple[Any, type, bool]]()
    current: Any = function
    while inspect.isfunction(current):
        wrapped = getattr(current, "__wrapped__", None)
        forwards = wrapped is not None and current.__code__.co_argcount == 0
        if not forwards and (current is function or _written_in_user_source(current)):
            functions.append((current, owner, takes_self))
        current = wrapped
    return functions


def _all_loader_members(loader_type: type, fields: set[str]) -> list[tuple[str, Any]]:
    """Every user member of a loader class but `load`, `save` and `empty` (which the loader's
    walk hashes itself) and its dataclass fields: each callable part user code owns and each
    plain value (`_member_values`) that a user class on the MRO defines and no earlier class
    overrides, under its defining class. Dunder values (`__module__`, `__dataclass_fields__`)
    and the ABC machinery's cache (`_abc_impl`) are the class machinery's, not members;
    user-defined dunder methods are members."""
    found = list[tuple[str, Any]]()
    seen = set[str]()
    for klass in loader_type.__mro__:
        names = [name for name in vars(klass) if name not in seen]
        seen.update(names)
        if not is_user_code(klass):
            continue
        owner_ref = _definition_ref(klass)
        for name in names:
            if name in _LOADER_METHODS or name in fields or name == "_abc_impl":
                continue
            for label, value in _member_values(name, vars(klass)[name]):
                if name.startswith("__") and name.endswith("__") and not callable(value):
                    continue
                found.append((f"{owner_ref}.{label}", value))
    return found


def _loader_member(
    mro: tuple[type, ...], name: str, start: int, fields: set[str]
) -> tuple[type, Any] | None:
    """The user class on `mro[start:]` that holds `name` and the raw member it holds, as an
    attribute lookup finds it; None for a dataclass field, an absent name, or a name a library
    class holds first."""
    if name in fields:
        return None
    for klass in mro[start:]:
        if name in vars(klass):
            return (klass, vars(klass)[name]) if is_user_code(klass) else None
    return None


# The dunder methods a builtin call on an instance runs, by the builtin's name; a call of any
# other function on `self` is a dynamic use (`_member_reads`).
_BUILTIN_DUNDERS: dict[str, tuple[str, ...]] = {
    "str": ("__str__", "__repr__"),
    "repr": ("__repr__",),
    "ascii": ("__repr__",),
    "format": ("__format__", "__str__", "__repr__"),
    "print": ("__str__", "__repr__"),
    "len": ("__len__",),
    "bool": ("__bool__", "__len__"),
    "hash": ("__hash__",),
    "iter": ("__iter__", "__getitem__"),
    "next": ("__next__",),
    "reversed": ("__reversed__", "__len__", "__getitem__"),
    "int": ("__int__", "__index__", "__trunc__"),
    "float": ("__float__", "__index__"),
    "complex": ("__complex__", "__float__", "__index__"),
    "abs": ("__abs__",),
    "round": ("__round__",),
    "list": ("__iter__", "__getitem__", "__len__"),
    "tuple": ("__iter__", "__getitem__", "__len__"),
    "set": ("__iter__", "__getitem__"),
    "frozenset": ("__iter__", "__getitem__"),
    "sorted": ("__iter__", "__getitem__"),
    "enumerate": ("__iter__", "__getitem__"),
}
# Builtins that read no member of the instance they are given.
_MEMBERLESS_BUILTINS = frozenset(("isinstance", "issubclass", "id", "callable"))
# Builtins that read the attribute their second argument names, when it is a literal.
_NAMED_ATTRIBUTE_BUILTINS = frozenset(("getattr", "hasattr", "setattr", "delattr"))
# f-string conversions (`ast.FormattedValue.conversion`) and the builtin each one calls.
_FSTRING_CONVERSIONS = {-1: "format", ord("s"): "str", ord("r"): "repr", ord("a"): "ascii"}


def _member_reads(tree: ast.Module, takes_self: bool) -> tuple[list[tuple[bool, str]], bool]:
    """The class members a method's body reads through its first parameter (the receiver, when
    it takes one), as `(via_super, name)`, and whether it uses the receiver in a way no static
    form names (a dynamic use).

    Static forms: `self.name`; `getattr(self, "name")` (also `hasattr`, `setattr`, `delattr`);
    `type(self).name` and `self.__class__.name`; `super().name` and `super(C, self).name`
    (`via_super`); a builtin call on `self` (`str(self)`, an f-string `{self}`), which reads the
    dunders it runs (`_BUILTIN_DUNDERS`); and `isinstance(self, C)`, which reads none. Any other
    use of the receiver (passed to another function, `getattr` with a computed name, which the
    walk also refuses, `vars(self)`, an operator, `type(self)(...)`, a rebinding) is dynamic. A
    method that takes the instance with only `*args` receives it unnamed, which is dynamic too."""
    function = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if function is None:
        return [], False
    parameters = [*function.args.posonlyargs, *function.args.args]
    if takes_self and not parameters and function.args.vararg is not None:
        return [], True
    receiver = parameters[0].arg if takes_self and parameters else None
    parents = {child: node for node in ast.walk(function) for child in ast.iter_child_nodes(node)}
    reads = list[tuple[bool, str]]()
    for node in ast.walk(function):
        if isinstance(node, ast.Attribute) and _is_bare_super_call(node.value):
            reads.append((True, node.attr))
            continue
        if not (isinstance(node, ast.Name) and node.id == receiver):
            continue
        if not isinstance(node.ctx, ast.Load):
            return reads, True
        found = _receiver_use(node, parents)
        if found is None:
            return reads, True
        reads.extend(found)
    return reads, False


def _is_bare_super_call(node: ast.AST) -> bool:
    """Whether `node` is `super()` with no arguments."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "super"
        and not node.args
    )


def _receiver_use(
    node: ast.Name, parents: Mapping[ast.AST, ast.AST]
) -> list[tuple[bool, str]] | None:
    """The members one load of the receiver reads, as `(via_super, name)`, or None for a
    dynamic use (`_member_reads`)."""
    parent = parents.get(node)
    if isinstance(parent, ast.Attribute) and parent.value is node:
        if parent.attr != "__class__":
            return [(False, parent.attr)]
        return _class_attribute_use(parent, parents)
    if isinstance(parent, ast.FormattedValue) and parent.value is node:
        builtin = _FSTRING_CONVERSIONS.get(parent.conversion, "format")
        return [(False, name) for name in _BUILTIN_DUNDERS[builtin]]
    if not (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name)):
        return None
    if node not in parent.args or parent.keywords:
        return None
    builtin = parent.func.id
    if builtin in _MEMBERLESS_BUILTINS:
        return []
    if builtin in _NAMED_ATTRIBUTE_BUILTINS and parent.args[0] is node:
        name = parent.args[1] if len(parent.args) > 1 else None
        if isinstance(name, ast.Constant) and isinstance(name.value, str):
            return [(False, name.value)]
        return None
    if builtin == "type" and len(parent.args) == 1:
        return _class_attribute_use(parent, parents)
    if builtin == "super" and len(parent.args) == 2 and parent.args[1] is node:
        holder = parents.get(parent)
        if isinstance(holder, ast.Attribute) and holder.value is parent:
            return [(True, holder.attr)]
        return None
    if builtin in _BUILTIN_DUNDERS and len(parent.args) == 1:
        return [(False, name) for name in _BUILTIN_DUNDERS[builtin]]
    return None


def _class_attribute_use(
    node: ast.AST, parents: Mapping[ast.AST, ast.AST]
) -> list[tuple[bool, str]] | None:
    """The member `type(self).<name>` or `self.__class__.<name>` reads, `node` being
    `type(self)` or `self.__class__`; None when the class is used otherwise (called, passed
    on), a dynamic use."""
    holder = parents.get(node)
    if isinstance(holder, ast.Attribute) and holder.value is node:
        return [(False, holder.attr)]
    return None


def _is_unsafe_fingerprinting_enabled() -> bool:
    if os.environ.get("PIVOT_UNSAFE_FINGERPRINTING") == "1":
        return True
    try:
        from pivot.config import io

        config = io.get_merged_config()
        return config.core.unsafe_fingerprinting
    except (ImportError, FileNotFoundError, AttributeError):
        # Config not available yet (early in pipeline lifecycle)
        return False
    except Exception:
        _logger.debug("Failed to read unsafe_fingerprinting config", exc_info=True)
        return False


def _check_non_scalar_capture(
    var_name: str, value: Any, stage_name: str, *, parameter: str | None = None
) -> None:
    subject = f"captures non-scalar variable '{var_name}'"
    if parameter is not None:
        subject += f" as the default of parameter '{parameter}'"
    message = (
        f"Stage '{stage_name}': {subject} (type: {type(value).__name__}).\n"
        + "Pivot tracks module-level bool, int, float, str, bytes, and None values, tuples and "
        + "frozensets of them, and compiled regexes. Other values may change without a reliable "
        + "fingerprint, which can cause silent wrong outputs.\n"
        + "Fix: move this value into the stage's StageParams, or read it from a pipeline input.\n"
        + "To suppress: set core.unsafe_fingerprinting=true or PIVOT_UNSAFE_FINGERPRINTING=1"
    )
    if _is_unsafe_fingerprinting_enabled():
        _unsafe_capture_warning.set(True)
        if (scope := _stage_scope.get()) is not None:
            scope["unsafe"] = True
        _logger.warning(message)
        return
    errors = _capture_errors.get()
    if errors is None:
        raise exceptions.StageDefinitionError(message)
    errors.append(message)


def _has_attribute(value: Any, name: str) -> bool:
    """`hasattr(value, name)` for a value the walk reads, where an error other than
    AttributeError from the lookup also means no attribute.

    Python calls an object's `__getattr__` only for a name its class and `__dict__` lack. One
    that raises KeyError or RuntimeError for that name (a dict-backed config, an `AttrDict`, a
    lazy object) says the attribute is absent, but `hasattr` lets the error through, and the
    walk would stop on it.
    """
    try:
        return hasattr(value, name)
    except Exception:  # the error of the value's own `__getattr__` or property
        return False


def _attribute_or_none(value: Any, name: str) -> Any:
    """`getattr(value, name, None)`, with the errors `_has_attribute` takes as no attribute."""
    try:
        return getattr(value, name, None)
    except Exception:  # the error of the value's own `__getattr__` or property
        return None


# `type`'s own descriptors for a class's MRO and namespace. Read through them, a class's real
# `tp_mro` and `tp_dict` are reached without asking its metaclass: no metaclass
# `__getattribute__`, `__mro__` property or `__dict__` property runs, and a walk over the MRO
# ends at its real length.
_class_mro = cast("Callable[[Any], tuple[type, ...]]", type.__dict__["__mro__"].__get__)
_class_namespace = cast("Callable[[type], Mapping[str, Any]]", type.__dict__["__dict__"].__get__)
# The C descriptors a class keeps for its instances' `__dict__` (a class's getset, a module's
# member); reading the instance dict through one runs no code of the object or its class. An
# entry's type is compared with them by identity: `==` would ask a metaclass `__eq__`.
_GETSET_DESCRIPTOR = types.GetSetDescriptorType
_MEMBER_DESCRIPTOR = types.MemberDescriptorType
# `type`'s own C member for a class's instance-dict offset: 0 when its instances have no
# `__dict__` (a `__slots__` class without one, a namedtuple), -1 for a managed dict. Read
# through it, no metaclass code runs.
_class_dictoffset = cast("Callable[[Any], int]", type.__dict__["__dictoffset__"].__get__)
# The generic C attribute lookup (`object.__getattribute__`): the type's MRO, then the
# instance dict, and no `__getattr__` fallback.
_generic_getattribute = object.__getattribute__


class _DumpRoute(enum.Enum):
    """How the canonical form decides whether a value is dumped by its `model_dump`."""

    # It may answer names it does not keep, and no class keeps `model_dump`: asked by attribute
    # lookup, where an error other than AttributeError is the object's own "absent" signal (a
    # `__getattr__` that raises KeyError or TypeError for a name it lacks) and means no.
    ASK = "ask"
    # It may answer names it does not keep, and a class keeps `model_dump`: the attribute exists,
    # so an error from the lookup (a `model_dump` property that raises) is real and propagates,
    # as it does from main's `hasattr`.
    ASK_KEPT = "ask_kept"
    # It keeps `model_dump` of its own, found with none of its code run.
    STATIC = "static"


# The routes the canonical form compares once per value, read as globals rather than as enum
# members through their class.
_ASK_KEPT = _DumpRoute.ASK_KEPT
_STATIC = _DumpRoute.STATIC


def _keeps_own_attribute(obj: Any, name: str) -> bool:
    """Whether `obj` keeps `name` of its own: in its instance `__dict__`, or in the namespace of
    a class on its real MRO (a slot, a property, a method, a class attribute). Nothing of `obj`,
    its class or its metaclass runs; see `_probe_own_attribute`."""
    return _probe_own_attribute(obj, name, False) is not None


def _read_type_attribute_probe(cls: type[Any], name: str) -> _TypeAttributeProbe:
    """Read `_probe_own_attribute`'s real-MRO facts without executing class-side code."""
    forwards = False
    keeps = False
    instance_dict: Any = None
    reports_class = False
    for klass in _class_mro(cls):
        namespace = _class_namespace(klass)
        if klass is not object:
            reports_class = reports_class or (
                "__class__" in namespace or "__getattribute__" in namespace
            )
        forwards = forwards or "__getattr__" in namespace
        if name in namespace:
            keeps = True
        elif instance_dict is None and "__dict__" in namespace:
            entry = namespace["__dict__"]
            if (
                type(entry) is _GETSET_DESCRIPTOR or type(entry) is _MEMBER_DESCRIPTOR
            ) and entry.__objclass__ is klass:
                instance_dict = entry.__get__
    has_instance_dict = instance_dict is not None or (not keeps and bool(_class_dictoffset(cls)))
    return forwards, keeps, instance_dict, has_instance_dict, reports_class


def _type_attribute_probe(cls: type[Any], name: str) -> _TypeAttributeProbe:
    """Read the static part of `_probe_own_attribute` once per real type and fingerprint pass.

    The cache is keyed by the type's id and holds the type strongly, so it cannot be reused. It
    never hashes the type: a metaclass may define `__hash__` or `__eq__` as user code. The
    instance-dict read and a reported `__class__` remain per value.
    """
    memo = _type_memo.get()
    if memo is None:
        return _read_type_attribute_probe(cls, name)
    facts = memo[0]
    key = id(cls)
    entry = facts.get(key)
    if entry is None:
        probes: dict[str, _TypeAttributeProbe] = {}
        facts[key] = (cls, probes)
    else:
        probes = entry[1]
    result = probes.get(name)
    if result is None:
        result = _read_type_attribute_probe(cls, name)
        probes[name] = result
    return result


def _probe_own_attribute(
    obj: Any, name: str, forwarding: bool, facts: _TypeAttributeProbe | None = None
) -> _DumpRoute | None:
    """`STATIC` when `obj` keeps `name` of its own; with `forwarding`, `ASK` or `ASK_KEPT` first
    when `obj` may answer names it does not keep; otherwise None.

    The real MRO and each class namespace are read through `type`'s own descriptors
    (`_class_mro`, `_class_namespace`), and types are compared by identity, so no metaclass
    `__getattribute__`, `__mro__` property, `__eq__` or `__hash__` is asked. That static part is
    memoized by real type within one `fingerprint_pass()`; the memo holds the type strongly and
    ends with the pass. The instance `__dict__` is still read from each value through the C
    descriptor a class on that MRO keeps for it. Without one, an instance of a class with no
    dict offset (a `__slots__` class without `__dict__`, a namedtuple) has no dict, and the
    answer is no; any other (a class whose `__dict__` is a property) is asked through
    `_generic_getattribute`, the generic C lookup main's `hasattr` makes minus the `__getattr__`
    fallback. It is reached only after the static walk has shown that no class on the MRO holds
    `name`, so it finds no descriptor to run and reads the instance dict the C way, past any
    `__dict__` property. An object that answers every name it lacks (a `unittest.mock.MagicMock`,
    an attribute tree) does not read as keeping it. Only AttributeError means no; any other
    error (a dict key's own `__eq__`) propagates, as it does from main's `hasattr`.

    With `forwarding`, a value whose real class defines `__getattr__` on its MRO (a pydantic
    model, whose `BaseModel` defines it; a hand-written forwarder that delegates to a target held
    in an attribute, a slot or a closure; celery's `local.Proxy`, `lazy_object_proxy`, wrapt), or
    that reports a class other than `type(obj)` as `__class__` (a proxy that forwards through
    `__getattribute__`), keeps its data where no static read reaches it, and the canonical form
    asks it by attribute lookup: `ASK_KEPT` when a class on the real MRO or on the reported
    class's MRO keeps `name`, `ASK` when none does (see `_DumpRoute`). The reported class is read
    as `isinstance` reads it, by `getattr(obj, "__class__", None)`, where only AttributeError
    means none, and `issubclass(type(reported), type)` asks the real type, not a `__class__`.
    """
    cls: type[Any] = type(obj)  # pyright: ignore[reportUnknownVariableType] - any class
    forwards, keeps, instance_dict, has_instance_dict, reports_class = (
        _type_attribute_probe(cls, name) if facts is None else facts
    )
    if forwarding:
        if forwards and keeps:
            return _DumpRoute.ASK_KEPT
        # Only a class other than `object` that defines `__class__` or `__getattribute__` can
        # report another class; without one, `obj.__class__` is `type(obj)` and is not read.
        if reports_class:
            reported = getattr(obj, "__class__", None)
            if reported is not cls and issubclass(type(reported), type):
                forwards = True
                if not keeps:
                    reported_class = cast("type[Any]", reported)
                    keeps = _type_attribute_probe(reported_class, name)[1]
        if forwards:
            return _DumpRoute.ASK_KEPT if keeps else _DumpRoute.ASK
    if keeps:
        return _DumpRoute.STATIC
    if instance_dict is not None:
        found = instance_dict(obj)
        if type(found) is dict:
            holds = name in found
        # An instance `__dict__` that is a dict subclass: its own `__contains__` is class-side
        # code, so the dict is asked the C way, as main's attribute lookup asks it. (A class asked
        # here, as `_mark_chain` asks one through `_source_object`, has a `mappingproxy` for its
        # namespace, which the last branch asks with `in`, a C lookup on the class's own dict.)
        elif issubclass(type(found), dict):
            holds = dict.__contains__(found, name)  # pyright: ignore[reportUnknownMemberType] - a dict subclass with unknown item types
        else:
            holds = name in found
        return _DumpRoute.STATIC if holds else None
    if not has_instance_dict:
        return None
    try:
        _generic_getattribute(obj, name)
    except AttributeError:
        return None
    return _DumpRoute.STATIC


def _model_dump_route(
    value: Any, value_type: type[Any], routes: _TypeDumpRoutes | None
) -> _DumpRoute | None:
    """The canonical form's `model_dump` route for a value that is not an exact built-in.

    A class is never dumped, whatever its metaclass reports: `isinstance(value, type)` is asked
    before any probe. It reads the real type, then the reported `__class__`, so a type is
    recorded in `routes` only when no class on its MRO but `object` defines `__class__` or
    `__getattribute__`: then every instance reports its real type, which is no metaclass, and
    the guard answers no for each without being asked again. Such a type is recorded when the
    type alone decides the route (see `_type_memo`): `ASK_KEPT` when its MRO defines
    `__getattr__` and keeps `model_dump`, as `_probe_own_attribute` answers first for it, or
    main's C `hasattr` when it has no class-side hook for `model_dump` either, which reads only
    the real instance dict, as `_probe_own_attribute` does. Later instances skip the probe.
    """
    if isinstance(value, type):
        return None
    facts = _type_attribute_probe(value_type, "model_dump")
    forwards, keeps, _, _, reports_class = facts
    if not reports_class:
        if forwards and keeps:
            if routes is not None:
                routes[id(value_type)] = (value_type, _ASK_KEPT)
            return _ASK_KEPT
        if not (forwards or keeps):
            if routes is not None:
                routes[id(value_type)] = (value_type, None)
            return _STATIC if hasattr(value, "model_dump") else None
    return _probe_own_attribute(value, "model_dump", True, facts)


def _is_user_class_instance(value: Any) -> bool:
    """Return whether `value` is an instance of a user-defined class."""
    return not isinstance(value, type) and is_user_code(type(value))


def _is_callable_instance(value: Any) -> bool:
    """Return whether `value` is a callable object rather than a function or class.

    A non-user module's own public callable (`numpy.log` is a `ufunc` instance) is a library
    function, not module-level state, so it is not a callable instance here.
    """
    if (
        not callable(value)
        or isinstance(value, (functools.partial, type))
        or inspect.isroutine(value)
    ):
        return False
    return not _is_library_export(value)


def _is_library_export(value: Any) -> bool:
    """Return whether a non-user module binds `value` under its own `__name__`."""
    return _library_module(value) is not None


def _library_module(value: Any) -> types.ModuleType | None:
    """Return the non-user module that binds `value` under its own `__name__`, if any.

    The module named by `__module__` is checked. An object without one (scipy.special's ufuncs,
    such as `expit`) is looked up in every loaded module; `np.frompyfunc` products stay
    refused because no module binds their generated names.
    """
    name = _attribute_or_none(value, "__name__")
    if not isinstance(name, str):
        return None
    module_name = getattr(value, "__module__", None)
    candidates: list[types.ModuleType | None]
    if isinstance(module_name, str):
        candidates = [sys.modules.get(module_name)]
    else:
        candidates = list(sys.modules.values())
    for module in candidates:
        if (
            isinstance(module, types.ModuleType)
            and module.__dict__.get(name) is value
            and not is_user_code(module)
        ):
            return module
    return None


def _library_name(value: Any) -> str:
    """Name a library callable by where it is defined: `<module>.<qualname>`.

    `np.mean` is `numpy.mean` and `from numpy import median` is `numpy.median`, so a binding
    that now holds another library function reads as changed. A method bound to a class or a
    scalar carries it in brackets (`builtins.str.join[',']`): the bound object is part of what
    the callable does. A marked class-based decorator without names of its own (one with
    `__slots__` that `functools.update_wrapper` did not make), or a library wrapper over one, is
    named by the helper at the end of its chain: the module-level function that
    `@fingerprint_by_name` checked.
    """
    bound = getattr(value, "__self__", None)
    named = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if not named and _recorded_by_name(value):
        return _library_name(unwrap(value))
    qualname = named or type(value).__qualname__
    module = getattr(value, "__module__", None)
    if not isinstance(module, str):
        owner_class = getattr(value, "__objclass__", None)
        if isinstance(bound, type):
            module = bound.__module__
        elif isinstance(owner_class, type):
            module = owner_class.__module__
        elif bound is not None and not isinstance(bound, types.ModuleType):
            module = type(bound).__module__
        else:
            exporter = _library_module(value)
            module = exporter.__name__ if exporter is not None else type(value).__module__
    name = f"{module}.{qualname}"
    if isinstance(bound, type):
        return f"{name}[{_definition_ref(bound)}]"
    if isinstance(bound, (bool, int, float, str, bytes)):
        return f"{name}[{bound!r}]"
    return name


def _carries_instance(value: Any) -> bool:
    """Return whether `value` is a method bound to an object whose state it may read.

    `Scaler(3).apply` and `np.random.default_rng(SEED).normal` carry an instance that no code
    hash records. A method bound to a module, a class, a scalar or a Logger does not, and
    neither does a library module's own export (`np.random.normal`, `random.random`), whose
    state belongs to the library.
    """
    bound = _attribute_or_none(value, "__self__")
    if bound is None or isinstance(
        bound, (types.ModuleType, type, bool, int, float, str, bytes, logging.Logger)
    ):
        return False
    return not _is_library_export(value)


def _is_library_callable(value: Any) -> bool:
    """Return whether `value` is a library routine, or a library module's own callable export.

    These are named by `_library_name`. A class is not: it stays a leaf of the canonical form.
    """
    if isinstance(value, type) or not callable(value) or is_user_code(value):
        return False
    return inspect.isroutine(value) or _is_library_export(value)


def _process_closure_values(
    values: Mapping[str, Any],
    func: Callable[..., Any],
    manifest: dict[str, str],
    visited: set[int],
    *,
    prefix: str,
) -> None:
    """Process the globals (`prefix` "g", dunders skipped) or closure cells ("c") `func` reads.

    A global is keyed by its module (`<module>.NAME`); a cell by the function that owns it
    (`_walk_owner`), since two closures from one factory hold different values under one name.
    Names are walked in sorted order: getclosurevars collects globals in a set.
    """
    stage_name = _current_stage_name.get() or getattr(func, "__name__", "<unknown>")
    owner = _walk_owner(func)
    for name, value in sorted(values.items()):
        if prefix == "g" and name.startswith("__"):
            continue
        binding_ref = _binding_ref(func, name) if prefix == "g" else f"{owner}.{name}"
        step = f"{prefix}:{name}"
        _record_read(func, step)
        with _walking(step):
            _process_captured_value(
                name, binding_ref, value, func, manifest, visited, stage_name, chain=name
            )


def _walk_owner(func: Callable[..., Any]) -> str:
    """Return the manifest reference under which `func`'s closure cells and defaults are keyed.

    That is the reference `func` itself was keyed by (`_add_callable_to_manifest`), or its
    definition for the stage function.
    """
    return _owner_ref.get() or _definition_ref(func)


def _process_captured_value(
    name: str,
    binding_ref: str,
    value: Any,
    func: Callable[..., Any],
    manifest: dict[str, str],
    visited: set[int],
    stage_name: str,
    *,
    chain: str | None,
    parameter: str | None = None,
) -> None:
    """Fingerprint one value `func` reads, or refuse it as a non-scalar capture.

    `chain` is the expression `func` reaches the value by (`Config`, `settings.Config`, a
    parameter name): the attributes read on a user class or user module are followed from it.
    None means the value is not reached by a name, and its attributes are not followed.
    `parameter` names the parameter when the value is a default argument.
    """
    match _value_kind(value):
        case _ValueKind.PARTIAL:
            _process_partial_dependency(binding_ref, value, manifest, visited)
        case _ValueKind.USER:
            _process_callable_dependency(binding_ref, value, manifest, visited)
            if isinstance(value, type) and chain is not None:
                _process_class_reads(chain, value, func, manifest, visited, stage_name)
        case _ValueKind.MODULE:
            if not is_user_code(value):
                # Its code is library code, not hashed; a worker imports it by name.
                _mark_by_name(value.__name__, manifest)
            elif chain is not None:
                _process_module_dependency(chain, value, func, manifest, visited)
        case _ValueKind.SCALAR:
            text = _immutable_value_text(value)
            assert text is not None
            manifest[f"const:{binding_ref}"] = text
        case _ValueKind.REFUSED:
            _check_non_scalar_capture(name, value, stage_name, parameter=parameter)
        case _ValueKind.CLASS_ABOVE_MARK:
            _refuse_class_above_mark(value)
        case _ValueKind.EXTERNAL:
            # A library function or class is named by where it is defined; its code is not
            # hashed, but rebinding the name to another one must change the manifest.
            manifest[f"ext:{binding_ref}"] = _library_name(value)
            _mark_by_name(getattr(value, "__module__", None), manifest)
        case _ValueKind.LOGGER:
            return


def _attributes_read(func: Callable[..., Any], chain: str) -> list[str]:
    """Each attribute `func` reads directly on the value it reaches as `chain`, in source order.

    For `chain` "settings", `settings.AGG(...)` and `settings.Config.SCALE` give `AGG` and
    `Config`; what is read on `Config` is found from the chain "settings.Config". The source
    read stops above a mark (`_source_object`), so a user decorator above a marked function
    gives the attributes its own lines read.
    """
    target = _source_object(func)
    if target is None:
        return []
    prefix = f"{chain}."
    found = dict[str, None]()
    for owner, path in ast_utils.extract_module_attr_usage(target):
        expression = f"{owner}.{path}"
        if expression.startswith(prefix):
            found[expression.removeprefix(prefix).split(".", 1)[0]] = None
    return list(found)


def _process_class_reads(
    chain: str,
    cls: type,
    func: Callable[..., Any],
    manifest: dict[str, str],
    visited: set[int],
    stage_name: str,
) -> None:
    """Track each attribute `func` reads from user class `cls`, reached as `chain`.

    `chain` is `Config` for a global, `settings.Config` through a module, or a parameter name.
    The class AST hash does not see a class attribute computed at import, so each attribute
    `func` reads that the user wrote in a class body (`_is_user_class_attribute`) gets the rule
    a module-level value gets: a scalar is `const:<class>.<NAME>`, a non-scalar is refused, a
    function or class is walked. A member of a user Enum is recorded by name and value. A
    pydantic or StageParams class is covered by its schema hash, field defaults included. A
    cache hit re-reads each attribute (`ca:NAME`). Any other attribute is left to the class
    AST hash.
    """
    if hasattr(cls, "model_fields"):
        _hash_pydantic_schema(cast("type[_PydanticModelProtocol]", cls), manifest, visited)
        return
    class_ref = _definition_ref(cls)
    for attribute in _attributes_read(func, chain):
        if attribute.startswith("__") or not _is_user_class_attribute(cls, attribute):
            continue
        step = f"ca:{attribute}"
        _record_read(cls, step)
        try:
            value = getattr(cls, attribute)
        except AttributeError:
            continue
        with _walking(step):
            if _is_user_enum_member(value):
                manifest[f"const:{class_ref}.{attribute}"] = _enum_member_text(value)
            else:
                _process_captured_value(
                    f"{class_ref}.{attribute}",
                    f"{class_ref}.{attribute}",
                    value,
                    func,
                    manifest,
                    visited,
                    stage_name,
                    chain=None,
                )


def _is_user_class_attribute(cls: type, attribute: str) -> bool:
    """Whether `attribute` on `cls` is a value the user wrote in a class body.

    The defining class is the first on the MRO that holds the name in its own `__dict__`, and
    it must be user code: a dict that `BaseHTTPRequestHandler` or a sklearn base defines is
    not. Also left out: descriptors other than plain functions (property, classmethod,
    staticmethod, cached_property, a NamedTuple field), the `_sunder_` names the enum
    machinery puts on an Enum class (`_member_names_`, `_value2member_map_`), and the
    `_fields` and `_field_defaults` that NamedTuple and namedtuple put on a tuple subclass.
    """
    owner = next((klass for klass in cls.__mro__ if attribute in vars(klass)), None)
    if owner is None or not is_user_code(owner):
        return False
    raw = vars(owner)[attribute]
    if isinstance(raw, (classmethod, staticmethod)) or (
        _has_attribute(raw, "__get__") and not callable(raw)
    ):
        return False
    sunder = len(attribute) > 2 and attribute[0] == attribute[-1] == "_"
    if issubclass(owner, enum.Enum) and sunder and "__" not in (attribute[:2], attribute[-2:]):
        return False
    return not (issubclass(owner, tuple) and attribute in ("_fields", "_field_defaults"))


def _is_user_enum_member(value: Any) -> bool:
    """Whether `value` is a member of an Enum defined in user code."""
    return isinstance(value, enum.Enum) and is_user_code(type(value))


def _enum_member_text(member: enum.Enum) -> str:
    """An Enum member as its class, name and canonical value: constants of the class body."""
    return f"{type(member).__qualname__}.{member.name}={_serialize_value_for_hash(member.value)}"


def _process_default_values(
    func: Callable[..., Any], manifest: dict[str, str], visited: set[int]
) -> None:
    """Fingerprint default argument values, which no code hash records.

    A default is evaluated once, when its module loads, so `cols=COLUMNS` puts the name
    `COLUMNS` in the AST and never its value. A scalar or a StageParams instance is tracked by
    value: a stage's own params hash does not cover a StageParams default on a helper. An
    ArtifactHandle default, or a list or tuple of them, is skipped: the pipeline wires it as a
    real input, which is hashed as a dependency. Any other value is treated like a captured
    module value, unless its default is a literal that the function's AST hash already
    records, such as `keys=("a", "b")`.
    """
    defaults = _function_defaults(func)
    if not defaults:
        return
    from pivot import compose, stage_def

    stage_name = _current_stage_name.get() or getattr(func, "__name__", "<unknown>")
    func_ref = _walk_owner(func)
    expressions: dict[str, ast.expr] | None = None
    for parameter, value in defaults:
        binding_ref = f"{func_ref}.{parameter}"
        step = f"d:{parameter}"
        _record_read(func, step)
        if isinstance(value, _SCALAR_TYPES):
            manifest[f"default:{binding_ref}"] = repr(value)
            continue
        if isinstance(value, stage_def.StageParams):
            serialized = _serialize_value_for_hash(value)
            manifest[f"default:{binding_ref}"] = xxhash.xxh64(serialized.encode()).hexdigest()
            continue
        if _is_artifact_handles(value, compose.ArtifactHandle):
            continue
        if expressions is None:
            expressions = _default_expressions(func)
        expression = expressions.get(parameter)
        if expression is not None and _is_literal(expression):
            continue
        text = _immutable_value_text(value)
        if text is not None:
            manifest[f"default:{binding_ref}"] = text
            continue
        source = ast.unparse(expression) if expression is not None else parameter
        with _walking(step):
            _process_captured_value(
                source,
                binding_ref,
                value,
                func,
                manifest,
                visited,
                stage_name,
                chain=parameter,
                parameter=parameter,
            )


def _is_artifact_handles(value: object, handle_type: type) -> bool:
    """Return whether `value` is a handle, or a non-empty list or tuple of handles."""
    if isinstance(value, handle_type):
        return True
    if isinstance(value, (list, tuple)):
        items = cast("list[object] | tuple[object, ...]", value)
        return bool(items) and all(isinstance(item, handle_type) for item in items)
    return False


def _function_defaults(func: Callable[..., Any]) -> list[tuple[str, Any]]:
    """Return (parameter, default value) for each parameter of `func` that has a default."""
    code = getattr(func, "__code__", None)
    if not isinstance(code, types.CodeType):
        return []
    positional: tuple[Any, ...] = getattr(func, "__defaults__", None) or ()
    keyword: dict[str, Any] = getattr(func, "__kwdefaults__", None) or {}
    names = code.co_varnames[: code.co_argcount]
    pairs = list(zip(names[len(names) - len(positional) :], positional, strict=True))
    pairs.extend(keyword.items())
    return pairs


def _default_expressions(func: Callable[..., Any]) -> dict[str, ast.expr]:
    """Map each defaulted parameter to its default expression in the hashed source.

    Empty when the code hash does not come from this function's own source: there is no
    source, or `func` is a `functools.wraps` wrapper, whose hash is bytecode, which holds no
    defaults. Every default is then checked by value.
    """
    if hasattr(func, "__wrapped__"):
        return {}
    try:
        node = ast_utils.get_function_ast(func)
    except (ValueError, SyntaxError):
        return {}
    if node.name != getattr(func, "__name__", None):
        return {}
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaulted = positional[len(positional) - len(args.defaults) :]
    expressions = {arg.arg: default for arg, default in zip(defaulted, args.defaults, strict=True)}
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        if default is not None:
            expressions[arg.arg] = default
    return expressions


def _is_literal(expression: ast.expr) -> bool:
    """Return whether `expression` is a Python literal, fully recorded by the AST."""
    try:
        ast.literal_eval(expression)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False
    return True


def _process_callable_dependency(
    binding_ref: str, func: Callable[..., Any], manifest: dict[str, str], visited: set[int]
) -> None:
    """Process a callable dependency and add to manifest."""
    key = f"class:{_unique_ref(func)}" if isinstance(func, type) else None
    _add_callable_to_manifest(
        key or f"func:{_callable_ref(func, binding_ref)}", func, manifest, visited
    )


def _process_partial_dependency(
    binding_ref: str,
    partial_obj: Any,
    manifest: dict[str, str],
    visited: set[int],
) -> None:
    """Process functools.partial: hash underlying func and bound arguments.

    The function is classified as any read value is (`_value_kind`): a method bound to an
    instance (`partial(Scaler(3).apply)`, `partial(default_rng(SEED).normal)`) or a callable
    instance is refused like a bare one, since no hash records the state it carries.
    """
    (args_str, kwargs_str), held = _serialize_collecting(_partial_parts(partial_obj))
    manifest[f"partial:{binding_ref}.args"] = xxhash.xxh64(args_str.encode()).hexdigest()
    manifest[f"partial:{binding_ref}.kwargs"] = xxhash.xxh64(kwargs_str.encode()).hexdigest()
    if held:
        _merge_held_loaders(manifest, held, read="hl")

    underlying = partial_obj.func
    _record_read(partial_obj, "pf")
    kind = _value_kind(underlying)
    if kind is _ValueKind.REFUSED:
        with _walking("pf"):
            _check_non_scalar_capture(
                f"{binding_ref}.func", underlying, _current_stage_name.get() or "<unknown>"
            )
    elif kind is _ValueKind.CLASS_ABOVE_MARK:
        with _walking("pf"):
            _refuse_class_above_mark(underlying)
    elif callable(underlying) and is_user_code(underlying):
        callable_ref = _callable_ref(underlying, f"{binding_ref}.func")
        with _walking("pf"):
            _add_callable_to_manifest(f"func:{callable_ref}", underlying, manifest, visited)
    elif callable(underlying):
        manifest[f"ext:{binding_ref}.func"] = _library_name(underlying)
        _mark_by_name(getattr(underlying, "__module__", None), manifest)
    _process_partial_arguments(binding_ref, partial_obj, manifest, visited)


def _process_partial_arguments(
    binding_ref: str,
    partial_obj: Any,
    manifest: dict[str, str],
    visited: set[int],
) -> None:
    """Walk the values bound in a partial's arguments (`_process_bound_values`)."""
    for index, item in enumerate(partial_obj.args):
        _process_bound_member(
            f"{binding_ref}.args[{index}]", partial_obj, f"pa:{index}", item, manifest, visited
        )
    for keyword, item in partial_obj.keywords.items():
        _process_bound_member(
            f"{binding_ref}.kwargs[{keyword}]",
            partial_obj,
            f"pk:{keyword}",
            item,
            manifest,
            visited,
        )


def _process_bound_member(
    binding_ref: str,
    owner: Any,
    step: str,
    value: Any,
    manifest: dict[str, str],
    visited: set[int],
) -> None:
    """Walk `value`, which `step` reads from `owner`, as a bound value."""
    _record_read(owner, step)
    with _walking(step):
        _process_bound_values(binding_ref, value, manifest, visited)


def _process_bound_values(
    binding_ref: str, value: Any, manifest: dict[str, str], visited: set[int]
) -> None:
    """Walk each user function and user class bound inside `value` as a stage helper.

    Loader config and `functools.partial` arguments hash a function by its name and code
    only, so the module values it reads would otherwise go unseen. The walk descends into
    partials, lists, tuples, sets and dict values. Each user function, including one behind
    a `functools.cache` wrapper, and each user class gets the walk a stage helper gets:
    globals, closure values, defaults, and the non-scalar refusal. An instance of a user
    class with `__call__`, or a method bound to an instance (`Scaler(3).apply`,
    `default_rng(SEED).normal`), is refused, as it is when a stage reads one: what it reads
    from its instance is not walked, and its repr does not record that state. A callable on a
    chain that holds a mark takes the mark decision (`_mark_decision`) before any unwrap: one
    recorded by name, or one with a user class instance above the mark, is left to the
    canonical form of the same binding, which every caller computes (`loader_config_text`, the
    partial's `args` and `keywords`): it names the first and refuses the second. A pivot loader
    among them is fingerprinted by its caller (`_merge_held_loaders`).
    """
    if isinstance(value, functools.partial):
        partial_obj = cast("functools.partial[Any]", value)
        _process_bound_member(
            f"{binding_ref}.func", partial_obj, "pf", partial_obj.func, manifest, visited
        )
        _process_partial_arguments(binding_ref, partial_obj, manifest, visited)
    elif isinstance(value, (list, tuple)):
        items = cast("list[Any] | tuple[Any, ...]", value)
        for index, item in enumerate(items):
            _process_bound_member(
                f"{binding_ref}[{index}]", items, f"i:{index}", item, manifest, visited
            )
    elif isinstance(value, (set, frozenset)):
        # Iteration order follows the hash seed, so walk members in canonical order.
        members = sorted(cast("set[Any] | frozenset[Any]", value), key=_serialize_value_for_hash)
        for index, item in enumerate(members):
            _process_bound_member(
                f"{binding_ref}{{{index}}}", value, f"s:{index}", item, manifest, visited
            )
    elif isinstance(value, dict):
        for key, item in cast("dict[Any, Any]", value).items():
            _process_bound_member(
                f"{binding_ref}[{key!r}]", value, f"k:{key!r}", item, manifest, visited
            )
    elif isinstance(value, type):
        if is_user_code(value):
            _process_callable_dependency(binding_ref, value, manifest, visited)
            if hasattr(value, "model_fields"):
                _hash_pydantic_schema(
                    cast("type[_PydanticModelProtocol]", value), manifest, visited
                )
    elif not callable(value):
        return
    elif (decision := _mark_decision(value)) is _MarkDecision.REFUSED or (
        decision is _MarkDecision.BY_NAME and not _carries_instance(value)
    ):
        # The binding's canonical form decides (`_canonicalize_value_for_hash`).
        return
    elif _is_user_class_instance(value) or _carries_instance(value):
        _check_non_scalar_capture(binding_ref, value, _current_stage_name.get() or "<unknown>")
    elif inspect.isfunction(unwrap(value, stop=_stop_at_mark)) and is_user_code(value):
        _process_callable_dependency(binding_ref, value, manifest, visited)


def _resolve_annotations_individually(func: Callable[..., Any]) -> dict[str, Any]:
    raw = getattr(func, "__annotations__", None)
    if not raw:
        return {}

    globalns = getattr(func, "__globals__", {})
    resolved = dict[str, Any]()

    for name, annotation in raw.items():
        if isinstance(annotation, type):
            resolved[name] = annotation
        elif isinstance(annotation, str):
            try:
                result = eval(annotation, globalns)
            except Exception:
                continue
            if isinstance(result, str) and result in globalns:
                result = globalns[result]
            resolved[name] = result
        else:
            resolved[name] = annotation

    return resolved


def _resolved_type_hints(func: Callable[..., Any]) -> dict[str, Any]:
    """Return `func`'s type hints, resolved once per function per process."""
    _t = metrics.start()
    # Check cache first. TypeError raised for non-weakly-referenceable functions (builtins).
    try:
        hints = _get_type_hints_cache.get(func)
    except TypeError:
        hints = None  # Not in cache and can't be cached

    if hints is None:
        # The walk enters no marked function, so a mark on `func`'s chain is below it.
        marked = _mark_on_chain(func)
        if marked is not None and getattr(func, "__annotations__", None) is getattr(
            marked, "__annotations__", None
        ):
            # `functools.wraps` copied these annotations from the marked function below `func`:
            # they are that function's type hints, which the walk does not read.
            hints = dict[str, Any]()
        else:
            try:
                # With a mark below, resolve in `func`'s own globals: by default
                # `get_type_hints` follows `__wrapped__` to the globals at the end of the chain.
                hints = typing.get_type_hints(
                    func, globalns=None if marked is None else getattr(func, "__globals__", {})
                )
            except Exception:
                hints = _resolve_annotations_individually(func)
        # Cache result if function is weakly referenceable
        with contextlib.suppress(TypeError):
            _get_type_hints_cache[func] = hints

    metrics.end("fingerprint.get_type_hints", _t)
    return hints


def _hint_classes(func: Callable[..., Any]) -> list[type]:
    """Return each user class in `func`'s type hints, including the origin of a generic
    (`MyContainer[int]`). Within one `fingerprint_pass()` each function's hints are collected
    once.
    """

    def collect() -> list[type]:
        found = list[type]()
        for hint in _resolved_type_hints(func).values():
            _collect_hint_classes(hint, found)
        return found

    return _memoized("hints", func, collect)


def _collect_hint_classes(hint: Any, found: list[type]) -> None:
    """Add the user classes in one type hint to `found`, recursing into generics."""
    if hint is type(None):
        return
    origin = typing.get_origin(hint)
    if origin is not None:
        # Track user-defined generic classes (e.g., MyGeneric[int] -> hash MyGeneric)
        if isinstance(origin, type) and is_user_code(origin):
            found.append(cast("type[Any]", origin))
        for arg in typing.get_args(hint):
            _collect_hint_classes(arg, found)
        return
    if isinstance(hint, type) and is_user_code(hint):
        found.append(cast("type[Any]", hint))


def _process_type_hint_dependencies(
    func: Callable[..., Any], manifest: dict[str, str], visited: set[int]
) -> None:
    """Process user-defined classes in type hints, including Pydantic model defaults."""
    _record_read(func, "hints")
    for index, hint_class in enumerate(_hint_classes(func)):
        with _walking(f"hc:{index}"):
            key = f"class:{_unique_ref(hint_class)}"
            if key not in manifest:
                _add_callable_to_manifest(key, hint_class, manifest, visited)
            # Always hash Pydantic schema (even if class was already added via closure vars)
            if hasattr(hint_class, "model_fields"):
                _hash_pydantic_schema(
                    cast("type[_PydanticModelProtocol]", hint_class), manifest, visited
                )


def _strip_schema_metadata(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip title/description to avoid false invalidation from docstring changes."""
    stripped = {k: v for k, v in schema.items() if k not in ("title", "description")}
    defs_raw = stripped.get("$defs")
    if isinstance(defs_raw, dict):
        defs = cast("dict[str, Any]", defs_raw)
        cleaned_defs = dict[str, Any]()
        for name, definition in defs.items():
            if isinstance(definition, dict):
                cleaned_defs[name] = _strip_schema_metadata(cast("dict[str, Any]", definition))
            else:
                cleaned_defs[name] = definition
        stripped["$defs"] = cleaned_defs
    properties_raw = stripped.get("properties")
    if isinstance(properties_raw, dict):
        properties = cast("dict[str, Any]", properties_raw)
        cleaned_properties = dict[str, Any]()
        for name, prop in properties.items():
            if isinstance(prop, dict):
                cleaned_properties[name] = _strip_schema_metadata(cast("dict[str, Any]", prop))
            else:
                cleaned_properties[name] = prop
        stripped["properties"] = cleaned_properties
    for key in ("items", "anyOf", "allOf", "oneOf"):
        val = stripped.get(key)
        if isinstance(val, dict):
            stripped[key] = _strip_schema_metadata(cast("dict[str, Any]", val))
        elif isinstance(val, list):
            cleaned_items = list[Any]()
            items = cast("list[Any]", val)
            for item in items:
                if isinstance(item, dict):
                    cleaned_items.append(_strip_schema_metadata(cast("dict[str, Any]", item)))
                else:
                    cleaned_items.append(item)
            stripped[key] = cleaned_items
    return stripped


def _hash_pydantic_schema(
    model: type[_PydanticModelProtocol],
    manifest: dict[str, str],
    visited: set[int],
) -> None:
    _record_read(model, "schema")
    schema_key = f"schema:{_unique_ref(model)}"
    if schema_key in manifest:
        return
    # Insert placeholder to prevent infinite recursion on self-referential models
    # (e.g., class Node(BaseModel): children: list["Node"])
    manifest[schema_key] = "<pending>"
    for index, field_class in enumerate(_model_field_classes(model)):
        with _walking(f"fc:{index}"):
            key = f"class:{_unique_ref(field_class)}"
            if key not in manifest:
                _add_callable_to_manifest(key, field_class, manifest, visited)
            if hasattr(field_class, "model_fields"):
                _hash_pydantic_schema(
                    cast("type[_PydanticModelProtocol]", field_class), manifest, visited
                )
    manifest[schema_key] = _schema_hash(model)


def _schema_hash(model: type[_PydanticModelProtocol]) -> str:
    """Hash a pydantic model's JSON schema and field defaults, once per class per process.

    A field default is evaluated when the class is created, so it is a live value: a cache hit
    reads it again through this hash. A class is created once per import, so the hash is
    memoized by class identity.
    """
    try:
        cached = _schema_hash_cache.get(model)
    except TypeError:
        cached = None
    if cached is not None:
        return cached
    from pivot import parameters

    schema_raw = model.model_json_schema(schema_generator=parameters.SortedSetJsonSchema)
    schema = _strip_schema_metadata(schema_raw)
    model_fields = cast("Mapping[str, Any]", getattr(model, "model_fields", {}))
    default_hashes = dict[str, str]()
    for field_name, field_info in model_fields.items():
        default = getattr(field_info, "default", None)
        default_factory = getattr(field_info, "default_factory", None)
        if type(default).__name__ == "PydanticUndefinedType":
            if default_factory is not None and callable(default_factory):
                default_hashes[field_name] = hash_function_ast(default_factory)
            continue
        value_str = _serialize_value_for_hash(default)
        default_hashes[field_name] = xxhash.xxh64(value_str.encode()).hexdigest()

    payload = {"schema": schema, "defaults": default_hashes}
    result = xxhash.xxh64(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    with contextlib.suppress(TypeError):
        _schema_hash_cache[model] = result
    return result


def _model_field_classes(model: Any) -> list[type]:
    """Return each user class in a pydantic model's field annotations, in field order."""
    model_fields = cast("Mapping[str, Any]", getattr(model, "model_fields", {}))
    found = list[type]()
    for field_info in model_fields.values():
        annotation = getattr(field_info, "annotation", None)
        if annotation is not None:
            found.extend(_annotation_classes(annotation))
    return found


def _annotation_classes(annotation: Any) -> list[type]:
    """Return the user classes in a field annotation, recursing into generic arguments."""
    if typing.get_origin(annotation) is not None:
        return [cls for arg in typing.get_args(annotation) for cls in _annotation_classes(arg)]
    if isinstance(annotation, type) and is_user_code(annotation):
        return [cast("type[Any]", annotation)]
    return []


# The exact built-in containers the canonical form takes without a probe (see its shortcut).
_BUILTIN_TYPES = frozenset({dict, list, tuple, set, frozenset, bytes})


def _serialize_value_for_hash(value: Any) -> str:
    """Serialize a value to a stable string for hashing."""
    return json.dumps(_canonicalize_value_for_hash(value))


def _canonicalize_value_for_hash(value: Any, active: tuple[int, ...] = ()) -> Any:
    """Return a JSON-ready form of `value` that does not depend on PYTHONHASHSEED.

    Collections become `[type, items]` nodes, so a set stays distinct from a list or from a
    string that looks like one. Set members and dict entries are sorted by their full
    canonical JSON text: a set's iteration order, and a dict built from a set, follow the
    hash seed. A Python function becomes `["function", function_identity]`, closure values
    and defaults included, because its `repr` holds a memory address and two closures from one
    factory share a definition. `active` holds the ids of the functions, containers and
    objects on the path being canonicalised: one met again (`self.me = self`, a list holding
    itself) becomes `["cycle", N]`, N steps back up the path, the same in every process. A
    library function, or a library module's own callable export,
    becomes `["external", _library_name]`: its repr can hold an address (`np.mean`). A
    `functools.partial` is canonicalised through its function and arguments. An instance is
    dumped by its `model_dump` as main dumps it, by one of two routes. A value that may answer
    names it does not keep, because its class defines `__getattr__` on its MRO (a pydantic model,
    a forwarder) or it reports a class other than its own type as `__class__` (a transparent
    proxy), is asked by attribute lookup, as main asks. On ASK, where no class keeps
    `model_dump`, it is dumped only when the answer is callable and returns an exact dict, so an
    object that answers every name with junk (a `MagicMock`, an attribute tree) gets the plain
    form. On ASK_KEPT, where a class keeps `model_dump`, the callable result is dumped as main
    dumps it; RootModel returns its root, which can be a list, a secret or a scalar. An error
    other than AttributeError from that lookup propagates, as from main's `hasattr`, when a class
    on the value's MRO or on its reported class's MRO keeps `model_dump`; when none does, it is
    the object's "absent" signal and means no. Beyond the `__class__` read each `isinstance`
    check makes, as main's do, that lookup is the one place the probe runs the value's own code.
    Any other value is dumped when it keeps `model_dump` of its own (`_keeps_own_attribute`,
    which runs none of its code). The dump is in python mode, which keeps each secret
    (`SecretStr`, `SecretBytes`, `Secret[T]`) as its own object; a secret becomes
    `["secret", digest]`, the keyed digest the lock file records for it
    (`parameters.secret_digest`), so changing it
    changes the hash and no fingerprint holds its value. A dataclass instance becomes
    `[module.qualname, canonical({field: value})]`, field by field, so a secret inside one
    reaches that leaf rather than the generated repr's mask. An object whose class keeps
    `object.__repr__` (which holds the address) and that has a `__dict__` becomes
    `[module.qualname, canonical(vars(obj))]`, without the `@fingerprint_by_name` mark that
    `functools.update_wrapper` copies into a wrapper when `vars(obj)` is an exact dict (a mark
    copied into a dict subclass stays, and is canonicalised by the marked object's name, so the
    form is stable). A compiled regex becomes `["re.Pattern", canonical(pattern), flags]`, so
    its full pattern is recorded. Other leaves keep their `repr`, tagged with their type. A class
    is a leaf, even a pydantic model class, whatever its metaclass reports as `__class__`.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, re.Pattern):
        pattern = cast("re.Pattern[Any]", value)
        return ["re.Pattern", _canonicalize_value_for_hash(pattern.pattern), pattern.flags]
    if id(value) in active:
        cycle = _form_cycle.get()
        if cycle is not None:
            cycle[0] = True
        return ["cycle", len(active) - active.index(id(value))]
    # An exact dict, list, tuple, set, frozenset or bytes skips every branch up to the container
    # branches: an exact instance has no `__dict__`, is not callable, and neither its type nor
    # `object` holds a mark, a `__wrapped__` or a `model_dump`, which a built-in type cannot gain.
    # Scalars, `str` and None returned above. The set lookup hashes the type, so it is reached
    # only for a type whose metaclass is `type` itself, whose `__hash__` and `__eq__` are identity
    # in C (a class cannot override them for itself); a type with another metaclass, whose
    # `__hash__` or `__eq__` may run code or raise, is no built-in and skips the lookup.
    value_type: type[Any] = type(value)  # pyright: ignore[reportUnknownVariableType] - any class
    if type(value_type) is not type or value_type not in _BUILTIN_TYPES:
        # The mark decision. The canonical form is where a value bound in a loader field or a
        # partial argument meets it (`_process_bound_values` leaves both answers here), so it
        # comes before every branch that could name the value without it: `model_dump`, the
        # function identity, and the library-callable test, which would name a `functools.cache`
        # wrapper over a class-based decorator over the mark and drop the class. A chain with a
        # user class instance above the mark is refused. A marked instance of a class-based
        # decorator is named like a marked function: its own `__dict__` is not dumped.
        # `_mark_decision` answers None before any read while no mark exists (`_marks_made`);
        # asking the flag here keeps that common case free of a call per value.
        if _marks_made:
            decision = _mark_decision(value)
            if decision is _REFUSED:
                _refuse_class_above_mark(value)
        else:
            decision = None
        # Main dumped every value with a `model_dump` (`hasattr`) that is not a class; so does
        # this walk, by one of the routes `_probe_own_attribute` picks (`_model_dump_route`). A
        # class is never dumped, whatever its metaclass reports. A real type whose route its MRO
        # alone decides takes it from `_type_memo`: `ASK_KEPT`, or main's C `hasattr`, which for
        # a type with no class-side hook reads only the real instance dict. No class's type is
        # recorded there.
        memo = _type_memo.get()
        routes = None if memo is None else memo[1]
        known = None if routes is None else routes.get(id(value_type))
        if known is None:
            route = _model_dump_route(value, value_type, routes)
        elif (route := known[1]) is None:
            route = _STATIC if hasattr(value, "model_dump") else None
        if route is _STATIC:
            # The static probe runs none of the value's code and agrees with `hasattr` for a
            # class that defines no `__getattr__`; the value is dumped as main dumps it.
            try:
                dump = value.model_dump
            except AttributeError:
                # A `model_dump` the class keeps but the object does not hold: an unset slot, or
                # a property that raises AttributeError. Main's `hasattr` is False there too, so
                # the value takes the branches below and is canonicalized as the plain object it
                # is.
                pass
            else:
                return _canonicalize_value_for_hash(dump(), (*active, id(value)))
        elif route is not None:
            # `ASK` or `ASK_KEPT`: a value that may answer names it does not keep (its class
            # defines `__getattr__`, or it reports another class) is asked by attribute lookup,
            # as main asks: beyond the `__class__` read each `isinstance` makes, the one place
            # the probe runs the value's code, as main does. When a class keeps `model_dump`, the
            # attribute exists and an error from the lookup propagates, as from main's `hasattr`;
            # when none does, an error is the object's "absent" signal and means no
            # (`_attribute_or_none`). On ASK, only an exact dict returned by a callable is
            # dumped, so a MagicMock or attribute tree that answers every name with junk gets the
            # plain form. On ASK_KEPT, a class really keeps the name, so its callable result is
            # dumped as main dumps it; a RootModel can therefore return a list, a secret or a
            # scalar root.
            if route is _ASK_KEPT:
                dump = getattr(value, "model_dump", None)
            else:
                dump = _attribute_or_none(value, "model_dump")
            if callable(dump):
                dumped = dump()
                if route is _ASK_KEPT or type(dumped) is dict:
                    return _canonicalize_value_for_hash(dumped, (*active, id(value)))
        if inspect.isfunction(value):
            return ["function", _function_identity(value, active)]
        if decision is _BY_NAME and not _carries_instance(value):
            return ["external", _library_name(value)]
        if _is_library_callable(value):
            return ["external", _library_name(value)]
        if isinstance(value, (functools.partial, functools.partialmethod)):
            partial_value = cast("Any", value)
            inner = (*active, id(partial_value))
            parts = (
                ("f", partial_value.func),
                ("a", partial_value.args),
                ("k", partial_value.keywords),
            )
            if _held_loaders.get() is None:
                return [
                    "partial",
                    *(_canonicalize_value_for_hash(part, inner) for _, part in parts),
                ]
            return ["partial", *(_canonical_at(step, part, inner) for step, part in parts)]
    if isinstance(value, dict):
        mapping = cast("dict[Any, Any]", value)
        inner = (*active, id(mapping))
        if _held_loaders.get() is not None:
            return ["dict", _canonical_entries_at(mapping, inner)]
        entries = [
            [_canonicalize_value_for_hash(key, inner), _canonicalize_value_for_hash(item, inner)]
            for key, item in mapping.items()
        ]
        entries.sort(key=json.dumps)
        return ["dict", entries]
    if isinstance(value, (list, tuple, set, frozenset)):
        collection = cast("list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any]", value)
        inner = (*active, id(collection))
        unordered = isinstance(collection, (set, frozenset))
        if _held_loaders.get() is not None:
            return [type(collection).__name__, _canonical_items_at(collection, inner, unordered)]
        items = [_canonicalize_value_for_hash(item, inner) for item in collection]
        if unordered:
            items.sort(key=json.dumps)
        return [type(collection).__name__, items]
    if isinstance(value, types.ModuleType):
        return ["module", value.__name__]
    from pivot import parameters

    if parameters.is_secret(value):
        return ["secret", parameters.secret_digest(value)]
    if not isinstance(value, type) and dataclasses.is_dataclass(cast("object", value)):
        inner = (*active, id(value))
        if _is_pivot_loader(value):
            return _canonical_loader(value, inner)
        fields = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
        return [_definition_ref(type(value)), _canonicalize_value_for_hash(fields, inner)]
    if type(value).__repr__ is object.__repr__ and _has_attribute(value, "__dict__"):
        inner = (*active, id(value))
        attributes: Any = vars(value)
        # A `__dict__` property may return anything (None, an int), so only an exact dict is
        # filtered. A mark copied into a dict subclass stays in the form; it is canonicalised by
        # the marked object's name, so the form is stable.
        if type(attributes) is dict and _BY_NAME_MARK in attributes:
            own = cast("dict[str, Any]", attributes)
            attributes = {name: item for name, item in own.items() if name != _BY_NAME_MARK}
        hidden = _hidden_state.get()
        if hidden is not None and _keeps_state_outside_dict(type(cast("object", value))):
            hidden[0] = True
        return [_definition_ref(type(value)), _canonicalize_value_for_hash(attributes, inner)]
    return [type(value).__name__, repr(value)]


def _canonical_at(step: str, value: Any, active: tuple[int, ...]) -> Any:
    """`_canonicalize_value_for_hash(value, active)`, one `step` further on `_held_path`."""
    token = _held_path.set((*_held_path.get(), [step]))
    try:
        return _canonicalize_value_for_hash(value, active)
    finally:
        _held_path.reset(token)


def _ranked(texts: list[str], cells: list[list[str]], prefix: str) -> list[int]:
    """The indexes of `texts` in sorted order. Each cell gets its text's rank in that order,
    after `prefix`; equal texts share a rank, so no rank follows the hash seed."""
    order = sorted(range(len(texts)), key=texts.__getitem__)
    rank = 0
    for position, index in enumerate(order):
        if position and texts[index] != texts[order[position - 1]]:
            rank = position
        cells[index][0] = f"{prefix}{rank}"
    return order


def _canonical_items_at(
    items: Collection[Any], active: tuple[int, ...], unordered: bool
) -> list[Any]:
    """The canonical forms of a list's, tuple's or set's items while held loaders are collected:
    each item one step further on `_held_path`, its index, or for a set its rank among the sorted
    forms (`_ranked`); a set's forms are sorted, as `_canonicalize_value_for_hash` sorts them."""
    path = _held_path.get()
    forms = list[Any]()
    cells = list[list[str]]()
    for index, item in enumerate(items):
        cell = ["" if unordered else str(index)]
        token = _held_path.set((*path, cell))
        try:
            forms.append(_canonicalize_value_for_hash(item, active))
        finally:
            _held_path.reset(token)
        cells.append(cell)
    if not unordered:
        return forms
    return [forms[index] for index in _ranked([json.dumps(form) for form in forms], cells, "s")]


def _canonical_entries_at(mapping: Mapping[Any, Any], active: tuple[int, ...]) -> list[Any]:
    """A dict's sorted canonical `[key, value]` entries while held loaders are collected: each
    entry one step further on `_held_path`, its rank among the sorted entries (`_ranked`), then
    `k` for its key or `v` for its value."""
    path = _held_path.get()
    entries = list[Any]()
    cells = list[list[str]]()
    for key, item in mapping.items():
        cell = [""]
        entries.append(
            [
                _canonical_at_cell(path, cell, "k", key, active),
                _canonical_at_cell(path, cell, "v", item, active),
            ]
        )
        cells.append(cell)
    return [entries[index] for index in _ranked([json.dumps(e) for e in entries], cells, "d")]


def _canonical_at_cell(
    path: tuple[list[str], ...], cell: list[str], step: str, value: Any, active: tuple[int, ...]
) -> Any:
    """`_canonicalize_value_for_hash(value, active)` at `path`, then `cell`, then `step`."""
    token = _held_path.set((*path, cell, [step]))
    try:
        return _canonicalize_value_for_hash(value, active)
    finally:
        _held_path.reset(token)


def _canonical_loader(loader: Any, inner: tuple[int, ...]) -> Any:
    """The canonical form of a pivot loader, `[module.qualname, canonical({field: value})]` as
    any dataclass instance's, `inner` ending with its id.

    While a holder's value is serialized (`_serialize_collecting`), the loader is noted with
    its form, the ids of the classes in it and its position, for the holder to fingerprint. The
    loaders inside it are its own fingerprint's, so they are not noted for the holder. A form
    that holds no cycle is the same wherever the loader sits, so it is kept for the stage
    (`_form_memo`): a chain of held loaders is canonicalized once, not once per level."""
    memo = _form_memo.get()
    form = None if memo is None else memo.get(id(loader))
    if form is None:
        loader_type: type[Any] = type(cast("object", loader))
        classes = list[int]()
        cycle = [False]
        held_token = _held_loaders.set(None)
        classes_token = _form_classes.set(classes)
        cycle_token = _form_cycle.set(cycle)
        try:
            fields = {
                field.name: getattr(loader, field.name) for field in dataclasses.fields(loader)
            }
            canonical = [_definition_ref(loader_type), _canonicalize_value_for_hash(fields, inner)]
        finally:
            _form_cycle.reset(cycle_token)
            _form_classes.reset(classes_token)
            _held_loaders.reset(held_token)
        form = _LoaderForm(loader, canonical, json.dumps(canonical), (id(loader_type), *classes))
        if not cycle[0]:
            if memo is not None:
                memo[id(loader)] = form
        elif (outer := _form_cycle.get()) is not None:
            outer[0] = True
    outer_classes = _form_classes.get()
    if outer_classes is not None:
        outer_classes.extend(form.classes)
    held = _held_loaders.get()
    if held is not None:
        held.append(_HeldLoader(loader, form.text, form.classes, _held_path.get()))
    return form.canonical


def _add_callable_to_manifest(
    key: str, func: Callable[..., Any], manifest: dict[str, str], visited: set[int]
) -> None:
    """Hash callable and merge its transitive dependencies into manifest.

    Its closure cells and defaults are keyed under the reference in `key`.
    """
    module_name = getattr(func, "__module__", None)
    if isinstance(module_name, str):
        _mark_module_covered(module_name, manifest)
    manifest[key] = hash_function_ast(func)
    token = _owner_ref.set(key.partition(":")[2])
    try:
        children = get_stage_fingerprint(func, visited)
    finally:
        _owner_ref.reset(token)
    for child_key, child_val in children.items():
        if not child_key.startswith("self:"):
            manifest[child_key] = child_val


def _process_module_dependency(
    chain: str,
    module: types.ModuleType,
    func: Callable[..., Any],
    manifest: dict[str, str],
    visited: set[int],
) -> None:
    """Fingerprint each attribute `func` reads on user module `module`, reached as `chain`.

    An attribute read as `settings.NAME` takes the path a global takes
    (`_process_captured_value`), with the binding `<module>.<NAME>`: a scalar is
    `const:settings.FACTOR`, a library function `ext:settings.AGG`, a partial or a logger gets
    the global rule, and a class or submodule has its own reads followed from the longer chain
    (`settings.Config.SCALE`). A cache hit re-reads each attribute (`a:NAME`). A library
    module's attributes are not read: they are library code, and `repr()` of objects like `np.c_`
    holds an address; the module is recorded by name (`_process_captured_value`).
    """
    _mark_module_covered(module.__name__, manifest)
    stage_name = _current_stage_name.get() or getattr(func, "__name__", "<unknown>")
    for attribute in _attributes_read(func, chain):
        step = f"a:{attribute}"
        _record_read(module, step)
        try:
            value = getattr(module, attribute)
        except AttributeError:
            continue
        binding_ref = f"{module.__name__}.{attribute}"
        with _walking(step):
            _process_captured_value(
                binding_ref,
                binding_ref,
                value,
                func,
                manifest,
                visited,
                stage_name,
                chain=f"{chain}.{attribute}",
            )


def _get_qualname_for_cache(func: Callable[..., Any]) -> str:
    """Get qualname, disambiguated for lambdas.

    Normal functions return their __qualname__ unchanged.
    Lambdas return qualname with line and column appended (e.g., "<lambda>:42:8")
    to avoid cache key collisions when multiple lambdas exist in the same file.
    """
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", "<unknown>")

    if "<lambda>" not in qualname:
        return qualname

    code = getattr(func, "__code__", None)
    if code is None:
        return qualname

    lineno = code.co_firstlineno
    col = 0
    if hasattr(code, "co_positions"):
        for _, _, c, _ in code.co_positions():
            # Skip col=0 which is the RESUME instruction, not the actual lambda
            if c is not None and c > 0:
                col = c
                break

    return f"{qualname}:{lineno}:{col}"


def _definition_ref(obj: Any) -> str:
    """Return an object's module-qualified definition name for a manifest key."""
    module = getattr(obj, "__module__", None) or "<unknown>"
    return f"{module}.{_get_qualname_for_cache(obj)}"


def _unique_ref(cls: type) -> str:
    """A class's reference in a manifest key: its definition name (`_definition_ref`), and for a
    class with no source (`_has_no_source`) its hash too (`<name>@<hash>`). Such classes share a
    name when one factory makes them (`collections.namedtuple` takes the factory's module), so
    the name alone would key two of them alike and drop the second."""
    ref = _definition_ref(cls)
    return f"{ref}@{hash_function_ast(cls)}" if _has_no_source(cls) else ref


def _binding_ref(func: Callable[..., Any], name: str) -> str:
    """Return a module-qualified reference for a value reached through `func`."""
    globals_dict = getattr(func, "__globals__", {})
    module = globals_dict.get("__name__") or getattr(func, "__module__", None) or "<unknown>"
    return f"{module}.{name}"


def _callable_ref(func: Callable[..., Any], binding_ref: str) -> str:
    """Use callable's definition, or the binding path where one definition makes many.

    A functools.wraps wrapper gets a separate owner from the function it wraps: update_wrapper
    copies the wrapped function's module and qualname, so without the marker the child's hash
    overwrites the wrapper's bytecode hash in the manifest.
    """
    if inspect.isfunction(func) and hasattr(func, "__wrapped__"):
        return f"{binding_ref}.<wrapper>"
    qualname = getattr(func, "__qualname__", "") or ""
    if "<lambda>" in qualname:
        return binding_ref
    if "<locals>" in qualname and (
        getattr(func, "__closure__", None)
        or getattr(func, "__defaults__", None)
        or getattr(func, "__kwdefaults__", None)
    ):
        return binding_ref
    return _definition_ref(func)


def _should_skip_persistent_cache(func: Callable[..., Any]) -> bool:
    """Check if function should skip persistent cache.

    Skip for:
    - Closures: `<locals>` in qualname → qualname collision risk
    - Wrapped functions: has `__wrapped__` → source file mismatch risk
    - A class with no source of its own (`_has_no_source`): its hash reads values that may
      come from other files, and two such classes can share a module and a name
    """
    qualname = getattr(func, "__qualname__", "")
    if "<locals>" in qualname:
        return True
    if _has_no_source(func):
        return True
    return hasattr(func, "__wrapped__")


def _has_no_source(obj: Any) -> bool:
    """Whether `obj` is a class with no class statement of its own (`collections.namedtuple`,
    `dataclasses.make_dataclass`, a functional `enum.Enum`, `type(...)`,
    `pydantic.create_model`): the compiler sets `__firstlineno__` in a class statement's
    namespace, and `inspect.getsource` finds a class by it."""
    return isinstance(obj, type) and "__firstlineno__" not in vars(obj)


def hash_function_ast(func: Callable[..., Any]) -> str:
    """Hash function AST (ignores whitespace, comments, docstrings).

    Uses persistent cache in StateDB when available, keyed by
    (file_path, source content digest, qualname, py_version, schema_version)
    for automatic invalidation on file changes or Python upgrades.

    Limitation: Lambdas and functions without source code fall back to id(func),
    which is non-deterministic across runs. This causes unnecessary re-runs for
    stages using lambdas. Mitigation: Use named functions instead of lambdas in
    pipeline stages for stable fingerprinting.
    """
    if _active_source_map is not None:
        info = _get_func_source_info(func)
        if info is not None:
            rel_path, digest = info
            _active_source_map.setdefault(rel_path, digest)
    _t = metrics.start()
    try:
        # 0. The mark decision (`_mark_decision`): an object the walk records by name gets a
        # hash of that name, and its body is not read. A chain with a user class instance
        # above the mark is refused (a default factory, a loader method or a function identity
        # reaches the hash without the walk's value classification). While no mark exists
        # (`_marks_made`) the decision is None, so it is not asked.
        if _marks_made:
            decision = _mark_decision(func)
            if decision is _MarkDecision.REFUSED:
                _refuse_class_above_mark(func)
            if decision is _MarkDecision.BY_NAME or decision is _MarkDecision.REFUSED:
                return _by_name_hash(func)

        # 1. Check in-memory WeakKeyDictionary cache first (fastest)
        # WeakKeyDictionary raises TypeError for non-weakly-referenceable functions (builtins)
        try:
            if (cached := _hash_function_ast_cache.get(func)) is not None:
                metrics.count("fingerprint.hash_function_ast.memory_cache_hit")
                return cached
        except TypeError:
            # Not weakly referenceable (builtins), compute directly
            return _compute_function_hash(func)

        # 2. Check persistent cache if appropriate
        source_info: tuple[str, str] | None = None
        qualname: str | None = None

        if not _should_skip_persistent_cache(func):
            source_info = _get_func_source_info(func)
            if source_info is not None:
                qualname = _get_qualname_for_cache(func)
                db = _get_state_db()
                if db is not None:
                    rel_path, digest = source_info
                    try:
                        persistent_cached = db.get_ast_hash(
                            rel_path, digest, qualname, _PYTHON_VERSION, _CACHE_SCHEMA_VERSION
                        )
                        if persistent_cached is not None:
                            metrics.count("fingerprint.hash_function_ast.persistent_cache_hit")
                            # Store in memory cache too
                            _hash_function_ast_cache[func] = persistent_cached
                            return persistent_cached
                    except Exception:
                        # LMDB errors shouldn't break fingerprinting
                        pass

        # 3. Compute hash
        result = _compute_function_hash(func)

        # 4. Store in memory cache
        _hash_function_ast_cache[func] = result

        # 5. Queue persistent write under this run's state directory (if source info
        # available and not skipping)
        if source_info is not None and qualname is not None:
            state_dir = _producing_state_dir()
            if state_dir is not None:
                rel_path, digest = source_info
                _pending_ast_writes.setdefault(state_dir, list[AstHashEntry]()).append(
                    (rel_path, digest, qualname, _PYTHON_VERSION, _CACHE_SCHEMA_VERSION, result)
                )
                metrics.count("fingerprint.hash_function_ast.persistent_cache_miss")

        return result
    finally:
        metrics.end("fingerprint.hash_function_ast", _t)


_PORTABLE_CO_FILENAME = "<pivot>"


def _hash_code_object(code: types.CodeType) -> str:
    """Hash code while excluding absolute source paths."""
    return xxhash.xxh64(marshal.dumps(_replace_code_filenames(code))).hexdigest()


def _replace_code_filenames(code: types.CodeType) -> types.CodeType:
    """Replace the source filename in `code` and nested code constants."""
    constants = tuple(
        _replace_code_filenames(value) if isinstance(value, types.CodeType) else value
        for value in code.co_consts
    )
    return code.replace(co_filename=_PORTABLE_CO_FILENAME, co_consts=constants)


def _resolve_static_callee(expr: ast.expr, globals_dict: dict[str, Any]) -> object | None:
    """Resolve a name or dotted attribute without executing user code."""
    if isinstance(expr, ast.Name):
        return globals_dict.get(expr.id)
    if not isinstance(expr, ast.Attribute):
        return None
    owner = _resolve_static_callee(expr.value, globals_dict)
    if not isinstance(owner, types.ModuleType):
        return None
    namespace = object.__getattribute__(owner, "__dict__")
    return namespace.get(expr.attr)


def _strip_stage_decorator(tree: ast.Module, globals_dict: dict[str, Any]) -> None:
    """Exclude Pivot's own stage decorator from a stage code hash.

    Drops each decorator on the stage's own FunctionDef that statically resolves to
    `pivot.compose.stage`: bare (`@stage`, `@compose.stage`) or called
    (`@stage(resources=...)`). Resources are scheduler admission, not stage code, so adding,
    removing or changing them keeps the hash. Other decorators stay hashed.
    """
    from pivot import compose

    outer = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if outer is None:
        return
    outer.decorator_list = [
        decorator
        for decorator in outer.decorator_list
        if _resolve_static_callee(
            decorator.func if isinstance(decorator, ast.Call) else decorator, globals_dict
        )
        is not compose.stage
    ]


def _compute_function_hash(func: Callable[..., Any]) -> str:
    """Compute hash for a function (uncached implementation)."""
    _t = metrics.start()
    try:
        # Builtins (list, dict, set, etc.) have no source - use stable name-based hash.
        # This is deterministic across Python sessions unlike id(func).
        if isinstance(func, type) and func.__module__ == "builtins":
            return xxhash.xxh64(f"builtin:{func.__qualname__}".encode()).hexdigest()

        # For wrapped functions (via functools.wraps), inspect.getsource() follows
        # __wrapped__ and returns the ORIGINAL function's source, making decorator
        # logic invisible. Use bytecode to capture the actual wrapper.
        if hasattr(func, "__wrapped__") and hasattr(func, "__code__"):
            return _hash_code_object(func.__code__)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType] - hasattr guards access

        # `inspect.getsource` follows `__wrapped__` with no stop, so the read takes the object
        # `_source_object` gives: the end of the chain, as on main, or the code of the user
        # decorator above a mark. Where the read would reach the mark, the hash is of the name.
        target = _source_object(func)
        if target is None:
            return _by_name_hash(func)
        _t_source = metrics.start()
        try:
            source = loaded_source.getsource(target)
        except (OSError, TypeError):
            metrics.end("fingerprint.inspect_getsource", _t_source)
            if isinstance(target, types.CodeType):
                return _hash_code_object(target)
            if hasattr(func, "__code__"):
                # Marshal captures constants as well as bytecode, so `x + 1` and `x + 999`
                # differ. Remove embedded source paths before hashing to keep it portable.
                return _hash_code_object(func.__code__)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType] - hasattr guards access
            if isinstance(func, type):
                return _sourceless_class_hash(func)
            # KNOWN ISSUE: Using id(func) is non-deterministic across runs
            # This affects lambdas without source code, causing unnecessary re-runs
            if is_user_code(func):
                message = (
                    "Cannot get source for '%s'. Using non-deterministic fallback — "
                    + "this stage will re-run every time. Use a named function defined "
                    + "in a source file for stable fingerprinting."
                )
                _logger.warning(message, getattr(func, "__qualname__", func))
            return xxhash.xxh64(str(id(func)).encode()).hexdigest()
        metrics.end("fingerprint.inspect_getsource", _t_source)

        _t_parse = metrics.start()
        dedented_source = textwrap.dedent(source)
        try:
            tree = ast.parse(dedented_source)
        except SyntaxError:
            metrics.end("fingerprint.ast_parse", _t_parse)
            # Fallback: hash dedented source (not raw source)
            return xxhash.xxh64(dedented_source.encode()).hexdigest()
        metrics.end("fingerprint.ast_parse", _t_parse)

        _t_norm = metrics.start()
        globals_dict = cast("dict[str, Any]", getattr(func, "__globals__", {}))
        _strip_stage_decorator(tree, globals_dict)
        tree = _normalize_ast(tree)
        ast_str = ast.dump(tree, annotate_fields=True, include_attributes=False)
        metrics.end("fingerprint.normalize_and_dump", _t_norm)
        return xxhash.xxh64(ast_str.encode()).hexdigest()
    finally:
        metrics.end("fingerprint._compute_function_hash", _t)


# A memory address in a repr (`<object object at 0x7f...>`, pydantic-core's `Py(0x...)`), which
# differs in each process.
_ADDRESS = re.compile(r"0x[0-9a-fA-F]{6,}")
# Namespace entries every class has whose text says nothing of what defines it; `_abc_impl` is
# the ABC machinery's cache of registered and checked subclasses.
_CLASS_NAMESPACE_NOISE = frozenset(("__dict__", "__weakref__", "__doc__", "_abc_impl"))
# Namespace entries pydantic generates for a model from what `_pydantic_model_parts` records
# (its fields, config, validators, private attributes and class variables): the core schema,
# validator and serializer, whose reprs hold addresses and vary between processes, and the
# rest of its bookkeeping.
_PYDANTIC_GENERATED = frozenset(
    (
        "__abstractmethods__",
        "__annotations__",
        "__annotate__",
        "__class_vars__",
        "__private_attributes__",
        "__signature__",
        "model_config",
    )
)
# Set while `_sourceless_parts` builds a form: `_canonicalize_value_for_hash` sets it when it
# meets an instance whose state its form does not hold (`_keeps_state_outside_dict`).
_hidden_state: contextvars.ContextVar[list[bool] | None] = contextvars.ContextVar(
    "_hidden_state", default=None
)


def _sourceless_class_hash(cls: type) -> str:
    """The hash of a class with no source of its own (`collections.namedtuple`,
    `dataclasses.make_dataclass`, a functional `enum.Enum`, `type(...)`, `pydantic.create_model`):
    what defines it (`_sourceless_parts`). It is the same in every process, and it changes when
    what defines the class changes: its fields, its members, or the code of a function it holds.

    An entry that holds a value whose form does not show it (a sentinel `object()` or a
    `__slots__` object, whose repr is an address; a `random.Random`, whose state is in C) keeps
    the rest of its form, without the address, so a change to any other entry is still seen. A
    change to such a value itself is not visible in any form, so the class is then also hashed
    with the source of the module that defines it (`_module_source_digest`): an edit to that
    module reruns the stages that read the class, as it did when the class was hashed by its
    address, and no new process and no fresh state database does."""
    parts, opaque = _sourceless_parts(cls)
    if opaque:
        parts.append(["module-source", _module_source_digest(cls)])
    return xxhash.xxh64(json.dumps(parts).encode()).hexdigest()


def _sourceless_parts(cls: type) -> tuple[list[Any], bool]:
    """What defines a class with no source, without addresses, and whether it holds a value its
    form does not show (an address, or `_keeps_state_in_c`): its module, name and bases, then,
    for a pydantic model, `_pydantic_model_parts`, and for any other class each entry of its own
    namespace (`_sourceless_form`)."""
    parts: list[Any] = [
        "class",
        cls.__module__,
        cls.__qualname__,
        [_definition_ref(base) for base in cls.__bases__],
    ]
    namespace = vars(cls)
    hidden = [False]
    token = _hidden_state.set(hidden)
    try:
        if "__pydantic_fields__" in namespace and "__pydantic_decorators__" in namespace:
            parts.extend(_pydantic_model_parts(cls))
        else:
            for name in sorted(namespace):
                if name not in _CLASS_NAMESPACE_NOISE:
                    parts.append([name, _sourceless_form(namespace[name])])
    finally:
        _hidden_state.reset(token)
    stripped = _without_addresses(parts, hidden)
    return stripped, hidden[0]


def _keeps_state_outside_dict(cls: type) -> bool:
    """Whether instances of `cls` may hold state their `__dict__` does not: an instance larger
    than a bare `object` keeps `__slots__` values or a C type's fields (`random.Random` keeps its
    generator's state in `_random.Random`). A class statement's own `__dict__` and
    `__weakref__` add no size."""
    return cls.__basicsize__ > object.__basicsize__


def _without_addresses(form: Any, found: list[bool]) -> Any:
    """`form` with each memory address in its text replaced by `0x?`; `found[0]` is set when one
    was."""
    if isinstance(form, str):
        text = _ADDRESS.sub("0x?", form)
        if text != form:
            found[0] = True
        return text
    if isinstance(form, list):
        return [_without_addresses(item, found) for item in cast("list[Any]", form)]
    if isinstance(form, dict):
        mapping = cast("dict[str, Any]", form)
        return {key: _without_addresses(item, found) for key, item in mapping.items()}
    return form


def _module_source_digest(cls: type) -> str | None:
    """The digest of the source of the module that defines `cls` (`_source_digest`), or None
    when the module has no source file."""
    module = sys.modules.get(cls.__module__)
    filename = None if module is None else vars(module).get("__file__")
    if not isinstance(filename, str):
        return None
    try:
        return _source_digest(filename)
    except OSError:
        return None


def _pydantic_model_parts(cls: type) -> list[Any]:
    """What defines a pydantic model with no source (`pydantic.create_model`), from what its
    author declared: each field's name and the public attributes of its `FieldInfo` (annotation,
    default, default factory, alias, constraints...), its config, the code and settings of each
    validator, serializer and computed field, its private attributes and class variables, and
    any other entry of its namespace. The core schema, validator and serializer pydantic builds
    from these are left out: their reprs hold addresses, and they differ between processes."""
    namespace = vars(cls)
    model = cast("Any", cls)
    fields = cast("dict[str, Any]", namespace["__pydantic_fields__"])
    field_forms = list[Any]()
    for name, info in fields.items():
        slot_names = cast("tuple[str, ...]", getattr(type(cast("object", info)), "__slots__", ()))
        slots = [slot for slot in slot_names if not slot.startswith("_")]
        field_forms.append(
            [name, [[slot, _sourceless_form(getattr(info, slot))] for slot in slots]]
        )
    decorators = namespace["__pydantic_decorators__"]
    decorator_forms = list[Any]()
    decorator_names = set[str]()
    for group in dataclasses.fields(decorators):
        for name, decorator in cast("dict[str, Any]", getattr(decorators, group.name)).items():
            decorator_names.add(name)
            function = getattr(decorator.func, "__func__", decorator.func)
            decorator_forms.append(
                [
                    group.name,
                    name,
                    _sourceless_form(function),
                    _decorator_info_form(decorator.info),
                ]
            )
    private = cast("dict[str, Any]", namespace.get("__private_attributes__", {}))
    private_forms = [
        [name, _sourceless_form(attr.default), _sourceless_form(attr.default_factory)]
        for name, attr in private.items()
    ]
    rest = [
        [name, _sourceless_form(namespace[name])]
        for name in sorted(namespace)
        if name not in _CLASS_NAMESPACE_NOISE
        and name not in _PYDANTIC_GENERATED
        and not name.startswith("__pydantic_")
        and name not in decorator_names
        and name not in private
    ]
    return [
        ["fields", field_forms],
        ["config", _sourceless_form(dict(model.model_config))],
        ["decorators", decorator_forms],
        ["private", private_forms],
        ["class-vars", sorted(cast("set[str]", namespace.get("__class_vars__", set[str]())))],
        ["namespace", rest],
    ]


def _decorator_info_form(info: Any) -> Any:
    """The settings of a pydantic validator, serializer or computed field (a dataclass such as
    `FieldValidatorDecoratorInfo`), field by field (`_sourceless_form`), so a property or
    function in it is recorded by its code."""
    if not dataclasses.is_dataclass(info) or isinstance(info, type):
        return _sourceless_form(info)
    return [
        _definition_ref(type(cast("object", info))),
        [
            [field.name, _sourceless_form(getattr(info, field.name))]
            for field in dataclasses.fields(info)
        ],
    ]


def _sourceless_form(value: Any) -> Any:
    """A form of one namespace entry of a class with no source (`_sourceless_class_hash`): a
    function by its code, a library function by its name, a class by its name, a dataclass's
    fields by their names, types and defaults, and any other value by its canonical form. A
    memory address in a form is left out by `_sourceless_parts`."""
    if isinstance(value, (staticmethod, classmethod)):
        wrapper = cast("staticmethod[..., Any] | classmethod[Any, ..., Any]", value)
        return [type(wrapper).__name__, _sourceless_form(cast("Any", wrapper).__func__)]
    if isinstance(value, property):
        return ["property", *(_sourceless_form(part) for part in (value.fget, value.fset))]
    if inspect.isfunction(value):
        return _sourceless_function_form(value)
    if isinstance(value, type):
        return ["class", _definition_ref(value)]
    if isinstance(value, dataclasses.Field):
        field = cast("dataclasses.Field[Any]", value)
        return [
            "field",
            field.name,
            _sourceless_form(field.type),
            _sourceless_form(field.default),
            _sourceless_form(field.default_factory),
        ]
    if value is dataclasses.MISSING:
        return ["missing"]
    if isinstance(value, dict) and any(
        isinstance(item, dataclasses.Field) for item in cast("dict[Any, Any]", value).values()
    ):
        mapping = cast("dict[str, Any]", value)
        return ["fields", [[name, _sourceless_form(item)] for name, item in mapping.items()]]
    return _canonicalize_value_for_hash(value)


def _sourceless_function_form(function: types.FunctionType) -> Any:
    """A function a class with no source holds: by its code hash when it was compiled from a
    user source file, by its name when from a library file, and otherwise (code generated by
    `exec`, as `namedtuple` and `make_dataclass` make it) by its code object and defaults."""
    filename = function.__code__.co_filename
    if _is_user_source_file(filename):
        return ["function", hash_function_ast(function)]
    if os.path.isfile(filename):
        return ["external", f"{function.__module__}.{function.__qualname__}"]
    return [
        "code",
        _code_form(function.__code__),
        _sourceless_form(function.__defaults__),
        _sourceless_form(function.__kwdefaults__),
    ]


def _code_form(code: types.CodeType) -> list[Any]:
    """The parts of a code object that decide what it does, without its file name or line
    numbers; constants by their canonical form, so a frozenset constant is not in hash-seed
    order."""
    return [
        code.co_code.hex(),
        [
            _code_form(constant)
            if isinstance(constant, types.CodeType)
            else _canonicalize_value_for_hash(constant)
            for constant in code.co_consts
        ],
        list(code.co_names),
        list(code.co_varnames),
        list(code.co_freevars),
        [code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount, code.co_flags],
    ]


def _normalize_ast(node: ast.AST) -> ast.AST:
    """Remove docstrings and normalize function names for stable hashing."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        node.name = "func"

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and _has_docstring(
        node
    ):
        node.body = node.body[1:]
        # Ensure body is never empty after removing docstring
        if not node.body:
            node.body = [ast.Pass()]

    for child in ast.iter_child_nodes(node):
        _normalize_ast(child)

    return node


def _has_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> bool:
    """Check if node has a docstring as first statement."""
    return (
        bool(node.body)
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    )


_BY_NAME_MARK = "__pivot_fingerprint_by_name__"
# The marked objects the mark cannot be stored on (see `mark_by_name`): a class-based decorator
# with `__slots__` and no `__dict__`, which may lack a `__weakref__` slot too, so no weak
# reference can hold it. The
# table is keyed by `id(obj)` because such a class may define `__eq__` without `__hash__`, which
# makes the object unusable as a key. The value is the object itself, a strong reference: the
# table keeps every marked object alive for the life of the process, so its address is never
# freed and no other object can get its id. `fingerprinted_by_name` also checks the stored value
# is `obj` itself, so a lookup by id alone never marks another object.
_marked_without_dict = dict[int, Any]()
# Whether `@pivot.fingerprint_by_name` has marked anything in this process. In a process that
# fingerprints, the decorator, run when discovery imports the pipeline's modules, is the only
# writer of a mark (`mark_by_name`), so until it runs no `__wrapped__` chain can hold one: every
# mark decision is None and the walk reads no chain. A project that marks nothing, the common
# one, pays one module-global read per classified value for the mark. Two states pivot never
# enters break that premise (an object unpickled by value with its `__dict__`, and
# `importlib.reload` of this module); in both the walk records the body: more, never less.
#
# `AGENTS.md` warns against module-level mutable state, and a flush of project-scoped state into
# the wrong project caused a real bug (#536). This flag is not that kind of state, so do not
# "fix" it into a `ContextVar` or a parameter: it records which objects the decorator marked in
# *this interpreter*, a property of the loaded code and never of a project, it is set once and
# never reset, it is never read as "which project am I in", and it gates only the probe's speed —
# the decision's result is the same whether it is True or False, because an unmarked value's
# chain holds no mark either way. Removing it puts a chain read and an MRO walk back on every
# classified value of every project, marked or not: the cost the benchmark gate caught.
_marks_made = False


def mark_by_name(obj: Any) -> None:
    """Mark `obj` itself for `@pivot.fingerprint_by_name` (see `fingerprinted_by_name`).

    The mark is written with `object.__setattr__`, the generic C store: no class `__setattr__`
    and no class-level `__dict__` property runs, and the mark lands in the real instance dict.
    An object with no instance dict (`__slots__` without `__dict__`), or whose type refuses the
    generic store, is held in `_marked_without_dict` instead.
    """
    global _marks_made
    _marks_made = True
    try:
        object.__setattr__(obj, _BY_NAME_MARK, obj)
    except (AttributeError, TypeError):
        _marked_without_dict[id(obj)] = obj


def fingerprinted_by_name(obj: Any) -> bool:
    """Whether `@pivot.fingerprint_by_name` marked `obj` itself.

    The fallback table is checked first by object identity, for a marked object whose type
    refused the generic store. Otherwise the mark is read with `object.__getattribute__`, the
    generic C lookup: no class `__getattr__`, no class-level `__dict__` property and no metaclass
    code runs. A class property with the mark's name can still run, so any error from that read
    means unmarked. `functools.update_wrapper` copies the instance dict into a wrapper, so the
    copy in a wrapper holds a different object and the wrapper is not marked.
    """
    if _marked_without_dict.get(id(obj)) is obj:
        return True
    try:
        mark = object.__getattribute__(obj, _BY_NAME_MARK)
    except Exception:
        return False
    return mark is obj


def wrapper_chain(obj: Any, stop: Callable[[Any], bool] | None = None) -> Iterator[Any]:
    """Yield `obj`, then each object its `__wrapped__` chain reaches, in order.

    Every step along a `__wrapped__` chain in pivot goes through this, so each path sees the
    chain `inspect.unwrap` sees; the mark decision first asks whether a value keeps a
    `__wrapped__` of its own (`_mark_chain`). It follows the chain as `inspect.unwrap` does: by
    attribute access, so a `__wrapped__` kept in a slot or a property counts as one in an
    instance `__dict__`; a class ends the chain; `stop` ends it at the first object it accepts;
    and its loop guard, by object identity and the recursion limit, ends the chain before an
    object it has already yielded. `inspect.unwrap` raises ValueError at a loop, and so does
    `unwrap`; a decision about the chain (`_mark_decision`, through `_mark_chain`) takes a loop
    as the chain's end, so a looped chain without a mark on it is decided as main decides it.
    """
    memo = {id(obj): obj}
    limit = sys.getrecursionlimit()
    current = obj
    yield current
    while not isinstance(current, type) and hasattr(current, "__wrapped__"):
        if stop is not None and stop(current):
            return
        current = current.__wrapped__
        if id(current) in memo or len(memo) >= limit:
            return
        memo[id(current)] = current
        yield current


def unwrap(obj: Any, stop: Callable[[Any], bool] | None = None) -> Any:
    """`inspect.unwrap(obj, stop=stop)`, through `wrapper_chain`: the last object on `obj`'s
    `__wrapped__` chain, or the first one `stop` accepts. A loop raises ValueError, as it does
    in `inspect.unwrap`.

    It differs from `inspect.unwrap` in two pathological chains, and each fails loudly and
    drops nothing: with a `__wrapped__` property that returns another object on each access,
    the second test of the end (`looped`, and `stop` asked again) can raise ValueError where
    `inspect.unwrap` returns; and on a chain longer than the recursion limit, `_stop_at_mark`
    asked of a lower link can see a mark that `_mark_decision` asked of the top does not, and
    the walk then raises.
    """
    *_, end = wrapper_chain(obj, stop)
    looped = not isinstance(end, type) and hasattr(end, "__wrapped__")
    if looped and (stop is None or not stop(end)):
        raise ValueError(f"wrapper loop when unwrapping {obj!r}")
    return end


def _mark_chain(obj: Any) -> Iterator[Any]:
    """`wrapper_chain(obj)` as the mark decision reads it, for a value main may not walk as code.

    `obj` is a wrapper only if it keeps a `__wrapped__` of its own (`_keeps_own_attribute`), so
    its `__getattr__` is not asked. Then the chain is read as `wrapper_chain` reads it, with the
    same reads, and a lookup that raises an exception other than AttributeError ends the chain.
    Main reads no `__wrapped__` of a value it does not unwrap, such as a callable user class
    instance, which it refuses as it is; so an object whose `__getattr__` answers every name, or
    raises KeyError for a missing one (a dict-backed config, an `AttrDict`), and a `__wrapped__`
    property that raises, give no chain, and the value is decided as main decides it. Where main
    does unwrap the value, `unwrap` reads the chain as `inspect.unwrap` does on main, errors
    included.
    """
    if not _keeps_own_attribute(obj, "__wrapped__"):
        yield obj
        return
    chain = wrapper_chain(obj)
    while True:
        try:
            current = next(chain)
        except Exception:  # StopIteration at the end, or the error of user code in a lookup
            return
        yield current


def _recorded_by_name(obj: Any) -> bool:
    """Whether the walk records `obj` by its module and qualified name for a mark.

    That is a marked object (a function, or an instance of a class-based decorator that
    `@fingerprint_by_name` marked), a method bound to one, and a library wrapper object without
    Python code of its own (functools.cache, lru_cache) whose `__wrapped__` chain reaches a
    marked object through such wrappers only. A Python function on the chain (a functools.wraps
    decorator) or an instance of a user class above the mark is user code: `_mark_decision`
    fingerprints the function by its own code and refuses the instance.
    """
    if not _marks_made:  # nothing in this process is marked (`_marks_made`)
        return False
    if fingerprinted_by_name(obj):
        return True
    if inspect.ismethod(obj):
        return _recorded_by_name(obj.__func__)
    if inspect.isfunction(obj) or isinstance(obj, (type, types.ModuleType)):
        return False
    if is_user_code(type(obj)):
        return False
    below = _mark_chain(obj)
    next(below)  # `obj` itself
    for current in below:
        if fingerprinted_by_name(current):
            return True
        if _is_user_class_instance(current):
            return False
        if inspect.isfunction(current) or isinstance(current, (type, types.ModuleType)):
            return False
    return False


class _MarkDecision(enum.Enum):
    """What the walk does with an object on a `__wrapped__` chain that holds a mark."""

    BY_NAME = "by_name"  # the marked object, or library wrappers over it: recorded by name
    OWN_CODE = "own_code"  # a user function above the mark: fingerprinted by its own code
    REFUSED = "refused"  # a user class instance above the mark: refused


# The decisions the canonical form compares once per value, read as globals.
_BY_NAME = _MarkDecision.BY_NAME
_REFUSED = _MarkDecision.REFUSED


def _mark_decision(obj: Any) -> _MarkDecision | None:
    """The one decision for an object whose `__wrapped__` chain holds a `@fingerprint_by_name`
    mark, or None when the chain holds no mark.

    Every path that reads a value (a global, closure cell, default, class member, module
    attribute, partial function or argument, loader field, default factory) asks it before it
    unwraps, so no path drops an object on such a chain without a decision. The decision and
    every unwrap read the chain through `wrapper_chain`, so they see the same objects of a chain
    whose objects keep their own `__wrapped__`. The decision reads it through `_mark_chain`, and
    only for a callable: main walks no other value as code, and reads no `__wrapped__` of a
    value it does not unwrap, so the decision does not ask the value's own `__getattr__`, and
    takes an error from a lookup as the end of the chain. Without a mark the answer is None and
    the walk is main's.

    Two short-circuits keep the decision off the hot path: without a mark in the process
    (`_marks_made`) it is None for every value, with no chain read; and within one
    `fingerprint_pass()` an object is decided once, so a value classified twice reads its chain
    once.
    """
    if not _marks_made or not callable(obj) or isinstance(obj, (type, types.ModuleType)):
        return None
    return _memoized("mark", obj, lambda: _chain_mark_decision(obj))


def _chain_mark_decision(obj: Any) -> _MarkDecision | None:
    """`_mark_decision` for a callable in a process that holds a mark: the chain reads."""
    target = obj.__func__ if inspect.ismethod(obj) else obj
    if _mark_on_chain(target) is None:
        return None
    if _recorded_by_name(obj):
        return _MarkDecision.BY_NAME
    if _class_above_mark(target) is not None:
        return _MarkDecision.REFUSED
    return _MarkDecision.OWN_CODE


def _class_above_mark(obj: Any) -> Any:
    """The first user class instance on `obj`'s `__wrapped__` chain (`_mark_chain`), `obj`
    included, that is above a marked object, or None."""
    found: Any = None
    for current in _mark_chain(obj):
        if fingerprinted_by_name(current):
            return found
        if found is None and _is_user_class_instance(current):
            found = current
    return None


def _refuse_class_above_mark(value: Any) -> None:
    """Refuse `value`, on whose chain a user class instance is above a mark, naming both.

    The mark records only the object below the instance, and no fingerprint records a class
    instance's code and state, so an edit to the class would go unseen. Unsafe fingerprinting
    does not suppress this refusal: the fixes are local to the helper.
    """
    instance = _class_above_mark(value.__func__ if inspect.ismethod(value) else value)
    marked = _mark_on_chain(instance)  # below the instance: `_class_above_mark` stops at a mark
    decorator = _definition_ref(type(instance))
    helper = _library_name(marked) if marked is not None else _definition_ref(instance)
    stage = _current_stage_name.get()
    prefix = f"Stage '{stage}': " if stage else ""
    message = (
        f"{prefix}the class-based decorator '{decorator}' is above '{helper}', which is marked "
        + "@fingerprint_by_name.\n"
        + f"The mark records '{helper}' by name only, and Pivot cannot fingerprint the code and "
        + f"state of a class-based decorator, so an edit to '{decorator}' would leave the stage "
        + "cached with an old output.\n"
        + "Fix one of:\n"
        + "  - put @fingerprint_by_name above the class-based decorator, so Pivot records the "
        + "whole helper by name;\n"
        + "  - replace the class with a function decorator built with functools.wraps, which "
        + "Pivot fingerprints by its own code."
    )
    errors = _capture_errors.get()
    if errors is None:
        raise exceptions.StageDefinitionError(message)
    errors.append(message)


def _mark_on_chain(obj: Any) -> Any:
    """The first marked object on `obj`'s `__wrapped__` chain (`_mark_chain`), `obj` included,
    or None. Without a mark in the process (`_marks_made`) no chain holds one, and none is read.
    """
    if not _marks_made:
        return None
    return next((current for current in _mark_chain(obj) if fingerprinted_by_name(current)), None)


def _stop_at_mark(obj: Any) -> bool:
    """Where an unwrap that leads to code stops: a marked object, or user code above one.

    The mark covers only the marked object. The walk reads the own code of a user function
    above it, and `_mark_decision` refuses a user class instance above it, so no unwrap passes
    either. A library wrapper (a `contextlib.contextmanager` helper, `functools.cache`) is not a
    stop. Without a mark on the chain nothing stops, and the unwrap goes to the end, as on main.
    Each unwrap with this stop is `unwrap`, which reads the chain as `_mark_decision` does, and
    raises where a `__wrapped__` lookup raises, as `inspect.unwrap` does on main.
    """
    if _mark_on_chain(obj) is None:
        return False
    return (
        fingerprinted_by_name(obj)
        or (inspect.isfunction(obj) and _written_in_user_source(obj))
        or _is_user_class_instance(obj)
    )


def _source_object(func: Any) -> Any:
    """What a read of `func`'s source parses, or None where that read would reach a mark.

    `inspect.getsource` follows `__wrapped__` to the end of the chain with no stop. Without a
    mark on the chain this returns that end, so the read is the one main makes. With a mark,
    `_mark_decision` decides: an object recorded by name, or a chain with a user class instance
    above the mark (which the walk refuses), gives None. Otherwise the unwrap stops at
    `_stop_at_mark`: for a user function decorator above the mark this returns the decorator's
    code object, whose own lines `inspect.getsource` reads (a code object has no
    `__wrapped__`); for the mark itself it returns None. No source read reaches a marked body.
    """
    decision = _mark_decision(func)
    if decision is _MarkDecision.BY_NAME or decision is _MarkDecision.REFUSED:
        return None
    target = unwrap(func, stop=_stop_at_mark)
    if fingerprinted_by_name(target):
        return None
    if _stop_at_mark(target):
        return target.__code__
    return target


def _by_name_hash(func: Any) -> str:
    """The code hash of a function the walk records by name: a hash of that name."""
    return xxhash.xxh64(f"by-name:{_library_name(func)}".encode()).hexdigest()


def is_user_code(obj: Any) -> bool:
    """Check if object is user code (not stdlib/site-packages/builtins, nor pivot's own
    packages: ``is_library_path``)."""
    _t = metrics.start()
    try:
        if obj is None:
            return False

        # Check cache (TypeError if obj not weakly referenceable)
        with contextlib.suppress(TypeError):
            if (cached := _is_user_code_cache.get(obj)) is not None:
                return cached

        result = _is_user_code_impl(obj)

        with contextlib.suppress(TypeError):
            _is_user_code_cache[obj] = result

        return result
    finally:
        metrics.end("fingerprint.is_user_code", _t)


def _is_user_code_impl(obj: Any) -> bool:
    """Internal implementation of is_user_code."""
    # A function the walk records by name (`_recorded_by_name`) is library code: the walk
    # records it by name and never descends into its body.
    if _recorded_by_name(obj):
        return False

    module = _get_module(obj)
    if module is None:
        return False

    # Built-in modules (sys, builtins, _io, etc.) are not user code
    module_name = getattr(module, "__name__", "")
    if module_name in sys.builtin_module_names:
        return False

    # A namespace package (PEP 420) has a __path__ but no __file__. It is a library's when one of
    # its directories is in site-packages or dist-packages or in pivot's own packages. A directory
    # under the interpreter's prefix alone, as pip's checkout of a `pip install -e git+...`
    # package in `<venv>/src`, leaves a namespace the user shares with it user code.
    if not hasattr(module, "__file__") or module.__file__ is None:
        if not hasattr(module, "__path__"):
            return True  # No __file__ and no __path__: user code (exec/notebook/interactive)
        for entry in module.__path__:
            resolved = pathlib.Path(entry).resolve()
            if any(part in _SITE_PACKAGE_PATHS for part in resolved.parts) or any(
                resolved.is_relative_to(prefix) for prefix in _PIVOT_PACKAGE_DIRS
            ):
                return False
        return True

    return not is_library_path(pathlib.Path(module.__file__).resolve())


def is_library_path(path: pathlib.Path) -> bool:
    """Whether resolved file or directory ``path`` belongs to a library, not to user code: it
    lies under the interpreter's prefix (the stdlib, or a virtualenv), in a site-packages or
    dist-packages directory, or in one of pivot's own packages (``_PIVOT_PACKAGE_DIRS``), which
    an editable install puts outside site-packages."""
    if any(part in _SITE_PACKAGE_PATHS for part in path.parts):
        return True
    return any(path.is_relative_to(prefix) for prefix in (*_STDLIB_PATHS, *_PIVOT_PACKAGE_DIRS))


def _get_module(obj: Any) -> types.ModuleType | None:
    """Get module for an object, handling both modules and module members."""
    if isinstance(obj, types.ModuleType):
        return obj

    if not hasattr(obj, "__module__"):
        return None

    module_name = obj.__module__
    if module_name == "builtins":
        return None

    return sys.modules.get(module_name)
