import ast
import atexit
import collections
import contextlib
import contextvars
import dataclasses
import datetime
import decimal
import enum
import fractions
import functools
import gc
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
import operator
import os
import pathlib
import re
import sys
import textwrap
import types
import typing
import uuid
import weakref
from collections.abc import (
    Callable,
    Collection,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
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
# Version 34 walks the class of each user class instance a partial's arguments, a loader's config,
# a dataclass field default or a StageParams default holds, as a stage that names the class gets.
# Version 35 records each value by its form over a closed type table (`_form`) and walks the code
# in it from one node table per stage (`_Walk`), keyed as a stage that names it is keyed: there
# are no `held:` entries, and a cache hit replays the reads of the code inside a value.
_CACHE_SCHEMA_VERSION = 38

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
# Maps rel_path -> content digest (`_source_digest`). Set by _collecting_sources(). A context
# variable, so a fingerprint in one thread (the engine fingerprints in worker threads, each with
# a copy of the context) never sees, or leaves behind, another thread's map.
_active_source_map: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "_active_source_map", default=None
)

# Stage name used for error reporting during fingerprinting.
_current_stage_name: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_stage_name", default=None
)

# Unsafe capture errors collected while recursively walking one stage's dependencies.
_capture_errors: contextvars.ContextVar[list["_Refusal"] | None] = contextvars.ContextVar(
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

# Steps from the node being expanded to the object its expansion is visiting (`_Walk`).
_walk_path: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "_walk_path", default=()
)


class _NodeKind(enum.Enum):
    """What a node of a `_Walk` is, which decides the edges its expansion follows."""

    FUNCTION = "function"  # a user function, or a C-level wrapper of one
    CLASS = "class"  # a user class
    LOADER_CLASS = "loader-class"  # a pivot loader class: its `load`, `save` and `empty`
    LOADER = "loader"  # a loader a stage reads or writes through: a projection root

    # A member is its own only instance, so it hashes by identity, in C: `_Walk.node` looks one
    # up for every edge, and Enum's own `__hash__` is Python code.
    __hash__ = object.__hash__


@dataclasses.dataclass(frozen=True, slots=True)
class _Refusal:
    """A refusal a node's expansion made: the steps from the node to where it was made, its
    message, and whether unsafe fingerprinting turns it into a warning."""

    path: tuple[str, ...]
    message: str
    downgradable: bool


@dataclasses.dataclass(slots=True, eq=False)
class _Node:
    """One code object of a `_Walk` and what its own edges found. Paths are relative to the node:
    `reads` holds each read's path, the object it reads from and its step (its token is taken only
    when a projection records reads), `children` each edge to another node, `refusals` what its
    edges refused, and `sources` the source files of the code its edges hashed."""

    obj: Any  # held, so its id is not reused while the walk lives
    kind: _NodeKind
    key: str  # the reference its entries and its children's bindings are keyed by
    entries: dict[str, str] = dataclasses.field(default_factory=dict[str, str])
    reads: list[tuple[tuple[str, ...], Any, str]] = dataclasses.field(
        default_factory=list[tuple[tuple[str, ...], Any, str]]
    )
    children: list[tuple[tuple[str, ...], "_Node"]] = dataclasses.field(
        default_factory=list[tuple[tuple[str, ...], "_Node"]]
    )
    refusals: list[_Refusal] = dataclasses.field(default_factory=list[_Refusal])
    sources: dict[str, str] = dataclasses.field(default_factory=dict[str, str])


@typing.final
class _Walk:
    """The code a stage reaches, as one node table: the walk's only memo and only cycle guard.

    `node` makes a node once per object and kind, before any of its edges runs, and puts it on the
    worklist; reaching it again adds an edge and nothing else. `run` expands each node once
    (`_expand`), from the worklist, so no node's walk recurses into another's and a cycle ends
    where it meets a node that exists. A node stores only what its own edges give, so nothing it
    stores depends on the order nodes were reached in; `_project` unions the nodes breadth-first
    from a root. A stage function's walk is its own (`_stage_manifest`), and the loaders of a stage
    share a second one (`_loaders_walk`), so code two of them reach is expanded once; a call outside
    a scope makes its own. A refusal stays in the node that made it, so nothing that outlives the
    walk keeps one.
    """

    __slots__ = ("nodes", "pending")

    def __init__(self) -> None:
        self.nodes = dict[tuple[int, _NodeKind], _Node]()
        self.pending = collections.deque[_Node]()

    def node(self, obj: Any, kind: _NodeKind, key: str) -> _Node:
        """The node of `obj` as `kind`, made with `key` and put on the worklist when it is new."""
        found = self.nodes.get((id(obj), kind))
        if found is None:
            found = _Node(obj, kind, key)
            self.nodes[(id(obj), kind)] = found
            self.pending.append(found)
        return found

    def run(self) -> None:
        """Expand every node on the worklist, and each node their edges make."""
        while self.pending:
            _expand(self.pending.popleft(), self)


# The walk the loaders of the stage `stage_fingerprint_scope` is fingerprinting share; None
# outside a scope. The stage function has a walk of its own (`_stage_manifest`): its manifest is
# cached and its loaders' are not, so a node both reach would take its key from whichever root
# reached it first, which differs between a cache hit and a cold walk.
_loaders_walk: contextvars.ContextVar[_Walk | None] = contextvars.ContextVar(
    "_loaders_walk", default=None
)

# The node being expanded and its walk (`_expand`); None outside an expansion.
_expanding: contextvars.ContextVar[tuple[_Node, _Walk] | None] = contextvars.ContextVar(
    "_expanding", default=None
)

# The ids of the classes whose identity is being computed (`_class_identity`), outermost first.
_identity_active: contextvars.ContextVar[tuple[int, ...]] = contextvars.ContextVar(
    "_identity_active", default=()
)


# Per-pass memo, keyed by kind and object id, for function identities (`function_identity`),
# type-hint classes (`_hint_classes`), user-class members (`_user_methods`), mark decisions
# (`_mark_decision`), class identities (`_class_identity`) and the parts of classes with no source
# (`_sourceless_parts`), and by file name for source file digests (`_source_digest`).
# Set only inside `fingerprint_pass()`; a reload, a mutated cell or an edit is seen by the next
# pass.
_pass_memo: contextvars.ContextVar[dict[tuple[str, int | str], Any] | None] = (
    contextvars.ContextVar("_pass_memo", default=None)
)


@contextlib.contextmanager
def fingerprint_pass() -> Iterator[None]:
    """Share function identities, type-hint classes, user-class members, mark decisions, class
    identities and source digests across one pass over many stages.

    A cache hit re-reads every recorded value, and stages of one pipeline read the same
    helpers, model classes and source files: inside the pass each is named or read once. The
    memo holds strong references, so an id is never reused within the pass, and it ends with
    the pass.
    """
    if _pass_memo.get() is not None:
        yield
        return
    token = _pass_memo.set({})
    try:
        yield
    finally:
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
    a cell. Values take their form (`_form`); an ArtifactHandle default is a wired input and is
    named only as one. Library code is named by module and qualname only.

    Used by the stage-manifest cache key, the form of a function value (loader config, partial
    arguments), and compose's check that one stage name is one stage. Within one
    `fingerprint_pass()` each function is named once.
    """
    return _memoized("identity", func, lambda: _FormBuilder().identity(func))


_EMPTY_CELL = object()


def _cell_value(cell: types.CellType) -> Any:
    """Return a closure cell's value, or `_EMPTY_CELL` for a name not yet bound."""
    try:
        return cell.cell_contents
    except ValueError:
        return _EMPTY_CELL


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
    source_map = dict[str, str]()
    token = _active_source_map.set(source_map)
    try:
        yield source_map
    finally:
        _active_source_map.reset(token)


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
    HOOKED = "hooked"  # a class whose metaclass answers attribute reads with user code


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
        return _form(value).text
    if isinstance(value, re.Pattern):
        return _form(value).text
    return None


def _value_kind(value: Any) -> _ValueKind:
    """Classify a value a walked function reads; the walk and a cache hit share this. Decided on
    the value's real type, so no `__class__` or `__getattribute__` of a user instance runs, and a
    class whose metaclass answers attribute reads with user code is HOOKED before anything of it
    is read (`_attribute_hook`).

    A value on a `__wrapped__` chain that holds a `@fingerprint_by_name` mark takes the mark
    decision (`_mark_decision`) first: a user class instance above the mark is
    CLASS_ABOVE_MARK, and an object the walk records by name is EXTERNAL, also when it is a
    marked instance of a class-based decorator.
    """
    mark = _mark_decision(value)
    if mark is _MarkDecision.REFUSED:
        return _ValueKind.CLASS_ABOVE_MARK
    value_type: type[Any] = type(value)  # pyright: ignore[reportUnknownVariableType] - any class
    if issubclass(value_type, type) and _attribute_hook(value) is not None:
        return _ValueKind.HOOKED
    # functools.partial and partialmethod fail is_user_code() (module is functools/stdlib).
    # `issubclass` with a class whose metaclass is `type` asks only the real MRO, in C.
    if issubclass(value_type, (functools.partial, functools.partialmethod)):
        return _ValueKind.PARTIAL
    if mark is _MarkDecision.BY_NAME and not _carries_instance(value):
        return _ValueKind.EXTERNAL
    if _is_user_class_instance(value) or _is_callable_instance(value) or _carries_instance(value):
        return _ValueKind.REFUSED
    if callable(value):
        return _ValueKind.USER if is_user_code(value) else _ValueKind.EXTERNAL
    if issubclass(value_type, types.ModuleType):
        return _ValueKind.MODULE
    if issubclass(value_type, logging.Logger):
        return _ValueKind.LOGGER
    if _immutable_value_text(value) is not None:
        return _ValueKind.SCALAR
    return _ValueKind.REFUSED


def _value_token(value: Any) -> str:
    """Summarise the walk's own entries for a read value; what it reads has its own steps.

    A user function or class is named by its identity, closure values and defaults included,
    so a binding that now holds another closure from the same factory reads as changed even
    where the walk did not descend into it again. A partial is named by its form (`_form`),
    which holds every value its arguments hold and names each class in them by its key.
    """
    kind = _value_kind(value)
    if kind is _ValueKind.PARTIAL:
        return f"partial:{xxhash.xxh64(_form(value).text.encode()).hexdigest()}"
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
    """Like `_value_token`, for a default argument (see `_process_default_values`): a value it
    records by its form (`_is_formed_default`) is named by that form."""
    from pivot import compose

    if _is_formed_default(value):
        return f"default:{xxhash.xxh64(_form(value).text.encode()).hexdigest()}"
    if _is_artifact_handles(value, compose.ArtifactHandle):
        return "handles"
    return _value_token(value)


def _is_formed_default(value: Any) -> bool:
    """Whether a default argument is recorded by its form (`_process_default_values`): a
    `StageParams` instance, or an instance of a user subclass of a scalar type (a `str` subclass,
    an IntEnum member). Decided on its real type's MRO, so no code of the value runs."""
    from pivot import stage_def

    value_type: type[Any] = type(value)  # pyright: ignore[reportUnknownVariableType] - any class
    if _on_mro(value_type, stage_def.StageParams):
        return True
    return (
        id(value_type) not in _SCALAR_TYPE_IDS
        and issubclass(value_type, _SCALAR_TYPES)
        and is_user_code(value_type)
    )


def _on_mro(cls: type[Any], base: type[Any]) -> bool:
    """Whether `base` is on `cls`'s real MRO (`_class_mro`), compared by identity: no
    `__subclasscheck__`, `__eq__` or `__hash__` of a metaclass runs."""
    return any(klass is base for klass in _class_mro(cls))


def _type_of(value: Any) -> type[Any]:
    """`type(value)`, typed: the real type, read in C, which no `__class__` a value defines
    changes."""
    return cast("type[Any]", type(value))


# A walk step names one move from an object to what it holds; `_read_step` follows each with the
# accessor the walk and `_form` used, so a cache hit reaches what the walk reached.
#
# Code edges: `g:NAME` a global (also recorded when absent), `c:NAME` a closure cell, `d:PARAM` a
# default, `a:ATTR` a module attribute, `ca:ATTR` a class attribute (`Config.SCALE`, or `self.S`
# in a member), read from the class body that holds it (`_class_attribute`), `mc` a pydantic
# model's `model_config` a member reads, `n:NAME` and `np:DOTTED` names in a class statement's
# annotations, `m:NAME` a callable member of a user class's own namespace, `b:INDEX` a class's
# base, `ob:INDEX` a base its class statement names that `__bases__` does not hold
# (`_statement_bases`), `fd:NAME` a dataclass field default, `bi:MODULE` a loaded module a function
# body imports, `lc` the native settings of a C cache wrapper, `hc:INDEX` a class in type hints,
# `fc:INDEX` a class in a model's field annotations, `lm:NAME` a loader class's `load`, `save` or
# `empty`, `lr:REF` a member those reach, `cfg:NAME` a loader's config field, `sd:NAME` a model's
# field default, and `ns:NAME` an entry of the namespace of a class with no source. `hints` and
# `schema` read the visited object itself.
#
# Value steps, from a value to an object inside it (`_form`): `pf`, `pa:INDEX` and `pk:KEY` a
# partial's parts, `i:INDEX` a list's, tuple's or deque's item, `s:RANK` a set's member by its
# rank among the sorted member forms (`_ranked`), `k:KEY` and `kk:KEY` a mapping's value and key by
# the key's form, `at:NAME` an instance attribute, `sl:LABEL` a slot (`_slot_values`), `bv` the
# built-in value of an instance of a subclass of a built-in type (`_builtin_value`), `cls` the
# class of an instance or a loader, `ea` an exception's arguments, `ec:NAME` a field of a
# built-in exception (`_exception_fields`), `ev` an Enum member's value, `mf` and `ms` a
# bound method's function and object, `cm` the function of a classmethod or staticmethod, `pg`,
# `ps` and `pd` a property's getter, setter and deleter, `cp` a cached property's function, `wr`
# the function a cache wrapper wraps, `df` a defaultdict's factory, `mp:INDEX` a ChainMap's map,
# `nf:INDEX` an item of an object array, `to` and `ta` a type alias's origin and arguments, and
# `px` the mapping a `MappingProxyType` shows.
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
            case "a" | "cfg":
                return getattr(owner, arg)
            case "ca":
                return _class_attribute(owner, arg)
            case "mc":
                return _class_attribute(owner, "model_config")
            case "n":
                return vars(sys.modules[owner.__module__])[arg]
            case "np":
                namespace = vars(sys.modules[owner.__module__])
                return _resolve_dotted_path(tuple(arg.split(".")), namespace)
            case "m":
                return dict(_user_methods(owner)).get(arg, _UNREAD)
            case "b":
                return _class_bases(owner)[int(arg)]
            case "ob":
                return _statement_bases(owner)[int(arg)]
            case "fd":
                return _dataclass_field_value(owner, arg)
            case "bi":
                module = _body_import_module(arg)
                return _UNREAD if module is None else module
            case "lc":
                attributes = _instance_dict(_type_of(owner), owner) or {}
                settings = _lru_cache_settings(attributes.get("cache_parameters"))
                return _UNREAD if settings is None else settings
            case "hc":
                return _hint_classes(owner)[int(arg)]
            case "fc":
                return _model_field_classes(owner)[int(arg)]
            case "lm":
                method = _user_defined_method(owner, arg)
                return _UNREAD if method is None else method
            case "lr":
                return dict(_reachable_loader_members(owner)).get(arg, _UNREAD)
            case "cls":
                return cast("type[Any]", type(owner))
            case _:
                return _value_part(owner, kind, arg)
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


def _value_part(owner: Any, kind: str, arg: str) -> Any:
    """What a value step (`_read_step`) reaches inside a value, read with the accessor `_form`
    reads it with."""
    match kind:
        case "pf":
            return owner.func
        case "pa":
            return owner.args[int(arg)]
        case "pk":
            return owner.keywords[arg]
        case "i":
            return owner[int(arg)]
        case "s":
            members = owner.items if type(owner) is _Contents else list(owner)
            order, ranks = _ranked([_form(member).text for member in members])
            rank = int(arg)
            return next(members[index] for index in order if ranks[index] == rank)
        case "k" | "kk":
            key, item = next(
                (key, item) for key, item in _mapping_items(owner) if _form(key).text == arg
            )
            return item if kind == "k" else key
        case "at":
            return cast("dict[str, Any]", _instance_dict(_type_of(owner), owner))[arg]
        case "sl":
            return dict(_slot_values(_type_of(owner), owner))[arg]
        case "bv":
            return _builtin_value(owner)
        case "ea":
            return _exception_args(owner)
        case "ec":
            return dict(_exception_fields(_type_of(owner), owner))[arg]
        case "ev":
            return cast("dict[str, Any]", _instance_dict(_type_of(owner), owner))["_value_"]
        case "mf" | "cm":
            return owner.__func__
        case "ms":
            return owner.__self__
        case "pg":
            return owner.fget
        case "ps":
            return owner.fset
        case "pd":
            return owner.fdel
        case "cp":
            return owner.func
        case "wr":
            return cast("dict[str, Any]", _instance_dict(_type_of(owner), owner))["__wrapped__"]
        case "df":
            return owner.default_factory
        case "mp":
            return owner.maps[int(arg)]
        case "nf":
            return owner.flat[int(arg)]
        case "to":
            return _GENERIC_ALIAS_ORIGIN(owner)
        case "ta":
            if _on_mro(_type_of(owner), types.GenericAlias):
                return _GENERIC_ALIAS_ARGS(owner)
            return _UNION_ARGS(owner)
        case "px":
            return _proxied_mapping(owner)
        case "ns":
            return _class_namespace(owner)[arg]
        case "sd":
            return owner.model_fields[arg].default
        case _:
            return _UNREAD


def _mapping_items(mapping: Any) -> Iterator[tuple[Any, Any]]:
    """The items of a mapping `_form` writes by its entries (a dict, a `Counter`, an
    `OrderedDict`, a `MappingProxyType` over a dict, the items of a dict subclass's built-in value:
    `_Contents`), read through the built-in types' own C methods, so no method of a dict subclass,
    of the mapping a proxy shows or of a key runs."""
    if type(mapping) is _Contents:
        return iter(mapping.items)
    if type(mapping) is types.MappingProxyType:
        return iter(_dict_items(_proxied_mapping(mapping)))
    return iter(_dict_items(mapping))


def _read_token(owner: Any, step: str) -> str:
    """Return the token for what `step` reads from `owner` now.

    Equal tokens mean the walk makes the same decision there and adds the same entries for
    the value itself. A value the walk records by its form is named by that form, which holds
    the key of each class in it, so a cache hit forms it again and walks no code in it.
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
    value = _read_step(owner, step)
    if value is _UNREAD:
        return "absent"
    if step in ("mc", "lc"):
        return f"config:{xxhash.xxh64(_form(value).text.encode()).hexdigest()}"
    if step.startswith("d:"):
        return _default_token(value)
    if step.startswith("ca:") and _is_user_enum_member(value):
        return f"enum:{_enum_member_text(value)}"
    return _value_token(value)


def _record_read(owner: Any, step: str) -> None:
    """Record, in the node being expanded, that it reads `step` from `owner` at the current path;
    a projection that records reads takes its token (`_project`)."""
    expanding = _expanding.get()
    if expanding is not None:
        expanding[0].reads.append(((*_walk_path.get(), step), owner, step))


@contextlib.contextmanager
def _walking(*steps: str) -> Iterator[None]:
    """Visit the value that `steps` reach from the object being visited."""
    token = _walk_path.set((*_walk_path.get(), *steps))
    try:
        yield
    finally:
        _walk_path.reset(token)


def _recorded_reads_hold(func: Callable[..., Any], reads: list[object]) -> bool:
    """Return whether every read recorded from `func` gives its recorded token again.

    A replay reads values and takes tokens only: it expands no node, so nothing it meets can
    record a refusal or an entry (`_record_refusal` outside an expansion would report one)."""
    assert _expanding.get() is None, "a cache hit is checked inside a node's expansion"
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
    """Follow `path` from the stage function, reusing the objects already reached (the empty path
    is the function), step by step, so a path of any length is followed."""
    reached = len(path)
    while path[:reached] not in owners:
        reached -= 1
    owner = owners[path[:reached]]
    for end in range(reached + 1, len(path) + 1):
        owner = owner if owner is _UNREAD else _read_step(owner, path[end - 1])
        owners[path[:end]] = owner
    return owner


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


def get_stage_fingerprint(func: Callable[..., Any]) -> dict[str, str]:
    """Generate fingerprint manifest capturing all code dependencies.

    Returns dict with keys:
    - 'self:<name>': Function itself (hash)
    - 'func:<module>.<qualname>': Referenced helper functions (hash, transitive)
    - 'class:<module>.<qualname>': Referenced class definitions (hash, transitive)
    - 'const:<module>.<name>': Primitive constants, read as globals or as `module.NAME`
    - 'ext:<module>.<name>': Library functions and classes, named by definition

    The stage function's walk is a node table of its own (`_stage_manifest`; its loaders share
    another, `_loaders_walk`), and the manifest is the projection of `func`'s node (`_project`).
    """
    _t = metrics.start()
    try:
        with _collected_capture_errors():
            return _stage_manifest(func, recording=False)[0]
    finally:
        metrics.end("fingerprint.get_stage_fingerprint", _t)


def _stage_manifest(
    func: Callable[..., Any], *, recording: bool
) -> tuple[dict[str, str], list[tuple[tuple[str, ...], str]]]:
    """The manifest of stage function `func`: its `self:` hash and the projection of its node, in
    a walk of its own (`_loaders_walk` says why); with `recording`, also the reads that projection
    takes."""
    walk = _Walk()
    root = walk.node(func, _NodeKind.FUNCTION, _definition_ref(func))
    walk.run()
    manifest = {f"self:{getattr(func, '__name__', '<lambda>')}": hash_function_ast(func)}
    reads = _project(root, manifest, recording=recording)
    return manifest, reads


def _expand(node: _Node, walk: _Walk) -> None:
    """Run `node`'s own edges, once: their entries, reads, children and refusals go into the node
    (`_record_read`, `_add_child`, `_record_refusal`), with paths relative to it, and, while a
    source map is collected (`_collecting_sources`), the source files of the code they hash into
    its `sources`. Only a cacheable stage walk collects one, and it expands every node of its own
    walk inside that collection; a loader's walk is never cached, so it reads no source file."""
    outer_sources = _active_source_map.get()
    sources_token = _active_source_map.set(None if outer_sources is None else node.sources)
    expanding_token = _expanding.set((node, walk))
    path_token = _walk_path.set(())
    try:
        match node.kind:
            case _NodeKind.FUNCTION:
                _expand_function(node.obj)
            case _NodeKind.CLASS:
                _expand_class(node.obj)
            case _NodeKind.LOADER_CLASS:
                _expand_loader_class(node.obj, node.key)
            case _NodeKind.LOADER:
                _expand_loader(node.obj, node.key)
    finally:
        _walk_path.reset(path_token)
        _expanding.reset(expanding_token)
        _active_source_map.reset(sources_token)


def _project(
    root: _Node, manifest: dict[str, str], *, recording: bool = False
) -> list[tuple[tuple[str, ...], str]]:
    """Add to `manifest` the entries of every node `root` reaches, breadth-first, each node's
    edges taken in the order of their paths, and report each node's refusals with the path that
    first reaches it (`_with_reach`). Node keys name what each entry records, never where it is
    reached, so the union does not depend on the order. With `recording`, return each node's
    reads under that first path, with the token each gives now (`_read_token`)."""
    reads = list[tuple[tuple[str, ...], str]]()
    queue = collections.deque([(root, tuple[str, ...]())])
    seen = {id(root)}
    while queue:
        node, path = queue.popleft()
        for key, value in node.entries.items():
            manifest.setdefault(key, value)
        if (active_sources := _active_source_map.get()) is not None:
            for rel_path, digest in node.sources.items():
                active_sources.setdefault(rel_path, digest)
        if recording:
            reads.extend(
                ((*path, *steps), _read_token(owner, step)) for steps, owner, step in node.reads
            )
        for refusal in node.refusals:
            _report_refusal(
                refusal.message,
                downgradable=refusal.downgradable,
                path=(*path, *refusal.path),
            )
        for steps, child in sorted(node.children, key=lambda edge: edge[0]):
            if id(child) not in seen:
                seen.add(id(child))
                queue.append((child, (*path, *steps)))
    return reads


# How a refusal's reach names each step (`_with_reach`), by the step's kind (`_read_step`).
_STEP_WORDS = {
    "g": "global '{}'",
    "c": "closure variable '{}'",
    "d": "the default of parameter '{}'",
    "a": "module attribute '{}'",
    "ca": "class attribute '{}'",
    "n": "'{}' in its class statement",
    "np": "'{}' in its class statement",
    "m": "member '{}'",
    "b": "base {}",
    "ob": "base {} its class statement names",
    "fd": "the default of field '{}'",
    "bi": "module '{}' its body imports",
    "hc": "class {} of its type hints",
    "fc": "class {} of its field annotations",
    "lm": "its method '{}'",
    "lr": "member '{}'",
    "cls": "its class",
    "pf": "the partial's function",
    "pa": "argument {}",
    "pk": "keyword argument '{}'",
    "i": "item {}",
    "s": "set member {}",
    "k": "the value at key {}",
    "kk": "the key {}",
    "at": "attribute '{}'",
    "sl": "slot {}",
    "bv": "its built-in value",
    "ea": "its arguments",
    "ec": "its field '{}'",
    "mf": "the method's function",
    "ms": "the object the method is bound to",
    "cm": "its function",
    "pg": "the property's getter",
    "ps": "the property's setter",
    "pd": "the property's deleter",
    "cp": "its function",
    "wr": "the function it wraps",
    "ev": "its value",
    "df": "its default factory",
    "mp": "map {}",
    "nf": "item {}",
    "to": "its origin",
    "ta": "its arguments",
    "px": "the mapping it shows",
    "mc": "its model_config",
    "ns": "entry '{}' of its namespace",
    "sd": "the default of field '{}'",
    "cfg": "config field '{}'",
}


# How many ways a merged refusal lists (`_merged_refusals`); the rest are counted.
_REACHES_SHOWN = 3


def _with_reach(message: str, paths: list[tuple[str, ...]]) -> str:
    """Render a grouped refusal once, with its first distinct reach paths in first-seen order."""
    lines = list[str]()
    for path in paths:
        if not path:
            continue
        words = list[str]()
        for step in path:
            kind, _, arg = step.partition(":")
            phrase = _STEP_WORDS.get(kind, step)
            words.append(phrase.format(arg) if "{}" in phrase else phrase)
        line = f"Reached through: {' -> '.join(words)}."
        if line not in lines:
            lines.append(line)
    shown = lines[:_REACHES_SHOWN]
    if len(lines) > _REACHES_SHOWN:
        shown.append(f"Reached through {len(lines) - _REACHES_SHOWN} more paths too.")
    head, newline, rest = message.partition("\n")
    return "\n".join([head, *shown]) + f"{newline}{rest}"


def _merged_refusals(refusals: list[_Refusal]) -> list[str]:
    """Group unchanged refusal bodies and their structured paths before rendering any text."""
    reaches = dict[str, list[tuple[str, ...]]]()
    for refusal in refusals:
        paths = reaches.setdefault(refusal.message, [])
        if refusal.path not in paths:
            paths.append(refusal.path)
    return [_with_reach(message, paths) for message, paths in reaches.items()]


@contextlib.contextmanager
def _collected_capture_errors() -> Iterator[None]:
    """Collect unsafe-capture errors and raise them together, unless a caller already does.

    Each projection reports the refusals of the nodes it reaches, so one variable read by a
    loader's `load()` and `save()`, or by one loader on two inputs, is reported several times,
    each with its reach; `_merged_refusals` names each refusal once, with its reaches.
    """
    if _capture_errors.get() is not None:
        yield
        return
    errors = list[_Refusal]()
    token = _capture_errors.set(errors)
    try:
        yield
    finally:
        _capture_errors.reset(token)
    if errors:
        raise exceptions.StageDefinitionError("\n\n".join(_merged_refusals(errors)))


def refuse(message: str) -> None:
    """Refuse the stage being fingerprinted: inside `stage_fingerprint_scope` the refusal is
    reported with the stage's other refusals when the scope ends, else it is raised now. Unsafe
    fingerprinting does not downgrade it (`loaded_source.check_worker_imports`)."""
    _record_refusal(message, downgradable=False)


def _record_refusal(message: str, *, downgradable: bool) -> None:
    """Refuse the stage with `message`. During a node's expansion the refusal is kept in that node,
    with the path to where it was made, and reported by each projection that reaches the node
    (`_project`): nothing that outlives the walk holds it. Elsewhere it is reported now
    (`_report_refusal`)."""
    expanding = _expanding.get()
    if expanding is None:
        _report_refusal(message, downgradable=downgradable)
        return
    expanding[0].refusals.append(_Refusal(_walk_path.get(), message, downgradable))


def _report_refusal(message: str, *, downgradable: bool, path: tuple[str, ...] = ()) -> None:
    """Collect the refusal and its absolute path, rendering only when it is reported."""
    if downgradable and _is_unsafe_fingerprinting_enabled():
        _unsafe_capture_warning.set(True)
        if (scope := _stage_scope.get()) is not None:
            scope["unsafe"] = True
        _logger.warning(_with_reach(message, [path]))
        return
    errors = _capture_errors.get()
    if errors is None:
        raise exceptions.StageDefinitionError(_with_reach(message, [path]))
    errors.append(_Refusal(path, message, downgradable))


class StageFingerprintScope(TypedDict):
    """What `stage_fingerprint_scope` learned about one stage's fingerprint."""

    unsafe: bool  # a refusal in it was downgraded to a warning by unsafe fingerprinting


@contextlib.contextmanager
def stage_fingerprint_scope(stage_name: str) -> Iterator[StageFingerprintScope]:
    """Name the stage being fingerprinted, raise all of its unsafe-capture errors together, and
    report whether unsafe fingerprinting downgraded any of them to a warning (`unsafe`).

    Wraps a stage's manifest and its input and output loader fingerprints, so a refusal found
    in loader config is reported with the stage's own refusals. The loaders are walked in one
    node table (`_Walk`, `_loaders_walk`), so code two of them reach is walked once.
    """
    scope = StageFingerprintScope(unsafe=False)
    token = _current_stage_name.set(stage_name)
    scope_token = _stage_scope.set(scope)
    walk_token = _loaders_walk.set(_Walk())
    try:
        with _collected_capture_errors():
            yield scope
    finally:
        _loaders_walk.reset(walk_token)
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
    with state_dir_mod.entry_invocation():
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
        token = _current_stage_name.set(stage_name)
        try:
            # Compute with source tracking
            with _collecting_sources() as source_map, _collected_capture_errors():
                manifest, reads = _stage_manifest(func, recording=True)

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
            _current_stage_name.reset(token)
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


def _current_node() -> _Node:
    """The node being expanded (`_expand`)."""
    return cast("tuple[_Node, _Walk]", _expanding.get())[0]


def _entries() -> dict[str, str]:
    """The entries of the node being expanded."""
    return _current_node().entries


def _add_child(obj: Any, kind: _NodeKind, key: str) -> _Node:
    """The node of `obj` as `kind` (made with `key` when new), with an edge to it from the node
    being expanded at the current path."""
    node, walk = cast("tuple[_Node, _Walk]", _expanding.get())
    child = walk.node(obj, kind, key)
    node.children.append((_walk_path.get(), child))
    return child


def _add_callable(value: Any, binding_ref: str = "", *, in_value: bool = False) -> None:
    """Add an edge from the node being expanded to user function or class `value`, at the current
    path, with the expanding node's entry for it: `func:<key>` or `class:<key>`, its code hash, and
    the coverage of its module. A function is keyed by `_callable_ref` of the binding it is reached
    through (`binding_ref`), a class by `_class_key`; a node keeps the key it was made with. A
    class whose metaclass answers attribute reads with user code (`_attribute_hook`) is refused
    instead, before anything of it is read."""
    if issubclass(type(value), type):
        if _attribute_hook(value) is not None:
            _record_refusal(_hooked_class_message(value), downgradable=True)
            return
        prefix, kind, key = "class", _NodeKind.CLASS, _class_key(value, in_value=in_value)
        module_name = _class_module(value)
    else:
        prefix, kind = "func", _NodeKind.FUNCTION
        key = _callable_ref(value, binding_ref)
        module_name = getattr(value, "__module__", None)
    entries = _entries()
    if isinstance(module_name, str):
        _mark_module_covered(module_name, entries)
    child = _add_child(value, kind, key)
    entries[f"{prefix}:{child.key}"] = hash_function_ast(value)


def _add_loader_class(loader_type: type, key: str) -> None:
    """Add an edge from the node being expanded to the node of pivot loader class `loader_type`
    (made with `key` when new), whose own entries are its `load`, `save` and `empty`
    (`_expand_loader_class`)."""
    _add_child(loader_type, _NodeKind.LOADER_CLASS, key)


def _check_dynamic_access(obj: Any, tree: ast.Module) -> None:
    """`_check_dynamic_name_access`, with its refusal kept with the stage's others."""
    try:
        _check_dynamic_name_access(obj, tree)
    except exceptions.StageDefinitionError as error:
        _record_refusal(str(error), downgradable=False)


def _expand_function(func: Callable[..., Any]) -> None:
    """The edges of a function node (`_Walk`): the globals and closure cells it reads, the
    defaults it takes, the classes in its type hints and the user modules its body imports.
    Every wrapper's immediate target (`wr`) takes normal user/library/value admission.
    Looping metadata is refused; terminating C caches retain their native settings."""
    if (message := _wrapper_loop_message(func)) is not None:
        _record_refusal(message, downgradable=False)
        return
    if type(func) is _LRU_CACHE_TYPE:
        _record_read(func, "lc")
        settings = _read_step(func, "lc")
        binding = _current_node().key
        if settings is _UNREAD:
            _admit(_form(func), binding, _bound_opaque_message(binding))
            return
        _entries()[f"default:{binding}.cache_parameters"] = xxhash.xxh64(
            _form(settings).text.encode()
        ).hexdigest()
    wrapped = _read_step(func, "wr")
    if wrapped is not _UNREAD:
        _record_read(func, "wr")
        binding = f"{_current_node().key}.__wrapped__"
        stage_name = _current_stage_name.get() or "<unknown>"
        with _walking("wr"):
            _process_captured_value("__wrapped__", binding, wrapped, func, stage_name, chain=None)
    tree: ast.Module | None = None
    if is_user_code(func):
        tree = _source_tree(func)
        # The refusals read the source `inspect.getsource` returns, which follows a
        # functools.wraps chain to the wrapped function, as on main, and stops at the user
        # decorator above a mark; the body-import walk below reads the wrapper's own source.
        check_tree = _wrapped_source_tree(func) if hasattr(func, "__wrapped__") else tree
        if check_tree is not None:
            _check_dynamic_access(func, check_tree)
    _t_closure = metrics.start()
    try:
        # Use cached result if available (getclosurevars is expensive at ~0.5ms)
        closure_vars = _getclosurevars_cache.get(func)
        if closure_vars is None:
            closure_vars = inspect.getclosurevars(func)
            _getclosurevars_cache[func] = closure_vars
    except (TypeError, AttributeError):
        metrics.end("fingerprint.getclosurevars", _t_closure)
        # The immediate `wr` edge already owns the wrapped function's code and reads. Adding
        # a second, terminal unwrap edge would give distinct functions one binding/key.
        return
    metrics.end("fingerprint.getclosurevars", _t_closure)

    _process_closure_values(closure_vars.globals, func, prefix="g")
    _process_closure_values(closure_vars.nonlocals, func, prefix="c")
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
                _process_closure_values(nested_globals_dict, func, prefix="g")

    _process_default_values(func)
    _process_type_hint_dependencies(func)
    if tree is not None:
        _process_body_imports(func, tree)


def _expand_class(cls: type) -> None:
    """The edges of a class node (`_Walk`): the names its statement's annotations use (`n:`,
    `np:`), each user class among its bases (`b:INDEX`) and among the bases its statement names
    that `__bases__` does not hold (`ob:INDEX`, a `TypedDict`'s), every callable member of its own
    namespace (`m:`), each class attribute a member reads through its receiver (`ca:`), its own
    dataclass field defaults (`fd:`), a pydantic model's schema and field defaults (`schema`,
    `fc:`, `sd:`), and for a class with no source the values its namespace holds (`ns:`).

    Every class's bases are its edges, a class statement's or a `type()` class's alike, and its
    members are its own namespace's: a member it inherits is its base's, walked once under the
    base. So a stage that names a class and one that holds an instance of it walk it alike. A base
    is reached through `__bases__`, not by a name, so it is keyed as a class inside a value is
    (`_class_key`): two bases one factory made keep apart.
    """
    if is_user_code(cls):
        tree = _source_tree(cls)
        if tree is not None:
            _check_dynamic_access(cls, tree)
    _process_class_body_dependencies(cls)
    bases = _class_bases(cls)
    for index, base in enumerate(bases):
        if base is object or not is_user_code(base):
            continue
        step = f"b:{index}"
        _record_read(cls, step)
        with _walking(step):
            _add_callable(base, in_value=True)
    for index, base in enumerate(_statement_bases(cls)):
        if any(base is known for known in bases) or not is_user_code(base):
            continue
        step = f"ob:{index}"
        _record_read(cls, step)
        with _walking(step):
            _add_callable(base, in_value=True)
    _process_user_methods(cls)
    _process_instance_class_attributes(cls)
    _process_dataclass_defaults(cls)
    if hasattr(cls, "model_fields"):
        _process_model_schema(cls)
    if _has_no_source(cls):
        _process_sourceless_namespace(cls)


def _statement_bases(cls: type) -> tuple[type, ...]:
    """The classes `cls`'s class statement names as its bases when `__bases__` holds others: a
    `TypedDict` or a `NamedTuple` keeps only `dict` or `tuple` there, and the bases it was written
    with in `__orig_bases__`. Read from its own namespace, so no code of its metaclass runs; a
    subscripted generic there is skipped, as its class is in `__bases__`."""
    written = _class_namespace(cls).get("__orig_bases__", ())
    if type(written) is not tuple:
        return ()
    return tuple(base for base in cast("tuple[Any, ...]", written) if issubclass(type(base), type))


def _process_user_methods(cls: type) -> None:
    """Walk each callable member of user class `cls`'s own namespace as a stage helper
    (`_user_methods`), or refuse it as opaque state, keyed under the class's node."""
    class_ref = _current_node().key
    stage_name = _current_stage_name.get() or _class_qualname(cls)
    for label, member in _user_methods(cls):
        step = f"m:{label}"
        _record_read(cls, step)
        with _walking(step):
            binding_ref = f"{class_ref}.{label}"
            _process_captured_value(binding_ref, binding_ref, member, cls, stage_name, chain=None)


def _own_dataclass_fields(cls: type) -> tuple[Mapping[str, "dataclasses.Field[Any]"], list[str]]:
    """The dataclass fields `cls`'s own namespace declares, and their names in the order its
    annotations give; none for a class that declares no fields of its own (a base's fields are
    that base's, walked as the base: `_expand_class`)."""
    namespace = _class_namespace(cls)
    fields = namespace.get("__dataclass_fields__")
    if fields is None:
        return {}, []
    annotations = cast("Mapping[str, Any]", namespace.get("__annotations__", {}))
    return cast("Mapping[str, dataclasses.Field[Any]]", fields), list(annotations)


def _dataclass_field_value(cls: type, name: str) -> Any:
    """Return one dataclass field's default or factory, or `_UNREAD` when absent."""
    field = _own_dataclass_fields(cls)[0].get(name)
    if field is None:
        return _UNREAD
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory
    return _UNREAD


def _dataclass_field_token(cls: type, name: str) -> str:
    """Return a stable token for a dataclass field's live default (its form) or factory."""
    field = _own_dataclass_fields(cls)[0].get(name)
    if field is None:
        return "absent"
    if field.default_factory is not dataclasses.MISSING:
        return _value_token(field.default_factory)
    if field.default is dataclasses.MISSING:
        return "absent"
    return f"field:{xxhash.xxh64(_form(field.default).text.encode()).hexdigest()}"


def _process_dataclass_defaults(cls: type) -> None:
    """Track each own dataclass field's class-level default (a `ClassVar`'s too) by its form
    (`_form`), and what it holds (`_admit`), or walk its default factory as a stage helper.

    The form holds no memory address and runs no code of the value: an `object()` sentinel is
    its type, a `__slots__` object its slots, a string, `bytes` or a path its content. A default
    the instance overrides is hashed too: a member may build an instance that takes it
    (`type(self)()`). A default holding a value whose state no form reads (a `random.Random`,
    whose state is in C) refuses the stage, naming the field."""
    fields, names = _own_dataclass_fields(cls)
    class_ref = _current_node().key
    stage_name = _current_stage_name.get() or _class_qualname(cls)
    for name in names:
        field = fields.get(name)
        if field is None:
            continue
        step = f"fd:{name}"
        _record_read(cls, step)
        binding_ref = f"{class_ref}.{name}"
        with _walking(step):
            if field.default is not dataclasses.MISSING:
                form = _form(field.default)
                _entries()[f"default:{binding_ref}"] = xxhash.xxh64(form.text.encode()).hexdigest()
                message = _field_opaque_message(f"{_definition_ref(cls)}.{name}")
                _admit(form, binding_ref, message)
            elif field.default_factory is not dataclasses.MISSING:
                _process_captured_value(
                    binding_ref, binding_ref, field.default_factory, cls, stage_name, chain=None
                )


def _class_member_parts(name: str, raw: Any) -> list[tuple[str, Any]]:
    """Return the concrete callable values one class-dict member can execute, decided on the
    member's real type, so no `__class__` it defines is read."""
    raw_type: type[Any] = type(raw)  # pyright: ignore[reportUnknownVariableType] - any class
    if issubclass(raw_type, property):
        return [(name, raw.fget), (f"{name}.fset", raw.fset), (f"{name}.fdel", raw.fdel)]
    if issubclass(raw_type, functools.cached_property):
        return [(name, cast("functools.cached_property[Any]", raw).func)]
    if issubclass(raw_type, (classmethod, staticmethod)):
        return [(name, raw.__func__)]
    if issubclass(raw_type, functools.partialmethod):
        return [
            (
                name,
                functools.partial(
                    cast("Callable[..., Any]", raw.func), *raw.args, **(raw.keywords or {})
                ),
            )
        ]
    if issubclass(raw_type, functools.singledispatchmethod):
        dispatcher = raw.dispatcher
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
    mark, or at the mark (`_stop_at_mark`), and a marked object it reaches is returned. A loop
    keeps the original member so its function node can refuse it by name.
    """
    decision = _mark_decision(value)
    if decision is _MarkDecision.BY_NAME or decision is _MarkDecision.REFUSED:
        return value
    if issubclass(type(value), functools.partial):
        return value
    if _wrapper_loop_message(value) is not None:
        return value
    if type(value) is types.FunctionType:
        if _written_in_user_source(value):
            return value
        unwrapped = unwrap(value, _stop_at_mark)
        if fingerprinted_by_name(unwrapped):
            return unwrapped
        if unwrapped is not value and inspect.isfunction(unwrapped):
            return unwrapped if _written_in_user_source(unwrapped) else _UNREAD
        return _UNREAD
    if issubclass(type(value), type):
        return value if is_user_code(value) else _UNREAD
    # A user class instance, decided on its real type, is kept when it is callable or a descriptor
    # (`__get__` on its class), and nothing of it is asked, so no `__getattr__` of it runs.
    if _is_user_class_instance(value):
        return value if callable(value) or _type_defines(_type_of(value), "__get__") else _UNREAD
    if _is_callable_instance(value) or _carries_instance(value):
        return value
    if callable(value):
        unwrapped = unwrap(value, _stop_at_mark)
        if fingerprinted_by_name(unwrapped):
            return unwrapped
        if unwrapped is not value and inspect.isfunction(unwrapped):
            return unwrapped if _written_in_user_source(unwrapped) else _UNREAD
    return _UNREAD


def _type_defines(cls: type[Any], name: str) -> bool:
    """Whether a class on `cls`'s real MRO holds `name` in its own namespace, as Python looks up a
    special method (`__get__`): read through `type`'s own descriptors, so nothing of an instance
    or of a metaclass runs."""
    return any(name in _class_namespace(klass) for klass in _class_mro(cls))


def _user_methods(cls: type) -> list[tuple[str, Any]]:
    """Trackable or refused callable members of user class `cls`'s own namespace, labelled for
    cache replay. A member `cls` inherits is its base's, walked under the base's node
    (`_expand_class`). The namespace is read through `type`'s own descriptor, so no code of the
    class's metaclass runs."""

    def collect() -> list[tuple[str, Any]]:
        found = list[tuple[str, Any]]()
        if not is_user_code(cls):
            return found
        for name, raw in _class_namespace(cls).items():
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


def manifest_entries(manifest: Mapping[str, str], outputs: Collection[str]) -> Iterator[str]:
    """Each entry of a stage manifest as the walk that made it keyed it: a stage's own key as it
    is, and a loader's without its `dep:`/`out:` prefix (`_loader_entry`, which the stage's
    output keys `outputs` place). An entry's kind is its first segment, so no dependency, output,
    field or stage name is read as a kind."""
    for key in manifest:
        entry = _loader_entry(key, outputs) if key.startswith(("dep:", "out:")) else key
        if entry is not None:
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


def _process_body_imports(func: Callable[..., Any], tree: ast.Module) -> None:
    """Walk body imports of user modules discovery loaded; refuse unloaded user modules.

    Pivot never imports a module to fingerprint it: importing a module discovery did not load
    can run any code, so reading its source cannot decide its fingerprint. The refusal is kept
    with the stage's others (`_record_refusal`), and unsafe fingerprinting does not lift it.
    """
    imports = _body_imports(func, tree)
    if not imports:
        return
    stage_name = _current_stage_name.get() or getattr(func, "__name__", "<unknown>")
    owner = _current_node().key
    entries = _entries()
    for body_import in imports:
        module = _body_import_module(body_import.module)
        if module is None:
            if _is_user_module_name(body_import.module):
                error = _unloaded_body_import_error(func, body_import.module)
                _record_refusal(str(error), downgradable=False)
                continue
            _mark_by_name(body_import.module, entries)
            continue
        _mark_module_covered(body_import.module, entries)
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
                        stage_name,
                        chain=body_import.bound,
                    )
            continue
        target = _body_import_module(body_import.target)
        if target is None:
            if _is_user_module_name(body_import.target):
                error = _unloaded_body_import_error(func, body_import.target)
                _record_refusal(str(error), downgradable=False)
                continue
            _mark_by_name(body_import.target, entries)
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


def _process_class_body_dependencies(cls: type) -> None:
    """Add an edge to each user class a class statement's annotations name (`n:NAME`, or
    `np:DOTTED` for `pkg.Type`), read in the namespace of the class's module. Its bases are its
    `b:` edges (`_expand_class`), the classes `__bases__` holds, not the names its statement
    wrote."""
    try:
        source = loaded_source.getsource(cls)
        tree = ast.parse(textwrap.dedent(source))
    except (OSError, TypeError, SyntaxError):
        return
    class_def = next((node for node in tree.body if isinstance(node, ast.ClassDef)), None)
    if class_def is None:
        return

    module_name = _class_module(cls)
    module = sys.modules.get(module_name) if type(module_name) is str else None
    if module is None:
        return
    ns = vars(module)

    names = set[str]()
    dotted_refs = list[tuple[str, ...]]()

    for stmt in class_def.body:
        if isinstance(stmt, ast.AnnAssign):
            _collect_annotation_names(stmt.annotation, names, dotted_refs)

    for name in sorted(names):
        _record_read(cls, f"n:{name}")
        if name not in ns:
            continue
        value = ns[name]
        if issubclass(type(value), type) and is_user_code(value):
            with _walking(f"n:{name}"):
                _add_callable(value)

    for parts in dotted_refs:
        step = f"np:{'.'.join(parts)}"
        _record_read(cls, step)
        resolved = _resolve_dotted_path(parts, ns)
        if not issubclass(type(resolved), type) or not is_user_code(resolved):
            continue
        with _walking(step):
            _add_callable(resolved)


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
    property it reads, transitively), and the code its config fields hold (a function, a class,
    the class of an instance, another pivot loader, at any depth of the field's form: `_form`),
    are walked like stage helpers, so the module values they read are tracked or refused. A
    member no such code reads is not walked. Methods inherited from `pivot.loaders` are
    framework code and are only hashed. A config field that holds a value whose state no form
    reads refuses the stage.

    The loader is the root of a projection (`_project`) of the walk the loaders of a stage share
    (`_loaders_walk`), or of a walk of its own outside a stage's scope. A loader its own walk
    reaches again (its config holds it) is a node that exists, and code two loaders of a stage
    reach is walked once.
    """
    walk = _loaders_walk.get() or _Walk()
    root = walk.node(loader, _NodeKind.LOADER, _definition_ref(type(cast("object", loader))))
    walk.run()
    manifest = dict[str, str]()
    if _capture_errors.get() is not None:  # a stage's scope collects its refusals already
        _project(root, manifest)
        return manifest
    with _collected_capture_errors():
        _project(root, manifest)
    return manifest


def _expand_loader(loader: Any, key: str) -> None:
    """The edges of a loader node (`_Walk`), a loader a stage reads or writes through: its class
    (`cls`, a loader-class node: `_expand_loader_class`), and each config field (`cfg:NAME`), a
    value recorded by its form in `loader:<key>:config` (the hash of `loader_config_text`), whose
    code is walked and whose unreadable state refuses the stage (`_admit`)."""
    if _attribute_hook(_type_of(loader)) is not None:
        _record_refusal(_hooked_class_message(_type_of(loader)), downgradable=True)
        return
    with _walking("cls"):
        _add_loader_class(_type_of(loader), key)
    texts = list[str]()
    for name, form in _loader_config_forms(loader):
        texts.append(f"{name}={form.text}")
        if not form.reports:
            continue  # nothing to act on (`_admit`): a scalar field, as most loaders' are
        binding = f"{key}.{name}"
        with _walking(f"cfg:{name}"):
            _admit(form, binding, _bound_opaque_message(binding))
    if texts:
        _entries()[f"loader:{key}:config"] = xxhash.xxh64(",".join(texts).encode()).hexdigest()


def _expand_loader_class(loader_type: type[Any], key: str) -> None:
    """The edges of a loader-class node (`_Walk`): the walks of its user-defined `load`, `save`
    and `empty` (`lm:NAME`) and of the members they reach (`lr:REF`,
    `_reachable_loader_members`), and the hashes of its `load`, `save` and overridden `empty`
    (`loader:<key>:load`, `:save`, `:empty`). A member the class itself defines is keyed under
    the node's key."""
    from pivot import loaders as loaders_module

    stage_name = _current_stage_name.get() or _class_qualname(loader_type)
    own = f"{_definition_ref(loader_type)}."
    for method_name in _LOADER_METHODS:
        method = _user_defined_method(loader_type, method_name)
        if method is not None:
            step = f"lm:{method_name}"
            _record_read(loader_type, step)
            with _walking(step):
                _add_callable(method, f"{key}.{method_name}")
    for ref, member in _reachable_loader_members(loader_type):
        step = f"lr:{ref}"
        binding = f"{key}.{ref.removeprefix(own)}" if ref.startswith(own) else ref
        _record_read(loader_type, step)
        with _walking(step):
            _process_captured_value(binding, binding, member, loader_type, stage_name, chain=None)

    # Use each method definition rather than a temporary bound method so the
    # weak AST hash cache retains the entry across loader fingerprint calls.
    methods = cast("Any", loader_type)
    entries = _entries()
    if _on_mro(loader_type, loaders_module.Writer):
        entries[f"loader:{key}:save"] = hash_function_ast(_method_function(methods.save))

    # Fingerprint load() if this is a Reader
    if _on_mro(loader_type, loaders_module.Reader):
        entries[f"loader:{key}:load"] = hash_function_ast(_method_function(methods.load))

        # Only fingerprint empty() if it's overridden in a Loader subclass
        if _on_mro(loader_type, loaders_module.Loader):
            for cls in _class_mro(loader_type):
                if cls is loaders_module.Loader:
                    # Reached Loader base class without finding override
                    break
                if "empty" in _class_namespace(cls):
                    # Found override in a subclass
                    entries[f"loader:{key}:empty"] = hash_function_ast(
                        _method_function(methods.empty)
                    )
                    break


def _loader_config_forms(loader: Any) -> list[tuple[str, "_Form"]]:
    """Each dataclass field of `loader`, a loader's config, with the form of its value."""
    return [
        (field.name, _form(getattr(loader, field.name))) for field in dataclasses.fields(loader)
    ]


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


def _is_pivot_loader_class(cls: type[Any]) -> bool:
    """Whether `cls` is a pivot loader class (`loaders.Reader` or `loaders.Writer`), decided by
    identity along its real MRO (`_class_mro`), so no code of its metaclass runs. While
    `pivot.loaders` is not imported no class can be one, so it is not imported here: that import
    brings in pandas."""
    loaders_module = sys.modules.get("pivot.loaders")
    if loaders_module is None:
        return False
    reader, writer = loaders_module.Reader, loaders_module.Writer
    return any(klass is reader or klass is writer for klass in _class_mro(cls))


def _is_user_class(cls: type[Any]) -> bool:
    """Whether `cls` is user code, decided from the module that defines it (its `__module__`, read
    through `type`'s own descriptor), so no code of its metaclass runs (`is_user_code`)."""
    name = _class_module(cls)
    module = sys.modules.get(name) if type(name) is str and name != "builtins" else None
    return module is not None and is_user_code(module)


def loader_config_text(loader: "loaders.Writer[Any] | loaders.Reader[Any]") -> str:
    """Return a loader's dataclass field values as `name=<form text>` (`_form`), comma-joined.

    This is the text hashed as `loader:<module>.<qualname>:config`. It does not depend on
    PYTHONHASHSEED or object identity: a function in a field is its definition and code hash,
    so a loader re-created by importing its pipeline file again gives the same text.
    """
    # Cast to Any for dataclass introspection - loader is a dataclass but generic params are unknown
    forms = _loader_config_forms(cast("Any", loader))
    return ",".join(f"{name}={form.text}" for name, form in forms)


def _user_defined_method(cls: type, name: str) -> Callable[..., Any] | None:
    """Return the function behind `cls`'s nearest `name` attribute when user code defines it,
    found along its real MRO (`_class_mro`, `_class_namespace`)."""
    for klass in _class_mro(cls):
        namespace = _class_namespace(klass)
        if name in namespace:
            member = namespace[name]
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

    A `__wrapped__` chain that loops is refused by its function node before code expansion.
    """

    def collect() -> list[tuple[str, Any]]:
        mro = _class_mro(loader_type)
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
                start = _mro_index(mro, defined_in) + 1 if via_super else 0
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
        elif not callable(member) and not _type_defines(_type_of(member), "__get__"):
            values.append((label, member))
    return values


def _member_functions(function: Any, owner: type, raw: Any) -> list[tuple[Any, type, bool]]:
    """The functions whose bodies the reachable walk reads for a member function, each with
    the class that holds the member and whether it takes the instance first: each function a
    `functools.wraps` chain wraps (`__wrapped__`, read through `wrapper_chain`, whose loop guard
    ends it before an object it has reached) that user source defines, and
    each wrapper on the chain but one that only forwards `*args` to what it wraps (it names no
    member, and receives the instance unnamed, which `_member_reads` would read as a dynamic
    use)."""
    takes_self = not isinstance(raw, staticmethod)
    functions = list[tuple[Any, type, bool]]()
    for current in wrapper_chain(function):
        if not inspect.isfunction(current):
            break
        forwards = hasattr(current, "__wrapped__") and current.__code__.co_argcount == 0
        if not forwards and (current is function or _written_in_user_source(current)):
            functions.append((current, owner, takes_self))
    return functions


def _mro_index(mro: tuple[type, ...], cls: type) -> int:
    """The position of `cls` on `mro`, found by identity: `tuple.index` would ask `==`, which a
    metaclass can define."""
    return next(index for index, klass in enumerate(mro) if klass is cls)


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


def _process_instance_class_attributes(cls: type) -> None:
    """Track each user class attribute a member of `cls`'s own namespace reads through its
    receiver (`ca:NAME`), as `_process_class_reads` tracks `Class.NAME`: a member's `self.NAME`,
    `cls.NAME`, `type(self).NAME`, `self.__class__.NAME` and their literal `getattr` forms
    (`_member_reads`). The class's abstract syntax tree alone cannot show the runtime value. A
    dataclass field read so is an instance's value, not the class's (`_process_dataclass_defaults`
    tracks its default). Each value is read from the class body that holds it
    (`_class_attribute`), so no descriptor's `__get__` and no metaclass code runs. A pydantic
    model's `model_config` is configuration its class body wrote: it is recorded by its form
    (`mc`), not refused as a dict."""
    stage_name = _current_stage_name.get() or _class_qualname(cls)
    class_ref = _current_node().key
    fields = _dataclass_field_names(cls)
    namespace = _class_namespace(cls)
    seen = set[str]()
    for label, member in _user_methods(cls):
        function = member.__func__ if type(member) is types.MethodType else member
        # A member recorded by name (`@fingerprint_by_name`) is not read: its body is the mark's.
        if type(function) is not types.FunctionType or _recorded_by_name(function):
            continue
        tree = _source_tree(function)
        if tree is None:
            continue
        raw = namespace.get(label)
        reads, dynamic = _member_reads(tree, takes_self=not issubclass(type(raw), staticmethod))
        if dynamic:
            continue
        for _via_super, name in reads:
            if (
                name in seen
                or name.startswith("__")
                or name in fields
                or not _is_user_class_attribute(cls, name)
            ):
                continue
            seen.add(name)
            binding_ref = f"{class_ref}.{name}"
            if name == "model_config" and _is_pydantic_model(cls):
                _process_model_config(cls, binding_ref)
                continue
            step = f"ca:{name}"
            value = _read_step(cls, step)
            if value is _UNREAD:
                continue
            _record_read(cls, step)
            with _walking(step):
                if _is_user_enum_member(value):
                    _entries()[f"const:{binding_ref}"] = _enum_member_text(value)
                else:
                    _process_captured_value(
                        binding_ref, binding_ref, value, function, stage_name, chain=None
                    )


def _is_pydantic_model(cls: type[Any]) -> bool:
    """Whether pydantic's `BaseModel` is on `cls`'s real MRO; none is while pydantic is not
    imported."""
    pydantic_main = sys.modules.get("pydantic.main")
    model = getattr(pydantic_main, "BaseModel", None) if pydantic_main is not None else None
    return model is not None and _on_mro(cls, model)


def _process_model_config(cls: type[Any], binding_ref: str) -> None:
    """Record the `model_config` of pydantic model `cls`, which a member reads through its
    receiver, by its form (`default:<class>.model_config`, step `mc`): pydantic writes the
    class's whole configuration there, and the schema hash leaves parts of it out (`title`). The
    code it holds (an alias generator, a `json_schema_extra` function) is walked, and a value no
    form reads refuses the stage (`_admit`)."""
    value = _class_attribute(cls, "model_config")
    if value is _UNREAD:
        return
    _record_read(cls, "mc")
    with _walking("mc"):
        form = _form(value)
        _entries()[f"default:{binding_ref}"] = xxhash.xxh64(form.text.encode()).hexdigest()
        _admit(form, binding_ref, _bound_opaque_message(binding_ref))


def _class_attribute(cls: type[Any], name: str) -> Any:
    """Class attribute `name` of `cls` as a class body wrote it: the entry of the nearest class on
    `cls`'s real MRO that holds it (`_class_namespace`), or `_UNREAD`. Neither a metaclass
    `__getattribute__` nor a descriptor's `__get__` runs; for what the walk tracks as a class
    attribute (`_is_user_class_attribute`) that is what `getattr(cls, name)` gives, but for a
    user descriptor, which stays itself: the walk refuses it as an instance."""
    for klass in _class_mro(cls):
        namespace = _class_namespace(klass)
        if name in namespace:
            return namespace[name]
    return _UNREAD


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
    """Refuse a value no fingerprint records reliably, with the stage's other refusals
    (`_record_refusal`); unsafe fingerprinting makes it a warning. The projection that reports it
    says how the stage reaches the value (`_with_reach`)."""
    subject = f"captures non-scalar variable '{var_name}'"
    if parameter is not None:
        subject += f" as the default of parameter '{parameter}'"
    message = (
        f"Stage '{stage_name}': {subject} (type: {type(value).__name__}).\n"
        + "Pivot tracks module-level bool, int, float, str, bytes, and None values, tuples and "
        + "frozensets of them, and compiled regexes. Other values may change without a reliable "
        + "fingerprint, which can cause silent wrong outputs.\n"
        + "Fix: move this value into the stage's StageParams, or read it from a pipeline input.\n"
        + _SUPPRESSION
    )
    _record_refusal(message, downgradable=True)


# The line every refusal unsafe fingerprinting downgrades ends with.
_SUPPRESSION = "To suppress: set core.unsafe_fingerprinting=true or PIVOT_UNSAFE_FINGERPRINTING=1"


def _field_opaque_message(binding: str) -> Callable[[tuple[str, ...], Any], str]:
    """The refusal of a value whose state no form reads (`_admit`) inside the class-level default
    of field `binding` (a dataclass field, a `ClassVar`, a pydantic model's field)."""

    def message(_path: tuple[str, ...], value: Any) -> str:
        stage_name = _current_stage_name.get() or "<unknown>"
        return (
            f"Stage '{stage_name}': the default of field '{binding}' keeps state pivot cannot "
            + f"read (type: {_definition_ref(type(value))}).\n"
            + "Pivot hashes a class-level default by its value, its attributes and its slots; "
            + "this one keeps its state elsewhere (in C), so a change to it would go unseen, which "
            + "can cause silent wrong outputs.\n"
            + "Fix: for a field, build the value with dataclasses.field(default_factory=...); for "
            + "a ClassVar, which takes no factory, create the value where a method uses it; or "
            + "move the value into the stage's StageParams.\n"
            + _SUPPRESSION
        )

    return message


def _bound_opaque_message(binding: str) -> Callable[[tuple[str, ...], Any], str]:
    """The refusal of a value whose state no form reads (`_admit`) inside value `binding`: a
    partial's arguments, a loader's config field, a `StageParams` default."""

    def message(path: tuple[str, ...], value: Any) -> str:
        stage_name = _current_stage_name.get() or "<unknown>"
        return (
            f"Stage '{stage_name}': bound value '{_binding_text(binding, path)}' keeps state Pivot "
            + f"cannot read (type: {_definition_ref(type(value))}).\n"
            + "Pivot fingerprints a value through its attributes, its slots and the value types "
            + "it knows; this value keeps its state outside them (in C), so a change to it could "
            + "leave an old output cached.\n"
            + "Fix: move the value into the stage's StageParams or read it from a pipeline "
            + "input.\n"
            + _SUPPRESSION
        )

    return message


def _sourceless_opaque_message(ref: str, name: str) -> Callable[[tuple[str, ...], Any], str]:
    """The refusal of a value whose state no form reads (`_admit`) in entry `name` of the
    namespace of class `ref`, which has no source of its own."""

    def message(_path: tuple[str, ...], value: Any) -> str:
        stage_name = _current_stage_name.get() or "<unknown>"
        return (
            f"Stage '{stage_name}': class '{ref}', which has no source of its own, holds a value "
            + f"whose state pivot cannot read in '{name}' (type: {_definition_ref(type(value))}).\n"
            + "Pivot hashes such a class by what defines it; this value keeps its state elsewhere "
            + "(in C), so a change to it would go unseen, which can cause silent wrong outputs.\n"
            + "Fix: define the class with a class statement and build the value with a "
            + "default_factory, or move the value into the stage's StageParams.\n"
            + _SUPPRESSION
        )

    return message


def _attribute_or_none(value: Any, name: str) -> Any:
    """`getattr(value, name, None)`, where an error other than AttributeError from the lookup also
    means no attribute: Python calls an object's `__getattr__` only for a name its class and
    `__dict__` lack, and one that raises KeyError or RuntimeError for that name (a dict-backed
    config, an `AttrDict`, a lazy object) says the attribute is absent."""
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
# `type`'s own descriptors for a class's `__module__`, `__qualname__`, `__name__` and
# `__bases__`, read so too.
_class_module = cast("Callable[[type], object]", type.__dict__["__module__"].__get__)
_class_qualname = cast("Callable[[type], str]", type.__dict__["__qualname__"].__get__)
_class_name = cast("Callable[[type], str]", type.__dict__["__name__"].__get__)
_class_bases = cast("Callable[[type], tuple[type, ...]]", type.__dict__["__bases__"].__get__)
# `BaseException`'s own descriptor for an exception's `args`, read so no override of it runs.
_exception_args = cast(
    "Callable[[BaseException], tuple[Any, ...]]", BaseException.__dict__["args"].__get__
)
# `type`'s own descriptors for the sizes `_read_layout` compares, read so too (`__dictoffset__` is
# `_class_dictoffset`, below).
_class_basicsize = cast("Callable[[type], int]", type.__dict__["__basicsize__"].__get__)
_class_weakrefoffset = cast("Callable[[type], int]", type.__dict__["__weakrefoffset__"].__get__)
# The size of a pointer: a tuple's item is one.
_POINTER_SIZE = tuple.__itemsize__
# The C descriptors a class keeps for its instances' `__dict__` (a class's getset, a module's
# member); reading the instance dict through one runs no code of the object or its class. An
# entry's type is compared with them by identity: `==` would ask a metaclass `__eq__`.
_GETSET_DESCRIPTOR = types.GetSetDescriptorType
_MEMBER_DESCRIPTOR = types.MemberDescriptorType
# `type`'s own C member for a class's instance-dict offset: 0 when its instances have no
# `__dict__` (a `__slots__` class without one, a namedtuple), -1 for a managed dict. Read
# through it, no metaclass code runs.
_class_dictoffset = cast("Callable[[Any], int]", type.__dict__["__dictoffset__"].__get__)


def _keeps_own_attribute(obj: Any, name: str) -> bool:
    """Whether `obj` keeps `name` of its own: in the namespace of a class on its real MRO (a
    slot, a property, a method, a class attribute), or in its instance `__dict__`, read through
    the C descriptor its class keeps for it (`_instance_dict`), or, when a class shadows that
    descriptor (a `__dict__` property), through `object.__getattribute__`, which no class on
    the MRO answers for `name` here and so reads only the instance dict, in C. Nothing of `obj`,
    its class or its metaclass runs, so an object whose `__getattr__` answers every name (a
    `MagicMock`, an attribute tree) does not read as keeping it; a key of the dict whose `__eq__`
    raises raises, as main's `hasattr` does."""
    cls: type[Any] = type(obj)  # pyright: ignore[reportUnknownVariableType] - any class
    if any(name in _class_namespace(klass) for klass in _class_mro(cls)):
        return True
    attributes = _instance_dict(cls, obj)
    if attributes is not None:
        return name in attributes
    if _class_dictoffset(cls) == 0:
        return False
    try:
        object.__getattribute__(obj, name)
    except AttributeError:
        return False
    return True


def _is_user_class_instance(value: Any) -> bool:
    """Return whether `value` is an instance of a user-defined class, decided on its real type,
    so no `__class__` it defines is read."""
    value_type: type[Any] = type(value)  # pyright: ignore[reportUnknownVariableType] - any class
    return not issubclass(value_type, type) and is_user_code(value_type)


def _is_callable_instance(value: Any) -> bool:
    """Return whether `value` is a callable object rather than a function or class.

    A non-user module's own public callable (`numpy.log` is a `ufunc` instance) is a library
    function, not module-level state, so it is not a callable instance here.
    """
    if (
        not callable(value)
        or issubclass(type(value), (functools.partial, type))
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


def _process_closure_values(
    values: Mapping[str, Any], func: Callable[..., Any], *, prefix: str
) -> None:
    """Process the globals (`prefix` "g", dunders skipped) or closure cells ("c") `func` reads.

    A global is keyed by its module (`<module>.NAME`); a cell by the key of the node being
    expanded (`_current_node`), since two closures from one factory hold different values under
    one name. Names are walked in sorted order: getclosurevars collects globals in a set.
    """
    stage_name = _current_stage_name.get() or getattr(func, "__name__", "<unknown>")
    owner = _current_node().key
    for name, value in sorted(values.items()):
        if prefix == "g" and name.startswith("__"):
            continue
        binding_ref = _binding_ref(func, name) if prefix == "g" else f"{owner}.{name}"
        step = f"{prefix}:{name}"
        _record_read(func, step)
        with _walking(step):
            _process_captured_value(name, binding_ref, value, func, stage_name, chain=name)


def _process_captured_value(
    name: str,
    binding_ref: str,
    value: Any,
    func: Callable[..., Any],
    stage_name: str,
    *,
    chain: str | None,
    parameter: str | None = None,
) -> None:
    """Fingerprint one value `func` reads by a name, or refuse it as a non-scalar capture: the
    module-level rule (a global, a cell, a default, a module or class attribute, a class member).

    `chain` is the expression `func` reaches the value by (`Config`, `settings.Config`, a
    parameter name): the attributes read on a user class or user module are followed from it.
    None means the value is not reached by a name, and its attributes are not followed.
    `parameter` names the parameter when the value is a default argument.
    """
    entries = _entries()
    match _value_kind(value):
        case _ValueKind.PARTIAL:
            _process_partial_dependency(binding_ref, value)
        case _ValueKind.USER:
            _add_callable(value, binding_ref)
            if isinstance(value, type) and chain is not None:
                _process_class_reads(chain, value, func, stage_name)
        case _ValueKind.MODULE:
            if not is_user_code(value):
                # Its code is library code, not hashed; a worker imports it by name.
                _mark_by_name(value.__name__, entries)
            elif chain is not None:
                _process_module_dependency(chain, value, func)
        case _ValueKind.SCALAR:
            text = _immutable_value_text(value)
            assert text is not None
            entries[f"const:{binding_ref}"] = text
        case _ValueKind.REFUSED:
            _check_non_scalar_capture(name, value, stage_name, parameter=parameter)
        case _ValueKind.CLASS_ABOVE_MARK:
            _refuse_class_above_mark(value)
        case _ValueKind.EXTERNAL:
            # A library function or class is named by where it is defined; its code is not
            # hashed, but rebinding the name to another one must change the manifest.
            entries[f"ext:{binding_ref}"] = _library_name(value)
            _mark_by_name(getattr(value, "__module__", None), entries)
            if (
                type(value) is _LRU_CACHE_TYPE
                and _mark_decision(value) is not _MarkDecision.BY_NAME
            ):
                _add_child(value, _NodeKind.FUNCTION, _callable_ref(value, binding_ref))
        case _ValueKind.LOGGER:
            return
        case _ValueKind.HOOKED:
            _record_refusal(_hooked_class_message(value), downgradable=True)


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


def _process_class_reads(chain: str, cls: type, func: Callable[..., Any], stage_name: str) -> None:
    """Track each attribute `func` reads from user class `cls`, reached as `chain`.

    `chain` is `Config` for a global, `settings.Config` through a module, or a parameter name.
    The class AST hash does not see a class attribute computed at import, so each attribute
    `func` reads that the user wrote in a class body (`_is_user_class_attribute`) gets the rule
    a module-level value gets: a scalar is `const:<class>.<NAME>`, a non-scalar is refused, a
    function or class is walked. A member of a user Enum is recorded by name and value. A
    pydantic or StageParams class is covered by its schema hash, field defaults included, which
    its class node records (`_process_model_schema`). A cache hit re-reads each attribute
    (`ca:NAME`). Any other attribute is left to the class AST hash.
    """
    if hasattr(cls, "model_fields"):
        return
    class_ref = _definition_ref(cls)
    for attribute in _attributes_read(func, chain):
        if attribute.startswith("__") or not _is_user_class_attribute(cls, attribute):
            continue
        step = f"ca:{attribute}"
        _record_read(cls, step)
        value = _class_attribute(cls, attribute)
        if value is _UNREAD:
            continue
        with _walking(step):
            if _is_user_enum_member(value):
                _entries()[f"const:{class_ref}.{attribute}"] = _enum_member_text(value)
            else:
                _process_captured_value(
                    f"{class_ref}.{attribute}",
                    f"{class_ref}.{attribute}",
                    value,
                    func,
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
    `_fields` and `_field_defaults` that NamedTuple and namedtuple put on a tuple subclass. The
    MRO and namespaces are read through `type`'s own descriptors, and a descriptor is told by its
    class (`_type_defines`), so nothing of the value runs.
    """
    owner = next((klass for klass in _class_mro(cls) if attribute in _class_namespace(klass)), None)
    if owner is None or not is_user_code(owner):
        return False
    raw = _class_namespace(owner)[attribute]
    raw_type: type[Any] = type(raw)  # pyright: ignore[reportUnknownVariableType] - any class
    if issubclass(raw_type, (classmethod, staticmethod)) or (
        _type_defines(raw_type, "__get__") and not callable(raw)
    ):
        return False
    sunder = len(attribute) > 2 and attribute[0] == attribute[-1] == "_"
    if _on_mro(owner, enum.Enum) and sunder and "__" not in (attribute[:2], attribute[-2:]):
        return False
    return not (_on_mro(owner, tuple) and attribute in ("_fields", "_field_defaults"))


def _is_user_enum_member(value: Any) -> bool:
    """Whether `value` is a member of an Enum defined in user code, decided on its real type."""
    value_type: type[Any] = type(value)  # pyright: ignore[reportUnknownVariableType] - any class
    return _on_mro(value_type, enum.Enum) and is_user_code(value_type)


def _enum_member_text(member: enum.Enum) -> str:
    """An Enum member as its class, name and value form: constants of the class body."""
    return f"{type(member).__qualname__}.{member.name}={_form(member.value).text}"


def _process_default_values(func: Callable[..., Any]) -> None:
    """Fingerprint default argument values, which no code hash records.

    A default is evaluated once, when its module loads, so `cols=COLUMNS` puts the name
    `COLUMNS` in the AST and never its value. A scalar, a user subclass of a scalar type
    included, is recorded by its repr, as on main; a StageParams instance by the hash of its form
    (`_form`): a stage's own params hash does not cover a StageParams default on a helper. The
    form of either, when it is a user class's instance (`_is_formed_default`), is then admitted
    (`_admit`): its class's code is walked and its unreadable state refuses the stage. An
    ArtifactHandle default, or a list or tuple of them, is skipped: the pipeline wires it as a
    real input, which is hashed as a dependency. Any other value is treated like a captured
    module value, unless its default is a literal that the function's AST hash already
    records, such as `keys=("a", "b")`.
    """
    defaults = _function_defaults(func)
    if not defaults:
        return
    from pivot import compose

    stage_name = _current_stage_name.get() or getattr(func, "__name__", "<unknown>")
    func_ref = _current_node().key
    entries = _entries()
    expressions: dict[str, ast.expr] | None = None
    for parameter, value in defaults:
        binding_ref = f"{func_ref}.{parameter}"
        step = f"d:{parameter}"
        _record_read(func, step)
        scalar = issubclass(type(value), _SCALAR_TYPES)
        if scalar:
            entries[f"default:{binding_ref}"] = repr(value)
        if _is_formed_default(value):
            form = _form(value)
            if not scalar:
                entries[f"default:{binding_ref}"] = xxhash.xxh64(form.text.encode()).hexdigest()
            with _walking(step):
                _admit(form, binding_ref, _bound_opaque_message(binding_ref))
            continue
        if scalar:
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
            entries[f"default:{binding_ref}"] = text
            continue
        source = ast.unparse(expression) if expression is not None else parameter
        with _walking(step):
            _process_captured_value(
                source,
                binding_ref,
                value,
                func,
                stage_name,
                chain=parameter,
                parameter=parameter,
            )


def _is_artifact_handles(value: object, handle_type: type) -> bool:
    """Return whether `value` is a handle, or a non-empty list or tuple of handles, decided on
    each object's real type, so no `__class__` a value defines is read."""
    if issubclass(type(value), handle_type):
        return True
    if issubclass(type(value), (list, tuple)):
        items = cast("list[object] | tuple[object, ...]", value)
        return bool(items) and all(issubclass(type(item), handle_type) for item in items)
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


def _process_partial_dependency(binding_ref: str, partial_obj: Any) -> None:
    """Process a `functools.partial` read by a name: its arguments and keywords are value roots,
    recorded by their forms (`partial:<binding>.args`, `.kwargs`), whose code is walked and whose
    unreadable state refuses the stage (`_admit`); its function (`pf`) takes the rule a global
    takes (`_value_kind`), so a method bound to an instance (`partial(Scaler(3).apply)`,
    `partial(default_rng(SEED).normal)`) or a callable instance is refused like a bare one.
    """
    if _read_step(partial_obj, "wr") is not _UNREAD:
        _add_child(partial_obj, _NodeKind.FUNCTION, f"{binding_ref}.<wrapper>")
    args_form, kwargs_form = _partial_forms(partial_obj)
    entries = _entries()
    entries[f"partial:{binding_ref}.args"] = xxhash.xxh64(args_form.text.encode()).hexdigest()
    entries[f"partial:{binding_ref}.kwargs"] = xxhash.xxh64(kwargs_form.text.encode()).hexdigest()
    message = _bound_opaque_message(binding_ref)
    _admit(args_form, binding_ref, message)
    _admit(kwargs_form, binding_ref, message)

    underlying = partial_obj.func
    function_ref = f"{binding_ref}.func"
    _record_read(partial_obj, "pf")
    with _walking("pf"):
        stage_name = _current_stage_name.get() or "<unknown>"
        _process_captured_value(
            function_ref, function_ref, underlying, partial_obj, stage_name, chain=None
        )


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
                # Explicit globals keep typing from following wrapper metadata itself. A
                # looping chain falls back to individual annotations; the node refuses it.
                if marked is not None:
                    globalns = getattr(func, "__globals__", {})
                elif hasattr(func, "__wrapped__"):
                    globalns = getattr(unwrap(func), "__globals__", {})
                else:
                    globalns = None
                hints = typing.get_type_hints(func, globalns=globalns)
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
    """Add the user classes in one type hint to `found`, recursing into generics. A class is
    decided on its real type first, so no metaclass attribute hook runs (`typing.get_origin`
    would read its `__class__`)."""
    if hint is type(None):
        return
    if issubclass(type(hint), type):
        if is_user_code(hint):
            found.append(cast("type[Any]", hint))
        return
    origin = typing.get_origin(hint)
    if origin is not None:
        # Track user-defined generic classes (e.g., MyGeneric[int] -> hash MyGeneric)
        if issubclass(type(origin), type) and is_user_code(origin):
            found.append(cast("type[Any]", origin))
        for arg in typing.get_args(hint):
            _collect_hint_classes(arg, found)


def _process_type_hint_dependencies(func: Callable[..., Any]) -> None:
    """Add an edge to each user class in `func`'s type hints (`hc:INDEX`); a pydantic model's
    schema and field defaults are its class node's (`_process_model_schema`)."""
    _record_read(func, "hints")
    for index, hint_class in enumerate(_hint_classes(func)):
        with _walking(f"hc:{index}"):
            _add_callable(hint_class)


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


def _process_model_schema(model: type) -> None:
    """The edges of a pydantic model's class node for its schema (`_Walk`): its schema hash
    (`schema:<key>`, `_schema_parts`), each user class its field annotations name (`fc:INDEX`),
    and each field default, a value root (`sd:NAME`) whose code is walked and whose unreadable
    state refuses the stage (`_admit`). A cache hit reads the schema hash again, field defaults
    included (`schema`)."""
    _record_read(model, "schema")
    key = _current_node().key
    for index, field_class in enumerate(_model_field_classes(model)):
        with _walking(f"fc:{index}"):
            _add_callable(field_class)
    schema_hash, defaults = _schema_parts(cast("type[_PydanticModelProtocol]", model))
    _entries()[f"schema:{key}"] = schema_hash
    for name, form in defaults:
        binding = f"{key}.{name}"
        with _walking(f"sd:{name}"):
            _admit(form, binding, _field_opaque_message(f"{_definition_ref(model)}.{name}"))


def _schema_hash(model: type[_PydanticModelProtocol]) -> str:
    """Hash a pydantic model's JSON schema and field defaults (`_schema_parts`)."""
    return _schema_parts(model)[0]


def _schema_parts(
    model: type[_PydanticModelProtocol],
) -> tuple[str, list[tuple[str, "_Form"]]]:
    """The model's schema and live default forms, reused only within this fingerprint pass.

    Validators and default reports can retain model classes, so none outlives the pass. The next
    pass reads rebuilt models and changed defaults again, without a process-lifetime cache.
    """
    return _memoized("schema-parts", model, lambda: _compute_schema_parts(model))


def _compute_schema_parts(
    model: type[_PydanticModelProtocol],
) -> tuple[str, list[tuple[str, "_Form"]]]:
    from pivot import parameters

    schema_raw = model.model_json_schema(schema_generator=parameters.SortedSetJsonSchema)
    schema = _strip_schema_metadata(schema_raw)
    model_fields = cast("Mapping[str, Any]", getattr(model, "model_fields", {}))
    default_hashes = dict[str, str]()
    defaults = list[tuple[str, _Form]]()
    for field_name, field_info in model_fields.items():
        default = getattr(field_info, "default", None)
        default_factory = getattr(field_info, "default_factory", None)
        if type(default).__name__ == "PydanticUndefinedType":
            if default_factory is not None and callable(default_factory):
                default_hashes[field_name] = hash_function_ast(default_factory)
            continue
        form = _form(default)
        defaults.append((field_name, form))
        default_hashes[field_name] = xxhash.xxh64(form.text.encode()).hexdigest()

    payload = {"schema": schema, "defaults": default_hashes}
    return (xxhash.xxh64(json.dumps(payload, sort_keys=True).encode()).hexdigest(), defaults)


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
    """Return the user classes in a field annotation, recursing into generic arguments; a class
    is decided on its real type first (`_collect_hint_classes`)."""
    if issubclass(type(annotation), type):
        return [cast("type[Any]", annotation)] if is_user_code(annotation) else []
    if typing.get_origin(annotation) is not None:
        return [cls for arg in typing.get_args(annotation) for cls in _annotation_classes(arg)]
    return []


# ---------------------------------------------------------------------------------------------
# Value forms (`_form`)
# ---------------------------------------------------------------------------------------------


class _ReportKind(enum.Enum):
    """What `_form` meets inside a value that the walk acts on (`_admit`)."""

    FUNCTION = "function"  # a user function: a function node
    CLASS = "class"  # a user class: a class node
    LOADER = "loader"  # the class of a pivot loader: a loader-class node
    MODULE = "module"  # a user module, by name: the manifest covers it
    EXTERNAL = "external"  # a library module, by name: a worker imports it by name
    OPAQUE = "opaque"  # a value whose state no row reads: the stage is refused
    MARK = "mark"  # a user class instance above a by-name mark: the stage is refused
    # A pivot loader whose class defines `__call__`: its loader-class node walks `load`, `save`
    # and `empty` only, so what a call runs would go unseen; the stage is refused, as main
    # refuses a callable instance
    CALLABLE = "callable"
    # A library instance whose class computes its attributes (`_forwards_attributes`): a celery or
    # lazy proxy, a mock; the stage is refused (`_proxy_message`)
    PROXY = "proxy"
    # A class whose metaclass answers attribute reads with user code (`_attribute_hook`), or an
    # instance of one: the stage is refused (`_hooked_class_message`)
    HOOKED = "hooked"


# One thing a form reports: the steps from the value to it (`_read_step`), its kind, and the
# object (for MODULE and EXTERNAL, the module's name).
type _Report = tuple[tuple[str, ...], _ReportKind, Any]


@dataclasses.dataclass(frozen=True, slots=True)
class _Form:
    """The form of a value (`_form`): its JSON text, and what the walk acts on inside it."""

    text: str
    reports: tuple[_Report, ...]


def _form(value: Any) -> _Form:
    """The form of `value`: JSON text that depends only on its object graph, the same in any
    process, under any hash seed and in any visit order, recording or replaying, and what in it
    the walk acts on (`_admit`): the code it holds (functions, classes, loaders, modules) and the
    values whose state no row reads, each with the steps from `value` to it.

    One closed table of rows decides each object by its real type (`type(value)`, then its real
    MRO), never by `isinstance`, which reads a `__class__` a proxy defines. State is read through
    C descriptors: an instance's `__dict__` and slots, a built-in base type's value, an
    exception's `args` and fields. A function, class or module is a label (`["function",
    identity]`, `["class", key]`, `["module", name]`), not walked. An object of a type no row knows,
    or whose layout holds state no descriptor shows (`random.Random`, a lock, a numpy
    `Generator`), is `["opaque", <type>]` and reported; no `repr` is read but for the exact value
    types whose C repr writes all of their content. No code of a user class runs, and no text is
    rewritten. `_FormBuilder` says how cycles, sets and dicts are written, and how a value nested
    deeper than the recursion limit is formed."""
    if id(_type_of(value)) in _SCALAR_TYPE_IDS:
        return _Form(json.dumps(value), ())  # the scalar row (`_FormBuilder.enter`), unbuilt
    builder = _FormBuilder()
    return builder.finish(builder.build(value))


def _partial_forms(partial_obj: Any) -> tuple[_Form, _Form]:
    """The forms of a partial's arguments and of its keywords, apart (the `partial:<ref>.args`
    and `.kwargs` entries), each reporting at its step from the partial (`pa:INDEX`, `pk:KEY`)."""
    args = _FormBuilder()
    keywords = _FormBuilder()
    return (
        args.finish(args.parts(args.argument_rows(partial_obj.args))),
        keywords.finish(keywords.parts(keywords.keyword_rows(partial_obj.keywords))),
    )


# The ids of the exact types whose instances are their own form (`None`, `bool`, `int`, `float`,
# `str`): a container whose items' types all have one of these ids is written as its items.
_SCALAR_TYPE_IDS = frozenset(map(id, (type(None), bool, int, float, str)))
# The depth a cycle met nowhere points to (`_FormBuilder.low`).
_NO_CYCLE = sys.maxsize
# The step a set member or a dict key takes while its form is built: its step (`s:RANK`,
# `kk:TEXT`) depends on its form and on the others'.
_PENDING_STEP = "?"
# What the row of a value with parts yields for each part (`_FormBuilder.run`): the part's step
# and the part. The row is sent the part's form and returns the value's.
type _Request = tuple[str, Any]
type _Rows = Generator[_Request, Any, Any]
_GENERATOR = types.GeneratorType


@typing.final
class _Frame:
    """One value `_FormBuilder.run` is forming: its row, the value (held, so its id is not reused
    while the memo keeps its form) and its id (None for a row that forms no value of its own:
    `_FormBuilder.parts`), its depth on the path, the `low` of the value that holds it, and where
    its reports start."""

    __slots__ = ("depth", "key", "low", "mark", "row", "value")

    def __init__(
        self, row: "_Rows | None", value: Any, key: int | None, depth: int, low: int, mark: int
    ) -> None:
        self.row = row
        self.value = value
        self.key = key
        self.depth = depth
        self.low = low
        self.mark = mark


@typing.final
class _FormBuilder:
    """One `_form` call.

    `active` maps the id of each object on the path being formed to its depth: one met again is
    `["cycle", N]`, N steps up. `path` holds the steps from the value to the object being formed,
    and `reports` what the form reports, each at its path. While `quiet` is set, inside a
    function's identity (whose cells and defaults the function's own node walks: `_admit`),
    nothing is reported. `memo` keeps, by id (the object held, so its id is not reused), the
    form of each subtree met whose cycles all point below its root, with its reports relative to
    it: no object in it then reaches its root, so none can be on the path where it is met again,
    and its text is the same wherever it is met. A subtree with a cycle to its root or above is
    formed again where it is met: an object in it may then be on the path, and a cycle there.
    `low` is the smallest depth a cycle met in the subtree being formed points to. `labels` and
    `layouts` keep each class's key and layout.

    A value is formed on an explicit stack, not by recursion (`run`): the row of a value with
    parts is a generator that yields each part's step and the part (`_Request`) and is sent the
    part's form, so a value nested deeper than Python's recursion limit is formed as any other.

    A set's members are written as their texts in sorted order, duplicates kept, each reporting
    under `s:<rank>` (`_ranked`); a dict's entries as `[key, value]` pairs sorted by their texts,
    a value reporting under `k:<key text>` and a key under `kk:<key text>`. Insertion order is
    not recorded for a dict (main's rule); `OrderedDict`, lists, tuples and deques keep their
    order.
    """

    __slots__ = (
        "active",
        "labels",
        "layouts",
        "low",
        "memo",
        "path",
        "quiet",
        "reports",
        "routes",
    )

    def __init__(self) -> None:
        self.active = dict[int, int]()
        self.path = list[str]()
        self.reports = list[_Report]()
        self.quiet = 0
        self.low = _NO_CYCLE
        self.memo = dict[int, tuple[Any, Any, int, int, int]]()
        self.labels = dict[int, str]()
        self.layouts = dict[int, _Layout]()
        self.routes = dict[int, Callable[[_FormBuilder, Any, type[Any]], Any]]()

    def finish(self, built: Any) -> _Form:
        return _Form(json.dumps(built), tuple(self.reports))

    def report(self, kind: _ReportKind, obj: Any, *steps: str) -> None:
        if not self.quiet:
            self.reports.append(((*self.path, *steps), kind, obj))

    # -- the stack -------------------------------------------------------------------------------

    def build(self, value: Any) -> Any:
        """The form of `value`: `enter`, then `run` for a value whose row forms parts."""
        entered = self.enter(value)
        if type(entered) is _Frame:
            return self.run(entered)
        return entered

    def parts(self, row: _Rows) -> Any:
        """The form `row` returns, for a row that forms no value of its own (a partial's arguments
        or keywords, `_partial_forms`)."""
        return self.run(_Frame(row, None, None, 0, self.low, len(self.reports)))

    def run(self, frame: _Frame) -> Any:
        """Run `frame`'s row, and the row of each part it asks for, on one stack, until `frame`'s
        row returns its form: a part's step is on `path` while the part is formed."""
        stack = [frame]
        sent: Any = None
        while True:
            top = stack[-1]
            try:
                step, part = cast("_Rows", top.row).send(sent)
            except StopIteration as done:
                built = self.leave(top, done.value)
                stack.pop()
                if not stack:
                    return built
                self.path.pop()
                sent = built
                continue
            self.path.append(step)
            entered = self.enter(part)
            if type(entered) is _Frame:
                stack.append(entered)
                sent = None
            else:
                self.path.pop()
                sent = entered

    def enter(self, value: Any) -> Any:
        """`value`'s form when it takes no row that forms parts: a scalar, a cycle, a mark
        decision, a leaf row, a subtree met before, or a row that returns its form at once; else
        the `_Frame` of its row, with `value` on the path (`active`)."""
        value_type: type[Any] = type(value)  # pyright: ignore[reportUnknownVariableType] - any class
        if id(value_type) in _SCALAR_TYPE_IDS:
            return value
        key = id(value)
        depth = self.active.get(key)
        if depth is not None:
            self.low = min(self.low, depth)
            return ["cycle", len(self.active) - depth]
        # The mark decision answers a callable only (`_mark_decision`), so an exact container or
        # value is never asked.
        if _marks_made and callable(value) and (marked := self.marked(value)) is not None:
            return marked
        leaf = _LEAF_ROWS.get(id(value_type))
        if leaf is not None:
            return leaf(self, value)
        known = None if self.quiet else self.memo.get(key)
        if known is not None:
            _, built, start, end, base = known
            if end > start:
                prefix = tuple(self.path)
                self.reports.extend(
                    (prefix + path[base:], kind, obj) for path, kind, obj in self.reports[start:end]
                )
            return built
        depth = len(self.active)
        self.active[key] = depth
        frame = _Frame(None, value, key, depth, self.low, len(self.reports))
        self.low = _NO_CYCLE
        row = _ROWS.get(id(value_type))
        made = self.instance(value) if row is None else row(self, value)
        if type(made) is _GENERATOR:
            frame.row = made
            return frame
        return self.leave(frame, made)

    def leave(self, frame: _Frame, built: Any) -> Any:
        """Take `frame`'s value off the path, and keep its form (`memo`) when no cycle in it points
        to it or above."""
        if frame.key is None:
            return built
        del self.active[frame.key]
        low = self.low
        self.low = min(frame.low, low)
        if not self.quiet and low > frame.depth:
            # Its reports stay where they are, with where its own steps start in their paths: a
            # set's or dict's `settle` rewrites a step above it only.
            self.memo[frame.key] = (
                frame.value,
                built,
                frame.mark,
                len(self.reports),
                len(self.path),
            )
        return built

    def part_rows(self, head: list[Any], *parts: _Request) -> _Rows:
        """`head`, then the form of each of `parts`, in order."""
        built = list(head)
        for request in parts:
            built.append((yield request))
        return built

    # -- labels, marks and refusals --------------------------------------------------------------

    def marked(self, value: Any) -> Any:
        """The form the mark decision gives `value` (`_mark_decision`), or None without one."""
        decision = _mark_decision(value)
        if decision is _REFUSED:
            self.report(_ReportKind.MARK, value)
            return ["mark", _definition_ref(type(value))]
        if decision is _BY_NAME and not _carries_instance(value):
            return self.external(value)
        return None

    def external(self, value: Any) -> Any:
        """A library callable, by where it is defined (`_library_name`); its module is reported,
        for the worker imports it by name."""
        module = _attribute_or_none(value, "__module__")
        if type(module) is str and isinstance(sys.modules.get(module), types.ModuleType):
            self.report(_ReportKind.EXTERNAL, module)
        return ["external", _library_name(value)]

    def opaque(self, value: Any) -> Any:
        self.report(_ReportKind.OPAQUE, value)
        return ["opaque", _definition_ref(type(value))]

    def hooked(self, cls: type[Any]) -> Any:
        """A class whose metaclass answers attribute reads with user code (`_attribute_hook`):
        reported (`_ReportKind.HOOKED`) and named by its definition name, so none of it is read."""
        self.report(_ReportKind.HOOKED, cls)
        return ["hooked", _definition_ref(cls)]

    def label(self, cls: type[Any]) -> str:
        """`cls`'s key inside a value (`_class_key`)."""
        found = self.labels.get(id(cls))
        if found is None:
            found = self.labels[id(cls)] = _class_key(cls, in_value=True)
        return found

    def layout(self, cls: type[Any]) -> "_Layout":
        found = self.layouts.get(id(cls))
        if found is None:
            found = self.layouts[id(cls)] = _layout(cls)
        return found

    # -- leaf rows: no state of their own to walk ------------------------------------------------

    def repr_row(self, value: Any) -> Any:
        """An exact library value type whose C repr writes all of its content (`bytes`, a
        `Decimal`, a path, a date, a `UUID`): main's `[type name, repr]`."""
        return [type(value).__name__, repr(value)]

    def zoned(self, value: Any) -> Any:
        """A `datetime` or `time`: its repr when its tzinfo is none, a `timezone` or a `ZoneInfo`,
        whose reprs are library code; a user tzinfo's repr would be user code, so opaque."""
        tzinfo = value.tzinfo
        zoneinfo = sys.modules.get("zoneinfo")
        if (
            tzinfo is None
            or type(tzinfo) is datetime.timezone
            or (zoneinfo is not None and type(tzinfo) is zoneinfo.ZoneInfo)
        ):
            return self.repr_row(value)
        return self.opaque(value)

    def slice_row(self, value: slice) -> Any:
        parts = (value.start, value.stop, value.step)
        if _SCALAR_TYPE_IDS.issuperset(map(id, map(type, parts))):
            return self.repr_row(value)
        return self.opaque(value)

    def pattern(self, value: re.Pattern[Any]) -> Any:
        return ["re.Pattern", self.build(value.pattern), value.flags]

    def class_label(self, cls: type[Any]) -> Any:
        """A class: a user class by its key (`label`), reported for its class node, or, when its
        metaclass answers attribute reads with user code, `hooked`; a library class by
        `type.__repr__`, which runs no metaclass code (main's text)."""
        if is_user_code(cls):
            if _attribute_hook(cls) is not None:
                return self.hooked(cls)
            self.report(_ReportKind.CLASS, cls)
            return ["class", self.label(cls)]
        return [_class_name(type(cls)), type.__repr__(cls)]

    def module(self, module: Any) -> Any:
        """A module, by its name: a user module is reported for the manifest's coverage, a library
        one for the worker's import by name."""
        name = _module_namespace(module).get("__name__")
        if type(name) is str:
            kind = _ReportKind.MODULE if is_user_code(module) else _ReportKind.EXTERNAL
            self.report(kind, name)
        return ["module", name]

    def tuplegetter(self, value: Any) -> Any:
        """A namedtuple's field accessor, by its index and doc, its C `__reduce__` arguments."""
        index, doc = value.__reduce__()[1]
        return ["tuplegetter", index, self.build(doc)]

    def type_name(self, value: Any) -> Any:
        """A `TypeVar`, `ParamSpec`, `TypeVarTuple` or `type` statement alias, by its kind and its
        name, read through its type's own C member: a bound, a constraint or an alias's value is
        evaluated lazily, by code, and says nothing a stage runs."""
        value_type = _type_of(value)
        name = _class_namespace(value_type)["__name__"].__get__(value)
        return ["typevar", _class_name(value_type), name]

    def function(self, func: Any) -> Any:
        """A Python function: main's `["function", function_identity]`, reported for its function
        node (a user function) or its module (a library one)."""
        if is_user_code(func):
            self.report(_ReportKind.FUNCTION, func)
        else:
            module = getattr(func, "__module__", None)
            if type(module) is str and isinstance(sys.modules.get(module), types.ModuleType):
                self.report(_ReportKind.EXTERNAL, module)
        return ["function", self.identity(func)]

    def identity(self, func: Any) -> FunctionIdentity:
        """`function_identity(func)`, its values formed in this call (`binding`), so a cell that
        holds what holds `func` is a cycle. A function being named already is named without its
        code. Nothing in it is reported: the function's node walks its cells and defaults."""
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
            if file is not None and (
                pathlib.PurePath(file).is_absolute() or _recorded_by_name(func)
            ):
                identity["file"] = None
            return identity
        key = id(func)
        if key in self.active:
            return identity
        self.active[key] = len(self.active)
        self.quiet += 1
        try:
            identity["code"] = hash_function_ast(func)
            code = getattr(func, "__code__", None)
            closure: tuple[types.CellType, ...] = getattr(func, "__closure__", None) or ()
            if isinstance(code, types.CodeType):
                identity["cells"] = [
                    (name, self.binding(_cell_value(cell)))
                    for name, cell in zip(code.co_freevars, closure, strict=True)
                ]
            identity["defaults"] = [
                (parameter, self.binding(value)) for parameter, value in _function_defaults(func)
            ]
        finally:
            self.quiet -= 1
            del self.active[key]
        return identity

    def binding(self, value: Any) -> Any:
        """The form of a closure cell or default value in a function identity."""
        from pivot import compose

        if value is _EMPTY_CELL:
            return ["empty"]
        if _is_artifact_handles(value, compose.ArtifactHandle):
            return ["handle"]
        return self.build(value)

    # -- composite rows: their state is formed, one step per part --------------------------------
    # A row returns the value's form when it forms no part, else the generator that forms them.

    def sequence(self, value: Any) -> Any:
        items = cast("list[Any] | tuple[Any, ...]", value)
        name = type(value).__name__
        if _SCALAR_TYPE_IDS.issuperset(map(id, map(type, items))):
            return [name, list(items)]
        return self.item_rows(name, items)

    def item_rows(self, name: str, items: Iterable[Any]) -> _Rows:
        """`[name, forms]`, each item formed at `i:INDEX`."""
        built = list[Any]()
        for index, item in enumerate(items):
            built.append((yield (f"i:{index}", item)))
        return [name, built]

    def unordered(self, value: Any) -> Any:
        """An exact set or frozenset (iterating it runs no code: its members are hashed already)."""
        return self.members(type(value).__name__, list(cast("set[Any] | frozenset[Any]", value)))

    def members(self, name: str, members: list[Any]) -> Any:
        """`[name, forms]` for a set's members, in sorted order (`member_rows`)."""
        if _SCALAR_TYPE_IDS.issuperset(map(id, map(type, members))):
            if len(members) > 1:
                members.sort(key=json.dumps)
            return [name, members]
        return self.member_rows(name, members)

    def member_rows(self, name: str, members: list[Any]) -> _Rows:
        """Each member's form, at `_PENDING_STEP` until all are formed, then sorted by text, its
        reports settled at `s:RANK` (`_ranked`)."""
        forms = list[Any]()
        spans = list[tuple[int, int]]()
        for member in members:
            start = len(self.reports)
            forms.append((yield (_PENDING_STEP, member)))
            spans.append((start, len(self.reports)))
        order, ranks = _ranked([json.dumps(form) for form in forms])
        for span, rank in zip(spans, ranks, strict=True):
            self.settle(span, f"s:{rank}")
        return [name, [forms[index] for index in order]]

    def settle(self, span: tuple[int, int], step: str) -> None:
        """Give the reports in `span` their step in place of `_PENDING_STEP`."""
        position = len(self.path)
        for index in range(*span):
            path, kind, obj = self.reports[index]
            self.reports[index] = ((*path[:position], step, *path[position + 1 :]), kind, obj)

    def mapped(self, tag: str, items: list[tuple[Any, Any]], *, ordered: bool) -> Any:
        """`[tag, entries]` for a mapping's `(key, value)` items (`mapped_rows`)."""
        if _SCALAR_TYPE_IDS.issuperset(
            map(id, map(type, (part for pair in items for part in pair)))
        ):
            return [tag, _sorted_entries([[key, item] for key, item in items], ordered)]
        return self.mapped_rows(tag, items, ordered=ordered)

    def mapped_rows(self, tag: str, items: list[tuple[Any, Any]], *, ordered: bool) -> _Rows:
        """Each entry's `[key form, value form]`: its key's reports settled at `kk:<key text>`, its
        value formed at `k:<key text>`; sorted by their texts unless `ordered`."""
        entries = list[Any]()
        for key, item in items:
            start = len(self.reports)
            key_form = yield (_PENDING_STEP, key)
            key_text = json.dumps(key_form)
            self.settle((start, len(self.reports)), f"kk:{key_text}")
            entries.append([key_form, (yield (f"k:{key_text}", item))])
        return [tag, _sorted_entries(entries, ordered)]

    def mapping(self, value: Any) -> Any:
        """A dict or `Counter`: its entries in sorted order (main's rule)."""
        tag = "dict" if type(value) is dict else type(value).__name__
        return self.mapped(tag, list(_dict_items(value)), ordered=False)

    def ordered_mapping(self, value: Any) -> Any:
        """An `OrderedDict` in its logical order, unless reading it would run key code."""
        contents = _ordered_value(value)
        return self.opaque(value) if contents is _UNREAD else self.contents(contents)

    def mapping_proxy(self, value: Any) -> Any:
        """A `MappingProxyType`, by the mapping it shows (`_proxied_mapping`): a dict's entries,
        read through `dict`'s own method, or any other mapping formed by its state (`px`), so no
        method of that mapping runs."""
        shown = _proxied_mapping(value)
        if type(shown) is dict:
            return self.mapped("mappingproxy", list(_dict_items(shown)), ordered=False)
        return self.part_rows(["mappingproxy"], ("px", shown))

    def contents(self, value: "_Contents") -> Any:
        """The members or entries of a subclass of a built-in hash container (`_Contents`), as the
        container writes them."""
        if value.tag == "set" or value.tag == "frozenset":
            return self.members(value.tag, list(value.items))
        return self.mapped(value.tag, value.items, ordered=value.tag == "OrderedDict")

    def default_mapping(self, value: Any) -> Any:
        return self.default_rows(value.default_factory, list(_dict_items(value)))

    def default_rows(self, factory: Any, items: list[tuple[Any, Any]]) -> _Rows:
        factory_form = yield ("df", factory)
        entries = self.mapped("defaultdict", items, ordered=False)
        if type(entries) is _GENERATOR:
            entries = yield from cast("_Rows", entries)
        return ["defaultdict", factory_form, entries[1]]

    def deque_row(self, value: Any) -> Any:
        items = list(value)
        if _SCALAR_TYPE_IDS.issuperset(map(id, map(type, items))):
            return ["deque", value.maxlen, items]
        return self.deque_rows(value.maxlen, items)

    def deque_rows(self, maxlen: int | None, items: list[Any]) -> _Rows:
        _, built = yield from self.item_rows("deque", items)
        return ["deque", maxlen, built]

    def chain_map(self, value: Any) -> Any:
        return self.part_rows(
            ["ChainMap"], *((f"mp:{index}", item) for index, item in enumerate(value.maps))
        )

    def namespace(self, value: Any) -> Any:
        return self.namespace_rows(value, _type_of(value))

    def namespace_rows(self, value: Any, value_type: type[Any]) -> _Rows:
        attributes = yield from self.attribute_rows(value, value_type)
        if attributes is None:
            return self.opaque(value)
        return ["namespace", attributes]

    def method(self, value: Any) -> Any:
        """A bound method: its function and the object it is bound to, each formed, so a method
        bound to an instance holds that instance's state. A library method bound to a class, a
        scalar, a module or a logger, or a library module's export, keeps main's
        `["external", name]`."""
        function = value.__func__
        if not is_user_code(function) and not _carries_instance(value):
            return self.external(value)
        return self.part_rows(["method"], ("mf", function), ("ms", value.__self__))

    def builtin(self, value: Any) -> Any:
        """A built-in function or method: by name (main's `["external", name]`), unless it is
        bound to an object whose state it may read, which is formed (`rng.random`: an opaque
        `random.Random`)."""
        if not _carries_instance(value):
            return self.external(value)
        return self.part_rows(["bmethod", value.__qualname__], ("ms", value.__self__))

    def partial(self, value: Any) -> Any:
        if _read_step(value, "wr") is not _UNREAD:
            self.report(_ReportKind.FUNCTION, value)
        return self.partial_rows(value.func, value.args, value.keywords)

    def partial_rows(self, func: Any, args: tuple[Any, ...], keywords: Mapping[str, Any]) -> _Rows:
        func_form = yield ("pf", func)
        args_form = yield from self.argument_rows(args)
        keywords_form = yield from self.keyword_rows(keywords)
        return ["partial", func_form, args_form, keywords_form]

    def argument_rows(self, args: tuple[Any, ...]) -> _Rows:
        if _SCALAR_TYPE_IDS.issuperset(map(id, map(type, args))):
            return ["tuple", list(args)]
        built = list[Any]()
        for index, item in enumerate(args):
            built.append((yield (f"pa:{index}", item)))
        return ["tuple", built]

    def keyword_rows(self, keywords: Mapping[str, Any]) -> _Rows:
        entries = list[Any]()
        for name, item in keywords.items():
            entries.append([name, (yield (f"pk:{name}", item))])
        if len(entries) > 1:
            entries.sort(key=json.dumps)
        return ["dict", entries]

    def method_wrapper(self, value: Any) -> Any:
        """A classmethod or staticmethod object: its function."""
        return self.part_rows([type(value).__name__], ("cm", value.__func__))

    def property_row(self, value: property) -> Any:
        return self.part_rows(
            ["property"], ("pg", value.fget), ("ps", value.fset), ("pd", value.fdel)
        )

    def cached_property_row(self, value: Any) -> Any:
        return self.part_rows(["cached_property"], ("cp", value.func))

    def cache_wrapper(self, value: Any) -> Any:
        """A `functools.lru_cache` wrapper: the function it wraps and its cache settings, read from
        the closure of the `cache_parameters` function `lru_cache` gave it (`_lru_cache_settings`),
        which is not called; one whose `cache_parameters` was replaced keeps its settings where no
        form reads them, in C, so it is opaque."""
        attributes = _instance_dict(_type_of(value), value) or {}
        settings = _lru_cache_settings(attributes.get("cache_parameters"))
        if settings is None:
            return self.opaque(value)
        return self.cache_rows(attributes.get("__wrapped__"), settings)

    def cache_rows(self, wrapped: Any, settings: list[Any]) -> _Rows:
        wrapped_form = yield ("wr", wrapped)
        return ["wrapped", "functools._lru_cache_wrapper", wrapped_form, settings]

    def type_alias(self, value: Any) -> Any:
        """A known exact stdlib alias (`list[int]`, `collections.abc.Callable[[int], str]`), by its
        native type, unpacked flag, origin (`to`) and arguments (`ta`), read through C members.
        User-defined subclasses do not take this row: native fields omit their Python state.
        A class in the expression is a label, as anywhere."""
        return self.part_rows(
            ["alias", _definition_ref(_type_of(value)), _GENERIC_ALIAS_UNPACKED(value)],
            ("to", _GENERIC_ALIAS_ORIGIN(value)),
            ("ta", _GENERIC_ALIAS_ARGS(value)),
        )

    def union_alias(self, value: Any) -> Any:
        """A `types.UnionType` (`str | None`), by its arguments (`ta`), read through its own C
        member."""
        return self.part_rows(["union"], ("ta", _UNION_ARGS(value)))

    def instance(self, value: Any) -> Any:
        """An object of a type with no exact row: the row its type's real MRO decides
        (`_instance_route`), decided once per type in this call."""
        value_type: type[Any] = type(value)  # pyright: ignore[reportUnknownVariableType] - any class
        route = self.routes.get(id(value_type))
        if route is None:
            route = self.routes[id(value_type)] = _instance_route(value_type)
        return route(self, value, value_type)

    def class_row(self, value: Any, _value_type: type[Any]) -> Any:
        return self.class_label(value)

    def hooked_instance(self, _value: Any, value_type: type[Any]) -> Any:
        """An instance of a class `hooked` names: its class's key would read the class."""
        return self.hooked(value_type)

    def secret(self, value: Any, _value_type: type[Any]) -> Any:
        from pivot import parameters

        return ["secret", parameters.secret_digest(value)]

    def numpy_row(self, value: Any, value_type: type[Any]) -> Any:
        return self.numpy(sys.modules["numpy"], value, value_type)

    def user_instance(self, value: Any, value_type: type[Any]) -> Any:
        return self.state("inst", value, value_type, user=True)

    def library_instance(self, value: Any, value_type: type[Any]) -> Any:
        """A library class's instance. A proxy, whose class computes its attributes
        (`_forwards_attributes`), is reported (`_ReportKind.PROXY`): what it forwards to is built
        by code no form reads. A library routine or a library module's own callable export (a
        ufunc such as `numpy.log`), bound to no object whose state it may read, is named as main
        names it (`_names_a_library_callable`). Any other one is formed by its state."""
        if _forwards_attributes(value_type):
            self.report(_ReportKind.PROXY, value)
            return ["proxy", _definition_ref(value_type)]
        if _names_a_library_callable(value, value_type):
            return self.external(value)
        return self.state("inst", value, value_type, user=False)

    def state(self, tag: str, value: Any, value_type: type[Any], *, user: bool) -> Any:
        """An instance by its state (`field_rows`): a user class by its key, reported for its class
        node at `cls`; a library class by its definition name. Opaque when its layout holds state
        no descriptor shows (`_layout`)."""
        layout = self.layout(value_type)
        if not layout.readable:
            return self.opaque(value)
        return self.state_rows(tag, value, value_type, layout, user=user)

    def state_rows(
        self, tag: str, value: Any, value_type: type[Any], layout: "_Layout", *, user: bool
    ) -> _Rows:
        fields = yield from self.field_rows(value, value_type, layout)
        if fields is None:
            return self.opaque(value)
        if not user:
            return [tag, _definition_ref(value_type), fields]
        self.report(_ReportKind.CLASS, value_type, "cls")
        return [tag, self.label(value_type), fields]

    def field_rows(self, value: Any, value_type: type[Any], layout: "_Layout") -> _Rows:
        """An instance's state: its `__dict__` (`d`, by attribute name), its slots (`s`, by
        `_slot_values` label) and the value of its built-in base type (`bv`), each read through C;
        None when its `__dict__` holds a name that is not a string."""
        state = dict[str, Any]()
        attributes = yield from self.attribute_rows(value, value_type)
        if attributes is None:
            return None
        if attributes:
            state["d"] = attributes
        slots = list[Any]()
        for label, item in _slot_values(value_type, value):
            slots.append([label, (yield (f"sl:{label}", item))])
        if slots:
            state["s"] = slots
        reader = _SOLID_READERS.get(id(layout.solid))
        if reader is not None:
            builtin = reader(value)
            if builtin is _UNREAD:
                return None
            state["bv"] = yield ("bv", builtin)
        return state

    def attribute_rows(self, value: Any, value_type: type[Any]) -> _Rows:
        """The `[name, form]` pairs of `value`'s instance `__dict__`, by name, or None when a name
        is not an exact string. The dict is read through `dict`'s own C methods (`_dict_items`),
        so no method of a dict subclass or of a key runs."""
        attributes = _instance_dict(value_type, value)
        items = list(_dict_items(attributes)) if attributes is not None else []
        if not all(type(name) is str for name, _ in items):
            return None
        items.sort(key=operator.itemgetter(0))
        if _SCALAR_TYPE_IDS.issuperset(map(id, map(type, (item for _, item in items)))):
            return [[name, item] for name, item in items]
        built = list[Any]()
        for name, item in items:
            built.append([name, (yield (f"at:{name}", item))])
        return built

    def loader(self, value: Any, value_type: type[Any]) -> Any:
        """A pivot loader: its class's key, reported for its loader-class node at `cls`, and its
        state, whose code its holder walks. A loader whose class defines `__call__` is reported
        as callable too (`_ReportKind.CALLABLE`)."""
        layout = self.layout(value_type)
        if not layout.readable:
            return self.opaque(value)
        return self.loader_rows(value, value_type, layout)

    def loader_rows(self, value: Any, value_type: type[Any], layout: "_Layout") -> _Rows:
        fields = yield from self.field_rows(value, value_type, layout)
        if fields is None:
            return self.opaque(value)
        self.report(_ReportKind.LOADER, value_type, "cls")
        if _user_defined_method(value_type, "__call__") is not None:
            self.report(_ReportKind.CALLABLE, value)
        return ["loader", self.label(value_type), fields]

    def exception(self, value: Any, value_type: type[Any]) -> Any:
        """An exception: its class, its `args`, the fields of the built-in exception classes on
        its MRO (`_exception_fields`: `OSError.filename`, `StopIteration.value`), its `__dict__`
        and its slots."""
        layout = self.layout(value_type)
        if not layout.readable:
            return self.opaque(value)
        return self.exception_rows(value, value_type, layout)

    def exception_rows(self, value: Any, value_type: type[Any], layout: "_Layout") -> _Rows:
        user = is_user_code(value_type)
        if user:
            self.report(_ReportKind.CLASS, value_type, "cls")
        state = dict[str, Any]()
        state["args"] = yield ("ea", _exception_args(value))
        found = _exception_fields(value_type, value)
        if found:
            fields = list[Any]()
            for name, item in found:
                fields.append([name, (yield (f"ec:{name}", item))])
            state["c"] = fields
        own = yield from self.field_rows(value, value_type, layout)
        if own is None:
            return self.opaque(value)
        state.update(own)
        label = self.label(value_type) if user else _definition_ref(value_type)
        return ["exc", label, state]

    def enum_member(self, value: Any, value_type: type[Any]) -> Any:
        """An Enum member: its class (a user Enum's by its key, reported for its class node; a
        library Enum's by its definition name), its name and its value's form (`ev`). A value is
        not in a class statement's syntax tree (`PICK = settings.VALUE`), so it is formed."""
        attributes = _instance_dict(value_type, value) or {}
        name = attributes.get("_name_")
        if is_user_code(value_type):
            self.report(_ReportKind.CLASS, value_type, "cls")
            label = self.label(value_type)
        else:
            label = _definition_ref(value_type)
        head = ["enum", label, name if type(name) is str else None]
        return self.part_rows(head, ("ev", attributes.get("_value_")))

    def logger(self, value: Any, value_type: type[Any]) -> Any:
        """A `logging.Logger`, by its name: configuration, not data (main's policy)."""
        attributes = _instance_dict(value_type, value) or {}
        return ["logger", attributes.get("name")]

    def module_subclass(self, value: Any, _value_type: type[Any]) -> Any:
        return self.module(value)

    def numpy(self, numpy: Any, value: Any, value_type: type[Any]) -> Any:
        """A numpy array, scalar or dtype by its data, read through numpy's own C descriptors and
        methods, so no `dtype` or `tobytes` a subclass defines runs; a user subclass of a numpy
        scalar adds its class's key, reported for its class node, its attributes and its slots.
        `_UNREAD` for any other object."""
        for klass in _class_mro(value_type):
            if klass is numpy.ndarray:
                if value_type is not numpy.ndarray:
                    return self.opaque(value)
                dtype = value.dtype
                if dtype.hasobject and dtype.fields is not None:
                    return self.opaque(value)
                if dtype.hasobject:
                    head = ["ndarray-obj", list(value.shape)]
                    items = ((f"nf:{index}", item) for index, item in enumerate(value.flat))
                    return self.object_array_rows(head, items)
                data = numpy.ascontiguousarray(value).tobytes()
                return ["ndarray", dtype.descr, list(value.shape), xxhash.xxh64(data).hexdigest()]
            if klass is numpy.generic:
                dtype = _class_namespace(numpy.generic)["dtype"].__get__(value)
                if dtype.hasobject:
                    return self.opaque(value)
                data = numpy.generic.tobytes(value).hex()
                if is_user_code(value_type):
                    return self.numpy_scalar_rows(value, value_type, dtype.descr, data)
                return ["np", dtype.descr, data]
            if klass is numpy.dtype:
                return ["np.dtype", value.descr]
        return _UNREAD

    def object_array_rows(self, head: list[Any], items: Iterator[_Request]) -> _Rows:
        built = list[Any]()
        for request in items:
            built.append((yield request))
        return [*head, built]

    def numpy_scalar_rows(self, value: Any, value_type: type[Any], descr: Any, data: str) -> _Rows:
        layout = self.layout(value_type)
        if layout.dict_descriptor is None and _class_dictoffset(value_type) != 0:
            return self.opaque(value)
        attributes = yield from self.attribute_rows(value, value_type)
        if attributes is None:
            return self.opaque(value)
        slots = list[Any]()
        for label, item in _slot_values(value_type, value):
            slots.append([label, (yield (f"sl:{label}", item))])
        self.report(_ReportKind.CLASS, value_type, "cls")
        return ["np", self.label(value_type), descr, data, attributes, slots]


def _sorted_entries(entries: list[Any], ordered: bool) -> list[Any]:
    """A mapping's `[key, value]` form pairs, sorted by their texts unless the mapping keeps an
    order of its own (main's rule for a dict: its insertion order is not recorded)."""
    if not ordered and len(entries) > 1:
        entries.sort(key=json.dumps)
    return entries


def _proxied_mapping(proxy: Any) -> Any:
    """The mapping a `MappingProxyType` shows, its one referent (`gc.get_referents`, the proxy's C
    traversal), so no method of the proxy or of the mapping runs."""
    return gc.get_referents(proxy)[0]


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class _Contents:
    """The members (`set`, `frozenset`) or `(key, value)` items (`dict`, `OrderedDict`) of an
    instance of a subclass of a built-in hash container, read through the container's own C
    iterator (`_SOLID_READERS`) into a list: building a container of the base type again would
    hash and compare them, running a member's `__hash__` or `__eq__`. `_form` writes it as the
    container (`_FormBuilder.contents`), and a cache hit reaches inside it (`_value_part`)."""

    tag: str
    items: list[Any]


def _lru_cache_settings(function: Any) -> list[Any] | None:
    """The `maxsize` and `typed` of a `functools.lru_cache` wrapper, from the closure of the
    `cache_parameters` function `lru_cache` gave it (one of `_LRU_PARAMETERS_CODES`), which is not
    called; None when the wrapper holds another one."""
    if type(function) is not types.FunctionType:
        return None
    code = function.__code__
    if not any(code is known for known in _LRU_PARAMETERS_CODES):
        return None
    closure = function.__closure__ or ()
    cells = dict(zip(code.co_freevars, closure, strict=True))
    return [_cell_value(cells["maxsize"]), _cell_value(cells["typed"])]


_LRU_CACHE_TYPE: type[Any] = functools._lru_cache_wrapper  # pyright: ignore[reportPrivateUsage] - the C cache wrapper has no public type name

# The code of the `cache_parameters` functions `functools.lru_cache` gives a wrapper: one where it
# wraps a function at once (`lru_cache(f)`), one where its decorator does (`lru_cache()(f)`).
_LRU_PARAMETERS_CODES = tuple(
    cast("types.FunctionType", vars(wrapper)["cache_parameters"]).__code__
    for wrapper in (functools.lru_cache(len), functools.lru_cache()(len))
)
# `types.GenericAlias`'s and `types.UnionType`'s own C members, read so no attribute an alias
# forwards to its origin answers instead.
_GENERIC_ALIAS_ORIGIN = cast(
    "Callable[[Any], Any]", _class_namespace(types.GenericAlias)["__origin__"].__get__
)
_GENERIC_ALIAS_ARGS = cast(
    "Callable[[Any], tuple[Any, ...]]", _class_namespace(types.GenericAlias)["__args__"].__get__
)
_GENERIC_ALIAS_UNPACKED = cast(
    "Callable[[Any], bool]", _class_namespace(types.GenericAlias)["__unpacked__"].__get__
)
_UNION_ARGS = cast(
    "Callable[[Any], tuple[Any, ...]]",
    _class_namespace(cast("type", cast("object", types.UnionType)))["__args__"].__get__,
)
# The type of a namedtuple's field accessor (`collections._tuplegetter`), which has no public name.
_TUPLEGETTER = _type_of(_class_namespace(typing.NamedTuple("_Fields", [("field", int)]))["field"])
_PURE_PATH_TYPES = (
    pathlib.PurePath,
    pathlib.PurePosixPath,
    pathlib.PureWindowsPath,
    pathlib.Path,
    pathlib.PosixPath,
    pathlib.WindowsPath,
)
# The rows `_FormBuilder.enter` takes by a value's exact type, by the type's id (a dict keyed by
# the type would hash it: a metaclass `__hash__`). Leaf rows hold no state of their own to walk.
_LEAF_ROWS: dict[int, Callable[[_FormBuilder, Any], Any]] = {
    **{
        id(value_type): _FormBuilder.repr_row
        for value_type in (
            bytes,
            bytearray,
            complex,
            range,
            decimal.Decimal,
            fractions.Fraction,
            uuid.UUID,
            datetime.date,
            datetime.timedelta,
            datetime.timezone,
            type(...),
            type(NotImplemented),
            *_PURE_PATH_TYPES,
        )
    },
    **{
        id(value_type): _FormBuilder.type_name
        for value_type in (
            typing.TypeVar,
            typing.ParamSpec,
            typing.TypeVarTuple,
            typing.TypeAliasType,
        )
    },
    id(datetime.datetime): _FormBuilder.zoned,
    id(datetime.time): _FormBuilder.zoned,
    id(slice): _FormBuilder.slice_row,
    id(re.Pattern): _FormBuilder.pattern,
    id(type): _FormBuilder.class_label,
    id(types.ModuleType): _FormBuilder.module,
    id(types.FunctionType): _FormBuilder.function,
    id(_TUPLEGETTER): _FormBuilder.tuplegetter,
}
_ROWS: dict[int, Callable[[_FormBuilder, Any], Any]] = {
    id(list): _FormBuilder.sequence,
    id(tuple): _FormBuilder.sequence,
    id(set): _FormBuilder.unordered,
    id(frozenset): _FormBuilder.unordered,
    id(dict): _FormBuilder.mapping,
    id(collections.Counter): _FormBuilder.mapping,
    id(types.MappingProxyType): _FormBuilder.mapping_proxy,
    id(collections.OrderedDict): _FormBuilder.ordered_mapping,
    id(collections.defaultdict): _FormBuilder.default_mapping,
    id(collections.deque): _FormBuilder.deque_row,
    id(collections.ChainMap): _FormBuilder.chain_map,
    id(types.SimpleNamespace): _FormBuilder.namespace,
    id(types.MethodType): _FormBuilder.method,
    id(types.BuiltinFunctionType): _FormBuilder.builtin,
    id(functools.partial): _FormBuilder.partial,
    id(functools.partialmethod): _FormBuilder.partial,
    id(classmethod): _FormBuilder.method_wrapper,
    id(staticmethod): _FormBuilder.method_wrapper,
    id(property): _FormBuilder.property_row,
    id(functools.cached_property): _FormBuilder.cached_property_row,
    id(_LRU_CACHE_TYPE): _FormBuilder.cache_wrapper,
    id(types.GenericAlias): _FormBuilder.type_alias,
    id(cast("type[Any]", type(Callable[..., object]))): _FormBuilder.type_alias,
    id(types.UnionType): _FormBuilder.union_alias,
    id(_Contents): _FormBuilder.contents,
}
# The rows `_FormBuilder.instance` takes by a class on a value's real MRO, by the class's id.
_MRO_ROWS: dict[int, Callable[[_FormBuilder, Any, type[Any]], Any]] = {
    id(BaseException): _FormBuilder.exception,
    id(enum.Enum): _FormBuilder.enum_member,
    id(logging.Logger): _FormBuilder.logger,
    id(types.ModuleType): _FormBuilder.module_subclass,
}


def _static_attribute(value: Any, value_type: type[Any], name: str) -> Any:
    """Attribute `name` of `value`, read without running code a Python class defines: its
    instance `__dict__` (`_instance_dict`), then the nearest entry on its type's real MRO when
    that entry is a plain value or a C getset or member descriptor. None otherwise."""
    attributes = _instance_dict(value_type, value)
    if attributes is not None and type(attributes) is dict and name in attributes:
        return attributes[name]
    for klass in _class_mro(value_type):
        namespace = _class_namespace(klass)
        if name in namespace:
            entry = namespace[name]
            if type(entry) is _GETSET_DESCRIPTOR or type(entry) is _MEMBER_DESCRIPTOR:
                try:
                    return entry.__get__(value, value_type)
                except AttributeError:
                    return None
            return entry if type(entry) is str else None
    return None


# The objects a library callable may be bound to without reading state no fingerprint records
# (`_carries_instance`).
_STATELESS_RECEIVERS = (types.ModuleType, type, bool, int, float, str, bytes, logging.Logger)


def _forwards_attributes(cls: type[Any]) -> bool:
    """Whether a class on `cls`'s real MRO, other than a built-in one, a `typing` one or pydantic's
    `BaseModel`, defines `__getattr__` or `__getattribute__`: its instances' attributes are
    computed by its code (a proxy forwards them to a target it builds when it is first used),
    read through `type`'s own descriptors. `BaseModel.__getattr__` answers from the private and
    extra fields the model's slots hold, which its form reads; a `typing` alias (`List[int]`,
    `Optional[str]`, `Literal["a"]`) forwards to its origin, and keeps its origin and arguments in
    its `__dict__`, which its form reads."""
    pydantic_main = sys.modules.get("pydantic.main")
    model = getattr(pydantic_main, "BaseModel", None) if pydantic_main is not None else None
    return any(
        klass is not model
        and _class_module(klass) not in ("builtins", "typing")
        and (
            "__getattr__" in _class_namespace(klass)
            or "__getattribute__" in _class_namespace(klass)
        )
        for klass in _class_mro(cls)
    )


def _attribute_hook(cls: type[Any]) -> type[Any] | None:
    """The user class on the real MRO of `cls`'s metaclass that defines `__getattribute__` or
    `__getattr__`, or None: with one, reading any attribute of `cls` (its source, its fields, its
    members) runs that code, so the walk refuses `cls` before it reads anything of it. Decided
    from `type`'s own descriptors, once per class per `fingerprint_pass()`."""
    meta: type[Any] = type(cls)
    if meta is type:
        return None
    return _memoized("attribute-hook", cls, lambda: _read_attribute_hook(meta))


def _read_attribute_hook(meta: type[Any]) -> type[Any] | None:
    for klass in _class_mro(meta):
        if klass is type or klass is object:
            return None
        namespace = _class_namespace(klass)
        hooks = "__getattribute__" in namespace or "__getattr__" in namespace
        if hooks and _is_user_class(klass):
            return klass
    return None


def _hooked_class_message(cls: type[Any]) -> str:
    """The refusal of class `cls`, whose metaclass answers attribute reads with user code
    (`_attribute_hook`)."""
    stage_name = _current_stage_name.get() or "<unknown>"
    hook = _attribute_hook(cls)
    meta = _definition_ref(hook) if hook is not None else "?"
    return (
        f"Stage '{stage_name}': class '{_definition_ref(cls)}' has the metaclass '{meta}', whose "
        + "`__getattribute__` or `__getattr__` is user code, so reading the class runs it.\n"
        + "Pivot reads a class through `type`'s own descriptors and runs none of its code; this "
        + "class answers attribute reads with code, so what it holds could change unseen, which "
        + "can cause silent wrong outputs.\n"
        + "Fix: remove `__getattribute__` and `__getattr__` from the metaclass, or move the "
        + "values the stage needs into its StageParams.\n"
        + _SUPPRESSION
    )


def _proxy_message(binding: str, value: Any) -> str:
    """The refusal of library instance `value`, a proxy (`_forwards_attributes`), at `binding`."""
    stage_name = _current_stage_name.get() or "<unknown>"
    return (
        f"Stage '{stage_name}': value '{binding}' is a proxy (type: "
        + f"{_definition_ref(type(value))}) whose class computes its attributes (`__getattr__` "
        + "or `__getattribute__`).\n"
        + "Pivot reads a value through its state and runs none of its code, so what the proxy "
        + "forwards to (a value built when it is first used, or read from a file) would go "
        + "unseen, which can cause silent wrong outputs.\n"
        + "Fix: read the value from a pipeline input, or move it into a StageParams field.\n"
        + _SUPPRESSION
    )


def _names_a_library_callable(value: Any, value_type: type[Any]) -> bool:
    """Main's `_is_library_callable(value) and not _carries_instance(value)` for an instance of
    a library class, its attributes read by `_static_attribute`, so an object that answers every
    name (a `MagicMock`) makes nothing new while it is asked: a routine (`inspect.isroutine`,
    decided on its type) or a library module's own export (`_is_static_library_export`), whose
    `__self__`, if any, holds no state the fingerprint misses."""
    if not callable(value):
        return False
    export = _is_static_library_export(value, value_type)
    if not export and not inspect.isroutine(value):
        return False
    bound = _static_attribute(value, value_type, "__self__")
    return export or bound is None or issubclass(type(bound), _STATELESS_RECEIVERS)


def _is_static_library_export(value: Any, value_type: type[Any]) -> bool:
    """Whether a library module binds `value` under its own `__name__` (`_library_module`), its
    name and module read by `_static_attribute`."""
    name = _static_attribute(value, value_type, "__name__")
    if type(name) is not str:
        return False
    module_name = _static_attribute(value, value_type, "__module__")
    candidates: list[object] = (
        [sys.modules.get(module_name)] if type(module_name) is str else list(sys.modules.values())
    )
    return any(
        issubclass(type(module), types.ModuleType)
        and _module_namespace(cast("types.ModuleType", module)).get(name) is value
        and not is_user_code(module)
        for module in candidates
    )


def _instance_route(value_type: type[Any]) -> Callable[[_FormBuilder, Any, type[Any]], Any]:
    """The row `_FormBuilder.instance` takes for an object of `value_type`, which has no exact
    row, decided along its real MRO: a class, a pydantic secret, a pivot loader, the nearest
    `_MRO_ROWS` class, a numpy array, scalar or dtype, then a user or a library instance."""
    if issubclass(value_type, type):
        return _FormBuilder.class_row
    if _attribute_hook(value_type) is not None:
        return _FormBuilder.hooked_instance
    mro = _class_mro(value_type)
    if _is_secret_type(mro):
        return _FormBuilder.secret
    if _is_pivot_loader_class(value_type):
        return _FormBuilder.loader
    for klass in mro:
        row = _MRO_ROWS.get(id(klass))
        if row is not None:
            return row
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        arrays = (numpy.ndarray, numpy.generic, numpy.dtype)
        if any(klass is array for klass in mro for array in arrays):
            return _FormBuilder.numpy_row
    if is_user_code(value_type):
        return _FormBuilder.user_instance
    return _FormBuilder.library_instance


def _is_secret_type(mro: tuple[type, ...]) -> bool:
    """Whether a real MRO holds a pydantic secret type (`SecretStr`, `SecretBytes`,
    `Secret[T]`), by identity; none can while pydantic is not imported."""
    if "pydantic" not in sys.modules:
        return False
    from pivot import parameters

    secrets = parameters._SECRET_TYPES  # pyright: ignore[reportPrivateUsage] - the secret types the lock file digests
    return any(klass is secret for klass in mro for secret in secrets)


def _ranked(texts: list[str]) -> tuple[list[int], list[int]]:
    """The indexes of `texts` in sorted order, and each text's rank: the position in that order
    of the first text equal to it, so equal texts share a rank and no rank follows the hash
    seed (a set's `s:RANK` step)."""
    order = sorted(range(len(texts)), key=texts.__getitem__)
    ranks = [0] * len(texts)
    rank = 0
    for position, index in enumerate(order):
        if position and texts[index] != texts[order[position - 1]]:
            rank = position
        ranks[index] = rank
    return order, ranks


@dataclasses.dataclass(frozen=True, slots=True)
class _Layout:
    """How `_form` reads an instance of one class (`_layout`): whether its layout holds only state
    its descriptors show (`readable`), its nearest built-in base type on its real MRO
    (`solid`; `object` when none), the member descriptors of the slots its classes above that
    declare, each with its label (`<mro index>:<qualname>.<slot>`), and the C descriptor of its
    instances' `__dict__` (None when they have none)."""

    readable: bool
    solid: type
    slots: tuple[tuple[str, Any], ...]
    dict_descriptor: Any


def _ordered_value(value: Any) -> Any:
    """List logical entries without running key code, or `_UNREAD` if that is not possible.

    Even CPython's native OrderedDict key iterator hashes keys. Inspect the backing dict first,
    whose item iterator does not hash, and allow only exact scalar keys with native hashes.
    """
    if any(id(_type_of(key)) not in _ORDERED_KEY_IDS for key, _ in _dict_items(value)):
        return _UNREAD
    return _Contents("OrderedDict", list(_native_ordered_items(value)))


_ORDERED_KEY_IDS = _SCALAR_TYPE_IDS | {id(bytes), id(complex)}


# How `_form` reads the value of an instance of a subclass of a built-in type (`bv`), as an exact
# object of that type, through the type's own C methods, so no method the subclass defines runs.
# A hash container's members are kept in a list (`_Contents`): building the container again would
# hash and compare them.
_SOLID_READERS: dict[int, Callable[[Any], Any]] = {
    id(int): int.__index__,
    id(float): float.__float__,
    id(complex): complex.__complex__,
    id(str): str.__str__,
    id(bytes): bytes.__bytes__,
    id(bytearray): lambda value: bytearray.__getitem__(value, slice(None)),
    id(tuple): lambda value: tuple(_tuple_iter(value)),
    id(list): lambda value: list(_list_iter(value)),
    id(dict): lambda value: _Contents("dict", list(_dict_items(value))),
    id(set): lambda value: _Contents("set", list(_set_iter(value))),
    id(frozenset): lambda value: _Contents("frozenset", list(_frozenset_iter(value))),
    id(collections.OrderedDict): _ordered_value,
    id(collections.deque): lambda value: collections.deque(
        _deque_iter(value), _DEQUE_MAXLEN(value)
    ),
    id(functools.partial): lambda value: functools.partial(
        _PARTIAL_FUNC(value), *_PARTIAL_ARGS(value), **_PARTIAL_KEYWORDS(value)
    ),
}
_DEQUE_MAXLEN = cast("Callable[[Any], int | None]", collections.deque.__dict__["maxlen"].__get__)
_PARTIAL_FUNC = cast("Callable[[Any], Any]", functools.partial.__dict__["func"].__get__)
_PARTIAL_ARGS = cast("Callable[[Any], tuple[Any, ...]]", functools.partial.__dict__["args"].__get__)
_PARTIAL_KEYWORDS = cast(
    "Callable[[Any], dict[str, Any]]", functools.partial.__dict__["keywords"].__get__
)
# The built-in types' own C methods `_SOLID_READERS` and `_mapping_items` read a value through,
# typed for any receiver: an override in a subclass does not run.
_tuple_iter = cast("Callable[[Any], Iterator[Any]]", tuple.__iter__)
_list_iter = cast("Callable[[Any], Iterator[Any]]", list.__iter__)
_set_iter = cast("Callable[[Any], Iterator[Any]]", set.__iter__)
_frozenset_iter = cast("Callable[[Any], Iterator[Any]]", frozenset.__iter__)
_deque_iter = cast("Callable[[Any], Iterator[Any]]", collections.deque.__iter__)
_dict_items = cast("Callable[[Any], Collection[tuple[Any, Any]]]", dict.items)
_native_ordered_items = cast(
    "Callable[[Any], Collection[tuple[Any, Any]]]", collections.OrderedDict.items
)


def _layout(cls: type[Any]) -> _Layout:
    """How `_form` reads an instance of `cls` (`_Layout`), from `type`'s own descriptors, once per
    class per `fingerprint_pass()`.

    The solid base is the first class on the real MRO with a reader (`_SOLID_READERS`), or any
    other built-in class but `object`. An instance shows all of its state when the solid base has
    a reader, is an exception (its fields are `_exception_fields`) or is `object`, and the
    instance's size is the solid base's, plus a pointer for each slot its classes above declare
    and for a `__dict__` or `__weakref__` pointer the solid base has none of (a managed dict adds
    none). A larger one keeps C state no descriptor shows: `random.Random`'s generator, a lock, a
    numpy `Generator`. The heap-type flag cannot decide it: `_random.Random`'s subclass is one."""
    return _memoized("layout", cls, lambda: _read_layout(cls))


def _read_layout(cls: type[Any]) -> _Layout:
    mro = _class_mro(cls)
    solid: type = object
    for klass in mro:
        if id(klass) in _SOLID_READERS or (
            klass is not object and _class_module(klass) == "builtins"
        ):
            solid = klass
            break
    slots = list[tuple[str, Any]]()
    for index, klass in enumerate(mro):
        if klass is solid:
            break
        for name, raw in _class_namespace(klass).items():
            if type(raw) is _MEMBER_DESCRIPTOR and name not in ("__dict__", "__weakref__"):
                slots.append((f"{index}:{_class_qualname(klass)}.{name}", raw))
    size = _class_basicsize(solid) + _POINTER_SIZE * len(slots)
    if _class_dictoffset(cls) > 0 and _class_dictoffset(solid) <= 0:
        size += _POINTER_SIZE
    if _class_weakrefoffset(cls) > 0 and _class_weakrefoffset(solid) <= 0:
        size += _POINTER_SIZE
    known = solid is object or id(solid) in _SOLID_READERS or _on_mro(solid, BaseException)
    descriptor = None
    for klass in mro:
        entry = _class_namespace(klass).get("__dict__")
        if (
            type(entry) is _GETSET_DESCRIPTOR or type(entry) is _MEMBER_DESCRIPTOR
        ) and entry.__objclass__ is klass:
            descriptor = entry
            break
    # An instance dict no C descriptor reads (a class shadows it with a `__dict__` property) is
    # state no form can show without running that class's code.
    shown = descriptor is not None or _class_dictoffset(cls) == 0
    return _Layout(
        known and shown and size == _class_basicsize(cls), solid, tuple(slots), descriptor
    )


def _instance_dict(cls: type[Any], value: Any) -> dict[str, Any] | None:
    """`value`'s instance `__dict__`, without the by-name mark, read through the C descriptor
    the class on `cls`'s real MRO that introduced it keeps for it (`_layout`), so no code of `cls`
    runs; None when it has none. A dict whose keys are not all exact strings is returned as it
    is: rebuilding it would hash and compare its keys, user code, and a form takes such a dict
    as state it cannot read (`_FormBuilder.attributes`)."""
    descriptor = _layout(cls).dict_descriptor
    if descriptor is None:
        return None
    found: object = descriptor.__get__(value, cls)
    if not issubclass(type(found), dict):
        return None
    own = cast("dict[str, Any]", found)
    items = list(_dict_items(own))
    if not all(type(key) is str for key, _ in items):
        return own
    return {key: item for key, item in items if key != _BY_NAME_MARK}


def _slot_values(cls: type[Any], value: Any) -> list[tuple[str, Any]]:
    """The values in the slots of `value`, an instance of `cls`, each with its label
    (`_Layout.slots`); an unset slot is left out. Each is read through its slot's own descriptor,
    so no code of `cls` runs."""
    found = list[tuple[str, Any]]()
    for label, descriptor in _layout(cls).slots:
        try:
            found.append((label, descriptor.__get__(value, cls)))
        except AttributeError:  # an unset slot
            continue
    return found


def _builtin_value(value: Any) -> Any:
    """The value of `value`'s built-in base type as an exact object of it (`_SOLID_READERS`), or
    `_UNREAD` when its type has none."""
    reader = _SOLID_READERS.get(id(_layout(_type_of(value)).solid))
    return _UNREAD if reader is None else reader(value)


def _exception_fields(cls: type[Any], value: BaseException) -> list[tuple[str, Any]]:
    """The fields of exception `value` that the built-in exception classes on its real MRO keep
    in C, outside `args` and `__dict__` (`OSError.errno`, `strerror`, `filename`, `filename2`,
    `StopIteration.value`, `SyntaxError` and `UnicodeError` fields), each read through its own
    member or getset descriptor, the nearest by name; one unset is left out. Where it was raised
    (its traceback, cause and context) is not its value."""
    found = list[tuple[str, Any]]()
    seen = set[str]()
    for klass in _class_mro(cls):
        if klass is object or _class_module(klass) != "builtins":
            continue
        for name, raw in _class_namespace(klass).items():
            if name in seen or name in _EXCEPTION_NONVALUE_ATTRIBUTES:
                continue
            if type(raw) is not _GETSET_DESCRIPTOR and type(raw) is not _MEMBER_DESCRIPTOR:
                continue
            seen.add(name)
            try:
                found.append((name, raw.__get__(value, cls)))
            except AttributeError:  # an unset field (`characters_written`)
                continue
    return found


# Exception attributes that describe where an exception was raised or duplicate state recorded
# elsewhere (`args`, `__dict__`), not its value (`_exception_fields`).
_EXCEPTION_NONVALUE_ATTRIBUTES = frozenset(
    (
        "args",
        "__dict__",
        "__traceback__",
        "__cause__",
        "__context__",
        "__suppress_context__",
        "__notes__",
    )
)


def _admit(form: _Form, binding: str, opaque: Callable[[tuple[str, ...], Any], str]) -> None:
    """Act on what `form` reports: the form of a value root (a value the node being expanded
    records by content, at the current path: a partial's arguments, a default, a field default, a
    loader's config field, an entry of a class with no source), named `binding`.

    Every object in a value root is accepted: a function gets an edge to its function node, keyed
    as a stage helper is (`_callable_ref` of its position in `binding`: `_binding_text`); a class,
    the class of an instance included, an edge to its class node, keyed as inside a value
    (`_class_key`); a pivot loader's class an edge to its loader-class node; a module is covered
    or imported by name. Those nodes then follow the module-level rule for what they read. A value
    whose state no form reads refuses the stage (`opaque` gives the message, downgradable); so
    does a callable pivot loader, as a non-scalar capture; and a user class instance above a
    by-name mark refuses it as main refused it. Each object is acted on once, at the first of its
    paths in sorted order, so many instances of one class add one edge, and the edges and
    messages follow no visit order."""
    first = dict[tuple[_ReportKind, object], _Report]()
    for report in form.reports:
        path, kind, obj = report
        key = (kind, obj if type(obj) is str else id(obj))
        found = first.get(key)
        if found is None or path < found[0]:
            first[key] = report
    for path, kind, obj in sorted(first.values(), key=lambda report: report[0]):
        with _walking(*path):
            match kind:
                case _ReportKind.FUNCTION:
                    _add_callable(obj, _binding_text(binding, path))
                case _ReportKind.CLASS:
                    _add_callable(obj, in_value=True)
                case _ReportKind.LOADER:
                    _add_loader_class(obj, _class_key(obj, in_value=True))
                case _ReportKind.MODULE:
                    _mark_module_covered(obj, _entries())
                case _ReportKind.EXTERNAL:
                    _mark_by_name(obj, _entries())
                case _ReportKind.OPAQUE:
                    _record_refusal(opaque(path, obj), downgradable=True)
                case _ReportKind.MARK:
                    _record_refusal(_class_above_mark_message(obj), downgradable=False)
                case _ReportKind.CALLABLE:
                    stage_name = _current_stage_name.get() or "<unknown>"
                    _check_non_scalar_capture(_binding_text(binding, path), obj, stage_name)
                case _ReportKind.PROXY:
                    message = _proxy_message(_binding_text(binding, path), obj)
                    _record_refusal(message, downgradable=True)
                case _ReportKind.HOOKED:
                    _record_refusal(_hooked_class_message(obj), downgradable=True)


def _binding_text(binding: str, path: tuple[str, ...]) -> str:
    """The name of the object `path` reaches inside value `binding`, as main named a value bound
    in a partial's arguments: `<binding>.args[0]`, `.kwargs[key]`, `[3]` an item, `{0}` a set
    member by rank, `["key"]` a dict value, `.func` a partial's function, `.NAME` an instance
    attribute, and `.<step>` any other step (`_read_step`)."""
    parts = [binding]
    for step in path:
        kind, _, arg = step.partition(":")
        match kind:
            case "pa":
                parts.append(f".args[{arg}]")
            case "pk":
                parts.append(f".kwargs[{arg}]")
            case "i" | "k":
                parts.append(f"[{arg}]")
            case "s":
                parts.append(f"{{{arg}}}")
            case "pf":
                parts.append(".func")
            case "at":
                parts.append(f".{arg}")
            case _:
                parts.append(f".{step}")
    return "".join(parts)


# `ModuleType`'s own C descriptor for a module's namespace, read so no module subclass's code runs.
_module_namespace = cast(
    "Callable[[types.ModuleType], dict[str, Any]]",
    _class_namespace(types.ModuleType)["__dict__"].__get__,
)
# Namespace entries every class has whose text says nothing of what defines it; `_abc_impl` is
# the ABC machinery's cache of registered and checked subclasses.
_CLASS_NAMESPACE_NOISE = frozenset(("__dict__", "__weakref__", "__doc__", "_abc_impl"))


def _class_key(cls: type[Any], *, in_value: bool = False) -> str:
    """The key a class node's entries take, and the label a form gives the class: its definition
    name (`_definition_ref`).

    Reached by a name (a global, a cell, a type hint, an annotation), a class keeps main's key:
    with no source of its own (`_has_no_source`) its hash too (`<name>@<hash>`), since one factory
    makes many classes under one name. Reached inside a value (`in_value`: an instance's class, a
    class in a container, a base), only a class with source that its module's name holds
    (`_resolves_by_name`) keeps its name; any other (a `<locals>` class, a `type()` or namedtuple
    class, a name rebound since) is `<name>@<identity8>`, the start of its identity
    (`_class_identity`), so two classes one factory made keep apart wherever a value holds them."""
    ref = _definition_ref(cls)
    no_source = _has_no_source(cls)
    if not in_value:
        return f"{ref}@{hash_function_ast(cls)}" if no_source else ref
    if not no_source and _resolves_by_name(cls):
        return ref
    return f"{ref}@{_class_identity(cls)[:8]}"


def _resolves_by_name(cls: type[Any]) -> bool:
    """Whether the module `cls` names as its own (`sys.modules`) holds `cls` at its qualified
    name, read through the module's and the classes' namespaces, so no code of either runs."""
    module_name = _class_module(cls)
    module = sys.modules.get(module_name) if type(module_name) is str else None
    if module is None:
        return False
    namespace: Mapping[str, Any] = _module_namespace(module)
    found: Any = None
    for part in _class_qualname(cls).split("."):
        found = namespace.get(part)
        if found is None or not issubclass(type(found), type):
            return False
        namespace = _class_namespace(found)
    return found is cls


# Namespace entries a class's identity leaves out (`_class_identity`): the noise every class has,
# what the compiler writes for its statement, and the signature pydantic writes for a model.
_IDENTITY_NOISE = frozenset(
    (
        *_CLASS_NAMESPACE_NOISE,
        "__module__",
        "__qualname__",
        "__firstlineno__",
        "__static_attributes__",
        "__signature__",
    )
)


def _class_identity(cls: type[Any]) -> str:
    """What makes two class objects the same class, as a hex digest: its definition name, its
    code hash (`hash_function_ast`: its statement's AST, or for a class with no source what
    defines it), the keys of its bases, and the form of each entry of its own namespace (`_form`:
    a function by its identity, cells and defaults included; a value by its state; a value no row
    reads by its type), so it never refuses. Two classes one factory made differ in it wherever
    what each was built with differs.

    A class whose identity is being computed (`_identity_active`), met again, is `^N`, N steps
    up, so classes that reach each other end. Within one `fingerprint_pass()` a class's identity
    is computed once at the top; one computed inside another's is kept (`free`) when no `^N` in
    it points to it or above, as its text is then the same wherever it is computed, so a chain of
    classes each holding the next costs one identity per class. Identities nest no deeper than
    `_IDENTITY_DEPTH` (`_identifying`), however long a chain of classes is."""
    active = _identity_active.get()
    key = id(cls)
    if not active:
        return _memoized(
            "class-identity", cls, lambda: _identifying(lambda: _computed_class_identity(cls))
        )
    scope = cast("_IdentityScope", _identity_scope.get())
    if key in active:
        index = active.index(key)
        scope.lows[-1] = min(scope.lows[-1], index)
        return f"^{len(active) - index}"
    known = scope.free.get(key)
    if known is None and (memo := _pass_memo.get()) is not None:
        known = memo.get(("free-class-identity", key))
    if known is not None:
        return cast("tuple[type[Any], str]", known)[1]
    contextual = scope.contextual.get((key, active))
    if contextual is not None:
        scope.lows[-1] = min(scope.lows[-1], contextual[2])
        return contextual[1]
    if len(active) - scope.base_depth >= _IDENTITY_DEPTH:
        raise _IdentityTooDeepError(cls)
    return _computed_class_identity(cls)


def _computed_class_identity(cls: type[Any]) -> str:
    active = _identity_active.get()
    depth = len(active)
    token = _enter_identity(cls)
    try:
        namespace = _class_namespace(cls)
        pydantic = "__pydantic_fields__" in namespace
        bases = [
            _class_key(base, in_value=True) if is_user_code(base) else _definition_ref(base)
            for base in _class_bases(cls)
        ]
        entries = [
            [name, _form(namespace[name]).text]
            for name in sorted(name for name in namespace if type(name) is str)
            if name not in _IDENTITY_NOISE
            and not (pydantic and name.startswith("__pydantic_") and name != "__pydantic_fields__")
        ]
        parts = [_definition_ref(cls), hash_function_ast(cls), bases, entries]
    finally:
        low = _leave_identity(token)
    identity = xxhash.xxh64(json.dumps(parts).encode()).hexdigest()
    scope = cast("_IdentityScope", _identity_scope.get())
    scope.contextual[(id(cls), active)] = (cls, identity, low)
    if low > depth:
        scope.free[id(cls)] = (cls, identity)
        if (memo := _pass_memo.get()) is not None:
            memo[("free-class-identity", id(cls))] = (cls, identity)
            memo.setdefault(("class-identity", id(cls)), (cls, identity))
    return identity


# How many class identities may nest on the Python stack before the worklist computes a
# dependency first. Its logical ancestor path stays intact, including across a long cycle.
_IDENTITY_DEPTH = 24


class _IdentityTooDeepError(Exception):
    """`cls`'s identity is needed deeper than `_IDENTITY_DEPTH` and is not known yet: the
    outermost identity being computed (`_identifying`) computes it first, then starts again."""

    def __init__(self, cls: type[Any]) -> None:
        super().__init__()
        self.cls: type[Any] = cls
        self.active: tuple[int, ...] = _identity_active.get()


@dataclasses.dataclass(slots=True)
class _IdentityScope:
    """One identity computation's worklist facts: the lowest logical ancestor each active result
    refers to (`lows`), context-free identities (`free`), and results keyed by their full logical
    ancestor path (`contextual`). `base_depth` is the ancestor prefix suspended off the Python
    stack, so cyclic and acyclic dependencies share the same hard stack-depth bound."""

    lows: list[int] = dataclasses.field(default_factory=list[int])
    free: dict[int, tuple[type[Any], str]] = dataclasses.field(
        default_factory=dict[int, tuple[type[Any], str]]
    )
    contextual: dict[tuple[int, tuple[int, ...]], tuple[type[Any], str, int]] = dataclasses.field(
        default_factory=dict[tuple[int, tuple[int, ...]], tuple[type[Any], str, int]]
    )
    base_depth: int = 0


# The scope of the class identities being computed (`_identifying`); None outside one.
_identity_scope: contextvars.ContextVar[_IdentityScope | None] = contextvars.ContextVar(
    "_identity_scope", default=None
)


def _identifying(compute: Callable[[], str]) -> str:
    """Run `compute`, suspending deep identity dependencies on an explicit worklist.

    A suspended dependency keeps its logical ancestors: its cycle distances and lowest
    back-reference stay the same as in recursive evaluation. Retrying its caller then reuses
    the result only in that context, or everywhere when it is free. No cyclic dependency is
    exempted from the stack-depth bound.
    """
    scope = _IdentityScope()
    token = _identity_scope.set(scope)
    try:
        needed = list[_IdentityTooDeepError]()
        while True:
            try:
                if not needed:
                    scope.base_depth = 0
                    return compute()
                first = needed[-1]
                scope.base_depth = len(first.active)
                active_token = _identity_active.set(first.active)
                try:
                    _computed_class_identity(first.cls)
                finally:
                    _identity_active.reset(active_token)
                needed.pop()
            except _IdentityTooDeepError as too_deep:
                needed.append(too_deep)
    finally:
        _identity_scope.reset(token)


def _enter_identity(cls: type[Any]) -> contextvars.Token[tuple[int, ...]]:
    """Put `cls` on `_identity_active` while its identity or hash is computed."""
    cast("_IdentityScope", _identity_scope.get()).lows.append(_NO_CYCLE)
    return _identity_active.set((*_identity_active.get(), id(cls)))


def _leave_identity(token: contextvars.Token[tuple[int, ...]]) -> int:
    """Take the class `token` put on `_identity_active` off it, and return the smallest depth a
    `^N` inside its computation pointed to, which the computation around it also takes."""
    lows = cast("_IdentityScope", _identity_scope.get()).lows
    low = lows.pop()
    if lows:
        lows[-1] = min(lows[-1], low)
    _identity_active.reset(token)
    return low


def _process_module_dependency(
    chain: str, module: types.ModuleType, func: Callable[..., Any]
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
    _mark_module_covered(module.__name__, _entries())
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
    """Return an object's module-qualified definition name for a manifest key. A class's names
    are read through `type`'s own descriptors (`_class_module`, `_class_qualname`), so no
    metaclass `__getattribute__` runs."""
    if isinstance(obj, type):
        module = _class_module(obj)
        name = module if type(module) is str and module else "<unknown>"
        return f"{name}.{_class_qualname(obj)}"
    module = getattr(obj, "__module__", None) or "<unknown>"
    return f"{module}.{_get_qualname_for_cache(obj)}"


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
    namespace (read through `type`'s own descriptor), and `inspect.getsource` finds a class by
    it."""
    return issubclass(type(obj), type) and "__firstlineno__" not in _class_namespace(obj)


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
    if (active_sources := _active_source_map.get()) is not None:
        info = _get_func_source_info(func)
        if info is not None:
            rel_path, digest = info
            active_sources.setdefault(rel_path, digest)
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
        # WeakKeyDictionary raises TypeError for non-weakly-referenceable functions (builtins).
        # A class whose metaclass defines `__hash__` or `__eq__` is not a key: the lookup would
        # run them. Nor is the hash of a class with no source computed inside a class identity
        # (`_identity_active`): it names a class being identified by its step up (`^N`), not by
        # its key, so it differs from the hash the class gets alone.
        if isinstance(func, type) and (
            not _hashes_by_identity(func) or (_identity_active.get() and _has_no_source(func))
        ):
            return _compute_function_hash(func)
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

        # 3. Compute hash. It never refuses: a class with no source that holds a value no form
        # reads is refused by its node's `ns:` edges (`_process_sourceless_namespace`).
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


def _sourceless_class_hash(cls: type) -> str:
    """The hash of a class with no source of its own (`collections.namedtuple`,
    `dataclasses.make_dataclass`, a functional `enum.Enum`, `type(...)`, `pydantic.create_model`):
    what defines it (`_sourceless_parts`). It is the same in every process, and it changes when
    what defines the class changes: its fields, its members, the code, cells and defaults of a
    function it holds, and the state of a value it holds, also one another module built.

    Nothing in the form is rewritten: a string keeps its content, a sentinel `object()` or a
    `__slots__` object is its state (`_form`). A value whose state no form reads is its type
    (`["opaque", <type>]`), so the hash never refuses: the class node's `ns:` edges refuse it,
    naming the class and the entry (`_process_sourceless_namespace`). The class is computed with
    itself on `_identity_active`, so a value that names it (an Enum member) names it by its step
    up (`^N`) in any order the hashes are asked for."""
    if _identity_scope.get() is None:
        return _identifying(lambda: _sourceless_class_hash(cls))
    if id(cls) in _identity_active.get():
        return xxhash.xxh64(json.dumps(_sourceless_parts(cls)).encode()).hexdigest()
    token = _enter_identity(cls)
    try:
        parts = _sourceless_parts(cls)
    finally:
        _leave_identity(token)
    return xxhash.xxh64(json.dumps(parts).encode()).hexdigest()


def _sourceless_parts(cls: type) -> list[Any]:
    """What defines a class with no source: its module, name and bases, then, for a pydantic
    model, `_pydantic_model_parts`, and for any other class each entry of its own namespace
    (`_sourceless_form`). A form holds no memory address, and no text in it is rewritten."""
    builder = _FormBuilder()
    builder.quiet += 1  # the hash is all this form is for: nothing in it is acted on
    parts: list[Any] = [
        "class",
        _class_module(cls),
        _class_qualname(cls),
        [_definition_ref(base) for base in _class_bases(cls)],
    ]
    namespace = _class_namespace(cls)
    if "__pydantic_fields__" in namespace and "__pydantic_decorators__" in namespace:
        parts.extend(_pydantic_model_parts(cls, builder))
    else:
        for name in sorted(namespace):
            if name not in _CLASS_NAMESPACE_NOISE:
                parts.append([name, _sourceless_form(namespace[name], builder)])
    return parts


def _process_sourceless_namespace(cls: type) -> None:
    """The edges of a class with no source for the values its namespace holds (`ns:NAME`), each a
    value root whose code is walked and whose unreadable state refuses the stage (`_admit`),
    naming the class and the entry. Its functions are its members (`_process_user_methods`) or
    code the class's hash records (`_sourceless_function_form`: what `namedtuple` and
    `make_dataclass` generate), and a pydantic model's generated entries are its schema's
    (`_process_model_schema`), so none of them is a value root. A cache hit reads the class's hash
    again where it reads the class."""
    key = _current_node().key
    namespace = _class_namespace(cls)
    pydantic = "__pydantic_fields__" in namespace
    for name in sorted(namespace):
        value = namespace[name]
        if (
            name in _IDENTITY_NOISE
            or (pydantic and (name in _PYDANTIC_GENERATED or name.startswith("__pydantic_")))
            or inspect.isfunction(value)
            or any(type(value) is wrapper for wrapper in _FUNCTION_WRAPPERS)
        ):
            continue
        with _walking(f"ns:{name}"):
            message = _sourceless_opaque_message(_definition_ref(cls), name)
            _admit(_form(value), f"{key}.{name}", message)


# The class members that hold a function for the class (`_class_member_parts`).
_FUNCTION_WRAPPERS = (staticmethod, classmethod, property, functools.cached_property)


def _pydantic_model_parts(cls: type, builder: _FormBuilder) -> list[Any]:
    """What defines a pydantic model with no source (`pydantic.create_model`), from what its
    author declared: each field's name and the public attributes of its `FieldInfo` (annotation,
    default, default factory, alias, constraints...), its config, the code and settings of each
    validator, serializer and computed field, its private attributes and class variables, and
    any other entry of its namespace. The core schema, validator and serializer pydantic builds
    from these are left out: their reprs hold addresses, and they differ between processes."""
    namespace = _class_namespace(cls)
    model = cast("Any", cls)
    fields = cast("dict[str, Any]", namespace["__pydantic_fields__"])
    field_forms = list[Any]()
    for name, info in fields.items():
        slot_names = cast("tuple[str, ...]", getattr(type(cast("object", info)), "__slots__", ()))
        slots = [slot for slot in slot_names if not slot.startswith("_")]
        field_forms.append(
            [name, [[slot, _sourceless_form(getattr(info, slot), builder)] for slot in slots]]
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
                    _sourceless_form(function, builder),
                    _decorator_info_form(decorator.info, builder),
                ]
            )
    private = cast("dict[str, Any]", namespace.get("__private_attributes__", {}))
    private_forms = [
        [
            name,
            _sourceless_form(attr.default, builder),
            _sourceless_form(attr.default_factory, builder),
        ]
        for name, attr in private.items()
    ]
    rest = [
        [name, _sourceless_form(namespace[name], builder)]
        for name in sorted(namespace)
        if name not in _CLASS_NAMESPACE_NOISE
        and name not in _PYDANTIC_GENERATED
        and not name.startswith("__pydantic_")
        and name not in decorator_names
        and name not in private
    ]
    return [
        ["fields", field_forms],
        ["config", _sourceless_form(dict(model.model_config), builder)],
        ["decorators", decorator_forms],
        ["private", private_forms],
        ["class-vars", sorted(cast("set[str]", namespace.get("__class_vars__", set[str]())))],
        ["namespace", rest],
    ]


def _decorator_info_form(info: Any, builder: _FormBuilder) -> Any:
    """The settings of a pydantic validator, serializer or computed field (a dataclass such as
    `FieldValidatorDecoratorInfo`), field by field (`_sourceless_form`), so a property or
    function in it is recorded by its code."""
    if not dataclasses.is_dataclass(info) or isinstance(info, type):
        return _sourceless_form(info, builder)
    return [
        _definition_ref(type(cast("object", info))),
        [
            [field.name, _sourceless_form(getattr(info, field.name), builder)]
            for field in dataclasses.fields(info)
        ],
    ]


def _sourceless_form(value: Any, builder: _FormBuilder) -> Any:
    """A form of one namespace entry of a class with no source (`_sourceless_class_hash`): a
    function by its code (`_sourceless_function_form`), a class by its name, a dataclass's fields
    by their names, types and defaults, and any other value by its form in `builder`."""
    if isinstance(value, (staticmethod, classmethod)):
        wrapper = cast("staticmethod[..., Any] | classmethod[Any, ..., Any]", value)
        return [type(wrapper).__name__, _sourceless_form(cast("Any", wrapper).__func__, builder)]
    if isinstance(value, property):
        return ["property", *(_sourceless_form(part, builder) for part in (value.fget, value.fset))]
    if inspect.isfunction(value):
        return _sourceless_function_form(value, builder)
    if isinstance(value, type):
        return ["class", _definition_ref(value)]
    if isinstance(value, dataclasses.Field):
        field = cast("dataclasses.Field[Any]", value)
        return [
            "field",
            field.name,
            _sourceless_form(field.type, builder),
            _sourceless_form(field.default, builder),
            _sourceless_form(field.default_factory, builder),
        ]
    if value is dataclasses.MISSING:
        return ["missing"]
    if isinstance(value, dict) and any(
        isinstance(item, dataclasses.Field) for item in cast("dict[Any, Any]", value).values()
    ):
        mapping = cast("dict[str, Any]", value)
        return [
            "fields",
            [[name, _sourceless_form(item, builder)] for name, item in mapping.items()],
        ]
    return builder.build(value)


def _sourceless_function_form(function: types.FunctionType, builder: _FormBuilder) -> Any:
    """A function a class with no source holds: by its identity (code hash, cells and defaults)
    when it was compiled from a user source file, by its name when from a library file, and
    otherwise (code generated by `exec`, as `namedtuple` and `make_dataclass` make it) by its
    code object and defaults."""
    filename = function.__code__.co_filename
    if _is_user_source_file(filename):
        return ["function", builder.identity(function)]
    if os.path.isfile(filename):
        return ["external", f"{function.__module__}.{function.__qualname__}"]
    return [
        "code",
        _code_form(function.__code__, builder),
        _sourceless_form(function.__defaults__, builder),
        _sourceless_form(function.__kwdefaults__, builder),
    ]


def _code_form(code: types.CodeType, builder: _FormBuilder) -> list[Any]:
    """The parts of a code object that decide what it does, without its file name or line
    numbers; constants by their form, so a frozenset constant is not in hash-seed order."""
    return [
        code.co_code.hex(),
        [
            _code_form(constant, builder)
            if isinstance(constant, types.CodeType)
            else builder.build(constant)
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


def _wrapper_loop_message(obj: Any) -> str | None:
    """Name a loop in statically readable wrapper links, including partial targets.

    This only inspects supported function/wrapper objects, never an arbitrary value's hooks.
    It is pure: replay and identity construction must not report refusals.
    """
    if not issubclass(_type_of(obj), functools.partial) and _read_step(obj, "wr") is _UNREAD:
        return None
    chain = list[Any]()
    positions = dict[int, int]()
    current = obj
    while issubclass(_type_of(current), (types.FunctionType, _LRU_CACHE_TYPE, functools.partial)):
        if (position := positions.get(id(current))) is not None:
            loop = chain[position:]
            names = list[str]()
            for item in loop:
                if issubclass(_type_of(item), functools.partial):
                    attributes = _instance_dict(_type_of(item), item) or {}
                    label = attributes.get("__qualname__")
                    names.append(
                        f"functools.partial '{label}'"
                        if type(label) is str
                        else "functools.partial"
                    )
                else:
                    names.append(_definition_ref(item))
            return (
                f"__wrapped__ loop involving {', '.join(dict.fromkeys(names))}.\n"
                "Fix: break the wrapper loop or move this value into the stage's StageParams."
            )
        positions[id(current)] = len(chain)
        chain.append(current)
        wrapped = _read_step(current, "wr")
        if wrapped is _UNREAD:
            if not issubclass(_type_of(current), functools.partial):
                return None
            wrapped = current.func
        current = wrapped
    return None


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
    """Refuse `value`, on whose chain a user class instance is above a mark, naming both
    (`_class_above_mark_message`), with the stage's other refusals (`_record_refusal`). Unsafe
    fingerprinting does not suppress this refusal: the fixes are local to the helper."""
    _record_refusal(_class_above_mark_message(value), downgradable=False)


def _class_above_mark_message(value: Any) -> str:
    """The refusal of `value`, on whose chain a user class instance is above a mark: the mark
    records only the object below the instance, and no fingerprint records a class instance's
    code and state, so an edit to the class would go unseen."""
    instance = _class_above_mark(value.__func__ if inspect.ismethod(value) else value)
    marked = _mark_on_chain(instance)  # below the instance: `_class_above_mark` stops at a mark
    decorator = _definition_ref(type(instance))
    helper = _library_name(marked) if marked is not None else _definition_ref(instance)
    stage = _current_stage_name.get()
    prefix = f"Stage '{stage}': " if stage else ""
    return (
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
    A loop has no source target and returns None; the function node reports its named refusal.
    """
    decision = _mark_decision(func)
    if decision is _MarkDecision.BY_NAME or decision is _MarkDecision.REFUSED:
        return None
    try:
        target = unwrap(func, stop=_stop_at_mark)
    except ValueError:
        return None
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
        if issubclass(type(obj), type):
            # Decided from the module that defines the class (`_is_user_class`), read through
            # `type`'s own descriptor: the weak cache would hash the class and `__module__` would
            # be asked of its metaclass, either of which can run the metaclass's code.
            return not (_marks_made and fingerprinted_by_name(obj)) and _is_user_class(obj)

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


def _hashes_by_identity(cls: type[Any]) -> bool:
    """Whether hashing class `cls` or comparing it for equality runs no Python code: its
    metaclass keeps `type`'s identity `__hash__` and `__eq__`, read along the metaclass's real
    MRO (`_class_mro`). Only such a class is a key of the weak caches (`is_user_code`,
    `hash_function_ast`)."""
    meta = type(cls)
    if meta is type:
        return True
    for klass in _class_mro(meta):
        namespace = _class_namespace(klass)
        if "__hash__" in namespace or "__eq__" in namespace:
            return klass is type or klass is object
    return True


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
