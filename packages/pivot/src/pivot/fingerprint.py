import ast
import atexit
import contextlib
import contextvars
import dataclasses
import functools
import inspect
import json
import logging
import marshal
import os
import pathlib
import sys
import textwrap
import types
import typing
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

import xxhash

from pivot import ast_utils, exceptions, metrics

if TYPE_CHECKING:
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
_CACHE_SCHEMA_VERSION = 10
_REPR_SIZE_LIMIT = 10_000

_SITE_PACKAGE_PATHS = ("site-packages", "dist-packages")

# Type alias for AST hash cache entry tuple.
# Format: (rel_path, mtime_ns, size, inode, qualname, py_version, schema_version, hash_hex)
type AstHashEntry = tuple[str, int, int, int, str, str, int, str]


class _PydanticModelProtocol(Protocol):
    model_fields: Mapping[str, Any]

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]: ...


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
_pending_ast_writes: list[AstHashEntry] = []

# Pending manifest cache writes, flushed at process exit.
# Format: list of (key_bytes, value_bytes) tuples.
_pending_manifest_writes: list[tuple[bytes, bytes]] = []


# Source files visited during a single stage's fingerprinting.
# Maps rel_path -> (mtime_ns, size, ino). Set by _collecting_sources().
_active_source_map: dict[str, tuple[int, int, int]] | None = None

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


def _close_state_db() -> None:
    """Close the readonly StateDB on process exit."""
    global _state_db
    if _state_db is not None:
        _state_db.close()
        _state_db = None


atexit.register(_close_state_db)


@atexit.register
def _flush_pending_caches() -> None:  # pyright: ignore[reportUnusedFunction] - called by atexit
    """Flush all pending cache writes at process exit.

    Registered AFTER _close_state_db so LIFO ordering runs this first.
    Flush opens its own writable StateDB, so readonly close is irrelevant.
    """
    flush_ast_hash_cache()
    flush_manifest_cache()


def _get_state_db() -> "StateDB | None":
    """Get readonly StateDB for fingerprint caching (graceful degradation).

    Not thread-safe, but fingerprinting runs single-threaded per process
    (see comment at line 33). Workers use separate memory spaces.
    """
    global _state_db, _state_db_init_attempted
    if _state_db is not None:
        return _state_db
    if _state_db_init_attempted:
        return None  # Already tried and failed
    _state_db_init_attempted = True
    try:
        from pivot.config import io
        from pivot.storage import state

        _state_db = state.StateDB(io.get_state_dir(), readonly=True)
        return _state_db
    except Exception:
        # OSError (filesystem), ImportError (module), lmdb.Error, etc.
        return None


def _make_manifest_cache_key(stage_name: str, func: Callable[..., Any]) -> bytes:
    """Build StateDB key for the manifest of `func` registered as `stage_name`.

    The key names the function as well as the stage, because two pipeline files can register
    the same stage name with different functions. A cached manifest for one never answers for
    the other.
    """
    module = getattr(func, "__module__", None) or "<unknown>"
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", "<unknown>")
    code = getattr(func, "__code__", None)
    source = code.co_filename if isinstance(code, types.CodeType) else "<unknown>"
    identity = f"{module}\x00{qualname}\x00{_portable_source_path(source)}"
    return f"sm:{stage_name}\x00{identity}\x00{_PYTHON_VERSION}\x00{_CACHE_SCHEMA_VERSION}".encode()


def _portable_source_path(source: str) -> str:
    """Return `source` relative to the project root, so a moved project keeps its cache."""
    from pivot import project

    try:
        return str(pathlib.Path(source).resolve().relative_to(project.get_project_root()))
    except (OSError, ValueError):
        return source


def _get_func_source_info(func: Callable[..., Any]) -> tuple[str, int, int, int] | None:
    """Get (rel_path, mtime_ns, size, inode) for function source file.

    Returns None if source info unavailable (builtins, exec'd code, outside project).
    """
    try:
        file = inspect.getsourcefile(func)
        if file is None:
            return None
        path = pathlib.Path(file).resolve()
        stat = path.stat()
        from pivot import project

        project_root = project.get_project_root()
        rel_path = str(path.relative_to(project_root))
        return (rel_path, stat.st_mtime_ns, stat.st_size, stat.st_ino)
    except (TypeError, OSError, ValueError):
        # TypeError: builtins, exec'd
        # OSError: file doesn't exist
        # ValueError: path outside project root
        return None


@contextlib.contextmanager
def _collecting_sources() -> Iterator[dict[str, tuple[int, int, int]]]:
    """Scope a source map for the duration of a fingerprint walk.

    Saves and restores any previously active source map, so nested calls
    (if they ever occur) don't clobber the outer collector.
    """
    global _active_source_map
    previous = _active_source_map
    source_map = dict[str, tuple[int, int, int]]()
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


def _record_project_modules(source_map: dict[str, tuple[int, int, int]]) -> None:
    """Add the file of every loaded project module to a manifest's source map.

    A manifest records values bound in module namespaces, and import-time code in any project
    module can change them without touching the walked functions' files: `from settings import
    FACTOR`, a module rebinding an imported helper, or one module assigning another's
    attribute. A cached manifest is therefore valid only while none of those files changed.
    """
    from pivot import project

    for rel_path, path in _project_module_files(project.get_project_root()):
        try:
            stat = path.stat()
        except OSError:
            continue
        source_map.setdefault(rel_path, (stat.st_mtime_ns, stat.st_size, stat.st_ino))


def _try_manifest_cache_hit(stage_name: str, func: Callable[..., Any]) -> dict[str, str] | None:
    """Try to load a cached manifest; returns None on miss."""
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
    if not isinstance(manifest_raw, dict) or not isinstance(sources_raw, dict):
        return None

    # Narrow types after validation: manifest is {str: str}, sources is {str: [int, int, int]}
    manifest = cast("dict[str, str]", manifest_raw)
    sources = cast("dict[str, list[int]]", sources_raw)

    # If no source files were tracked, we have nothing to validate against.
    # This happens for stages with only builtins/exec'd code. Force recompute
    # rather than returning a stale manifest that can never be invalidated.
    if not sources:
        return None

    # Stat-check every source file
    from pivot import project

    project_root = project.get_project_root()
    for rel_path, stats in sources.items():
        if len(stats) != 3:
            return None  # Corrupted cache entry
        cached_mtime, cached_size, cached_ino = stats
        # Guard against path traversal from corrupted/malicious cache entries
        if rel_path.startswith("/") or ".." in pathlib.Path(rel_path).parts:
            return None
        try:
            st = (project_root / rel_path).stat()
        except OSError:
            return None  # File deleted or inaccessible
        if st.st_mtime_ns != cached_mtime or st.st_size != cached_size or st.st_ino != cached_ino:
            return None  # File changed

    return manifest


def flush_ast_hash_cache() -> None:
    """Flush pending AST hash writes to StateDB (call from coordinator)."""
    global _pending_ast_writes
    if not _pending_ast_writes:
        return

    pending = _pending_ast_writes
    _pending_ast_writes = []

    try:
        from pivot.config import io
        from pivot.storage import state

        with state.StateDB(io.get_state_dir(), readonly=False) as db:
            db.save_ast_hash_many(pending)
        metrics.count("fingerprint.ast_hash_cache.flush")
    except Exception:
        # OSError (filesystem), ImportError (module), lmdb.Error, etc.
        # Restore pending writes on failure so they can be retried on next flush.
        # This prevents permanent loss of cache entries on transient failures.
        _pending_ast_writes.extend(pending)
        _logger.debug("Failed to flush AST hash cache (%d entries)", len(pending), exc_info=True)


def flush_manifest_cache() -> None:
    """Flush pending manifest writes to StateDB."""
    global _pending_manifest_writes
    if not _pending_manifest_writes:
        return

    pending = _pending_manifest_writes
    _pending_manifest_writes = []

    try:
        from pivot.config import io
        from pivot.storage import state

        with state.StateDB(io.get_state_dir(), readonly=False) as db:
            db.put_raw_many(pending)
        metrics.count("fingerprint.manifest_cache.flush")
    except Exception:
        _pending_manifest_writes.extend(pending)
        _logger.debug("Failed to flush manifest cache (%d entries)", len(pending), exc_info=True)


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
    sources = cast("dict[str, list[int]]", sources_raw)
    return any(source_path in changed_paths for source_path in sources)


def invalidate_manifests_for_paths(
    paths: Sequence[str | os.PathLike[str] | pathlib.Path],
) -> None:
    """Invalidate cached manifests that reference changed source files."""
    global _pending_manifest_writes

    if not paths:
        return
    changed_paths = _normalize_changed_paths(paths)
    if not changed_paths:
        return

    from pivot.config import io
    from pivot.storage import state

    keys_to_delete = set[bytes]()
    try:
        with state.StateDB(io.get_state_dir(), readonly=True) as db:
            for key, raw in db.iter_prefix(b"sm:"):
                if _manifest_references_paths(raw, changed_paths):
                    keys_to_delete.add(key)
    except Exception:
        _logger.debug("Failed to scan manifest cache for invalidation", exc_info=True)

    if _pending_manifest_writes:
        pending_keys = set[bytes]()
        for key, raw in _pending_manifest_writes:
            if _manifest_references_paths(raw, changed_paths):
                pending_keys.add(key)
        if pending_keys:
            keys_to_delete.update(pending_keys)
            _pending_manifest_writes = [
                (key, value) for key, value in _pending_manifest_writes if key not in pending_keys
            ]

    if not keys_to_delete:
        return

    try:
        with state.StateDB(io.get_state_dir(), readonly=False) as db:
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
    - 'mod:<module>.<attr>': User-code module attributes (hash for callables, repr for primitives)
    - 'const:<module>.<name>': Global primitive constants (repr value)
    """
    if visited is None:
        visited = set()

    errors = _capture_errors.get()
    token: contextvars.Token[list[str] | None] | None = None
    if errors is None:
        errors = list[str]()
        token = _capture_errors.set(errors)

    _t = metrics.start()
    try:
        result = _get_stage_fingerprint_impl(func, visited)
        if token is not None and errors:
            raise exceptions.StageDefinitionError("\n\n".join(errors))
        return result
    finally:
        metrics.end("fingerprint.get_stage_fingerprint", _t)
        if token is not None:
            _capture_errors.reset(token)


def get_stage_fingerprint_cached(stage_name: str, func: Callable[..., Any]) -> dict[str, str]:
    """Like get_stage_fingerprint, but with manifest-level caching.

    On hit, skips the entire closure walk. On miss, computes normally and
    queues the result for flush at process exit.
    """
    _t = metrics.start()

    # Try cache hit
    cached = _try_manifest_cache_hit(stage_name, func)
    if cached is not None:
        metrics.count("fingerprint.manifest_cache.hit")
        metrics.end("fingerprint.get_stage_fingerprint_cached", _t)
        return cached

    metrics.count("fingerprint.manifest_cache.miss")
    unsafe_token = _unsafe_capture_warning.set(False)
    try:
        # Compute with source tracking
        token = _current_stage_name.set(stage_name)
        try:
            with _collecting_sources() as source_map:
                manifest = get_stage_fingerprint(func)
        finally:
            _current_stage_name.reset(token)

        if not _unsafe_capture_warning.get():
            _record_project_modules(source_map)
            key = _make_manifest_cache_key(stage_name, func)
            value = json.dumps(
                {
                    "m": manifest,
                    "s": {rel_path: list(stats) for rel_path, stats in source_map.items()},
                },
                separators=(",", ":"),
            ).encode()
            _pending_manifest_writes.append((key, value))

        return manifest
    finally:
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


def _check_dynamic_name_access(func: Callable[..., Any]) -> None:
    try:
        source = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(source))
    except (OSError, TypeError, SyntaxError):
        return

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
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
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

    if is_user_code(func):
        _check_dynamic_name_access(func)

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
        return manifest
    metrics.end("fingerprint.getclosurevars", _t_closure)

    _process_closure_values(closure_vars.globals, func, manifest, visited, skip_dunders=True)
    _process_closure_values(closure_vars.nonlocals, func, manifest, visited, skip_dunders=False)

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
            for name in new_names:
                if name in func_globals:
                    nested_globals_dict[name] = func_globals[name]
            if nested_globals_dict:
                _process_closure_values(
                    nested_globals_dict, func, manifest, visited, skip_dunders=True
                )

    _process_default_values(func, manifest, visited)
    _process_type_hint_dependencies(func, manifest, visited)

    return manifest


def _process_class_body_dependencies(
    cls: type, manifest: dict[str, str], visited: set[int]
) -> None:
    try:
        source = inspect.getsource(cls)
        tree = ast.parse(textwrap.dedent(source))
    except (OSError, TypeError, SyntaxError):
        return

    class_def = next((node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)), None)
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

    for name in names:
        if name not in ns:
            continue
        value = ns[name]
        if isinstance(value, type) and is_user_code(value):
            key = f"class:{_definition_ref(value)}"
            if key not in manifest:
                _add_callable_to_manifest(key, value, manifest, visited)
            if hasattr(value, "model_fields"):
                _hash_pydantic_schema(
                    cast("type[_PydanticModelProtocol]", value), manifest, visited
                )

    for parts in dotted_refs:
        resolved = _resolve_dotted_path(parts, ns)
        if resolved is None or not isinstance(resolved, type) or not is_user_code(resolved):
            continue
        key = f"class:{_definition_ref(resolved)}"
        if key not in manifest:
            _add_callable_to_manifest(key, resolved, manifest, visited)
        if hasattr(resolved, "model_fields"):
            _hash_pydantic_schema(cast("type[_PydanticModelProtocol]", resolved), manifest, visited)


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


def get_loader_fingerprint(loader: "loaders.Writer[Any] | loaders.Reader[Any]") -> dict[str, str]:
    """Generate fingerprint manifest for a loader instance.

    Handles Reader, Writer, and Loader (which inherits from both) instances.
    Only fingerprints methods that exist on the handler type:
    - Reader: 'loader:<module>.<qualname>:load' + empty() if overridden
    - Writer: 'loader:<module>.<qualname>:save'
    - Loader: All of the above
    - Always: 'loader:<module>.<qualname>:config' for dataclass field values
    """
    manifest = dict[str, str]()
    class_name = _definition_ref(type(loader))

    # Import here to avoid circular import
    from pivot import loaders as loaders_module

    # Fingerprint save() if this is a Writer
    if isinstance(loader, loaders_module.Writer):
        manifest[f"loader:{class_name}:save"] = hash_function_ast(loader.save)

    # Fingerprint load() if this is a Reader
    if isinstance(loader, loaders_module.Reader):
        manifest[f"loader:{class_name}:load"] = hash_function_ast(loader.load)

        # Only fingerprint empty() if it's overridden in a Loader subclass
        if isinstance(loader, loaders_module.Loader):
            # Cast to Loader[Any, Any] - isinstance narrows but basedpyright keeps Unknown params
            typed_loader = cast("loaders_module.Loader[Any, Any]", loader)
            for cls in type(typed_loader).__mro__:
                if cls is loaders_module.Loader:
                    # Reached Loader base class without finding override
                    break
                if "empty" in cls.__dict__:
                    # Found override in a subclass
                    manifest[f"loader:{class_name}:empty"] = hash_function_ast(typed_loader.empty)
                    break

    # Config hash from dataclass fields
    # Cast to Any for dataclass introspection - loader is a dataclass but generic params are unknown
    loader_any = cast("Any", loader)
    field_values = list[str]()
    for field in dataclasses.fields(loader_any):
        value = getattr(loader_any, field.name)
        field_values.append(f"{field.name}={value!r}")
    if field_values:
        config_str = ",".join(field_values)
        manifest[f"loader:{class_name}:config"] = xxhash.xxh64(config_str.encode()).hexdigest()

    return manifest


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
        + "Pivot tracks only module-level bool, int, float, str, bytes, and None values. "
        + "Other values may change without a reliable fingerprint, which can cause silent wrong "
        + "outputs.\n"
        + "Fix: move this value into the stage's StageParams, or read it from a pipeline input.\n"
        + "To suppress: set core.unsafe_fingerprinting=true or PIVOT_UNSAFE_FINGERPRINTING=1"
    )
    if _is_unsafe_fingerprinting_enabled():
        _unsafe_capture_warning.set(True)
        _logger.warning(message)
        return
    errors = _capture_errors.get()
    if errors is None:
        raise exceptions.StageDefinitionError(message)
    errors.append(message)


def _check_data_class_methods(cls: type) -> None:
    allowed = frozenset({"model_post_init", "model_validate", "model_rebuild"})
    user_methods = [
        name
        for name, val in vars(cls).items()
        if callable(val)
        and not (name.startswith("__") and name.endswith("__"))
        and name not in allowed
    ]
    if user_methods:
        message = (
            f"Data class '{cls.__qualname__}' has methods {user_methods}. "
            + "Move these to standalone functions for reliable change detection."
        )
        raise exceptions.StageDefinitionError(message)


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
    """Return whether a non-user module binds `value` under its own `__name__`.

    The module named by `__module__` is checked. An object without one (scipy.special's ufuncs,
    such as `expit`) is looked up in every loaded module; `np.frompyfunc` products stay
    refused because no module binds their generated names.
    """
    name = getattr(value, "__name__", None)
    if not isinstance(name, str):
        return False
    module_name = getattr(value, "__module__", None)
    candidates: list[types.ModuleType | None]
    if isinstance(module_name, str):
        candidates = [sys.modules.get(module_name)]
    else:
        candidates = list(sys.modules.values())
    return any(
        isinstance(module, types.ModuleType)
        and module.__dict__.get(name) is value
        and not is_user_code(module)
        for module in candidates
    )


def _process_closure_values(
    values: Mapping[str, Any],
    func: Callable[..., Any],
    manifest: dict[str, str],
    visited: set[int],
    *,
    skip_dunders: bool,
) -> None:
    """Process closure variable values (globals or nonlocals) and add to manifest."""
    stage_name = _current_stage_name.get() or getattr(func, "__name__", "<unknown>")
    for name, value in values.items():
        if skip_dunders and name.startswith("__"):
            continue
        _process_captured_value(
            name, _binding_ref(func, name), value, func, manifest, visited, stage_name
        )


def _process_captured_value(
    name: str,
    binding_ref: str,
    value: Any,
    func: Callable[..., Any],
    manifest: dict[str, str],
    visited: set[int],
    stage_name: str,
    *,
    parameter: str | None = None,
) -> None:
    """Fingerprint one value `func` reads, or refuse it as a non-scalar capture.

    `parameter` names the parameter when the value is a default argument.
    """
    # functools.partial fails is_user_code() (module is functools/stdlib).
    if isinstance(value, functools.partial):
        _process_partial_dependency(
            binding_ref, cast("functools.partial[Any]", value), manifest, visited
        )
    elif _is_callable_instance(value) or _is_user_class_instance(value):
        _check_non_scalar_capture(name, value, stage_name, parameter=parameter)
    elif callable(value) and is_user_code(value):
        _process_callable_dependency(binding_ref, value, manifest, visited)
    elif callable(value):
        # External functions, classes, and builtins are not user dependencies.
        return
    elif isinstance(value, types.ModuleType):
        _process_module_dependency(parameter or name, value, func, manifest, visited)
    elif isinstance(value, logging.Logger):
        return
    elif isinstance(value, (bool, int, float, str, bytes, type(None))):
        manifest[f"const:{binding_ref}"] = repr(value)
    else:
        _check_non_scalar_capture(name, value, stage_name, parameter=parameter)


def _process_default_values(
    func: Callable[..., Any], manifest: dict[str, str], visited: set[int]
) -> None:
    """Fingerprint default argument values, which no code hash records.

    A default is evaluated once, when its module loads, so `cols=COLUMNS` puts the name
    `COLUMNS` in the AST and never its value. A scalar or a StageParams instance is tracked by
    value: a stage's own params hash does not cover a StageParams default on a helper. Any
    other value is treated like a captured module value, unless its default is a literal that
    the function's AST hash already records, such as `keys=("a", "b")`.
    """
    defaults = _function_defaults(func)
    if not defaults:
        return
    from pivot import stage_def

    stage_name = _current_stage_name.get() or getattr(func, "__name__", "<unknown>")
    func_ref = _definition_ref(func)
    expressions: dict[str, ast.expr] | None = None
    for parameter, value in defaults:
        binding_ref = f"{func_ref}.{parameter}"
        if isinstance(value, (bool, int, float, str, bytes, type(None))):
            manifest[f"default:{binding_ref}"] = repr(value)
            continue
        if isinstance(value, stage_def.StageParams):
            serialized = _serialize_value_for_hash(value)
            manifest[f"default:{binding_ref}"] = xxhash.xxh64(serialized.encode()).hexdigest()
            continue
        if expressions is None:
            expressions = _default_expressions(func)
        expression = expressions.get(parameter)
        if expression is not None and _is_literal(expression):
            continue
        source = ast.unparse(expression) if expression is not None else parameter
        _process_captured_value(
            source, binding_ref, value, func, manifest, visited, stage_name, parameter=parameter
        )


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
    prefix = "class" if isinstance(func, type) else "func"
    _add_callable_to_manifest(
        f"{prefix}:{_callable_ref(func, binding_ref)}", func, manifest, visited
    )


def _process_partial_dependency(
    binding_ref: str,
    partial_obj: functools.partial[Any],
    manifest: dict[str, str],
    visited: set[int],
) -> None:
    """Process functools.partial: hash underlying func and bound arguments."""
    args_str = _serialize_value_for_hash(partial_obj.args)
    kwargs_str = _serialize_value_for_hash(partial_obj.keywords)
    manifest[f"partial:{binding_ref}.args"] = xxhash.xxh64(args_str.encode()).hexdigest()
    manifest[f"partial:{binding_ref}.kwargs"] = xxhash.xxh64(kwargs_str.encode()).hexdigest()

    underlying = partial_obj.func
    if callable(underlying) and is_user_code(underlying):
        callable_ref = _callable_ref(underlying, f"{binding_ref}.func")
        _add_callable_to_manifest(f"func:{callable_ref}", underlying, manifest, visited)


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


def _process_type_hint_dependencies(
    func: Callable[..., Any], manifest: dict[str, str], visited: set[int]
) -> None:
    """Process user-defined classes in type hints, including Pydantic model defaults."""
    _t = metrics.start()
    # Check cache first. TypeError raised for non-weakly-referenceable functions (builtins).
    try:
        hints = _get_type_hints_cache.get(func)
    except TypeError:
        hints = None  # Not in cache and can't be cached

    if hints is None:
        try:
            hints = typing.get_type_hints(func)
        except Exception:
            hints = _resolve_annotations_individually(func)
        # Cache result if function is weakly referenceable
        with contextlib.suppress(TypeError):
            _get_type_hints_cache[func] = hints

    metrics.end("fingerprint.get_type_hints", _t)

    for hint in hints.values():
        _process_type_hint(hint, manifest, visited)


def _process_type_hint(hint: Any, manifest: dict[str, str], visited: set[int]) -> None:
    """Process a single type hint, recursively handling generics."""
    if hint is type(None):
        return

    origin = typing.get_origin(hint)
    if origin is not None:
        # Track user-defined generic classes (e.g., MyGeneric[int] -> hash MyGeneric)
        if isinstance(origin, type) and is_user_code(origin):
            origin_type = cast("type[Any]", origin)
            key = f"class:{_definition_ref(origin_type)}"
            if key not in manifest:
                _add_callable_to_manifest(key, origin_type, manifest, visited)
            if hasattr(origin_type, "model_fields"):
                _hash_pydantic_schema(
                    cast("type[_PydanticModelProtocol]", origin_type), manifest, visited
                )
        for arg in typing.get_args(hint):
            _process_type_hint(arg, manifest, visited)
        return

    if not isinstance(hint, type):
        return

    hint_type = cast("type[Any]", hint)

    if not is_user_code(hint_type):
        return
    if dataclasses.is_dataclass(hint_type):
        _check_data_class_methods(hint_type)

    key = f"class:{_definition_ref(hint_type)}"
    if key not in manifest:
        _add_callable_to_manifest(key, hint_type, manifest, visited)

    # Always hash Pydantic schema (even if class was already added via closure vars)
    if hasattr(hint_type, "model_fields"):
        _hash_pydantic_schema(cast("type[_PydanticModelProtocol]", hint_type), manifest, visited)


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
    schema_model = model
    schema_key = f"schema:{_definition_ref(schema_model)}"
    if schema_key in manifest:
        return
    # Insert placeholder to prevent infinite recursion on self-referential models
    # (e.g., class Node(BaseModel): children: list["Node"])
    manifest[schema_key] = "<pending>"
    schema_raw = schema_model.model_json_schema()
    schema = _strip_schema_metadata(schema_raw)

    model_fields = cast("Mapping[str, Any]", getattr(schema_model, "model_fields", {}))
    default_hashes = dict[str, str]()
    for field_name, field_info in model_fields.items():
        annotation = getattr(field_info, "annotation", None)
        if annotation is not None:
            _discover_pydantic_field_types(annotation, manifest, visited)

        default = getattr(field_info, "default", None)
        default_factory = getattr(field_info, "default_factory", None)
        if type(default).__name__ == "PydanticUndefinedType":
            if default_factory is not None and callable(default_factory):
                default_hashes[field_name] = hash_function_ast(default_factory)
            continue

        value_str = _serialize_value_for_hash(default)
        default_hashes[field_name] = xxhash.xxh64(value_str.encode()).hexdigest()

    payload = {"schema": schema, "defaults": default_hashes}
    manifest[schema_key] = xxhash.xxh64(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _discover_pydantic_field_types(
    annotation: Any, manifest: dict[str, str], visited: set[int]
) -> None:
    origin = typing.get_origin(annotation)
    if origin is not None:
        for arg in typing.get_args(annotation):
            _discover_pydantic_field_types(arg, manifest, visited)
        return
    if isinstance(annotation, type) and is_user_code(annotation):
        annotation_type = cast("type[Any]", annotation)
        key = f"class:{_definition_ref(annotation_type)}"
        if key not in manifest:
            _add_callable_to_manifest(key, annotation_type, manifest, visited)
        if hasattr(annotation_type, "model_fields"):
            _hash_pydantic_schema(
                cast("type[_PydanticModelProtocol]", annotation_type), manifest, visited
            )


def _serialize_value_for_hash(value: Any) -> str:
    """Serialize a value to a stable string for hashing."""
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(), sort_keys=True, default=str)

    if isinstance(value, (list, tuple)):
        items: list[Any] = []
        for item in cast("list[Any]", value):
            if hasattr(item, "model_dump"):
                items.append(item.model_dump())
            else:
                items.append(item)
        return json.dumps(items, sort_keys=True, default=str)

    if isinstance(value, (set, frozenset)):
        # Sort for deterministic ordering
        items_to_sort = cast("set[Any] | frozenset[Any]", value)
        return json.dumps(
            sorted(items_to_sort, key=lambda x: (type(x).__name__, str(x))), default=str
        )

    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)

    return repr(value)


def _add_callable_to_manifest(
    key: str, func: Callable[..., Any], manifest: dict[str, str], visited: set[int]
) -> None:
    """Hash callable and merge its transitive dependencies into manifest."""
    manifest[key] = hash_function_ast(func)
    for child_key, child_val in get_stage_fingerprint(func, visited).items():
        if not child_key.startswith("self:"):
            manifest[child_key] = child_val


def _process_module_dependency(
    name: str,
    module: types.ModuleType,
    func: Callable[..., Any],
    manifest: dict[str, str],
    visited: set[int],
) -> None:
    """Process module attribute dependencies and add to manifest."""
    # Only track attributes from user-code modules.
    # Non-user-code modules (numpy, pandas, stdlib) are not tracked because:
    # - AST already captures which functions are called
    # - We can't detect library implementation changes anyway
    # - repr() of objects like np.c_ contains memory addresses (non-deterministic)
    if not is_user_code(module):
        return

    _t = metrics.start()
    attrs = ast_utils.extract_module_attr_usage(func)
    metrics.end("fingerprint.extract_module_attr_usage", _t)
    module_name = getattr(module, "__name__", name)

    for mod_name, attr_name in attrs:
        if mod_name not in (name, module_name):
            continue
        key = f"mod:{module_name}.{attr_name}"
        if key in manifest:
            continue
        try:
            attr_value = getattr(module, attr_name)
        except AttributeError:
            manifest[key] = "unknown"
            continue
        if _is_callable_instance(attr_value) or _is_user_class_instance(attr_value):
            _check_non_scalar_capture(
                f"{module_name}.{attr_name}",
                attr_value,
                _current_stage_name.get() or getattr(func, "__name__", "<unknown>"),
            )
        elif callable(attr_value) and is_user_code(attr_value):
            _add_callable_to_manifest(key, attr_value, manifest, visited)
        elif isinstance(attr_value, (bool, int, float, str, bytes, type(None))):
            manifest[key] = repr(attr_value)
        else:
            _check_non_scalar_capture(
                f"{module_name}.{attr_name}",
                attr_value,
                _current_stage_name.get() or getattr(func, "__name__", "<unknown>"),
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


def _binding_ref(func: Callable[..., Any], name: str) -> str:
    """Return a module-qualified reference for a value reached through `func`."""
    module = getattr(func, "__module__", None) or "<unknown>"
    return f"{module}.{name}"


def _callable_ref(func: Callable[..., Any], binding_ref: str) -> str:
    """Use callable's definition, retaining the binding path for anonymous lambdas."""
    qualname = getattr(func, "__qualname__", "") or ""
    if "<lambda>" in qualname:
        return binding_ref
    return _definition_ref(func)


def _should_skip_persistent_cache(func: Callable[..., Any]) -> bool:
    """Check if function should skip persistent cache.

    Skip for:
    - Closures: `<locals>` in qualname → qualname collision risk
    - Wrapped functions: has `__wrapped__` → source file mismatch risk
    """
    qualname = getattr(func, "__qualname__", "")
    if "<locals>" in qualname:
        return True
    return hasattr(func, "__wrapped__")


def hash_function_ast(func: Callable[..., Any]) -> str:
    """Hash function AST (ignores whitespace, comments, docstrings).

    Uses persistent cache in StateDB when available, keyed by
    (file_path, mtime_ns, size, inode, qualname, py_version, schema_version)
    for automatic invalidation on file changes or Python upgrades.

    Limitation: Lambdas and functions without source code fall back to id(func),
    which is non-deterministic across runs. This causes unnecessary re-runs for
    stages using lambdas. Mitigation: Use named functions instead of lambdas in
    pipeline stages for stable fingerprinting.
    """
    if _active_source_map is not None:
        info = _get_func_source_info(func)
        if info is not None:
            rel_path, mtime_ns, size, ino = info
            if rel_path not in _active_source_map:
                _active_source_map[rel_path] = (mtime_ns, size, ino)
    _t = metrics.start()
    try:
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
        source_info: tuple[str, int, int, int] | None = None
        qualname: str | None = None

        if not _should_skip_persistent_cache(func):
            source_info = _get_func_source_info(func)
            if source_info is not None:
                qualname = _get_qualname_for_cache(func)
                db = _get_state_db()
                if db is not None:
                    rel_path, mtime_ns, size, inode = source_info
                    try:
                        persistent_cached = db.get_ast_hash(
                            rel_path,
                            mtime_ns,
                            size,
                            inode,
                            qualname,
                            _PYTHON_VERSION,
                            _CACHE_SCHEMA_VERSION,
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

        # 5. Queue persistent write (if source info available and not skipping)
        if source_info is not None and qualname is not None:
            rel_path, mtime_ns, size, inode = source_info
            _pending_ast_writes.append(
                (
                    rel_path,
                    mtime_ns,
                    size,
                    inode,
                    qualname,
                    _PYTHON_VERSION,
                    _CACHE_SCHEMA_VERSION,
                    result,
                )
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

        _t_source = metrics.start()
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):
            metrics.end("fingerprint.inspect_getsource", _t_source)
            if hasattr(func, "__code__"):
                # Marshal captures constants as well as bytecode, so `x + 1` and `x + 999`
                # differ. Remove embedded source paths before hashing to keep it portable.
                return _hash_code_object(func.__code__)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType] - hasattr guards access
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
        if hasattr(func, "__pivot_resources__"):
            globals_dict = cast("dict[str, Any]", getattr(func, "__globals__", {}))
            _strip_stage_decorator(tree, globals_dict)
        tree = _normalize_ast(tree)
        ast_str = ast.dump(tree, annotate_fields=True, include_attributes=False)
        metrics.end("fingerprint.normalize_and_dump", _t_norm)
        return xxhash.xxh64(ast_str.encode()).hexdigest()
    finally:
        metrics.end("fingerprint._compute_function_hash", _t)


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


def is_user_code(obj: Any) -> bool:
    """Check if object is user code (not stdlib/site-packages/builtins)."""
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
    module = _get_module(obj)
    if module is None:
        return False

    # Built-in modules (sys, builtins, _io, etc.) are not user code
    module_name = getattr(module, "__name__", "")
    if module_name in sys.builtin_module_names:
        return False

    # Pivot's own code is framework, not user code (matters for editable installs
    # where pivot isn't in site-packages)
    if module_name == "pivot" or module_name.startswith("pivot."):
        return False

    # Check for namespace packages (PEP 420): they have __path__ but no __file__
    # If __path__ points to site-packages, it's a third-party namespace package
    if not hasattr(module, "__file__") or module.__file__ is None:
        if not hasattr(module, "__path__"):
            return True  # No __file__ and no __path__: user code (exec/notebook/interactive)

        # Namespace package - check if any path is in site-packages
        for path_entry in module.__path__:
            parts = pathlib.Path(path_entry).parts
            for sp in _SITE_PACKAGE_PATHS:
                if sp in parts:
                    return False
        return True  # Namespace package not in site-packages: user code

    module_file = pathlib.Path(module.__file__).resolve()

    if _is_stdlib_path(module_file):
        return False

    return not any(path in module_file.parts for path in _SITE_PACKAGE_PATHS)


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


def _is_stdlib_path(module_file: pathlib.Path) -> bool:
    """Check if path is in Python stdlib (but not site-packages)."""
    for stdlib_path in _STDLIB_PATHS:
        try:
            if stdlib_path in module_file.parents:
                return not any(path in module_file.parts for path in _SITE_PACKAGE_PATHS)
        except (ValueError, OSError):
            continue

    return False
