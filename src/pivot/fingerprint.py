import ast
import atexit
import contextlib
import dataclasses
import functools
import inspect
import json
import logging
import marshal
import pathlib
import sys
import types
import typing
import weakref
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

import xxhash

from pivot import ast_utils, metrics

if TYPE_CHECKING:
    from pivot import loaders
    from pivot.storage.state import StateDB

_logger = logging.getLogger(__name__)

_SITE_PACKAGE_PATHS = ("site-packages", "dist-packages")


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
_pending_ast_writes: list[tuple[str, int, int, int, str, str]] = []


def _close_state_db() -> None:
    """Close the readonly StateDB on process exit."""
    global _state_db
    if _state_db is not None:
        _state_db.close()
        _state_db = None


atexit.register(_close_state_db)


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

        _state_db = state.StateDB(io.get_state_db_path(), readonly=True)
        return _state_db
    except Exception:
        # OSError (filesystem), ImportError (module), lmdb.Error, etc.
        return None


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

        with state.StateDB(io.get_state_db_path(), readonly=False) as db:
            db.save_ast_hash_many(pending)
        metrics.count("fingerprint.ast_hash_cache.flush")
    except Exception:
        # OSError (filesystem), ImportError (module), lmdb.Error, etc.
        # Log but don't fail - cache is a performance optimization, not critical
        _logger.debug("Failed to flush AST hash cache (%d entries)", len(pending), exc_info=True)


def get_stage_fingerprint(
    func: Callable[..., Any], visited: set[int] | None = None
) -> dict[str, str]:
    """Generate fingerprint manifest capturing all code dependencies.

    Returns dict with keys:
    - 'self:<name>': Function itself (hash)
    - 'func:<name>': Referenced helper functions (hash, transitive)
    - 'class:<name>': Referenced class definitions (hash, transitive)
    - 'mod:<module>.<attr>': User-code module attributes (hash)
    - 'const:<name>': Global constants (repr value)
    """
    if visited is None:
        visited = set()
        # TODO (future): If parallel fingerprinting is needed, use threading.Lock
        # to protect visited set from race conditions. Current single-threaded
        # usage is safe.

    with metrics.timed("fingerprint.get_stage_fingerprint"):
        return _get_stage_fingerprint_impl(func, visited)


def _get_stage_fingerprint_impl(func: Callable[..., Any], visited: set[int]) -> dict[str, str]:
    """Internal implementation of get_stage_fingerprint."""
    manifest = dict[str, str]()

    func_id = id(func)
    if func_id in visited:
        return manifest
    visited.add(func_id)

    func_name = getattr(func, "__name__", "<lambda>")
    manifest[f"self:{func_name}"] = hash_function_ast(func)

    with metrics.timed("fingerprint.getclosurevars"):
        try:
            # Use cached result if available (getclosurevars is expensive at ~0.5ms)
            cached_closure = _getclosurevars_cache.get(func)
            if cached_closure is not None:
                closure_vars = cached_closure
            else:
                closure_vars = inspect.getclosurevars(func)
                _getclosurevars_cache[func] = closure_vars
        except (TypeError, AttributeError):
            return manifest

    _process_closure_values(
        closure_vars.globals,
        func,
        manifest,
        visited,
        skip_dunders=True,
        include_modules=True,
    )
    _process_closure_values(
        closure_vars.nonlocals,
        func,
        manifest,
        visited,
        skip_dunders=False,
        include_modules=False,
    )

    _process_type_hint_dependencies(func, manifest, visited)

    return manifest


def get_loader_fingerprint(loader: "loaders.Writer[Any] | loaders.Reader[Any]") -> dict[str, str]:
    """Generate fingerprint manifest for a loader instance.

    Handles Reader, Writer, and Loader (which inherits from both) instances.
    Only fingerprints methods that exist on the handler type:
    - Reader: 'loader:<classname>:load' + empty() if overridden
    - Writer: 'loader:<classname>:save'
    - Loader: All of the above
    - Always: 'loader:<classname>:config' for dataclass field values
    """
    manifest = dict[str, str]()
    class_name = type(loader).__name__

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
            for cls in type(loader).__mro__:
                if cls is loaders_module.Loader:
                    # Reached Loader base class without finding override
                    break
                if "empty" in cls.__dict__:
                    # Found override in a subclass
                    manifest[f"loader:{class_name}:empty"] = hash_function_ast(loader.empty)
                    break

    # Config hash from dataclass fields
    field_values = list[str]()
    for field in dataclasses.fields(loader):
        value = getattr(loader, field.name)
        field_values.append(f"{field.name}={value!r}")
    if field_values:
        config_str = ",".join(field_values)
        manifest[f"loader:{class_name}:config"] = xxhash.xxh64(config_str.encode()).hexdigest()

    return manifest


def _process_closure_values(
    values: Mapping[str, Any],
    func: Callable[..., Any],
    manifest: dict[str, str],
    visited: set[int],
    *,
    skip_dunders: bool,
    include_modules: bool,
) -> None:
    """Process closure variable values (globals or nonlocals) and add to manifest."""
    for name, value in values.items():
        if skip_dunders and name.startswith("__"):
            continue

        # functools.partial fails is_user_code() (module is functools/stdlib)
        # Must check before general callable check
        if isinstance(value, functools.partial):
            _process_partial_dependency(
                name, cast("functools.partial[Any]", value), manifest, visited
            )
        elif callable(value) and is_user_code(value):
            _process_callable_dependency(name, value, manifest, visited)
        elif include_modules and isinstance(value, types.ModuleType):
            _process_module_dependency(name, value, func, manifest, visited)
        elif isinstance(value, (bool, int, float, str, bytes, type(None))):
            manifest[f"const:{name}"] = repr(value)
        elif isinstance(value, (dict, list, tuple, set, frozenset)):
            _process_collection_dependency(
                name,
                cast(
                    "dict[Any, Any] | list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any]",
                    value,
                ),
                manifest,
                visited,
            )
        elif _is_user_class_instance(value):
            _process_instance_dependency(name, value, manifest, visited)


def _process_callable_dependency(
    name: str, func: Callable[..., Any], manifest: dict[str, str], visited: set[int]
) -> None:
    """Process a callable dependency and add to manifest."""
    # Use 'class:' prefix for type objects, 'func:' for functions
    prefix = "class" if isinstance(func, type) else "func"
    _add_callable_to_manifest(f"{prefix}:{name}", func, manifest, visited)


def _process_partial_dependency(
    name: str,
    partial_obj: functools.partial[Any],
    manifest: dict[str, str],
    visited: set[int],
) -> None:
    """Process functools.partial: hash underlying func and bound arguments."""
    # Hash bound args and kwargs (changes to these should invalidate cache)
    args_str = _serialize_value_for_hash(partial_obj.args)
    kwargs_str = _serialize_value_for_hash(partial_obj.keywords)
    manifest[f"partial:{name}.args"] = xxhash.xxh64(args_str.encode()).hexdigest()
    manifest[f"partial:{name}.kwargs"] = xxhash.xxh64(kwargs_str.encode()).hexdigest()

    # Recursively fingerprint the underlying function if it's user code
    underlying = partial_obj.func
    if callable(underlying) and is_user_code(underlying):
        _add_callable_to_manifest(f"func:{name}.func", underlying, manifest, visited)


def _is_user_class_instance(value: Any) -> bool:
    """Check if value is an instance of a user-defined class."""
    cls = cast("type[Any]", type(value))
    # Skip built-in types and common stdlib types
    if cls.__module__ == "builtins":
        return False
    return is_user_code(cls)


def _process_instance_dependency(
    name: str, instance: Any, manifest: dict[str, str], visited: set[int]
) -> None:
    """Track the class definition of a user-defined instance."""
    cls = cast("type[Any]", type(instance))
    _add_callable_to_manifest(f"class:{name}.__class__", cls, manifest, visited)


def _process_type_hint_dependencies(
    func: Callable[..., Any], manifest: dict[str, str], visited: set[int]
) -> None:
    """Process user-defined classes in type hints, including Pydantic model defaults."""
    with metrics.timed("fingerprint.get_type_hints"):
        # Use cached result if available
        try:
            cached = _get_type_hints_cache.get(func)
            if cached is not None:
                hints = cached
            else:
                try:
                    hints = typing.get_type_hints(func)
                except Exception:
                    return
                _get_type_hints_cache[func] = hints
        except TypeError:
            # Not weakly referenceable
            try:
                hints = typing.get_type_hints(func)
            except Exception:
                return

    for hint in hints.values():
        _process_type_hint(hint, manifest, visited)


def _process_type_hint(hint: Any, manifest: dict[str, str], visited: set[int]) -> None:
    """Process a single type hint, recursively handling generics."""
    if hint is type(None):
        return

    origin = typing.get_origin(hint)
    if origin is not None:
        for arg in typing.get_args(hint):
            _process_type_hint(arg, manifest, visited)
        return

    if not isinstance(hint, type):
        return

    if not is_user_code(hint):
        return

    key = f"class:{hint.__name__}"
    if key not in manifest:
        _add_callable_to_manifest(key, hint, manifest, visited)

    # Always hash Pydantic defaults (even if class was already added via closure vars)
    if hasattr(hint, "model_fields"):
        _hash_pydantic_defaults(hint, manifest)


def _hash_pydantic_defaults(model: type, manifest: dict[str, str]) -> None:
    """Hash Pydantic model field default values including default_factory."""
    model_fields = getattr(model, "model_fields", {})
    if not model_fields:
        return

    for field_name, field_info in model_fields.items():
        default = getattr(field_info, "default", None)
        default_factory = getattr(field_info, "default_factory", None)

        # Check for PydanticUndefined (no default) - don't skip None, it's a valid default
        if type(default).__name__ == "PydanticUndefinedType":
            # No static default, check for default_factory
            if default_factory is not None and callable(default_factory):
                # Hash the factory function itself (tracks code changes)
                key = f"pydantic:{model.__name__}.{field_name}"
                manifest[key] = hash_function_ast(default_factory)
            continue

        value_str = _serialize_value_for_hash(default)
        key = f"pydantic:{model.__name__}.{field_name}"
        manifest[key] = xxhash.xxh64(value_str.encode()).hexdigest()


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


def _process_collection_dependency(
    name: str,
    collection: dict[Any, Any] | list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any],
    manifest: dict[str, str],
    visited: set[int],
) -> None:
    """Scan collection for callable user code and add to manifest."""
    if isinstance(collection, dict):
        # Use sorted keys for deterministic ordering
        for key in sorted(collection.keys(), key=_sort_key):
            value = collection[key]
            if callable(value) and is_user_code(value):
                _add_callable_to_manifest(f"func:{name}[{key!r}]", value, manifest, visited)
    else:
        # For sequences and sets, use enumerate for index-based keys
        # Sort sets for deterministic ordering
        items = (
            sorted(collection, key=_sort_key)
            if isinstance(collection, (set, frozenset))
            else collection
        )
        for i, value in enumerate(items):
            if callable(value) and is_user_code(value):
                _add_callable_to_manifest(f"func:{name}[{i}]", value, manifest, visited)


def _sort_key(value: Any) -> tuple[str, str]:
    """Sort key that handles mixed types safely."""
    return (type(value).__name__, str(value))


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

    with metrics.timed("fingerprint.extract_module_attr_usage"):
        attrs = ast_utils.extract_module_attr_usage(func)
    module_name = getattr(module, "__name__", name)

    for mod_name, attr_name in attrs:
        if mod_name not in (name, module_name):
            continue
        key = f"mod:{mod_name}.{attr_name}"
        if key in manifest:
            continue
        try:
            attr_value = getattr(module, attr_name)
        except AttributeError:
            manifest[key] = "unknown"
            continue
        if callable(attr_value) and is_user_code(attr_value):
            _add_callable_to_manifest(key, attr_value, manifest, visited)
        else:
            manifest[key] = repr(attr_value)


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
    (file_path, mtime_ns, size, inode, qualname) for automatic invalidation.

    Limitation: Lambdas and functions without source code fall back to id(func),
    which is non-deterministic across runs. This causes unnecessary re-runs for
    stages using lambdas. Mitigation: Use named functions instead of lambdas in
    pipeline stages for stable fingerprinting.
    """
    with metrics.timed("fingerprint.hash_function_ast"):
        # 1. Check in-memory WeakKeyDictionary cache first (fastest)
        # WeakKeyDictionary raises TypeError for non-weakly-referenceable functions (builtins)
        try:
            cached = _hash_function_ast_cache.get(func)
            if cached is not None:
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
                qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", None)
                if qualname is None:
                    raise RuntimeError(
                        f"Cannot fingerprint {func!r}: missing __qualname__ and __name__. This indicates a bug in pivot's fingerprinting logic."
                    )
                db = _get_state_db()
                if db is not None:
                    rel_path, mtime_ns, size, inode = source_info
                    try:
                        persistent_cached = db.get_ast_hash(
                            rel_path, mtime_ns, size, inode, qualname
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
            _pending_ast_writes.append((rel_path, mtime_ns, size, inode, qualname, result))
            metrics.count("fingerprint.hash_function_ast.persistent_cache_miss")

        return result


def _compute_function_hash(func: Callable[..., Any]) -> str:
    """Compute hash for a function (uncached implementation)."""
    with metrics.timed("fingerprint._compute_function_hash"):
        # Builtins (list, dict, set, etc.) have no source - use stable name-based hash.
        # This is deterministic across Python sessions unlike id(func).
        if isinstance(func, type) and func.__module__ == "builtins":
            return xxhash.xxh64(f"builtin:{func.__qualname__}".encode()).hexdigest()

        # For wrapped functions (via functools.wraps), inspect.getsource() follows
        # __wrapped__ and returns the ORIGINAL function's source, making decorator
        # logic invisible. Use __code__ bytecode to capture the actual wrapper.
        if hasattr(func, "__wrapped__") and hasattr(func, "__code__"):
            return xxhash.xxh64(marshal.dumps(func.__code__)).hexdigest()  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType] - hasattr guards access

        with metrics.timed("fingerprint.inspect_getsource"):
            try:
                source = inspect.getsource(func)
            except (OSError, TypeError):
                if hasattr(func, "__code__"):
                    # marshal.dumps captures full code object including co_consts
                    # (co_code alone doesn't include constants - x+1 and x+999 have same co_code!)
                    return xxhash.xxh64(marshal.dumps(func.__code__)).hexdigest()  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType] - hasattr guards access
                # KNOWN ISSUE: Using id(func) is non-deterministic across runs
                # This affects lambdas without source code, causing unnecessary re-runs
                return xxhash.xxh64(str(id(func)).encode()).hexdigest()

        with metrics.timed("fingerprint.ast_parse"):
            try:
                tree = ast.parse(source)
            except SyntaxError:
                return xxhash.xxh64(source.encode()).hexdigest()

        with metrics.timed("fingerprint.normalize_and_dump"):
            tree = _normalize_ast(tree)
            ast_str = ast.dump(tree, annotate_fields=True, include_attributes=False)
        return xxhash.xxh64(ast_str.encode()).hexdigest()


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
    with metrics.timed("fingerprint.is_user_code"):
        if obj is None:
            return False

        # Use cached result if available (called 10K+ times with many repeats)
        try:
            cached = _is_user_code_cache.get(obj)
            if cached is not None:
                return cached
        except TypeError:
            # Not weakly referenceable
            pass

        result = _is_user_code_impl(obj)

        with contextlib.suppress(TypeError):
            _is_user_code_cache[obj] = result

        return result


def _is_user_code_impl(obj: Any) -> bool:
    """Internal implementation of is_user_code."""
    module = _get_module(obj)
    if module is None:
        return False

    # Built-in modules (sys, builtins, _io, etc.) are not user code
    module_name = getattr(module, "__name__", "")
    if module_name in sys.builtin_module_names:
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
