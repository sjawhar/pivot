from __future__ import annotations

import ast
import inspect
import textwrap
import types
from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable[..., object])


def no_fingerprint(code_deps: list[str] | None = None) -> Callable[[F], F]:
    """Disable AST fingerprinting for a stage. Use file-level hashing instead.

    Must be called with parentheses: ``@no_fingerprint()`` not ``@no_fingerprint``.
    """
    if callable(code_deps):
        raise TypeError("Use @no_fingerprint() with parentheses, not @no_fingerprint")

    def decorator(func: F) -> F:
        # Marks only the object decorated: a function shared by several @stage wrappers must
        # not change for all of them. @stage's functools.wraps copies marks set below it.
        setattr(func, "__pivot_no_fingerprint__", True)  # noqa: B010
        setattr(func, "__pivot_code_deps__", code_deps or [])  # noqa: B010
        return func

    return decorator


_FINGERPRINT_BY_NAME_ATTR = "__pivot_fingerprint_by_name__"


def fingerprint_by_name[F: Callable[..., object]](func: F) -> F:
    """Record a helper by its module and qualified name only.

    A stage that calls a marked helper does not rerun when the helper's body
    changes. It reruns once when the helper moves to another module, when the
    helper is renamed, or when the mark is removed.

    Pivot does not walk code that only a marked helper calls. If that code
    imports a user module in its body, the run-time check counts the import,
    and the stage warns and reruns each run. Move that import to the top of
    its module to clear it.

    The fingerprint walk never reads the body of a marked function. A decorator above the
    mark is user code: Pivot adds a function decorator's own code to the fingerprint and
    records the marked function inside it by name, also when `functools.cache` wraps that
    decorator. A library wrapper directly above the mark (`functools.cache`,
    `functools.lru_cache`, `contextlib.contextmanager`) keeps the record by name. Pivot
    refuses a class-based decorator above the mark, however the stage reaches the helper and
    however the decorator keeps `__wrapped__` (in its `__dict__`, a slot or a property),
    because the decorator instance can carry state Pivot cannot fingerprint soundly. Unsafe
    fingerprinting does not suppress that refusal. To clear it, put the mark above the
    class-based decorator (the decorator accepts an instance that `functools.update_wrapper`
    made), or use a `functools.wraps` function decorator. Without a mark below it, a
    `functools.wraps` decorator is fingerprinted as in earlier versions: Pivot does not see what
    its own lines read through a module (`settings.FACTOR`), or its body when `functools.cache`
    wraps it.

    A stage cannot carry the mark: Pipeline refuses a stage with the mark
    anywhere on its `__wrapped__` chain.

    Soundness is the user's choice: if a marked helper can change a stage's
    output, Pivot does not see the change.

    Use bare: ``@fingerprint_by_name``, not ``@fingerprint_by_name()``.
    """
    _validate_fingerprint_by_name_target(func)
    # Store the function itself. `functools.update_wrapper` copies `__dict__`, so the copy in
    # a wrapper holds another function, and the wrapper does not read as marked. An identity
    # check also holds after the function's address is reused.
    func.__dict__[_FINGERPRINT_BY_NAME_ATTR] = func
    return func


def _validate_fingerprint_by_name_target(func: object) -> None:
    """Validate that *func* is a suitable target for ``@fingerprint_by_name``."""
    if not callable(func):
        raise TypeError(f"@fingerprint_by_name requires a callable, got {type(func).__name__}")
    if isinstance(func, (type, staticmethod, classmethod)):
        inner = getattr(func, "__func__", func)
        inner_name = getattr(inner, "__qualname__", type(func).__name__)
        msg = (
            f"@fingerprint_by_name requires a module-level function, "
            f"not {type(func).__name__} ({inner_name})"
        )
        raise TypeError(msg)
    from pivot import fingerprint as _fp

    unwrapped = _fp.unwrap(func)
    if not isinstance(unwrapped, types.FunctionType):
        raise TypeError(f"@fingerprint_by_name requires a function, got {type(func).__name__}")
    qualname = getattr(unwrapped, "__qualname__", "")
    name = getattr(unwrapped, "__name__", "")
    if "<lambda>" in qualname:
        raise TypeError("@fingerprint_by_name does not accept a lambda")
    if qualname != name:
        msg = (
            f"@fingerprint_by_name requires a module-level function, "
            f"not a method or nested function ({qualname})"
        )
        raise TypeError(msg)
    # Refuse whatever the walk refuses in a function body: user-module body
    # imports and dynamic imports (importlib.import_module, __import__).
    _check_body_at_decoration(unwrapped)


def _check_body_at_decoration(func: types.FunctionType) -> None:
    """Refuse *func* if its body has a user-module import or a dynamic import."""
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        # Unreadable source (zipapp, exec'd, C extension). Skip the check.
        # The runtime import check (#496) catches any uncovered import.
        return
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return
    from pivot import exceptions as _exc
    from pivot import fingerprint as _fp

    # The walk's own dynamic-import detector. The walk does not read a marked body, so a
    # `__import__` call in it is refused here rather than counted on every run.
    _fp._check_dynamic_name_access(func, tree, dunder_import=True)  # pyright: ignore[reportPrivateUsage] - internal API within pivot

    for imp in _fp._body_imports(func, tree):  # pyright: ignore[reportPrivateUsage] - internal API within pivot
        if _fp._is_user_module_name(imp.module):  # pyright: ignore[reportPrivateUsage] - internal API within pivot
            msg = (
                f"@fingerprint_by_name helper '{func.__qualname__}' imports user "
                f"module '{imp.module}' in its body. The fingerprint walk does not "
                f"descend into a marked helper, so this import would not be covered. "
                f"Fix: import '{imp.module}' at the top of '{func.__module__}', or "
                "remove @fingerprint_by_name."
            )
            raise _exc.StageDefinitionError(msg)
