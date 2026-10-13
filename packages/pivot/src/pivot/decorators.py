from __future__ import annotations

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
        # Mark every layer of a functools.wraps chain: @stage keeps the undecorated function
        # for execution and fingerprinting, so the marks must reach it on either side of @stage.
        layer: object = func
        while layer is not None:
            setattr(layer, "__pivot_no_fingerprint__", True)  # noqa: B010
            setattr(layer, "__pivot_code_deps__", code_deps or [])  # noqa: B010
            layer = getattr(layer, "__wrapped__", None)
        return func

    return decorator
