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
        setattr(func, "__pivot_no_fingerprint__", True)  # noqa: B010
        setattr(func, "__pivot_code_deps__", code_deps or [])  # noqa: B010
        return func

    return decorator
