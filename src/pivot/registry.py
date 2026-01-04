"""Stage registry for collecting pipeline stages.

Provides the @stage decorator for marking functions as pipeline stages and
a registry for managing all registered stages.

Example:
    >>> from pivot import stage
    >>>
    >>> @stage(deps=['data.csv'], outs=['output.txt'])
    >>> def process(input_file: str = 'data.csv'):
    ...     with open(input_file) as f:
    ...         data = f.read()
    ...     with open('output.txt', 'w') as f:
    ...         f.write(data.upper())
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from . import fingerprint

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class stage:
    """Decorator for marking functions as pipeline stages.

    Args:
        deps: Input dependencies (files or 'stage:<name>')
        outs: Output files produced by stage
        params_cls: Optional Pydantic model for parameters

    Example:
        >>> @stage(deps=['input.txt'], outs=['output.txt'])
        >>> def process(input_file: str, output_file: str):
        ...     # Process files...
        ...     pass
    """

    deps: list[str] = field(default_factory=list)
    outs: list[str] = field(default_factory=list)
    params_cls: type[BaseModel] | None = None

    def __call__(self, func: F) -> F:
        """Register function as a stage (returns original function unmodified)."""
        REGISTRY.register(
            func,
            name=func.__name__,
            deps=self.deps,
            outs=self.outs,
            params_cls=self.params_cls,
        )
        return func


class StageRegistry:
    """Global registry for all pipeline stages."""

    def __init__(self) -> None:
        self._stages: dict[str, dict[str, Any]] = {}

    def register(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        deps: list[str] | None = None,
        outs: list[str] | None = None,
        params_cls: type[BaseModel] | None = None,
    ) -> None:
        """Register a stage function with metadata."""
        stage_name = name if name is not None else func.__name__

        # TODO (future): Warn or error on duplicate stage names to prevent
        # accidental overwrites. Current behavior silently replaces existing stage.

        # TODO (future): Validate deps/outs paths:
        # - Check for invalid characters (e.g., '..')
        # - Warn on absolute paths outside project
        # - Detect circular dependencies in stage references
        self._stages[stage_name] = {
            "func": func,
            "name": stage_name,
            "deps": deps if deps is not None else [],
            "outs": outs if outs is not None else [],
            "params_cls": params_cls,
            "signature": inspect.signature(func),
            "fingerprint": fingerprint.get_stage_fingerprint(func),
        }

    def get(self, name: str) -> dict[str, Any]:
        """Get stage info by name (raises KeyError if not found)."""
        return self._stages[name]

    def list_stages(self) -> list[str]:
        """Get list of all stage names."""
        return list(self._stages.keys())

    def clear(self) -> None:
        """Clear all registered stages (for testing)."""
        self._stages.clear()


REGISTRY = StageRegistry()
