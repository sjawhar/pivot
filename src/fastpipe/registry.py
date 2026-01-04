"""Stage registry for collecting pipeline stages.

Provides the @stage decorator for marking functions as pipeline stages and
a registry for managing all registered stages.

Example:
    >>> from fastpipe import stage
    >>>
    >>> @stage(deps=['data.csv'], outs=['output.txt'])
    >>> def process(input_file: str = 'data.csv'):
    ...     with open(input_file) as f:
    ...         data = f.read()
    ...     with open('output.txt', 'w') as f:
    ...         f.write(data.upper())
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


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

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Register function as a stage (returns original function unmodified)."""
        # TODO: Implement stage registration
        # 1. Get function signature with inspect.signature()
        # 2. Generate fingerprint with get_stage_fingerprint()
        # 3. Register in REGISTRY with stage info
        # 4. Return original function unmodified
        raise NotImplementedError("Week 1 implementation in progress")


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
        # TODO: Implement registration logic
        # 1. Validate inputs
        # 2. Generate stage name
        # 3. Create stage info dict
        # 4. Store in _stages
        raise NotImplementedError("Week 1 implementation in progress")

    def get(self, name: str) -> dict[str, Any]:
        """Get stage info by name (raises KeyError if not found)."""
        return self._stages[name]

    def list_stages(self) -> list[str]:
        """Get list of all stage names."""
        return list(self._stages.keys())

    def clear(self) -> None:
        """Clear all registered stages (for testing)."""
        self._stages.clear()


# Global registry instance
REGISTRY = StageRegistry()
