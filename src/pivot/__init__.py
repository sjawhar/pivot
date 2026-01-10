from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0-dev"

# Public API - only exports that users need when writing pipelines
# Internal modules like REGISTRY, BaseOut, and show.* are accessible
# via their full paths (e.g., pivot.registry.REGISTRY) for advanced use

if TYPE_CHECKING:
    from pivot.outputs import IncrementalOut as IncrementalOut
    from pivot.outputs import Metric as Metric
    from pivot.outputs import Out as Out
    from pivot.outputs import Plot as Plot
    from pivot.pipeline import Pipeline as Pipeline
    from pivot.registry import Variant as Variant
    from pivot.registry import stage as stage

# Lazy import mapping for runtime
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "IncrementalOut": ("pivot.outputs", "IncrementalOut"),
    "Metric": ("pivot.outputs", "Metric"),
    "Out": ("pivot.outputs", "Out"),
    "Plot": ("pivot.outputs", "Plot"),
    "Pipeline": ("pivot.pipeline", "Pipeline"),
    "Variant": ("pivot.registry", "Variant"),
    "stage": ("pivot.registry", "stage"),
}


def __getattr__(name: str) -> object:
    """Lazily import public API members on first access."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        # Cache in module globals for subsequent access
        globals()[name] = value
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """List available attributes including lazy imports."""
    return list(globals().keys()) + list(_LAZY_IMPORTS.keys())
