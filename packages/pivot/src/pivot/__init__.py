# pyright: reportImportCycles=false, reportImplicitRelativeImport=false
from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.2.0a1"

# Public API - only exports that users need when writing pipelines
# Internal modules like BaseOut and show.* are accessible via their full paths

if TYPE_CHECKING:
    from pivot import loaders as loaders
    from pivot import merkle as merkle
    from pivot import stage_def as stage_def
    from pivot.compose import Pipeline as Pipeline
    from pivot.compose import metric as metric
    from pivot.compose import plot as plot
    from pivot.compose import stage as stage
    from pivot.decorators import no_fingerprint as no_fingerprint
    from pivot.stage_def import StageParams as StageParams

# Lazy import mapping for runtime: (module_path, attr_name or None for module import)
_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "no_fingerprint": ("pivot.decorators", "no_fingerprint"),
    "merkle": ("pivot.merkle", None),
    "loaders": ("pivot.loaders", None),
    "stage_def": ("pivot.stage_def", None),
    "Pipeline": ("pivot.compose", "Pipeline"),
    "stage": ("pivot.compose", "stage"),
    "metric": ("pivot.compose", "metric"),
    "plot": ("pivot.compose", "plot"),
    "StageParams": ("pivot.stage_def", "StageParams"),
}


def __getattr__(name: str) -> object:
    """Lazily import public API members on first access."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = module if attr_name is None else getattr(module, attr_name)
        # Cache in module globals for subsequent access
        globals()[name] = value
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """List available attributes including lazy imports."""
    return list(globals().keys()) + list(_LAZY_IMPORTS.keys())
